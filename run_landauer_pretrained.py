"""
THERMODYNAMICS OF ATTENTION — LANDAUER LIMIT (PRETRAINED MODELS)

Validates the neural Landauer bound  |ΔL| ≥ T|ΔS|  at scale by
fine-tuning pretrained Vision Transformers from timm at various
temperatures on Food-101 (101 classes, auto-downloaded).

For each temperature T the script:
  1. Loads a fresh copy of the pretrained model
  2. Replaces the ImageNet head with a Food-101 head
  3. Injects temperature T into every attention layer
  4. Measures the initial state (loss, attention entropy, MI)
  5. Fine-tunes for a few epochs
  6. Measures the final state
  7. Records  |ΔL|,  Q = T·|ΔS|,  W = ΔI

"""

import os
import math
import copy
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm

# ============================================================
# 1) CONFIGURATION
# ============================================================

MODEL_NAME = "vit_base_patch16_224"
# Also tested: "vit_base_patch16_224", "vit_large_patch16_224"

# Food-101: 101 classes, auto-downloads (~5 GB), used for Landauer fine-tuning.
# The pretrained ImageNet backbone is retained; only the classification head
# is replaced. This avoids requiring the 138 GB ImageNet training set.
DATA_DIR = "./data"
NUM_CLASSES = 101

# Temperature sweep (log-spaced, matching the paper's grid)
TEMPS = np.geomspace(0.1, 10.0, 10).tolist()

# Fine-tuning config
FINETUNE_EPOCHS = 2
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
TRAIN_LIMIT = 20_000
EVAL_LIMIT = 5_000
NUM_SEEDS = 1
BASE_SEED = 42

NUM_WORKERS = 1

# Output
OUT_DIR = "./landauer_pretrained_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ImageNet normalization (standard for pretrained timm models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ============================================================
# 2) ATTENTION PATCHING — TEMPERATURE INJECTION + ENTROPY
# ============================================================
class ThermalAttentionPatcher:
    """
    Patches all timm Attention modules in a ViT to:
      1. Scale attention logits by 1/T  (temperature injection)
      2. Accumulate per-sample normalized attention entropy
    """

    def __init__(self, model, T=1.0, track_entropy=True):
        self.model = model
        self.T = T
        self.track_entropy = track_entropy
        self._entropy_accumulators = []
        self._n_blocks = len(model.blocks)

        for idx, block in enumerate(model.blocks):
            acc = []
            self._entropy_accumulators.append(acc)
            self._patch_module(block.attn, acc)

    def _patch_module(self, mod, acc):
        patcher = self
        scale = mod.scale
        num_heads = mod.num_heads
        head_dim = getattr(mod, 'head_dim', mod.qkv.in_features // num_heads)

        if hasattr(mod, 'fused_attn'):
            mod.fused_attn = False

        def thermal_forward(x, **kwargs):
            B, N, C = x.shape
            qkv = mod.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            logits = (q * scale) @ k.transpose(-2, -1)
            logits = logits / max(patcher.T, 1e-6)
            logits = logits - logits.amax(dim=-1, keepdim=True)

            attn = logits.softmax(dim=-1)

            if patcher.track_entropy:
                with torch.no_grad():
                    n_keys = attn.size(-1)
                    S = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
                    S_norm = S / math.log(n_keys)
                    acc.append(S_norm.mean(dim=(1, 2)))

            attn = mod.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = mod.proj(x)
            if hasattr(mod, 'proj_drop'):
                x = mod.proj_drop(x)
            return x

        mod.forward = thermal_forward

    def collect_batch_entropy(self):
        block_means = []
        for acc in self._entropy_accumulators:
            if len(acc) == 0:
                continue
            block_means.append(torch.cat(acc, dim=0))

        if len(block_means) == 0:
            return []

        stacked = torch.stack(block_means, dim=0)
        per_sample = stacked.mean(dim=0)
        return per_sample.cpu().tolist()

    def reset(self):
        for acc in self._entropy_accumulators:
            acc.clear()


# ============================================================
# 3) MUTUAL INFORMATION (Soft Confusion)
# ============================================================
def compute_soft_mi_nats(soft_confusion, pseudocount=1e-8):
    conf = soft_confusion
    if pseudocount > 0:
        conf = conf + pseudocount
    total = conf.sum()
    if total.item() <= 0:
        return 0.0

    P_joint = conf / total
    P_y = P_joint.sum(dim=1, keepdim=True)
    P_yhat = P_joint.sum(dim=0, keepdim=True)
    eps = 1e-12
    mi = (P_joint * (torch.log(P_joint + eps)
                      - torch.log(P_y + eps)
                      - torch.log(P_yhat + eps))).sum()
    return float(mi.item())


# ============================================================
# 4) STATE MEASUREMENT
# ============================================================
@torch.no_grad()
def measure_state(model, patcher, loader, criterion, device, num_classes):
    model.eval()
    patcher.reset()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    soft_confusion = torch.zeros(num_classes, num_classes,
                                  dtype=torch.float64, device='cpu')

    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        bs = x.size(0)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += bs

        probs = torch.softmax(logits, dim=-1).cpu().to(torch.float64)
        idx = y.cpu().view(-1, 1).expand(-1, num_classes)
        soft_confusion.scatter_add_(0, idx, probs)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    mi = compute_soft_mi_nats(soft_confusion)

    S_samples = patcher.collect_batch_entropy()
    S_norm_mean = float(np.mean(S_samples)) if S_samples else 0.0

    pe = model.patch_embed
    img_size = pe.img_size[0] if isinstance(pe.img_size, (tuple, list)) else pe.img_size
    ps = pe.patch_size[0] if isinstance(pe.patch_size, (tuple, list)) else pe.patch_size
    n_tokens = (img_size // ps) ** 2 + 1
    S_raw_mean = S_norm_mean * math.log(n_tokens)

    return {
        "loss": avg_loss,
        "acc": acc,
        "S_norm": S_norm_mean,
        "S_raw": S_raw_mean,
        "mi_nats": mi,
    }


# ============================================================
# 5) DATASET
# ============================================================
def get_dataloaders():
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dset = datasets.Food101(DATA_DIR, split="train", download=True, transform=tfm_train)
    eval_dset = datasets.Food101(DATA_DIR, split="test", download=True, transform=tfm_eval)

    rng = np.random.default_rng(BASE_SEED)
    if TRAIN_LIMIT and len(train_dset) > TRAIN_LIMIT:
        idx = rng.choice(len(train_dset), size=TRAIN_LIMIT, replace=False).tolist()
        train_dset = Subset(train_dset, idx)
    if EVAL_LIMIT and len(eval_dset) > EVAL_LIMIT:
        idx = rng.choice(len(eval_dset), size=EVAL_LIMIT, replace=False).tolist()
        eval_dset = Subset(eval_dset, idx)

    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=False)
    eval_loader = DataLoader(eval_dset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, eval_loader


# ============================================================
# 6) MAIN LOOP
# ============================================================
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")
    print(f"Dataset: Food-101 (train subset: {TRAIN_LIMIT}, eval subset: {EVAL_LIMIT})")
    print(f"Temperatures: {len(TEMPS)} points from {TEMPS[0]:.4f} to {TEMPS[-1]:.2f}")
    print(f"Seeds: {NUM_SEEDS}, Fine-tune epochs: {FINETUNE_EPOCHS}")

    train_loader, eval_loader = get_dataloaders()

    # Load pretrained weights once
    print(f"Loading pretrained {MODEL_NAME}...")
    base_model = timm.create_model(MODEL_NAME, pretrained=True)
    base_state = copy.deepcopy(base_model.state_dict())
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_classes = NUM_CLASSES

    # Storage for results
    all_results = {
        "dL": np.zeros((NUM_SEEDS, len(TEMPS))),
        "dS": np.zeros((NUM_SEEDS, len(TEMPS))),
        "dS_norm": np.zeros((NUM_SEEDS, len(TEMPS))),
        "Q": np.zeros((NUM_SEEDS, len(TEMPS))),
        "dI": np.zeros((NUM_SEEDS, len(TEMPS))),
        "acc_init": np.zeros((NUM_SEEDS, len(TEMPS))),
        "acc_final": np.zeros((NUM_SEEDS, len(TEMPS))),
    }

    criterion = nn.CrossEntropyLoss()

    for si, seed in enumerate(range(BASE_SEED, BASE_SEED + NUM_SEEDS)):
        print(f"\n--- Seed {si + 1}/{NUM_SEEDS} (seed={seed}) ---")

        for ti, T in enumerate(tqdm(TEMPS, desc=f"Seed {si+1} temps")):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Fresh model from pretrained weights
            model = timm.create_model(MODEL_NAME, pretrained=False)
            model.load_state_dict(base_state, strict=False)

            # Replace head
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            nn.init.xavier_uniform_(model.head.weight)
            nn.init.zeros_(model.head.bias)

            model = model.to(device)

            # Inject temperature
            patcher = ThermalAttentionPatcher(model, T=T, track_entropy=True)

            # Measure initial state
            s0 = measure_state(model, patcher, eval_loader, criterion,
                               device, num_classes)

            # Fine-tune
            optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR,
                                    weight_decay=WEIGHT_DECAY)
            model.train()
            patcher.track_entropy = False
            for epoch in range(FINETUNE_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device).long()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

            # Measure final state
            patcher.track_entropy = True
            sf = measure_state(model, patcher, eval_loader, criterion,
                               device, num_classes)

            # Thermodynamic quantities
            dL = s0["loss"] - sf["loss"]
            dS_raw = s0["S_raw"] - sf["S_raw"]
            dS_norm = s0["S_norm"] - sf["S_norm"]
            Q = T * dS_raw
            dI = sf["mi_nats"] - s0["mi_nats"]

            all_results["dL"][si, ti] = dL
            all_results["dS"][si, ti] = dS_raw
            all_results["dS_norm"][si, ti] = dS_norm
            all_results["Q"][si, ti] = Q
            all_results["dI"][si, ti] = dI
            all_results["acc_init"][si, ti] = s0["acc"]
            all_results["acc_final"][si, ti] = sf["acc"]

            print(f"  T={T:.4f}  |dL|={abs(dL):.4f}  Q={Q:.4f}  "
                  f"acc: {s0['acc']:.3f}->{sf['acc']:.3f}")

            # Clean up
            del model, patcher, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    save_data = {
        "model_name": MODEL_NAME,
        "dataset": "Food101",
        "num_classes": int(NUM_CLASSES),
        "temps": np.array(TEMPS, dtype=np.float64),
        "finetune_epochs": int(FINETUNE_EPOCHS),
        "finetune_lr": float(FINETUNE_LR),
        "weight_decay": float(WEIGHT_DECAY),
        "train_limit": int(TRAIN_LIMIT) if TRAIN_LIMIT else -1,
        "eval_limit": int(EVAL_LIMIT) if EVAL_LIMIT else -1,
        "num_seeds": int(NUM_SEEDS),
        "dL_mean": all_results["dL"].mean(axis=0),
        "dL_std": all_results["dL"].std(axis=0),
        "Q_mean": all_results["Q"].mean(axis=0),
        "Q_std": all_results["Q"].std(axis=0),
        "dI_mean": all_results["dI"].mean(axis=0),
        "dI_std": all_results["dI"].std(axis=0),
        "dS_mean": all_results["dS"].mean(axis=0),
        "dS_norm_mean": all_results["dS_norm"].mean(axis=0),
        "acc_init_mean": all_results["acc_init"].mean(axis=0),
        "acc_final_mean": all_results["acc_final"].mean(axis=0),
        "dL_all": all_results["dL"],
        "Q_all": all_results["Q"],
        "dI_all": all_results["dI"],
        "acc_final_all": all_results["acc_final"],
    }

    safe_name = MODEL_NAME.replace("/", "_")
    fname = os.path.join(OUT_DIR, f"landauer_{safe_name}_food101.mat")
    sio.savemat(fname, save_data)
    print(f"\nSaved: {fname}")

    # Summary
    print("\n--- Landauer Bound Check ---")
    print(f"{'T':>8s}  {'|ΔL|':>8s}  {'Q=T|ΔS|':>8s}  {'Valid':>6s}  {'Acc_f':>6s}")
    for ti, T in enumerate(TEMPS):
        dL_avg = abs(all_results["dL"].mean(axis=0)[ti])
        Q_avg = all_results["Q"].mean(axis=0)[ti]
        valid = "✓" if dL_avg >= Q_avg - 1e-6 else "✗"
        acc_f = all_results["acc_final"].mean(axis=0)[ti]
        print(f"{T:8.4f}  {dL_avg:8.4f}  {Q_avg:8.4f}  {valid:>6s}  {acc_f:6.4f}")

    print("\n--- First Law Check ---")
    print(f"{'T':>8s}  {'|ΔL|':>8s}  {'Q+W':>8s}  {'Regime':>12s}")
    for ti, T in enumerate(TEMPS):
        dL_avg = abs(all_results["dL"].mean(axis=0)[ti])
        Q_avg = all_results["Q"].mean(axis=0)[ti]
        W_avg = all_results["dI"].mean(axis=0)[ti]
        QW = Q_avg + W_avg
        regime = "Dissipative" if dL_avg > QW else "Deficit"
        print(f"{T:8.4f}  {dL_avg:8.4f}  {QW:8.4f}  {regime:>12s}")


if __name__ == "__main__":
    main()
