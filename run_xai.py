"""
THERMODYNAMICS OF ATTENTION- XAI 

"""

import os
import math
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================================
# 0. OPTIONAL DEPENDENCY (OrganAMNIST)
# ============================================================
try:
    from medmnist import OrganAMNIST
except Exception:
    OrganAMNIST = None

# ============================================================
# 1. CONFIGURATION (SELECT DATASET HERE)
# ============================================================
DATASET_NAME = "CIFAR10"  # Options: "MNIST", "CIFAR10", "OrganAMNIST"

OUT_DIR = "./xai_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset-Specific Defaults (Restored)
if DATASET_NAME == "MNIST":
    OPTIMAL_EPOCHS = 20
elif DATASET_NAME == "OrganAMNIST":
    OPTIMAL_EPOCHS = 40
elif DATASET_NAME == "CIFAR10":
    OPTIMAL_EPOCHS = 150
else:
    OPTIMAL_EPOCHS = 50

BASE_CONFIG = {
    "d_model": 64,
    "n_head": 4,
    "n_layer": 2,
    "lr": 1e-3,
    "epochs": OPTIMAL_EPOCHS,
    "batch_size": 32,
    "num_ablation_runs": 10,
    "train_limit": None,
    "test_limit": None,
    "weight_decay": 1e-4,
    "seed": 42,
    "ig_steps": 20,
}

if DATASET_NAME == "MNIST":
    CONFIG = {
        **BASE_CONFIG,
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 10,
    }
elif DATASET_NAME == "CIFAR10":
    CONFIG = {
        **BASE_CONFIG,
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "num_classes": 10,
    }
elif DATASET_NAME == "OrganAMNIST":
    if OrganAMNIST is None:
        raise ImportError("Install medmnist: pip install medmnist")
    CONFIG = {
        **BASE_CONFIG, 
        "img_size": 28, 
        "patch_size": 4, 
        "in_channels": 1, 
        "num_classes": 11
    }
else:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(BASE_CONFIG["seed"])

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"--- Configuration: {DATASET_NAME} ---")
print(f"Device: {device}")
print(CONFIG)

# ============================================================
# 2. MODEL
# ============================================================
class ThermalAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model % n_head != 0")
        self.head_dim = d_model // n_head
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.attn_map_cache = None

    def forward(self, x, T=1.0, track_stats=False, head_mask=None):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        logits = (q @ k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * T)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        att = F.softmax(logits, dim=-1)

        if head_mask is not None:
            att = att * head_mask.view(1, -1, 1, 1)

        if track_stats:
            self.attn_map_cache = att.detach()

        out = (att @ v).transpose(1, 2).reshape(B, S, C)
        return self.proj(out)

class ThermodynamicViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.n_patches = (config["img_size"] // self.patch_size) ** 2
        self.n_head = config["n_head"]

        d_model = config["d_model"]
        patch_dim = config["in_channels"] * (self.patch_size ** 2)

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.n_patches, d_model))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": ThermalAttention(d_model, config["n_head"]),
                "ln1": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
                "ln2": nn.LayerNorm(d_model),
            }) for _ in range(config["n_layer"])
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, config["num_classes"])

    def forward(self, x, T=1.0, track_stats=False, head_masks=None):
        B = x.size(0)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb

        for li, blk in enumerate(self.blocks):
            hm = head_masks[li] if head_masks is not None else None
            x = x + blk["attn"](blk["ln1"](x), T=T, track_stats=track_stats, head_mask=hm)
            x = x + blk["mlp"](blk["ln2"](x))

        x = self.ln_f(x)
        return self.head(x[:, 0])

# ============================================================
# 3. DATA UTILS
# ============================================================
def _stratified_subset(dataset, limit: int, seed: int, num_classes: int):
    # Handles limit=None case implicitly (len check)
    if limit is None or limit >= len(dataset):
        return dataset if not isinstance(dataset, Subset) else dataset

    # Build label -> indices map
    rng = np.random.default_rng(seed)
    indices_by_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if isinstance(y, torch.Tensor):
            yv = int(y.item()) if y.numel() == 1 else int(y.view(-1)[0].item())
        else:
            yv = int(y)
        indices_by_class[yv].append(idx)

    # Shuffle each class list
    for c in range(num_classes):
        rng.shuffle(indices_by_class[c])

    # Round-robin selection to fill limit
    chosen = []
    ptrs = [0] * num_classes
    while len(chosen) < limit:
        made_progress = False
        for c in range(num_classes):
            if len(chosen) >= limit:
                break
            if ptrs[c] < len(indices_by_class[c]):
                chosen.append(indices_by_class[c][ptrs[c]])
                ptrs[c] += 1
                made_progress = True
        if not made_progress:
            break  # dataset exhausted

    return Subset(dataset, chosen)

def get_dataloaders(name, config):
    if name == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dset_cls = datasets.MNIST

    elif name == "CIFAR10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dset_cls = datasets.CIFAR10

    elif name == "OrganAMNIST":
        if OrganAMNIST is None:
             raise ImportError("medmnist not installed.")
        
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # Use global OrganAMNIST class
        tr_full = OrganAMNIST(split="train", download=True, transform=tfm, root="./data")
        te_full = OrganAMNIST(split="test", download=True, transform=tfm, root="./data")

        # Stratified Subsets (limit might be None, which is fine)
        tr_full = _stratified_subset(tr_full, config["train_limit"], config["seed"], config["num_classes"])
        te_full = _stratified_subset(te_full, config["test_limit"], config["seed"] + 999, config["num_classes"])

        tr_loader = DataLoader(tr_full, batch_size=config["batch_size"], shuffle=True)
        te_loader = DataLoader(te_full, batch_size=config["batch_size"], shuffle=True)
        return tr_loader, te_loader, te_full

    else:
        raise ValueError(f"Unknown dataset: {name}")

    tr_full = dset_cls("./data", train=True, download=True, transform=tfm)
    te_full = dset_cls("./data", train=False, download=True, transform=tfm)

    # Stratified Subsets
    tr_full = _stratified_subset(tr_full, config["train_limit"], config["seed"], config["num_classes"])
    te_full = _stratified_subset(te_full, config["test_limit"], config["seed"] + 999, config["num_classes"])

    tr_loader = DataLoader(tr_full, batch_size=config["batch_size"], shuffle=True)
    te_loader = DataLoader(te_full, batch_size=config["batch_size"], shuffle=True)
    return tr_loader, te_loader, te_full

def to_label(lbl, device):
    y = lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl)
    if DATASET_NAME == "OrganAMNIST":
        y = y.view(-1).long()
    else:
        if y.dim() == 0:
            y = y.unsqueeze(0)
        y = y.long()
    return y.to(device)

# ============================================================
# 4. XAI MAPS (Work + IG + Attention)
# ============================================================
def normalized_fill_value(in_channels: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(in_channels, 1, 1, device=device)

@torch.no_grad()
def compute_patch_work(model, img, label, criterion, device):
    """Work per patch: W_ij = L(mask ij) - L(base). Mean-mask in normalized space (fill=0)."""
    model.eval()
    x = img.to(device)
    y = to_label(label, device)
    fill = normalized_fill_value(x.size(0), device)

    loss_base = criterion(model(x.unsqueeze(0)), y).item()

    _, h, w = x.shape
    p = model.patch_size
    n_h, n_w = h // p, w // p
    work_map = np.zeros((n_h, n_w), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            masked = x.clone()
            masked[:, i*p:(i+1)*p, j*p:(j+1)*p] = fill
            loss_m = criterion(model(masked.unsqueeze(0)), y).item()
            work_map[i, j] = loss_m - loss_base

    return work_map

def compute_integrated_gradients(model, img, label, criterion, device, steps=20):
    """
    Integrated Gradients in normalized space.
    """
    model.eval()
    x = img.clone().detach().to(device)
    y = to_label(label, device)

    baseline = torch.zeros_like(x)
    accumulated_grads = torch.zeros_like(x)

    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        x_step = baseline + alpha * (x - baseline)
        x_step.requires_grad_(True)

        out = model(x_step.unsqueeze(0))
        loss = criterion(out, y)

        grad_x, = torch.autograd.grad(
            loss, x_step,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )
        accumulated_grads += grad_x.detach()

    avg_grads = accumulated_grads / float(steps)
    ig = (x - baseline) * avg_grads

    ig_full = ig.abs().sum(dim=0)
    p = model.patch_size
    n_h, n_w = ig_full.shape[0] // p, ig_full.shape[1] // p

    ig_map = np.zeros((n_h, n_w), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            ig_map[i, j] = ig_full[i*p:(i+1)*p, j*p:(j+1)*p].sum().item()

    return ig_map

@torch.no_grad()
def get_cls_attn(model, img, device):
    """Mean attention from CLS token to patches from last layer."""
    model.eval()
    _ = model(img.unsqueeze(0).to(device), track_stats=True)
    att = model.blocks[-1]["attn"].attn_map_cache
    if att is None:
        raise RuntimeError("attn_map_cache is None. Ensure the forward pass used track_stats=True.")
    att = att[0]  # [H,S,S]
    score = att[:, 0, 1:].mean(dim=0).cpu().numpy()  # [n_patches]
    n = int(math.sqrt(score.shape[0]))
    return score.reshape(n, n)

# ============================================================
# 5. ROBUST ABLATION
# ============================================================
def run_robust_ablation(model, dataloader, device, num_runs, ig_steps):
    print(f"Running robust ablation ({num_runs} batches)...")
    model.eval()

    # Need criterion for work/IG
    criterion = nn.CrossEntropyLoss()

    hist_attn, hist_grad, hist_work = [], [], []
    iterator = iter(dataloader)
    ratios = np.linspace(0.0, 0.9, 10)

    n_patches = model.n_patches
    grid = int(math.sqrt(n_patches))
    p = model.patch_size

    for run_idx in range(num_runs):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, labels = next(iterator)

        images = images.to(device)
        labels = to_label(labels, device)
        B = images.size(0)

        fill = normalized_fill_value(images.size(1), device)

        # Precompute maps for this batch
        maps_w, maps_g, maps_a = [], [], []
        for k in range(B):
            maps_w.append(compute_patch_work(model, images[k], labels[k], criterion, device).flatten())
            maps_g.append(compute_integrated_gradients(model, images[k], labels[k], criterion, device, steps=ig_steps).flatten())
            maps_a.append(get_cls_attn(model, images[k], device).flatten())

        maps_w = np.asarray(maps_w)
        maps_g = np.asarray(maps_g)
        maps_a = np.asarray(maps_a)

        def mask_with_scores(img_in, scores, k_mask):
            idx = np.argsort(scores)[::-1][:k_mask]
            im = img_in.clone()
            for ix in idx:
                row, col = int(ix // grid), int(ix % grid)
                im[:, row*p:(row+1)*p, col*p:(col+1)*p] = fill
            return im

        run_acc_a, run_acc_g, run_acc_w = [], [], []

        for r in ratios:
            k_mask = int(n_patches * float(r))
            ca = cg = cw = 0

            for k in range(B):
                img_a = mask_with_scores(images[k], maps_a[k], k_mask).unsqueeze(0)
                img_g = mask_with_scores(images[k], maps_g[k], k_mask).unsqueeze(0)
                img_w = mask_with_scores(images[k], maps_w[k], k_mask).unsqueeze(0)

                lbl_val = int(labels[k].item())
                with torch.no_grad():
                    if model(img_a).argmax(1).item() == lbl_val: ca += 1
                    if model(img_g).argmax(1).item() == lbl_val: cg += 1
                    if model(img_w).argmax(1).item() == lbl_val: cw += 1

            run_acc_a.append(ca / B)
            run_acc_g.append(cg / B)
            run_acc_w.append(cw / B)

        hist_attn.append(run_acc_a)
        hist_grad.append(run_acc_g)
        hist_work.append(run_acc_w)
        print(f"  Ablation run {run_idx+1:02d}/{num_runs} done.")

    return np.asarray(hist_attn), np.asarray(hist_grad), np.asarray(hist_work)


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print(f"Using {device} for {DATASET_NAME}")

    train_loader, test_loader, _ = get_dataloaders(DATASET_NAME, CONFIG)

    model = ThermodynamicViT(CONFIG).to(device)
    optimizer_ = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    train_epoch_loss = []
    train_epoch_acc = []

    print("Training...")
    for ep in range(CONFIG["epochs"]):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = to_label(y, device)
            optimizer_.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer_.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            n_seen += bs
            correct += (logits.argmax(1) == y).sum().item()

        avg_loss = loss_sum / max(1, n_seen)
        avg_acc = correct / max(1, n_seen)
        train_epoch_loss.append(avg_loss)
        train_epoch_acc.append(avg_acc)
        print(f"Epoch {ep+1:03d}/{CONFIG['epochs']}  loss={avg_loss:.6f}  acc={avg_acc:.4f}")

    print("Computing robust faithfulness curves (with Integrated Gradients)...")
    h_a, h_g, h_w = run_robust_ablation(
        model,
        test_loader,
        device,
        num_runs=CONFIG["num_ablation_runs"],
        ig_steps=CONFIG["ig_steps"],
    )

    save_data = {
        "dataset": DATASET_NAME,
        "faith_attn_mean": h_a.mean(axis=0),
        "faith_attn_std": h_a.std(axis=0),
        "faith_grad_mean": h_g.mean(axis=0),
        "faith_grad_std": h_g.std(axis=0),
        "faith_work_mean": h_w.mean(axis=0),
        "faith_work_std": h_w.std(axis=0),
        "train_epoch_loss": np.asarray(train_epoch_loss, dtype=np.float32),
        "train_epoch_acc": np.asarray(train_epoch_acc, dtype=np.float32),
        "ig_steps": int(CONFIG["ig_steps"]),
        "num_ablation_runs": int(CONFIG["num_ablation_runs"]),
        "batch_size": int(CONFIG["batch_size"]),
    }

    fname = os.path.join(OUT_DIR, f"thermo_xai_{DATASET_NAME}.mat")
    sio.savemat(fname, save_data)
    print(f"Saved {fname}.")

if __name__ == "__main__":
    main()