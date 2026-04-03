"""
THERMODYNAMICS OF ATTENTION — XAI FAITHFULNESS (PRETRAINED MODELS)

"""

import os
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm
from timm.models.vision_transformer import Attention as TimmAttention

# ============================================================
# 1) CONFIGURATION
# ============================================================

# Models to evaluate (any timm ViT with patch embedding)
MODEL_NAMES = [
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",     
]

# Path to dataset root — should contain  val/  with 1000 class subfolders
DATA_DIR = "./data"

# Evaluation parameters
BATCH_SIZE = 32
NUM_ABLATION_RUNS = 10          # number of evaluation batches
IG_STEPS = 20                   # interpolation steps for Integrated Gradients
SEED = 42
NUM_WORKERS = 4

# Output directory
OUT_DIR = "./xai_pretrained_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ImageNet normalization (standard for pretrained timm models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ============================================================
# 2) ATTENTION PATCHING — EXTRACT ATTENTION MAPS FROM TIMM VITS
# ============================================================
class AttentionMapExtractor:
    """
    Patches timm Attention modules to disable fused attention and
    cache the raw attention weights from the last transformer block.

    Only the last block's attention is cached (for CLS-token XAI).
    All blocks use explicit softmax path so attention is well-defined.
    """

    def __init__(self, model):
        self.model = model
        self.last_attn_map = None  # [B, H, S, S]
        self._patch_all_blocks()

    def _patch_all_blocks(self):
        blocks = self.model.blocks
        n_blocks = len(blocks)

        for idx, block in enumerate(blocks):
            attn_module = block.attn
            is_last = (idx == n_blocks - 1)

            # Disable fused / flash attention if available
            if hasattr(attn_module, 'fused_attn'):
                attn_module.fused_attn = False

            self._replace_forward(attn_module, cache_map=is_last)

    def _replace_forward(self, mod, cache_map=False):
        extractor = self  # closure reference
        scale = mod.scale
        num_heads = mod.num_heads
        head_dim = getattr(mod, 'head_dim', mod.qkv.in_features // mod.num_heads)

        def thermal_forward(x, **kwargs):
            # kwargs absorbs attn_mask, is_causal from newer timm versions
            B, N, C = x.shape
            qkv = mod.qkv(x).reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            logits = (q * scale) @ k.transpose(-2, -1)
            attn = logits.softmax(dim=-1)

            if cache_map:
                extractor.last_attn_map = attn.detach()

            attn = mod.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = mod.proj(x)
            if hasattr(mod, 'proj_drop'):
                x = mod.proj_drop(x)
            return x

        mod.forward = thermal_forward

    def clear(self):
        self.last_attn_map = None


# ============================================================
# 3) XAI METHODS
# ============================================================
def get_patch_grid(model):
    """Return (patch_size, grid_h, grid_w) for a timm ViT."""
    pe = model.patch_embed
    patch_size = pe.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    img_size = pe.img_size
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[0]
    grid = img_size // patch_size
    return patch_size, grid, grid


@torch.no_grad()
def compute_patch_work(model, img, label, criterion, device, patch_size, grid):
    """
    Thermodynamic work per patch:  W_p = L(mask_p) - L(original).
    Masking fills with normalized zero (dataset mean after normalization).
    """
    model.eval()
    x = img.to(device)
    y = torch.tensor([label], device=device).long()

    loss_base = criterion(model(x.unsqueeze(0)), y).item()

    work_map = np.zeros((grid, grid), dtype=np.float32)
    p = patch_size

    for i in range(grid):
        for j in range(grid):
            masked = x.clone()
            masked[:, i * p:(i + 1) * p, j * p:(j + 1) * p] = 0.0
            loss_m = criterion(model(masked.unsqueeze(0)), y).item()
            work_map[i, j] = loss_m - loss_base

    return work_map


def compute_integrated_gradients(model, img, label, criterion, device,
                                  patch_size, grid, steps=20):
    """
    Integrated Gradients (Sundararajan et al., 2017).
    Baseline = zero (dataset mean in normalized space).
    Returns patch-level IG importance map.
    """
    model.eval()
    x = img.clone().detach().to(device)
    y = torch.tensor([label], device=device).long()
    baseline = torch.zeros_like(x)

    accumulated_grads = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        x_step = baseline + alpha * (x - baseline)
        x_step.requires_grad_(True)
        out = model(x_step.unsqueeze(0))
        loss = criterion(out, y)
        grad_x, = torch.autograd.grad(loss, x_step,
                                       retain_graph=False, create_graph=False)
        accumulated_grads += grad_x.detach()

    avg_grads = accumulated_grads / float(steps)
    ig = (x - baseline) * avg_grads
    ig_full = ig.abs().sum(dim=0)  # [H, W]

    p = patch_size
    ig_map = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            ig_map[i, j] = ig_full[i * p:(i + 1) * p, j * p:(j + 1) * p].sum().item()

    return ig_map


@torch.no_grad()
def get_cls_attention(extractor, model, img, device, grid):
    """
    Mean attention from CLS token (position 0) to all patches
    in the last transformer block, averaged across heads.
    """
    model.eval()
    extractor.clear()
    _ = model(img.unsqueeze(0).to(device))
    att = extractor.last_attn_map  # [1, H, S, S]
    if att is None:
        raise RuntimeError("Attention map not captured. Check patching.")
    # CLS query (pos 0) attending to patch keys (pos 1:)
    cls_to_patches = att[0, :, 0, 1:].mean(dim=0).cpu().numpy()  # [n_patches]
    return cls_to_patches.reshape(grid, grid)


# ============================================================
# 4) ABLATION PROTOCOL
# ============================================================
def run_ablation(model, extractor, dataloader, device, num_runs, ig_steps):
    """
    Progressive masking ablation: for each XAI method, rank patches
    by importance, mask top-k%, and measure accuracy degradation.
    """
    criterion = nn.CrossEntropyLoss()
    patch_size, grid, _ = get_patch_grid(model)
    n_patches = grid * grid

    ratios = np.linspace(0.0, 0.9, 10)
    hist_attn, hist_grad, hist_work = [], [], []
    iterator = iter(dataloader)

    for run_idx in range(num_runs):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            images, labels = next(iterator)

        images = images.to(device)
        labels = labels.to(device).long()
        B = images.size(0)

        # Compute importance maps for each image in batch
        maps_w, maps_g, maps_a = [], [], []
        for k in range(B):
            lbl = labels[k].item()
            maps_w.append(compute_patch_work(
                model, images[k], lbl, criterion, device, patch_size, grid
            ).flatten())
            maps_g.append(compute_integrated_gradients(
                model, images[k], lbl, criterion, device,
                patch_size, grid, steps=ig_steps
            ).flatten())
            maps_a.append(get_cls_attention(
                extractor, model, images[k], device, grid
            ).flatten())

        maps_w = np.asarray(maps_w)
        maps_g = np.asarray(maps_g)
        maps_a = np.asarray(maps_a)

        def mask_and_predict(img, scores, k_mask):
            idx = np.argsort(scores)[::-1][:k_mask]
            im = img.clone()
            for ix in idx:
                row, col = int(ix // grid), int(ix % grid)
                im[:, row * patch_size:(row + 1) * patch_size,
                      col * patch_size:(col + 1) * patch_size] = 0.0
            return im

        run_acc_a, run_acc_g, run_acc_w = [], [], []

        for r in ratios:
            k_mask = int(n_patches * float(r))
            ca = cg = cw = 0

            for k in range(B):
                img_a = mask_and_predict(images[k], maps_a[k], k_mask).unsqueeze(0)
                img_g = mask_and_predict(images[k], maps_g[k], k_mask).unsqueeze(0)
                img_w = mask_and_predict(images[k], maps_w[k], k_mask).unsqueeze(0)

                lbl_val = int(labels[k].item())
                with torch.no_grad():
                    if model(img_a).argmax(1).item() == lbl_val:
                        ca += 1
                    if model(img_g).argmax(1).item() == lbl_val:
                        cg += 1
                    if model(img_w).argmax(1).item() == lbl_val:
                        cw += 1

            run_acc_a.append(ca / B)
            run_acc_g.append(cg / B)
            run_acc_w.append(cw / B)

        hist_attn.append(run_acc_a)
        hist_grad.append(run_acc_g)
        hist_work.append(run_acc_w)
        print(f"  Ablation batch {run_idx + 1}/{num_runs} done.")

    return np.asarray(hist_attn), np.asarray(hist_grad), np.asarray(hist_work)


# ============================================================
# 5) DATASET
# ============================================================
def get_eval_loader():
    """Load ImageNet validation set."""
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_dir = os.path.join(DATA_DIR, "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet val directory not found at {val_dir}. "
            f"Set DATA_DIR to the folder containing val/."
        )
    dset = datasets.ImageFolder(val_dir, transform=tfm)

    # Use a random subset for efficiency (same subset across models)
    rng = np.random.default_rng(SEED)
    max_images = NUM_ABLATION_RUNS * BATCH_SIZE
    if len(dset) > max_images:
        indices = rng.choice(len(dset), size=max_images, replace=False).tolist()
        dset = Subset(dset, indices)

    loader = DataLoader(
        dset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader


# ============================================================
# 6) MAIN
# ============================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Dataset: ImageNet")

    eval_loader = get_eval_loader()

    for model_name in MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load pretrained model
        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device).eval()

        # Verify it's a ViT with patch embedding
        if not hasattr(model, 'patch_embed') or not hasattr(model, 'blocks'):
            print(f"  SKIP: {model_name} does not appear to be a standard timm ViT.")
            continue

        patch_size, grid, _ = get_patch_grid(model)
        n_patches = grid * grid
        print(f"  Patch size: {patch_size}x{patch_size}, Grid: {grid}x{grid}, "
              f"Patches: {n_patches}")

        # Patch attention modules for map extraction
        extractor = AttentionMapExtractor(model)

        # Baseline accuracy (no masking)
        print("  Computing baseline accuracy...")
        correct = total = 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        baseline_acc = correct / max(1, total)
        print(f"  Baseline accuracy: {baseline_acc:.4f}")

        # Run faithfulness ablation
        print("  Running faithfulness ablation...")
        h_a, h_g, h_w = run_ablation(
            model, extractor, eval_loader, device,
            num_runs=NUM_ABLATION_RUNS, ig_steps=IG_STEPS,
        )

        # Save results
        save_data = {
            "model_name": model_name,
            "dataset": "ImageNet",
            "baseline_acc": float(baseline_acc),
            "n_patches": int(n_patches),
            "patch_size": int(patch_size),
            "grid_size": int(grid),
            "faith_attn_mean": h_a.mean(axis=0),
            "faith_attn_std": h_a.std(axis=0),
            "faith_grad_mean": h_g.mean(axis=0),
            "faith_grad_std": h_g.std(axis=0),
            "faith_work_mean": h_w.mean(axis=0),
            "faith_work_std": h_w.std(axis=0),
            "masking_ratios": np.linspace(0.0, 0.9, 10),
            "ig_steps": int(IG_STEPS),
            "num_ablation_runs": int(NUM_ABLATION_RUNS),
            "batch_size": int(BATCH_SIZE),
        }

        safe_name = model_name.replace("/", "_")
        fname = os.path.join(OUT_DIR, f"xai_pretrained_{safe_name}_imagenet.mat")
        sio.savemat(fname, save_data)
        print(f"  Saved: {fname}")

        # Print summary
        auc_w = float(np.trapz(h_w.mean(axis=0), np.linspace(0, 0.9, 10)))
        auc_a = float(np.trapz(h_a.mean(axis=0), np.linspace(0, 0.9, 10)))
        auc_g = float(np.trapz(h_g.mean(axis=0), np.linspace(0, 0.9, 10)))
        print(f"  AUC — Work: {auc_w:.4f}  Attention: {auc_a:.4f}  IG: {auc_g:.4f}")
        print(f"  (Lower AUC = more faithful)")

    print("\nAll models done.")


if __name__ == "__main__":
    main()
