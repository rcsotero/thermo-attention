"""
THERMODYNAMICS OF ATTENTION- HYSTERESIS & LANDAU POTENTIALS

"""

import os
import math
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Optional dependency for OrganAMNIST
try:
    from medmnist import OrganAMNIST
except Exception:
    OrganAMNIST = None

# ============================================================
# 1) CONFIGURATION (SELECT DATASET HERE)
# ============================================================
DATASET_NAME = "OrganAMNIST"  # Options: "MNIST", "CIFAR10", "OrganAMNIST"

EXPERIMENT_NAME = f"hysteresis_landau_{DATASET_NAME.lower()}_v3"
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

SAVE_DIR = os.path.join(SCRIPT_DIR, f"results_{EXPERIMENT_NAME}")
os.makedirs(SAVE_DIR, exist_ok=True)

# 10 Seeds for statistical smoothing
SEEDS = list(range(42, 52))

# Temperature Sweep
TEMPS_HYST = np.geomspace(0.01, 10.0, 40).tolist()

# Landau Potential Config
HIST_BINS = 100
HIST_RANGE = (0.0, 1.0)  # Order Parameter in [0,1]

# Dataset-specific config
BASE_CONFIG = {
    "d_model": 64,
    "n_head": 4,
    "n_layer": 2,
    "lr": 1e-3,
    "batch_size": 512,
    "weight_decay": 0.0001,
    "pretrain_epochs": 2,
    "step_epochs": 2,
    "workers": 1 if torch.backends.mps.is_available() else 2,
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
        raise ImportError("OrganAMNIST selected but medmnist is not installed. Install medmnist or choose MNIST/CIFAR10.")
    CONFIG = {
        **BASE_CONFIG,
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 11,
    }
else:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

# ============================================================
# 2) MODEL
# ============================================================
class ThermalAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.head_dim = d_model // n_head
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.batch_entropies = None

    def forward(self, x, T=1.0, track_stats=False):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]

        raw_dot = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = raw_dot / max(T, 1e-6)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        att = F.softmax(logits, dim=-1)

        if track_stats:
            n_keys = att.size(-1)
            S_raw = -(att * torch.log(att + 1e-9)).sum(dim=-1)
            S_norm = S_raw / math.log(n_keys)
            self.batch_entropies = S_norm.mean(dim=(1, 2))

        out = (att @ v).transpose(1, 2).reshape(B, S, C)
        return self.proj(out)


class UniversalViT(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.patch_size = cfg["patch_size"]
        self.n_patches = (cfg["img_size"] // self.patch_size) ** 2
        self.patch_embed = nn.Linear(cfg["in_channels"] * (self.patch_size ** 2), cfg["d_model"])
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, cfg["d_model"]))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": ThermalAttention(cfg["d_model"], cfg["n_head"]),
                "ln1": nn.LayerNorm(cfg["d_model"]),
                "mlp": nn.Sequential(
                    nn.Linear(cfg["d_model"], 4 * cfg["d_model"]),
                    nn.GELU(),
                    nn.Linear(4 * cfg["d_model"], cfg["d_model"]),
                ),
                "ln2": nn.LayerNorm(cfg["d_model"]),
            }) for _ in range(cfg["n_layer"])
        ])
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], num_classes)

    def forward(self, x, T=1.0, track_stats=False):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        x = self.patch_embed(x) + self.pos_emb
        for b in self.blocks:
            x = x + b["attn"](b["ln1"](x), T=T, track_stats=track_stats)
            x = x + b["mlp"](b["ln2"](x))
        return self.head(self.ln_f(x).mean(dim=1))


# ============================================================
# 3) DATASET LOADER
# ============================================================
def load_datasets():
    if DATASET_NAME == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_full = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        test_full = datasets.MNIST("./data", train=False, download=True, transform=tfm)
        return train_full, test_full

    if DATASET_NAME == "CIFAR10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        train_full = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        test_full = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
        return train_full, test_full

    if DATASET_NAME == "OrganAMNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_full = OrganAMNIST(split="train", download=True, transform=tfm, root=SAVE_DIR)
        test_full = OrganAMNIST(split="test", download=True, transform=tfm, root=SAVE_DIR)
        return train_full, test_full

    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")


def maybe_fix_labels(y):
    # OrganAMNIST often yields y shape [B,1]
    if DATASET_NAME == "OrganAMNIST":
        if isinstance(y, torch.Tensor):
            return y.squeeze().long()
    return y.long()


# ============================================================
# 4) MAIN LOOP
# ============================================================
def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Running {DATASET_NAME} Landau Hysteresis on {device}")

    train_full, test_full = load_datasets()

    # Histogram config
    bin_edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    global_hist_heat = np.zeros((len(TEMPS_HYST), HIST_BINS), dtype=np.float64)
    global_hist_cool = np.zeros((len(TEMPS_HYST), HIST_BINS), dtype=np.float64)
    results_agg = {"heat_phi": [], "cool_phi": []}

    for seed in tqdm(SEEDS, desc="Seeds"):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(
            train_full,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["workers"],
            pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_full,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["workers"],
            pin_memory=(device.type == "cuda"),
        )

        model = UniversalViT(CONFIG, num_classes=CONFIG["num_classes"]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        def run_step(train: bool, T: float):
            if train:
                model.train()
                for x, y in train_loader:
                    x = x.to(device)
                    y = maybe_fix_labels(y.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    out = model(x, T=T, track_stats=False)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                return None

            model.eval()
            phis = []
            with torch.no_grad():
                for x, _y in test_loader:
                    x = x.to(device)
                    _ = model(x, T=T, track_stats=True)
                    S = torch.stack([b["attn"].batch_entropies for b in model.blocks]).mean(0)
                    phis.extend((1.0 - S).cpu().tolist())
            return np.asarray(phis, dtype=np.float64)

        # Pretrain at lowest temperature
        for _ in range(CONFIG["pretrain_epochs"]):
            run_step(True, TEMPS_HYST[0])

        # HEATING
        seed_heat_phi_means = []
        last_heat_phis = None

        for i, T in enumerate(TEMPS_HYST):
            for _ in range(CONFIG["step_epochs"]):
                run_step(True, T)
            raw_phis = run_step(False, T)

            hist, _ = np.histogram(raw_phis, bins=HIST_BINS, range=HIST_RANGE)
            global_hist_heat[i] += hist
            seed_heat_phi_means.append(float(np.mean(raw_phis)))

            if i == len(TEMPS_HYST) - 1:
                last_heat_phis = raw_phis

        # COOLING
        seed_cool_phi_means = [0.0] * len(TEMPS_HYST)
        max_t_idx = len(TEMPS_HYST) - 1

        # Carry over max temp evaluation from heating
        hist_max, _ = np.histogram(last_heat_phis, bins=HIST_BINS, range=HIST_RANGE)
        global_hist_cool[max_t_idx] += hist_max
        seed_cool_phi_means[max_t_idx] = float(np.mean(last_heat_phis))

        # Walk down excluding max temp
        for ii, T in enumerate(reversed(TEMPS_HYST[:-1])):
            original_idx = (len(TEMPS_HYST) - 2) - ii

            for _ in range(CONFIG["step_epochs"]):
                run_step(True, T)
            raw_phis = run_step(False, T)

            hist, _ = np.histogram(raw_phis, bins=HIST_BINS, range=HIST_RANGE)
            global_hist_cool[original_idx] += hist
            seed_cool_phi_means[original_idx] = float(np.mean(raw_phis))

        results_agg["heat_phi"].append(seed_heat_phi_means)
        results_agg["cool_phi"].append(seed_cool_phi_means)

    # Normalize distributions safely
    row_sums_heat = global_hist_heat.sum(axis=1, keepdims=True)
    P_phi_heat = np.divide(
        global_hist_heat,
        row_sums_heat,
        out=np.zeros_like(global_hist_heat, dtype=np.float64),
        where=row_sums_heat != 0,
    )

    row_sums_cool = global_hist_cool.sum(axis=1, keepdims=True)
    P_phi_cool = np.divide(
        global_hist_cool,
        row_sums_cool,
        out=np.zeros_like(global_hist_cool, dtype=np.float64),
        where=row_sums_cool != 0,
    )

    final_res = {
        "dataset": DATASET_NAME,
        "config": {k: (v if v is not None else "None") for k, v in CONFIG.items()},
        "temps": np.array(TEMPS_HYST, dtype=np.float64),

        "phi_heat_mean": np.array(np.mean(results_agg["heat_phi"], axis=0), dtype=np.float64),
        "phi_heat_std":  np.array(np.std(results_agg["heat_phi"], axis=0), dtype=np.float64),

        "phi_cool_mean": np.array(np.mean(results_agg["cool_phi"], axis=0), dtype=np.float64),
        "phi_cool_std":  np.array(np.std(results_agg["cool_phi"], axis=0), dtype=np.float64),

        "dist_heat": P_phi_heat,
        "dist_cool": P_phi_cool,

        "hist_edges": bin_edges.astype(np.float64),
        "hist_centers": bin_centers.astype(np.float64),
    }

    mat_path = os.path.join(SAVE_DIR, f"{DATASET_NAME.lower()}_hysteresis_landau_v3.mat")
    scipy.io.savemat(mat_path, {"results": final_res}, do_compression=True)
    print(f"{DATASET_NAME} Landau Hysteresis Done. Saved to: {mat_path}")


if __name__ == "__main__":
    main()
