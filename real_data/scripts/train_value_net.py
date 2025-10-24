#!/usr/bin/env python3
# scripts/train_value_net.py
import os, json, time, math, argparse, hashlib, random, gc
from typing import Dict, Any, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from contextlib import nullcontext

# ---------- Paths ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEWS_ROOT    = os.path.join(PROJECT_ROOT, "data", "views")
MODELS_ROOT   = os.path.join(PROJECT_ROOT, "models", "value_net")

# ---------- Model ----------
class TimeEmbed(nn.Module):
    def __init__(self, fourier_dim=32, mlp_dim=128, out_dim=128):
        super().__init__()
        # fixed Gaussian matrix for Fourier features
        self.register_buffer("B", torch.randn(1, fourier_dim) * 4.0, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(2*fourier_dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, out_dim)
        )
    def forward(self, t_scalar: torch.Tensor):
        # t_scalar: (B,) standardized noise scalar
        t = t_scalar[:, None]  # (B,1)
        feats = torch.cat([torch.sin(t @ self.B * 2*math.pi), torch.cos(t @ self.B * 2*math.pi)], dim=-1)
        return self.mlp(feats)  # (B, out_dim)

class FiLM(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.to_gb = nn.Linear(t_dim, 2*ch)
    def forward(self, x, t_emb):
        gamma, beta = self.to_gb(t_emb).chunk(2, dim=-1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]

class DWConvBlock(nn.Module):
    def __init__(self, ch, t_dim):
        super().__init__()
        self.gn  = nn.GroupNorm(num_groups=max(1, ch // 4), num_channels=ch)
        self.dw  = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.pw  = nn.Conv2d(ch, ch, 1)
        self.film = FiLM(ch, t_dim)
        self.se_fc1 = nn.Linear(ch, max(1, ch//4))
        self.se_fc2 = nn.Linear(max(1, ch//4), ch)
    def forward(self, x, t_emb):
        h = self.gn(x)
        h = F.gelu(h)
        h = self.dw(h); h = F.gelu(h)
        h = self.pw(h)
        # squeeze-excitation
        s = h.mean(dim=(2,3))                   # (B,C)
        s = F.gelu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s))[:, :, None, None]
        h = h * s
        h = self.film(h, t_emb)
        return x + h

class LatentHead(nn.Module):
    """
    h(xt, eta) -> E[exp(r) | xt, eta], positivity enforced via Softplus.
    """
    def __init__(self, in_ch=4, width=128, depth=3, t_dim=128):
        super().__init__()
        self.pre_gn  = nn.GroupNorm(num_groups=1, num_channels=in_ch)
        self.in_proj = nn.Conv2d(in_ch, width, 1)
        self.tok     = TimeEmbed(out_dim=t_dim)
        self.blocks  = nn.ModuleList([DWConvBlock(width, t_dim) for _ in range(depth)])
        self.head    = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1)
        )
        self.softplus = nn.Softplus(beta=1.0)
    def forward(self, xt: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # xt: (B,4,H,W) float32; t_scalar: (B,) standardized noise scalar
        x = self.pre_gn(xt)
        x = F.gelu(self.in_proj(x))
        t_emb = self.tok(t_scalar)              # (B, t_dim)
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = x.mean(dim=(2,3))                   # GAP -> (B,width)
        y = self.head(x)                        # (B,1)
        y = self.softplus(y) + 1e-6             # strictly positive
        return y.squeeze(-1)                    # (B,)

# ---------- Dataset ----------
class ViewDataset(Dataset):
    """
    Streams rows from data/views/<view_id>/rows.parquet and loads latents
    from shard paths stored as relative paths like '.../latents-00012.pt:row=345'.
    """
    def __init__(self, view_dir: str, df: pd.DataFrame, noise_mean: float, noise_std: float, cache_limit: int = 8):
        self.view_dir = view_dir
        self.df = df.reset_index(drop=True)
        self.mu = float(noise_mean)
        self.sd = float(noise_std) if noise_std != 0 else 1.0
        self.cache: Dict[str, Any] = {}
        self.cache_order: list[str] = []
        self.cache_limit = cache_limit

    def __len__(self): return len(self.df)

    def _get_shard(self, rel_with_row: str):
        rel, row_s = rel_with_row.split(":row=")
        row_idx = int(row_s)
        abs_path = os.path.join(self.view_dir, rel)
        # tiny LRU
        if abs_path not in self.cache:
            if len(self.cache_order) >= self.cache_limit:
                evict = self.cache_order.pop(0)
                self.cache.pop(evict, None)
                gc.collect()
            self.cache[abs_path] = torch.load(abs_path, map_location="cpu")
            self.cache_order.append(abs_path)
        return self.cache[abs_path], row_idx

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        shard, idx = self._get_shard(r["latent_path"])
        row = shard[idx]
        xt: torch.Tensor = row["xt"]  # [4,H,W] float32 CPU
        noise = (float(r["noise_scalar"]) - self.mu) / self.sd
        y = math.exp(float(r["reward"]))  # exp(r)
        return xt, torch.tensor(noise, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def collate(batch):
    xt = torch.stack([b[0] for b in batch], dim=0)     # (B,4,H,W)
    noise = torch.stack([b[1] for b in batch], dim=0)  # (B,)
    y = torch.stack([b[2] for b in batch], dim=0)      # (B,)
    return xt, noise, y

# ---------- Utils ----------
def load_view(view_id: str):
    view_dir = os.path.join(VIEWS_ROOT, view_id)
    rows_p   = os.path.join(view_dir, "rows.parquet")
    splits_p = os.path.join(view_dir, "splits.json")
    manifest_p = os.path.join(view_dir, "manifest.json")
    for p in [rows_p, splits_p, manifest_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")
    rows = pd.read_parquet(rows_p)
    with open(splits_p, "r") as f: splits = json.load(f)
    with open(manifest_p, "r") as f: man = json.load(f)
    return view_dir, rows, splits, man

def subset_by_pairs(rows: pd.DataFrame, pairs: list[Tuple[str,str]]) -> pd.DataFrame:
    # pairs are [ [run_id, sample_id], ... ]
    key = pd.MultiIndex.from_frame(rows[["run_id","sample_id"]])
    target = pd.MultiIndex.from_tuples([tuple(p) for p in pairs])
    mask = key.isin(target)
    return rows[mask].copy()

def make_sampler_uniform_stepidx(df: pd.DataFrame) -> WeightedRandomSampler:
    # weight each example inversely by count(step_idx)
    counts = df["step_idx"].value_counts()
    w = df["step_idx"].map(lambda s: 1.0 / counts[s]).to_numpy()
    w_tensor = torch.tensor(w, dtype=torch.double)
    sampler = WeightedRandomSampler(w_tensor, num_samples=len(df), replacement=True)
    return sampler

def save_manifest(out_dir: str, payload: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(payload, f, indent=2)

# ---------- Train ----------
def train_one_epoch(model, loader, opt, scaler, device, amp, loss_name):
    model.train()
    total = 0.0
    n = 0
    for xt, noise, y in loader:
        xt = xt.to(device)
        noise = noise.to(device)
        y = y.to(device)
        ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if amp else nullcontext()
        with ctx:
            pred = model(xt, noise)
            if loss_name == "l1":
                loss = F.l1_loss(pred, y)
            else:
                loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        total += loss.item() * xt.size(0)
        n += xt.size(0)
    return total / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, device, loss_name):
    model.eval()
    total = 0.0
    n = 0
    for xt, noise, y in loader:
        xt = xt.to(device)
        noise = noise.to(device)
        y = y.to(device)
        pred = model(xt, noise)
        loss = F.l1_loss(pred, y) if loss_name == "l1" else F.mse_loss(pred, y)
        total += loss.item() * xt.size(0)
        n += xt.size(0)
    return total / max(1, n)

def main():
    ap = argparse.ArgumentParser(description="Train value net on a view")
    ap.add_argument("view_id", type=str, help="Folder under data/views/")
    ap.add_argument("--model-id", type=str, default=None, help="Subfolder under models/value_net/. Default: auto timestamp")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--t-dim", type=int, default=128)
    ap.add_argument("--loss", type=str, default="l1", choices=["l1","mse"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (fp16)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cache-limit", type=int, default=8, help="Number of latent shards to keep in RAM")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load view
    view_dir, rows, splits, man = load_view(args.view_id)

    # Basic schema checks
    needed = {"run_id","sample_id","step_idx","timestep","noise_scalar","noise_type","latent_path","reward"}
    missing = needed - set(rows.columns)
    if missing:
        raise AssertionError(f"rows.parquet missing columns: {sorted(missing)}")

    # Build splits at (run_id, sample_id)
    train_df = subset_by_pairs(rows, splits["train"])
    val_df   = subset_by_pairs(rows, splits["val"]) if len(splits.get("val",[])) else rows.iloc[0:0]
    test_df  = subset_by_pairs(rows, splits["test"]) if len(splits.get("test",[])) else rows.iloc[0:0]

    # Compute noise normalization on training split (store; don't rely solely on manifest)
    mu = float(train_df["noise_scalar"].mean())
    sd = float(train_df["noise_scalar"].std(ddof=0) or 1.0)

    # Datasets / loaders
    train_ds = ViewDataset(view_dir, train_df, noise_mean=mu, noise_std=sd, cache_limit=args.cache_limit)
    val_ds   = ViewDataset(view_dir, val_df,   noise_mean=mu, noise_std=sd, cache_limit=args.cache_limit) if len(val_df) else None
    test_ds  = ViewDataset(view_dir, test_df,  noise_mean=mu, noise_std=sd, cache_limit=args.cache_limit) if len(test_df) else None

    sampler = make_sampler_uniform_stepidx(train_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate, drop_last=False) if test_ds else None

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = LatentHead(in_ch=4, width=args.width, depth=args.depth, t_dim=args.t_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # Train
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device, args.amp, args.loss)
        if val_loader:
            val_loss = eval_epoch(model, val_loader, device, args.loss)
        else:
            val_loss = float("nan")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        print(f"[epoch {epoch}] train {tr_loss:.6f} | val {val_loss:.6f}")

        if val_loader and val_loss < best_val:
            best_val = val_loss
            # save interim best
            tag = args.model_id or time.strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = os.path.join(MODELS_ROOT, tag)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ckpt.pt"))

    # Final eval on test if present
    test_loss = eval_epoch(model, test_loader, device, args.loss) if test_loader and len(test_df) else float("nan")
    print(f"[final] test {test_loss:.6f}")

    # Persist final artifacts
    tag = args.model_id or time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(MODELS_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ckpt.pt"))

    # Write manifest
    manifest = {
        "model_id": tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": "latent_conv_film",
        "hparams": {
            "width": args.width, "depth": args.depth, "t_dim": args.t_dim,
            "loss": args.loss, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "amp": args.amp,
        },
        "data": {
            "view_id": args.view_id,
            "rows_path": os.path.join("..","..","data","views", args.view_id, "rows.parquet"),
            "splits_path": os.path.join("..","..","data","views", args.view_id, "splits.json"),
            "rows_sha256_csv": _hash_rows_csv(os.path.join(VIEWS_ROOT, args.view_id, "rows.parquet")),
            "noise_type": rows["noise_type"].iloc[0] if len(rows) else "unknown",
            "noise_norm": {"mean": mu, "std": sd},
        },
        "metrics": {
            "best_val_loss": best_val if val_loader else None,
            "test_loss": test_loss if test_loader else None,
            "history": history,
        },
        "env": {
            "torch": torch.__version__,
        }
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[save] model -> {out_dir}")
    print(f"[norm] noise mean={mu:.6f}, std={sd:.6f}")

def _hash_rows_csv(rows_path: str) -> str:
    try:
        rows = pd.read_parquet(rows_path)
        b = rows.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return "unavailable"

if __name__ == "__main__":
    main()
