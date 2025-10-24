#!/usr/bin/env python3
# scripts/check_value_net.py
import os, json, argparse, torch
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIEWS_ROOT  = os.path.join(PROJECT_ROOT, "data", "views")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models", "value_net")

from train_value_net import LatentHead, ViewDataset, collate  # reuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view_id", type=str)
    ap.add_argument("model_id", type=str)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    view_dir = os.path.join(VIEWS_ROOT, args.view_id)
    rows = pd.read_parquet(os.path.join(view_dir, "rows.parquet"))
    with open(os.path.join(view_dir,"splits.json")) as f: splits = json.load(f)

    # small sample from train pairs
    import random
    pairs = splits["train"][: max(1, args.n // 2)]
    df = rows.merge(pd.DataFrame(pairs, columns=["run_id","sample_id"]), on=["run_id","sample_id"], how="inner")
    mu = float(df["noise_scalar"].mean()); sd = float(df["noise_scalar"].std(ddof=0) or 1.0)
    ds = ViewDataset(view_dir, df.sample(n=min(args.n, len(df)), random_state=123), mu, sd, cache_limit=2)
    xt, noise, y = collate([ds[i] for i in range(min(args.n, len(ds)))])

    ckpt_p = os.path.join(MODELS_ROOT, args.model_id, "ckpt.pt")
    state = torch.load(ckpt_p, map_location="cpu")
    model = LatentHead()
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        pred = model(xt, noise)
    print("pred shape:", tuple(pred.shape), "min/max:", float(pred.min()), float(pred.max()))
    print("target shape:", tuple(y.shape), "min/max:", float(y.min()), float(y.max()))
    assert torch.all(pred > 0), "Outputs must be strictly positive (Softplus)"

if __name__ == "__main__":
    main()
