#!/usr/bin/env python3
# scripts/views/preview_view.py

import os, argparse, json, time, random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIEWS_ROOT   = os.path.join(PROJECT_ROOT, "data", "views")

def main():
    ap = argparse.ArgumentParser(description="Preview a random 2x2 grid of images from a view with rewards.")
    ap.add_argument("view_id", type=str, help="Folder under data/views/")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output", type=str, default=None, help="Optional output filename (.png). Defaults inside the view folder.")
    args = ap.parse_args()

    view_dir = os.path.join(VIEWS_ROOT, args.view_id)
    rows_path = os.path.join(view_dir, "rows.parquet")
    if not os.path.exists(rows_path):
        raise FileNotFoundError(f"Missing {rows_path}")

    df = pd.read_parquet(rows_path)

    # Minimal schema checks
    need = {"run_id","sample_id","image_path","reward"}
    missing = need - set(df.columns)
    if missing:
        raise AssertionError(f"rows.parquet missing required columns: {sorted(missing)}")

    # One row per (run_id, sample_id) — the same image appears on all steps
    keep_cols = ["run_id","sample_id","image_path","reward"]
    # Keep raw score too if available (may be 'score_raw' or 'pickscore_raw')
    if "score_raw" in df.columns: keep_cols.append("score_raw")
    if "pickscore_raw" in df.columns and "score_raw" not in df.columns: keep_cols.append("pickscore_raw")

    samples = df[keep_cols].drop_duplicates(subset=["run_id","sample_id"]).reset_index(drop=True)
    if len(samples) == 0:
        raise RuntimeError("No samples found in view.")

    n_slots = args.rows * args.cols
    if len(samples) < n_slots:
        print(f"[warn] view has only {len(samples)} unique samples; grid will use that many.")

    # Deterministic sample with seed
    sample_df = samples.sample(n=min(n_slots, len(samples)), random_state=args.seed)

    # Prepare figure
    fig_w, fig_h = 4 * args.cols, 4 * args.rows
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    # Render each tile
    for ax, (_, row) in zip(axes_flat, sample_df.iterrows()):
        img_abs = os.path.join(view_dir, row["image_path"])
        if not os.path.exists(img_abs):
            ax.set_axis_off()
            ax.set_title(f"[missing]\n{row['run_id']}/{row['sample_id']}", fontsize=9)
            continue

        im = Image.open(img_abs).convert("RGB")
        ax.imshow(im)
        ax.set_axis_off()

        # Build caption: reward and (optionally) raw score
        reward = float(row["reward"])
        if "score_raw" in row and pd.notna(row["score_raw"]):
            cap = f"r={reward:.3f} | s={float(row['score_raw']):.2f}"
        elif "pickscore_raw" in row and pd.notna(row["pickscore_raw"]):
            cap = f"r={reward:.3f} | ps={float(row['pickscore_raw']):.2f}"
        else:
            cap = f"r={reward:.3f}"
        ax.set_title(cap, fontsize=10)

    # If fewer tiles than grid slots, blank the rest
    for ax in axes_flat[len(sample_df):]:
        ax.set_axis_off()

    # Save to view folder (or custom path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_name = args.output or f"preview_{args.view_id}_{timestamp}.png"
    out_path = out_name if os.path.isabs(out_name) else os.path.join(view_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[preview_view] saved {out_path}")

if __name__ == "__main__":
    main()
