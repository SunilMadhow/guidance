#!/usr/bin/env python3
# scripts/views/make_view.py

import os, re, json, argparse, hashlib, time, math, random
from typing import Dict, Any, List, Tuple
import pandas as pd

# Resolve PROJECT_ROOT no matter where this file lives (scripts/views/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "data", "runs")
VIEWS_ROOT = os.path.join(PROJECT_ROOT, "data", "views")

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def rel_from_view(path_abs: str, view_dir: str) -> str:
    return os.path.relpath(path_abs, start=view_dir)

def load_run_frames(run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = os.path.join(RUNS_ROOT, run_id)
    steps_p = os.path.join(run_dir, "steps.parquet")
    rewards_p = os.path.join(run_dir, "rewards.parquet")
    if not os.path.exists(steps_p):
        raise FileNotFoundError(f"[{run_id}] missing {steps_p}")
    if not os.path.exists(rewards_p):
        raise FileNotFoundError(f"[{run_id}] missing {rewards_p}")
    return pd.read_parquet(steps_p), pd.read_parquet(rewards_p)

def filter_rows(df: pd.DataFrame,
                reward_min: float = None,
                step_idx_range: Tuple[int,int] = None,
                prompts_regex: str = None) -> pd.DataFrame:
    out = df
    if reward_min is not None:
        out = out[out["reward"] >= reward_min]
    if step_idx_range is not None:
        a, b = step_idx_range
        out = out[(out["step_idx"] >= a) & (out["step_idx"] <= b)]
    if prompts_regex:
        rx = re.compile(prompts_regex)
        out = out[out["prompt"].apply(lambda s: bool(rx.match(str(s))))]
    return out

def build_splits(pairs: List[Tuple[str,str]],
                 val_frac: float,
                 test_frac: float,
                 seed: int = 123) -> Dict[str, List[Tuple[str,str]]]:
    assert 0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1
    rng = random.Random(seed)
    dedup = sorted(set(pairs))
    rng.shuffle(dedup)
    n = len(dedup)
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    val = dedup[:n_val]
    test = dedup[n_val:n_val+n_test]
    train = dedup[n_val+n_test:]
    return {"train": train, "val": val, "test": test}

def main():
    ap = argparse.ArgumentParser(description="Build a frozen multi-run view under data/views/<view_id>")
    ap.add_argument("view_id", type=str, help="Name of the view to create (folder under data/views/)")
    ap.add_argument("--runs", nargs="+", required=True, help="One or more run_id values")
    ap.add_argument("--reward-min", type=float, default=None, help="Drop rows with reward < this value")
    ap.add_argument("--step-idx-range", type=str, default=None, help="e.g., 0:24 to keep a contiguous range")
    ap.add_argument("--prompts-regex", type=str, default=None, help="Regex to match prompts to keep")
    ap.add_argument("--noise-type", type=str, default="logsnr", help="Require rows to have this noise_type")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--balance-by-run", action="store_true", help="Interleave runs when building rows for stability")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    view_dir = os.path.join(VIEWS_ROOT, args.view_id)
    os.makedirs(view_dir, exist_ok=True)

    # Parse range
    step_range = None
    if args.step_idx_range:
        parts = args.step_idx_range.split(":")
        if len(parts) != 2:
            raise ValueError("--step-idx-range must be like A:B")
        step_range = (int(parts[0]), int(parts[1]))

    sources_meta = []
    frames = []
    for run_id in args.runs:
        run_dir = os.path.join(RUNS_ROOT, run_id)
        steps, rewards = load_run_frames(run_id)

        # Enforce noise_type if requested
        if args.noise_type:
            if "noise_type" not in steps.columns:
                raise AssertionError(f"[{run_id}] steps.parquet missing 'noise_type'")
            steps = steps[steps["noise_type"] == args.noise_type]
            if steps.empty:
                raise AssertionError(f"[{run_id}] no rows with noise_type={args.noise_type}")

        # Join steps (many) to rewards (one per sample)
        df = steps.merge(
            rewards[["run_id","sample_id","reward","pickscore_raw","prompt","negative_prompt"]] \
                .rename(columns={"pickscore_raw":"score_raw"}),
            on=["run_id","sample_id"],
            how="inner",
            validate="many_to_one",
        )

        # Make absolute then re-relativize paths relative to the view folder
        df["latent_path"] = df["latent_path"].apply(lambda rel: os.path.join(run_dir, rel))
        df["image_path"]  = df["image_path"].apply(lambda rel: os.path.join(run_dir, rel))
        df["latent_path"] = df["latent_path"].apply(lambda p: rel_from_view(p, view_dir))
        df["image_path"]  = df["image_path"].apply(lambda p: rel_from_view(p, view_dir))

        # Filter
        df = filter_rows(df,
                         reward_min=args.reward_min,
                         step_idx_range=step_range,
                         prompts_regex=args.prompts_regex)

        if df.empty:
            raise AssertionError(f"[{run_id}] all rows filtered out")

        # Record source hashes for lineage
        steps_p = os.path.join(run_dir, "steps.parquet")
        rewards_p = os.path.join(run_dir, "rewards.parquet")
        sources_meta.append({
            "run_id": run_id,
            "steps_parquet": rel_from_view(steps_p, view_dir),
            "steps_sha256": sha256_file(steps_p),
            "rewards_parquet": rel_from_view(rewards_p, view_dir),
            "rewards_sha256": sha256_file(rewards_p),
            "rows_after_filter": int(len(df)),
            "unique_samples_after_filter": int(df[["run_id","sample_id"]].drop_duplicates().shape[0]),
        })
        frames.append(df)

    # Concatenate deterministically
    if args.balance_by_run:
        # simple interleave by run to reduce locality bias
        chunks = [f.sort_values(["sample_id","step_idx"]).reset_index(drop=True) for f in frames]
        interleaved = []
        idxs = [0]*len(chunks)
        total = sum(len(c) for c in chunks)
        while len(interleaved) < total:
            for i, c in enumerate(chunks):
                if idxs[i] < len(c):
                    interleaved.append(c.iloc[idxs[i]])
                    idxs[i] += 1
        all_rows = pd.DataFrame(interleaved)
    else:
        all_rows = pd.concat(frames, ignore_index=True)
        all_rows = all_rows.sort_values(["run_id","sample_id","step_idx"]).reset_index(drop=True)

    # Minimal schema normalization
    needed_cols = [
        "run_id","sample_id","step_idx","timestep",
        "noise_scalar","noise_type",
        "latent_path","image_path",
        "prompt","negative_prompt",
        "score_raw","reward",
    ]
    missing = [c for c in needed_cols if c not in all_rows.columns]
    if missing:
        raise AssertionError(f"Missing columns in merged rows: {missing}")

    # Build splits at (run_id, sample_id) level
    pairs = list(all_rows[["run_id","sample_id"]].drop_duplicates().itertuples(index=False, name=None))
    splits = build_splits(pairs, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    # Write rows.parquet (full per-step table)
    rows_path = os.path.join(view_dir, "rows.parquet")
    all_rows.to_parquet(rows_path, index=False)

    # Write splits.json
    splits_json = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "seed": args.seed,
        "level": "sample",  # pairs of (run_id, sample_id)
    }
    with open(os.path.join(view_dir, "splits.json"), "w") as f:
        json.dump(splits_json, f, indent=2)

    # Compute a content hash of rows.parquet for lineage (hash of CSV view for stability)
    try:
        rows_bytes = pd.read_parquet(rows_path).to_csv(index=False).encode("utf-8")
        rows_sha256 = hashlib.sha256(rows_bytes).hexdigest()
    except Exception:
        rows_sha256 = "unavailable"

    # Noise stats (handy for training standardization)
    mu = float(all_rows["noise_scalar"].mean())
    sd = float(all_rows["noise_scalar"].std(ddof=0) or 1.0)

    # Write manifest.json
    manifest = {
        "view_id": args.view_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": args.runs,
        "sources": sources_meta,
        "filters": {
            "reward_min": args.reward_min,
            "step_idx_range": args.step_idx_range,
            "prompts_regex": args.prompts_regex,
        },
        "constraints": {
            "noise_type": args.noise_type,
        },
        "row_count": int(len(all_rows)),
        "unique_samples": int(len(pairs)),
        "rows_sha256_csv": rows_sha256,
        "noise_stats": {"mean": mu, "std": sd},
        "paths": {
            "rows_parquet": rel_from_view(rows_path, view_dir),
            "splits_json": "splits.json",
        },
        "schema": needed_cols,
    }
    with open(os.path.join(view_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[make_view] Wrote view '{args.view_id}' to {view_dir}")
    print(f"[make_view] rows: {len(all_rows)}, samples: {len(pairs)}, noise_mean={mu:.4f}, noise_std={sd:.4f}")
    print(f"[make_view] rows_sha256_csv={rows_sha256}")

if __name__ == "__main__":
    main()
