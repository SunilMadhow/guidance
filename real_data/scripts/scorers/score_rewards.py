#!/usr/bin/env python3
# scripts/score_rewards.py

import os, sys, json, argparse, hashlib, time
from typing import List, Sequence, Dict, Any, Optional

import pandas as pd
from PIL import Image
from PIL import ImageStat
from tqdm import tqdm

from base import TextImageScorer
from pickscore import PickScoreScorer
from pixel_average_scorer import PixelAvgMockScorer



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "data", "runs")

# -----------------------------
# Helpers
# -----------------------------

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def default_transform(score_raw: float, offset: float, scale: float) -> float:
    """reward = (score_raw - offset) * scale"""
    return (score_raw - offset) * scale

def collect_final_images_and_prompts(steps_df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """
    Build one row per (run_id, sample_id) with image path + prompt text.
    """
    base = (
        steps_df[["run_id", "sample_id", "image_path"]]
        .drop_duplicates()
        .sort_values(["sample_id"])
        .reset_index(drop=True)
    )
    base["image_path_abs"] = base["image_path"].apply(lambda p: os.path.join(run_dir, p))

    prompts_path = os.path.join(run_dir, "prompts.jsonl")
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Missing {prompts_path}")

    recs = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))

    prompts_df = pd.DataFrame(recs)
    if "negative_prompt" not in prompts_df.columns:
        prompts_df["negative_prompt"] = ""

    df = base.merge(
        prompts_df[["sample_id", "prompt", "negative_prompt"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if df["prompt"].isna().any():
        missing = df[df["prompt"].isna()]["sample_id"].tolist()[:5]
        raise RuntimeError(f"Prompts missing for some sample_ids, e.g. {missing}")

    df["prompt_sha256"] = df["prompt"].apply(sha256_text)
    return df


def make_scorer(kind: str, device: Optional[str]) -> TextImageScorer:
    kind = kind.lower()
    if kind in ("pixel-avg", "pixelavg", "mock"):
        return PixelAvgMockScorer()
    if kind in ("pickscore", "ps"):
        return PickScoreScorer(device=device)
    raise ValueError(f"Unknown scorer '{kind}'. Use 'pickscore' or 'pixel-avg'.")
# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", type=str, help="<run_id> from path: data/runs/<run_id>")
    ap.add_argument("--scorer", type=str, default="pickscore", help="pickscore | pixel-avg")
    ap.add_argument("--offset", type=float, default=20.0, help="Reward offset (reward = (score - offset) * scale)")
    ap.add_argument("--scale", type=float, default=0.05, help="Reward scale (reward = (score - offset) * scale)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing rewards.parquet")
    ap.add_argument("--device", type=str, default=None, help="Device hint for scorer (if applicable)")
    args = ap.parse_args()

    # run_dir = args.run_dir
    run_id = args.run_id
    run_dir = os.path.join(RUNS_ROOT, run_id)
    steps_path = os.path.join(run_dir, "steps.parquet")
    rewards_path = os.path.join(run_dir, "rewards.parquet")
    meta_path = os.path.join(run_dir, "meta.json")
    prompts_path = os.path.join(run_dir, "prompts.jsonl")

    # Presence checks
    for p in [steps_path, meta_path, prompts_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    if (not args.overwrite) and os.path.exists(rewards_path):
        print(f"[score_rewards] {rewards_path} exists. Use --overwrite to replace.")
        sys.exit(0)

    # Load steps & meta
    steps_df = pd.read_parquet(steps_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Build one row per sample with prompt + image path
    samples_df = collect_final_images_and_prompts(steps_df, run_dir)

    # Verify images exist and are readable
    missing = [p for p in samples_df["image_path_abs"].tolist() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, first: {missing[0]}")
    for p in tqdm(samples_df["image_path_abs"], desc="Verifying images", leave=False):
        try:
            Image.open(p).verify()
        except Exception as e:
            raise RuntimeError(f"Image corrupt/unreadable: {p}: {e}")

    # Hash images (integrity)
    print("[score_rewards] Hashing images...")
    samples_df["image_sha256"] = [
        sha256_file(p) for p in tqdm(samples_df["image_path_abs"], desc="SHA256", leave=False)
    ]

    # Score with chosen scorer
    scorer = make_scorer(args.scorer, args.device)
    print(f"[score_rewards] Using scorer: {scorer.name}@{scorer.version}")
    print("[score_rewards] Scoring (prompt-aware)...")
    scores = scorer.score_pairs(
        samples_df["image_path_abs"].tolist(),
        samples_df["prompt"].tolist(),
        batch_size=args.batch_size
    )
    if len(scores) != len(samples_df):
        raise RuntimeError(f"Scorer returned {len(scores)} scores for {len(samples_df)} samples")

    # Transform to reward
    samples_df["pickscore_raw"] = scores  # keep name 'pickscore_raw' for clarity even for mocks
    samples_df["reward"] = samples_df["pickscore_raw"].apply(lambda x: default_transform(x, args.offset, args.scale))

    # Assemble output table (one row per sample_id)
    out_cols = [
        "run_id",
        "sample_id",
        "image_path",
        "image_sha256",
        "prompt",
        "negative_prompt",
        "prompt_sha256",
        "pickscore_raw",
        "reward",
    ]
    out_df = samples_df[out_cols].copy()

    # Sidecar meta (and repetition inside parquet attrs for redundancy)
    computed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    transform_id = f"ps_minus_{int(args.offset)}_times_{args.scale:g}"
    rmeta: Dict[str, Any] = {
        "computed_at": computed_at,
        "scorer_name": scorer.name,
        "scorer_version": scorer.version,
        "prompt_aware": True,
        "prompt_hash_algo": "sha256(utf8)",
        "reward_transform_id": transform_id,
        "reward_formula": "reward = (pickscore_raw - offset) * scale",
        "reward_offset": float(args.offset),
        "reward_scale": float(args.scale),
        "run_id": os.path.basename(run_dir.rstrip("/")),
    }
    out_df.attrs["meta"] = rmeta

    # Write
    print(f"[score_rewards] Writing {rewards_path} with {len(out_df)} rows")
    out_df.to_parquet(rewards_path, index=False)
    with open(os.path.join(run_dir, "rewards_meta.json"), "w") as f:
        json.dump(rmeta, f, indent=2)

    print("[score_rewards] Done.")
    print(f"- samples scored: {len(out_df)}")
    print(f"- scorer: {scorer.name}@{scorer.version}")
    print(f"- transform: {transform_id}")


if __name__ == "__main__":
    main()
