#!/usr/bin/env python3
# scripts/check_phase2.py

import os
import sys
import json
import argparse
import hashlib
import random
import pandas as pd
import numpy as np
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "data", "runs")

print(f"RUNS_ROOT: {RUNS_ROOT}")

REQUIRED_REWARD_COLS = {
    "run_id",
    "sample_id",
    "image_path",
    "image_sha256",
    "prompt",
    "prompt_sha256",
    "pickscore_raw",   # rename here if you chose a different score column
    "reward",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_prompts_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "negative_prompt" not in df.columns:
        df["negative_prompt"] = ""
    return df[["sample_id", "prompt", "negative_prompt"]]


def check_phase2(
    run_id: str,
    score_col: str = "pickscore_raw",
    rehash_sample: int = 8,
    strict_sample_match: bool = True,
) -> None:
    run_dir = os.path.join(RUNS_ROOT, run_id)
    steps_p = os.path.join(run_dir, "steps.parquet")
    rewards_p = os.path.join(run_dir, "rewards.parquet")
    rmeta_p = os.path.join(run_dir, "rewards_meta.json")
    prompts_p = os.path.join(run_dir, "prompts.jsonl")

    # 1) Presence
    for p in [steps_p, rewards_p, rmeta_p, prompts_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    steps = pd.read_parquet(steps_p)
    rewards = pd.read_parquet(rewards_p)
    with open(rmeta_p, "r") as f:
        rmeta = json.load(f)
    prompts = read_prompts_jsonl(prompts_p)

    # 2) Schema & columns
    missing_cols = REQUIRED_REWARD_COLS - set(rewards.columns)
    if score_col not in rewards.columns:
        missing_cols.add(score_col)
    if missing_cols:
        raise AssertionError(f"rewards.parquet missing columns: {sorted(missing_cols)}")

    # 3) One reward per sample_id
    counts = rewards.groupby("sample_id").size()
    dup = counts[counts != 1]
    assert dup.empty, f"Found sample_ids with != 1 reward row: {dup.head().to_dict()}"

    # 4) Sample set equality (optional strict)
    step_samples = set(steps["sample_id"].unique().tolist())
    reward_samples = set(rewards["sample_id"].unique().tolist())

    missing_in_rewards = step_samples - reward_samples
    extra_in_rewards = reward_samples - step_samples

    if strict_sample_match:
        assert not missing_in_rewards, f"Samples present in steps but missing in rewards: {list(missing_in_rewards)[:10]}"
        assert not extra_in_rewards, f"Samples present in rewards but not in steps: {list(extra_in_rewards)[:10]}"
    else:
        if missing_in_rewards:
            print(f"[WARN] {len(missing_in_rewards)} samples in steps missing from rewards (first 5): {list(missing_in_rewards)[:5]}")
        if extra_in_rewards:
            print(f"[WARN] {len(extra_in_rewards)} samples in rewards not in steps (first 5): {list(extra_in_rewards)[:5]}")

    # 5) Joinability: steps x rewards is many_to_one on (run_id, sample_id)
    merged = steps.merge(
        rewards[["run_id", "sample_id", "reward"]],
        on=["run_id", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    assert merged["reward"].notna().all(), "Some steps rows are missing reward after join"

    # 6) Prompt consistency (string & hash) vs prompts.jsonl
    rewards_prompts = rewards[["sample_id", "prompt", "prompt_sha256"]].copy()
    prompts_map = prompts.set_index("sample_id")["prompt"].to_dict()
    # Check string equality
    bad_strings: List[str] = []
    for sid, ptxt in rewards_prompts[["sample_id", "prompt"]].itertuples(index=False):
        ref = prompts_map.get(sid, None)
        if ref is None or ref != ptxt:
            bad_strings.append(sid)
            if len(bad_strings) >= 10:
                break
    assert not bad_strings, f"Prompt text mismatch for sample_ids: {bad_strings[:10]}"

    # Check hash equality
    def sha256_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    rewards_prompts["prompt_sha256_check"] = rewards_prompts["prompt"].map(sha256_text)
    bad_hash = rewards_prompts[
        rewards_prompts["prompt_sha256_check"] != rewards_prompts["prompt_sha256"]
    ]
    assert bad_hash.empty, f"prompt_sha256 mismatch for sample_ids: {bad_hash['sample_id'].head(10).tolist()}"

    # 7) Reward transform correctness
    # Prefer values from rewards_meta.json
    try:
        offset = float(rmeta.get("reward_offset", rmeta.get("offset")))
        scale = float(rmeta.get("reward_scale", rmeta.get("scale")))
    except Exception as e:
        raise AssertionError(f"Could not read offset/scale from rewards_meta.json: {e}")

    rec = (rewards[score_col] - offset) * scale
    max_abs_diff = (rewards["reward"] - rec).abs().max()
    assert max_abs_diff < 1e-6, f"Reward != (score - offset)*scale; max abs diff={max_abs_diff}"

    # 8) Image existence and optional hash verification on a sample
    rewards["image_path_abs"] = rewards["image_path"].apply(lambda p: os.path.join(run_dir, p))
    missing = rewards[~rewards["image_path_abs"].apply(os.path.exists)]
    assert missing.empty, f"Missing image files, first few: {missing['image_path_abs'].head(5).tolist()}"

    if rehash_sample > 0:
        sample_rows = rewards.sample(n=min(rehash_sample, len(rewards)), random_state=123)
        mismatches = []
        for _, row in sample_rows.iterrows():
            h = sha256_file(row["image_path_abs"])
            if h != row["image_sha256"]:
                mismatches.append(row["sample_id"])
                if len(mismatches) >= 5:
                    break
        assert not mismatches, f"image_sha256 mismatch for sample_ids: {mismatches}"

    # 9) Compact summary
    scorer_name = rmeta.get("scorer_name", "unknown")
    scorer_version = rmeta.get("scorer_version", "unknown")
    transform_id = rmeta.get("reward_transform_id", f"offset_{offset}_scale_{scale}")

    n_samples = len(rewards)
    n_prompts = rewards["prompt"].nunique()
    print("Phase 2 checks passed ✅")
    print(f"- run: {os.path.basename(os.path.normpath(run_dir))}")
    print(f"- samples with rewards: {n_samples}")
    print(f"- unique prompts: {n_prompts}")
    print(f"- scorer: {scorer_name}@{scorer_version}")
    print(f"- transform: {transform_id}  (offset={offset}, scale={scale})")
    print(f"- score column: {score_col}")
    if rehash_sample > 0:
        print(f"- verified image hashes on {min(rehash_sample, n_samples)} samples")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help="<run_id> where run is at path data/runs/<run_id>")
    parser.add_argument("--score-col", type=str, default="pickscore_raw",
                        help="Column name for raw score in rewards.parquet")
    parser.add_argument("--rehash-sample", type=int, default=8,
                        help="Random subset size to verify image_sha256 (0 to skip)")
    parser.add_argument("--strict-sample-match", action="store_true",
                        help="Require exact equality of sample_id sets between steps and rewards")
    args = parser.parse_args()

    check_phase2(
        run_id=args.run_id,
        score_col=args.score_col,
        rehash_sample=args.rehash_sample,
        strict_sample_match=args.strict_sample_match,
    )
