import os, json
import pandas as pd
import torch

def main(run_dir: str):
    steps = pd.read_parquet(os.path.join(run_dir, "steps.parquet"))
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)

    # Basic counts
    n_rows = len(steps)
    n_samples = steps["sample_id"].nunique()
    n_steps = steps["step_idx"].nunique()
    print(f"rows={n_rows}, samples={n_samples}, steps_per_trajectory≈{n_rows//n_samples}")

    # Monotonicity check over timesteps
    timesteps = meta["timesteps"]
    assert all(timesteps[i] >= timesteps[i+1] for i in range(len(timesteps)-1)), \
    "timesteps should be non-increasing"

    # Spot-load one shard row and match to steps row
    first_row = steps.iloc[0]
    shard_rel, row_s = first_row["latent_path"].split(":row=")
    shard_path = os.path.join(run_dir, shard_rel)
    row_idx = int(row_s)
    shard = torch.load(shard_path)
    xt = shard[row_idx]["xt"]
    assert xt.shape[0] == 4 and xt.ndim == 3 and xt.dtype == torch.float32
    print("Loaded one latent OK:", xt.shape, xt.dtype)

    # Noise sanity: logSNR should increase with step_idx (if using logsnr)
    if first_row["noise_type"] == "logsnr":
        by_step = steps.groupby("step_idx")["noise_scalar"].mean().sort_index()
        assert (by_step.diff().fillna(0) >= -1e-6).all(), "logSNR should be non-decreasing over step_idx"
        print("logSNR monotonicity looks OK.")

    print("All basic checks passed.")

if __name__ == "__main__":
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "data/runs/2025-10-22_16-39-41"
    main(run_dir)
