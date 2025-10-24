#!/usr/bin/env python3
# scripts/views/check_view.py

import os, json, argparse, hashlib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIEWS_ROOT = os.path.join(PROJECT_ROOT, "data", "views")

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view_id", type=str, help="Folder under data/views/")
    ap.add_argument("--rehash", action="store_true", help="Rehash sources to verify manifest")
    args = ap.parse_args()

    view_dir = os.path.join(VIEWS_ROOT, args.view_id)
    man_p = os.path.join(view_dir, "manifest.json")
    rows_p = os.path.join(view_dir, "rows.parquet")
    splits_p = os.path.join(view_dir, "splits.json")

    for p in [man_p, rows_p, splits_p]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {p}")

    with open(man_p, "r") as f:
        man = json.load(f)
    rows = pd.read_parquet(rows_p)

    # Basic schema & counts
    expected_cols = man.get("schema", [])
    missing = [c for c in expected_cols if c not in rows.columns]
    assert not missing, f"rows.parquet missing columns: {missing}"

    # rows hash consistency (soft check)
    try:
        rows_bytes = rows.to_csv(index=False).encode("utf-8")
        rows_sha = hashlib.sha256(rows_bytes).hexdigest()
        if man.get("rows_sha256_csv") not in ("", "unavailable", None):
            assert rows_sha == man["rows_sha256_csv"], "rows_sha256_csv mismatch"
    except Exception:
        pass

    # Noise monotonic sanity (per run_id, per step_idx mean should be non-decreasing)
    g = rows.groupby(["run_id","step_idx"])["noise_scalar"].mean().reset_index()
    ok = True
    for rid, sub in g.groupby("run_id"):
        sub = sub.sort_values("step_idx")
        diffs = sub["noise_scalar"].diff().fillna(0.0)
        if (diffs < -1e-6).any():
            ok = False
            bad_at = sub.iloc[(diffs < -1e-6).to_numpy().nonzero()[0][0]]["step_idx"]
            raise AssertionError(f"[{rid}] noise_scalar not non-decreasing at step_idx {int(bad_at)}")
    if ok:
        pass  # good

    # Splits cover all (run_id, sample_id) and are disjoint
    with open(splits_p, "r") as f:
        splits = json.load(f)
    pairs_all = set(tuple(x) for x in rows[["run_id","sample_id"]].drop_duplicates().to_records(index=False))
    pairs_train = set(tuple(x) for x in splits["train"])
    pairs_val   = set(tuple(x) for x in splits["val"])
    pairs_test  = set(tuple(x) for x in splits["test"])
    assert (pairs_train & pairs_val) == set()
    assert (pairs_train & pairs_test) == set()
    assert (pairs_val & pairs_test) == set()
    assert (pairs_train | pairs_val | pairs_test) == pairs_all, "splits do not cover all (run_id,sample_id)"

    # Optional: verify source file hashes
    if args.rehash:
        for src in man.get("sources", []):
            sp = os.path.join(view_dir, src["steps_parquet"])
            rp = os.path.join(view_dir, src["rewards_parquet"])
            assert os.path.exists(sp) and os.path.exists(rp), f"missing source files: {sp} or {rp}"
            s_hash = sha256_file(sp)
            r_hash = sha256_file(rp)
            assert s_hash == src["steps_sha256"], f"steps_sha256 mismatch for run {src['run_id']}"
            assert r_hash == src["rewards_sha256"], f"rewards_sha256 mismatch for run {src['run_id']}"

    # Summary
    print("View checks passed ✅")
    print(f"- view_id: {man.get('view_id')}")
    print(f"- rows: {len(rows)} | samples: {rows[['run_id','sample_id']].drop_duplicates().shape[0]}")
    print(f"- runs: {', '.join(man.get('runs', []))}")
    print(f"- noise_type: {man.get('constraints',{}).get('noise_type')}")
    ns = man.get("noise_stats", {})
    if ns:
        print(f"- noise mean={ns.get('mean'):.4f}, std={ns.get('std'):.4f}")

if __name__ == "__main__":
    main()
