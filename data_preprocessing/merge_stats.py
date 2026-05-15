"""
Merges stats.json files from HuggingFace datasets t1-e1 through t1-e10
into a single combined stats.json with recomputed statistics.

Combining rules:
  - min/max   : element-wise min / max
  - count     : sum
  - mean      : weighted average  Σ(mean_i * n_i) / Σn_i
  - std       : parallel formula  sqrt(Σ(n_i*(std_i² + mean_i²)) / N - μ²)
  - quantiles : weighted average  (approximation without raw data)
"""

import json
import math
import os
import argparse
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUANTILE_KEYS = ("q01", "q10", "q50", "q90", "q99")
STAT_KEYS = ("min", "max", "mean", "std", "count") + QUANTILE_KEYS


# ---------------------------------------------------------------------------
# Recursive numpy-free helpers that work on plain Python scalars / lists
# ---------------------------------------------------------------------------

def map_nested(fn, *args):
    """Apply fn element-wise across one or more nested list structures."""
    if isinstance(args[0], list):
        return [map_nested(fn, *[a[i] for a in args]) for i in range(len(args[0]))]
    return fn(*args)


def nest_min(*arrays):
    return map_nested(min, *arrays)

def nest_max(*arrays):
    return map_nested(max, *arrays)

def nest_add(*arrays):
    return map_nested(lambda *xs: sum(xs), *arrays)

def nest_scale(array, scalar):
    return map_nested(lambda x: x * scalar, array)

def nest_div(array, scalar):
    return map_nested(lambda x: x / scalar, array)

def nest_sqrt(array):
    return map_nested(math.sqrt, array)

def nest_sq(array):
    return map_nested(lambda x: x * x, array)


# ---------------------------------------------------------------------------
# Core combination logic
# ---------------------------------------------------------------------------


def combine_stats(all_stats: list[dict]) -> dict:
    """Combine a list of per-dataset stat dicts into one merged dict.
 
    Features may appear in different orders or be absent from some datasets.
    Only datasets that contain a given feature contribute to its combined stats.
    Features are emitted in the order they are first encountered across all datasets.
    """
 
    # Union of all feature keys, preserving first-seen order
    seen: set[str] = set()
    feature_keys: list[str] = []
    for stats in all_stats:
        for key in stats:
            if key not in seen:
                seen.add(key)
                feature_keys.append(key)
 
    combined = {}
 
    for feat in feature_keys:
        # Only include datasets that actually have this feature
        blocks = [s[feat] for s in all_stats if feat in s]
        absent = len(all_stats) - len(blocks)
        if absent:
            print(f"  warning: '{feat}' missing from {absent} dataset(s) — combining {len(blocks)} only")
 
        counts = [b["count"][0] for b in blocks]
        N = sum(counts)
 
        # -- min / max --------------------------------------------------------
        combined_min = blocks[0]["min"]
        combined_max = blocks[0]["max"]
        for b in blocks[1:]:
            combined_min = nest_min(combined_min, b["min"])
            combined_max = nest_max(combined_max, b["max"])
 
        # -- mean (weighted average) ------------------------------------------
        weighted_means = [nest_scale(b["mean"], n) for b, n in zip(blocks, counts)]
        combined_mean = weighted_means[0]
        for wm in weighted_means[1:]:
            combined_mean = nest_add(combined_mean, wm)
        combined_mean = nest_div(combined_mean, N)
 
        # -- std (parallel formula, sample std ddof=1) ------------------------
        # Total SS = sum_i [ (n_i - 1)*s_i^2  +  n_i*(mu_i - mu_combined)^2 ]
        # combined_s = sqrt(total_SS / (N - 1))
        def ss_term(s, mu, n, _mu=combined_mean):
            within  = nest_scale(nest_sq(s), n - 1)
            between = nest_scale(nest_sq(map_nested(lambda a, b: a - b, mu, _mu)), n)
            return nest_add(within, between)
 
        total_ss = ss_term(blocks[0]["std"], blocks[0]["mean"], counts[0])
        for b, n in zip(blocks[1:], counts[1:]):
            total_ss = nest_add(total_ss, ss_term(b["std"], b["mean"], n))
 
        total_ss    = map_nested(lambda v: max(v, 0.0), total_ss)   # clamp fp noise
        combined_std = nest_sqrt(nest_div(total_ss, N - 1))
 
        # -- quantiles (weighted average – approximation) ---------------------
        combined_quantiles = {}
        for qk in QUANTILE_KEYS:
            weighted_q = [nest_scale(b[qk], n) for b, n in zip(blocks, counts)]
            cq = weighted_q[0]
            for wq in weighted_q[1:]:
                cq = nest_add(cq, wq)
            combined_quantiles[qk] = nest_div(cq, N)
 
        combined[feat] = {
            "min":   combined_min,
            "max":   combined_max,
            "mean":  combined_mean,
            "std":   combined_std,
            "count": [N],
            **combined_quantiles,
        }
 
    return combined


# ---------------------------------------------------------------------------
# Fetch helper
# ---------------------------------------------------------------------------

def fetch_stats(root_dir: str, t: int, i: int) -> dict:
    path = Path(root_dir + f"/data_raw/t{t}-e{i}/meta/stats.json")
    with open(path) as f:
        d = json.load(f)
   
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser(description="Merge the raw data into a correct format per each task.")
    parser.add_argument("--root-dir", required=True, help="Path to the root project directory")
    parser.add_argument("--task", required=True, help="Task number")
    

    args = parser.parse_args()
    
    root_dir = args.root_dir
    task = args.task


    output_file = Path(root_dir + "/data_merged/stats.json")

    dataset_filenames = os.listdir(root_dir + "/data_raw")
    ls = []
    for fil in dataset_filenames:
        i = int(re.findall("(?<=-e)\d+", fil)[0])
        ls.append(i)
    min_datas_id = min(ls)
    max_datas_id = max(ls)


    all_stats = []
    for i in range(min_datas_id, max_datas_id + 1):
        try:
            stats = fetch_stats(root_dir = root_dir, t=task, i=i)
            all_stats.append(stats)
            print(f"    ✓ t1-e{i}: count={_frame_count(stats)}")
        except Exception as exc:
            print(f"    ✗ t1-e{i}: {exc} — skipping")

    if not all_stats:
        raise RuntimeError("No stats files could be found.")

    print(f"\nCombining {len(all_stats)} datasets …")
    combined = combine_stats(all_stats)

    output_file .write_text(json.dumps(combined, indent=2))
    print(f"\nWritten to { output_file .resolve()}")

    # Quick sanity-print
    for feat, stats in combined.items():
        print(f"  {feat}: count={stats['count']}, mean_shape={_shape(stats['mean'])}")


def _shape(x):
    if not isinstance(x, list):
        return "scalar"
    dims = []
    cur = x
    while isinstance(cur, list):
        dims.append(len(cur))
        cur = cur[0]
    return str(dims)


def _frame_count(stats: dict) -> int:
    """Return the frame-level count for a dataset.
 
    Image features store pixel counts (H*W*frames), not frame counts, so we
    prefer known scalar features and fall back to the first feature whose
    count is not inflated by spatial dimensions.
    """
    # Prefer canonical scalar features if present
    for key in ("frame_index", "index", "timestamp", "episode_index", "task_index"):
        if key in stats:
            return stats[key]["count"][0]
    # Fall back: first feature with a plain scalar min (not an image)
    for feat, block in stats.items():
        if not isinstance(block["min"][0], list):
            return block["count"][0]
    return next(iter(stats.values()))["count"][0]


if __name__ == "__main__":
    main()