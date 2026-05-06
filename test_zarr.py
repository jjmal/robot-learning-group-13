#!/usr/bin/env python3
"""
Two-level test for the custom zarr dataset produced by process_custom.py.

Level 1 — Zarr structure (standalone, no mimic-video required):
    Validates every episode zarr contains the expected keys, shapes, dtypes,
    and that timestamps are sane.

Level 2 — ChunkReader integration (requires mimic-video model in PYTHONPATH):
    Constructs a real ChunkReader, reads chunks across the dataset, and
    validates every returned array has the expected shape.

Run from anywhere for Level 1:
    python test_zarr_dataset.py --zarr-dir /path/to/zarr_output

Run from mimic-video/model for Level 2 (with venv active):
    PYTHONPATH=. python test_zarr_dataset.py --zarr-dir /path/to/zarr_output
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import zarr

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — must match process_custom.py and your training YAML
# ──────────────────────────────────────────────────────────────────────────────

# Keys that every episode zarr must contain
EXPECTED_KEYS = {
    "workspace_rgb",
    "workspace_rgb_timestamps",
    "ee_pose",
    "ee_pose_timestamps",
    "observation_state",
    "observation_state_timestamps",
    "language_embedding",
    "language_embedding_timestamps",
}

# Expected dtypes
EXPECTED_DTYPES = {
    "workspace_rgb": np.uint8,
    "workspace_rgb_timestamps": np.int64,
    "ee_pose": np.float32,
    "ee_pose_timestamps": np.int64,
    "observation_state": np.float32,
    "observation_state_timestamps": np.int64,
    "language_embedding": np.float32,
    "language_embedding_timestamps": np.int64,
}

# Expected trailing dimensions (None = any)
EXPECTED_TRAILING_DIMS = {
    "workspace_rgb": (None, None, 3),  # (T, H, W, 3)
    "ee_pose": (5,),
    "observation_state": (5,),
    "language_embedding": (None,),     # (1, embedding_dim)
}

# ChunkReader config — horizon / frequency match your training YAML
CHUNK_READER_DATA_COMPONENTS = {
    # key format: "<category>/<zarr_array_name>"
    "obs/workspace_rgb": {
        "horizon": 5,
        "target_frequency": 30.0,
        "shift_right_by": 0.0,
    },
    "obs/ee_pose": {
        "horizon": 1,
        "target_frequency": 30.0,
        "shift_right_by": 0.0,
    },
    "obs/observation_state": {
        "horizon": 1,
        "target_frequency": 30.0,
        "shift_right_by": 0.0,
    },
    "obs/language_embedding": {
        "horizon": 1,
        "target_frequency": 1.0,
        "shift_right_by": 0.0,
    },
    "action/ee_pose": {
        "horizon": 16,
        "target_frequency": 30.0,
        "shift_right_by": 0.0,
    },
}

TIMESTEP_ANCHOR = "workspace_rgb"

# Expected output shapes from ChunkReader (horizon × trailing_dim)
CHUNK_READER_EXPECTED_SHAPES = {
    "obs/workspace_rgb":       (5, None, None, 3),
    "obs/ee_pose":             (1, 5),
    "obs/observation_state":   (1, 5),
    "obs/language_embedding":  (1, None),
    "action/ee_pose":          (16, 5),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def check(condition: bool, msg: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {msg}")
    return condition


def shape_matches(actual: tuple, expected: tuple) -> bool:
    """Allow None as wildcard in expected shape."""
    if len(actual) != len(expected):
        return False
    return all(e is None or a == e for a, e in zip(actual, expected))


# ──────────────────────────────────────────────────────────────────────────────
# Level 1 — Zarr structure
# ──────────────────────────────────────────────────────────────────────────────

def test_zarr_structure(zarr_paths: list[Path]) -> int:
    """Returns number of failed checks."""
    print("\n" + "=" * 60)
    print("LEVEL 1: Zarr structure validation")
    print("=" * 60)

    failures = 0

    for ep_path in zarr_paths:
        print(f"\n  Episode: {ep_path.name}")
        try:
            root = zarr.open(str(ep_path), "r")
        except Exception as e:
            print(f"  [{FAIL}] Could not open zarr: {e}")
            failures += 1
            continue

        present_keys = set(root.array_keys())

        # 1. All expected keys present
        for key in EXPECTED_KEYS:
            ok = check(key in present_keys, f"key '{key}' exists")
            if not ok:
                failures += 1

        # Per-key checks
        for key in EXPECTED_KEYS:
            if key not in present_keys:
                continue
            arr = root[key]

            # 2. Dtype
            if key in EXPECTED_DTYPES:
                ok = check(
                    arr.dtype == EXPECTED_DTYPES[key],
                    f"'{key}' dtype={arr.dtype} (expected {EXPECTED_DTYPES[key]})",
                )
                if not ok:
                    failures += 1

            # 3. Trailing dimensions
            if key in EXPECTED_TRAILING_DIMS:
                exp = EXPECTED_TRAILING_DIMS[key]
                ok = check(
                    shape_matches(arr.shape[-len(exp):], exp),
                    f"'{key}' trailing shape={arr.shape[-len(exp):]} (expected {exp})",
                )
                if not ok:
                    failures += 1

        # 4. Timestamp arrays are non-empty and monotonically increasing
        for ts_key in [k for k in present_keys if k.endswith("_timestamps")]:
            ts = root[ts_key][:]
            ok = check(len(ts) > 0, f"'{ts_key}' non-empty (len={len(ts)})")
            if not ok:
                failures += 1
                continue
            if len(ts) > 1:
                ok = check(
                    bool(np.all(np.diff(ts) >= 0)),
                    f"'{ts_key}' monotonically non-decreasing",
                )
                if not ok:
                    failures += 1

        # 5. Value array length matches timestamp array length
        for base_key in ["workspace_rgb", "ee_pose", "observation_state"]:
            ts_key = f"{base_key}_timestamps"
            if base_key in present_keys and ts_key in present_keys:
                n_vals = root[base_key].shape[0]
                n_ts = root[ts_key].shape[0]
                ok = check(
                    n_vals == n_ts,
                    f"'{base_key}' length {n_vals} == '{ts_key}' length {n_ts}",
                )
                if not ok:
                    failures += 1

        # 6. Language embedding is shape (1, D) — persistent single entry
        if "language_embedding" in present_keys:
            shape = root["language_embedding"].shape
            ok = check(shape[0] == 1, f"'language_embedding' first dim == 1, got {shape}")
            if not ok:
                failures += 1

        # 7. Pixel values are in [0, 255]
        if "workspace_rgb" in present_keys:
            sample = root["workspace_rgb"][0]  # read first frame only
            ok = check(
                int(sample.min()) >= 0 and int(sample.max()) <= 255,
                f"'workspace_rgb' pixel range [{sample.min()}, {sample.max()}] ⊆ [0, 255]",
            )
            if not ok:
                failures += 1

    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — ChunkReader integration
# ──────────────────────────────────────────────────────────────────────────────

def build_data_components(zarr_paths: list[Path]) -> dict:
    """
    Merge the hardcoded horizon/frequency config with the obs_type/repr
    inferred from the actual zarr data (avoids hard-coding ObsType enums
    that live inside cosmos_predict2).
    """
    from cosmos_predict2.data.action.types import LieRepr, NormalizationType, ObsType

    OBS_TYPE_MAP = {
        "workspace_rgb":     ObsType.RGB,
        "ee_pose":           ObsType.CARTESIAN_POS,
        "observation_state": ObsType.JOINT_POS,
        "language_embedding": ObsType.LANGUAGE_EMBEDDING,
    }
    REPR_MAP = {
        "workspace_rgb":     None,
        "ee_pose":           LieRepr.OTHER,
        "observation_state": LieRepr.OTHER,
        "language_embedding": None,
    }

    data_components = {}
    for full_key, spec in CHUNK_READER_DATA_COMPONENTS.items():
        arr_name = full_key.split("/")[1]
        data_components[full_key] = {
            **spec,
            "obs_type": OBS_TYPE_MAP[arr_name],
            "repr": REPR_MAP[arr_name],
            "normalization_type": NormalizationType.NONE,
        }
    return data_components


def test_chunk_reader(zarr_paths: list[Path]) -> int:
    print("\n" + "=" * 60)
    print("LEVEL 2: ChunkReader integration")
    print("=" * 60)

    try:
        from cosmos_predict2.data.action.chunk_reader import ChunkReader
    except ImportError:
        print(f"  [{SKIP}] cosmos_predict2 not importable.")
        print("         Run from mimic-video/model with PYTHONPATH=. to enable Level 2.")
        return 0

    failures = 0

    try:
        data_components = build_data_components(zarr_paths)
    except Exception as e:
        print(f"  [{FAIL}] Could not build data_components: {e}")
        traceback.print_exc()
        return 1

    print(f"\n  Building ChunkReader over {len(zarr_paths)} episode(s)…")
    try:
        reader = ChunkReader(
            zarr_paths,
            data_components=data_components,
            timestep_anchor=TIMESTEP_ANCHOR,
            should_include_padded_tails=True,
            verbose=True,
        )
    except Exception as e:
        print(f"  [{FAIL}] ChunkReader.__init__ raised: {e}")
        traceback.print_exc()
        return 1

    total = len(reader)
    ok = check(total > 0, f"ChunkReader.__len__() = {total} > 0")
    if not ok:
        return 1

    print(f"\n  Dataset has {total} valid timesteps across {len(zarr_paths)} episode(s).")

    # Sample indices: first, middle, last, plus a few random ones
    rng = np.random.default_rng(0)
    sample_indices = sorted(set([
        0,
        total // 4,
        total // 2,
        total * 3 // 4,
        total - 1,
        *rng.integers(0, total, size=min(10, total)).tolist(),
    ]))

    print(f"\n  Reading {len(sample_indices)} chunks…")
    for idx in sample_indices:
        try:
            chunk = reader.read_chunk(idx)
        except Exception as e:
            print(f"  [{FAIL}] read_chunk({idx}) raised: {e}")
            traceback.print_exc()
            failures += 1
            continue

        # Check all expected keys are present
        for key in CHUNK_READER_EXPECTED_SHAPES:
            if key not in chunk:
                print(f"  [{FAIL}] idx={idx}: key '{key}' missing from chunk")
                failures += 1
                continue

            arr = chunk[key]
            exp_shape = CHUNK_READER_EXPECTED_SHAPES[key]

            shape_ok = shape_matches(arr.shape, exp_shape)
            ok = check(
                shape_ok,
                f"idx={idx} '{key}': shape={arr.shape} (expected {exp_shape})",
            )
            if not ok:
                failures += 1

            # No NaNs or Infs in numeric arrays
            if arr.dtype in (np.float32, np.float64):
                finite_ok = check(
                    bool(np.all(np.isfinite(arr))),
                    f"idx={idx} '{key}': all values finite",
                )
                if not finite_ok:
                    failures += 1

    return failures


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test zarr dataset produced by process_custom.py.")
    parser.add_argument(
        "--zarr-dir",
        type=Path,
        required=True,
        help="Directory containing per-episode .zarr stores.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit to the first N episodes (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    zarr_paths = sorted(args.zarr_dir.glob("*.zarr"))
    if not zarr_paths:
        print(f"No .zarr stores found in {args.zarr_dir}")
        sys.exit(1)

    if args.max_episodes is not None:
        zarr_paths = zarr_paths[: args.max_episodes]

    print(f"Found {len(zarr_paths)} episode(s) in {args.zarr_dir}")

    total_failures = 0
    total_failures += test_zarr_structure(zarr_paths)
    total_failures += test_chunk_reader(zarr_paths)

    print("\n" + "=" * 60)
    if total_failures == 0:
        print(f"  All tests passed.")
    else:
        print(f"  {total_failures} check(s) FAILED.")
    print("=" * 60)

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
