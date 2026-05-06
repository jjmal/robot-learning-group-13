#!/usr/bin/env python3
"""
Convert a custom robot learning dataset to the mimic-video zarr format.

Expected folder structure under --data-root:
    <data_root>/
        action/     *.parquet   (columns: action, observation.state, timestamp, ...)
        t5_xxl/     *.pickle    (per-episode T5 language embeddings)
        video/      *.mp4       (per-episode RGB recordings)

An episode is identified by the shared stem across all three files, e.g.:
    action/episode_0042.parquet
    t5_xxl/episode_0042.pickle
    video/episode_0042.mp4

Output: one zarr group per episode at <output-dir>/<stem>.zarr containing:
    workspace_rgb              (T_vid, H, W, 3)  uint8 — raw video frames
    workspace_rgb_timestamps   (T_vid,)           int64 — nanoseconds
    ee_pose                    (T_act, 5)         float32 — first 5 dims of action
    ee_pose_timestamps         (T_act,)           int64 — nanoseconds
    observation_state          (T_act, 5)         float32 — first 5 dims of obs state
    observation_state_timestamps (T_act,)         int64 — nanoseconds
    language_embedding         (1, D)             float32 — T5 embedding
    language_embedding_timestamps (1,)            int64 — nanoseconds (episode start)

Usage:
    python process_custom.py \
        --data-root /path/to/data \
        --output-dir /path/to/zarr_output \
        [--video-freq 30] \
        [--action-freq 30]
"""

import argparse
import pickle
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd
import tqdm
import zarr
from numcodecs import Blosc

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

S_TO_NS: int = int(1e9)  # seconds → nanoseconds multiplier

# Compressor presets
#   Images   → LZ4 + byte-shuffle  (fast, decent ratio on uint8 spatial data)
#   Lowdim   → LZ4 + bit-shuffle   (better ratio on slowly-varying float sequences)
IMG_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
TRAJ_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    Read a DataFrame column that stores vectors (as lists or numpy arrays)
    and return a 2-D float32 array of shape (T, D).
    Works whether the column is stored as Python lists, object arrays, or
    already as a numeric multi-column layout.
    """
    sample = df[col].iloc[0]
    if hasattr(sample, "__len__"):
        return np.stack(df[col].values).astype(np.float32)
    # Scalar column (shouldn't happen for 6-D vectors, but guard anyway)
    return df[col].values.astype(np.float32)[:, np.newaxis]


def read_video(video_path: Path, video_freq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode all frames from an MP4 file using PyAV (supports AV1, H.264, H.265, …).

    Returns
    -------
    frames : np.ndarray  shape (T, H, W, 3)  dtype uint8  RGB
    timestamps_ns : np.ndarray  shape (T,)  dtype int64  nanoseconds from episode start
    """
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Disable threading issues with AV1 by forcing single-threaded decode
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        for packet in container.demux(stream):
            for frame in packet.decode():
                # Convert to RGB numpy array
                frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    frames_arr = np.stack(frames, axis=0).astype(np.uint8)  # (T, H, W, 3)
    T = len(frames_arr)
    timestamps_ns = (np.arange(T, dtype=np.float64) / video_freq * S_TO_NS).astype(np.int64)
    return frames_arr, timestamps_ns


def load_language_embedding(pickle_path: Path) -> np.ndarray:
    """
    Load a T5 embedding from a .pickle file.

    Returns a 2-D array of shape (1, D) so it looks like a persistent
    single-entry time series to the chunk_reader.
    """
    with open(pickle_path, "rb") as f:
        embedding = pickle.load(f)

    embedding = np.array(embedding, dtype=np.float32)
    if embedding.ndim == 1:
        embedding = embedding[np.newaxis, :]   # (1, D)
    elif embedding.ndim == 2 and embedding.shape[0] != 1:
        # If multiple embeddings are stored (e.g. one per frame), take the first
        embedding = embedding[:1]
    return embedding


# ──────────────────────────────────────────────────────────────────────────────
# Episode processing
# ──────────────────────────────────────────────────────────────────────────────

def process_episode(
    parquet_path: Path,
    video_path: Path,
    pickle_path: Path,
    output_dir: Path,
    action_freq: float,
    video_freq: float,
    overwrite: bool = False,
) -> None:
    """
    Convert a single episode to zarr and write it to output_dir.
    """
    episode_name = parquet_path.stem
    output_path = output_dir / f"{episode_name}.zarr"

    if output_path.exists() and not overwrite:
        print(f"  Skipping {episode_name} (already exists; use --overwrite to force)")
        return

    # ── Parquet ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(parquet_path)

    # Timestamps: parquet stores seconds (float) relative to episode start.
    # Fall back to uniform grid if the column is absent.
    if "timestamp" in df.columns:
        timestamps_ns = (df["timestamp"].values.astype(np.float64) * S_TO_NS).astype(np.int64)
    else:
        print(f"  Warning: no 'timestamp' column in {parquet_path.name}; generating from action_freq")
        timestamps_ns = (np.arange(len(df), dtype=np.float64) / action_freq * S_TO_NS).astype(np.int64)

    # Action — drop 6th dimension → shape (T, 5)
    action = read_vector_column(df, "action")[:, :5]

    # Observation state — drop 6th dimension → shape (T, 5)
    obs_state = read_vector_column(df, "observation.state")[:, :5]

    T_act = len(action)
    assert len(timestamps_ns) == T_act, (
        f"Timestamp length {len(timestamps_ns)} does not match action length {T_act}"
    )

    # ── Video ─────────────────────────────────────────────────────────────────
    frames, video_timestamps_ns = read_video(video_path, video_freq)
    T_vid, H, W, C = frames.shape

    # ── Language embedding ────────────────────────────────────────────────────
    lang_embedding = load_language_embedding(pickle_path)  # (1, D)

    # ── Write zarr ────────────────────────────────────────────────────────────
    root = zarr.open(str(output_path), mode="w")

    # Video frames — chunk (1, H, W, C) so individual frames can be read
    # without loading the whole episode
    root.array(
        "workspace_rgb",
        frames,
        chunks=(1, H, W, C),
        dtype=np.uint8,
        compressor=IMG_COMPRESSOR,
    )
    root.array(
        "workspace_rgb_timestamps",
        video_timestamps_ns,
        chunks=(T_vid,),
        dtype=np.int64,
        compressor=TRAJ_COMPRESSOR,
    )

    # Action (ee_pose)
    root.array(
        "ee_pose",
        action,
        chunks=(T_act, 5),
        dtype=np.float32,
        compressor=TRAJ_COMPRESSOR,
    )
    root.array(
        "ee_pose_timestamps",
        timestamps_ns,
        chunks=(T_act,),
        dtype=np.int64,
        compressor=TRAJ_COMPRESSOR,
    )

    # Observation / proprioception state
    root.array(
        "observation_state",
        obs_state,
        chunks=(T_act, 5),
        dtype=np.float32,
        compressor=TRAJ_COMPRESSOR,
    )
    root.array(
        "observation_state_timestamps",
        timestamps_ns,
        chunks=(T_act,),
        dtype=np.int64,
        compressor=TRAJ_COMPRESSOR,
    )

    # Language embedding — persistent (one value for the whole episode)
    root.array(
        "language_embedding",
        lang_embedding,
        chunks=lang_embedding.shape,
        dtype=np.float32,
        compressor=TRAJ_COMPRESSOR,
    )
    root.array(
        "language_embedding_timestamps",
        np.array([timestamps_ns[0]], dtype=np.int64),  # anchor to episode start
        chunks=(1,),
        dtype=np.int64,
        compressor=TRAJ_COMPRESSOR,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert custom robot dataset to mimic-video zarr format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root folder containing action/, t5_xxl/, and video/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-episode .zarr stores will be written.",
    )
    parser.add_argument(
        "--action-freq",
        type=float,
        default=30.0,
        help="Recording frequency of action / proprioception data in Hz.",
    )
    parser.add_argument(
        "--video-freq",
        type=float,
        default=30.0,
        help="Recording frequency of video in Hz (no default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process episodes whose zarr already exists.",
    )
    args = parser.parse_args()

    action_dir = args.data_root / "action"
    t5_dir = args.data_root / "t5_xxl"
    video_dir = args.data_root / "video"

    for d in (action_dir, t5_dir, video_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected directory not found: {d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover episodes from parquet files; matching video and pickle by stem
    parquet_files = sorted(action_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {action_dir}")

    print(f"Found {len(parquet_files)} episodes under {action_dir}.")
    print(f"action_freq={args.action_freq} Hz  |  video_freq={args.video_freq} Hz")
    print(f"Output → {args.output_dir}\n")

    skipped, failed = [], []

    for parquet_path in tqdm.tqdm(parquet_files, desc="Episodes"):
        stem = parquet_path.stem
        video_path = video_dir / f"{stem}.mp4"
        pickle_path = t5_dir / f"{stem}.pickle"

        missing = [p for p in (video_path, pickle_path) if not p.exists()]
        if missing:
            skipped.append((stem, [str(p) for p in missing]))
            continue

        try:
            process_episode(
                parquet_path=parquet_path,
                video_path=video_path,
                pickle_path=pickle_path,
                output_dir=args.output_dir,
                action_freq=args.action_freq,
                video_freq=args.video_freq,
                overwrite=args.overwrite,
            )
        except Exception as e:  # noqa: BLE001
            failed.append((stem, str(e)))
            tqdm.tqdm.write(f"  ERROR on {stem}: {e}")
            tqdm.tqdm.write(traceback.format_exc())

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Processed : {len(parquet_files) - len(skipped) - len(failed)}")
    if skipped:
        print(f"  Skipped (missing files) : {len(skipped)}")
        for stem, paths in skipped:
            print(f"    {stem}: missing {paths}")
    if failed:
        print(f"  Failed : {len(failed)}")
        for stem, err in failed:
            print(f"    {stem}: {err}")
    print("Done.")


if __name__ == "__main__":
    main()
