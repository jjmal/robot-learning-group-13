#!/usr/bin/env python3
"""
Convert LeRobot v3 format data to backbone format (video/*.mp4 + t5_xxl/ + metas/*.txt).

LeRobot v3: one big concatenated video per chunk with episode timestamps in parquet.
Backbone format: individual H.264 mp4 files numbered sequentially.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_ffmpeg_path():
    """Get path to ffmpeg binary bundled with imageio_ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        # Fallback: try common locations
        for path in [
            "/home/ubuntu/robot-learning-group-13/model/.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]:
            if os.path.exists(path):
                return path
        raise RuntimeError("ffmpeg not found")


def convert_chunk(chunk_dir: Path, out_dir: Path, episode_start: int) -> int:
    """
    Convert one LeRobot chunk to backbone format.
    Returns the next episode number to assign.
    """
    # Read episode metadata
    ep_parquet = list(chunk_dir.glob("meta/episodes/chunk-*/file-*.parquet"))[0]
    eps_df = pd.read_parquet(ep_parquet)

    # Find the video file (auto-detect camera key)
    video_files = list(chunk_dir.glob("videos/observation.images.*/chunk-*/file-*.mp4"))
    if not video_files:
        raise RuntimeError(f"No video files found in {chunk_dir}")
    video_file = video_files[0]

    # Read task name (take first, should be same for all episodes in chunk)
    tasks_parquet = chunk_dir / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_parquet)
    task_name = tasks_df.index[0]  # task name is the index

    # Create output directories
    (out_dir / "video").mkdir(parents=True, exist_ok=True)
    (out_dir / "metas").mkdir(parents=True, exist_ok=True)

    ep_count = 0
    for _, row in tqdm(eps_df.iterrows(), total=len(eps_df), desc=f"Converting {chunk_dir.name}"):
        ep_idx = episode_start + ep_count

        # Extract timestamps (auto-detect camera key)
        ts_cols = [c for c in eps_df.columns if c.endswith("/from_timestamp")]
        cam_key = ts_cols[0].replace("/from_timestamp", "")
        from_ts = row[f"{cam_key}/from_timestamp"]
        to_ts = row[f"{cam_key}/to_timestamp"]
        duration = to_ts - from_ts

        # Output file
        out_video = out_dir / "video" / f"ep{ep_idx}.mp4"
        out_meta = out_dir / "metas" / f"ep{ep_idx}.txt"

        # Skip if already exists
        if out_video.exists():
            ep_count += 1
            continue

        # ffmpeg: seek to timestamp, extract clip, re-encode to H.264
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path, "-v", "error",
            "-ss", str(from_ts),
            "-i", str(video_file),
            "-t", str(duration),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-an",  # no audio
            str(out_video),
            "-y"
        ]

        subprocess.run(cmd, check=True)

        # Write metadata file
        out_meta.write_text(task_name)

        ep_count += 1

    return episode_start + ep_count


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3 data to backbone format")
    parser.add_argument("--data-dir", required=True, help="Path to /ephemeral/data_raw")
    parser.add_argument("--out-dir", required=True, help="Output directory (e.g., /ephemeral/rl-group13-merged-h264)")

    parser.add_argument("--episode-start", type=int, default=None,
                        help="Episode number to start from. Auto-detected from output dir if not set.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Auto-detect starting episode from existing files in output dir
    if args.episode_start is not None:
        ep_idx = args.episode_start
    else:
        existing = list((out_dir / "video").glob("ep*.mp4")) if (out_dir / "video").exists() else []
        ep_idx = len(existing)

    # Get all chunks and sort
    chunks = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("t1-e")])

    print(f"Found {len(chunks)} chunks")
    print(f"Output: {out_dir}")
    print(f"Starting from episode {ep_idx}")
    print()

    for chunk_dir in chunks:
        ep_idx = convert_chunk(chunk_dir, out_dir, ep_idx)

    print()
    print(f"✓ Converted {ep_idx} episodes to {out_dir}")
    print(f"  video/ep0.mp4 ... ep{ep_idx-1}.mp4")
    print(f"  metas/ep0.txt ... ep{ep_idx-1}.txt")


if __name__ == "__main__":
    main()
