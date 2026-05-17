#!/usr/bin/env python3
"""
Convert LeRobot v3 chunked parquet data + pre-converted mp4s to mimic-video zarr format.

Reads action/obs data from LeRobot v3 data parquets and video from already-converted
per-episode H.264 mp4 files (produced by convert_lerobot_to_backbone.py).

Output zarr per episode:
    workspace_rgb              (T_vid, H, W, 3)  uint8
    workspace_rgb_timestamps   (T_vid,)           int64  nanoseconds
    actions                    (T_act, 5)         float32  (gripper dropped)
    actions_timestamps         (T_act,)           int64  nanoseconds
    observation_state          (T_act, 5)         float32  (gripper dropped)
    observation_state_timestamps (T_act,)         int64
    language_instruction       (1,)               str
"""

import argparse
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd
import tqdm
import zarr
from numcodecs import Blosc, VLenUTF8

S_TO_NS = int(1e9)
IMG_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
TRAJ_COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)


def read_video(video_path: Path, video_freq: float):
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_type = av.codec.context.ThreadType.NONE
        for packet in container.demux(stream):
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    frames_arr = np.stack(frames).astype(np.uint8)
    T = len(frames_arr)
    timestamps_ns = (np.arange(T, dtype=np.float64) / video_freq * S_TO_NS).astype(np.int64)
    return frames_arr, timestamps_ns


def write_zarr(output_path, frames, video_ts, actions, obs_state, act_ts, prompt):
    T_vid, H, W, C = frames.shape
    T_act = len(actions)
    root = zarr.open(str(output_path), mode="w")
    root.array("workspace_rgb", frames, chunks=(1, H, W, C),
               dtype=np.uint8, compressor=IMG_COMPRESSOR)
    root.array("workspace_rgb_timestamps", video_ts, chunks=(T_vid,),
               dtype=np.int64, compressor=TRAJ_COMPRESSOR)
    root.array("actions", actions, chunks=(T_act, 5),
               dtype=np.float32, compressor=TRAJ_COMPRESSOR)
    root.array("actions_timestamps", act_ts, chunks=(T_act,),
               dtype=np.int64, compressor=TRAJ_COMPRESSOR)
    root.array("observation_state", obs_state, chunks=(T_act, 5),
               dtype=np.float32, compressor=TRAJ_COMPRESSOR)
    root.array("observation_state_timestamps", act_ts, chunks=(T_act,),
               dtype=np.int64, compressor=TRAJ_COMPRESSOR)
    root.array("language_instruction", prompt, dtype=object, object_codec=VLenUTF8())


def process_chunk(chunk_dir: Path, video_dir: Path, output_dir: Path,
                  episode_start: int, prompt: str, video_freq: float,
                  overwrite: bool) -> int:
    # Load all frames for this chunk
    data_parquet = list(chunk_dir.glob("data/chunk-*/file-*.parquet"))[0]
    df = pd.read_parquet(data_parquet)

    ep_count = 0
    for local_ep_idx in sorted(df["episode_index"].unique()):
        ep_global = episode_start + ep_count
        output_path = output_dir / f"ep{ep_global}.zarr"

        if output_path.exists() and not overwrite:
            ep_count += 1
            continue

        ep_df = df[df["episode_index"] == local_ep_idx].sort_values("frame_index")

        # Actions and obs — drop gripper (last column, index 5)
        actions = np.stack(ep_df["action"].values).astype(np.float32)[:, :5]
        obs_state = np.stack(ep_df["observation.state"].values).astype(np.float32)[:, :5]
        timestamps_ns = (ep_df["timestamp"].values.astype(np.float64) * S_TO_NS).astype(np.int64)

        # Video from pre-converted per-episode mp4
        video_path = video_dir / f"ep{ep_global}.mp4"
        if not video_path.exists():
            print(f"  WARNING: missing video {video_path}, skipping ep{ep_global}")
            ep_count += 1
            continue

        frames, video_ts = read_video(video_path, video_freq)

        write_zarr(output_path, frames, video_ts, actions, obs_state, timestamps_ns, prompt)
        ep_count += 1

    return episode_start + ep_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", required=True,
                        help="One or more LeRobot data_raw directories (e.g. /ephemeral/data_raw /ephemeral/data_raw_scaevitas)")
    parser.add_argument("--video-dir", required=True,
                        help="Directory with pre-converted ep*.mp4 files")
    parser.add_argument("--output-dir", required=True,
                        help="Output zarr directory (e.g. /ephemeral/rl-group13-processed)")
    parser.add_argument("--prompt", default="push the white ball into the white circle")
    parser.add_argument("--video-freq", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_idx = 0
    failed = []

    for data_root in args.data_dirs:
        data_root = Path(data_root)
        chunks = sorted([d for d in data_root.iterdir()
                         if d.is_dir() and d.name.startswith("t1-e")])
        print(f"\n{data_root.name}: {len(chunks)} chunks")

        for chunk_dir in tqdm.tqdm(chunks, desc=data_root.name):
            try:
                ep_idx = process_chunk(chunk_dir, video_dir, output_dir,
                                       ep_idx, args.prompt, args.video_freq, args.overwrite)
            except Exception as e:
                failed.append((str(chunk_dir), str(e)))
                tqdm.tqdm.write(f"  ERROR {chunk_dir.name}: {e}")
                tqdm.tqdm.write(traceback.format_exc())

    print(f"\n✓ Wrote {ep_idx} episodes to {output_dir}")
    if failed:
        print(f"  {len(failed)} chunks failed:")
        for name, err in failed:
            print(f"    {name}: {err}")


if __name__ == "__main__":
    main()
