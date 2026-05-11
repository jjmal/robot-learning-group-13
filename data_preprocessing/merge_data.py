import os
import shutil
import argparse
import subprocess
import pandas as pd
# import pyarrow
# import fastparquet
from pathlib import Path


def load_lenghts(meta_path) -> list[tuple[int, int]]:
    """Load (start_frame, end_frame) intervals from a Parquet file."""
    df = pd.read_parquet(meta_path, columns=["length"])
    return list(df["length"].astype(int))


def split_video(video_path: str, meta_path: str, out_dir: str, start_ep: int, fps: int = 30):
    
    start_frame = 0

    lengths = load_lenghts(meta_path)

    for i, n_frames in enumerate(lengths):
        true_i = start_ep + i
        out_path = out_dir + f"/ep{true_i}.mp4"
 
        # if out_path.exists():
        #     print(f"[{i+1}/{len(frame_counts)}] Skipping {out_path.name} (already exists)")
        #     start_frame += n_frames
        #     continue
 
        start_sec = start_frame / fps
        duration_sec = n_frames / fps
 
        print(f"Writing {out_path} "
              f"(frames {start_frame}–{start_frame + n_frames - 1}, "
              f"{start_sec:.3f}s – {start_sec + duration_sec:.3f}s)")
        
 
        subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-ss", str(start_sec),       # seek to start (fast, keyframe-aligned)
                "-i", str(video_path),
                "-t", str(duration_sec),     # duration of this episode
                "-c", "copy",                # no re-encoding — fast and lossless
                "-avoid_negative_ts", "1",
                out_path,
            ],
            check=True,
        )
        
        
        start_frame += n_frames
 

def split_parquet(parquet_path: str, out_dir: str, start_ep: int) -> int:

    df = pd.read_parquet(parquet_path)

    for ep in range(int(df['episode_index'].max() + 1)):
        df_ep = df[df['episode_index'] == ep]
        df_ep.to_parquet(out_dir + f"/ep{start_ep + ep}.parquet", index=False)

    return int(df['episode_index'].max())

def merge_task(root_dir: str, task_nr: int) -> int:
    dir_raw = root_dir + "/data_raw"
    out_dir = root_dir + f"/data_merged/t{task_nr}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_dir_vid = out_dir + "/video"
    out_dir_action = out_dir + "/action"
    Path(out_dir_vid).mkdir(exist_ok=True)
    Path(out_dir_action).mkdir(exist_ok=True)

    start_ep = 0

    datafolders = os.listdir(dir_raw)
    for fold in datafolders:
        fold_path = dir_raw + f"/{fold}"
        if f"t{task_nr}" in fold:
            split_video(
                fold_path + "/videos/observation.images.top/chunk-000/file-000.mp4", 
                fold_path + "/meta/episodes/chunk-000/file-000.parquet",
                out_dir_vid,
                start_ep
                )
            nr_of_eps = split_parquet(
                fold_path +  "/data/chunk-000/file-000.parquet",
                out_dir_action,
                start_ep
            ) + 1
            start_ep += nr_of_eps

    return start_ep
        

def main():
    parser = argparse.ArgumentParser(description="Merge the raw data into a correct format per each task.")
    parser.add_argument("--root-dir", required=True, help="Path to the root project directory")

    args = parser.parse_args()
    
    root_dir = args.root_dir
    Path(root_dir + "/data_merged").mkdir(parents=True, exist_ok=True)

    for t in range(1,4):
        merge_task(root_dir,t)

    
if __name__ == "__main__":
    main()


