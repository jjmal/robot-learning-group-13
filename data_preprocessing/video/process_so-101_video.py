import os
import subprocess
import argparse
from tqdm import tqdm


def main():

    parser = argparse.ArgumentParser(description="Merge the raw data into a correct format per each task.")
    parser.add_argument("--data-dir", required=True, help="Path to the data_merged directory")

    args = parser.parse_args()

    data_dir  = args.data_dir
    
    # Re-encode the videos
    video_dir = data_dir  + "/video"
    video_files = os.listdir(video_dir)

    for filename in tqdm(video_files, desc="Re-encoding videos"):
        if not filename.endswith(".mp4"):
            continue
        path = os.path.join(video_dir, filename)
        tmp = path + ".tmp.mp4"
        print(f"Processing: {filename}")
        subprocess.run([
            "ffmpeg", "-i", path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an", tmp, "-y"
        ], check=True)
        os.replace(tmp, path)
        print(f"Done: {filename}")

if __name__ == "__main__":
    main()