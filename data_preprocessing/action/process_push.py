import argparse
import logging
import pathlib
import pickle
import re
import numpy as np
import tqdm
import zarr


from functools import partial
from multiprocessing import Pool
from typing import Literal



def _convert(dataset_path: pathlib.Path, out_dir: pathlib.Path):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        required=True,
        help="Dataset folder",
    )
    ap.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Folder to write per-demo .zarr files",
    )
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Used to artificially create timestamps (ns)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(list(args.dataset_dir.glob("**/*.parquet")) + list(args.dataset_dir.glob("**/*.parquet")))
    assert len(h5_paths) > 0, f"No Parquet files found in {args.dataset_dir}"

    with Pool(processes=args.num_workers) as pool:
        for msg in tqdm.tqdm(
            pool.imap_unordered(
                partial(
                    _convert,
                    out_dir=args.output_dir,
                    fps=args.fps,
                    overwrite=args.overwrite,
                ),
                h5_paths,
            ),
            total=len(h5_paths),
            desc="push -> zarr",
        ):
            print(msg)


if __name__ == "__main__":
    main()
