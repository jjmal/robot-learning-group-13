import os
import shutil
import re
import argparse
import subprocess
import pandas as pd
# import pyarrow
# import fastparquet
from pathlib import Path

def add_metas(root_dir: str, path_to_embedding: str, task_nr: int):
    out_path = root_dir + f"/data_merged/t{task_nr}/t5_xxl"
    task_action_data_path = root_dir + f"/data_merged/t{task_nr}/action"

    task_numbers = []
    for fil in os.listdir(task_action_data_path):
        task_nr = re.findall('\d+', fil)[0]
        task_numbers.append(int(task_nr))
    nr_of_eps = max(task_numbers)

    for i in range(nr_of_eps + 1):
        shutil.copy(path_to_embedding, out_path + f"/ep{i}.pickle") 

def main():
    parser = argparse.ArgumentParser(description="Merge the raw data into a correct format per each task.")

    parser.add_argument("--root-dir", required=True, help="Path to the root project directory")
    parser.add_argument("--emb-path", required=True, help="Path to the root project directory")
    parser.add_argument("--task-nr", required=True, help="Path to the root project directory")

    args = parser.parse_args()
    
    root_dir = args.root_dir
    emd_path = args.emb_path
    task_nr = int(args.task_nr)

    Path(root_dir + f"/data_merged/t{task_nr}/t5_xxl").mkdir(parents=True, exist_ok=True)

    add_metas(root_dir, emd_path, task_nr)

if __name__ == "__main__":
    main()