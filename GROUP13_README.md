# Setup

In addition to performing the setup as indicated in the original README file, install the additional Python packages by running:
```
pip install av pyarrow
``` 
if you are using WSL, you may need to install pip first (as is it not included by default in the built environment):
```
python -m ensurepip --upgrade
python -m pip install av pyarrow
```
Furthermore, make sure that you have ffmpeg installed (for working with video data)

# Data access

Run

```
bash data_preprocessing/get_data.sh <name> <task> <start> <end> [local_dir]
```
where `<name>` indicates account name, `<task>` task number (1-3), `<start>, <end>` starting and finishing dataset name, `[local_dir]` directory name to which to download the files. Example usage:
```
bash data_preprocessing/get_data.sh Scaevitas 1 1 10 ./data_raw
```
This will create a `data_raw` folder 

# Data preprocessing




## Structuring the dataset

Having downloaded the dataset, run from the main project folder:
```
python data_processing/merge_data.py --root_dir .
```

This will create a `data_merged` folder, which contains the data structure intended by the original repo authors. You can then use further preprocessing commands on this folder.

## Adding word embeddings

You can find the task prompts in `prompts.py`. Copy each prompt into ...

Nake sure that the embedding file is named `ep.pickle`. Then, run from the main project folder:

```
python data_processing/add_embeddings_to_merged.py --root_dir .
```

## Video

From the main project folder run:
```
python data_processing/merge_data.py --root_dir .
```


## Action

## Embedding textual instructions
To extract text embeddings from raw data, run data_preprocesing/video/get_t5_embeddings.py . In --cache_dir you specify the model name (!) from hugginhface. Example: python data_preprocessing/video/get_t5_embeddings.py --dataset_path dataset --cache_dir google-t5/t5-small 

