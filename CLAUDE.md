# Robot Learning Group 13 — Claude Context

## Project
Fine-tuning the Cosmos Predict2 video2world backbone and action head for SO-100 robot manipulation.
Task: "push the white ball into the white circle"

## Environment
- Working directory: `/home/ubuntu/robot-learning-group-13`
- Model code: `model/`
- Venv: `source model/.venv/bin/activate` (always activate before running any python)
- Large data/checkpoints go on `/ephemeral` (wiped on reboot, 500GB)
- Root disk is nearly full — never write large files outside `/ephemeral`
- GPU: A100 80GB on Brev (`ubuntu@brev-cfcpd4pd9`)
- Use `tmux` to keep training alive after SSH disconnect

## Pipeline Overview

### 1. Backbone fine-tuning (video2world)
Fine-tunes the 2B Cosmos DiT on domain-specific robot videos using LoRA.

**Data format**: `video/*.mp4` + `t5_xxl/*.pickle` per episode
- Videos must be H.264 encoded (AV1 not supported by decord)
- Re-encode AV1 videos: `ffmpeg-linux-x86_64-v7.0.2 -i input.mp4 -c:v libx264 -crf 18 -preset fast output.mp4`
- T5 pickle format: `pickle.dump([emb.squeeze(0).astype(np.float32)], f)` — must be a `list` with one `np.ndarray` of shape `(512, 1024)`

**Dataset registration**: add entry to `model/cosmos_predict2/configs/defaults/data_video.py`

**Training command** (from `model/`):
```bash
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGINAIRE_OUTPUT_ROOT=/ephemeral/training_output \
torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=v2w_rl_group13_lora_rank256_lr1.778e-04_bsz32 \
  video_dataset_train.dataset_dir=/ephemeral/rl-group13-merged-h264 \
  video_dataset_val.dataset_dir=/ephemeral/rl-group13-merged-h264 \
  trainer.logging_iter=100 \
  dataloader_train.batch_size=2 \
  trainer.max_iter=10000 \
  checkpoint.save_iter=100
```

- Resume: run the exact same command — auto-detects latest checkpoint
- Checkpoints saved to `/ephemeral/training_output/posttraining/video2world/<exp_name>/checkpoints/model/`
- `_fused.pt` files are LoRA already merged — no manual fusing needed
- Hydra overrides: NO `--` prefix, after the `--` separator (e.g. `trainer.logging_iter=100`)

**Gotchas**:
- `predict2_video2world_ddp_2b_480p_10fps` model name was missing — added alias in `model/cosmos_predict2/configs/defaults/video2world_model.py`
- batch_size=4 works on A100 without action head training running; batch_size=1 if memory tight
- Loss is very noisy for diffusion models — individual values don't mean much, check trends

### 2. Precompute T5 text embeddings (action head data)
**Data format**: zarr episodes with fields: `workspace_rgb`, `actions`, `observation_state`, `language_instruction`, `language_embedding`

```bash
cd /home/ubuntu/robot-learning-group-13
python -m data_preprocessing.action.precompute_t5 \
  --dataset-path /ephemeral/rl-group13-processed \
  --prompt "push the white ball into the white circle"
```

**Gotcha**: delete HuggingFace download cache first or glob picks up broken zarrs:
```bash
rm -rf /ephemeral/rl-group13-processed/.cache
```

### 3. Precompute video backbone embeddings (TA hint)
The action head training runs the 2B backbone every step — very slow. Pre-compute `crossattn_emb` (hidden states at layer 20) once per episode at a fixed sigma and save to zarr. Then action head training loads from disk, no backbone needed.

```bash
cd /home/ubuntu/robot-learning-group-13
python -m data_preprocessing.action.precompute_video_embeddings \
  --dataset-path /ephemeral/rl-group13-processed \
  --dit-path /ephemeral/training_output/posttraining/video2world/v2w_rl_group13_lora_rank256_lr1.778e-04_bsz32/checkpoints/model/iter_000000800_fused.pt \
  --stride 1 \
  --sigma 0.4
```

- `--stride 1`: maximum data → ~190 windows/episode (~3hr precompute, run in tmux)
- `--sigma 0.4`: geometric mean of training sigma range [0.002, 80.0]
- Omit `--dit-path` to use base pretrained model instead of fine-tuned
- Saves `crossattn_emb` (n_windows, N_tokens, D) and `crossattn_emb_window_starts` to each zarr
- Script: `data_preprocessing/action/precompute_video_embeddings.py`

### 4. Action head training
**Data**: `/ephemeral/rl-group13-processed` (6 episodes currently, needs more)
**Config**: SO-100-specific `policy_io` yaml already created (`rl_group13`) — 6-dim actions at 30Hz

**Training command** (from `model/`):
```bash
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGINAIRE_OUTPUT_ROOT=/ephemeral/training_output \
torchrun --nproc_per_node=1 --master_port=12342 \
  -m scripts.train \
  --config=cosmos_predict2/configs/config.py \
  -- experiment=w2a_rl_group13_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1 \
  job.name=w2a_rl_group13 \
  trainer.logging_iter=10 \
  trainer.max_iter=5000 \
  checkpoint.save_iter=500 \
  trainer.run_validation=False
```

- Use port `12342` (backbone training uses `12341`)
- `trainer.run_validation=False`: validation runs the full 2B backbone on every val sample (~2min each) — skip it during training
- Checkpoints saved to `/ephemeral/training_output/posttraining/world2action/w2a_rl_group13/`
- Resume: run the exact same command — auto-detects latest checkpoint

**Gotchas**:
- Precomputed `crossattn_emb` is loaded automatically from zarr (configured in `rl_group13` dataset config) — training skips the 2B backbone forward pass each step, making training fast
- Validation still runs the backbone (for MSE sweep + video generation) — always use `trainer.run_validation=False` unless you specifically need val metrics
- OOM during normalization stats: if you see SIGKILL at "Iterating dataset to get normalization", it means `crossattn_emb` arrays are being loaded into RAM — this is fixed by the `restrict_keys` guard in `chunk_reader.py`
- `ep13.zarr` warning about missing `workspace_rgb_timestamps` is harmless — that episode is skipped

## Data Notes
- SO-100 robot data: 30Hz, 6-dim actions, 6-dim proprioception, 480×640 RGB
- Dataset on HuggingFace: `smoothmoth121/rl-group13-processed` (zarr format)
- Downloading zarr from HuggingFace hits rate limits fast (1000 req/5min) — zip before uploading
- Visual issue: black robot on black mat — poor contrast hurts learning. Consider colored tape on joints.
- Currently only 6 episodes — need significantly more data

## Video Generation (evaluation)
```bash
cd model/
python scripts/run_video2world.py \
  --dit_path /ephemeral/training_output/posttraining/video2world/<exp>/checkpoints/model/iter_XXXXXX_fused.pt \
  --input_path <video.mp4> \
  --prompt "push the white ball into the white circle" \
  --num_conditional_frames 5 \
  --save_path /ephemeral/output.mp4 \
  --disable_guardrail \
  --use_first_frames
```

- `--use_first_frames`: conditions on first 5 frames (start of task) instead of last 5 (end, static)
- Base model (no `--dit_path`): uses `checkpoints/video_backbone/v2w_pretrained_cosmos.pt`

## Key Files
- `model/cosmos_predict2/configs/defaults/data_video.py` — register new video datasets
- `model/cosmos_predict2/configs/defaults/video2world_model.py` — model registration
- `model/cosmos_predict2/configs/dataloading/policy_io/rl_group13.yaml` — SO-100 action/obs horizons (6-dim, 30Hz)
- `model/cosmos_predict2/configs/dataloading/dataset/rl_group13.yaml` — SO-100 dataset config (has `load_precomputed_crossattn_emb: True`)
- `model/cosmos_predict2/configs/dataloading/rl_group13.yaml` — top-level data config pointing to `/ephemeral/rl-group13-processed`
- `model/cosmos_predict2/configs/defaults/world2action_pipe.py` — action head network configs (rl_group13: 6-dim in/out)
- `model/cosmos_predict2/configs/experiment/world2action.py` — action head experiment configs
- `model/cosmos_predict2/models/world2action_model.py` — action head training logic (modified: uses precomputed crossattn_emb if in batch)
- `model/cosmos_predict2/data/action/chunk_reader.py` — zarr data loader (modified: loads precomputed crossattn_emb)
- `model/cosmos_predict2/pipelines/video2world.py` — video generation pipeline (modified: added `--use_first_frames`)
- `data_preprocessing/action/precompute_video_embeddings.py` — precomputes crossattn_emb from backbone and saves to zarr
