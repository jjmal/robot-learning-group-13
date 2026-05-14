"""
Precompute video backbone embeddings (crossattn_emb) for action head training.

Implements the TA hint: "pre-compute the embedded space".

For each zarr episode, slides a window of 61 frames (5 obs + 56 action),
runs the frozen video backbone at a fixed sigma, and saves the resulting
cross-attention embeddings to zarr. Action head training can then load
these embeddings from disk instead of running the 2B backbone every step.

Usage:
    python -m data_preprocessing.action.precompute_video_embeddings \
        --dataset-path /ephemeral/rl-group13-processed \
        --dit-path /ephemeral/training_output/.../iter_000000800_fused.pt \
        --stride 10 \
        --sigma 1.0
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
import zarr
from einops import rearrange
from numcodecs import Blosc
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "model"))

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint

NUM_OBS_FRAMES = 5
NUM_ACTION_FRAMES = 56
TOTAL_FRAMES = NUM_OBS_FRAMES + NUM_ACTION_FRAMES  # 61
XATTN_LAYER_IDX = 20


def load_pipeline(dit_path: str) -> Video2WorldPipeline:
    config = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    config.guardrail_config.enabled = False
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()
    return pipe


@torch.no_grad()
def compute_crossattn_emb(
    pipe: Video2WorldPipeline,
    frames_uint8: np.ndarray,   # (61, H, W, 3) uint8
    language_embedding: np.ndarray,  # (1, 512, 1024)
    sigma: float,
) -> np.ndarray:
    # (61, H, W, 3) -> (1, 3, 61, H, W) uint8
    frames_t = torch.from_numpy(frames_uint8).permute(3, 0, 1, 2).unsqueeze(0)
    frames_t = frames_t.to("cuda")

    obs_frames = frames_t[:, :, :NUM_OBS_FRAMES, :, :]    # (1, 3, 5, H, W)
    action_frames = frames_t[:, :, NUM_OBS_FRAMES:, :, :]  # (1, 3, 56, H, W)

    lang_emb = torch.from_numpy(language_embedding).to("cuda", dtype=torch.bfloat16)

    data_batch = {
        "obs/workspace_rgb": obs_frames,
        "action/workspace_rgb": action_frames,
        "obs/language_embedding": lang_emb,
        "num_conditional_frames": pipe.tokenizer.get_latent_num_frames(NUM_OBS_FRAMES),
        "is_preprocessed": True,
    }

    _, video_latent, condition = pipe.get_mimic_data_and_condition(data_batch)

    sigma_t = torch.tensor([[sigma]], device="cuda", dtype=torch.float32)
    noise = torch.randn_like(video_latent)
    noisy_latent = video_latent + noise * rearrange(sigma_t, "b t -> b 1 t 1 1")

    world_pred = pipe.denoise(
        noisy_latent,
        sigma_t,
        condition,
        use_cuda_graphs=False,
        return_only_hidden_states_up_to=XATTN_LAYER_IDX,
        return_decoded_video=False,
    )

    crossattn_emb = world_pred.hidden_states[XATTN_LAYER_IDX]  # (1, T, H, W, D)
    B, T, H, W, D = crossattn_emb.shape
    crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)
    return crossattn_emb.squeeze(0).cpu().float().numpy()  # (N_tokens, D)


def process_episode(
    pipe: Video2WorldPipeline,
    zarr_path: pathlib.Path,
    stride: int,
    sigma: float,
) -> None:
    with zarr.open(str(zarr_path), "r") as root:
        workspace_rgb = root["workspace_rgb"][:]       # (N, H, W, 3) uint8
        language_embedding = root["language_embedding"][:]  # (1, 512, 1024)

    n_frames = workspace_rgb.shape[0]
    if n_frames < TOTAL_FRAMES:
        print(f"Skipping {zarr_path.name}: only {n_frames} frames (need {TOTAL_FRAMES})")
        return

    starts = list(range(0, n_frames - TOTAL_FRAMES + 1, stride))
    embeddings = []

    for start in tqdm(starts, desc=zarr_path.name, leave=False):
        frames = workspace_rgb[start : start + TOTAL_FRAMES]
        emb = compute_crossattn_emb(pipe, frames, language_embedding, sigma=sigma)
        embeddings.append(emb)

    embeddings_arr = np.stack(embeddings, axis=0)  # (n_windows, N_tokens, D)
    window_starts = np.array(starts, dtype=np.int32)

    compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
    with zarr.open(str(zarr_path), "r+") as root:
        root.create_dataset(
            "crossattn_emb",
            data=embeddings_arr,
            chunks=(1, embeddings_arr.shape[1], embeddings_arr.shape[2]),
            compressor=compressor,
            overwrite=True,
        )
        root.create_dataset(
            "crossattn_emb_window_starts",
            data=window_starts,
            overwrite=True,
        )
        root.attrs["crossattn_sigma"] = sigma
        root.attrs["crossattn_layer_idx"] = XATTN_LAYER_IDX
        root.attrs["crossattn_stride"] = stride

    print(f"{zarr_path.name}: saved {len(starts)} windows, shape {embeddings_arr.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dit-path",
        type=str,
        default=get_cosmos_predict2_video2world_checkpoint(),
        help="Path to fused backbone checkpoint (default: pretrained base)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride in frames. Smaller = more windows = more training data but slower precompute.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.4,
        help="Fixed noise level for embedding extraction. 0.4 = geometric mean of training range [0.002, 80.0].",
    )
    args = parser.parse_args()

    print(f"Loading pipeline from: {args.dit_path}")
    pipe = load_pipeline(args.dit_path)

    zarr_paths = sorted(args.dataset_path.glob("*.zarr"))
    print(f"Found {len(zarr_paths)} episodes")

    for zarr_path in zarr_paths:
        process_episode(pipe, zarr_path, stride=args.stride, sigma=args.sigma)

    print("Done.")


if __name__ == "__main__":
    main()
