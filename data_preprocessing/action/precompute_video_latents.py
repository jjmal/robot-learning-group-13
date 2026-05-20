"""
Precompute VAE latents (clean video embeddings) for action head training.

Implements Algorithm 2 from the paper: saves clean latents zfuture_0 and zpast_0
to disk. During action head training, noise is added on-the-fly with a randomly
sampled τv (logit-normal, matching backbone pre-training), and the DiT runs each
training step to produce hτv = v_ϕ(zpast_0, zfuture_τv, l, τv).

This is different from precompute_video_embeddings.py which baked in a fixed
sigma=0.4 and saved the final crossattn_emb — incorrect per Algorithm 2.

Usage:
    cd /home/ubuntu/robot-learning-group-13
    source model/.venv/bin/activate
    python -m data_preprocessing.action.precompute_video_latents \
        --dataset-dir /ephemeral/rl-group13-merged-h264 \
        --target-fps 30 \
        --data-fps 30 \
        --num-samples-per-second 2 \
        --obs-history 5 \
        --num-frames 61

Output: dataset_dir/video_embeddings/<episode>.safetensors with keys:
    video_embeddings       — (N, C, T, H, W) clean VAE latents
    video_embeddings_idxs — (N,) anchor frame indices into original video
    video_len              — scalar, total frames in original video
    fps                    — scalar, target fps
    num_conditional_frames — scalar, latent frames for obs history
"""

import argparse
import math
import operator
import os
import pathlib

import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F
import tqdm
from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from decord import VideoReader, cpu
from imaginaire.lazy_config import instantiate
from torch.utils.data import Dataset as _Dataset

os.environ["SAFETENSORS_FAST_GPU"] = "1"
torch.backends.cudnn.benchmark = True


class Dataset(_Dataset):
    def __init__(
        self,
        video_paths: list[pathlib.Path],
        num_frames: int,
        video_height: int,
        video_width: int,
        target_fps: int,
        seed: int,
        num_samples_per_second: int | None,
        data_fps: int | None = None,
        obs_history: int = 5,
    ) -> None:
        super().__init__()

        self._rng = np.random.default_rng(seed)

        self._video_paths = video_paths
        self._sequence_length = num_frames
        self._video_height = video_height
        self._video_width = video_width
        self._target_fps = target_fps
        self._num_samples_per_second = num_samples_per_second
        self._data_fps = data_fps
        self._obs_history = obs_history

    def __len__(self) -> int:
        return len(self._video_paths)

    def _get_frames_batch(self, video_path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, int, pathlib.Path]:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=0, fault_tol=0.01)
        n = len(vr)
        if n == 0:
            raise ValueError(f"Empty video: {video_path}")

        fps = self._data_fps or vr.get_avg_fps()
        step = fps / self._target_fps

        high = int(n + step * (self._obs_history - 1))

        if (
            self._num_samples_per_second is not None
            and (num_samples := round(self._num_samples_per_second * high / fps)) < high
        ):
            i = self._rng.choice(high, size=num_samples, replace=False, shuffle=False)
            i.sort()
        else:
            num_samples = high
            i = np.arange(high)

        T = self._sequence_length
        k = np.arange(T, dtype=np.float64) - (self._obs_history - 1)

        idx = np.rint(i[:, None] + k[None, :] * step).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)

        uniq, inverse = np.unique(idx, return_inverse=True)

        frames_np = vr.get_batch(uniq).asnumpy()

        del vr

        x = torch.from_numpy(frames_np).permute(3, 0, 1, 2).contiguous()
        C, _num_uniq, H, W = x.shape

        x = F.interpolate(
            x.float().view(-1, 1, H, W),
            size=(self._video_height, self._video_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(C, -1, self._video_height, self._video_width)
        x = x.round().clamp_(0, 255).to(torch.uint8)

        j = torch.from_numpy(inverse).to(torch.long)
        x = x.index_select(1, j)

        x = x.view(C, num_samples, T, self._video_height, self._video_width).permute(1, 0, 2, 3, 4).contiguous()

        return x, torch.from_numpy(i), n, video_path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, pathlib.Path]:
        return self._get_frames_batch(self._video_paths[index])


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Compute video embeddings.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, help="Root path to the dataset")
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--target-fps", type=int, required=True)
    parser.add_argument(
        "--num-samples-per-second",
        type=int,
        default=None,
        help=(
            "Anchor frames to sample per second of video. Default (None) = all frames."
        ),
    )
    parser.add_argument("--data-fps", type=int, required=False)
    parser.add_argument("--obs-history", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-slice-length",
        type=int,
        default=1,
        help=(
            "How big a minibatch that goes into the tokenizer is. Tune this. On GH200, actually 1 seems best."
            " If >1, consider padding the minibatches."
        ),
    )
    parser.add_argument("--include-only-with-substring", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)

    args = parser.parse_args()

    video_paths = sorted(
        [
            video
            for video in (args.dataset_dir / "video").iterdir()
            if video.suffix == ".mp4" and args.include_only_with_substring in str(video)
        ]
    )
    video_paths = video_paths[args.rank :: args.world_size]

    (args.dataset_dir / "video_embeddings").mkdir(exist_ok=True)
    tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

    ds = Dataset(
        video_paths,
        args.num_frames,
        480,
        640,
        args.target_fps,
        args.seed,
        args.num_samples_per_second,
        args.data_fps,
        args.obs_history,
    )
    tokenizer: TokenizerInterface = instantiate(
        get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10).tokenizer
    ).to(**tensor_kwargs)
    tokenizer.encode = torch.compile(tokenizer.encode, dynamic=False, backend="inductor")  # ty:ignore[invalid-assignment]

    slice_len = args.tokenizer_slice_length
    for video_frames, idxs, video_len, video_path in tqdm.tqdm(
        torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            collate_fn=operator.itemgetter(0),
            num_workers=0,
        ),
        desc="Encoding video samples.",
        total=len(ds),
    ):
        target_file = args.dataset_dir / "video_embeddings" / f"{video_path.stem}.safetensors"
        if not args.overwrite and target_file.exists():
            continue

        # process slice-by-slice to avoid OOM (video_frames stays on CPU)
        video_embeddings = torch.cat(
            [
                tokenizer.encode(
                    (video_frames[i * slice_len : (i + 1) * slice_len].to(**tensor_kwargs) / 127.5 - 1.0)
                ).cpu()
                for i in range(math.ceil(video_frames.shape[0] / slice_len))
            ]
        )

        st.save_file(
            {
                "video_embeddings": video_embeddings.float(),
                "video_embeddings_idxs": idxs,
                "video_len": torch.tensor(video_len),
                "fps": torch.tensor(args.target_fps),
                "num_conditional_frames": torch.tensor(tokenizer.get_latent_num_frames(args.obs_history)),
            },
            target_file,
        )

    print("done.")


if __name__ == "__main__":
    main()
