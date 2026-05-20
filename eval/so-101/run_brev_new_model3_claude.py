"""
run_brev_new_model3_claude.py — runs on BREV, connects to relay_server.py on laptop.

Usage:
    cd /home/ubuntu/robot-learning-group-13/eval/so-101
    conda activate mimicvideo
    python run_brev_new_model3_claude.py --relay-url http://localhost:5000
"""

import argparse
import base64
import json
import pathlib
import sys
import threading
import time
import cv2
import imageio
import numpy as np
import requests
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "model"))

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.config_world2action import SchedulerConfig, World2ActionPipelineConfig
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import LazyCall as L

# ── paths ─────────────────────────────────────────────────────────────────────
BACKBONE_PATH    = "/ephemeral/checkpoints/backbone/iter_000020000_fused.pt"
ACTION_HEAD_PATH = "/ephemeral/training_output/vam/rl_group13/w2a_rl_group13/checkpoints/model/iter_000050000.pt"
NORM_STATS_PATH  = "/ephemeral/rl-group13-processed/.statistics_cache/20d299217656659812638ab8f0988362d9b69bb6a6b9d0f7678e3779b0b98f3c"
LANG_EMB_PATH    = "/ephemeral/language_embedding.npy"
ROLLOUT_DIR      = "./video_rollouts" 

# ── constants (must match training) ───────────────────────────────────────────
NUM_OBS_FRAMES        = 5
XATTN_LAYER_IDX       = 20
OBS_DIM               = 5
ACTION_HORIZON        = 60    # actions executed per chunk (out of 90 predicted)
IMG_H, IMG_W          = 480, 640
ACTION_FREQ           = 30    # Hz — robot control rate (action execution)
OBS_FREQ              = 30    # Hz — obs frame collection rate (5 for old 5Hz policy, 30 for new 30fps policy)

# Partial-denoising hyperparameters (paper-style imagined future):
# - NUM_SAMPLING_STEPS=35 matches training scheduler
# - STOP_AFTER_STEP=24 lands at sigma≈0.377 (closest to training sigma 0.4)
# Lowering STOP_AFTER_STEP makes inference faster but moves sigma away from 0.4,
# creating a train/inference noise-level mismatch.
NUM_SAMPLING_STEPS   = 35
STOP_AFTER_STEP_DEFAULT = 0     # τv=1: pure noise future, single DiT pass (Algorithm 1)
MAX_CHUNKS_DEFAULT      = 10

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_FIXED_POS = 0.0


# ── relay client ──────────────────────────────────────────────────────────────

class RelayClient:
    def __init__(self, base_url: str, timeout: float = 50.0):
        self.base    = base_url.rstrip("/")
        self.timeout = timeout

    def get_observation(self) -> tuple[np.ndarray, np.ndarray, bool]:
        r = requests.get(f"{self.base}/observation", timeout=self.timeout).json()
        img_bytes = base64.b64decode(r["image"])
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        image     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        joints_dict = r["joints"]
        joints = np.array(
            [joints_dict[f"{j}.pos"] for j in JOINT_NAMES],
            dtype=np.float32,
        )
        return image, joints, bool(r.get("stop", False))

    def send_action(self, action: dict) -> None:
        requests.post(f"{self.base}/action", json=action, timeout=self.timeout)

    def reset_robot(self) -> None:
        requests.post(f"{self.base}/reset", json={"steps": 50, "hz": 25.0}, timeout=120.0)

    def clear_stop(self) -> None:
        requests.post(f"{self.base}/clear_stop", timeout=self.timeout)

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base}/health", timeout=2.0).ok
        except Exception:
            return False


# ── model loading ─────────────────────────────────────────────────────────────

def load_backbone() -> Video2WorldPipeline:
    print("Loading backbone...")
    config = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    config.guardrail_config.enabled = False
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=BACKBONE_PATH,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()
    print("  Backbone loaded.")
    return pipe


def load_action_head() -> World2ActionPipeline:
    print("Loading action head...")
    net_config = L(World2ActionDIT)(
        max_horizon=91,
        in_channels=5,
        out_channels=5,
        model_channels=512,
        num_blocks=10,
        num_heads=8,
        mlp_ratio=4.0,
        atten_backend="flash_attn_no_cp",
        crossattn_emb_channels=2048,
        use_adaln_lora=True,
        adaln_lora_dim=64,
        pair_timestep_feature_rank=512,
        sac_config=SACConfig(mode="none", every_n_blocks=1),
    )
    config = World2ActionPipelineConfig(
        precision="bfloat16",
        scheduler=SchedulerConfig(alpha=1.0, beta=1.0, num_denoising_steps=10),
        net=net_config,
        ema=EMAConfig(enabled=False),
        xattn_layer_idx=XATTN_LAYER_IDX,
    )
    pipe = World2ActionPipeline.from_config(
        config=config,
        dit_path=ACTION_HEAD_PATH,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()
    print("  Action head loaded.")
    return pipe


def load_norm_stats(pipe: World2ActionPipeline) -> None:
    print("Loading norm stats...")
    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)
    normalization_types = {
        "action/actions":        NormalizationType.VARIANCE,
        "obs/observation_state": NormalizationType.VARIANCE,
    }
    concat_groups = {
        "action/lowdim_concat": ["action/actions"],
        "obs/lowdim_concat":    ["obs/observation_state"],
    }
    pipe.normalizer.build_from_stats(
        stats,
        normalization_types=normalization_types,
        concat_groups=concat_groups,
        dtype=torch.bfloat16,
        device="cuda",
    )
    print("  Norm stats loaded.")


# ── backbone forward (paper-style partial denoising) ──────────────────────────

@torch.no_grad()
def backbone_forward(
    backbone: Video2WorldPipeline,
    obs_frames_uint8: np.ndarray,   # (5, H, W, 3) — obs frames only
    lang_emb: np.ndarray,           # (1, 512, 1024)
    stop_after_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # (5, H, W, 3) uint8 → (1, 3, 5, H, W)
    obs_t = torch.from_numpy(obs_frames_uint8).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
    lang_t = torch.from_numpy(lang_emb).to("cuda", dtype=torch.bfloat16)

    # Algorithm 1 at τv=1: future slots = pure noise, 0 denoising steps.
    # DiT runs once on (clean obs + pure noise future) → extract layer 20 hidden states.
    # Action head then runs full 10-step denoising conditioned on hτv.
    crossattn_emb, video_sigma = backbone.generate_video(
        vid_input=obs_t,
        num_latent_conditional_frames=backbone.tokenizer.get_latent_num_frames(NUM_OBS_FRAMES),
        prompt_embedding=lang_t,
        num_sampling_step=NUM_SAMPLING_STEPS,
        return_context_at_step=stop_after_step,
        hidden_state_layer_idx=XATTN_LAYER_IDX,
        guidance=0.0,
        use_cuda_graphs=False,
    )

    # video_sigma: (B,) → (B, 1) for action head
    video_sigma = video_sigma.unsqueeze(-1)

    B, T, H, W, D = crossattn_emb.shape
    crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)  # (1, 19200, 2048)
    return crossattn_emb, video_sigma


# ── action conversion ─────────────────────────────────────────────────────────

def action_to_dict(action_vec: np.ndarray) -> dict:
    """Convert (5,) action vector to robot command dict."""
    return {
        "shoulder_pan.pos":  float(action_vec[0]),
        "shoulder_lift.pos": float(action_vec[1]),
        "elbow_flex.pos":    float(action_vec[2]),
        "wrist_flex.pos":    float(action_vec[3]),
        "wrist_roll.pos":    float(action_vec[4]),
        "gripper.pos":       GRIPPER_FIXED_POS,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relay-url", default="http://localhost:5000")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS_DEFAULT)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=STOP_AFTER_STEP_DEFAULT,
        help=(
            "Number of DiT sampling steps to run in the backbone's generate_video(). "
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--obs-freq", type=int, default=OBS_FREQ,
        help="Hz at which the 5 obs frames are collected (30 for new 30fps policy, 5 for old 5Hz policy)")
    args = parser.parse_args()

    # check relay
    relay = RelayClient(args.relay_url)
    print(f"Checking relay at {args.relay_url} ...")
    if not relay.health():
        raise RuntimeError("Relay not reachable. Is relay_server.py running? Is SSH tunnel open?")
    print("Relay reachable ✓\n")

    # load language embedding
    print("Loading language embedding...")
    lang_emb = np.load(LANG_EMB_PATH).astype(np.float16)
    print(f"  lang_emb shape: {lang_emb.shape} dtype: {lang_emb.dtype}")

    # load models
    backbone    = load_backbone()
    action_pipe = load_action_head()
    load_norm_stats(action_pipe)

    print(f"\nPartial denoising: {args.stop_after_step}/{NUM_SAMPLING_STEPS} steps "
          f"(more steps = closer to training sigma, slower inference)")

    # output dir
    pathlib.Path(ROLLOUT_DIR).mkdir(parents=True, exist_ok=True)

    # reset robot
    # relay.reset_robot()  # skipped — position robot manually before running
    relay.clear_stop()

    input("\nPlace object in start circle, then press Enter...")

    replay_images = []
    stopped = False

    for chunk_idx in range(args.max_chunks):
        # Grab 5 rapid consecutive frames — matches training precompute (--target-fps 30,
        # 5 consecutive frames ≈ 0.13s of history). Do NOT use a rolling 5Hz buffer here.
        obs_frames_list = []
        joints = None
        for i in range(NUM_OBS_FRAMES):
            t_frame = time.time()
            image, joints, stop = relay.get_observation()
            obs_frames_list.append(image)
            # sleep remainder of 1/obs_freq to match training frame spacing
            if i < NUM_OBS_FRAMES - 1:
                elapsed = time.time() - t_frame
                sleep_time = (1.0 / args.obs_freq) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        image = obs_frames_list[-1]   # most recent frame
        print(f"\n[chunk {chunk_idx+1}/{args.max_chunks}] joints: {joints[:5]}")
        if stop:
            print("Emergency stop. Ending.")
            stopped = True
            break

        replay_images.append(image)
        obs_frames = np.stack(obs_frames_list, axis=0)   # (5, H, W, 3)
        state = (
            torch.from_numpy(joints[:OBS_DIM].astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to("cuda", dtype=torch.bfloat16)
        )   # (1, 1, 5)

        # backbone: partial denoising → crossattn_emb + sigma at stop step
        print(f"  backbone ({args.stop_after_step} DiT passes)...", end=" ", flush=True)
        t0 = time.time()
        crossattn_emb, video_sigma = backbone_forward(
            backbone, obs_frames, lang_emb, stop_after_step=args.stop_after_step
        )
        backbone_ms = (time.time() - t0) * 1000
        print(f"{backbone_ms:.0f}ms  sigma={video_sigma[0, 0].item():.3f}")

        # action head: predicts 90-step chunk; we execute action_horizon of them
        with torch.no_grad():
            actions = action_pipe(
                state_B_HO_O=state,
                crossattn_emb=crossattn_emb,
                context_timesteps_B_1=video_sigma.to(dtype=torch.bfloat16),
            )   # (1, 90, 5)

        print(f"  crossattn_emb mean: {crossattn_emb.mean().item():.4f}  "
              f"std: {crossattn_emb.std().item():.4f}")
        print(f"  actions[0:3] = {actions[0, :3].float().cpu().numpy().round(3)}")

        # collect frames in background while executing actions
        frames_during_chunk: list[np.ndarray] = []
        stop_collecting = threading.Event()

        def collect_frames_bg():
            while not stop_collecting.is_set():
                try:
                    img, _, _ = relay.get_observation()
                    frames_during_chunk.append(img)
                except Exception:
                    pass
                time.sleep(1 / 5)  # 5Hz to reduce load

        collector = threading.Thread(target=collect_frames_bg)
        collector.start()

        # execute first `action_horizon` actions at ACTION_FREQ Hz
        t_chunk_start = time.time()
        for action_vec in actions[0][:args.action_horizon]:
            t_step = time.time()
            relay.send_action(action_to_dict(action_vec.float().cpu().numpy()))
            sleep_time = (1.0 / ACTION_FREQ) - (time.time() - t_step)
            if sleep_time > 0:
                time.sleep(sleep_time)

        stop_collecting.set()
        collector.join()

        # final observation for stop check
        image, joints, stop = relay.get_observation()
        replay_images.extend(frames_during_chunk)
        replay_images.append(image)

        chunk_ms = (time.time() - t_chunk_start) * 1000
        print(f"  chunk exec: {chunk_ms:.0f}ms  (frames collected: {len(frames_during_chunk)})")

        if stop:
            print("Emergency stop after chunk. Ending.")
            stopped = True
            break

    # save rollout video
    if replay_images:
        out_path = f"{ROLLOUT_DIR}/rollout.mp4"
        imageio.mimwrite(out_path, replay_images, fps=20)
        print(f"\nSaved rollout: {out_path}")

    print("Done." + (" (stopped early)" if stopped else ""))


if __name__ == "__main__":
    main()
