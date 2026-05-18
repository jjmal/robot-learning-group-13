"""
run_brev.py  —  run this on BREV
Loads the new backbone + action head, drives the control loop
with action chunking, and talks to relay_server.py on your laptop.

Architecture:
  - Backbone runs once per chunk (~2.2s), predicts 90 actions
  - Actions execute on robot at ACTION_FREQ Hz (smooth 30 Hz motion)
  - Camera buffer of 61 frames (5 obs + 56 action frames) feeds backbone

Usage:
    python run_brev.py \
        --video-model-path ../../model/checkpoints/video_backbone/iter_000010000_fused.pt \
        --action-model-path ../../model/checkpoints/action_decoder/iter_000025000.pt \
        --norm-stats-path ../../model/checkpoints/stats.json \
        --lang-emb-path ../../model/checkpoints/language_embedding.npy \
        --task task1 \
        --relay-url http://localhost:5000
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable

SCRIPT_START_TIME = time.time()

import cv2
import imageio
import numpy as np
import requests
import torch
import tyro
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.config_world2action import SchedulerConfig, World2ActionPipelineConfig
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import LazyCall as L

# ── constants (must match mock_inference.py) ─────────────────────────────────
NUM_OBS_FRAMES        = 5
NUM_ACTION_FRAMES     = 56
TOTAL_FRAMES          = NUM_OBS_FRAMES + NUM_ACTION_FRAMES   # 61
XATTN_LAYER_IDX       = 20
CROSSATTN_POOL_FACTOR = 10    # 19200 → 1920 tokens
VIDEO_SIGMA           = 0.4
OBS_DIM               = 5
ACTION_HORIZON        = 90    # actions predicted per backbone call
ACTION_FREQ           = 30    # Hz at which actions are sent to robot
CAMERA_HEIGHT         = 480
CAMERA_WIDTH          = 640
NUM_ATTEMPTS          = 5
MAX_CHUNKS_PER_ATTEMPT = 20   # backbone calls per attempt (20 × 90/30s = 60s max)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_FIXED_POS = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Relay client
# ─────────────────────────────────────────────────────────────────────────────

class RelayClient:
    """Thin wrapper around the relay_server.py HTTP API."""

    def __init__(self, base_url: str, timeout: float = 50.0):
        self.base    = base_url.rstrip("/")
        self.timeout = timeout

    def get_observation(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns image (H,W,3) uint8 RGB, joints float32 (6,), stop bool."""
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
        requests.post(f"{self.base}/reset", json={"steps": 50, "hz": 25.0},
                      timeout=30.0)

    def clear_stop(self) -> None:
        requests.post(f"{self.base}/clear_stop", timeout=self.timeout)

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base}/health", timeout=2.0).ok
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (matches mock_inference.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(video_model_path: str) -> Video2WorldPipeline:
    print("Loading backbone...")
    config = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    config.guardrail_config.enabled = False
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=video_model_path,
        use_text_encoder=False,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()
    print("  Backbone loaded.")
    return pipe


def load_action_head(action_model_path: str) -> World2ActionPipeline:
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
        dit_path=action_model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pipe.requires_grad_(False)
    pipe.eval()
    print("  Action head loaded.")
    return pipe


def load_norm_stats(pipe: World2ActionPipeline, norm_stats_path: str) -> None:
    print("Loading norm stats...")
    with open(norm_stats_path) as f:
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


# ─────────────────────────────────────────────────────────────────────────────
# Backbone forward  (matches mock_inference.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def backbone_forward(
    backbone: Video2WorldPipeline,
    frames_uint8: np.ndarray,   # (61, H, W, 3)
    lang_emb: np.ndarray,       # (1, 512, 1024)
) -> torch.Tensor:
    frames_t = torch.from_numpy(frames_uint8).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
    obs_frames    = frames_t[:, :, :NUM_OBS_FRAMES]
    action_frames = frames_t[:, :, NUM_OBS_FRAMES:]
    lang_t = torch.from_numpy(lang_emb).to("cuda", dtype=torch.bfloat16)

    data_batch = {
        "obs/workspace_rgb":      obs_frames,
        "action/workspace_rgb":   action_frames,
        "obs/language_embedding": lang_t,
        "num_conditional_frames": backbone.tokenizer.get_latent_num_frames(NUM_OBS_FRAMES),
        "is_preprocessed":        True,
    }

    _, video_latent, condition = backbone.get_mimic_data_and_condition(data_batch)

    sigma_t      = torch.tensor([[VIDEO_SIGMA]], device="cuda", dtype=torch.float32)
    noise        = torch.randn_like(video_latent)
    noisy_latent = video_latent + noise * rearrange(sigma_t, "b t -> b 1 t 1 1")

    world_pred = backbone.denoise(
        noisy_latent,
        sigma_t,
        condition,
        use_cuda_graphs=False,
        return_only_hidden_states_up_to=XATTN_LAYER_IDX,
        return_decoded_video=False,
    )

    crossattn_emb = world_pred.hidden_states[XATTN_LAYER_IDX]   # (1, T, H, W, D)
    B, T, H, W, D = crossattn_emb.shape
    crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)
    crossattn_emb = crossattn_emb.reshape(
        B, CROSSATTN_POOL_FACTOR, T * H * W // CROSSATTN_POOL_FACTOR, D
    ).mean(dim=1)
    return crossattn_emb   # (1, 1920, 2048)


# ─────────────────────────────────────────────────────────────────────────────
# Inference class
# ─────────────────────────────────────────────────────────────────────────────

class VAMInference:
    """
    Maintains a rolling 61-frame buffer.
    Calls backbone once per chunk → executes 90 actions at 30 Hz.
    """

    def __init__(
        self,
        backbone: Video2WorldPipeline,
        action_pipe: World2ActionPipeline,
        lang_emb: np.ndarray,
    ):
        self.backbone    = backbone
        self.action_pipe = action_pipe
        self.lang_emb    = lang_emb
        self.video_sigma = torch.tensor([[VIDEO_SIGMA]], device="cuda", dtype=torch.bfloat16)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=TOTAL_FRAMES)

    def reset(self) -> None:
        self._frame_buffer.clear()

    def add_frame(self, image: np.ndarray) -> None:
        """Push a uint8 (H,W,3) frame; pads buffer with copies until full."""
        self._frame_buffer.append(image)
        while len(self._frame_buffer) < TOTAL_FRAMES:
            self._frame_buffer.appendleft(image.copy())

    def predict_chunk(self, joints: np.ndarray) -> np.ndarray:
        """Run backbone + action head. Returns float32 (90, 5)."""
        frames = np.stack(list(self._frame_buffer), axis=0)   # (61, H, W, 3)
        state  = (
            torch.from_numpy(joints[:OBS_DIM].astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to("cuda", dtype=torch.bfloat16)
        )   # (1, 1, 5)

        crossattn_emb = backbone_forward(self.backbone, frames, self.lang_emb)

        with torch.no_grad():
            actions = self.action_pipe(
                state_B_HO_O=state,
                crossattn_emb=crossattn_emb,
                context_timesteps_B_1=self.video_sigma,
            )   # (1, 90, 5)

        return actions[0].float().cpu().numpy()   # (90, 5)

    @staticmethod
    def action_to_dict(action_vec: np.ndarray) -> dict:
        return {
            "shoulder_pan.pos":  float(action_vec[0]),
            "shoulder_lift.pos": float(action_vec[1]),
            "elbow_flex.pos":    float(action_vec[2]),
            "wrist_flex.pos":    float(action_vec[3]),
            "wrist_roll.pos":    float(action_vec[4]),
            "gripper.pos":       GRIPPER_FIXED_POS,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Control loop
# ─────────────────────────────────────────────────────────────────────────────

def run_attempt(
    relay: RelayClient,
    policy: VAMInference,
    max_chunks: int,
) -> tuple[bool, list[np.ndarray]]:
    """
    One eval attempt using action chunking:
      1. Get frame from relay, add to buffer
      2. Run backbone (~2.2s) → 90 actions
      3. Execute all 90 actions at ACTION_FREQ Hz, collecting frames
      4. Repeat up to max_chunks times
    """
    relay.clear_stop()
    policy.reset()
    replay_images: list[np.ndarray] = []
    success   = False
    action_dt = 1.0 / ACTION_FREQ
    stopped   = False

    for chunk_idx in range(max_chunks):
        # get observation before backbone call
        image, joints, stop = relay.get_observation()
        if stop:
            print("Emergency stop. Ending attempt.")
            break

        policy.add_frame(image)
        replay_images.append(image)

        print(f"  [chunk {chunk_idx+1}/{max_chunks}] backbone...", end=" ", flush=True)
        t0      = time.time()
        actions = policy.predict_chunk(joints)   # (90, 5) — ~2.2s
        print(f"{(time.time()-t0)*1000:.0f}ms  → executing {ACTION_HORIZON} actions")

        # execute chunk at ACTION_FREQ Hz
        for action_vec in actions:
            t_step = time.time()

            image, joints, stop = relay.get_observation()
            if stop:
                print("Emergency stop mid-chunk. Ending attempt.")
                stopped = True
                break

            policy.add_frame(image)
            replay_images.append(image)
            relay.send_action(policy.action_to_dict(action_vec))

            sleep_time = action_dt - (time.time() - t_step)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if stopped:
            break

    return success, replay_images


def save_rollout_video(
    rollout_images: Iterable[np.ndarray],
    idx: int,
    success: bool,
    task: str,
    rollout_dir: Path,
) -> Path:
    rollout_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = rollout_dir / f"episode{idx}_{'success' if success else 'failure'}_{task}.mp4"
    writer = imageio.get_writer(mp4_path, fps=20)
    try:
        for img in rollout_images:
            writer.append_data(img)
    finally:
        writer.close()
    print(f"Saved rollout: {mp4_path}")
    return mp4_path


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pushing_eval(
    video_model_path: str,
    action_model_path: str,
    norm_stats_path: str,
    lang_emb_path: str,
    task: str,
    relay_url: str,
    max_chunks: int = MAX_CHUNKS_PER_ATTEMPT,
    save_video: bool = True,
    seed: int = 42,
) -> None:
    """
    Example:
        python run_brev.py \\
            --video-model-path ../../model/checkpoints/video_backbone/iter_000010000_fused.pt \\
            --action-model-path ../../model/checkpoints/action_decoder/iter_000025000.pt \\
            --norm-stats-path ../../model/checkpoints/stats.json \\
            --lang-emb-path ../../model/checkpoints/language_embedding.npy \\
            --task task1 \\
            --relay-url http://localhost:5000
    """
    set_seed_everywhere(seed)

    # check relay first — fast fail before loading heavy models
    relay = RelayClient(relay_url)
    print(f"Checking relay server at {relay_url} ...")
    if not relay.health():
        raise RuntimeError(
            f"Cannot reach relay server at {relay_url}. "
            "Is mock_relay_server.py or relay_server.py running? Is SSH tunnel open?"
        )
    print("Relay server reachable ✓\n")

    # load models
    print("Loading language embedding...")
    lang_emb = np.load(lang_emb_path).astype(np.float16)
    print(f"  lang_emb shape: {lang_emb.shape}")

    backbone    = load_backbone(video_model_path)
    action_pipe = load_action_head(action_model_path)
    load_norm_stats(action_pipe, norm_stats_path)

    policy = VAMInference(backbone, action_pipe, lang_emb)

    rollout_dir = Path("./results") / Path(action_model_path).stem / task
    rollout_dir.mkdir(parents=True, exist_ok=True)

    overall_success = False
    total_attempts  = 0
    total_successes = 0

    for attempt_idx in range(1, NUM_ATTEMPTS + 1):
        print(f"\nResetting robot to start position (attempt {attempt_idx})...")
        relay.reset_robot()

        input(
            f"\n[Attempt {attempt_idx}/{NUM_ATTEMPTS}] "
            "Place object in start circle, then press Enter here on Brev..."
        )
        total_attempts += 1

        success, frames = run_attempt(
            relay=relay,
            policy=policy,
            max_chunks=max_chunks,
        )

        if success:
            total_successes += 1
            overall_success  = True

        print(f"Attempt {attempt_idx}: {'SUCCESS ✓' if success else 'no success'}")
        print(f"Success rate: {total_successes}/{total_attempts} "
              f"= {total_successes / total_attempts:.2%}")

        if save_video:
            save_rollout_video(frames, attempt_idx, success, task, rollout_dir)

    print(f"\n=== Eval {task} complete ===")
    print(f"Result:          {'SUCCESS' if overall_success else 'FAILURE'}")
    print(f"Total attempts:  {total_attempts}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate:    {total_successes / total_attempts:.2%}")
    print(f"Videos saved to: {rollout_dir}")


if __name__ == "__main__":
    elapsed = (time.time() - SCRIPT_START_TIME) / 60
    print(f"Starting SO-101 Brev inference at {time.strftime('%H:%M:%S')}")
    print(f"Time since script start (imports): {elapsed:.1f} min")
    tyro.cli(run_pushing_eval)