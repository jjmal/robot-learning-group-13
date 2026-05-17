"""
run_brev.py  —  run this on BREV
Loads the VAM model on GPU, drives the control loop,
and talks to relay_server.py on your laptop for robot I/O.

Usage:
    python run_brev.py \
        --experiment-name w2a_libero_goal_... \
        --video-model-path checkpoints/video_backbone/cosmos.pt \
        --action-model-path checkpoints/action_decoder/pushing.pt \
        --dataset-statistics-path checkpoints/stats.json \
        --task eval1 \
        --relay-url https://abc123.ngrok-free.app
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import random
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
from scipy.spatial.transform import Rotation

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

from prompts import TASK_PROMPTS

# ── constants ────────────────────────────────────────────────────────────────
MAX_STEPS_PER_ATTEMPT = 200
PROMPT_EMBEDDINGS_PATH = Path("../../model/checkpoints/prompt_embeddings.pt")
CAMERA_HEIGHT = 480
CAMERA_WIDTH  = 640
NUM_ATTEMPTS  = 5
GRIPPER_FIXED_POS = 0.0

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ─────────────────────────────────────────────────────────────────────────────
# Relay helpers  (replace all direct robot/camera calls)
# ─────────────────────────────────────────────────────────────────────────────

class RelayClient:
    """Thin wrapper around the relay_server.py HTTP API."""

    def __init__(self, base_url: str, timeout: float = 50.0):
        self.base  = base_url.rstrip("/")
        self.timeout = timeout

    # -- observation ----------------------------------------------------------

    def get_observation(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Returns:
            image:  uint8 (H, W, 3) RGB
            joints: float32 (6,) in degrees
            stop:   bool — True if ESC was pressed on laptop
        """
        r = requests.get(f"{self.base}/observation", timeout=self.timeout).json()

        img_bytes = base64.b64decode(r["image"])
        img_arr   = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)          # BGR
        image     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        joints_dict = r["joints"]
        joints = np.array(
            [joints_dict[f"{j}.pos"] for j in JOINT_NAMES],
            dtype=np.float32,
        )
        return image, joints, bool(r.get("stop", False))

    # -- action ---------------------------------------------------------------

    def send_action(self, action: dict) -> None:
        requests.post(f"{self.base}/action", json=action, timeout=self.timeout)

    # -- control --------------------------------------------------------------

    def reset_robot(self) -> None:
        requests.post(f"{self.base}/reset", json={"steps": 50, "hz": 25.0},
                      timeout=30.0)

    def clear_stop(self) -> None:
        requests.post(f"{self.base}/clear_stop", timeout=self.timeout)

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=2.0)
            return r.ok
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged from run.py
# ─────────────────────────────────────────────────────────────────────────────

def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    print("Loading config...")
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    print("Loading video backbone...")
    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
        use_text_encoder=False,
    )

    print("Loading action decoder...")
    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)

    print("Loading dataset statistics...")
    with dataset_statistics_path.open("rb") as f:
        stats = json.load(f)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=dtype,
    )

    print("Pipeline ready.")
    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda()


class VAMInference:
    """Unchanged from run.py — all GPU work stays here on Brev."""

    def __init__(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: pathlib.Path,
        task: str,
        img_horizon: int,
        lowdim_horizon: int,
        stop_video_denoising_step: int,
        num_execute_actions: int,
        rollout_dir: pathlib.Path,
    ):
        self.model = load_video2world2action_pipeline(
            experiment_name,
            video_model_path,
            action_model_path,
            dataset_statistics_path,
        )
        self._image_horizon   = img_horizon
        self._lowdim_horizon  = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps  = 35
        self.rollout_dir = rollout_dir

        # FAKE embeddings - only for testing pipeline, NOT for real inference
        self._prompt_embeddings = {
            task: torch.randn(1, 512, 1024, dtype=torch.bfloat16)
            for task in TASK_PROMPTS
        }
        print("WARNING: Using FAKE prompt embeddings - for testing only!")

        # if PROMPT_EMBEDDINGS_PATH.exists():
        #     self._prompt_embeddings = torch.load(PROMPT_EMBEDDINGS_PATH)
        #     print("Loaded saved T5 prompt embeddings.")
        # else:
        #     print("Pre-computing T5 embeddings for all tasks...")
        #     self._prompt_embeddings = {
        #         t: self.model.video2world_pipeline.encode_prompt(p).to(dtype=torch.bfloat16)
        #         for t, p in TASK_PROMPTS.items()
        #     }
        #     torch.save(self._prompt_embeddings, PROMPT_EMBEDDINGS_PATH)
        #     print("Saved T5 prompt embeddings.")

        self._current_task   = None
        self.prompt_embedding = None
        self.reset(task)

    def reset(self, task: str) -> None:
        if task != self._current_task:
            if task not in TASK_PROMPTS:
                raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_PROMPTS)}")
            self.prompt_embedding = self._prompt_embeddings[task]
            self._current_task = task

        self._image_history: deque[np.ndarray] = deque(
            maxlen=(self._image_horizon - 1) * 4 + 1)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0
        self._execute_horizon  = 0

    def step(self, image: np.ndarray, task: str, joint_angles: np.ndarray) -> dict:
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}.")
        if task != self._current_task:
            self.reset(task)

        self._add_image_to_history(self._process_image(image))
        self._add_lowdim_to_history(self._state_from_joints(joint_angles))

        if self.action_buffer is None:
            self._query_policy()

        current_action = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self._execute_horizon:
            self.action_buffer = None

        return self._convert_action(current_action, joint_angles)

    def _query_policy(self) -> None:
        images   = np.concatenate(list(self._image_history)[::4], axis=1)
        lowdims  = np.stack(list(self._lowdim_history), axis=0)
        input_vid    = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        with torch.no_grad():
            pred_actions = self.model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt="",
                prompt_embedding=self.prompt_embedding,
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                use_cuda_graphs=True,
            )

        self.action_buffer     = pred_actions[0].float().cpu().numpy()
        self._execute_horizon  = self.num_execute_actions
        self.action_buffer_idx = 0

    def _state_from_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        return joint_angles[:6].astype(np.float32)

    @staticmethod
    def _matrix_from_6d(orient6: np.ndarray) -> np.ndarray:
        r1 = orient6[:3]
        r2 = orient6[3:]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        return np.stack([r1_norm, r2_norm, r3], axis=0)

    def _convert_action(self, action: np.ndarray, current_joints: np.ndarray) -> dict:
        return {
            "shoulder_pan.pos":  float(action[0]),
            "shoulder_lift.pos": float(action[1]),
            "elbow_flex.pos":    float(action[2]),
            "wrist_flex.pos":    float(action[3]),
            "wrist_roll.pos":    float(action[4]),
            "gripper.pos":       GRIPPER_FIXED_POS,
        }

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self._image_history.append(image)
        while len(self._image_history) < self._image_history.maxlen:
            self._image_history.append(image.copy())

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())


# ─────────────────────────────────────────────────────────────────────────────
# Control loop  (robot I/O now goes through RelayClient)
# ─────────────────────────────────────────────────────────────────────────────

def run_attempt(
    relay: RelayClient,
    policy: VAMInference,
    max_steps: int,
    control_hz: float = 5.0,
) -> tuple[bool, list[np.ndarray]]:
    relay.clear_stop()
    replay_images: list[np.ndarray] = []
    success  = False
    step_dt  = 1.0 / control_hz

    for step_idx in range(max_steps):
        t_start = time.time()

        image, joint_angles, stop = relay.get_observation()
        if stop:
            print("Emergency stop received from laptop. Ending attempt.")
            break

        replay_images.append(image)
        action = policy.step(image, policy._current_task, joint_angles)
        relay.send_action(action)

        elapsed    = time.time() - t_start
        sleep_time = step_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

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


def run_pushing_eval(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: str,
    task: str,
    relay_url: str,                          # e.g. https://abc123.ngrok-free.app
    img_horizon: int = 5,
    lowdim_horizon: int = 1,
    stop_video_denoising_step: int = 10,
    num_execute_actions: int = 32,
    control_hz: float = 5.0,
    save_video: bool = True,
    seed: int = 42,
) -> None:
    """
    Example:
        python run_brev.py \\
            --experiment-name w2a_libero_goal_... \\
            --video-model-path checkpoints/video_backbone/cosmos.pt \\
            --action-model-path checkpoints/action_decoder/pushing.pt \\
            --dataset-statistics-path checkpoints/stats.json \\
            --task eval1 \\
            --relay-url https://abc123.ngrok-free.app
    """
    set_seed_everywhere(seed)

    relay = RelayClient(relay_url)
    print(f"Checking relay server at {relay_url} ...")
    if not relay.health():
        raise RuntimeError(
            f"Cannot reach relay server at {relay_url}. "
            "Is relay_server.py running? Is ngrok up?"
        )
    print("Relay server reachable ✓\n")

    run_label   = (
        f"{Path(action_model_path).stem}"
        f"_stopafter{stop_video_denoising_step}"
        f"_execute{num_execute_actions}"
    )
    rollout_dir = Path("./results") / run_label / task
    rollout_dir.mkdir(parents=True, exist_ok=True)

    policy = VAMInference(
        experiment_name=experiment_name,
        video_model_path=video_model_path,
        action_model_path=action_model_path,
        dataset_statistics_path=Path(dataset_statistics_path),
        task=task,
        img_horizon=img_horizon,
        lowdim_horizon=lowdim_horizon,
        stop_video_denoising_step=stop_video_denoising_step,
        num_execute_actions=num_execute_actions,
        rollout_dir=rollout_dir,
    )

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
        policy.reset(task)

        success, frames = run_attempt(
            relay=relay,
            policy=policy,
            max_steps=MAX_STEPS_PER_ATTEMPT,
            control_hz=control_hz,
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
    print(f"Result:        {'SUCCESS' if overall_success else 'FAILURE'}")
    print(f"Total attempts: {total_attempts}")
    print(f"Total successes: {total_successes}")
    print(f"Success rate:   {total_successes / total_attempts:.2%}")
    print(f"Videos saved to: {rollout_dir}")


if __name__ == "__main__":
    elapsed = (time.time() - SCRIPT_START_TIME) / 60
    print(f"Starting SO-101 Brev inference at {time.strftime('%H:%M:%S')}")
    print(f"Time since script start (imports): {elapsed:.1f} min")
    tyro.cli(run_pushing_eval)