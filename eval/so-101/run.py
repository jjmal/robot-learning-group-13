from __future__ import annotations

from datetime import datetime
import json
import os
import pathlib
import random
import time
from collections import deque
from pathlib import Path
from typing import Iterable

SCRIPT_START_TIME = time.time()  # for measuring total runtime 

import imageio
import numpy as np
import torch
import tyro
from einops import rearrange
from scipy.spatial.transform import Rotation
import cv2
import threading
from pynput import keyboard

from model.cosmos_predict2.configs.config import make_config
from model.cosmos_predict2.data.action.utils import extract_normalization_types
from model.cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from model.cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from model.cosmos_predict2.pipelines.world2action import World2ActionPipeline
from model.imaginaire.lazy_config import instantiate
from model.imaginaire.utils.config_helper import override

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsEEToJoints

from prompts import TASK_PROMPTS

# TODO: adjust constants as needed
MAX_STEPS_PER_ATTEMPT = 200

PROMPT_EMBEDDINGS_PATH = Path("../../checkpoints/prompt_embeddings.pt")

CAMERA_HEIGHT = 480 # actual camera resolution
CAMERA_WIDTH = 640
NUM_ATTEMPTS = 5 # num attempts allowed per task; only the best counts for eval

URDF_PATH = os.path.join(os.path.dirname(__file__), "SO101", "so101_new_calib.urdf")

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_FIXED_POS = 0.0  # closed, range [0, 100]

TARGET_CENTER_PX = (320, 240)   # TODO: manually determine from camera view
TARGET_RADIUS_PX = 50

OBJECT_HSV_LOWER = np.array([0, 0, 0])
OBJECT_HSV_UPPER = np.array([255, 255, 255])

START_POSITION = {
    "shoulder_pan.pos":  0.0,
    "shoulder_lift.pos": -90.0,  # TODO: tune these starting joint angles
    "elbow_flex.pos":    90.0,
    "wrist_flex.pos":    0.0,
    "wrist_roll.pos":    0.0,
    "gripper.pos":       GRIPPER_FIXED_POS,
}


def set_seed_everywhere(seed: int) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Video2World2ActionPipeline:
    """Instantiate the video-to-world-to-action pipeline and load normalizer statistics."""
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
        use_text_encoder=False, # TODO: let it have the 45GB text encoder as well 
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
    with dataset_statistics_path.open("rb") as stats_file:
        stats = json.load(stats_file)
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
    """Maintains temporal context and queries the VAM policy for the physical SO-101."""

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

        self._image_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps = 35
        self.rollout_dir = rollout_dir

        # TODO:
        # Forward kinematics: joint angles -> EEF pose
        # they train on EEF pose, but the robot gives joint angles, so we need FK to convert
        # also must match the dataset rep
        self.kinematics = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="gripper_frame_link",
            joint_names=JOINT_NAMES,
        )

        # TODO: for future, we will precompute the embeddings, save them and load them here
        if PROMPT_EMBEDDINGS_PATH.exists():
            self._prompt_embeddings = torch.load(PROMPT_EMBEDDINGS_PATH)
            print("Loaded saved T5 prompt embeddings.")
        else:
            print("Pre-computing T5 embeddings for all tasks...")
            self._prompt_embeddings = {
                task: self.model.video2world_pipeline.encode_prompt(prompt).to(dtype=torch.bfloat16)
                for task, prompt in TASK_PROMPTS.items()
            }
            torch.save(self._prompt_embeddings, PROMPT_EMBEDDINGS_PATH)
            print("Saved T5 prompt embeddings.")
        
        self._current_task = None
        self.prompt_embedding = None

        self.reset(task)

    def reset(self, task: str) -> None:
        """Reset internal state for a new task/episode."""
        
        if task != self._current_task:
            if task not in TASK_PROMPTS:
                raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_PROMPTS)}")
            self.prompt_embedding = self._prompt_embeddings[task]
            self._current_task = task
        
        self._image_history: deque[np.ndarray] = deque(maxlen=(self._image_horizon - 1) * 4 + 1)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0
        self._execute_horizon = 0

    def step(self, image: np.ndarray, task: str, joint_angles: np.ndarray) -> np.ndarray:
        """
        Return next robot action given current camera frame and joint angles.

        Args:
            image:        uint8 numpy (H, W, 3)
            joint_angles: float32 numpy (6,) in degrees — from LeRobot

        Returns:
            action: float32 numpy (7,) — [delta_xyz(3) + rotvec(3) + gripper(1)]
        """
        if image.dtype != np.uint8:
            raise ValueError(f"Expected image dtype uint8, got {image.dtype}.")

        if task != self._current_task:
            self.reset(task)

        processed_image = self._process_image(image)
        self._add_image_to_history(processed_image)

        state_vec = self._state_from_joints(joint_angles)
        self._add_lowdim_to_history(state_vec)

        if self.action_buffer is None:
            self._query_policy()

        current_action = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self._execute_horizon:
            self.action_buffer = None

        return self._convert_action(current_action, joint_angles)

    def _query_policy(self) -> None:
        """Query the model and cache the planned action sequence."""
        images = np.concatenate(list(self._image_history)[::4], axis=1)  # 20fps -> 5fps
        lowdims = np.stack(list(self._lowdim_history), axis=0)

        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        with torch.no_grad():
            pred_actions = self.model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt="",                              # workaround to embeddings; will use directly the precomputed embeddings
                prompt_embedding=self.prompt_embedding,
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                use_cuda_graphs=True,
            )

        self.action_buffer = pred_actions[0].float().cpu().numpy()
        self._execute_horizon = self.num_execute_actions
        self.action_buffer_idx = 0

    # State: joint angles -> 10-dim vector [xyz(3) + rot6d(6) + gripper(1)]
    def _state_from_joints(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Convert 6 joint angles (degrees, from LeRobot) to the 10-dim state vector
        the model expects: [eef_xyz(3) + rot6d(6) + gripper(1)].
        """
        # # degrees -> radians, build dict for FK
        # joint_dict = {
        #     name: np.deg2rad(val)
        #     for name, val in zip(JOINT_NAMES, joint_angles)
        # }

        # # Forward kinematics -> 4x4 EEF transform
        # fk_poses = self.kinematics.forward_kinematics(joint_dict)
        # eef_matrix = fk_poses["gripper_frame_link"]  # (4, 4)

        # eef_pos = eef_matrix[:3, 3]                  # (3,)
        # rot_6d = eef_matrix[:3, :3][:2].reshape(6,)  # first two rows of R -> (6,)

        # # Typical SO-101: ~0 (closed) to ~100 degrees (open) -> map to [-1, 1]
        # gripper_raw = joint_angles[-1]
        # gripper = np.clip(gripper_raw / 50.0 - 1.0, -1.0, 1.0)

        # return np.concatenate([eef_pos, rot_6d, [gripper]]).astype(np.float32)

        eef_matrix = self.kinematics.forward_kinematics(joint_angles)  # use directly
        eef_pos = eef_matrix[:3, 3]
        rot_6d = eef_matrix[:3, :3][:2].reshape(6,)
        # gripper = np.clip(joint_angles[-1] / 50.0 - 1.0, -1.0, 1.0) # [0,100] -> [-1,1]; remove gripper anyways
        # return np.concatenate([eef_pos, rot_6d, [gripper]]).astype(np.float32)
        return np.concatenate([eef_pos, rot_6d]).astype(np.float32)

    # Action conversion - same as before
    @staticmethod
    def _matrix_from_6d(orient6: np.ndarray) -> np.ndarray:
        """Reconstruct rotation matrix from 6D representation (Gram-Schmidt)."""
        r1 = orient6[:3]
        r2 = orient6[3:]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        return np.stack([r1_norm, r2_norm, r3], axis=0)

    def _convert_action(self, action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """
        Convert model output [xyz(3) + rot6d(6) + gripper(1)] to 
        joint position dict accepted by LeRobot SO-101.

        Prev: to [delta_xyz(3) + rotvec(3) + gripper(1)].
        """
        delta_pos = action[:3]
        rot_matrix = self._matrix_from_6d(action[3:9])

        # FK to get current EEF pose
        eef_current = self.kinematics.forward_kinematics(current_joints)  # 4x4

        # target EEF = current EEF + delta_pos
        eef_target = eef_current.copy()
        eef_target[:3, 3] += delta_pos
        eef_target[:3, :3] = rot_matrix
        
        # IK: target EEF -> joint positions
        joint_positions = self.kinematics.inverse_kinematics(
            current_joint_pos=current_joints,  # degrees
            desired_ee_pose=eef_target,        # 4x4
        )  # returns degrees
        
        # remove gripper
        # gripper_pos = float(np.clip((np.sign(action[9]) + 1) * 50.0, 0.0, 100.0))
        gripper_pos = float(current_joints[-1])  # keep gripper as it is

        return {
            "shoulder_pan.pos":  float(joint_positions[0]),
            "shoulder_lift.pos": float(joint_positions[1]),
            "elbow_flex.pos":    float(joint_positions[2]),
            "wrist_flex.pos":    float(joint_positions[3]),
            "wrist_roll.pos":    float(joint_positions[4]),
            # "gripper.pos":       gripper_pos,
            "gripper.pos":       GRIPPER_FIXED_POS,
        }
    
    
    # image processing - same as before
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self._image_history.append(image)
        while len(self._image_history) < self._image_history.maxlen: # for beginning, to fill history with copies of first frame
            self._image_history.append(image.copy())

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon: # same but for states
            self._lowdim_history.append(lowdim.copy())


# Robot helpers
def connect_robot(port: str) -> SO101Follower:
    config = SO101FollowerConfig(port=port, id="so101_pusher")
    robot = SO101Follower(config)
    robot.connect()
    print(f"Robot connected on {port}")
    return robot


def get_robot_image(robot: SO101Follower) -> np.ndarray:
    """
    Read one RGB frame from the robot camera.
    """
    obs = robot.get_observation()
    # TODO: adjust key if your camera has a different name
    image = obs["front"]
    if image.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise ValueError(f"Unexpected image shape {image.shape}.")
    return image.astype(np.uint8)


def get_joint_angles(robot: SO101Follower) -> np.ndarray:
    """
    Read current joint angles in degrees from the robot.
    Returns float32 array of shape (6,).
    """
    obs = robot.get_observation()
    # TODO: adjust key if needed
    return np.array([
        obs["shoulder_pan.pos"],
        obs["shoulder_lift.pos"],
        obs["elbow_flex.pos"],
        obs["wrist_flex.pos"],
        obs["wrist_roll.pos"],
        obs["gripper.pos"],
    ], dtype=np.float32)


def send_action(robot: SO101Follower, action: np.ndarray) -> None:
    """
    Send joint position commands to the robot.
    
    Args:
        action: dict with keys like {"shoulder_pan.pos": float, ..., "gripper.pos": float}
                values in degrees, range [-180, 180] for joints, [0, 100] for gripper.
                Returned by _convert_action().
    """
    robot.send_action(action)


def reset_robot_to_start(robot: SO101Follower) -> None:
    """Move robot to start position before each attempt."""
    print("Resetting robot to start position...")
    robot.send_action(START_POSITION)
    time.sleep(2.0)  # wait for it to actually move


def object_in_target_circle(image: np.ndarray) -> bool:
    """
    Detect if the pushed object has reached the target circle.

    Prerequisites before using:
        1. Mount camera in fixed position
        2. Run debug_hsv.py to find OBJECT_HSV_LOWER / OBJECT_HSV_UPPER
        3. Run debug_target.py to find TARGET_CENTER_PX / TARGET_RADIUS_PX
    """
    # TODO: improve target circle detection logic
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, OBJECT_HSV_LOWER, OBJECT_HSV_UPPER)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    M = cv2.moments(mask)
    if M["m00"] == 0:
        return False

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    dist = np.sqrt((cx - TARGET_CENTER_PX[0])**2 + (cy - TARGET_CENTER_PX[1])**2)
    return bool(dist < TARGET_RADIUS_PX)

    
def run_attempt(
    robot: SO101Follower,
    policy: VAMInference,
    max_steps: int,
    control_hz: float = 10.0,
) -> tuple[bool, list[np.ndarray]]:
    """
    Execute one pushing attempt. Returns (success, replay_frames).
    """

    # emergency stop listener
    stop_event = threading.Event()
    def on_press(key):
        if key == keyboard.Key.esc:  # ESC pentru emergency stop
            stop_event.set()
            return False
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    replay_images: list[np.ndarray] = []
    success = False
    step_dt = 1.0 / control_hz

    for step_idx in range(max_steps):
        if stop_event.is_set():
            print("Emergency stop triggered! Ending attempt.")
            break

        t_start = time.time()

        image = get_robot_image(robot)
        joint_angles = get_joint_angles(robot)
        replay_images.append(image)

        action = policy.step(image, joint_angles)
        send_action(robot, action)

        if object_in_target_circle(image):
            success = True
            print(f"Goal reached at step {step_idx}!")
            break

        elapsed = time.time() - t_start
        sleep_time = step_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    listener.stop()

    return success, replay_images


def save_rollout_video(
    rollout_images: Iterable[np.ndarray], # frames
    idx: int,
    success: bool,
    task: str,
    rollout_dir: Path,
) -> Path:
    """Save an MP4 replay of the episode."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = rollout_dir / f"episode{idx}_{'success' if success else 'failure'}_{task}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    try:
        for img in rollout_images:
            video_writer.append_data(img)
    finally:
        video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def run_pushing_eval(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: str,
    task: str,
    robot_port: str = "/dev/ttyACM0", # "/dev/ttyUSB0",
    img_horizon: int = 5,
    lowdim_horizon: int = 1,
    stop_video_denoising_step: int = 10,
    num_execute_actions: int = 8,
    control_hz: float = 10.0,
    save_video: bool = True,
    seed: int = 42,
) -> None:
    """
    Run up to NUM_ATTEMPTS pushing attempts. Only the best counts (per eval rules).

    Example:
        python run_so101.py \\
            --experiment-name w2a_libero_goal_... \\
            --video-model-path checkpoints/video_backbone/cosmos.pt \\
            --action-model-path checkpoints/action_decoder/pushing.pt \\
            --dataset-statistics-path checkpoints/stats.json \\
            --task eval1 \\
            --robot-port /dev/ttyUSB0
    """
    print('yay!', time.time())
    set_seed_everywhere(seed)
    run_label = (
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

    robot = connect_robot(robot_port)
    overall_success = False
    total_attempts = 0
    total_successes = 0

    try:
        for attempt_idx in range(1, NUM_ATTEMPTS + 1):
            # for now, manually reset the scene before each attempt; 
            # automated reset logic to a fixed initial position, although the object still needs manual replacement
            # TODO: tune angles and use the automatic reset
            # reset_robot_to_start(robot)
            input(
                f"\n[Attempt {attempt_idx}/{NUM_ATTEMPTS}] "
                "Place object in start circle, then press Enter..."
            )
            total_attempts += 1

            policy.reset(task)  # reset policy state for new attempt

            success, frames = run_attempt(
                robot=robot,
                policy=policy,
                max_steps=MAX_STEPS_PER_ATTEMPT,
                control_hz=control_hz,
            )

            if success:
                total_successes += 1
                overall_success = True

            print(f"Attempt {attempt_idx}: {'SUCCESS ✓' if success else 'no success'}")
            print(f"Current success rate: {total_successes}/{total_attempts} = {total_successes / total_attempts:.2%}")

            if save_video:
                save_rollout_video(frames, attempt_idx, success, task, rollout_dir)

        print(f"\n=== Eval {task} complete ===")
        print(f"Result: {'SUCCESS' if overall_success else 'FAILURE'}")
        print(f"Videos saved to: {rollout_dir}")
        print(f"Total attempts: {total_attempts}")
        print(f"Total successes: {total_successes}")
        print(f"Success rate: {total_successes / total_attempts:.2%}")

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    elapsed = (time.time() - SCRIPT_START_TIME) / 60
    print(f'Starting SO-101 pushing evaluation at {time.strftime("%H:%M:%S")}')
    print(f'Time since script start (imports etc.): {elapsed:.1f} min')
    tyro.cli(run_pushing_eval)