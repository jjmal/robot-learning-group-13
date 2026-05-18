"""
Mock inference script — no robot required.
Loads backbone, action head, norm stats, and language embedding,
runs a full inference step with fake camera/state inputs,
and reports backbone forward pass timing.

Usage:
    cd /home/ubuntu/robot-learning-group-13
    source model/.venv/bin/activate
    python mock_inference.py
"""

import json
import pathlib
import sys
import time
import imageio


import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).parent / "model"))

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.config_world2action import SchedulerConfig, World2ActionPipelineConfig
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.data.action.types import NormalizationType
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.world2action_dit import World2ActionDIT
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import LazyCall as L

# ── paths ────────────────────────────────────────────────────────────────────
BACKBONE_PATH = (
    "../../model/checkpoints/video_backbone/iter_000010000_fused.pt"
)
ACTION_HEAD_PATH = (
    "../../model/checkpoints/action_decoder/iter_000025000.pt"
)
NORM_STATS_PATH = (
    "../../model/checkpoints/stats.json"
)
LANG_EMB_PATH = "../../model/checkpoints/language_embedding.npy"

# ── constants (must match training) ─────────────────────────────────────────
NUM_OBS_FRAMES = 5
NUM_ACTION_FRAMES = 56
TOTAL_FRAMES = NUM_OBS_FRAMES + NUM_ACTION_FRAMES  # 61
XATTN_LAYER_IDX = 20
CROSSATTN_POOL_FACTOR = 10   # 19200 → 1920 tokens
VIDEO_SIGMA = 0.4
OBS_DIM = 5
ACTION_HORIZON = 90
IMG_H, IMG_W = 480, 640

ACTION_FREQ = 30  # Hz, robot control frequency
CHUNK_SIZE  = 90  # steps predicted per backbone call (3 seconds)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

GRIPPER_FIXED_POS = 0.0

max_chunks = 10  # for testing, how many backbone+action_head iterations to run

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
        "action/actions": NormalizationType.VARIANCE,
        "obs/observation_state": NormalizationType.VARIANCE,
    }
    concat_groups = {
        "action/lowdim_concat": ["action/actions"],
        "obs/lowdim_concat": ["obs/observation_state"],
    }
    pipe.normalizer.build_from_stats(
        stats,
        normalization_types=normalization_types,
        concat_groups=concat_groups,
        dtype=torch.bfloat16,
        device="cuda",
    )
    print("  Norm stats loaded.")


@torch.no_grad()
def backbone_forward(
    backbone: Video2WorldPipeline,
    frames_uint8: np.ndarray,   # (61, H, W, 3)
    lang_emb: np.ndarray,       # (1, 512, 1024)
) -> torch.Tensor:
    frames_t = torch.from_numpy(frames_uint8).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
    obs_frames = frames_t[:, :, :NUM_OBS_FRAMES]
    action_frames = frames_t[:, :, NUM_OBS_FRAMES:]
    lang_t = torch.from_numpy(lang_emb).to("cuda", dtype=torch.bfloat16)

    data_batch = {
        "obs/workspace_rgb": obs_frames,
        "action/workspace_rgb": action_frames,
        "obs/language_embedding": lang_t,
        "num_conditional_frames": backbone.tokenizer.get_latent_num_frames(NUM_OBS_FRAMES),
        "is_preprocessed": True,
    }

    _, video_latent, condition = backbone.get_mimic_data_and_condition(data_batch)

    sigma_t = torch.tensor([[VIDEO_SIGMA]], device="cuda", dtype=torch.float32)
    noise = torch.randn_like(video_latent)
    noisy_latent = video_latent + noise * rearrange(sigma_t, "b t -> b 1 t 1 1")

    world_pred = backbone.denoise(
        noisy_latent,
        sigma_t,
        condition,
        use_cuda_graphs=False,
        return_only_hidden_states_up_to=XATTN_LAYER_IDX,
        return_decoded_video=True,
    )
    
    # Save the world prediction
    decoded_video = world_pred.decoded_video  # (B, C, T, H, W), [-1, 1]
    decoded_np = ((decoded_video[0].permute(1,2,3,0).cpu().float().numpy() + 1) * 127.5).clip(0,255).astype(np.uint8)
    # decoded_np is (T, H, W, 3) — save as mp4
    imageio.mimwrite(f"video_rollouts/imagined_video_q{self._query_count:04d}.mp4", decoded_np, fps=5)


    crossattn_emb = world_pred.hidden_states[XATTN_LAYER_IDX]  # (1, T, H, W, D)
    B, T, H, W, D = crossattn_emb.shape
    crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)
    # 10× pooling
    crossattn_emb = crossattn_emb.reshape(
        B, CROSSATTN_POOL_FACTOR, T * H * W // CROSSATTN_POOL_FACTOR, D
    ).mean(dim=1)

    
    return crossattn_emb  # (1, 1920, 2048)


def main():
    # ── load language embedding ───────────────────────────────────────────
    print("Loading language embedding...")
    lang_emb = np.load(LANG_EMB_PATH)   # (1, 512, 1024)
    print(f"  lang_emb shape: {lang_emb.shape} dtype: {lang_emb.dtype}")

    # ── load models ───────────────────────────────────────────────────────
    backbone = load_backbone()
    action_pipe = load_action_head()
    load_norm_stats(action_pipe)
    
    # ── main loop ───────────────────────────────────────────────────────
    
    for chunk_idx in range(max_chunks):
        # get observation before backbone call
        frames, state, stop = relay.get_observation()

        if stop:
            print("Emergency stop. Ending attempt.")
            break
        crossattn_emb = backbone_forward(backbone, frames, lang_emb)  # (1, 1920, 2048)

        actions = action_pipe(
            state_B_HO_O=state,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma,
        )  # (1, 90, 5)

        # 4. execute all 90 actions on the robot at 30 Hz
        for action in actions[0]:             # action: (5,) tensor
            relay.send_action(action.cpu().numpy())
            time.sleep(1 / ACTION_FREQ)

        if stopped:
            break

    return success, replay_images
    

# ── Real robot control loop (action chunking) ─────────────────────────────
#
# DO NOT re-run the backbone every step — it takes ~seconds.
# Instead: run backbone once, execute all 90 predicted actions, then repeat.
#
# ACTION_FREQ = 30  # Hz, robot control frequency
# CHUNK_SIZE  = 90  # steps predicted per backbone call (3 seconds)
#
# lang_emb = np.load(LANG_EMB_PATH)
# backbone, action_pipe = load_backbone(), load_action_head()
# load_norm_stats(action_pipe)
# video_sigma = torch.tensor([[VIDEO_SIGMA]], device="cuda", dtype=torch.bfloat16)
#
# while task_not_done:
#     # 1. collect last 61 frames from camera buffer (5 obs + 56 action frames)
#     frames = get_camera_frames()          # (61, H, W, 3) uint8
#     state  = get_robot_state()            # (1, 1, 5) torch bfloat16 on cuda
#
#     # 2. backbone: slow, run once per chunk
#     crossattn_emb = backbone_forward(backbone, frames, lang_emb)  # (1, 1920, 2048)
#
#     # 3. action head: fast, predicts full 90-step chunk at once
#     actions = action_pipe(
#         state_B_HO_O=state,
#         crossattn_emb=crossattn_emb,
#         context_timesteps_B_1=video_sigma,
#     )  # (1, 90, 5)
#
#     # 4. execute all 90 actions on the robot at 30 Hz
#     for action in actions[0]:             # action: (5,) tensor
#         robot.send_joint_command(action.cpu().numpy())
#         time.sleep(1 / ACTION_FREQ)
#
#     # then loop: get new frames, run backbone again


if __name__ == "__main__":
    main()
