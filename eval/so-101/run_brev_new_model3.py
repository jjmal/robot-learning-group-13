"""
mock_inference.py — runs on BREV, connects to relay_server.py on laptop.
Loads backbone, action head, norm stats, and language embedding,
runs the full control loop with real camera/robot via relay.

Usage:
    cd /home/ubuntu/robot-learning-group-13/eval/so-101
    conda activate mimicvideo
    python mock_inference.py --relay-url http://localhost:5000
"""

import argparse
import base64
import json
import pathlib
import sys
import time

import cv2
import imageio
import numpy as np
import requests
import torch
from einops import rearrange
import threading

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
BACKBONE_PATH    = "../../model/checkpoints/video_backbone/iter_000010000_fused.pt"
ACTION_HEAD_PATH = "../../model/checkpoints/action_decoder/iter_000025000.pt"
NORM_STATS_PATH  = "../../model/checkpoints/stats.json"
LANG_EMB_PATH    = "../../model/checkpoints/language_embedding.npy"
ROLLOUT_DIR      = "./video_rollouts"

# ── constants (must match training) ───────────────────────────────────────────
NUM_OBS_FRAMES        = 5
NUM_ACTION_FRAMES     = 56
TOTAL_FRAMES          = NUM_OBS_FRAMES + NUM_ACTION_FRAMES   # 61
XATTN_LAYER_IDX       = 20
CROSSATTN_POOL_FACTOR = 10    # 19200 → 1920 tokens
VIDEO_SIGMA           = 0.4
OBS_DIM               = 5
ACTION_HORIZON        = 15  # 90
IMG_H, IMG_W          = 480, 640
ACTION_FREQ           = 30    # Hz
MAX_CHUNKS            = 10    # backbone calls per attempt

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
        requests.post(f"{self.base}/reset", json={"steps": 50, "hz": 25.0}, timeout=30.0)

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


# ── backbone forward ──────────────────────────────────────────────────────────

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
        use_cuda_graphs=True, # False,
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


# ── action conversion ─────────────────────────────────────────────────────────

def action_to_dict(action_vec: np.ndarray) -> dict:
    """Convert (5,) action vector to robot command dict."""
    # print(f"  action values: {action_vec}")  # add this
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
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
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

    video_sigma = torch.tensor([[VIDEO_SIGMA]], device="cuda", dtype=torch.bfloat16)

    # rolling frame buffer
    from collections import deque
    frame_buffer: deque[np.ndarray] = deque(maxlen=TOTAL_FRAMES)

    # output dir
    import pathlib
    pathlib.Path(ROLLOUT_DIR).mkdir(parents=True, exist_ok=True)

    # reset robot
    print("\nResetting robot to start position...")
    relay.reset_robot()
    relay.clear_stop()

    input("\nPlace object in start circle, then press Enter...")

    replay_images = []
    stopped = False

    for chunk_idx in range(args.max_chunks):
        # get observation
        image, joints, stop = relay.get_observation()
        print(f"  current joints: {joints[:5]}")
        if stop:
            print("Emergency stop. Ending.")
            stopped = True
            break

        # update frame buffer
        frame_buffer.append(image)
        while len(frame_buffer) < TOTAL_FRAMES:
            frame_buffer.appendleft(image.copy())
        replay_images.append(image)

        frames = np.stack(list(frame_buffer), axis=0)   # (61, H, W, 3)
        print(f"  frame mean: {frames.mean():.1f}  std: {frames.std():.1f}")
        print(f"  unique frames in buffer: {len(set(frames[i].tobytes() for i in range(len(frames))))}/{TOTAL_FRAMES}")
        state  = (
            torch.from_numpy(joints[:OBS_DIM].astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to("cuda", dtype=torch.bfloat16)
        )   # (1, 1, 5)

        print(f"  [chunk {chunk_idx+1}/{args.max_chunks}] backbone...", end=" ", flush=True)
        t0 = time.time()
        crossattn_emb = backbone_forward(backbone, frames, lang_emb)
        print(f"{(time.time()-t0)*1000:.0f}ms → executing {ACTION_HORIZON} actions")

        

        with torch.no_grad():
            actions = action_pipe(
                state_B_HO_O=state,
                crossattn_emb=crossattn_emb,
                context_timesteps_B_1=video_sigma,
            )   # (1, 90, 5)

        print(f"  crossattn_emb mean: {crossattn_emb.mean().item():.4f}  std: {crossattn_emb.std().item():.4f}")
        print(f"  state fed to model: {state[0,0].cpu().float().numpy()}")
        print(f"  actions mean: {actions[0].float().mean(dim=0).cpu().numpy()}")
        print(f"  actions std:  {actions[0].float().std(dim=0).cpu().numpy()}")

        # execute chunk at ACTION_FREQ Hz
        # t_chunk_start = time.time()
        # for action_vec in actions[0]:
        #     t_step = time.time()

        #     # TODO: re-rewrite this; just to move faster
        #     # image, joints, stop = relay.get_observation()
        #     # if stop:
        #     #     print("Emergency stop mid-chunk.")
        #     #     stopped = True
        #     #     break

        #     # frame_buffer.append(image)
        #     # replay_images.append(image)
        #     relay.send_action(action_to_dict(action_vec.float().cpu().numpy()))

        #     sleep_time = (1.0 / ACTION_FREQ) - (time.time() - t_step)
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)

        # image, joints, stop = relay.get_observation()

        # collect frames in background while executing actions
        frames_during_chunk = []
        stop_collecting = threading.Event()

        def collect_frames_bg():
            while not stop_collecting.is_set():
                try:
                    img, _, _ = relay.get_observation()
                    frames_during_chunk.append(img)
                except:
                    pass
                # time.sleep(1/10)  # collect at 10Hz
                time.sleep(1/5)  # collect at 5Hz to reduce load

        collector = threading.Thread(target=collect_frames_bg)
        collector.start()

        # execute actions at ACTION_FREQ Hz
        t_chunk_start = time.time()
        for action_vec in actions[0][:ACTION_HORIZON]: # CHANGE HERE ACTION_HORIZON
            t_step = time.time()
            relay.send_action(action_to_dict(action_vec.float().cpu().numpy()))
            sleep_time = (1.0 / ACTION_FREQ) - (time.time() - t_step)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # stop frame collection
        stop_collecting.set()
        collector.join()

        # update frame buffer with all collected frames
        for f in frames_during_chunk:
            frame_buffer.append(f)

        # get final observation for joints/stop check
        image, joints, stop = relay.get_observation()
        if frames_during_chunk:
            frame_buffer.append(image)

        print(f"Chunk execution time: {time.time()-t_chunk_start:.2f}s  frames collected: {len(frames_during_chunk)}")
        

        print(f"Chunk execution time: {time.time()-t_chunk_start:.2f}s  (expected 3.0s)")

        if stopped:
            break

    # save rollout video
    if replay_images:
        out_path = f"{ROLLOUT_DIR}/rollout.mp4"
        imageio.mimwrite(out_path, replay_images, fps=20)
        print(f"\nSaved rollout: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()