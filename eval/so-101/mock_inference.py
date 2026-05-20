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

import numpy as np
import torch

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
    "/ephemeral/training_output/posttraining/video2world"
    "/v2w_rl_group13_lora_rank256_lr1.778e-04_bsz32"
    "/checkpoints/model/iter_000010000_fused.pt"
)
ACTION_HEAD_PATH = (
    "/ephemeral/training_output/vam/rl_group13/w2a_rl_group13"
    "/checkpoints/model/iter_000025000.pt"
)
NORM_STATS_PATH = (
    "/ephemeral/rl-group13-processed/.statistics_cache"
    "/800a79c08667b0f5b41714c2865ff39f86978995df137124ffc3d3cb7919c328"
)
LANG_EMB_PATH = "/ephemeral/language_embedding.npy"

# ── constants (must match training) ─────────────────────────────────────────
NUM_OBS_FRAMES = 5
NUM_ACTION_FRAMES = 56
TOTAL_FRAMES = NUM_OBS_FRAMES + NUM_ACTION_FRAMES  # 61
XATTN_LAYER_IDX = 20
VIDEO_SIGMA = 0.4            # training sigma (reference only)
# Partial denoising: run 24 of 35 scheduler steps from pure noise.
# scheduler.sigmas[24] ≈ 0.377 — closest step to training sigma 0.4.
# This matches the paper (imagined future via partial denoising) instead of
# the broken past-frames approach used previously.
NUM_SAMPLING_STEPS = 35
STOP_AFTER_STEP = 0          # τv=1: pure noise future, single DiT pass (Algorithm 1)
OBS_DIM = 5
ACTION_HORIZON = 90
IMG_H, IMG_W = 480, 640


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
    obs_frames_uint8: np.ndarray,   # (5, H, W, 3) — obs frames only
    lang_emb: np.ndarray,           # (1, 512, 1024)
) -> tuple[torch.Tensor, torch.Tensor]:
    # (5, H, W, 3) uint8 → (1, 3, 5, H, W)
    obs_t = torch.from_numpy(obs_frames_uint8).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
    lang_t = torch.from_numpy(lang_emb).to("cuda", dtype=torch.bfloat16)

    # Partially denoise from pure Gaussian noise to sigma ≈ 0.4 (STOP_AFTER_STEP=24).
    # Future frame slots are pure noise that gets partially denoised — the model
    # imagines a plausible future rather than conditioning on stale past frames.
    crossattn_emb, video_sigma = backbone.generate_video(
        vid_input=obs_t,
        num_latent_conditional_frames=backbone.tokenizer.get_latent_num_frames(NUM_OBS_FRAMES),
        prompt_embedding=lang_t,
        num_sampling_step=NUM_SAMPLING_STEPS,
        return_context_at_step=STOP_AFTER_STEP,
        hidden_state_layer_idx=XATTN_LAYER_IDX,
        guidance=0.0,
        use_cuda_graphs=False,
    )

    # video_sigma: (B,) → (B, 1) for action head
    video_sigma = video_sigma.unsqueeze(-1)

    B, T, H, W, D = crossattn_emb.shape
    crossattn_emb = crossattn_emb.reshape(B, T * H * W, D)  # (1, 19200, 2048)
    return crossattn_emb, video_sigma


def main():
    # ── load language embedding ───────────────────────────────────────────
    print("Loading language embedding...")
    lang_emb = np.load(LANG_EMB_PATH)   # (1, 512, 1024)
    print(f"  lang_emb shape: {lang_emb.shape} dtype: {lang_emb.dtype}")

    # ── load models ───────────────────────────────────────────────────────
    backbone = load_backbone()
    action_pipe = load_action_head()
    load_norm_stats(action_pipe)

    # ── mock inputs ───────────────────────────────────────────────────────
    print("\nCreating mock inputs...")
    fake_obs_frames = np.random.randint(0, 255, (NUM_OBS_FRAMES, IMG_H, IMG_W, 3), dtype=np.uint8)
    fake_state = torch.zeros(1, 1, OBS_DIM, device="cuda", dtype=torch.bfloat16)
    print(f"  obs frames: {fake_obs_frames.shape}  state: {fake_state.shape}")
    print(f"  partial denoising: {NUM_SAMPLING_STEPS} steps, stop at step {STOP_AFTER_STEP} (sigma≈0.377)")

    # ── backbone forward (timed) ──────────────────────────────────────────
    print("\nRunning backbone forward pass (warmup)...")
    with torch.no_grad():
        _ = backbone_forward(backbone, fake_obs_frames, lang_emb)
    torch.cuda.synchronize()

    print("Running backbone forward pass (timed)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        crossattn_emb, video_sigma = backbone_forward(backbone, fake_obs_frames, lang_emb)
    torch.cuda.synchronize()
    backbone_ms = (time.perf_counter() - t0) * 1000

    print(f"  crossattn_emb shape: {crossattn_emb.shape}")
    print(f"  video_sigma: {video_sigma[0, 0].item():.4f}  (expected ≈ 0.377)")
    print(f"  Backbone forward ({STOP_AFTER_STEP} denoising steps): {backbone_ms:.1f} ms")

    # ── action head forward ───────────────────────────────────────────────
    print("\nRunning action head...")
    t0 = time.perf_counter()
    with torch.no_grad():
        actions = action_pipe(
            state_B_HO_O=fake_state,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma.to(dtype=torch.bfloat16),
        )
    torch.cuda.synchronize()
    action_ms = (time.perf_counter() - t0) * 1000

    print(f"  actions shape: {actions.shape}  (expected: [1, {ACTION_HORIZON}, {OBS_DIM}])")
    print(f"  Action head forward: {action_ms:.1f} ms")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n── Timing summary ─────────────────────────────")
    print(f"  Backbone ({STOP_AFTER_STEP}/{NUM_SAMPLING_STEPS} steps): {backbone_ms:7.1f} ms")
    print(f"  Action head:                  {action_ms:7.1f} ms")
    print(f"  Total:                        {backbone_ms + action_ms:7.1f} ms")
    print(f"  Control freq (approx): {1000 / (backbone_ms + action_ms):.2f} Hz")
    print(f"  Note: backbone now runs {STOP_AFTER_STEP} DiT passes vs 1 before — expect ~{STOP_AFTER_STEP}x slower")


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
#
# while task_not_done:
#     # 1. collect 5 obs frames from camera buffer
#     obs_frames = get_camera_frames()      # (5, H, W, 3) uint8
#     state      = get_robot_state()        # (1, 1, 5) torch bfloat16 on cuda
#
#     # 2. backbone: partially denoise imagined future (STOP_AFTER_STEP denoising steps)
#     crossattn_emb, video_sigma = backbone_forward(backbone, obs_frames, lang_emb)
#
#     # 3. action head: fast, predicts full 90-step chunk at once
#     actions = action_pipe(
#         state_B_HO_O=state,
#         crossattn_emb=crossattn_emb,
#         context_timesteps_B_1=video_sigma.to(dtype=torch.bfloat16),
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
