# Running SO-101 Inference on Brev Cluster

## Architecture

The inference runs split across two machines:

```
YOUR LAPTOP                          BREV (GPU cluster)
─────────────────────────            ──────────────────────────────
relay_server.py                      run_brev.py
  - robot on /dev/ttyACM0              - VAMInference (GPU)
  - camera (USB)                       - control loop
  - Flask HTTP server        ◄──────►  - calls laptop for obs/action
```

Brev runs the heavy GPU inference. Your laptop owns the robot and camera and serves them over HTTP. An SSH tunnel connects the two without any third-party service.

---

## What's confirmed working (as of 2026-05-17)

- Model loads correctly on Brev (video backbone + action decoder + dataset stats)
- CUDA graphs compile successfully (~45s on first run, cached after)
- Full inference pipeline runs end-to-end on GPU
- SSH tunnel is stable through CUDA graph compilation pauses
- HTTP relay architecture works (Brev ↔ SSH tunnel ↔ laptop)
- Action loop ticks at ~1s/step at 5 Hz with inference
- Action values are in expected degree ranges for all joints
- Reset, clear_stop, health endpoints all work

## What still needs testing with real hardware

- Real robot — serial on `/dev/ttyACM0`, USB forwarding via `usbipd` on WSL
- Real camera — V4L2 capture, correct camera index, real frames
- Real prompt embeddings — currently using fake ones; model behaviour may differ
- Action quality — whether outputs make physical sense on the real robot
- Emergency stop — ESC key on laptop halting the loop
- Success detection — `object_in_target_circle` is currently commented out

---

## Prerequisites

### On your laptop
- WSL2 (Ubuntu) with the `cosmos-predict2` conda environment
- `flask`, `opencv-python` installed (`pip install flask opencv-python`)
- SSH access to Brev via `~/.brev/brev.pem`

### On Brev
- `mimicvideo` conda environment with all model dependencies
- Model checkpoints at:
  - `../../model/checkpoints/video_backbone/iter_000001800_fused.pt`
  - `../../model/checkpoints/action_decoder/iter_000010000.pt`
  - `../../model/checkpoints/stats_mimic.json`
- `requests` installed (`pip install requests`)

---

## Step-by-step: testing with mock (no robot needed)

Use this to verify the full pipeline before you have the hardware.

### Step 1 — Start the mock relay on your laptop

Open a terminal on your laptop:

```bash
cd /mnt/c/Users/<you>/Desktop/ETH/robot-learning-group-13/eval/so-101
conda activate cosmos-predict2
python mock_relay_server.py --port 5000
```

You should see:
```
Mock relay server running on port 5000
No robot or camera needed — returning fake data.
* Running on http://127.0.0.1:5000
```

### Step 2 — Open the SSH tunnel

Open a second terminal on your laptop and leave it open for the whole session:

```bash
ssh -R 5000:localhost:5000 -p 22 -i "/home/<you>/.brev/brev.pem" shadeform@216.81.200.28
```

> Replace `<you>` with your WSL username. The IP and key path come from `cat ~/.brev/ssh_config`.
> **Do not run anything in this terminal** — just keep it open.

### Step 3 — Run inference on Brev

In your existing Brev terminal:

```bash
cd ~/robot-learning-group-13/eval/so-101
conda activate mimicvideo
python run_brev.py \
    --experiment-name w2a_rl_group13_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128 \
    --video-model-path ../../model/checkpoints/video_backbone/iter_000001800_fused.pt \
    --action-model-path ../../model/checkpoints/action_decoder/iter_000010000.pt \
    --dataset-statistics-path ../../model/checkpoints/stats_mimic.json \
    --task task1 \
    --relay-url http://localhost:5000
```

### Step 4 — Verify it's working

On Brev you should see the model load, CUDA graphs compile (~45s), then:
```
[Attempt 1/5] Place object in start circle, then press Enter here on Brev...
```

Press Enter. On your laptop's mock terminal you should see actions arriving:
```
[step 001] received action: pan=-2.88  lift=-76.50  elbow=82.00
[step 002] received action: pan=-0.50  lift=-79.00  elbow=82.00
...
```

Varying action values = the model is running inference correctly. ✅

### Step 5 — Verify relay API directly (optional sanity check)

In a third laptop terminal:

```bash
python - <<'EOF'
import requests, base64, numpy as np, cv2
URL = "http://localhost:5000"
r = requests.get(f"{URL}/observation").json()
img = np.frombuffer(base64.b64decode(r["image"]), np.uint8)
frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
print("image shape:", frame.shape)    # expect (480, 640, 3)
print("joints:", r["joints"])         # expect dict with 6 keys
print("stop:", r["stop"])             # expect False
resp = requests.post(f"{URL}/action", json={
    "shoulder_pan.pos": 12.3, "shoulder_lift.pos": -45.0,
    "elbow_flex.pos": 60.0, "wrist_flex.pos": 30.0,
    "wrist_roll.pos": 0.0, "gripper.pos": 0.0,
})
print("action response:", resp.json())  # expect {"ok": True}
EOF
```

---

## Step-by-step: running with real robot

Do this once you have the hardware.

### Step 1 — Forward the robot's USB port into WSL

In **Windows PowerShell (as admin)**:

```powershell
winget install usbipd
usbipd list                        # find the robot, e.g. "USB Serial Device (COM3)"
usbipd bind --busid <busid>
usbipd attach --wsl --busid <busid>
```

In WSL, confirm it appeared:
```bash
ls /dev/tty*                       # look for /dev/ttyACM0 or /dev/ttyUSB0
# or
lerobot-find-port
```

### Step 2 — Start the real relay server

```bash
cd /mnt/c/Users/<you>/Desktop/ETH/robot-learning-group-13/eval/so-101
conda activate cosmos-predict2
python relay_server.py --robot-port /dev/ttyACM0 --camera-index 1 --port 5000
```

> If the robot doesn't connect, try `--robot-port /dev/ttyUSB0`.
> If the camera is wrong, try `--camera-index 0`.

### Step 3 — SSH tunnel and Brev

Same as mock steps 2 and 3 above — nothing changes on the Brev side.

### Step 4 — Prompt embeddings

Make sure the real embeddings file exists on Brev:
```
../../model/checkpoints/prompt_embeddings.pt
```

If it doesn't exist, the script will compute it automatically on first run (requires the text encoder — comment out `use_text_encoder=False` temporarily). After it saves, restore the flag.

---

## Troubleshooting

**SSH tunnel permission denied**
```bash
# Make sure you're using the right key and user
ssh -R 5000:localhost:5000 -p 22 -i "/home/<you>/.brev/brev.pem" shadeform@216.81.200.28
```

**Relay not reachable from Brev**
- Check the SSH tunnel terminal is still open
- Check mock/relay is still running on laptop
- Test: `curl http://localhost:5000/health` from inside Brev

**ngrok SSL drops during CUDA graph compilation**
- Don't use ngrok — use the SSH tunnel instead (ngrok free tier drops idle connections after ~30s, CUDA graph compilation takes ~45s)

**Robot port not found**
- Run `lerobot-find-port` or `ls /dev/tty*` after attaching via usbipd
- Port may be `/dev/ttyUSB0` instead of `/dev/ttyACM0`

**Camera index wrong**
- Try `--camera-index 0` or `--camera-index 1`
- Check available cameras: `ls /dev/video*`

**CUDA graphs recompile every run**
- Normal on first run only; subsequent runs reuse the cache and start much faster

---

## File reference

| File | Runs on | Purpose |
|---|---|---|
| `relay_server.py` | Laptop | Real robot + camera → HTTP server |
| `mock_relay_server.py` | Laptop | Fake robot + camera for testing |
| `run_brev.py` | Brev | GPU inference + control loop |
