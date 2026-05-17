"""
relay_server.py  —  run this on YOUR LAPTOP
Bridges the SO-101 robot and camera to Brev over HTTP.

Usage:
    python relay_server.py --port 5000 --camera-index 1 --robot-port /dev/ttyACM0
"""

import argparse
import base64
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── constants (must match run.py) ────────────────────────────────────────────
CAMERA_HEIGHT = 480
CAMERA_WIDTH  = 640

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

START_POSITION = {
    "shoulder_pan.pos":   0.0,
    "shoulder_lift.pos": -90.0,
    "elbow_flex.pos":     90.0,
    "wrist_flex.pos":     45.0,
    "wrist_roll.pos":      0.0,
    "gripper.pos":         0.0,
}

# ── globals set at startup ────────────────────────────────────────────────────
robot: SO101Follower = None
cap:   cv2.VideoCapture = None
app = Flask(__name__)

# ── emergency-stop flag (Brev sets it via POST /stop) ────────────────────────
stop_event = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/observation", methods=["GET"])
def get_observation():
    """Return current camera frame (JPEG/base64) + joint angles."""
    # camera frame
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "camera read failed"}), 500
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame_rgb.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        frame_rgb = cv2.resize(frame_rgb, (CAMERA_WIDTH, CAMERA_HEIGHT))

    _, buf = cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode()

    # joint angles
    obs = robot.get_observation()
    joints = {k: float(obs[k]) for k in (f"{j}.pos" for j in JOINT_NAMES)}

    return jsonify({"image": img_b64, "joints": joints, "stop": stop_event.is_set()})


@app.route("/action", methods=["POST"])
def send_action():
    """Receive joint position commands from Brev and forward to robot."""
    action = request.json          # dict like {"shoulder_pan.pos": 12.3, ...}
    robot.send_action(action)
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset_robot():
    """Smoothly move robot back to START_POSITION (called between attempts)."""
    steps = int(request.json.get("steps", 50))
    hz    = float(request.json.get("hz", 25.0))
    dt    = 1.0 / hz

    obs     = robot.get_observation()
    current = {k: float(obs[k]) for k in START_POSITION}

    for i in range(1, steps + 1):
        alpha  = i / steps
        interp = {k: current[k] + alpha * (START_POSITION[k] - current[k])
                  for k in START_POSITION}
        robot.send_action(interp)
        time.sleep(dt)

    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def emergency_stop():
    """Set the stop flag so Brev's control loop knows to halt."""
    stop_event.set()
    return jsonify({"ok": True})


@app.route("/clear_stop", methods=["POST"])
def clear_stop():
    """Clear stop flag before a new attempt."""
    stop_event.clear()
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-port",   default="/dev/ttyACM0")
    parser.add_argument("--camera-index", default=1, type=int)
    parser.add_argument("--port",         default=5000, type=int)
    args = parser.parse_args()

    global robot, cap

    print(f"Connecting robot on {args.robot_port}...")
    config = SO101FollowerConfig(port=args.robot_port, id="so101_pusher", cameras={})
    robot  = SO101Follower(config)
    robot.connect()
    print("Robot connected.")

    print(f"Opening camera index {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")
    print(f"Camera ready: {cap.isOpened()}")

    print(f"\nRelay server starting on port {args.port} ...")
    print("Waiting for Brev to connect...\n")

    # use_reloader=False is important — avoids double-init of robot/camera
    app.run(host="0.0.0.0", port=args.port, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()