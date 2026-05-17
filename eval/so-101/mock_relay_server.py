"""
mock_relay_server.py  —  run on YOUR LAPTOP
Simulates relay_server.py without any real robot or camera.
Returns a random noise image and zeroed joint angles.

Usage:
    python mock_relay_server.py --port 5000
"""

import argparse
import base64
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request

CAMERA_HEIGHT = 480
CAMERA_WIDTH  = 640

app = Flask(__name__)
stop_event = threading.Event()
step_counter = 0  # so you can see actions arriving


@app.route("/observation", methods=["GET"])
def get_observation():
    # fake image: random noise (looks like static)
    fake_frame = np.random.randint(0, 255, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    _, buf = cv2.imencode(".jpg", fake_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode()

    # fake joints: all zeros (robot at neutral)
    joints = {
        "shoulder_pan.pos":   0.0,
        "shoulder_lift.pos": -90.0,
        "elbow_flex.pos":     90.0,
        "wrist_flex.pos":     45.0,
        "wrist_roll.pos":      0.0,
        "gripper.pos":         0.0,
    }

    return jsonify({"image": img_b64, "joints": joints, "stop": stop_event.is_set()})


@app.route("/action", methods=["POST"])
def send_action():
    global step_counter
    step_counter += 1
    action = request.json
    print(f"[step {step_counter:03d}] received action: "
          f"pan={action.get('shoulder_pan.pos', '?'):.2f}  "
          f"lift={action.get('shoulder_lift.pos', '?'):.2f}  "
          f"elbow={action.get('elbow_flex.pos', '?'):.2f}")
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset_robot():
    print("[mock] reset_robot called")
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def emergency_stop():
    stop_event.set()
    print("[mock] stop_event set")
    return jsonify({"ok": True})


@app.route("/clear_stop", methods=["POST"])
def clear_stop():
    stop_event.clear()
    print("[mock] stop_event cleared")
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    print(f"Mock relay server running on port {args.port}")
    print("No robot or camera needed — returning fake data.\n")
    app.run(host="0.0.0.0", port=args.port, use_reloader=False, threaded=True)