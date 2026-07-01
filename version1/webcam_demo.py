#!/usr/bin/env python3
"""
Live webcam gesture detection for version1 models.

Usage:
  python version1/webcam_demo.py
  python version1/webcam_demo.py --model version1/models/gesture_transformer_v1.pt
  python version1/webcam_demo.py --model version1/models/gesture_bilstm_v1.pt --threshold 0.6

Press q to quit.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VERSION_DIR = Path(__file__).resolve().parent
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from gesture_models import NUM_FRAMES, load_gesture_model, resolve_device
PROJECT_ROOT = VERSION_DIR.parent
DEFAULT_MODEL = VERSION_DIR / "models" / "gesture_bilstm_v1.pt"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = PROJECT_ROOT / "mediapipe_model" / "hand_landmarker.task"

NO_GESTURE_IDX = 0


def hand_detected_mask(seq: np.ndarray) -> np.ndarray:
    return np.any(seq != 0, axis=1)


def gesture_window(mask: np.ndarray) -> tuple[int | None, int | None]:
    if not mask.any():
        return None, None
    start = int(np.argmax(mask))
    end = int(len(mask) - 1 - np.argmax(mask[::-1]))
    return start, end


def interpolate_gaps(seq: np.ndarray) -> np.ndarray:
    """Match final_data preprocessing from data_processing.ipynb."""
    seq = seq.copy()
    detected = hand_detected_mask(seq)
    start_win, end_win = gesture_window(detected)
    if start_win is None:
        return seq

    i = start_win
    while i <= end_win:
        if detected[i]:
            i += 1
            continue

        gap_start = i
        while i <= end_win and not detected[i]:
            i += 1
        gap_end = i
        gap = gap_end - gap_start

        if gap <= 0 or gap_end > end_win:
            continue

        prev_frame = seq[gap_start - 1]
        next_frame = seq[gap_end]
        start = gap_start - 1

        for k in range(1, gap + 1):
            alpha = k / (gap + 1)
            seq[start + k] = (1 - alpha) * prev_frame + alpha * next_frame

    return seq


class HandLandmarkExtractor:
    def __init__(self):
        HAND_LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not HAND_LANDMARKER_PATH.exists():
            print(f"Downloading MediaPipe hand model to {HAND_LANDMARKER_PATH} ...")
            urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        frame_landmarks = np.zeros(63, dtype=np.float32)
        hand_landmarks = []
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            for i, lm in enumerate(hand_landmarks):
                frame_landmarks[i * 3 : i * 3 + 3] = [lm.x, lm.y, lm.z]

        return frame_landmarks, hand_landmarks

    def close(self):
        self._landmarker.close()


def draw_hand_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    if not hand_landmarks:
        return

    h, w = frame.shape[:2]
    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for start_idx, end_idx in connections:
        x1 = int(hand_landmarks[start_idx].x * w)
        y1 = int(hand_landmarks[start_idx].y * h)
        x2 = int(hand_landmarks[end_idx].x * w)
        y2 = int(hand_landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)


def predict_gesture(
    model: torch.nn.Module,
    labels: list[str],
    sequence: np.ndarray,
    device: torch.device,
) -> tuple[str, float, np.ndarray]:
    seq = interpolate_gaps(sequence.astype(np.float32))
    tensor = torch.from_numpy(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return labels[pred_idx], float(probs[pred_idx]), probs


def best_non_idle_gesture(
    labels: list[str],
    probs: np.ndarray,
    threshold: float,
) -> tuple[str, float] | None:
    best_idx = None
    best_prob = threshold
    for idx in range(1, len(labels)):
        if probs[idx] >= best_prob:
            best_prob = probs[idx]
            best_idx = idx
    if best_idx is None:
        return None
    return labels[best_idx], best_prob


def draw_overlay(
    frame: np.ndarray,
    *,
    model_path: Path,
    buffer_size: int,
    detection: tuple[str, float] | None,
    status: str,
) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Model: {model_path.name}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        frame,
        status,
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        2,
    )
    cv2.putText(
        frame,
        f"Buffer: {buffer_size}/{NUM_FRAMES}",
        (12, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        2,
    )

    if detection is not None:
        label, prob = detection
        text = f"{label} detected ({prob:.0%})"
        cv2.putText(
            frame,
            text,
            (12, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 120),
            3,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam gesture detection (version1)")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a .pt checkpoint in version1/models/",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum probability to show a non-idle gesture",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=True,
        help="Mirror the webcam preview (default: on)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_false",
        dest="mirror",
        help="Disable mirrored preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")

    model, labels, hyperparameters = load_gesture_model(model_path, device=device)
    print(f"Model class: {type(model).__name__}")
    print(f"Classes: {labels}")
    print(f"Expected input: ({hyperparameters.get('num_frames', NUM_FRAMES)}, 63)")

    extractor = HandLandmarkExtractor()
    frame_buffer: deque[np.ndarray] = deque(maxlen=NUM_FRAMES)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open camera {args.camera}")

    print("Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            landmarks, hand_landmarks = extractor.extract(frame)
            frame_buffer.append(landmarks)
            draw_hand_landmarks(frame, hand_landmarks)

            detection = None
            if len(frame_buffer) < NUM_FRAMES:
                status = "Warming up — perform a gesture when ready"
            else:
                sequence = np.stack(frame_buffer)
                _, _, probs = predict_gesture(model, labels, sequence, device)
                detection = best_non_idle_gesture(labels, probs, args.threshold)
                status = "Ready"

            draw_overlay(
                frame,
                model_path=model_path,
                buffer_size=len(frame_buffer),
                detection=detection,
                status=status,
            )

            cv2.imshow("Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
