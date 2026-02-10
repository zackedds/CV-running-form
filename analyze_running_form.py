#!/usr/bin/env python3
"""
Running Form Analyzer using MediaPipe Pose.

Analyzes running form from video footage using 33-landmark pose estimation.
Detects foot strike pattern, knee angles, cadence, trunk lean, and vertical oscillation.

Usage:
    python analyze_running_form.py original_slowmo.mp4 --slowmo-factor 8.0
"""

import sys
import os
import argparse

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is not installed. Run: pip install mediapipe")
    sys.exit(1)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MediaPipe landmark indices (33 total, same for legacy and tasks API)
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_KNEE = 25
LM_RIGHT_KNEE = 26
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28
LM_LEFT_HEEL = 29
LM_RIGHT_HEEL = 30
LM_LEFT_FOOT_INDEX = 31
LM_RIGHT_FOOT_INDEX = 32

# Pose connections for drawing skeleton
POSE_CONNECTIONS = list(mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS)

# Model path (relative to script directory)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_heavy.task")


def calculate_angle(a, b, c):
    """Calculate angle at point b formed by points a-b-c. Returns degrees."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def smooth(data, window=5):
    """Simple moving average smoothing. Handles NaN values."""
    result = np.copy(data)
    half = window // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        vals = data[start:end]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result


def _create_pose_landmarker(model_path):
    """Create a PoseLandmarker using the tasks API."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Download it with:")
        print('  curl -L -o pose_landmarker_heavy.task '
              '"https://storage.googleapis.com/mediapipe-models/'
              'pose_landmarker/pose_landmarker_heavy/float16/latest/'
              'pose_landmarker_heavy.task"')
        sys.exit(1)

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _draw_skeleton(frame, landmarks_list, width, height):
    """Draw pose skeleton on a frame from landmark data."""
    if not landmarks_list:
        return

    landmarks = landmarks_list[0]  # first (only) detected pose

    # Draw connections
    for connection in POSE_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        pt1 = (int(start.x * width), int(start.y * height))
        pt2 = (int(end.x * width), int(end.y * height))
        cv2.line(frame, pt1, pt2, (0, 200, 0), 2)

    # Draw landmark points
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    # Highlight heel (red) and toe (orange) landmarks
    for idx, color in [
        (LM_LEFT_HEEL, (0, 0, 255)),
        (LM_RIGHT_HEEL, (0, 0, 255)),
        (LM_LEFT_FOOT_INDEX, (0, 165, 255)),
        (LM_RIGHT_FOOT_INDEX, (0, 165, 255)),
    ]:
        lm = landmarks[idx]
        px = int(lm.x * width)
        py = int(lm.y * height)
        cv2.circle(frame, (px, py), 6, color, -1)


class RunningFormAnalyzer:
    def __init__(self, video_path, slowmo_factor=1.0):
        self.video_path = video_path
        self.slowmo_factor = slowmo_factor

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            sys.exit(1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.real_fps = self.fps * slowmo_factor
        self.frame_data = []
        self.ground_contacts = []
        self.metrics = {}

        print(f"Video: {video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps:.2f} (real-time: {self.real_fps:.1f})")
        print(f"Frames: {self.frame_count}")
        print(f"Duration: {self.frame_count / self.fps:.2f}s "
              f"(real-time: {self.frame_count / self.real_fps:.2f}s)")
        print(f"Slowmo factor: {self.slowmo_factor}x")

    def extract_landmarks(self):
        """Phase 1: Run MediaPipe Pose on every frame, extract 33 landmarks."""
        print("\n--- Phase 1: Extracting landmarks ---")
        landmarker = _create_pose_landmarker(MODEL_PATH)

        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        detected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / self.fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                pose_lms = results.pose_landmarks[0]
                landmarks = {}
                for idx, lm in enumerate(pose_lms):
                    landmarks[idx] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "vis": lm.visibility,
                        "px_x": int(lm.x * self.width),
                        "px_y": int(lm.y * self.height),
                    }
                self.frame_data.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / self.fps,
                    "landmarks": landmarks,
                })
                detected += 1
            else:
                self.frame_data.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / self.fps,
                    "landmarks": None,
                })

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{self.frame_count} frames...")

        cap.release()
        landmarker.close()
        print(f"  Done. Landmarks detected in {detected}/{frame_idx} frames "
              f"({100 * detected / max(frame_idx, 1):.1f}%)")

    def generate_skeleton_video(self, output_path):
        """Generate video with skeleton overlay (Phase 1 visual output)."""
        print(f"\n--- Generating skeleton video: {output_path} ---")
        landmarker = _create_pose_landmarker(MODEL_PATH)

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                              (self.width, self.height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / self.fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.pose_landmarks:
                _draw_skeleton(frame, results.pose_landmarks,
                               self.width, self.height)

            # Frame counter
            cv2.putText(
                frame,
                f"Frame {frame_idx}/{self.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            out.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Written {frame_idx}/{self.frame_count} frames...")

        cap.release()
        out.release()
        landmarker.close()
        print(f"  Done. Saved to {output_path}")

    def run(self, output_dir="output", generate_video=True):
        """Run the full analysis pipeline."""
        os.makedirs(output_dir, exist_ok=True)

        # Phase 1: Extract landmarks
        self.extract_landmarks()

        # Phase 1: Skeleton video
        if generate_video:
            video_out = os.path.join(output_dir, "skeleton_overlay.mp4")
            self.generate_skeleton_video(video_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Form Analyzer")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--slowmo-factor", type=float, default=1.0,
        help="Slowmo factor (e.g. 8.0 for 8x slowmo video)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output)",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video generation (faster)",
    )
    args = parser.parse_args()

    analyzer = RunningFormAnalyzer(args.video, slowmo_factor=args.slowmo_factor)
    analyzer.run(output_dir=args.output_dir, generate_video=not args.no_video)
