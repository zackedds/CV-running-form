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

# MediaPipe landmark indices (33 total)
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
NUM_LANDMARKS = 33

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


def _draw_skeleton_from_arrays(frame, lm_x, lm_y, width, height):
    """Draw pose skeleton on a frame from corrected landmark arrays.

    Args:
        frame: BGR image to draw on.
        lm_x: array of shape (33,) with normalized x coords for this frame.
        lm_y: array of shape (33,) with normalized y coords for this frame.
        width, height: frame dimensions.
    """
    if np.isnan(lm_x[0]):
        return

    # Draw connections
    for connection in POSE_CONNECTIONS:
        x1 = int(lm_x[connection.start] * width)
        y1 = int(lm_y[connection.start] * height)
        x2 = int(lm_x[connection.end] * width)
        y2 = int(lm_y[connection.end] * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Draw landmark dots
    for i in range(NUM_LANDMARKS):
        px = int(lm_x[i] * width)
        py = int(lm_y[i] * height)
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    # Highlight heel (red) and toe (orange)
    for idx, color in [
        (LM_LEFT_HEEL, (0, 0, 255)),
        (LM_RIGHT_HEEL, (0, 0, 255)),
        (LM_LEFT_FOOT_INDEX, (0, 165, 255)),
        (LM_RIGHT_FOOT_INDEX, (0, 165, 255)),
    ]:
        px = int(lm_x[idx] * width)
        py = int(lm_y[idx] * height)
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
        self.ground_contacts = []
        self.metrics = {}

        # Landmark arrays: shape (n_frames, 33) for x, y, z, vis
        self.lm_x = None
        self.lm_y = None
        self.lm_z = None
        self.lm_vis = None
        self.detected = None  # bool array: was pose detected in this frame?

        print(f"Video: {video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps:.2f} (real-time: {self.real_fps:.1f})")
        print(f"Frames: {self.frame_count}")
        print(f"Duration: {self.frame_count / self.fps:.2f}s "
              f"(real-time: {self.frame_count / self.real_fps:.2f}s)")
        print(f"Slowmo factor: {self.slowmo_factor}x")

    def extract_landmarks(self):
        """Phase 1: Run MediaPipe Pose on every frame, store as numpy arrays."""
        print("\n--- Phase 1: Extracting landmarks ---")
        landmarker = _create_pose_landmarker(MODEL_PATH)

        cap = cv2.VideoCapture(self.video_path)
        n = self.frame_count

        self.lm_x = np.full((n, NUM_LANDMARKS), np.nan)
        self.lm_y = np.full((n, NUM_LANDMARKS), np.nan)
        self.lm_z = np.full((n, NUM_LANDMARKS), np.nan)
        self.lm_vis = np.full((n, NUM_LANDMARKS), 0.0)
        self.detected = np.zeros(n, dtype=bool)

        frame_idx = 0
        det_count = 0

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
                for idx, lm in enumerate(pose_lms):
                    self.lm_x[frame_idx, idx] = lm.x
                    self.lm_y[frame_idx, idx] = lm.y
                    self.lm_z[frame_idx, idx] = lm.z
                    self.lm_vis[frame_idx, idx] = lm.visibility
                self.detected[frame_idx] = True
                det_count += 1

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{n} frames...")

        cap.release()
        landmarker.close()
        print(f"  Done. Landmarks detected in {det_count}/{frame_idx} frames "
              f"({100 * det_count / max(frame_idx, 1):.1f}%)")

    def correct_landmarks(self, delta_threshold_px=30.0, smooth_window=5):
        """Detect and fix landmark jitter by interpolating outlier frames,
        then apply light smoothing to all landmarks.

        Args:
            delta_threshold_px: max allowed frame-to-frame pixel jump before
                marking a frame as an outlier.
            smooth_window: moving average window size for final smoothing.
        """
        print(f"\n--- Correcting landmark jitter ---")
        print(f"  Delta threshold: {delta_threshold_px}px, "
              f"smooth window: {smooth_window}")

        n = self.lm_x.shape[0]
        total_fixed = 0

        for lm_idx in range(NUM_LANDMARKS):
            x = self.lm_x[:, lm_idx].copy()
            y = self.lm_y[:, lm_idx].copy()

            # Convert to pixel space for threshold comparison
            x_px = x * self.width
            y_px = y * self.height

            # Compute frame-to-frame delta in pixels
            dx = np.diff(x_px)
            dy = np.diff(y_px)
            deltas = np.sqrt(dx**2 + dy**2)

            # Find outlier frames: where delta exceeds threshold
            # An outlier frame i means the jump FROM i-1 TO i was too big
            outlier_mask = np.zeros(n, dtype=bool)
            for i in range(len(deltas)):
                if deltas[i] > delta_threshold_px:
                    outlier_mask[i + 1] = True

            # Also check if the frame AFTER an outlier snaps back
            # (teleport-and-return pattern): mark the outlier frame
            for i in range(1, n - 1):
                if outlier_mask[i]:
                    # Check if next frame snaps back near the frame before
                    if i + 1 < n and not np.isnan(x_px[i - 1]):
                        snap_back = np.sqrt(
                            (x_px[i + 1] - x_px[i - 1])**2 +
                            (y_px[i + 1] - y_px[i - 1])**2
                        )
                        # If it snaps back within threshold, this is a single-
                        # frame glitch — mark only frame i
                        if snap_back < delta_threshold_px:
                            continue
                    # If it doesn't snap back, this might be a multi-frame
                    # glitch — keep scanning forward
                    j = i + 1
                    while j < n:
                        d = np.sqrt(
                            (x_px[j] - x_px[i - 1])**2 +
                            (y_px[j] - y_px[i - 1])**2
                        )
                        if d < delta_threshold_px:
                            break
                        outlier_mask[j] = True
                        j += 1

            n_outliers = np.sum(outlier_mask)
            if n_outliers > 0:
                total_fixed += n_outliers
                # Replace outlier positions with NaN, then interpolate
                x[outlier_mask] = np.nan
                y[outlier_mask] = np.nan

                # Linear interpolation over NaN gaps
                valid = ~np.isnan(x)
                if np.sum(valid) >= 2:
                    frames = np.arange(n)
                    x = np.interp(frames, frames[valid], x[valid])
                    y = np.interp(frames, frames[valid], y[valid])

            # Apply light smoothing
            x = smooth(x, window=smooth_window)
            y = smooth(y, window=smooth_window)

            self.lm_x[:, lm_idx] = x
            self.lm_y[:, lm_idx] = y

        print(f"  Fixed {total_fixed} outlier landmark-frames across all landmarks")

    def generate_jitter_report(self, output_dir):
        """Generate before/after jitter diagnostic plots."""
        print(f"\n--- Generating jitter diagnostic plots ---")

        # We need to re-extract raw landmarks to show before vs after
        # Instead, just plot the corrected foot positions
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle("Corrected Foot Landmark Y-Position (pixels)",
                     fontsize=14, fontweight="bold")

        times = np.arange(self.frame_count) / self.fps

        for ax_idx, (name, idx) in enumerate([
            ("Left Heel", LM_LEFT_HEEL),
            ("Left Toe", LM_LEFT_FOOT_INDEX),
            ("Left Ankle", LM_LEFT_ANKLE),
        ]):
            y_px = self.lm_y[:, idx] * self.height
            ax = axes[ax_idx]
            ax.plot(times, y_px, linewidth=1.0, color="green",
                    label="corrected")
            ax.set_ylabel("Y position (px)")
            ax.set_title(name)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        path = os.path.join(output_dir, "corrected_positions.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    def generate_skeleton_video(self, output_path):
        """Generate video with skeleton overlay from corrected landmark data."""
        print(f"\n--- Generating corrected skeleton video: {output_path} ---")

        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                              (self.width, self.height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < self.lm_x.shape[0] and self.detected[frame_idx]:
                _draw_skeleton_from_arrays(
                    frame,
                    self.lm_x[frame_idx],
                    self.lm_y[frame_idx],
                    self.width,
                    self.height,
                )

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
        print(f"  Done. Saved to {output_path}")

    def run(self, output_dir="output", generate_video=True):
        """Run the full analysis pipeline."""
        os.makedirs(output_dir, exist_ok=True)

        # Phase 1: Extract landmarks
        self.extract_landmarks()

        # Correct jitter
        self.correct_landmarks()
        self.generate_jitter_report(output_dir)

        # Skeleton video from corrected data
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
