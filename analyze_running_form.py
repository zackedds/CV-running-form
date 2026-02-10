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

    def fix_lr_swaps(self):
        """Detect and fix left/right landmark identity swaps.

        MediaPipe sometimes swaps which foot is left vs right. During
        normal running the feet cross naturally (gradual), but a swap
        event shows as a sudden large jump in the L-R heel X difference
        (~200px in one frame). We detect these jump events, pair them
        (swap-in / swap-out), and unswap the frames between each pair.
        """
        print("\n--- Fixing left/right landmark swaps ---")

        lr_pairs = [
            (LM_LEFT_HIP, LM_RIGHT_HIP),
            (LM_LEFT_KNEE, LM_RIGHT_KNEE),
            (LM_LEFT_ANKLE, LM_RIGHT_ANKLE),
            (LM_LEFT_HEEL, LM_RIGHT_HEEL),
            (LM_LEFT_FOOT_INDEX, LM_RIGHT_FOOT_INDEX),
            (LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER),
        ]

        n = self.lm_x.shape[0]
        l_x = self.lm_x[:, LM_LEFT_HEEL] * self.width
        r_x = self.lm_x[:, LM_RIGHT_HEEL] * self.width
        diff = l_x - r_x

        # Detect swap events: sudden jumps in the L-R difference.
        # A real swap causes ~200px swing in one frame; natural crossing
        # changes by only a few px per frame.
        diff_delta = np.diff(diff)
        swap_threshold = 80.0  # px jump in L-R diff in one frame

        # Find swap event frames
        swap_events = np.where(np.abs(diff_delta) > swap_threshold)[0]

        if len(swap_events) == 0:
            print("  No swaps detected")
            return

        # Group nearby events into swap regions. Each region is a
        # contiguous block where swaps are happening. Within each region,
        # find the start (first event) and end (last event).
        regions = []
        region_start = swap_events[0]
        region_end = swap_events[0]
        for i in range(1, len(swap_events)):
            if swap_events[i] - region_end < 30:  # within ~1s
                region_end = swap_events[i]
            else:
                regions.append((region_start, region_end))
                region_start = swap_events[i]
                region_end = swap_events[i]
        regions.append((region_start, region_end))

        # For each swap region, determine which frames are swapped by
        # comparing the L-R diff inside the region to the stable diff
        # just outside the region.
        swapped = np.zeros(n, dtype=bool)
        for start, end in regions:
            # Get reference diff from frames just before the region
            ref_start = max(0, start - 10)
            ref_diff = np.nanmedian(diff[ref_start:start + 1])

            # Mark frames within the region where diff sign disagrees
            # with the reference (with significant magnitude)
            for f in range(start + 1, min(end + 2, n)):
                if np.sign(diff[f]) != np.sign(ref_diff) and \
                   np.abs(diff[f]) > 30.0:
                    swapped[f] = True

        n_swapped = np.sum(swapped)
        if n_swapped == 0:
            print("  No swaps detected after region analysis")
            return

        t_ranges = []
        for start, end in regions:
            region_count = np.sum(swapped[start:end + 2])
            if region_count > 0:
                t_ranges.append(
                    f"{start / self.fps:.1f}-{(end + 1) / self.fps:.1f}s "
                    f"({region_count} frames)")
        print(f"  Detected {n_swapped} swapped frames in {len(t_ranges)} regions:")
        for tr in t_ranges:
            print(f"    {tr}")

        for l_idx, r_idx in lr_pairs:
            self.lm_x[swapped, l_idx], self.lm_x[swapped, r_idx] = \
                self.lm_x[swapped, r_idx].copy(), self.lm_x[swapped, l_idx].copy()
            self.lm_y[swapped, l_idx], self.lm_y[swapped, r_idx] = \
                self.lm_y[swapped, r_idx].copy(), self.lm_y[swapped, l_idx].copy()
            self.lm_z[swapped, l_idx], self.lm_z[swapped, r_idx] = \
                self.lm_z[swapped, r_idx].copy(), self.lm_z[swapped, l_idx].copy()

    def correct_landmarks(self, mad_multiplier=3.0, smooth_window=5):
        """Detect and fix landmark jitter using adaptive per-landmark thresholds.

        Fits a baseline from the median of frame-to-frame deltas, then flags
        anything beyond median + mad_multiplier * MAD as an outlier. Outlier
        frames are replaced with linearly interpolated values from the nearest
        valid neighbors, then a light smoothing pass is applied.

        Args:
            mad_multiplier: how many MADs above the median to set the cutoff.
                8.0 gives ~7px threshold for typical slowmo running video.
            smooth_window: moving average window for final smoothing pass.
        """
        print(f"\n--- Correcting landmark jitter (adaptive threshold) ---")

        n = self.lm_x.shape[0]
        total_fixed = 0
        self._jitter_stats = []  # store per-landmark stats for plotting

        for lm_idx in range(NUM_LANDMARKS):
            x = self.lm_x[:, lm_idx].copy()
            y = self.lm_y[:, lm_idx].copy()

            x_px = x * self.width
            y_px = y * self.height

            # Frame-to-frame deltas in pixel space
            dx = np.diff(x_px)
            dy = np.diff(y_px)
            deltas = np.sqrt(dx**2 + dy**2)
            valid_deltas = deltas[~np.isnan(deltas)]

            if len(valid_deltas) == 0:
                continue

            # Fit baseline: median delta is the "normal" movement per frame
            baseline = np.median(valid_deltas)
            mad = np.median(np.abs(valid_deltas - baseline))
            # Guard against MAD=0 (perfectly static landmark): use 0.5px floor
            mad = max(mad, 0.5)
            threshold = baseline + mad_multiplier * mad

            self._jitter_stats.append({
                "lm_idx": lm_idx,
                "baseline": baseline,
                "mad": mad,
                "threshold": threshold,
            })

            # Iterative delta-based cleaning: flag destination frames of any
            # delta that exceeds threshold, NaN them out, interpolate from
            # valid neighbors, then recompute deltas. Repeat until no new
            # outliers are found. This avoids the "stuck anchor" problem.
            outlier_mask = np.zeros(n, dtype=bool)
            frames_arr = np.arange(n)
            for _pass in range(10):  # safety cap on iterations
                cx_px = x * self.width
                cy_px = y * self.height
                cdx = np.diff(cx_px)
                cdy = np.diff(cy_px)
                cdeltas = np.sqrt(cdx**2 + cdy**2)

                new_outliers = 0
                for i in range(len(cdeltas)):
                    if cdeltas[i] > threshold and not outlier_mask[i + 1]:
                        outlier_mask[i + 1] = True
                        new_outliers += 1

                if new_outliers == 0:
                    break

                # NaN out outliers and interpolate from valid neighbors
                x_clean = self.lm_x[:, lm_idx].copy()
                y_clean = self.lm_y[:, lm_idx].copy()
                x_clean[outlier_mask] = np.nan
                y_clean[outlier_mask] = np.nan
                valid = ~np.isnan(x_clean)
                if np.sum(valid) >= 2:
                    x = np.interp(frames_arr, frames_arr[valid],
                                  x_clean[valid])
                    y = np.interp(frames_arr, frames_arr[valid],
                                  y_clean[valid])

            total_fixed += np.sum(outlier_mask)

            # Light smoothing pass
            x = smooth(x, window=smooth_window)
            y = smooth(y, window=smooth_window)

            self.lm_x[:, lm_idx] = x
            self.lm_y[:, lm_idx] = y

        # Print summary for foot landmarks
        foot_ids = {27, 28, 29, 30, 31, 32}
        for s in self._jitter_stats:
            if s["lm_idx"] in foot_ids:
                print(f"  LM {s['lm_idx']:2d}: baseline={s['baseline']:.1f}px  "
                      f"MAD={s['mad']:.1f}px  threshold={s['threshold']:.1f}px")

        print(f"  Fixed {total_fixed} outlier landmark-frames total")

    def generate_jitter_report(self, output_dir, raw_lm_x=None, raw_lm_y=None):
        """Generate jitter diagnostic plots showing raw deltas with fitted
        baseline/threshold, and corrected Y-positions."""
        print(f"\n--- Generating jitter diagnostic plots ---")
        times = np.arange(self.frame_count) / self.fps

        # --- Plot 1: Raw deltas with baseline & threshold for foot landmarks ---
        if raw_lm_x is not None:
            foot_lms = [
                ("L_Heel", LM_LEFT_HEEL), ("R_Heel", LM_RIGHT_HEEL),
                ("L_Toe", LM_LEFT_FOOT_INDEX), ("R_Toe", LM_RIGHT_FOOT_INDEX),
            ]
            fig, axes = plt.subplots(len(foot_lms), 1, figsize=(16, 14),
                                     sharex=True)
            fig.suptitle("Foot Landmark Deltas — Raw with Adaptive Threshold",
                         fontsize=14, fontweight="bold")

            stats_by_id = {s["lm_idx"]: s for s in self._jitter_stats}

            for ax_idx, (name, idx) in enumerate(foot_lms):
                raw_x_px = raw_lm_x[:, idx] * self.width
                raw_y_px = raw_lm_y[:, idx] * self.height
                dx = np.diff(raw_x_px)
                dy = np.diff(raw_y_px)
                deltas = np.sqrt(dx**2 + dy**2)

                ax = axes[ax_idx]
                ax.plot(times[1:], deltas, linewidth=0.5, alpha=0.7,
                        color="steelblue", label="raw delta")

                s = stats_by_id.get(idx)
                if s:
                    ax.axhline(s["baseline"], color="green", linestyle="-",
                               linewidth=1.5, alpha=0.8,
                               label=f"baseline (median={s['baseline']:.1f}px)")
                    ax.axhline(s["threshold"], color="red", linestyle="--",
                               linewidth=1.5, alpha=0.8,
                               label=f"threshold={s['threshold']:.1f}px")
                    ax.fill_between(times[1:], 0, s["threshold"],
                                    alpha=0.05, color="green")
                    outliers = deltas > s["threshold"]
                    if np.any(outliers):
                        ax.scatter(times[1:][outliers], deltas[outliers],
                                   c="red", s=12, zorder=5,
                                   label=f"outliers ({np.sum(outliers)})")

                ax.set_ylabel("Delta (px)")
                ax.set_title(name)
                ax.legend(loc="upper right", fontsize=8)

            axes[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            path = os.path.join(output_dir, "delta_with_threshold.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved {path}")

        # --- Plot 2: Corrected Y-positions ---
        fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig2.suptitle("Corrected Foot Landmark Y-Position (pixels)",
                      fontsize=14, fontweight="bold")

        for ax_idx, (name, idx) in enumerate([
            ("Left Heel", LM_LEFT_HEEL),
            ("Left Toe", LM_LEFT_FOOT_INDEX),
            ("Left Ankle", LM_LEFT_ANKLE),
        ]):
            y_px = self.lm_y[:, idx] * self.height
            ax = axes2[ax_idx]

            # Show raw as faded background if available
            if raw_lm_y is not None:
                raw_y = raw_lm_y[:, idx] * self.height
                ax.plot(times, raw_y, linewidth=0.5, alpha=0.3, color="red",
                        label="raw")

            ax.plot(times, y_px, linewidth=1.0, color="green",
                    label="corrected")
            ax.set_ylabel("Y position (px)")
            ax.set_title(name)
            ax.legend(loc="upper right")

        axes2[-1].set_xlabel("Time (s)")
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
        # Create per-video subfolder inside output_dir
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(output_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Phase 1: Extract landmarks
        self.extract_landmarks()

        # Fix L/R swaps first, then jitter
        self.fix_lr_swaps()

        # Save raw copies for diagnostic plots, then correct jitter
        raw_lm_x = self.lm_x.copy()
        raw_lm_y = self.lm_y.copy()
        self.correct_landmarks()
        self.generate_jitter_report(output_dir, raw_lm_x, raw_lm_y)

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
