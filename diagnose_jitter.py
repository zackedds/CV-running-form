#!/usr/bin/env python3
"""Diagnostic: analyze foot landmark jitter frame-to-frame."""

import sys
import os
import mediapipe as mp
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_heavy.task")

# Foot landmark indices
FOOT_LANDMARKS = {
    "L_Heel": 29, "R_Heel": 30,
    "L_Toe": 31, "R_Toe": 32,
    "L_Ankle": 27, "R_Ankle": 28,
}

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "original_slowmo.mp4"

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Collect positions: {name: [(x_px, y_px), ...]}
    positions = {name: [] for name in FOOT_LANDMARKS}
    visibilities = {name: [] for name in FOOT_LANDMARKS}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(frame_idx * 1000 / fps)
        results = landmarker.detect_for_video(mp_image, ts)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            lms = results.pose_landmarks[0]
            for name, idx in FOOT_LANDMARKS.items():
                lm = lms[idx]
                positions[name].append((lm.x * width, lm.y * height))
                visibilities[name].append(lm.visibility)
        else:
            for name in FOOT_LANDMARKS:
                positions[name].append((np.nan, np.nan))
                visibilities[name].append(0.0)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{n_frames}...")

    cap.release()
    landmarker.close()
    print(f"Extracted {frame_idx} frames")

    # Compute deltas (pixel distance between consecutive frames)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("Foot Landmark Frame-to-Frame Deltas (pixels)", fontsize=14, fontweight="bold")

    frames = np.arange(frame_idx)
    times = frames / fps

    for ax_idx, (name, idx) in enumerate(list(FOOT_LANDMARKS.items())[:4]):
        pos = np.array(positions[name])
        vis = np.array(visibilities[name])

        # Frame-to-frame pixel distance
        dx = np.diff(pos[:, 0])
        dy = np.diff(pos[:, 1])
        delta = np.sqrt(dx**2 + dy**2)

        ax = axes[ax_idx]
        ax.plot(times[1:], delta, linewidth=0.5, alpha=0.8, label="delta (px)")

        # Mark outlier frames (delta > mean + 3*std)
        mean_d = np.nanmean(delta)
        std_d = np.nanstd(delta)
        threshold = mean_d + 3 * std_d
        outliers = delta > threshold
        if np.any(outliers):
            ax.scatter(times[1:][outliers], delta[outliers],
                       c="red", s=10, zorder=5, label=f"outliers (>{threshold:.1f}px)")

        ax.axhline(mean_d, color="gray", linestyle="--", alpha=0.5,
                   label=f"mean={mean_d:.1f}px")
        ax.axhline(threshold, color="red", linestyle="--", alpha=0.3,
                   label=f"3σ={threshold:.1f}px")
        ax.set_ylabel("Delta (px)")
        ax.set_title(f"{name} (idx {idx}) — mean Δ={mean_d:.1f}px, "
                     f"max Δ={np.nanmax(delta):.1f}px, outliers={np.sum(outliers)}")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("output/foot_jitter_analysis.png", dpi=150)
    print("Saved output/foot_jitter_analysis.png")

    # Also plot raw Y positions (vertical) to see the noise pattern
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig2.suptitle("Foot Landmark Y-Position (vertical, px) — Raw vs Smoothed",
                  fontsize=14, fontweight="bold")

    for ax_idx, (name, idx) in enumerate([
        ("L_Heel", 29), ("L_Toe", 31), ("L_Ankle", 27)
    ]):
        pos = np.array(positions[name])
        raw_y = pos[:, 1]

        # Simple moving average
        kernel = 7
        smoothed = np.convolve(raw_y, np.ones(kernel)/kernel, mode="same")

        ax = axes2[ax_idx]
        ax.plot(times, raw_y, linewidth=0.5, alpha=0.6, label="raw", color="blue")
        ax.plot(times, smoothed, linewidth=1.5, alpha=0.9, label=f"smoothed (w={kernel})",
                color="red")
        ax.set_ylabel("Y position (px)")
        ax.set_title(f"{name}")
        ax.legend(loc="upper right")

    axes2[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("output/foot_position_raw_vs_smooth.png", dpi=150)
    print("Saved output/foot_position_raw_vs_smooth.png")

    # Print summary stats
    print("\n=== JITTER SUMMARY ===")
    for name, idx in FOOT_LANDMARKS.items():
        pos = np.array(positions[name])
        dx = np.diff(pos[:, 0])
        dy = np.diff(pos[:, 1])
        delta = np.sqrt(dx**2 + dy**2)
        mean_d = np.nanmean(delta)
        std_d = np.nanstd(delta)
        threshold = mean_d + 3 * std_d
        outliers = np.sum(delta > threshold)
        print(f"  {name:10s}: mean={mean_d:5.1f}px  std={std_d:5.1f}px  "
              f"max={np.nanmax(delta):6.1f}px  outliers(3σ)={outliers}")


if __name__ == "__main__":
    main()
