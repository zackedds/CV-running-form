# Signal Processing: Landmark Correction

## The Pyramid Wave Pattern

The most important diagnostic plot is `lr_corrected_analysis.png` — the L vs R Heel X (horizontal) position over time. In a correctly tracked running video, each foot traces a **pyramid wave** (triangle wave) pattern:

```
    /\        /\        /\
   /  \      /  \      /  \      ← Left heel X
  /    \    /    \    /    \
 /      \  /      \  /      \
          \/        \/
```

This shape comes from the biomechanics of running:
- The foot moves forward at roughly constant speed during the swing phase (linear ramp up)
- Brief direction change at the front of the stride (peak)
- The foot moves backward relative to the body during stance (linear ramp down)
- Brief direction change at the back (valley)

The left and right feet are **180 degrees out of phase** — when one is at the front of the stride, the other is at the back. The two pyramid waves should interleave cleanly without crossing identities.

## What Goes Wrong (and How We Fix It)

### 1. Landmark Jitter (High-Frequency Noise)

**Symptom**: Individual landmarks teleport to wrong positions for 1-3 frames, especially in slowmo video where inter-frame motion is small and the model loses confidence.

**Detection**: Compute frame-to-frame pixel delta per landmark. Fit a baseline from the median delta, then use MAD (median absolute deviation) to set an adaptive threshold: `threshold = median + 3 * MAD`. For slowmo running video this typically gives 3-5px thresholds against a ~1.7px baseline.

**Correction**: Iterative delta-based cleaning:
1. Flag destination frames of any delta exceeding the threshold
2. NaN out flagged frames, linearly interpolate from valid neighbors
3. Recompute deltas on cleaned data
4. Repeat until no new outliers (typically converges in 2-3 passes)
5. Apply a light moving average (window=5) as final smoothing

### 2. Left/Right Identity Swaps

**Symptom**: MediaPipe swaps which foot is "left" vs "right". The feet smoothly cross to each other's position, so per-landmark deltas stay small — jitter correction cannot catch this. Visible as the pyramid waves suddenly inverting.

**Detection**: Monitor the L-R heel X difference signal. Compute its frame-to-frame delta. A real swap causes a ~200px jump in the L-R difference in a single frame (both feet teleport to each other's position simultaneously). Natural foot crossing during stride changes the difference by only a few px per frame.

**Algorithm**:
1. Compute `diff = L_heel_x - R_heel_x` for each frame
2. Compute `diff_delta = diff[i+1] - diff[i]` (rate of change of L-R separation)
3. Flag frames where `|diff_delta| > 80px` as swap events
4. Group nearby events (within 30 frames / ~1s) into swap regions
5. For each region, compute a reference L-R diff from the 10 frames before the region
6. Within the region, mark frames where the diff sign disagrees with the reference AND magnitude > 30px
7. Swap all paired L/R landmarks (hip, knee, ankle, heel, toe, shoulder) on marked frames

**Why 80px threshold**: In side-view running video at 640px width, the L-R heel separation oscillates between roughly -100px and +100px during normal stride. A swap causes this to jump ~200px in one frame. Natural stride-to-stride variation changes by <10px per frame.

## Processing Order

The corrections must run in this order:
1. **L/R swap fix** — must run first because swaps create false high-deltas that would confuse jitter correction
2. **Jitter correction** — runs on swap-corrected data, catches remaining noise
3. **Smoothing** — final pass within jitter correction

## Validation

After correction, verify `lr_corrected_analysis.png`:
- Heel X plot should show clean pyramid waves, L and R alternating 180° out of phase
- Heel Y plot should show alternating ground contacts (high Y = foot on ground)
- L-R difference should maintain consistent sign (no flips = no remaining swaps)

If any of these are violated, the correction thresholds may need tuning for the specific video.
