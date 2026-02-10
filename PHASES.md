# CV Running Form Analyzer — Build Phases

## Phase 1: Landmark Extraction & Skeleton Overlay
- Run MediaPipe Pose on every frame, extract 33 landmarks
- Draw skeleton overlay on video, write annotated output
- Validate tracking visually

## Phase 2: Ground Contact Detection
- Smoothed heel_y analysis for each foot
- Ground plane estimation (90th percentile)
- Local maxima + velocity sign change detection
- Visualize contacts on annotated video

## Phase 3: Running Form Metrics
- 3a: Foot strike classification (heel/midfoot/forefoot)
- 3b: Knee angle at contact
- 3c: Cadence estimation (with slowmo correction)
- 3d: Trunk lean angle
- 3e: Vertical oscillation

## Phase 4: Annotated Video with HUD
- Semi-transparent metrics panel overlay
- Color-coded contact markers
- Knee angle arc visualization

## Phase 5: Matplotlib Charts
- 2x3 grid: strike timeline, knee angles, trunk lean, vertical oscillation, cadence, strike distribution

## Phase 6: Text Report
- Summary stats + actionable recommendations
