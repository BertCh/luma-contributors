# fix-website report

Verifier: `node website/scripts/verify-runs.mjs --port 3301 --out website/.reports/w1-fix{,-a,-b} <ids>`; all 22 runs PASS, restDifference 0 for every run, dark theme only. Screenshots read for: pointcloud-smrf, pointcloud-ground, spike-repair, terrain-critical-points, variogram-model-fit, datum-grid-transform, multiscale-gwr, coverage-validity. Not read (numbers only): nearest-feature-join, linear-referencing, spatial-pearson and the nine reverted runs. Light theme not checked. `npx tsc -p website/tsconfig.json --noEmit`: no errors in my groups (one error in space-time-models, not mine). Biome ignores `website/`, so nothing was formatted; I kept the existing style by hand.

## 1. Most serious
- pointcloud-smrf (H1): tilt removed, caption restored. Picture: real relief in the left surface, small blue band at right (as in 74dddcf5). `window` was inert (0 change) so the slider is removed; window is a fixed 0.3 (`SMOOTHING_WINDOW`), stated in the caption. Sliders: threshold 0.9%, slope 13.9%, cut 7.0%. Gesture 25.8%.
- pointcloud-ground (H4): tilt removed, parameters restored to 74dddcf5 ranges. `slope` and `maximumThreshold` were inert (0%), so both sliders are removed; passed as fixed constants SLOPE 0.1 and MAXIMUM_THRESHOLD 0.8 (>= the largest initialThreshold 0.8, so there is no silent raising and no Math.max). Caption says both are fixed. Remaining sliders: initialThreshold 2.7%, radiusCount 0.4%. Gesture 8.9%.
- spike-repair (H2): six-pass smoothing removed, real tile shown; per-block +-30 m deviation kept and captioned truthfully as a random scenario. `jump` and `tolerance` stayed inert (0%) on the real tile, so both sliders are removed (fixed 200 m and 40 m, stated in the caption); `Data: spikes` (5.6%) remains. Finding: on real relief one damaged block stays visible in the repaired half (operation finding, relates to WEB-PIC-17).
- coverage-validity (H5): back to diameter mapping (gapWidth = 2 * distance, ring radius = distance, rest search at default/2). Seven graded defects kept. Sliders gapWidth 4.6%, gap 8.3%; gesture 0.5 to 2.4%.
- terrain-critical-points (H6): dimmer alpha 0.85 to 0.6; saddle penalty and 5-cell search kept. Other classes (blue, red) and faint elevation stay readable. Gesture up to 7.7%, sliders 6.3% and 3.5%.

## 2. Next tier
- multiscale-gwr: `interceptBandwidth` default 1. Left map is now uniform, middle map drifts, right constant. Sliders 6.8, 4.8, 8.1%; pad gesture 3.9 to 5.8%. All three maps still unlabelled.
- variogram-model-fit: `pairCounts` added to outputs; readout is `N pairs (sample of M)` with N read from the operation's `pairCounts[bin]`. The CPU scan remains only to pick the drawn sample. Gesture 2.8 to 4.8%.
- nearest-feature-join: radius ceiled to the 0.01 step. linear-referencing: radius ceiled to the 0.005 step. Both gestures 1.0 to 3.8%, sliders 3.3% and 3.1%/4.9%.
- pointcloud-statistical-filter: capacity removed from the gesture; settles on standardDeviations only (5 builds), buildCache 5, hint and readout restored. The file is identical to 74dddcf5. Verifier warning "only 2 distinct pictures at 5 positions" also applies to the original (x axis no longer acts).
- datum-grid-transform: marker tests `(status & GPU_DATUM_GRID_TRANSFORM_STATUS.inverseExhausted) != 0u` (value 8, imported from `@luma-contributors/analysis/geospatial-analysis`). No marker visible at the default limit 3 (accepted by the review). Sliders 3.5, 2.8, 2.7%.
- spatial-pearson: noise default back to 0.25 (dots 5.5 kept). Sliders 6.8%, 3.0%.

## 3. Size-only verdicts
Reverted by Edit (each file is again identical to 74dddcf5 except pair-gather's slider range): line-locate, building-tessellation-recipe, group-statistics (3.6), pick-region-mask, network-flow, network-snapping, track-quality, pair-gather (draw sizes only; `radius` max 0.24 kept because it is not a size change), profile-peaks. Accepted ones left alone.

## Rest picture differs from 74dddcf5 (regenerate posters)
Changed by me or still different from 74dddcf5: spike-repair, coverage-validity, terrain-critical-points, multiscale-gwr, variogram-model-fit (hovered-bin pairs at rest), nearest-feature-join, linear-referencing, datum-grid-transform. Other runs in the twelve groups that earlier agents changed still differ; I did not re-judge them.
Rest picture now equal to 74dddcf5 (poster need not change): pointcloud-smrf, pointcloud-ground, pointcloud-statistical-filter, spatial-pearson (apart from dots 5.5), line-locate, building-tessellation-recipe, group-statistics, pick-region-mask, network-flow, network-snapping, track-quality, pair-gather, profile-peaks.

## Not done
Review items M5, M7 captions (label-point, raster-sampling, map-matching hint), M8 notes (quantile-lisa import, terrain-depression-hierarchy kernel, reach doc) were not in the assignment; left untouched.
