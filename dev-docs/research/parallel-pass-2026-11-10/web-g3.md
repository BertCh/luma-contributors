# web-g3 (saved by the lead from the agent's final message; the agent wrote no report file)

Groups: regression-weights, spatial-statistics, terrain-hydrology-flow, terrain-relief-light.
Verified run by run (no full group pass); website typecheck and biome clean for these groups.

## Inert sliders
- multiscale-gwr interceptBandwidth: third map (intercept surface) drawn; slider 6.4%. True intercept is constant; slider shows noise appearing as bandwidth shrinks. Bandwidths rescaled to 0.08-1. Pad gesture = x1 bandwidth by intercept bandwidth; gesture 6.5 -> 4.7%, distinct 2 -> 3.
- terrain-depression-hierarchy stages: GPU kernel writes a default chart; 0.1 -> 10.0%.
- point-horizon-visibility targetHeight: maximumDistance 3600 -> 20000; slider 1.5%; gesture 2.1 -> 7.6%.
- spike-repair jump / tolerance: run now SMOOTHS THE GROUND and gives each block a seam of one step plus up to 30 m (caption says scenario). jump 1.1%, tolerance 2.6%; swipe 1.9 -> 3.1%.
- island-attachment reciprocal: band radius default 0.06 -> 0.04; 0.25 -> 0.5% (still small).
- spatial-two-stage-least-squares instrumentOrder: already 3.5%, left.

## Gestures (max % before -> after; distinct)
- terrain-critical-points 0.5 -> 7.6 (1 -> 3); probit-regression 1.3 -> 14.8; global-spatial-statistics now reach (band radius, 5 settled values) 1.7 -> 5.3 (1 -> 3); global-permutation-test 2.3 -> 4.9; spatial-cross-validated-ols 2.1 -> 5.0; spatial-pearson 2.6 -> 4.0 (default noise 0.25 -> 0.5); pair-gather 1.5 -> 2.6 (thicker); profile-peaks 1.7 -> 3.0 (bigger dots).
- Left under 3%: optics-clustering 1.5, neighborhood-summary 2.3, peak-snap 1.2, losh-residual-permutation-test 2.8, spatial-scan-statistic 2.9, spatial-interaction 2.9.

## WEB-PIC-17
Cross-map links are the run's configuration, not an operation defect: the run shrinks tracts by `gap`, and the old tolerance exceeded the nearest non-touching distance. New defaults: contiguity-weights gap 0.04, snapTolerance 0.018; fuzzy-contiguity gap 0.05, tolerance 0.021. GPU links were not counted against the measured distances.

## Not re-verified after last edit
- pair-gather `how` slider flagged inert again (0.5%); radius default reverted to 0.1 afterwards, not re-run.
- peak-snap not re-run after reverting a spotlight experiment.
- optics-clustering got a type fix after its verification.

## Posters to regenerate
multiscale-gwr, terrain-depression-hierarchy, point-horizon-visibility, spike-repair, island-attachment, contiguity-weights, fuzzy-contiguity, terrain-critical-points, probit-regression, global-spatial-statistics, global-permutation-test, spatial-pearson, pair-gather, profile-peaks, peak-snap, optics-clustering, neighborhood-summary, spatial-cross-validated-ols.

## Shared helper edits inside its groups
regression-weights/shared/regression.ts (unique shifted-map buffer id per offset); terrain-relief-light/support/class-highlight.ts (drawClassDimmer optional color).

## Framework note
The verifier's pointer-change metric compares against the rest picture, which already draws glyphs, so structure shown at rest cannot raise a drag's score.
