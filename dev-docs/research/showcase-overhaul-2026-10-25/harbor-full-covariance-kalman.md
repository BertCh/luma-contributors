# Harbor track quality: full covariance Kalman assumptions

The harbor track-quality scene keeps its existing isotropic filter and smoother as the default. Its raw quality thresholds (20 kn derived step speed and 300 s interval) are unchanged. A second GPU `GPUTrajectoryKalman` run now evaluates full correlated position and velocity covariance on the same AIS observations, so the scene can compare the two models without substituting CPU estimates.

The full-covariance parameter vector follows the public contributor contract `[qxx, qxy, qyy, rxx, rxy, ryy, vxx, vxy, vyy, maximumGap]`. The defaults are acceleration spectral-density matrix `Q = [[1.5, 0], [0, 1.5]] m²/s³`, position measurement covariance `R = [[2500, 0], [0, 2500]] m²` (50 m standard deviation on each axis), initial velocity covariance `V = [[2500, 0], [0, 2500]] (m/s)²`, and a 300 s model gap. These are illustrative model assumptions, not fitted receiver statistics.

The controls use an x/y variance ratio from 1 to 4 and a shared correlation from -0.8 to 0.8 for Q and R. Each matrix is constructed as `[[base × sqrt(ratio), ρ × base], [ρ × base, base / sqrt(ratio)]]`, so its diagonal variance ratio equals the displayed control and it remains positive definite over the exposed range. Initial velocity covariance stays isotropic. Ratio 1 and correlation 0 are the parity setting; the scene reports the maximum filtered and smoothed position differences between full and isotropic runs over valid states.

Full covariance is stored as four column-major `float32x4` rows per observation. For the projected position submatrix, the scene reads `Pxx`, averages the symmetric `Pxy` and `Pyx` entries, and reads `Pyy`. It derives the two one-standard-deviation axes from that 2×2 matrix and draws the corresponding ellipse at every fourth valid estimate. The uncertainty chart reports the same major/minor position standard deviations in metres. Covariance cross terms are reported in square metres in the nearest-fix tooltip. State residuals compare estimates to the observations used to fit them; neither those residuals nor the ellipses are independent accuracy evidence.

Scoped checks: the showcase story-step check passes for this scene and its WGSL identifier check passes. The scoped TypeScript check reports no diagnostics for the density or harbor files; repository-wide diagnostics remain in unrelated OSM-history, route-windowing, and raster grid-alignment files. GPU/browser execution was not performed in this implementation wave.

## Epoch preparation

The scene now feeds both GPU models with the relative times returned by
`prepareGPUTrajectoryObservations`. Source timestamps are whole seconds since 12 June 2024
00:00 UTC; reconstructing their integer epoch and subtracting the first observation before
float32 conversion preserves all 265 distinct report times. A deliberately naïve comparison
that casts absolute Unix seconds directly to float32 retains only 237 distinct values and
introduces up to 64 seconds of error. No subsecond source precision is invented. The helper’s
ENU position output is not mixed into the existing map: the models retain caller-owned AEQD
coordinates and their original spatial context. This preparation utility is not an estimator.
