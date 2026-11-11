# Math correctness review: shared numerics and point cloud (`core-numerics`)

Date: 2026-11-10. Scope: `modules/analysis/src/analysis-core` and
`modules/analysis/src/pointcloud-analysis`. Paths are relative to `modules/analysis/src` unless they
start with `docs/`, `node_modules/` or `modules/`. Line numbers are those of the working tree on
the review date; other sessions were editing nine of the scope files, so re-grep before acting.

Scratch scripts are in the session scratchpad under `core-numerics/` (`philox.mjs`,
`shuffle-wrap.mjs`, `quantile.mjs`, `chi2.mjs`, `icp-plane.mjs`, `sphere.mjs`). They are not kept
in the repository. No spec file was run.

## 1. Coverage

Read in full (56 of the 84 scope files):

- `analysis-core/kernels`: `philox-random.ts`, `permutation-random.ts`, `permutation-shuffle.ts`,
  `hash-uint32.ts`, `fixed-point-accumulator.ts`, `uint64-wgsl.ts`, `exact-orientation-wgsl.ts`,
  `exact-lattice-wgsl.ts`, `exact-geometry-wgsl.ts`, `dense-solve-wgsl.ts`,
  `symmetric-eigen-wgsl.ts`, `binary-search-wgsl.ts`, `sphere-distance-wgsl.ts`,
  `incomplete-gamma-wgsl.ts`, `normal-quantile-wgsl.ts`, `wgsl-poisson-observed-term.ts`,
  `wgsl-theil-sen-median.ts`, `sorted-segment-sums.ts`, `sorted-segment-offsets.ts`,
  `scan-total.ts`, `wgsl-common.ts`, `stable-lexicographic-index-sort.ts`,
  `masked-weak-components.ts`, `parallel-merge-tree.ts`, `study-window-wgsl.ts`,
  `gpu-prediction-utils.ts`.
- `analysis-core/contracts`: `earth.ts`, `spatial-context.ts`, `local-frame.ts`,
  `spatial-frame.ts`, `spatial-frame-crs.ts`, `temporal-context.ts`, `inference.ts`,
  `vertical-datum.ts`, `study-window.ts`.
- `analysis-core` root and others: `cell-key-wgsl.ts`, `lon-lat-ring-wgsl.ts`,
  `spherical-cell-area.ts`, `raster-grid.ts`, `graph-support/arrow-analytics-time-upload.ts`,
  `graph-support/arrow-analytics-wide-numeric.ts`, `ring-assembly/ring-vertex-quantizer.ts`.
- `pointcloud-analysis`: `gpu-pointcloud-neighborhoods.ts`, `gpu-pointcloud-features.ts`,
  `gpu-pointcloud-statistical-filter.ts`, `gpu-pointcloud-change.ts`, `gpu-pointcloud-icp.ts`,
  `pointcloud-plane-icp.ts`, `gpu-pointcloud-ground.ts`, `gpu-pointcloud-smrf.ts`,
  `gpu-pointcloud-surfaces.ts`, `gpu-pointcloud-voxel-merge.ts`, `pointcloud-grid-index.ts`,
  `pointcloud-local-origin.ts`, `pointcloud-kernel-nodes.ts`, `index.ts`.

Read in part (3 files): `kernels/load-balanced-for-each.ts` (lines 1-60, 97-180, 290-469: the
merge-path search and walk, not the segment-sum scan and fixup), `kernels/polygon-edge-slabs.ts`
(lines 1-140: the invariant and layout, not the build or query kernels),
`graph-support/arrow-analytics-normalization.ts` (lines 150-379: Float64 policy, categories, time
words).

Not opened (25 files, not verified): `kernels/gpu-contributor-types.ts`,
`kernels/gpu-contributor-utils.ts`, `kernels/wgsl-kernel-nodes.ts`, `kernels/wgsl-kernel-tiles.ts`,
`kernels/row-access-wgsl.ts`; `contracts/capacity-planning.ts`, `contract-names.ts`, `index.ts`,
`model-result.ts`, `parameter-schema.ts`, `partitioning.ts`, `ports.ts`, `status.ts`;
`graph-support/index.ts`; `analysis-core/index.ts`; `network-topology.ts`; all four files of
`polygon-triangulation/` (3,499 lines); `ring-assembly/gpu-segment-ring-assembly.ts`,
`ring-assembly/index.ts`, `ring-assembly/segment-polygonization-diagnostics.ts`;
`pointcloud-analysis/pointcloud-attribute.ts`, `pointcloud-limits.ts`. Most of these are plumbing,
but `polygon-triangulation` and `gpu-segment-ring-assembly.ts` hold geometry math that nobody in
this review has checked.

Outside the scope, read only to compare a private copy with the shared one:
`spatial-regression/spatial-regression-solve.ts` (1-110),
`spatial-interpolation/equilibrated-solve-wgsl.ts`, `geometry-measures/geodesic-wgsl.ts` (100-150,
210-270), and the lines quoted in section 4.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| core-numerics-01 | wrong-edge-case | CONFIRMED (Node replica of the kernel arithmetic, `icp-plane.mjs`) | `pointcloud-analysis/pointcloud-plane-icp.ts:13-16`, `:39-45` | Point-to-plane ICP forms the rotational Jacobian `cross(position, normal)` about the coordinate origin and rejects a pivot at or below `1e-7` times the largest diagonal, with no scaling | Low (2004) linearisation; PCL and Open3D solve it in float64. In float32 the rotation and translation columns must be decoupled (centre on the matched centroid) and the rank test must be scale free | A 10 m room corner (three orthogonal faces, 3,000 points, 2 cm shift) solves at the origin, but 500 m from the origin returns status 3 `DEGENERATE` (pivot 0.065, threshold 34). Centred with a 10 km extent, or a 30 m scene in millimetres: also `DEGENERATE` | Take the Jacobian about the weighted centroid of the matched source points and convert the translation back; solve with the shared `getDenseLUWGSL` with equilibration |
| core-numerics-02 | wrong-edge-case | CONFIRMED (u32 arithmetic by hand, and an 8-bit analog, `shuffle-wrap.mjs`) | `analysis-core/kernels/permutation-shuffle.ts:145` | WGSL `swapOrNotKeys[round] + count - position` in `u32`. For `count > 2^31` the sum wraps | The TypeScript twin (`:77`) adds in float64 and does not wrap. The header says `PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS` serves "any uint32 domain" | `count = 3,000,000,000`, key `2,000,000,000`, position 5: WGSL partner `705,032,699`, twin `1,999,999,995`. The round is then not an involution, so the result is not a permutation (analog with 8-bit words: `count = 200` is never a bijection in 2,000 trials, `count = 129` fails in 40%) | `partner = select(key + (count - position), key - position, key >= position)`; add a spec at `count > 2^31` |
| core-numerics-03 | precision | CONFIRMED (hand arithmetic) | `analysis-core/cell-key-wgsl.ts:9`, `:53-60` | `cellScaleValue` rounds every row to a multiple of `1 / sumScale` (default `1 / 65536`) before the 64-bit sum | `docs/api/gpu-dataframe-analysis.md:710` calls these sums "exact"; a user expects float32 accuracy of a sum or mean | A column of constant `1e-4`: `round(6.5536) = 7`, so the group mean is `1.0681e-4` (+6.8%). Any value below `7.6e-6` contributes 0 | Use the data-scaled 96-bit accumulator (`fixed-point-accumulator.ts`) for `GPUGroupStatistics`, `GPUKeyJoin` and the cell tables; until then, document the resolution next to "exact" |
| core-numerics-04 | precision | CONFIRMED (against the AS 241 table in the CPython `statistics` source, `quantile.mjs`) | `analysis-core/kernels/normal-quantile-wgsl.ts:21`, `:29`, `:34` | Three AS 241 coefficients are mistyped: `2509.0859297301226727`, `4.6303378461565456959`, `0.59983322206588879377` | Wichura (1988): `2509.0809287301226727`, `4.6303378461565452959`, `0.59983220655588793769` | Evaluated in float64, the far tail gives `-7.03448225` for `p = 1e-12` (true `-7.03448383`, relative error 2.2e-7) and 6.6e-7 at `p = 1e-30`. The central typo moves the float64 result by at most 3e-11; the third has no effect | Correct the digits. Add a spec: no test references this kernel |
| core-numerics-05 | inconsistent | CONFIRMED (`Math.fround(16777215 + 0.5) = 16777216`) | `gpu-spatial-analysis/scan-statistics/expectation-based-scan-poisson-wgsl.ts:15`; `spatial-interpolation/gpu-conditional-gaussian-simulation.ts:625`; `spatial-interpolation/conditional-simulation-blocked.ts:436`; `spatial-clustering/gpu-kmeans.ts:744` | `(f32(word >> 8u) + 0.5) / 16777216.0`, meant as a uniform in the open interval `(0, 1)` | The shared `philoxUnitFloat` (`philox-random.ts:138`) is `[0, 1)`. In float32 `k + 0.5` is not representable for `k >= 2^23`: it rounds to `k` or `k + 1` | `word >> 8 = 16777215` gives exactly `1.0` (once in 2^24 draws), so `log(u) = 0`: a Box-Muller normal of exactly 0, and a k-means++ race key of `-0.0` (bits `0x80000000`), which can never win an `atomicMin`, where the intended draw would almost surely win | Add one open-interval helper to core, for example `(f32(word >> 9u) + 0.5) * (1.0 / 8388608.0)` (exact in float32), and use it in the four files |
| core-numerics-06 | inconsistent | CONFIRMED (hand derivation; PCL and PDAL formulas from memory, not run) | `pointcloud-analysis/gpu-pointcloud-ground.ts:189-193` | Threshold of stage k is `initial + slope * cellSize * (radius_k - radius_(k-1))`, with `radius_(-1) = 0` | Zhang et al. (2003) eq. 7, PCL and PDAL `filters.pmf`: `initial` alone at the first stage, then `slope * (window_k - window_(k-1)) * cellSize + initial` with full window widths `2 r + 1` | `radii [1, 2]`, cell 1, slope 0.1, initial 0.2: the kernel uses 0.3 and 0.3; the reference gives 0.2 and 0.4. The same `slope` is half as permissive after the first stage | `docs/api/pointcloud-analysis.md:254` states the kernel's formula, so this is a convention. Say in the class doc that `slope` is per radius, not per window, or adopt the reference form |
| core-numerics-07 | doc-mismatch | CONFIRMED (published `.d.ts`) | `docs/api/pointcloud-analysis.md:219-224`; `pointcloud-analysis/gpu-pointcloud-voxel-merge.ts:30-37` | `GPUPointCloudVoxelMerge` merges per-part `centroids` and `attribute` means | The docs say `GPUPointCloudVoxelDownsample` has optional `centroids` and `attribute`. Published alpha.10 `GPUPointCloudVoxelDownsampleProps` has neither (`node_modules/@luma.gl/graph/dist/gpu-pointcloud/gpu-pointcloud-voxel-downsample.d.ts`) | Nothing in this repository or in published luma.gl produces the `centroids` input. Passing the published `output` (the representative point) gives a count-weighted mean of representatives, not a centroid | Remove the paragraph or mark it as not in the published release; propose `centroids` upstream |
| core-numerics-08 | doc-mismatch | CONFIRMED (`sphere.mjs`) | `analysis-core/kernels/sphere-distance-wgsl.ts:22-23` | `getSphereAngleFromHaversineTerm` is `2 atan2(sqrt(h), sqrt(1 - h))` | The comment says it keeps precision near antipodes "unlike `2 asin(sqrt(h))`". Both read the same rounded `h`, so both lose it | Angle `pi - 3e-4`: `h` rounds to 1.0 and both forms return `pi`, 1.9 km off on Earth. At `pi - 1e-3` both are 150 m off | Fix the comment. For antipodal accuracy use the vector form of `geodesicCentralAngle` |
| core-numerics-09 | doc-mismatch | CONFIRMED (reading) | `analysis-core/kernels/normal-quantile-wgsl.ts:12`, `:16-17` | Returns `-1e15` for `p <= 0` and `1e15` for `p >= 1` | The comment says `-inf` and `+inf` | `p = 0` gives `-1e15`, which is finite | Say "a large finite value" in the comment. The one caller clamps to 8.21 |
| core-numerics-10 | doc-mismatch | CONFIRMED (reading) | `pointcloud-analysis/gpu-pointcloud-features.ts:38`, `:119`; `gpu-pointcloud-icp.ts:75`, `:357` | 24 and 48 single Jacobi rotations (one pivot each) | The status docs say "24-sweep" and "48-sweep" | None: classical Jacobi on 3x3 and 4x4 converges well inside these counts | Say "rotations" |
| core-numerics-11 | untested | n/a | `analysis-core/kernels/normal-quantile-wgsl.ts`, `wgsl-theil-sen-median.ts`, `wgsl-poisson-observed-term.ts` | See section 6 | A direct spec would have caught core-numerics-04 | No spec names these kernels | One node spec each against a float64 formula |

### Notes on the findings

**core-numerics-01.** The point-to-point path is not affected: Horn's method centres both clouds on
their means. In the point-to-plane path the rotation columns `p x n` of a cloud at distance `c`
from the origin are almost linear combinations of the translation columns, so after eliminating
them the translation pivots hold only rounding noise. At 500 m the pivot is 0.065 against matrix
entries near 3e8, so the float32 system is really unusable there, not only mis-flagged. At a few
kilometres of extent around the origin the system is fine but the unscaled threshold rejects it.
The failure is loud (status 3, last good transform kept), not a silent wrong transform. The spec
(`modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts:195`) uses 12 points within
3 units of the origin. The `origin` prop documentation says absolute float32 positions lose
millimetres "beyond ~1e5 m", which understates when this metric needs a centred frame.

**core-numerics-02.** The path is reached by `GPUSpatialScanStatistic`
(`gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:1088`, `:1105`, `:1115`),
whose props allow a total population "below 2^32" (`:161`). A population between 2^31 and 2^32
draws labels that are not a sample without replacement, and the CPU oracle no longer matches. The
other five callers use the default 150 rounds (domains below 2^24) and are safe.

**core-numerics-03.** The library review (3.8) already said there are three fixed-point schemes
and recommended the 96-bit one. It did not state the accuracy consequence, which is why it is
listed here. The arithmetic inside `cellScaleValue` is itself correct (round half even, exact
split into words, saturation at `2^62 - 2^38`).

**core-numerics-04.** In float32 the central typo changes 29 of 84,901 sampled outputs by one ulp.
The far-tail typo is reachable by the only caller, `GPURasterStandardizedIndex`, for indices
between 6.7 and its 8.21 clamp. The kernel comment attributes "about 5e-7 relative error" to
float32 input precision; part of it is this coefficient.

**core-numerics-05.** The files are in other reviewers' scopes; the finding is that core offers no
open-interval conversion, so four kernels wrote the same one that does not do what it says. The
Poisson sampler tolerates `u = 1` (the PTRS branch rejects the draw). In the k-means++ race the
row that drew the top value is excluded for that round instead of being the likely winner. All
four effects are rare (2^-24 per draw); the point is the convention, not the size.

## 3. Verified correct

Compared against the definition, by a Node replica where a script is named, otherwise by hand
derivation.

- `philox-random.ts`: Philox4x32-10 multipliers, Weyl constants, round function, key schedule (ten
  rounds, nine bumps). The three Random123 known-answer vectors (zeros, ones, pi digits) match;
  the 16-bit-limb multiply matches BigInt on 200,005 pairs (`philox.mjs`). `philoxUnitFloat` is
  the top 24 bits over 2^24: largest value `1 - 2^-24`, never 1.0.
- `philox-random.ts` purpose registry: key streams differ in key word 1, counter streams in
  counter word 3, and the two kinds differ in counter word 3. No two purposes can share a block.
- `permutation-random.ts`: `nextBelow` is Lemire's multiply-shift with the correct rejection
  threshold `(2^32 - n) mod n`; WGSL and TypeScript agree.
- `permutation-shuffle.ts` for `count <= 2^31`: swap-or-not rounds are involutions; all `n!`
  permutations of `n = 2..7` appear with chi-square z between -1.3 and -0.6; pairs for `n = 10, 37`
  are uniform; the WGSL `countLeadingZeros` round count equals the TypeScript `ceil(log2)` for
  every `n < 100,000` and around every power of two (`philox.mjs`).
- `hash-uint32.ts`: lowbias32 constants and shifts as published by Wellons.
- `fixed-point-accumulator.ts`: unit exponent bounds the sum below 2^95; quantisation is round
  half even including the `shift = -24` tie; the atomic three-word add is exact for any
  interleaving (carry count of a wrapping counter is order independent); decode rounds to nearest
  even and overflows to infinity only above the largest float32. NaN and infinite terms contribute
  0, as documented.
- `uint64-wgsl.ts`: add and subtract carries, both compares, shifts at 0, 32 and 64, the 32x32
  product, and the restoring division with the overflow bit.
- `exact-orientation-wgsl.ts`: the six-term expansion equals `orient2d`; `multiply24` and the
  shifted 256-bit accumulate cannot overflow for a span of 200 (products below 2^248, six terms
  below 2^251); the float filter constant 3e-7 is above Shewchuk's 1.8e-7 bound and strict
  comparisons send underflow, overflow and NaN to the exact path.
- `exact-lattice-wgsl.ts`: crossing numerator and denominator signs; all products stay below 2^70
  for coordinates up to 2^22 + 1 (limit 2^80); the rounded crossing is the largest `k` with
  `k - 1/2 <= x`, found in at most 23 of the 24 candidate steps.
- `exact-geometry-wgsl.ts`: the simulation-of-simplicity tie rules for point and segment sides and
  the ray straddle rule, derived from the perturbation `(-e^2, -e)`.
- `dense-solve-wgsl.ts`: Cholesky with symmetric equilibration (solve, inverse diagonal, log
  determinant) and LU with partial pivoting and Ruiz scaling (solve, determinant sign and log).
- `symmetric-eigen-wgsl.ts`: rotation formulas (small root, overflow guard), convergence test,
  ascending sort with vector columns. Eigenvector sign is not fixed, and is not documented as fixed.
- `binary-search-wgsl.ts`: lower and upper bound, empty range, ordered float key.
- `incomplete-gamma-wgsl.ts`: series, modified Lentz fraction, Stirling log-gamma, Wilson-Hilferty
  switch. Float32 replica against closed forms for 49 degrees of freedom from 1 to 20,000
  (`chi2.mjs`): up to 500, relative error at most 1.0e-4 for p above 1e-6; up to 2,000, at most
  6.7e-4; above 2,000, absolute error at most 5.8e-6. At most 162 of the 500 iterations are used.
  The documented 6e-5 is slightly optimistic in the replica (which uses JavaScript `log` and `exp`).
- `wgsl-poisson-observed-term.ts`: Taylor coefficients `1 / (k (k - 1))` to `k = 14` with the right
  signs; the identity with `o ln(o / e) - o + e`.
- `wgsl-theil-sen-median.ts`: the heap of the `pairs / 2 + 1` smallest slopes gives the exact
  median for odd and even pair counts.
- `sorted-segment-sums.ts`, `sorted-segment-offsets.ts`, `scan-total.ts`,
  `stable-lexicographic-index-sort.ts`: fixed-order tree, lower-bound offsets, carry detection,
  key packing order and LSD pass order.
- `study-window-wgsl.ts`: the circle fraction and the Epanechnikov, quartic and Gaussian kernel
  masses, derived from the fan decomposition (`(1 - s^2)^2`, `(1 - s^2)^3`, `exp(-s^2 / 2)` tail
  functions); the 16-point Gauss-Legendre nodes and weights.
- `contracts/earth.ts`: IUGG mean radius, WGS84 authalic radius, semi-major axis, flattening.
- `contracts/spatial-context.ts`: radius selection by unit; the float32 longitude quantum 2^-16
  degrees.
- `contracts/local-frame.ts`: spherical azimuthal equidistant on the mean radius and ellipsoidal
  Lambert equal-area, delegated to math.gl; the stated 0.1% distance error at 200 km is
  conservative (0.016%).
- `raster-grid.ts`: geographic metres per degree (`N cos(lat)`, `M`) and Web Mercator ground scale
  (`N cos(lat) / a`, `M cos(lat) / a`), row-centre edge interpolation, point-pixel half-row shift.
- `spherical-cell-area.ts`: `2 R^2 dLon cos(mid) sin(dLat / 2)`, row mirroring for north-up grids.
- `graph-support`: time words from Arrow 17 `BigInt64Array` values, day dates times 86,400,000,
  the Float64 error budget.
- `GPUPointCloudFeatures`: centred population covariance about a local anchor, smallest
  eigenvector as the normal, viewpoint flip, linearity, planarity and scattering as CloudCompare
  defines them. Surface variation `l3 / (l1 + l2 + l3)` is not an output.
- `GPUPointCloudStatisticalFilter`: mean neighbour distance, centred variance, mean plus k
  standard deviations, inclusive threshold.
- `GPUPointCloudChange`: M3C2 cylinder test, Welford moments, sample variance, level of detection
  `1.96 (sqrt(s1^2 / n1 + s2^2 / n2) + reg)`, the correlated-error variant, the cylinder bounding
  box `h |n| + r sqrt(1 - n^2)`.
- `GPUPointCloudICP` point-to-point: weighted running co-moments and block merge, Horn's 4x4
  matrix, quaternion to rotation, translation, composition, Huber and Tukey weights.
- `pointcloud-plane-icp.ts`: Jacobian `[p x n, n]`, normal equations and Rodrigues update are
  right; only the conditioning is not (core-numerics-01).
- `GPUPointCloudSMRF`: stages as PDAL 2.8.4 (low outliers at 5 cells, net cut, progressive
  diamond openings with threshold `slope * cellSize * radius`, eight-nearest inpainting with a
  correct ring cutoff, gradient, `threshold + scalar * gradient`).
- `GPUPointCloudSurfaces`, `GPUPointCloudHeight`, `GPUPointCloudVoxelMerge`,
  `pointcloud-local-origin.ts`, `pointcloud-grid-index.ts`: cell minimum and maximum, bilinear
  and IDW sampling, weighted mean merge, `T_local = S^-1 T S`, sort key width.

Read but not verified beyond reading: `masked-weak-components.ts`, `parallel-merge-tree.ts`,
`lon-lat-ring-wgsl.ts`, the parts of `load-balanced-for-each.ts` and `polygon-edge-slabs.ts`
listed above.

## 4. Standards consistency

**Shared numerics that no contributor uses.** Contract section 11 names them, but these have no
importer outside `analysis-core` today (grep on the working tree):

| Shared helper | Importers outside core | What contributors use instead |
| --- | --- | --- |
| `getDenseCholeskyWGSL` | 0 | `getCholeskyWGSL` (`spatial-regression/spatial-regression-solve.ts`, 20 files): same loop order, so bit-identical to the shared one with `equilibrate = false`; it reports only a boolean |
| `getDenseLUWGSL` | 0 | `getEquilibratedSolveWGSL` (`spatial-interpolation/equilibrated-solve-wgsl.ts`, 6 files): same elimination and pivot tests, but one scale vector updated row by row, where the shared one updates row and column scales together. Same solution, not bit-identical. Also `getEquilibratedCholeskyWGSL` (5 files, not read), `solveSystem` in `gpu-kriging.ts:638` and `gpu-spatial-two-stage-least-squares.ts:849`, `solveMatrix` in `gpu-spatial-regression-diagnostics.ts:711`, `factorInformation` in `gpu-generalized-geographically-weighted-regression.ts:305`, and the 6x6 Gauss-Jordan of `pointcloud-plane-icp.ts:39-46` |
| `getSymmetricEigenWGSL` | 0 | Private Jacobi in `gpu-pointcloud-features.ts:117-132` (3x3) and `gpu-pointcloud-icp.ts:355-369` (4x4): classical largest-pivot Jacobi with `atan2`, relative tolerance 1e-6; mathematically equivalent, different rotation order. Also `getSymmetricEigenvalueRangeWGSL` in `spatial-regression-solve.ts` |
| `getBinarySearchWGSL` | 0 (one use inside core) | 43 files define a named lower or upper bound function and about 70 files hold a `low < high` loop, including `pointcloud-grid-index.ts:150` in this scope. Not compared one by one |
| `SPHERE_DISTANCE_WGSL` | 0 | `GEODESIC_WGSL` (37 files) is the working shared module. Inline haversine terms remain in `spatial-interpolation/separation-wgsl.ts:91`, `spatial-join/nearest-wgsl.ts:425`, `neighbor-search/neighbor-search-kernels.ts:762`, `geometry-measures/geodesic-wgsl.ts:266`, `gpu-terrain/point-horizon/gpu-point-horizon-visibility.ts:196` (all `2 asin(sqrt(clamp(h)))`, same value as the shared form) and `geometry-measures/geometry-measures-kernels.ts:139` (`atan2` form without the antipode guard) |
| `UINT64_WGSL` | 0 | Add-with-carry written out per file, as the library review listed |

**Two word orders for 64-bit integers.** `cell-key-wgsl.ts:35` is `vec2u(high, low)`;
`uint64-wgsl.ts` and the time words are `(low, high)`.

**Error function and chi-square copies.** The Numerical Recipes `erfcc` fit appears twice inside
core (`wgsl-common.ts:155`, `incomplete-gamma-wgsl.ts:59`) and once outside
(`space-time-statistics/gpu-space-time-cross-correlation.ts:218`); coefficients are identical.
Chi-square survival is shared by five files and re-implemented in
`pair-statistics/gpu-point-pattern-indices.ts:112-200` (same algorithm, other tolerances and a
second Wilson-Hilferty switch at 100 degrees of freedom),
`spatial-regression/gpu-ordinary-least-squares.ts:949` (same series, log-gamma from the CPU) and
`spatial-regression/gpu-spatial-regression-diagnostics.ts:696` (closed forms for 1 and 2 degrees
of freedom, which are the only values its callers pass). Log-gamma is also copied in
`raster-analysis/change-detection/change-detection-wgsl.ts:31` (same series as the shared one).

**Random numbers that are not Philox.**

| File:line | Generator | Compared with the shared stream |
| --- | --- | --- |
| `spatial-clustering/gpu-kmeans.ts:728` | `hashUint32(row ^ roundSeed)` | Uses the shared hash as a generator. Two rounds draw the same multiset of values in a different row order, so rounds are not independent |
| `minimum-bounds/gpu-minimum-bounds.ts:576` | PCG-style `hashPair` | Different generator, no registered purpose |
| `gpu-network/network-optimization/gpu-network-louvain.ts:171-174` | xorshift32 seeded with `seed ^ 0x9e3779b9`, `% remaining` | Modulo bias; a seed of `0x9e3779b9` gives state 0, which never leaves 0 |
| `delaunay-tessellation/gpu-delaunay-tessellation.ts:493` | murmur3 `fmix32` priorities | A bijective hash, used as an order, not as a distribution |

Hashes for tables, not randomness (no defect seen): `ring-assembly/ring-vertex-quantizer.ts:95`,
`map-matching/gpu-map-matching.ts:1458`, `spatial-regionalization/gpu-azp-regions.ts:826`,
`spatial-regionalization/ward-active-graph.ts:236`, `column-profile/column-profile-kernels.ts:152`.
No `fract(sin(...))` and no LCG were found in WGSL.

**Unit-float conversion.** See core-numerics-05. `gpu-network/network-k-function/gpu-network-k-function.ts:617`
uses the same expression as `philoxUnitFloat` inline.

**Earth constants.** Every radius and ellipsoid value in `modules/analysis/src` comes from
`contracts/earth.ts`, with one exception:
`gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:20-21` writes the
WGS84 axes as literals (`6378137`, `6356752.314245179`; the values agree with `earth.ts`).
`6371000` appears only in comments that explain parity with other tools.

**Degrees of freedom.** `GPUPointCloudStatisticalFilter` uses the population standard deviation;
PDAL, PCL and Open3D use `n - 1`. The docs say so. `GPUPointCloudChange` uses the sample variance.
`GPUPointCloudFeatures` uses the population covariance (the eigenvectors do not depend on it).

**Self in the neighbourhood.** `GPUPointCloudNeighborhoods` excludes the query point, so features
are computed from the neighbours only. PCL and PDAL include the point. The docs note it for the
PDAL normal fixture.

## 5. luma.gl official implementations

- `@luma.gl/graph/gpu-core` publishes no random-number, hash, dense-solver, eigen or
  segmented-reduction helper, so Philox, lowbias32, the dense solvers and `sorted-segment-sums`
  duplicate nothing. `@luma.gl/shaders` `random` is `fract(sin(dot(...)))`; the local comment in
  `philox-random.ts:8` describes it correctly. The shaders `fp64` module is not used locally and
  nothing local re-implements it.
- `GPUHaversineDistance` (`@luma.gl/graph/geospatial`): a small-angle planar branch, the `asin`
  haversine below `h = 0.5` and the vector `atan2` form above. The local `geodesicCentralAngle`
  uses the vector form everywhere. The two agree mathematically. The published default radius is
  `6371` (kilometres); `earth.ts` gives `6371.0088`, a 1.4e-6 relative difference. No local
  contributor instantiates the published class (it is a node, not a WGSL function), so no
  argument is passed with a changed meaning.
- `GPUSinusoidalProjection`: the local `GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE` equals the published
  `40000 / 360`.
- `@luma.gl/graph/gpu-pointcloud`: no file under `modules/analysis/src` imports it. The local
  package does not re-implement voxel downsampling or the octree. `GPUPointCloudNeighborhoods`
  overlaps `GPUPointCloudRadiusFilter`: both count other points within an inclusive radius with a
  float32 squared threshold, so the counts agree; the published one is all pairs, the local one is
  grid indexed. A radius outlier mask could be offered locally as `counts >= minimum` without new
  math. `GPUPointCloudVoxelMerge` depends on inputs the published downsample does not produce
  (core-numerics-07).
- `pointcloud-grid-index.ts` beside `GPUGridIndex`: the file lists six missing capabilities; the
  reasons hold (exact `origin + k * cellSize` cells, half-open domain, stable order in a cell).
- `sorted-segment-offsets.ts` beside `GPUGallopingSearch` and `GPUSegmentOffsets`: the published
  `GPUSegmentOffsets` converts start flags to list offsets and is not a substitute.
- `scan-total.ts` beside `GPUScanUint64`: the local helper saturates a `u32` total; the published
  64-bit scan would give the true total where a contributor needs more than a saturation flag.
- `masked-weak-components.ts` beside the published connected components: same 32-iteration
  default; the reason for the local copy (a masked CSR) is stated in the file.

## 6. Test oracles

Independent oracle:

- Philox: Random123 vectors in
  `modules/analysis/test/gpu-spatial-analysis/permutation-inference/permutation-random.node.spec.ts`
  and in three Python generators.
- Fixed-point accumulator: BigInt model (`test/utils/fixed-point-oracle.ts`).
- Exact orientation and geometry: BigInt reference
  (`test/gpu-spatial-analysis/segment-intersection/exact-geometry-reference.ts`).
- `uint64-wgsl.ts`, `binary-search-wgsl.ts`: BigInt (`test/utils/wgsl-words-search.spec.ts`).
- Dense solve and symmetric eigen: float64 CPU references (`test/utils/dense-solve-wgsl.spec.ts`,
  `symmetric-eigen-wgsl.spec.ts`).
- Incomplete gamma: SciPy values (`distribution-dynamics/incomplete-gamma.spec.ts`).
- Sphere distance: float64 (`test/utils/wgsl-sphere-rows.spec.ts`).
- Local frame: Python parity generators.
- Point cloud: PDAL 2.8.4 fixtures for SMRF, M3C2, normals, point-to-point ICP and the outlier
  filter (`test/gpu-pointcloud/fixtures`, `pdal-m3c2-upstream-golden.json`).

Own formula or hand-made expectation only:

- `GPUPointCloudGround`: a hand-made expectation in `gpu-pointcloud-analysis.spec.ts`; no PCL or
  PDAL `filters.pmf` fixture.
- Point-to-plane ICP: recovery of a synthetic transform near the origin.
- Swap-or-not shuffle: uniformity and the CPU twin, never above 2^24 rows.

No spec found by name:

- `normal-quantile-wgsl.ts`, `wgsl-theil-sen-median.ts`, `wgsl-poisson-observed-term.ts`,
  `masked-weak-components.ts`, `parallel-merge-tree.ts`, `hash-uint32.ts`. They may be exercised
  through their consumers; that was not checked.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis/src` unless they start with `docs/` or `modules/`. Specs are
under `modules/analysis/test`. Every listed spec was run on the headless WebGPU device (or in Node
for `*.node.spec.ts`) and passes; specs written before a fix were first run against the old code
and failed there, as noted.

| ID | Outcome | What changed (`file:line`) | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| core-numerics-01 | fixed | `pointcloud-analysis/pointcloud-plane-icp.ts:41` (each 256-row block takes the rotational Jacobian about its own weighted centre), `:79-90` (the solve moves every block to the centroid of all matched points with the exact transfer `J = [a + d x n, n]`), `:92-94` (the system is scaled to a unit diagonal; a diagonal that is not positive is `DEGENERATE`), `:97` (pivot limit 1e-5 on the scaled system), `:108-112` (Rodrigues update without the `1 - cos` cancellation, composed back to the origin frame). Block moments grow from 45 to 48 words (`gpu-pointcloud-icp.ts:235`). Class documentation and `docs/api/pointcloud-analysis.md` "Point-to-plane registration" | `gpu-pointcloud/gpu-pointcloud-icp-plane-conditioning.spec.ts` (the report's three scenes with 3,000 points, a rotation and a translation, expected transform from the construction in float64; all three returned status 3 on the old code on the GPU; plus a single tilted plane that must stay `DEGENERATE`) | Point-to-plane runs that returned status 3 with the identity transform far from the origin, at large extents or in small units now converge (status 0). The rank test is now a pivot at or below 1e-5 of the unit-diagonal system (was 1e-7 of the largest unscaled diagonal). The history's translation step keeps its meaning (origin-frame translation change) |
| core-numerics-02 | fixed | `analysis-core/kernels/permutation-shuffle.ts:147` (WGSL: `select(key + (count - position), key - position, key >= position)`), `:79` (TypeScript twin, same form) | `utils/permutation-shuffle-full-range.spec.ts` (WGSL at counts 2^31, 2^31 + 1, 3e9, 2^32 - 2, 2^32 - 1 against the exact-integer definition in `utils/permutation-shuffle-reference.ts`, plus inversion by the reversed rounds; failed on the old kernel at count 3e9), `utils/permutation-shuffle-full-range.node.spec.ts` (hand values of the report) | For a domain above 2^31 the WGSL shuffle is now a permutation and equals the CPU twin. `GPUSpatialScanStatistic` with a total population between 2^31 and 2^32 draws different (now valid) labels. Unchanged for domains up to 2^31 |
| core-numerics-03 | fixed (documentation and a helper; the default constant is unchanged) | `analysis-core/cell-key-wgsl.ts:8-18` (the quantisation is stated on `GPU_CELL_DEFAULT_SUM_SCALE`), `:45` (new `getGPUCellSumScale(maximumMagnitude, maximumRowCount)`), exported at `gpu-spatial-analysis/cell-aggregation/index.ts:20`; "exact" removed and the resolution stated in `gpu-spatial-analysis/cell-aggregation/cell-table.ts:50`, `gpu-cell-aggregation.ts:88` and `docs/api/gpu-spatial-analysis/operators-cells.md:73-87`. `GPUGroupStatistics` and `GPUKeyJoin` and their section of `docs/api/gpu-dataframe-analysis.md` were changed by the `dataframe` stream (dataframe-03) | `utils/cell-sum-scale.spec.ts` (GPU: 1,000 rows of 1e-4 sum to `7000 / 65536` at the default scale and to 0.1 within 1e-7 with the helper's scale), `utils/cell-sum-scale.node.spec.ts` (hand-computed scales) | None for existing calls. New optional helper |
| core-numerics-04 | fixed | `analysis-core/kernels/normal-quantile-wgsl.ts:24` (`2509.0809287301226727`), `:32` (`4.6303378461565452959`), `:37` (`0.59983220655588793769`). All 45 coefficients were compared with the table in CPython 3.15 `statistics.py`; these three were the only differences | `utils/normal-quantile-wgsl.node.spec.ts` (every literal against the published table, digit for digit), `utils/normal-quantile-wgsl.spec.ts` (GPU against `statistics.NormalDist().inv_cdf` at 32 probabilities in all three branches, down to 1.5e-38, tolerance 3.6e-7 relative; the old far tail was 4.7e-7 to 7.7e-7 off on the GPU and failed) | Far-tail quantiles (tail probability below 1.4e-11) move by 2e-7 to 8e-7 relative; `GPURasterStandardizedIndex` values between 6.7 and its 8.21 limit change in the last digits. Measured GPU error is now at most 2.1e-7 relative on the spec's rows |
| core-numerics-05 | fixed for the part in this scope | The helper was added by the lead (`analysis-core/kernels/philox-random.ts:152`, `:207`); this stream added its documentation in the `PHILOX_WGSL` comment and its specs. The module is not in a public barrel, so there is no further export. The four kernels with the faulty 24-bit form are in other streams | `utils/philox-open-unit-float.node.spec.ts` (largest word gives `1 - 2^-24`, smallest `2^-24`, by hand), `utils/philox-open-unit-float.spec.ts` (WGSL equals the CPU form bit for bit on the edge words and 2,000 random words; `log(u)` is finite and negative on the GPU) | None in this scope |
| core-numerics-06 | fixed (documentation; formula unchanged) | `pointcloud-analysis/gpu-pointcloud-ground.ts:37-54` (`slope` prop), class comment; `docs/api/pointcloud-analysis.md:269` | `gpu-pointcloud/gpu-pointcloud-ground-threshold.spec.ts` (pins the documented thresholds 0.3 and 0.3 with bumps of 0.25 and 0.35 that Zhang's 0.2 and 0.4 would classify the other way) | None |
| core-numerics-07 | fixed (documentation) | `docs/api/pointcloud-analysis.md:220-234` (the published class has no `centroids` or `attribute`; how to compute centroids today; why `output` must not be passed), `:461`; `pointcloud-analysis/gpu-pointcloud-voxel-merge.ts:32-37` | none (words only) | None |
| core-numerics-08 | fixed (documentation) | `analysis-core/kernels/sphere-distance-wgsl.ts:22-28` | none (words only; the existing `utils/wgsl-sphere-rows.spec.ts` already allows 1e-3 rad near the antipode) | None |
| core-numerics-09 | fixed (documentation) | `analysis-core/kernels/normal-quantile-wgsl.ts:15` | `utils/normal-quantile-wgsl.spec.ts` (`p <= 0` gives `-1e15`, `p >= 1` gives `1e15`) | None |
| core-numerics-10 | fixed (documentation) | `pointcloud-analysis/gpu-pointcloud-features.ts:38-39`, `gpu-pointcloud-icp.ts:83-86`, `:367` (loop variable), `docs/api/pointcloud-analysis.md:64-65` | none (words only) | None |
| core-numerics-11 | fixed | New specs for the three kernels named. `masked-weak-components.ts`, `parallel-merge-tree.ts` and `hash-uint32.ts` were not given a direct spec (not done in this pass) | `utils/normal-quantile-wgsl.spec.ts`, `utils/wgsl-theil-sen-median.spec.ts` (median of all pairwise slopes by a full float64 sort, 128 series), `utils/wgsl-poisson-observed-term.spec.ts` (float64 `o log1p(d) - (o - e)`, 618 pairs) | See core-numerics-12 |
| core-numerics-12 (new, found by the spec for 11) | fixed, wrong-edge-case | `analysis-core/kernels/wgsl-poisson-observed-term.ts:37-42`: `getEntropyExcess(-1)` evaluated `0 * log(0)` and returned NaN. `getObservedTerm` reaches it when the observed count is positive but below 2^-25 of the expected count, because `(o - e) / e` rounds to exactly -1 in float32 (one case against an expectation above 3.4e7; `o = 1e-6`, `e = 50` gave NaN on the GPU, true value 49.99998). The limit is 0, so the function now returns `-d` there | `utils/wgsl-poisson-observed-term.spec.ts` (`[1e-6, 50]` and `[1, 5e7]`; failed with NaN on the old kernel) | A NaN Poisson term (scan statistics, emerging hot spots, change points) becomes the expectation `e` in that corner |
| luma-official-05 | fixed | `analysis-core/kernels/uint64-wgsl.ts` (every function, `(high, low)` in the shader), `analysis-core/kernels/binary-search-wgsl.ts:34-37`, `:58`, `:77`, `:79` (the `'u64'` key is `(high, low)`; storage stays low word first). No importer outside `analysis-core` existed (grep over `modules`, `website`, `examples`) | `utils/wgsl-words-search.spec.ts` (BigInt reference unchanged; only the packing of the words in the test kernel moved) | Internal only: a future caller of `UINT64_WGSL` or of the `'u64'` binary search passes `vec2<u32>(high, low)` |
| luma-official order of work 14 (`GPUPointCloudSMRF`) | fixed (delegated) | New `pointcloud-analysis/pointcloud-diamond-morphology.ts`: the diamond minimum and maximum windows are the published `GPURasterErosion` and `GPURasterDilation` (`'cross'`, `'nodata'` border, `'ignore'` policy), in passes of radius at most 8; used at `gpu-pointcloud-smrf.ts:270-280`. Not added to the entry barrel | `gpu-pointcloud/pointcloud-diamond-morphology.spec.ts` (bit-identical to a full clipped-diamond scan on a 37 by 23 raster for radii 1, 3, 8, 9, 18, 37, minimum and maximum), and the existing SMRF fixtures (`gpu-pointcloud-smrf.spec.ts`, PDAL fixtures in `gpu-pointcloud-advanced.spec.ts`) | Same classifications. The object openings take about 50 passes at the default window of 18 (was about 190) and 350 at 64 (was about 2,100); each pass adds a transient validity buffer |
| luma-official order of work 14 (`GPUPointCloudGround`) | deferred | Not delegated: see "Upstream gaps" | n/a | None |
| luma-official-07 (`validatePackedView`) | deferred | Reason: roadmap 12.4 (rename of an exported symbol) | n/a | None |

Behaviour changes in one list:

1. Point-to-plane ICP: status 3 with the identity transform becomes status 0 with the solved
   transform for well-posed scenes far from the origin, of large extent, or in small units; the
   rank limit is 1e-5 on the unit-diagonal system.
2. Swap-or-not shuffle above 2^31 values: WGSL result changes to the correct permutation.
3. Normal quantile below a tail probability of 1.4e-11: values move by 2e-7 to 8e-7 relative.
4. Poisson observed term with `0 < o < 2^-25 e`: NaN becomes `e`.
5. `UINT64_WGSL` and the `'u64'` binary search take `(high, low)` (no caller outside core).
6. SMRF: same output, fewer and wider morphology passes.

### Float compensated summation (task added by the lead during the pass)

The trajectories stream measured that the shader compiler of the test GPU removes the float
correction of a Kahan or Neumaier sum. In this scope:

- `getAdditionRoundingError` now lives in `analysis-core/kernels/addition-rounding-error-wgsl.ts`
  as `ADDITION_ROUNDING_ERROR_WGSL` (same function name and signature), and
  `gpu-spatial-analysis/line-segmentize/line-segmentize-kernels.ts` imports it and re-exports it
  under the same name. It was extended from non-negative to finite operands of either sign: the
  three values are combined as wrapping `u32` integers in units of the smaller operand's last
  place (the result is at most `2^25` units), and operands more than 25 binades apart return the
  smaller operand. For non-negative operands the result is unchanged (both forms are exact).
- `utils/addition-rounding-error-wgsl.spec.ts`: 4,022 pairs of both signs against the exact
  float64 error (bit for bit), a million terms with mean 1234.567 and a quarter of them negative
  (and their negation) against the exact sum, and the pinned reason: in the same kernel the float
  form `(sum - next) + term` leaves a correction of exactly 0 and a total 16,115 from the true sum
  of 1,234,041,587.7, while the integer form is 0.44 away before the final rounding and the
  fixed-point accumulator gives the nearest float32.
- A second fact from that spec: the plain float32 running sum of the kernel (1,234,025,472) is not
  the IEEE serial sum of the same terms in the same order (1,234,033,152). The running sum that
  feeds `getAdditionRoundingError` is. A CPU twin cannot expect a plain GPU loop sum bit for bit.
- Own directories: `grep -rliE "neumaier|kahan|compensat|two-?sum|error-free"` over `analysis-core`
  and `pointcloud-analysis` finds only the comment in `exact-orientation-wgsl.ts` that says float
  error-free transforms are not used. The fixed-point accumulator quantises, adds and decodes with
  integer words and `ldexp` only; its GPU spec against the BigInt oracle
  (`utils/fixed-point-accumulator.spec.ts`) passes, and the new spec adds a million-term run of it.
  The point cloud kernels use running means and block merges, not compensated sums.
- `dev-docs/contributor-contract.md` section 11 has two new bullets with the accepted cures.

### Review follow-up

After the independent review (`fix-review-geometry.md`) and the second compiler fault isolated by
the `network-paths` stream (fix review fr-fields-05). Where this subsection and the table above
disagree, this subsection is current.

**Function-scope identity tables.** Reproduced in a kernel of one binding
(`utils/function-scope-table-readback.spec.ts`): a function-scope `var order: array<u32, N>`
written only by `order[i] = i` and read by a loop index inside a loop nest reads back as all zeros
at 40, 48, 56, 64, 72 and 96 entries and right at 8 to 32 and at 128; the same table as a
module-scope `var<private>` is right at all 13 sizes, and so is a float table
(`order[i] = f32(i)`) and a table written by `N - 1 - i`.

Search: all of `modules/analysis/src` (595 function-scope array declarations), for a write of the
form `table[i] = i`, `table[i] = u32(i)`, `table[i] = f32(i)` or `table[i] = base + i`, including
writes whose value is a template expression, and for arrays whose only writes are constants. Every
site found:

| Site | Shape | Verdict | Evidence |
| --- | --- | --- | --- |
| `analysis-core/kernels/dense-solve-wgsl.ts:319` (`result.pivots`, with the constant fills of `rowScale` and `columnScale` at `:317-318`) | Struct member of a function-scope `var`, identity loop, then `pivots[column] = pivotRow` in the elimination | Not the failing shape (later writes), and right on the GPU. Not changed | New spec: LU at 9 capacities from 8 to 64, a system that needs no exchange (the table stays the identity) and one that exchanges every row, with 0 and 2 equilibration passes; the solution a float64 right-hand side was built from is recovered to 2e-4 |
| `gpu-network/network-optimization/gpu-network-louvain.ts:225` (`groups`) | Identity loop, then `groups[vertex] = communities[...]` at `:285` after every level | Not the failing shape (later data writes). Other stream's file | Pinned by `gpu-network-louvain-visit-order.spec.ts` through the class at 10 sizes, 4 to 32 (the class limit) |
| `gpu-network/network-optimization/gpu-network-louvain.ts:59` (`louvainVisitOrder`) | Already a module-scope `var<private>` | Fixed by `network-paths` | Same spec |
| `gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan.ts:736-738` (`order` in `getOrder`) | Identity loop, then the merge-exchange sort swaps entries by data (`:809-810`); with one neighbor the table has one entry | Not the failing shape (the "identity loop followed by swaps" row of fr-fields-05 read right at every size). Other stream's file; not tested here beyond its own spec, which passes | Reading; `stx07-expectation-based-scan.spec.ts` |
| `gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton.ts:159`, `:179` (`edgeIds`) | `edgeIds[vertex] = first + vertex` with a run-time `first`, then data writes at `:247` | Not the failing shape (run-time base and later writes). Other stream's file | Reading |
| `gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor.ts:208`, `:255` (`sampleIndices`, direct path without `triangles`) | **The failing shape**: identity loop only, read by loop indices in single and nested loops and as an index into storage | Right on the GPU at every size tried; recommended change is for the owning stream: drop the table on the direct path (the index is the value) or move it to module scope | New spec through the class: 10 sample counts from 8 to 64 (the limit of the direct path), with and without the `weights` output, a linear field that Sibson interpolation reproduces exactly, 3 queries each |
| `raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts:246`, `:251` (`times`, without `dateAxis`) | Float identity table (`times[slice] = f32(slice)`), read by computed indices | Right on the GPU at every size tried, and the float form did not fail in the one-binding kernel either. Recommended for the owning stream: compute `f32(slice)` in place, or move the table to module scope | New spec through the class: 13 slice counts from 8 to 128, a triangular curve whose peak time `n / 2` and start time `n / 8` are known in closed form |
| Arrays whose only writes are constants (`exact-orientation-wgsl.ts:84` `present`, `polygon-overlap-metric.ts:26` `present`, `gpu-network-flow.ts:221`, `gpu-network-rational-access.ts:197`, the unit vectors of the regression solvers, `spatial-validation-design.ts:278`, `gpu-raster-whittaker-smoother.ts:307`) | Conditional or single-element writes, or a constant fill followed by data writes; none is a constant table that is only read | Not the failing shape | Reading |

Nothing was moved to module scope in `analysis-core` or `pointcloud-analysis`: the one table
there (`pivots`) is written after its identity loop, is part of a struct that the shared solver
returns and takes by pointer (moving it would change a shared signature), and reads back right at
every capacity tried. The cause inside the compiler is still unknown, so a table that passes at
the sizes tried is evidence for those compiled kernels, not a proof for every configuration.

**fix-review-geometry rows in this scope.**

- fr-geometry-11: the missing space after `+` in `analysis-core/kernels/normal-quantile-wgsl.ts:24`
  is restored (no value change; the digit spec still passes).
- fr-geometry-04: the byte-for-byte pin `computation-pack/local-permutation-kernels.node.spec.ts`
  passes on the current tree (run here; not edited by this stream).
- Files the review did not read: `dense-solve-wgsl.ts` and `row-access-wgsl.ts` were not changed by
  this stream; `sphere-distance-wgsl.ts` changed in the comment at `:23-29` only (core-numerics-08).
  `symmetric-eigen-wgsl.ts` was not touched (the other session's file).
- Section 5 of the review: the `mask` and `validity` check of `GPUPointCloudStatisticalFilter`
  (`pointcloud-analysis/gpu-pointcloud-statistical-filter.ts:53-57`) is not a change of this
  stream; `PHILOX_WGSL` gained `philoxOpenUnitFloat` (added by the lead, core-numerics-05), which
  changes the text of every kernel that embeds it; the removal in `analysis-core/network-topology.ts`
  belongs to `terrain-surface`.

### Upstream gaps

- `GPURasterErosion` and `GPURasterDilation` (`@luma.gl/graph/gpu-raster`): a radius above 8 in one
  pass. `GPUPointCloudGround` uses square windows up to radius 32 over a raster with holes, and a
  sequence of radius-8 passes with the `'ignore'` policy is not the same window there, because an
  invalid centre never becomes valid and so cannot pass a value along.
- The same classes: a per-cell count of the valid samples in the window (or a completeness flag
  beside the `'ignore'` result). `GPUPointCloudGround` reports `INCOMPLETE_WINDOW` from it; today
  that needs a second `'propagate'` pass per window.
- `GPUPointCloudVoxelDownsample`: optional per-voxel `centroids` and attribute means, which
  `GPUPointCloudVoxelMerge` takes as input and nothing produces (core-numerics-07).
- A small 64-bit word shader module with multiply and divide (`UINT64_WGSL` now uses the official
  `(high, low)` order, so it could be replaced by one).
- A counter-based random generator with an open-interval unit float (`philoxOpenUnitFloat`).

### `analysis-core/polygon-triangulation` (not covered by the review)

Read in full: the ear-clipping kernel (`gpu-polygon-triangulation.ts:495-1278`), compared function
by function with Earcut. Skimmed only: `polygon-triangulation-monotone-wgsl.ts` and
`polygon-triangulation-monotone.ts` for floating-point arithmetic. The monotone path decides
everything through the exact `orientOf` wrapper and ordered integer keys and has no float
division; its combinatorics (sweep, helper selection, pointer jumping) were not verified.
`polygon-triangulation-layout.ts` was not read.

Matches Earcut: `areaSign` (minus the orientation), `pointInTriangle`, `isEar` and `isEarHashed`
(including the except-first-vertex rule), `locallyInside`, `intersectsPolygon`, `splitPolygon`,
`sectorContainsSector`, the hole order, the 15-bit z-order code. Deliberate differences that are
sound: ring winding from the exact orientation at the lexicographically smallest vertex instead of
a float signed area; `middleInside` by exact parity with simulation of simplicity;
`cureLocalIntersections` accepts proper crossings only.

Not fixed, because no input was found that gives a wrong result (a fix needs a pinning test):

| ID | Status | `file:line` | What differs from Earcut | Evidence |
| --- | --- | --- | --- | --- |
| core-numerics-13 | SUSPECTED | `analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts:1136` | The tie-break of `findHoleBridge` for two candidates with the same tangent compares `pp.x > pm.x`, where `pm` is the first candidate `m` of the ray. Earcut compares with the current best (`p.x > m.x`), which keeps the candidate nearer to the hole. Here a later, farther candidate on the same ray replaces a nearer one, so the bridge runs through the nearer vertex | A shell with two notch tips collinear with the hole's leftmost vertex, in both visiting orders: the output is still a valid tiling (areas sum exactly, every triangle inside), with a T-junction at the nearer tip, which Earcut's collinear filter can also produce. Pinned as a tiling check in `analysis-core/polygon-triangulation-hole-bridge.spec.ts` |
| core-numerics-14 | SUSPECTED | `gpu-polygon-triangulation.ts:919` | `isValidDiagonal` rejects `a.next.i == b.i` but not `a.prev.i == b.i`. `findSplit` excludes the node `a.prev`, not a duplicate node of the same vertex (bridges and splits create duplicates). In the one-bridge case the "opposite-facing sectors" test rejects the diagonal anyway | Reading only; no failing input constructed |
| core-numerics-15 | doc-mismatch, fixed | `gpu-polygon-triangulation.ts:1090-1094` | The comment said the bridge is validated by `isValidDiagonal`; `eliminateHoles` does not call it (nor does Earcut). The comment now says so | Reading |
| core-numerics-16 | precision, not fixed | `gpu-polygon-triangulation.ts:1109-1110` | The ray crossing `x` that selects the hit edge is a float32 division compared with `hp.x` and with the best `x` so far. Earcut does the same in float64. Two shell edges whose crossings differ by less than a float32 rounding can be ranked the wrong way | Reading; no failing input constructed |

### Closing pass (2026-11-11)

Scope: `analysis-core/polygon-triangulation` (stream `triangulation`). Where this subsection and
the table above disagree, this subsection is current. Source lines are those of
`analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts` after the pass.

**Method.** A seeded generator (`gpu-spatial-analysis/polygon-triangulation/polygon-triangulation-search-cases.ts`)
builds valid polygons with holes, kept by rejection with an exact validity check: a 2-opt shell of
6 to 24 vertices with 1 to 6 holes (lattice sides 14 to 200), a shell of 4 to 6 vertices with 7 to
12 holes, and a star-shaped shell of 70 to 159 vertices (the z-order hash path) with 2 to 8 holes,
each without and with holes moved to touch another ring at one vertex. Every polygon is placed on
an integer lattice, at projected-metre magnitude (500000, 4500000) with the lattice at float32
resolution (1/32 m by 1/2 m), and at that magnitude on a 2 m lattice. Four more families aim at
the bridge: two shell edges that the ray from the hole crosses less than one float32 step apart,
two bridge candidates whose tangents differ by less than a float32 division resolves, a hole
vertex one to four float32 steps right of a slanted shell edge, and two holes that share their
leftmost vertex with first edges of almost one slope. The GPU ran in batches of up to 3,000
polygons per graph (headless Chromium, WebGPU, Apple Metal). The checker (`getTilingDefect`) is
exact on the float32 values (a float64 filter, then BigInt): every triangle has non-zero area, its
centroid is inside the polygon, no triangle edge crosses a polygon edge, no polygon vertex is
strictly inside a triangle, no two triangles overlap, and the areas sum to the polygon area. The
reference library is the mapbox `earcut` package 3.2.4 (the version at `node_modules/earcut`; a
transitive dependency, imported by the specs as an oracle only) on the same float32 coordinates in
float64. Counts below are polygons.

| ID | Outcome | What was wrong, what changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| core-numerics-13 | fixed (confirmed) | `findHoleBridge`: of two bridge candidates on one ray from the hole vertex the tie went to any later candidate right of the first one, so the bridge could run through the nearer vertex. The tie now keeps the nearer candidate, compared with the current best as Earcut does (`:1190-1192`) | `gpu-polygon-triangulation-search.spec.ts` (2-opt seeds 731 and 1475, every frame); `analysis-core/polygon-triangulation-hole-bridge.spec.ts` now also pins the triangle count of the package (13; the old kernel gave 12) | On the old kernel the two seeds came back `valid = 0` (`'none'`), and the package tiles them. A copy of the package with the old tie rule fails on the same two polygons |
| core-numerics-14 | verified, no change for the suspected test; a related defect fixed as core-numerics-20 | The test `a.prev.i == b.i` is absent from the installed package too (3.2.4 dropped it; 2.2.4 had it). Adding it, by index or by position, at either end, to a copy of the package that uses the port's midpoint test changes no outcome on 12,613 polygons | n/a | In the search about half of the 2-opt polygons and 95% of the many-hole polygons bridge two or more holes to one vertex (1,489 of 2,928; 2,815 of 2,948), and 1 to 5% need a diagonal split (391 splits); all tile |
| core-numerics-15 | fixed (comment) | The comment of `findHoleBridge` (`:1125-1136`) states what is decided exactly and that the bridge is still not checked afterwards | n/a | Reading |
| core-numerics-16 | fixed (confirmed) | The hit edge was the largest float32 crossing `x`. Two shell edges crossed closer than float32 resolution tied, the first one met won, and the bridge crossed the nearer edge: `valid = 1` with triangles outside the polygon. The order of two hit edges is now exact (`crossesRowRightOf`, `:1105-1123`): two edges of a simple polygon do not cross, so the orientation of one edge's ends against the other's line orders their crossings; the float32 `x` is kept only for edges that do cross (invalid input) | Same spec, "the hole bridge does not depend on float32 rounding" | Old kernel: 500 of 700 polygons of the close-crossing family wrong at float32 resolution (0 on the integer lattice, where the same polygons are exact), 60 of the 84 pinned. The package tiles all 700 |
| core-numerics-17 (new) | fixed, wrong-edge-case | Same function, the threshold `x <= hp.x`: a hole vertex within a few float32 steps of a slanted shell edge had a computed crossing right of the vertex, the edge was skipped, no bridge was found, and the hole was dropped without a flag: `valid = 1` with the hole triangulated over. The test is now the exact side of the hole vertex against the edge (`:1153-1154`), and "on the edge" is orientation zero | Same spec, near-edge family (400 polygons in a frame that spans zero, 400 at projected-metre magnitude) | Old kernel: 56 of 2,000 wrong in the frame that spans zero (the hole vertex is far finer than the edge's ends), 0 of 2,000 at projected-metre magnitude; 11 of the 400 pinned. The package tiles all |
| core-numerics-18 (new) | fixed, precision | Same function, second walk: the bridge triangle used the float32 crossing as a corner and the candidates were ranked by a float32 tangent. Two reflex vertices in almost one direction tied, and with the old tie rule the farther one won: `valid = 1` with triangles across a notch. Membership of the triangle is now the row side, the side of the hit edge's line and the side of the line to the first candidate (`:1181-1183`), and the smaller angle is an orientation (`:1190`) | Same spec, close-tangent family | Old kernel: 224 of 600 wrong in each frame (lattice coordinates of 6,000 to 33,000 are enough), 52 of the 160 pinned. The package tiles all |
| core-numerics-19 (new) | fixed, precision | `isLeftOf`: holes that share their leftmost vertex were ordered by a float32 slope; a tie left them in input order, and the second bridge then failed: `valid = 0`. The order is the exact orientation of the two first edges (`:1086-1090`) | Same spec, shared-leftmost family | Old kernel: 278 of 600 `valid = 0` in each frame, 76 of the 160 pinned. The package tiles all |
| core-numerics-20 (new) | fixed, wrong-edge-case | `isValidDiagonal`: a zero-length diagonal between two nodes at one point (two holes that touch) with both sectors convex passed the sector tests trivially and was then decided by the midpoint test at the shared point. Earcut's float test rejects it; the port's exact parity perturbs the point and accepted it, which turned one loop inside out and left a ring with no valid diagonal: `valid = 0`. Such a pair is now rejected unless it is Earcut's special case of two reflex sectors (`:924-929`) | Same spec (many-hole seed 1936 with touching holes) | Old kernel: `valid = 0` on that polygon in both frames, and the package tiles it. A copy of the package with the port's midpoint test fails on exactly the polygons the port flagged, and with the new rule on exactly those a ring-order Earcut fails |
| Monotone path | verified, no change | Read in full: `polygon-triangulation-monotone-wgsl.ts`, `polygon-triangulation-monotone.ts`, `polygon-triangulation-layout.ts`. Checked by reasoning: the sweep order is total and is the order of a tilted sweep; the class of a vertex follows from the ranks of its neighbours and one exact turn; a group of an edge holds its insertion first, its deletion last and the left touches between, so the predecessor in the group is de Berg's `helper`; the rule "later is a split or earlier is a merge" is the union of de Berg's six handlers; the four diagonal slots have one writer each; `next` is a bijection on half-edges, so its cycles are the faces; the count check `sum(piece - 2) = n + 2h - 2` fails when a hole is left unconnected; the stack triangulation emits the triangles of de Berg 3.3 with a collinear triple as a zero-area triangle | Same spec, "the monotone path alone" (`monotoneMinimumVertices: 3`, `maximumWork: 1`, so a valid result cannot be the Earcut fallback) | GPU: 6,276 polygons without touching rings in each of three frames: all valid, exactly `n - 2 + 2h` triangles, an exact tiling once zero-area triangles are skipped (4,496 polygons have collinear vertices), directed edges cancel to the boundary. With touching rings 5,431 of 6,337 are flagged, as documented, and the 906 that are valid tile. The CPU model of the same passes agrees on the same polygons |

**Counts on the final kernel.**

- Earcut path (`largePolygonMethod: 'none'`), holes not touching: 6,276 polygons in each of three
  frames (18,828 GPU triangulations): none flagged, none wrong. The triangle count equals the
  package's except in 7 runs, all at float32 resolution, where both sides tile the polygon (the
  package's own bridge changes when the same polygon is translated). The package returns a wrong
  triangulation for one of these polygons in every frame (many-hole seed 2417; its ring-order
  variant does not), and the port tiles it.
- Earcut path, holes touching (5,728 of 6,337 polygons have rings that share a vertex), two frames
  (12,674 triangulations): none wrong; 16 polygons flagged `valid = 0` in both frames. The package
  returns a wrong triangulation for 15 of the 16 and tiles the other (many-hole seed 2370, where
  its block-order scan takes another bridge; a ring-order copy of the package fails all 16). The
  triangle count differs from the package's on 13 to 15 polygons per frame, each a valid tiling.
- Default method (`'monotone'`): run on all of the above before the last fix, with the same counts
  of flagged, wrong and count differences as `'none'`; the existing bitwise spec still passes.
- Bridge families: 700 close-crossing, 600 close-tangent and 600 shared-leftmost polygons in two
  frames and 2,000 near-edge polygons in two frames: all valid tilings with the package's count.
- Translation: the indices on the 2 m lattice at projected-metre magnitude are bit for bit those of
  the integer lattice on 5,976 polygons. At float32 resolution 9 of 2,928 2-opt polygons and 19 of
  2,948 many-hole polygons take a different, valid, triangulation (next bullet list).

**Left, with the reason.**

- `middleInside` (`:862-885`) still places the midpoint of a candidate diagonal in float32. Its
  parity is exact for that rounded point, but the point itself moves when a midpoint is not a
  float32 value, which is the only decision left that depends on rounding: it changed which valid
  diagonal was taken in about 0.3% to 0.6% of polygons at float32 resolution, and no result was wrong or
  flagged because of it. An exact midpoint side needs the sign of a sum of two orientation
  determinants (twelve products), which `analysis-core/kernels` does not have; not added without
  a failing input.
- Holes that touch another ring at a vertex: the 16 polygons above stay `valid = 0` on both paths
  (the monotone path does not take touching rings). They are valid polygons. The port flags them;
  it does not return a wrong triangulation.
- A hole for which no bridge is found is skipped without a flag (`:1251`), as in Earcut. After
  core-numerics-17 this needs a hole outside its shell, which is invalid input; not changed.
- The package's failure on a polygon whose holes do not touch (seed 2417) is an upstream defect of
  `earcut` 3.2.4; not reported upstream from here.

**Not verified.** Other GPUs and the software adapter (one Apple GPU was used). Rings above 159
vertices on the Earcut path beyond the existing specs. Holes that touch the shell along an edge
or at more than one vertex, self-intersecting rings and other invalid input. The lens records
under `docs/lens-records` that store the hash of the Earcut kernel were not regenerated (the
kernel text changed; `modules/lenses` was off limits).

**Specs and commands.** `npx vitest run --project headless` on
`modules/analysis/test/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation-search.spec.ts`
(3 tests; 749 random polygons on each path and 1,288 bridge polygons, 55 s),
`gpu-polygon-triangulation.spec.ts`, `gpu-polygon-triangulation-monotone.spec.ts` and
`modules/analysis/test/analysis-core/polygon-triangulation-hole-bridge.spec.ts`: 4 files, 21 tests
pass. `npx vitest run --project node modules/analysis/test/gpu-spatial-analysis/polygon-triangulation/`:
4 files, 45 tests pass. On the kernel as it was before the pass the search spec fails in two of its
three tests (2-opt seeds 731 and 1475; 199 of the 1,288 bridge polygons) and the hole-bridge spec
fails on the count; the monotone test passes on both kernels.

### Closing pass, review fixes (2026-11-11)

Scope: the five triangulation findings of
[fix-review-closing-geometry.md](./fix-review-closing-geometry.md) (frc-geometry-01 to 05; 06 is
ring assembly, in [polygon-topology.md](./polygon-topology.md)). Where this subsection and the one
above disagree, this one is current: in particular "it does not return a wrong triangulation",
"none flagged, none wrong" and the two "Left" items on touching rings and on a hole without a
bridge are superseded. Source lines are those of
`analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts` after these fixes.

**The rule, and why it now holds.** A result flagged `valid = 1` is a tiling of the polygon
whenever the polygon is valid. This is an argument, not a count. Take every triangle
counter-clockwise. The number of triangles over a point is the winding number about that point
of the sum of the triangle boundaries (as directed segments, where overlapping opposite segments
cancel). If that sum is the polygon boundary (shell counter-clockwise, holes clockwise), the
count is the polygon's own winding number, 1 inside and 0 outside: a tiling. On the Earcut path
every step keeps "triangles written plus the working ring" equal to the polygon boundary (a
bridge or a split adds a segment in both directions, the collinear filter replaces two collinear
edges by their sum, an ear replaces two edges by the third side of a counter-clockwise triangle
of exact positive area) except three, which are now failures: Earcut's cure of a local
self-intersection, which writes a triangle across two crossing edges; a hole for which no bridge
is found, which Earcut drops; and a predicate that cannot be certified. On the monotone path the
pieces are the cycles of a permutation of the half-edges and the stack triangulation replaces two
consecutive edges of a piece by a third at each step, so the triangle boundaries sum to the
polygon boundary by construction, and the only way to a wrong result is a clockwise triangle,
which is now a failure. The same argument is the checker of the specs
(`getBoundaryChainDefect`), which is why it is complete.

**Method.** A CPU model of the Earcut kernel (a line-by-line port, scratch) was checked against
the GPU on every polygon of the search: equal `valid` and equal index lists on 126,351 of 126,351
(the 31 polygons of about 4,300 vertices were run on the GPU only). Causes were found by tracing
the model, candidate fixes were tried in the model, and the final kernel was run on the GPU.
Inputs: the reviewer's files (36,941 polygons: the seven touching-ring families, the two sets
with every touched edge noded, the apart families and the hand cases), four more sets from the
reviewer's generator with other seeds (34,438), the author's families of
`gpu-polygon-triangulation-search.spec.ts` with seeds 1 to 4,000 (19,575), and the reviewer's 31
enlarged polygons: 90,985 polygons with up to twelve holes, four in five with touching rings.
35,397 of them were run again at projected-metre magnitude (2 m lattice and 1/32 m by 1/2 m),
126,382 in all. Validity is Shapely's (2.2.0 on GEOS 3.14.1) for the reviewer's
generator and the exact check of the search cases for the author's. Verdicts are those of the
new checker; the `earcut` package 3.2.4 ran on the same float32 coordinates. Three kernels were
run: the one before the first closing pass (the reviewer's copy), the one after it ("before"
below), and the final one.

| Finding | Outcome | Cause, and what changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| frc-geometry-01 (major) | fixed | Reproduced: both review inputs come back `valid = 1` and wrong. Cause: the first walk of `findHoleBridge` only looked at downward edges, so a hole vertex inside a horizontal or upward edge of the outer ring was not seen to touch it. In the first review input `(4, 9)` lies inside the horizontal edge `(5, 9) - (3, 9)` of the hole merged before it; the ray went on to the shell and the bridge `(4, 9) - (0, 11)` cut through that hole. The ring then crosses itself, and Earcut's `cureLocalIntersections` wrote a triangle across the crossing: in the model every wrong result on the lattice sets (20 of 20) had passed through the cure, and none of the 79,610 tilings had. Two changes. (a) Where Earcut would cure, the polygon fails (`hasLocalIntersection`, `:1009`, called at `:1091`): a triangulation that contains a cure triangle is not a tiling (the triangle and the piece it drops do not add up to the ring they replace), so nothing correct is lost. Also failures now: a hole without a bridge (`:1340`) and an uncertified predicate (`:587-601`). (b) A hole whose leftmost vertex lies inside an edge of the outer ring, any direction, horizontal included, is joined there: a node with the hole vertex's row is inserted into the edge and returned, so the bridge has zero length (`:1220-1223`; box test, then `orientOf == 0`). This is the reviewer's scratch fix; it makes the `side == 0` return and the `touchesEdge` case of the second walk unreachable, and both are removed. One node per hole is reserved for it: `getPolygonTriangulationNodeCount` is `2 * vertices + 5 * holes + 8 * polygons` (`:82-91`, `baseNodeOf` `:1352`), validated where it was (`validatePolygonTriangulationScratch`). The index capacity is unchanged: a zero-length bridge leaves two coincident nodes that the filter removes, so the ring has fewer nodes than with an ordinary bridge. | `gpu-polygon-triangulation-touching-rings.spec.ts` (16 review inputs and 1,043 polygons of fixture `polygon-triangulation-touching-rings.json`: Shapely validity and area) | Search, Earcut path: 32 wrong before, 0 after. In the model on 79,953 lattice polygons, (a) alone turns the 20 wrong results into `valid = 0` and loses no tiling (79,610 of 79,610 stay tilings); (b) turns them into tilings and brings the flagged from 343 to 68. That no wrong result is left is an argument, not only a count: see "The rule" |
| frc-geometry-02 (major) | fixed (the monotone path detects it) | Reproduced: the 15-vertex input and the enlarged polygon in the default configuration come back `valid = 1` and wrong. Cause: `fanStack` (`polygon-triangulation-monotone-wgsl.ts:1067`) wrote its triangles without looking at their orientation. With touching rings a piece can pass every check (count, monotone chains, closed walks, wedge order) and still not be a simple monotone polygon; a fan triangle is then clockwise and lies partly in a hole. Every wrong result had one: the triangle areas exceeded the polygon area by twice the clockwise area (29 against 28; 122,159,104 against 120,061,952). The fan now fails the piece on a clockwise triangle (exact `orientOf`); zero-area triangles between collinear vertices are still written. The polygon is flagged, takes route 4 and is given to Earcut. The test-side CPU model (`polygon-triangulation-monotone-model.ts`) has the same test. | Same spec: "the monotone path alone flags a polygon or tiles it" and "the default method tiles polygons of more than 4096 vertices whose rings touch, as method none does" (the 16 review inputs enlarged to 4,099 to 6,147 vertices) | Search, monotone path alone: 106 wrong of 38,207 valid before; after, 0 wrong, the 106 are flagged and every other result is unchanged (30,051 tilings with zero-area triangles, 8,050 without). Default configuration: 34 wrong before (32 of them Earcut's), 0 after; never `valid = 0` where `'none'` is valid (0 of 126,382). The documentation sentence "never does worse than `'none'`" is true as written and now says why |
| frc-geometry-03 (minor) | fixed | Reproduced: both inputs are `valid = 0`. (-19) `isLeftOf` took `nextOf` of the leftmost node, which is at the same point when the leftmost vertex is repeated: orientation zero both ways and the two holes kept their input order. It now steps to the next different point (`nextDistinctOf`, `:1118`). (-13) The tie rule itself is right and stays: with the tie rule of the old kernel put back, 41 polygons of the search fail. In the 17-vertex input the two holes meet at `(5, 5)`, a collinear vertex of the first hole; the collinear filter removes it when the first hole is bridged, so the leftmost vertex of the second hole lies inside the edge `(6, 5) - (4, 5)`. That is the case of frc-geometry-01 (b) and is fixed by it (without the insertion the input stays `valid = 0`). | Same spec (review inputs "frc-geometry-03 ..." and the `failedBefore` entries); `gpu-polygon-triangulation-search.spec.ts` passes unchanged | Three kernels on the GPU. On the reviewer's sets where the first pass lost polygons (`tri.json`, `tri-edge3.json`, `tri-edge3-noded.json`, 24,951 polygons): the first pass repaired 549 (416 flagged, 133 wrong) and lost 18 (2, 9 and 7); all 18 are tilings again, none of the 549 is lost, and all 24,951 are valid tilings. Over all the reviewer's lattice sets (36,941): no polygon that either earlier kernel tiled is flagged or wrong now. Against the first-pass kernel over the whole search: 566 gained (534 flagged, 32 wrong), 1 still flagged, 0 lost |
| frc-geometry-04 (minor) | fixed, cause established | The first hole's bridge ends at shell vertex `(14, 6)` in line with the shell edge before it, so the collinear filter removes one of the two nodes at that vertex and leaves the other, whose sector is the sliver between the bridge and the next shell edge, inside the merged edge `(12, 8) - (18, 2)`. The ray of the second hole meets both that edge and the sliver node's edge at `(14, 6)`. The two crossings are equal, the walk order chose, and the hole was joined to the sliver sector, which cannot be clipped. Of two hit edges that meet the row at one point, an edge that ends there at a node whose sector does not hold the hole now gives way to one that passes through the point or whose node holds the hole (`hitEndOnRow` `:1175`, `:1229-1247`). | Same spec (review inputs "frc-geometry-04 ...") | Rings apart: 7 of 23,480 flagged before (the review's polygon, its variants, and 5 of the new sets), 0 after. Every polygon with rings apart that was valid before has bit-identical indices after (23,473 of 23,473 in the model, which equals the GPU). With the tie-break left out, 31 of 79,953 lattice polygons stay flagged in the model, most of them with touching rings. Earcut's own collinear filter is kept: a variant that kept collinear nodes at the cuts also fixed the input but changed the triangles of 5,152 of those 23,473 polygons |
| frc-geometry-05 (minor) | fixed (test helper) | `getTilingDefect` passed the review's two triangles (`undefined` on the old helper, confirmed). It now ends with `getBoundaryChainDefect`: every triangle counter-clockwise, shell counter-clockwise, holes clockwise, each directed edge counted +1 at its start and -1 at its end on its line (BigInt on the float32 values, direction reduced by the gcd), and every sum must be zero. That is the argument of "The rule", so it is a complete test: it sees overlap and gaps, and accepts a triangle side that runs past a collinear vertex. The older tests stay in front of it for their messages. | `polygon-triangulation-search-cases.node.spec.ts` (7 tests: the review's counterexample, a tiling of the same polygon, a side past a collinear vertex, zero-area triangles, gap, double cover, a doubled triangle with the right area, the wrong and a right triangulation of the first frc-geometry-01 input) | The right triangulation in that spec is `shapely.constrained_delaunay_triangles` (11 triangles, union equal to the polygon). The new check agrees with the reviewer's independent checker on every count of the review that was recomputed (7 wrong and 98 flagged on the Earcut path, 3 and 13 wrong on the monotone path, 2 flagged apart) |

**Counts on the final kernel** (GPU, headless Chromium on Apple Metal; 126,382 triangulations per
configuration: 90,985 polygons on lattices and 35,397 of them again at projected-metre
magnitude).

| Configuration | Tilings | Flagged `valid = 0` | Wrong with `valid = 1` | Before these fixes |
| --- | --- | --- | --- | --- |
| Earcut path (`'none'`) | 126,381 | 1 | 0 | 125,815 tilings, 535 flagged, 32 wrong |
| Monotone path alone | 38,101 (30,051 with zero-area triangles) | 88,281 | 0 | 38,101 tilings, 88,175 flagged, 106 wrong |
| Default | 126,381 | 1 | 0 | 125,830 tilings, 518 flagged, 34 wrong |
| `earcut` package 3.2.4 | 125,741 | (it has no flag) | 641 | |

The package's 641 wrong results are 640 polygons with touching rings and one with rings apart.
The one polygon the kernel flags is tiled by the package. Of the 101,889 polygons with touching
rings the monotone path alone flags 88,281 and tiles 13,608; it tiles every polygon with rings
apart.

**Behaviour that changed for results that were valid before.** Compared index by index in the
model on 58,814 polygons that were valid before. Rings apart (23,473): none changed. Touching
rings (35,341): 944 take another valid triangulation, 129 of them with another triangle count;
all 944 have a hole vertex inside an edge (some only after the collinear filter removed the
vertex it touched), which is now joined by the inserted vertex. Input that is not a valid polygon
and that Earcut's cure used to return as `valid = 1` is now `valid = 0` on the Earcut path (the
monotone retry of the default method may still complete it), and so is a polygon with a hole for
which no bridge is found (a hole outside its shell) or with non-finite coordinates.

**Assertions changed in existing specs.** None in the triangulation specs:
`gpu-polygon-triangulation-search.spec.ts` lost its private copy of the batch helper (now
`polygon-triangulation-search-harness.ts`) and asserts what it asserted. See "Specs and commands"
for the specs of other contributors.

**Left, with the reason.**

- One polygon is flagged: the reviewer's `touch-hole-vertex+sliver` polygon at index 2192 of
  `tri.json` (32 vertices, 6 holes, the lattice shear `diag-4096`) at 1/32 m by 1/2 m
  resolution. (Corrected after the re-review: this account first called it polygon 1988, which
  is its position among the 5,447 polygons that map to that grid exactly, and gave it 37
  vertices.) It is a tiling
  on the lattice and on the 2 m lattice. The cause is the one named under "Left" above:
  `middleInside` places the midpoint of a candidate diagonal in float32, and in this sliver the
  rounded midpoint is outside the polygon, so no diagonal is accepted. It was flagged before
  these fixes too. The exact test needs the sign of a sum of two orientation determinants; it
  was not added in this pass (it is none of the six findings), and there is now a failing input
  for it.
- A touching vertex that is not the hole's leftmost one gets no inserted vertex. No polygon of the
  search needed one: Earcut's zero-length split between two reflex nodes and the ear order
  resolved every such touch. If one does not resolve, the polygon is flagged, not wrong.
- `ringWinding` of a ring whose lowest-leftmost vertex is collinear with its neighbours (a spike,
  not a valid ring) keeps the input direction; the result then follows the winding number of the
  rings as given.

**Not verified.** Other GPUs and the software adapter. Polygons above 6,147 vertices with touching
rings. Holes that share an edge with another ring, rings that cross and other invalid input,
beyond the statement above and the three inputs of the spec (a hole outside its shell, a hole
across a shell edge, a coordinate that is not finite). Cost: not measured apart from the spec
run times, which rose under shared load (the 46 headless spec files that use the two modules:
154 s of test time before, 177 s after; the search spec 46 s to 52 s, the ring-assembly spec
4.5 s to 6.4 s); compile time and run time were not separated. Lens records: not regenerated (`modules/lenses` is
off limits); the kernels whose WGSL changed are the Earcut kernel (nodes `<id>-earcut` and
`<id>-earcut-fallback`, also through `GPURandomPointsInPolygon`, `GPUDotDensity`,
`GPUSubdividePolygons` and `GPURasterFractionalCoverage`; its text also changes with the node
count for any layout that has holes) and the monotone `triangulate` kernel.

**Specs and commands.** See the end of the review-fixes subsection of
[polygon-topology.md](./polygon-topology.md), which lists the commands for both modules.
