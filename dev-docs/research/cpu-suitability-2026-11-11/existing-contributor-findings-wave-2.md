# Things noticed in existing GPU contributors during the second twin wave (2026-11-11)

Collected from the implementors' reports of roadmap section 13.6, as they wrote them, grouped by stream. The same caveats as [existing-contributor-findings.md](./existing-contributor-findings.md): none was fixed in that pass, line numbers are from the tree as it was that day, and almost all are reasoned from the code, not run on a GPU. Verify before acting. Where a twin differs from its contributor because of one of these, the twin's "What differs from the GPU contributor" list says so.

### followup-cell-measures

- `modules/analysis/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts:283`: deduplication examines only stored segments. After capacity fills, repeated unstored segments can inflate `counts[2]`. The CPU test pins four unique segments for coincident squares with capacity three. This GPU finding follows from source inspection; it was not executed here.
- The same file, lines 143 and 221: unknown runtime `operation` values silently select symmetric difference.
- GPUCellMeasures: no additional suspected defect found beyond its documented float32 precision limitations.

### followup-error-ml

These are static findings; GPU execution was unavailable. All references are in `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood.ts`.

- **Line 875:** residuals use the raw intercept and predictors. Large offsets can cancel away residual variation despite the centered likelihood calculation.
- **Line 1219:** pseudo R² multiplies sums of squares. Both products can overflow float32, producing `NaN` for a finite correlation while status remains 0.
- **Line 267:** the raw-power trace overflow warning appears outdated: the current trace contributor computes normalized powers.
- **Line 1049:** singular-design classification can return status 1 alongside non-finite weights because this branch examines unfiltered design moments. The documentation promises status 3 for non-finite weights.

### followup-lag-ml

- `gpu-spatial-lag-maximum-likelihood.ts:1473` calls `fail(3u)` on stationarity rejection; `fail` at line 1428 clears the finite table and summary. This conflicts with the brief’s requirement to publish the rejected iterate. The twin follows the brief and documents the difference.
- `gpu-spatial-lag-maximum-likelihood.ts:331` describes overflow of raw `tr(W^q)`, although the implementation uses normalized traces `tr((W/s)^q)`. That warning appears stale.

### followup-map-matching

- `modules/analysis/src/gpu-network/map-matching/gpu-map-matching.ts:1989`: `sumWhole` is an `i32` accumulator without overflow handling. Many individually accepted terms can exceed its range and corrupt a finite likelihood. The new CPU closed-form test covers approximately −2.5 billion; this GPU concern was identified by inspection, not execution.
- `modules/analysis/src/gpu-network/map-matching/gpu-map-matching.ts:1913`: backward decoding initializes `chosen` only inside track ranges, while the matched-output kernel reads every point at line 1154. Prefix/suffix points outside all tracks can therefore read unwritten scratch.

### followup-nested-validation

- `modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts:74`: construction validates capacities and views, but seed, buffer distance and candidate models are checked only when child contributors are built in `getCommandNodes` (lines 183, 243 and 257). This delays invalid-prop errors beyond construction.
- Same file, lines 136, 206 and 208: command-node count grows with outer folds, candidates and inner folds, but the contributor exposes no `estimatedCommandNodeCount` or command-budget validation required by contract section 4.

No existing files were edited.

### followup-panel-lag-ml

`modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts:367,402–414`: zero variance receives a finite objective penalty, while success depends on phi convergence and bracket width without a profile stationarity check. An admissible zero-RSS pole could therefore receive status 0 at a nearby golden-section point despite having no finite Gaussian maximum.

The new node spec pins a concrete example: directed four-cycle weights, two opposing-sign periods, and `x = y − 0.5 Wy`. The CPU recognizes its exact zero variance. The GPU outcome is **unverified**; existing code was left unchanged.

### followup-raster-paths

Static reading indicates a capacity bug in `modules/analysis/src/raster-analysis/isolines/gpu-isolines.ts:442`: publication passes `segmentLevels` as the compact IDs. `createPublishNode` takes capacity from `output.ids.length` (`analysis-core/kernels/wgsl-kernel-nodes.ts:754`), while scatter is bounded by `segments.length` (`gpu-isolines.ts:416`). Construction permits a longer `segmentLevels` view at line 207. Such a configuration can report unwritten segments as present and miss overflow. The new operations allocate equal capacities. The existing contributor was left unchanged.

The extreme finite isoline case confirms an overflow risk in `src/raster-analysis/isolines/gpu-isolines.ts:988`: `b - a` can overflow float32. The CPU midpoint result, `x = 1`, is tested and documented. Actual GPU behavior remains unverified.

No existing contributor files were changed.

### followup-regions

- `gpu-max-p-regions.ts:177` describes `improvementMoveCount` as moves, but line 1273 copies AZP’s successful-round count (`gpu-azp-regions.ts:1068`). The twin preserves rounds.
- `gpu-azp-regions.ts:952` and line 961 make the gain threshold depend on uncentered squared moments. A large column translation can suppress otherwise improving moves. The twin retains this explicit rule and tests that behavior.
- `gpu-max-p-regions.ts:1278` publishes construction feasibility after refinement without recomputing it. This could leave stale feasibility words if refinement repairs an under-threshold region.

### followup-trajectory-restriction

- `modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-kernels.ts:276`: the polygon path consumes `exact.signs` without propagating `exact.uncertain`. The shared predicate can fall back to float32 signs, so classification can be uncertain without an output diagnostic.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-kernels.ts:391` and `:414`: compact summary counts describe retained upstream slots. Overflow adds one per overflowing source segment, rather than the number of discarded intervals. Consequently, the summary does not provide total required/dropped interval counts after upstream slot overflow.

- `trajectory-restriction-kernels.ts:395`: the compactor’s forward track cursor assumes ordered ranges. Reversed nonempty ranges can cause equal-time singletons from different tracks to merge. The twin preserves this kernel behavior; node and headless regressions cover it.
- `trajectory-restriction-kernels.ts:194`: overlapping nonempty geometry ranges write shared output slots concurrently. The twin retains an `offsets` error for those ranges and documents this distinction.

### geometry-normalize

- `modules/analysis/src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:119`: coincident rhumb pairs evaluate `atan2(0, 0)`, whose WGSL result is unspecified. The comparison spec uses an east-west limiting case instead.
- `modules/analysis/src/gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:210`: midpoint latitude uses the original, unclamped mean, although the preceding midpoint calculations clamp latitude. The twin matches this.
- `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap.ts:201`: validity bit 1 tests `required == 1`, so empty lines receive validity 0 despite the “too few distinct vertices” documentation at line 43. The twin matches this.
- `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry.ts:228`: selecting only the first minimum can produce different normalized sequences for cyclic rotations of a zero-winding ring with repeated minima. The twin preserves this rule.
- `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry.ts:213` and `:327`: overlapping vertex ranges or child-slot ranges permit conflicting writes without detection. The CPU function rejects these layouts.
- `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry.ts:269`: maximum-depth validation occurs during command-node construction rather than in the constructor.

### geometry-skeleton

All paths below are under `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/`.

- `gpu-convex-straight-skeleton.ts:85`, `gpu-vector-straight-skeleton.ts:74`, `gpu-offset-curve-components.ts:67`, and `gpu-line-loop-removal.ts:62` use generic validation errors without the contributor ID. These depart from the required diagnostic contract.
- `gpu-offset-curve-components.ts:116` derives potentially large cut/fragment scratch products without an explicit storage-binding-size check before allocation.
- `gpu-line-loop-removal.ts:106` forwards spatial-context validation to a delegate built during `getCommandNodes`, so unsupported contexts are rejected after construction.

These findings were not changed.

### geometry-triangulation

- `modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts:105`: the inclusive `u32` loop can wrap when `maximumFlips = 0xffffffff`, which construction accepts at line 57. If an illegal edge remains at that iteration, the loop can restart instead of terminating. Inspection finding; not exercised.
- `modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull.ts:69` and `:89`: validation errors omit the contributor ID and accepted range or required capacity, contrary to the contributor contract.

### inference

- `modules/analysis/src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts:167`: transform props are documented as finite, but the constructor does not validate them. Nonfinite values can reach WGSL literal generation and produce invalid shader text.
- `modules/analysis/src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope.ts:55`: status documentation includes invalid weights, but extreme-rank inference never reads reference/weight values. Its bindings at `gpu-point-pattern-rank-envelope.ts:28` read only the curves and rank output. The twin follows this behavior.

### likelihoods

- `modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring.ts:100` documents status 3 for unsupported covariance codes. With usable conditioning rows, the NaN derivative can instead fail the solve at line 287 while status remains 2 from line 237.
- `modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts:503` retains positive query-pivot cancellation residue. Unlike alias handling in `conditional-simulation-blocked.ts:133`, the cooperative path can therefore introduce small spurious variance for identical latent variables.
- `modules/analysis/src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes.ts:597` updates beta as `triggeredSum / lagSum`. For the finite-window likelihood, differentiating the offspring compensator also contributes `-omega * Σ remainingTime * exp(-beta * remainingTime)`. This appears to make the documented update an approximate M-step. The twin preserves that documented update.

### network-routing

- `modules/analysis/src/gpu-network/network-reachability/gpu-network-reachability.ts:162` says starts receive no predecessor. The strict-parent kernel at `modules/analysis/src/gpu-network/network-reachability/network-reachability-passes.ts:548` can assign a predecessor to a start with a strict tight incoming edge. The twin follows the kernel.
- `modules/analysis/src/gpu-network/alternative-routes/gpu-alternative-routes.ts:77`, `:87` and `:93` throw validation messages without the contributor ID; the output-length message also omits the affected port and expected length. These conflict with the construction-validation contract.

### raster

- `modules/analysis/src/raster-analysis/gpu-raster-fractional-coverage.ts:445`: the `1e-30` mean denominator floor changes a singleton’s mean when its positive coverage is smaller than that floor. For calibrated value 5 and coverage approximately `1e-32`, the GPU returns approximately 0.05 instead of 5. The CPU invariant test covers this difference.
- The same line scans every coverage tuple once per zone, contrary to the contract’s parallel aggregation rule. This remains an existing baseline finding.
- `modules/analysis/src/raster-analysis/gpu-raster-reconstruction.ts:48` and `:227` describe `maximumIterations` as the round budget, while `:149` schedules one additional round. That round also changes the published result when unconverged; this is a documentation ambiguity.
- `modules/analysis/src/raster-analysis/isolines/gpu-isolines.ts:150` documents that row 0 is placed at `minY`, independently of the grid affine. North-up inputs can therefore be mirrored. This is an existing documented limitation.

No existing contributor was changed.

### regions-allocation

- `modules/analysis/src/gpu-spatial-analysis/location-allocation/gpu-location-allocation.ts:1093`: finite costs at or above the rounded `1e30` sentinel are treated as unreachable, although the cost documentation identifies positive infinity as unreachable. The twin preserves this rule pending clarification.
- The same file at lines **1470** and **1516** uses different sparse/dense assignment comparisons. A cost exactly equal to the sentinel can receive a dense assignment but remains unassigned through the sparse path.
- The same file at line **1323** initializes the p-median candidate bound to `1e30`. An unreachable-demand weight exceeding that bound can prevent every candidate from being selected.
- `modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra.ts:88` describes a truncated row as a sorted prefix. The emit pass at `spatial-weights-algebra-wgsl.ts:200` truncates before the sort at line **207**. Higher-order output therefore retains a generation-order prefix and sorts that retained subset. The twin matches the implementation.
- `modules/analysis/src/gpu-spatial-analysis/spatial-weights/spatial-weights.ts:258` permits `candidateOverflow`, but algebra does not write it. The new operation omits that optional port.

### regression-error

- `gpu-spatial-error-gm.ts:107` promises NaN residuals on failure, but `fail()` at line 1301 clears only table and summary. Residuals written at line 732 can remain finite when the report fails. The twin follows the kernels.
- `gpu-spatial-error-gm.ts:215` says stochastic traces leave lambda and coefficient standard errors unaffected. The trace-dependent weight at lines 1104–1121 changes the second-step lambda, which also changes its filtered coefficient covariance.
- `gpu-spatial-error-maximum-likelihood.ts:267` describes raw power-trace overflow as a function of `maximumPower`. For ordinary finite weight scales, `GPUSpatialWeightsTraces` normalizes power chains by the infinity norm; its documentation at `gpu-spatial-weights-traces.ts:169` describes that scaling. The error-ML restriction appears stale.
- `gpu-multiscale-geographically-weighted-regression.ts:71` describes nonpositive variance as inference failure. `multiscale-inference.ts:87` accepts zero variance through `variance >= 0`. The twin follows the kernel.

The GPU also ignores `traceMode` for Kelejian–Prucha, so the CPU guard now covers both ignored trace settings. It still validates malformed `traceParameters`, matching the GPU constructor. No additional contributor defects were found.

### regression-held

`modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts:1392`: by inspection, non-finite inputs that invalidate the initial solve leave `W_DELTA` non-finite and select status 1. The documented non-finite status is 3. The twin mirrors the branch as written; GPU execution has not confirmed this observation.

### regression-linear

These were left unchanged.

| File and line | Observation |
|---|---|
| `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces.ts:454` | The norm merge selects scale 1 whenever the norm is at least `3e38`, including finite representable norms. Subsequent power chains can overflow despite the documented scaling guarantee. The CPU twin retains the finite norm. |
| `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts.ts:138` | `rhoBound` validation accepts positive infinity and numbers that round to float32 infinity. At line 206 the value is rounded, then interpolated into WGSL as `Infinity`, producing an invalid shader literal. The CPU twin rejects these bounds during construction. |

### spherical-cells

- `modules/analysis/src/gpu-spatial-analysis/cell-indexing/cell-index-families.ts:53`: `isGPUCellIndexFamily` uses `in`, which accepts inherited names such as `toString`. The following resolution validation can consequently accept an unsupported family.
- `modules/analysis/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay.ts:283`: deduplication checks only stored segments. After capacity is exhausted, duplicates of unstored segments can increment `required` again, potentially overstating `counts[2]`. This is a code-inspection finding; it was not GPU-tested.

Neither concern was changed.

### statistics-small

| File and line | Finding |
|---|---|
| `modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary.ts:449` | Initial extrema ±3e38 clip legitimate finite float32 values beyond those magnitudes. |
| `modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary.ts:583` | Categorical membership includes `0xffffffff` as an ordinary category, despite the contract identifying it as the missing-category code. The twin matches the kernel. |
| `modules/analysis/src/gpu-spatial-analysis/change-points/gpu-location-change-points.ts:612` | A lone finite Poisson observation receives cost zero. The documented cost gives `-8 ln(4)` for a lone observation of 4. The twin preserves this special case. |
| `modules/analysis/src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence.ts:223` | A 1×1 grid defaults to blur radius zero, then construction rejects it because the minimum radius is one. |
| `modules/analysis/src/dataframe-analysis/column-classification/gpu-class-breaks.ts:796` | `maximum - minimum` can overflow float32 for finite endpoints such as ±3e38, producing an incorrect equal-interval midpoint. |
| `modules/analysis/src/dataframe-analysis/column-classification/gpu-class-breaks.ts:999` | Maximum-break midpoint computation has the same subtraction-overflow problem; the published clamp can turn the midpoint into an endpoint. |

- The GPU penalty recurrence at `gpu-location-change-points.ts:495` and `:541` still permits cancellation. Device execution must determine whether reassociation preserves the residual.
- Signed-zero distinctness is method-specific: maximum breaks compares ordered keys, while exact natural breaks compares decoded values numerically (`exact-natural-breaks-nodes.ts:233`, `:285`). The twin preserves that distinction.

Only previously created files were changed.

### terrain

- `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:79` says a code without a receiver ends the stream. The seed kernel at lines 313–318 instead leaves status 2 for a nonzero code without an in-grid receiver. The twin matches the kernel.
- `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-breach.ts:88` and `:93` accept finite JavaScript props that overflow float32. They subsequently fail in `getWGSLFloatLiteral` at `:167` or `:208`, rather than producing a construction-time error naming the contributor and prop. This was identified by inspection.

### trajectory-large

- `modules/analysis/src/gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts:143`: the `frechetCapExceeded` documentation omits LCSS, although its kernel also skips capped pairs at line 954.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman.ts:368`: the unscaled determinant can overflow even when covariance entries and the inverse are finite, causing `smoothingSingular`. The CPU uses a scaled solve.

### trajectory-serial

- `modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split.ts:302` writes unclamped segment starts; line 430 clamps the terminal offset independently. Overflow can therefore produce nonmonotonic offsets such as `[0, 5, 3]`. Segment-capacity overflow also makes the terminal offset include rows from omitted segments. The twin preserves this.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split.ts:232` prefers the input `modes` buffer as prediction scratch. Supplying both modes and speed breaks mutates that input and can leave `output.predictedModes` unwritten. The operation omits this optional input during prediction.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split.ts:259` converts every nonfinite speed break to positive float32 maximum. An accepted first break of `-Infinity` consequently changes meaning. The twin preserves this conversion.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators.ts:458` counts active `0xffffffff` location keys as a category, affecting entropy and home/work selection. The twin preserves this.
- `modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators.ts:557` consumes tripleg columns without corresponding constructor format/length checks. Malformed tables can reach out-of-range shader reads; the CPU function rejects inconsistent arrays.

### validation-sampling

- `modules/analysis/src/gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts:286` publishes the nearest **surviving** neighbor distance as the NNDM radius. `modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts:260` then excludes distances `<=` that radius, excluding that survivor too. The hand-derived case produces empty training sets despite a positive minimum training fraction. The twin preserves this behavior.
- `modules/analysis/src/gpu-spatial-analysis/spatial-validation/spatial-validation-design.ts:264` publishes environmental assignments only after convergence. An iteration limit discards the available assignments, contrary to contributor-contract section 4.
- `modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts:74` does not validate candidate models, seed, or inner buffer distance during construction. Those checks occur through contributors created at lines 183, 243 and 257 during `getCommandNodes`.
