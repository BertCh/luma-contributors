# CPU twins that disagree with their GPU class

## Summary

I executed 79 named CPU twins across five families. I found 14 issues involving 15 distinct twins. GPU results below are source traces, never device measurements.  
The most ordinary mismatch takes 11 integers and a public parameter packer. The CPU returns `6` for the lower 70th percentile where the documented GPU rank selects `7`. The same discrepancy changes which rows pass a filter.  
A one-band raster medoid can select a different acquisition on an exact mathematical tie. Four ordinary decimal observations suffice. The provenance changes along with the pixel value.  
A point-pattern envelope can report a Monte Carlo p-value of `0.05` on CPU and, by shader trace, `1` on GPU. Both return success.  
The highest-value change is to make **discrete decisions** part of the twin contract. Ranks, selected IDs, membership masks, exceedance counts and status words need independent, exact expectations even when the associated floating-point estimate has a tolerance.

## Census

**COUNT.** These commands count files containing the public class declaration and the standard CPU departure comment, respectively. Both returned `128` in `r19`.

```sh
rg -l '^export class CPU\w+Operation' modules/analysis/src | wc -l
rg -l 'What differs from the GPU' modules/analysis/src -g 'cpu-*-operation.ts' | wc -l
```

| Source family | Public CPU operation classes | Distinct twins executed |
|---|---:|---:|
| `gpu-spatial-analysis` | 79 | 56 |
| `gpu-network` | 17 | 12 |
| `raster-analysis` | 13 | 5 |
| `gpu-terrain` | 11 | 5 |
| `dataframe-analysis` | 7 | 1 |
| `analysis-core` | 1 | 0 |
| `pointcloud-analysis`, `geospatial-analysis` | 0 | 0 |
| **Total** | **128** | **79** |

**COUNT.** I extracted each comment from `What differs from the GPU` to its closing `*/` and searched that text. The counts overlap. They measure what authors already explain in source, not how many implementations are defective.

| Wording in the departure comment | Classes |
|---|---:|
| `float32` | 112 |
| `float64` | 99 |
| `[Tt]ie` | 76 |
| `[Tt]hreshold` | 33 |
| `[Cc]utoff` | 11 |

**What ran.** I extracted self-contained literal inputs from `cpu-*-operation.node.spec.ts` using the TypeScript parser and called 72 real CPU implementations directly. I separately ran 27 twins on selected boundary or control inputs. Twenty names overlapped, giving 79 distinct twins. The first pass establishes executable coverage only. I used the boundary probes and source traces below for findings. No test runner, build, GPU or network was used.

| Family | Twins executed, with the common `compute…OnCPU` spelling omitted |
|---|---|
| Dataframe | ColumnQuantiles |
| Raster | CostDistancePath, RasterFillHoles, RasterGeometricMedianComposite, RasterMedoidComposite, RasterReconstruction |
| Terrain | ProfilePeaks, TerrainFlowDistance, TerrainPeakProminence, TerrainStreamOrder, TerrainWatersheds |
| Network | KShortestPaths, NetworkCentrality, NetworkFlow, NetworkLouvain, NetworkPathExtraction, NetworkPointRouting, NetworkRationalAccess, NetworkReachability, NetworkStructure, NetworkTimeRouting, NetworkTravelTimePercentiles, RadiationFlowModel |
| Spatial, A–M | BrownianBridgeUtilization, ContinuousSpatialSample, ConvexStraightSkeleton, DelaunayTessellation, EnsembleScores, GeodesicPairs, HealpixPointToCell, LabelPoint, LineLoopRemoval, LineSimplification, LocationAllocation, LocationChangePoints, MantelTest, MapColoring, MinimumBounds, MobilityIndicators, ModalSplit, MultipleTestingAdjustment, MultiscaleGeographicallyWeightedRegression |
| Spatial, N–R | NeighborhoodSummary, NormalizeGeometry, NormalizePolygons, OPTICS, OffsetCurveComponents, PettittTest, PointPatternEnvelope, PointToCell, PopulationSupportAlignment, ProbitRegression, RankMarkov, RatioConcaveHull, RestrictedRegimeRegression |
| Spatial, S–W | S2CapCovering, SegmentSnap, SinkhornDivergence, SpaceTimeHawkes, SpatialBalancedSample, SpatialCrossValidation, SpatialErrorGM, SpatialErrorMaximumLikelihood, SpatialLagImpacts, SpatialLagMaximumLikelihood, SpatialPanelLagMaximumLikelihood, SpatialTwoStageLeastSquares, SpatialWeightsAlgebra, SpatialWeightsTraces, SpatialWeightsValidation, TrackSimilarity, TrajectoryKalman, TrajectoryLocations, TrajectoryRecursions, TrajectoryRestriction, TrajectoryStaypoints, TrajectoryThinning, TransitionDiagnostics, WardRegions |

**Docs-first use.** The dataframe page specifies a float32 rank and gives the public parameter packer. A fallback caller can reasonably write the following. It runs successfully. The surprise is the answer, not a missing import or an unfamiliar buffer layout. See `docs/api/gpu-dataframe-analysis.md:368`, `:375`, `:390`.

```ts
const parameters = getGPUColumnQuantilesParameterValues({
  quantiles: [0.7],
  interpolation: 'lower',
  filterRange: [0.7, 1]
});
const values = Float32Array.from({length: 11}, (_, row) => row);
const result = computeColumnQuantilesOnCPU({values, parameters}, 1);
result.quantiles[0]; // Executed: 6. The page's rank expression selects 7.
```

## Proposed convention

For the same representable inputs and named method, a CPU twin and its GPU class should use the same input domain, decision precision, tie order and loss-status precedence. Check every discrete output against a literal or independently derived expectation on a fixture that distinguishes the competing rules. Apply numeric tolerances to continuous outputs separately. Give intentionally different estimators or approximations a public option that selects the same behavior on either backend.

## Findings

Paths in the details are relative to `modules/analysis/src` unless they begin with `docs`, `modules/analysis/test` or `node_modules`. **KI** identifies the newer `luma-contributors-ki` copy of a staging-owned file. Other implementation locations are from `r19`. For breaking changes, the direct-call inventory counts named calls and constructions in repository code, including tests and operation wrappers. **All proposed fixes require zero caller syntax edits.** The counts identify callers whose assumptions or expected results merit review.

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| TWN-01 | W | ORDINARY | BREAKING | S | ColumnQuantiles | A public packed `0.7` selects a different row and filter mask. |
| TWN-02 | W | ORDINARY | BREAKING | S | RasterMedoidComposite | An exact one-band tie can select the later acquisition. |
| TWN-03 | W | UNUSUAL | BREAKING | M | PointPatternEnvelope | Rounding the statistic before comparison changes a p-value from `0.05` to `1`. |
| TWN-04 | W, C | ORDINARY | BREAKING | S | ModalSplit | The documented speed breaks classify the same tripleg differently. |
| TWN-05 | W, C | UNUSUAL | BREAKING | M | Three spatial twins | One radius decision changes a track match, a location ID and whether a fold is usable. |
| TWN-06 | W | UNUSUAL | BREAKING | S | RasterGeometricMedianComposite | One retained observation can move during a GPU iteration. |
| TWN-07 | W | UNUSUAL | ADDITIVE | S | EnsembleScores | Both paths admit a negative variogram score from an invalid weight. |
| TWN-08 | W | UNUSUAL | BREAKING | M | TerrainPeakProminence | Calibrated integer elevations `0` and `1` can become a GPU plateau. |
| TWN-09 | W, C | UNUSUAL | BREAKING | S | ProfilePeaks | One rounded subtraction changes whether a peak exists. |
| TWN-10 | W | EXTREME | BREAKING | S | RadiationFlowModel | Finite-size normalization can discard the only destination on CPU. |
| TWN-11 | S | UNUSUAL | ADDITIVE | S | MantelTest, MobilityIndicators | CPU throws on row contents after an upstream loss flag has invalidated them. |
| TWN-12 | D | ORDINARY | ADDITIVE | S | RasterGeometricMedianComposite | The API page promises an index identity that fails for NDVI. |
| TWN-13 | C, D | UNUSUAL | ADDITIVE | S | BrownianBridgeUtilization | The GPU cutoff can remove the entire representable density of a cell. |
| TWN-14 | E | ORDINARY | ADDITIVE | S | ColumnQuantiles | The CPU supports 65 probabilities, but both public packers reject them. |

### TWN-01: A packed percentile changes the selected row

**Where.** `dataframe-analysis/column-classification/cpu-column-quantiles-operation.ts:92,110` (both copies), `column-quantiles-nodes.ts:186,193,263,267`. The CPU source comment already acknowledges possible rank differences. The additional finding is that a commonplace decimal changes both a value and an ordinary filter, with no rational-filter escape hatch.

**Evidence and current call.** **EXECUTED** the docs-first example above on both trees. Eleven rows give lower quantile `[6]`. With ten rows `0…9` and the same parameters, CPU filter bounds are `[6,9]`, retaining row `6`. **SOURCE** the GPU product is `f32(10 * f32(0.7)) = 7`. The corresponding answers are quantile `[7]` and filter bounds `[7,9]`. The rational probability input changes the quantile rank only. The smallest GPU probe is these two dispatches, reading one quantile and mask word `6`.

**Wanted.** Use the shader's float32 product before `floor`, `ceil` and nearest-rank decisions in the packed-float CPU path, including filter ranks. Preserve exact-integer rational ranks. Publish both fixtures beside the public formula. This is independent of the row-shape work in `KNOWN(DPC-117)`.

**Callers, value and cost.** 15 direct CPU calls and 3 CPU constructions, chiefly the column-classification tests and operation wrapper. A fallback user gets the same selected records. **S.**

### TWN-02: A one-band medoid loses the earliest acquisition on an exact tie

**Where.** `raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation.ts:49,62`, `gpu-raster-medoid-composite.ts:63,159,168,170`. The GPU class promises earliest-slice provenance.

**Evidence and current call.** **EXECUTED** one cell, one band, four slices, `Float32Array.of(.1,.2,.3,.4)`. CPU selects slice `1`, value `0.20000000298023224`, status `0`. For any four ordered scalar values the middle two have equal total absolute distance. Here both exact totals, using the stored inputs, are `0.4000000134110451`. An executed float32 transcription of the shader loop gives candidate totals `[0.6000000238, 0.4000000358, 0.4000000060, 0.6000000238]`. **SOURCE** the strict minimum selects slice `2`, value about `0.3`, status `0`. One GPU dispatch reading `slice` and `composite` confirms it.

**Wanted.** For one band, select the earliest valid acquisition whose value lies in the closed interval between the two central order statistics. This handles even counts, source order and duplicates directly. Add the fixture in chronological order and with the two middle observations exchanged.

**Callers, value and cost.** 13 GPU constructions, primarily raster-time-series tests and operation consumers. A user tracing a pixel to its acquisition gets the promised date. **S.**

### TWN-03: Rounding before the envelope comparison changes the statistical decision

**Where.** `gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation.ts:175,184,186` and `gpu-point-pattern-envelope.ts:229,232,245,251,255` (KI). The GPU operation spec uses a twin comparison at `modules/analysis/test/gpu-spatial-analysis/point-process/point-pattern-envelope-operation.spec.ts:73`.

**Evidence and current call.** **EXECUTED** `statistic: 'integrated-squared-deviation'`, specified null, no fitted parameters, observed `[1,2**-13]`, 19 simulated curves `[1,0]`, reference `[0,0]`, weights `[1,1]`. Both CPU copies return `statistics=[1,0.05000000074505806,0.05000000074505806,19]`, status `0`. The independently computed observed statistic is `1 + 2^-26`. All 19 simulated statistics are exactly `1`. **SOURCE** the GPU stores each statistic as one float32 before `value >= observed`, producing 19 exceedances and p-value `1`. At significance level `0.1` the inference reverses. Dispatch those 20 two-element curves and read `statistics[1]`.

**Wanted.** Preserve a residual through the exceedance comparison, for example by storing and comparing a two-word accumulation. Rounding the displayed statistic is fine. Casting a compensated sum back to one word before comparing would retain the bug. Assert the independent exceedance count of zero.

**Callers, value and cost.** 22 GPU constructions across point-process tests and operation consumers. Monte Carlo inference survives a change of backend. **M.**

### TWN-04: A tripleg changes transport mode at the documented speed break

**Where.** `gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation.ts:63,165`, `gpu-modal-split.ts:90,214,224,342,358,372`.

**Evidence and current call.** **EXECUTED** one tripleg from `(0,0)` to `(f32(15/3.6),0)` in one second, `metric: 'count'`, `modeSpeedBreaks: [15/3.6,100/3.6,Infinity]`. CPU predicts mode `0`, counts `[1,0,0]`. It compares `4.166666507720947` with the unrounded JavaScript break `4.166666666666667`. **SOURCE** the first GPU break equals the stored speed, so the exclusive upper edge gives mode `1`, counts `[0,1,0]`. Read `predictedModes[0]` on that leg. **EXECUTED** CPU also accepts `[NaN]` and `[-Infinity,Infinity]`; GPU construction rejects both. Both constructors accept `[.7,.7000000001,Infinity]`, whose first two breaks become identical as float32.

**Wanted.** Normalize breaks to float32 once in both constructors and the pure function. Check strict ascending order **after** rounding and allow positive infinity only as the terminal break. Compare the speed and break at the same declared precision. Use the actual documented `15/3.6` example as a regression fixture.

**Callers, value and cost.** 17 pure calls, 4 CPU and 13 GPU constructions, mainly trajectory-mobility tests and consumers. A published transport category acquires one boundary definition. **S.**

### TWN-05: The same radius discrepancy changes three different public decisions

**Where.** Under `gpu-spatial-analysis`, see `track-similarity/cpu-track-similarity-operation.ts:289,293,485,487` (KI), `track-similarity/track-similarity-dynamic-programs.ts:155,195,384`, `track-similarity/track-similarity-edit-programs.ts:53` (KI), `trajectory-staypoints/cpu-trajectory-locations-operation.ts:145` and `gpu-trajectory-locations.ts:344` (KI), and `spatial-validation/cpu-spatial-cross-validation-operation.ts:486,491` and `gpu-spatial-cross-validation.ts:278,304` (KI).

**Evidence and current call.** **EXECUTED** both copies on points `A=(0,0)` and `B=(1,2^-13)` with radius `1`. The exact squared distance is `1+2^-26`, strictly outside. The float32 squared distance rounds to `1`. **SOURCE** each GPU branch includes the boundary. These are three minimal GPU probes with different user-visible consequences.

| Operation and additional input | Executed CPU output | GPU output by source trace |
|---|---|---|
| TrackSimilarity, one vertex per line, compare A with B, epsilon `1`, default EDR and LCSS tolerances | Fréchet `1`, within `0`, EDR `1`, normalized LCSS `0`, status `0` | Within `1`, EDR `0`, LCSS `1` |
| TrajectoryLocations, two centroids on track `0`, parameters `[1,0,0,0]` | IDs `[0,1]`, two locations, converged | IDs `[0,0]`, one location |
| SpatialCrossValidation, groups `[0,1]`, two folds, buffer `1` | Training `[0,1,1,0]`, status `[0,0]` | Training `[0,0,0,0]`, status `[1,1]`, both folds lack training rows |

**Wanted.** Specify inclusive Euclidean membership on the stored coordinates. Keep the fast comparison when rounding cannot affect the decision and resolve ambiguous squared-distance comparisons with a residual. Use the same rule for all three planar operations. Document that the displayed Fréchet distance can round to the radius even when its membership flag is false. Assert IDs, masks and fold status independently.

**Callers, value and cost.** Direct GPU constructions are TrackSimilarity `61`, TrajectoryLocations `9`, SpatialCrossValidation `35`, largely their tests, operation consumers and examples. A GIS user can move a workload between backends without merging locations or losing a training fold. **M.**

### TWN-06: The geometric median of one observation can move

**Where.** `raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation.ts:70,85,92`, `gpu-raster-geometric-median-composite.ts:243,256,258,262,266`. The staged status oracle includes a singleton, but its chosen value does not expose this arithmetic.

**Evidence and current call.** **EXECUTED** `values=[456787.5]`, one cell, one band, one slice, one iteration, epsilon `1e-5`. CPU returns `456787.5`, last step `0`, status `0`. An executed float32 transcription gives `weight=100000` and `f32(f32(weight*456787.5)/weight)=456787.53125`. **SOURCE** the shader updates from this weighted absolute coordinate, permitting a spurious step of `0.03125`. Device division behavior warrants measuring the exact value. The smallest probe reads `composite` and the last-step output after one iteration. The independent expectation is exact identity and a zero step on every device.

**Wanted.** Return the sole retained observation directly. For the general iteration, accumulate weighted displacements from the current estimate and add their weighted mean to that estimate. Include an all-identical, nonzero stack as a companion invariant.

**Callers, value and cost.** 12 GPU constructions, chiefly raster-time-series tests and consumers. A single clear acquisition survives compositing exactly. **S.**

### TWN-07: A negative weight lets a worse forecast beat a perfect score

**Where.** `gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts:85,161`, `gpu-ensemble-scores.ts:100,417`. The GPU prop specifies nonnegative weights. I also checked the KI scoring-rule specs.

**Evidence and current call.** **EXECUTED** one ensemble member `[0,1]`, observation `[0,0]`, pair `[0,1]`, default variogram order `0.5`. Weights `-1`, `0`, `1` return scores `-1`, `0`, `1` respectively. A perfect forecast has zero discrepancy. **SOURCE** WGSL multiplies the squared discrepancy by the weight without inspecting its sign. A one-pair dispatch with weight `-1` tests both backends against the scoring rule's nonnegativity invariant. A twin comparison alone accepts `-1`.

**Wanted.** Handle negative and nonfinite weights in the existing per-pair evaluation. Return `NaN` for the requested variogram score on either backend and document the invalid-input result. Preserve other requested scores and valid zero weights.

**Callers, value and cost.** No valid caller changes. Forecast comparisons stop rewarding invalid inputs. **S.**

### TWN-08: Integer calibration can reverse which terrain peak survives

**Where.** `gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation.ts:64,115`, `gpu-terrain-peak-prominence.ts:129`, `terrain-peak-prominence-tree.ts:80,117`, `raster-analysis/raster-grid/raster-operation-band.ts:140` (KI). The installed upstream raster conversion is in `node_modules/@luma.gl/graph/src/gpu-raster/gpu-raster-neighborhood.ts:345`.

**Evidence and current call.** **EXECUTED** two adjacent `sint32` cells `[16777216,16777217]`, scale `1`, offset `-16777216`. Calibrated heights are exactly `[0,1]`. CPU gives `parentPeak=[1,NONE]`, prominence `[0,Infinity]`. **SOURCE** the native GPU path first casts each raw integer to `f32`, producing equal samples, then applies the offset. The plateau uses its lower cell ID, predicting `parentPeak=[NONE,0]`. A `2×1` GPU DEM reading the parent IDs distinguishes the cases without a float tolerance.

**Wanted.** Document the immediate workaround of subtracting the integer origin before upload. Fix the shared conversion upstream so small calibrated differences survive cancellation, then adopt the published implementation here. Test the parent IDs as well as the calibrated samples.

**Callers, value and cost.** 12 direct GPU peak-prominence constructions. The shared raster conversion has wider reach, which I did not quantify. Users of offset-encoded DEMs retain the actual summit. **M.**

### TWN-09: Profile prominence uses two precisions for a yes-or-no threshold

**Where.** `gpu-terrain/terrain-features/cpu-profile-peaks-operation.ts:136,137,160`, `gpu-profile-peaks.ts:254,596,597,690`. I checked the KI profile status fixtures.

**Evidence and current call.** **EXECUTED** profile `[2^-26,1,2^-26]`, offsets `[0,3]`, minimum prominence `1`, `window=minimumSide=capacity=1`. CPU gives count `0`, mask `[0,0,0]`, converged `1`. **SOURCE** the shader's prominence subtraction yields `f32(1−2^-26)=1`. The inclusive threshold accepts the center. Read the count and mask from this three-sample GPU profile.

**Wanted.** Use the documented float32 prominence in the CPU threshold and candidate ordering. State that the minimum applies to the reported precision. Test both acceptance and the winning index.

**Callers, value and cost.** 17 pure calls and 7 CPU constructions, chiefly terrain-feature tests and examples. A reported peak can be reproduced using its published threshold. **S.**

### TWN-10: The radiation CPU twin can lose the entire outflow

**Where.** `gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts:67,117`, `gpu-radiation-flow-model.ts:185,190,205`. `KNOWN(NET-111)` identified cancellation on GPU. The current GPU code sums the outside population directly. The reverse failure remains on CPU.

**Evidence and current call.** **EXECUTED** two zones one unit apart, populations `[2**54,1]`, outflows `[100,100]`, `normalization: 'finite-size'`. CPU returns row-major flows `[0,0,100,0]`. Populations `[1,2**-54]` reproduce it. With one non-origin destination, the finite-size expression simplifies to that origin's entire outflow. **SOURCE** the current GPU accumulation predicts `[0,100,100,0]` on the first fixture. Read both off-diagonal entries in a two-zone dispatch.

**Wanted.** Accumulate non-origin population directly in the CPU pass, as the GPU now does. Add the one-destination identity as an independent oracle. It catches this bug whichever backend changes next.

**Callers, value and cost.** 8 pure calls and 3 CPU constructions in flow-aggregation code and tests. An extreme population imbalance no longer erases a valid OD flow. **S.**

### TWN-11: An upstream overflow suppresses GPU results but can throw on CPU

**Where.** `gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation.ts:128,132` and `gpu-mantel-test.ts:697` (KI), `trajectory-mobility/cpu-mobility-indicators-operation.ts:164,328` and `gpu-mobility-indicators.ts:785,835,874`. This extends `KNOWN(PTS-104)` with two consumers that inspect invalidated row contents before returning their loss result.

**Evidence and current calls.**

- **MantelTest, EXECUTED on both copies.** Use `offsets=[0,1,2]`, `neighbors=[1]`, weights and distances `[1]`, times `[0,1]`, parameters `[0,0,3,0]`, maximum permutations `3`, upstream `overflow=[1]`. CPU throws `"offsets must be nondecreasing and within neighbors capacity"`. Changing the final offset to `1` while retaining overflow gives four NaN statistics, twelve NaN permutation values and zero draws. **SOURCE** GPU schedules terminal suppression. Run the first fixture and read the statistics and draw count.
- **MobilityIndicators, EXECUTED.** One timestamp, one staypoint, `startRows=[0xffffffff]`, `count=[1]`, `overflow=[1]`, one track. CPU throws `"staypoints.startRows must index timestamps"`. **SOURCE** the GPU suppression nodes overwrite affected dense outputs and report loss. Run that one row and read an affected float output and the loss word.

**Wanted.** Validate fixed buffer shapes first. When a producer declares its frame incomplete, publish the specified suppressed result before interpreting invalidated offsets or row IDs. Verify this precedence with both well-formed and unusable row contents under the same loss flag.

**Callers, value and cost.** No signature changes. Pipeline users receive the same recoverable status on CPU fallback. **S.**

### TWN-12: The raster page gives a false rule for computing NDVI

**Where.** `docs/api/gpu-raster/raster-time-series.md:223` in `r19`, `:232` in KI. Both say that an index computed after geometric-median compositing equals the index computed before it. The relevant implementations are `raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation.ts:70,85,92` and `gpu-raster-geometric-median-composite.ts:243,256,262`.

**Evidence and current call.** **EXECUTED** three `(NIR,red)` observations `[(1,1),(3,1),(1,2)]`, epsilon `1e-5`, 64 iterations. The two-band CPU composite is `[1.2545692921,1.3045036793]`, giving NDVI `−0.0195126859`. Per-acquisition NDVIs are `[0,0.5,−1/3]`, whose one-band composite is `0`. Independent SciPy minimization of total Euclidean distance gives `[1.25456932,1.30450369]`. The disagreement follows from the estimators themselves. **SOURCE** the GPU evaluates the same joint-band objective. Two one-cell dispatches, one per ordering, suffice to demonstrate it there.

**Wanted page text.** “The geometric median fits all bands jointly and can produce a synthetic spectrum. Compute NDVI from that spectrum when you want the index of the composite. To summarize acquisition-level NDVI, compute NDVI per acquisition and composite that scalar. These quantities can differ.”

**Callers, value and cost.** No code changes. A remote-sensing user can choose the intended estimand from the page alone. **S.**

### TWN-13: Seven-sigma truncation can remove all density in a cell

**Where.** `gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation.ts:46,108,114`, `gpu-brownian-bridge-utilization.ts:63,68,306`. The CPU comment discloses the cutoff difference. The GPU comment claims the omitted density is below float32 resolution **of the cell total**.

**Evidence and current call.** **EXECUTED** two stationary fixes at the origin, times `[0,1]`, one midpoint, motion variances `[4*s*s,0]`, zero location errors, one cell with extent `[7*s,-s,8*s,s]`. Its center is `7.5` standard deviations away. With `s=1`, CPU density is `9.7115336064e-14`. With `s=2^-16`, it is `0.0004171072`. Both match `exp(-7.5²/2)/(2πs²)`. **SOURCE** the GPU skips the segment in both cases and returns zero. Dispatch the second fixture and read its one density. A downstream log-density changes from finite to `−Infinity`.

**Wanted.** Correct the comment to describe a bound relative to peak density. Add `tailPolicy?: 'full' | 'truncate-at-seven-sigma'` to both forms, with identical behavior when specified. Document the existing defaults beside the option. This gives a caller an additive route to reproducibility.

**Callers, value and cost.** No existing caller must change. Users evaluating tails can select their approximation explicitly. **S.**

### TWN-14: Using the advertised CPU capacity requires hand-packing a private-looking header

**Where.** `dataframe-analysis/column-classification/column-quantiles-parameters.ts:54,61,121,128`, `cpu-column-quantiles-operation.ts:125` (both copies), `modules/analysis/test/gpu-dataframe/column-classification/cpu-column-quantiles-operation.node.spec.ts:60`. The test distinguishes CPU and GPU capacity at declaration.

**Evidence and current call.** **EXECUTED** `getGPUColumnQuantilesParameterValues({quantiles: Array(65).fill(.5)})`. It throws `"GPUColumnQuantiles quantileCount must be an integer in [1, 64]"`. The rational packer rejects 65 half-probabilities with the same message. For input values `[0,2]`, calling the CPU twin with `Float32Array.of(3,0,1,0,...Array(65).fill(.5))` succeeds and returns 65 ones. The caller has to discover the header order and the numeric interpolation code before using a supported CPU shape.

**Wanted.** Export backend-neutral `getColumnQuantilesParameterValues` and `getColumnQuantilesRationalProbabilityValues` that validate the data layout independently of GPU capacity. Keep the 64-slot check at GPU declaration and preserve the existing exports. Show the neutral packer in the fallback example.

**Callers, value and cost.** No existing caller must change. CPU users can use the extra capacity through named inputs. **S.**

## Departures by name

The 15 distinct operation pairs with findings are grouped below. A name can occur twice when separate fixes apply.

| Departure | Public operation names, common `CPU…Operation` and `GPU…` prefixes omitted |
|---|---|
| Rounding changes a rank, class, tie winner or membership result | ColumnQuantiles, RasterMedoidComposite, PointPatternEnvelope, ModalSplit, TrackSimilarity, TrajectoryLocations, SpatialCrossValidation, ProfilePeaks, TerrainPeakProminence |
| Arithmetic violates a simple estimator invariant | RasterGeometricMedianComposite, RadiationFlowModel |
| Both paths accept an invalid scoring weight | EnsembleScores |
| Incomplete-input precedence differs | MantelTest, MobilityIndicators |
| Approximation or usable capacity depends on backend | BrownianBridgeUtilization, ColumnQuantiles |
| Public method interpretation fails an independent example | RasterGeometricMedianComposite |

**Faithful on the particular boundary checked.** For these 12 twins, the executed CPU value matched the independent expectation and the GPU source branch I traced. Each row identifies the specific behavior checked.

| Twin | Separating input | Executed CPU result consistent with GPU source |
|---|---|---|
| PettittTest | `[0,NaN,0,4,4]` | `K=4`, first maximizing split after three input slices, p about `0.60238844` |
| RankMarkov | Two periods of tied `[4,4]`, `tieOrder: 'row-id'` | Ranks `[0,1,0,1]` |
| TransitionDiagnostics | Two-class counts `[1,1,1,1,999]` with one condition | Stationary probabilities `[.5,.5]`, capacity tail ignored |
| SpatialWeightsValidation | Offsets `[0,1,1]`, slots `[0,5]`, weights `[0,NaN]` | Row flags `[8,0]`, one invalid row, unused tail ignored |
| MultipleTestingAdjustment | Benjamini–Hochberg on `[.01,.01,NaN,.04]` | About `[.015,.015,NaN,.04]`, three tests |
| RasterFillHoles | `3×3` ring of ones around zero, four-connectivity | All ones, convergence reported |
| RasterReconstruction | Dilation marker `[1,0,0]`, mask `[1,1,1]`, one allowed round | `[1,1,0]`, iteration limit reported |
| CostDistancePath | Two cells, directions `[0,16]`, target `1`, capacity `1` | IDs `[1]`, emitted `1`, required `2`, overflow `1` |
| TerrainWatersheds | Directions `[1,0]`, duplicate pour cells `[1,1]` | Labels `[0,0]`, first duplicate wins |
| TerrainStreamOrder | Directions `[1,0]`, membership `[1,1]` | Orders `[1,1]` |
| TerrainFlowDistance | Directions `[1,0]`, maximum steps `1` | Distances `[NaN,0]`, terminals `[0xfffffffe,1]` |
| NetworkStructure | CSR offsets `[0,1,2]`, neighbors `[1,0]` | Both vertices in component `0` |

## Quick wins

- [ ] Replace the geometric-median index sentence with the two NDVI workflows and the three-observation example from TWN-12.
- [ ] Add the backend-neutral quantile packers and the 65-median CPU example from TWN-14.
- [ ] Handle invalid variogram weights in the existing pair loop and assert a zero-weight control alongside the negative case.
- [ ] Give MantelTest and MobilityIndicators an overflow fixture whose row contents would fail ordinary validation. Assert their documented suppressed outputs.
- [ ] Correct the Brownian density bound and expose the tail policy on both forms.
- [ ] Add the literal expectations from TWN-01 through TWN-06 as paired CPU and headless fixtures. The expected rank, slice, exceedance count, mode, mask and singleton value can all be derived without running either implementation.

## Larger proposals

1. **Make discrete parity a pre-alpha contract.** Start with the local fixes for quantile rank, modal break packing, profile prominence, scalar medoid selection, singleton compositing and radiation normalization. Their behavior changes are breaking for affected inputs, but none requires an in-repository API migration. A single “CPU uses float64” disclaimer cannot tell a caller whether a provenance ID or fold mask is portable.
2. **Spend extra precision at the decision that needs it.** For the envelope, rounding both backends to the present GPU statistic would make them agree on an independently incorrect exceedance count. For the planar radius predicates, retaining the residual only on ambiguous comparisons preserves the stated Euclidean boundary. I recommend these targeted changes before any package-wide arithmetic rewrite.
3. **Fix native raster calibration at its owner.** An analysis-side terrain tie rule cannot reconstruct a distinction already lost during raster conversion. Document integer recentering now, take the calibration case upstream, then verify the two parent IDs against the published fix.

## Not read

I read the scalar medoid CPU and GPU implementations, their observation-selection code, and the Pettitt CPU implementation in full. For the other detailed findings I read the relevant input handling, decision loops, output or suppression branches and adjacent documentation. I sampled the wider census by extracting defaults, limits, comparisons and termination branches, then followed delegated WGSL where a concrete candidate emerged. Executing a twin on a literal fixture does not establish parity for its other methods or layouts.

I searched operation specs for GPU construction and `compute…OnCPU` comparisons, then checked adjacent literal expectations and available independent oracles. The modal-split and point-pattern operation specs contain twin comparisons. The raster medoid already has a one-dimensional order-statistic test. I did not establish that any whole family has **only** GPU-versus-twin tests. Independent checks here used the scalar medoid identity, squared Euclidean distance, Monte Carlo exceedance counting, the two-zone radiation identity, variogram nonnegativity, Gaussian density and SciPy minimization. I checked candidates against the earlier synthesis, area reports, API/catalog and direction reports, and decision list. Staging-owned implementation and documentation findings above use the newer copy as indicated.

The remaining 49 named CPU twins were not executed. I did not audit every shader variant, device limit, multi-band medoid tie, geographic-distance implementation or downstream application. No GPU result in this report was measured.