# Changes for callers: Spatial statistics, regression, weights, autocorrelation and permutation inference

Part of [the record of caller-visible changes](README.md) from the 2026-11-19 and 2026-11-20 fix passes. Each bullet condenses what the implementers of those passes recorded; [README.md](README.md) says how the changes were tested and what that does and does not establish.

Counts in this file: results 53, rejections 42, added 31, deprecated 5, docs 16.

## Results that change

**Global statistics and permutation inference**

- `GPUGlobalSpatialStatistics` and `GPUGlobalPermutationTest`: sparse General G keeps its off-diagonal
  mass on both the analytic and permutation routes, so sparse nonnegative inputs that returned zero,
  NaN or an inaccurate denominator now return the correct statistic. The analytic variance, z and
  normal probability are stable as well (the two-positive case gives variance 0.25, z = 1). The
  negative-value rejection counts the original signs, so underflow during scaling can no longer
  hide a negative value.
- `GPUGlobalPermutationTest`: global bivariate Moran and Lee's L no longer change when measurement
  units change, and exact power-of-two scaling keeps closely spaced observations distinct (values
  45 + [0, 1, 2, 3]·2^-18 now give Moran 0.4 and Lee 0.2).
- `GPUGlobalSpatialStatistics` global Geary on complete-support weights: C stays exactly 1 for uniform
  weights. A very small positive C comes from the direct quadratic numerator instead of being
  rounded to zero through 1 + (C − 1), and tiny positive randomization variances are kept. Weights
  near 2^-112, where the ordinary float32 ratio underflows, now return finite C, variances and z
  instead of NaN.
- `GPUQuantileLISA`: the inclusive alternatives (`'greater'`, `'two-sided'`) keep tied draws and
  compare the exact count ratio with the float32 cutoff the GPU holds. `'geoda-legacy'` keeps the
  former calculation, except that an exact 1/25 is no longer significant at the packed .04 cutoff,
  and classes follow the corrected flag. An invalid local cutoff held on the GPU now suppresses
  inference and sets the aggregate overflow and optional `invalidInput` until a clean resubmission.
- `GPULocalPermutationTest` local Getis-Ord G: when one focus dominates, the normalization
  complement keeps small observations. For example, a ten-row fixture's row-zero statistic changes
  from 1 to 0.9999995231628418, and long runs of small values are kept even when the main sum is
  near 2^24. Replay output stays in raw-statistic units.
- `GPULocalPermutationTest`, `GPUGlobalPermutationTest` and `GPUQuantileLISA` honour positive
  `weights.status.complete` certificates on the root, chunk roots and children: zero suppresses
  inference, and the next corrected submission recovers. Final and candidate capacity diagnostics
  keep separate meanings.
- Permutation budget planning: the rejection floor is computed from the p-value construction
  separately from sidedness, so the .05 upper-tail floor is reached at 19 draws. Inverse estimates
  use the forward cutoff, so the three-row, .15 example needs 40 draws rather than 39.
  Benjamini-Yekutieli advice resolves harmonic boundaries exactly: at 73 rows and
  alpha = 0.03808209987270469, 255 two-sided draws cannot reject and 256 are required.

**Local statistics and rates**

- `GPUEmpiricalBayesRates` and `GPUSpatialEmpiricalBayesRates` exclude negative event counts from
  focal results and from prior fitting, and count them in `invalidCount` when requested. Negative
  missing-data sentinels no longer shift the smoothed rates of valid neighbours.
- `GPUSpatialPearson` and the centered `GPUGammaStatistic` keep the sign and magnitude of small
  variations around large float32 baselines. Gamma's edge centering uses an eligible reference only
  where subtraction is exact, so the distant-minimum case changes from 0 to 1.
- `GPULOSHResidualPermutationTest`: exactly harmonic neighbourhoods keep a zero residual instead of
  a subtraction-rounding artifact, while genuinely small residuals stay nonzero. Normalized
  residuals and downstream LOSH inference can change.
- `GPUMultivariateLocalGearyPermutationTest` (both strategies) and the common Moran mask honour a
  supplied false completeness certificate (`weights.status.complete` or
  `membershipStatus.complete`) and suppress inference.
- `GPUColocationQuotient` and `GPULocalCategoryDependence` suppress dependent results to NaN when
  producer overflow flags are set or a supplied `status.complete` is zero, and recover when the
  flags clear. Category counts and excess storage are kept. Co-location now gives unit neighbour
  shares for all-positive subnormal neighbourhoods on devices that divide by reciprocal, including
  Metal, instead of nonfinite shares.

**Multiple testing**

- `GPURasterFalseDiscoveryRateMask` and raster correction classify p-value bits before arithmetic.
  A negative subnormal p-value is excluded instead of being flushed to a valid zero, which can
  reduce inflated Bonferroni adjustments of other rows on flush-to-zero devices.

**Spatial weights**

- `GPUSpatialWeightsTransform`, `GPUChunkedSpatialWeightsTransform`, `GPUSpatialWeightsTranspose` and
  `GPUChunkedSpatialWeightsTranspose` pass the producer's `status.complete` and `neighborCounts`
  through. The chunked transpose folds `complete = 0` into its `candidateOverflow`, so downstream
  statistics no longer treat incomplete weights as complete.
- Transposed `double` normalization metadata is now `'unknown'` unless the output is square and
  the input is declared `symmetric: true`.
- A chunk whose provenance or information field is present but `undefined` no longer erases the
  matrix value. For example, an adaptive kernel after a binary transform keeps `criterion: 'knn'`.
- Chunked row, binary and kernel transforms now build when each chunk fits even though the whole
  matrix exceeds a storage-binding limit. Whole and chunked transforms no longer reject unused
  neighbour, distance or status bindings whose alignment prefix exceeds the device limit.
- CPU contiguity and weights algebra results keep their loss metadata. An incomplete flat result
  that was analysed silently now carries the algebra loss, or a regression consumer rejects it.
  Algebra propagates positive completeness independently of candidate flags, including from
  `GPUNetworkCostWeights`.
- `decodeGPUAnalysisContiguityStatistics` reports an exact profile only when validity is exactly 1.
  Validity 2 or 0xffffffff no longer certifies a profile, either positive loss flag (flat or nested)
  prevents exactness, and an exact profile with a nonfinite median now throws.
- Capacity planning treats an omitted observation as unknown rather than as an exact count sized to
  the allocation. Candidate-limited contiguity observations grow candidates before final slots are
  priced.
- Native Polygon topology preparation keeps valid rows, batch alignment and original-coordinate
  equality, including coordinates that coincide only after float32 rounding. Empty polygon
  hierarchies and zero-row regression CSR inputs are accepted, and regression fits on them report
  their existing insufficient-support status.
- CPU spatial weights traces recognise whole numerical CSR by its typed-array shape, so attaching
  clean loss metadata no longer prevents traces or whole-CSR Gram moments.
- `GPUSpatialWeightsValidation` accepts flags in a disjoint range of a distances or `neighborCounts`
  buffer, which it refused before.

**Local regression**

- `GPUGeographicallyWeightedRegression` and `GPUGeographicallyWeightedRegressionNonstationarityTest`
  solve on response differences from the focal observation. Large response offsets no longer
  contaminate local slopes, residuals, R² or nonstationarity spreads, and a compensated
  reconstruction keeps small intercepts after large origins cancel.
- GWR grid extents publish four NaNs for zero input rows and keep the true minimum and maximum
  for finite coordinates beyond ±3.0e38.
- `GPUGeographicallyTemporallyWeightedRegression` (dense and indexed) computes residuals, summaries
  and bandwidth-search scores from centered fit-time residuals. Callers can now see finite nonzero
  RSS, likelihood and AICc where rounded fitted values used to report a false exact fit. Fits use a
  shared neighbourhood response origin, so large mixed-sign focal outliers no longer give different
  fits at identical designs, and negligible-weight extreme responses no longer set the origin.
  Exact fits still publish infinities.
- `GPUMultiscaleGeographicallyTemporallyWeightedRegression` keeps exact large-intercept lines
  through backfitting, scores component searches from fit-time residuals and publishes the final
  error state.
- `GPUGeneralizedGeographicallyWeightedRegression` (Poisson): a change of exposure units, including
  a fractional one, no longer causes a spurious initialization failure. Leave-one-out fitting and
  bandwidth selection use positive-weight training anchors, so query predictors no longer change
  training rank and valid extrapolating fits and bandwidth winners are kept.
- GWR Student t inference uses a shared host numerical core that is more accurate at low degrees of
  freedom.

**Global regression**

- `GPUOrdinaryLeastSquares` and `CPUOrdinaryLeastSquaresOperation`: log likelihood, AIC and BIC now
  depend on the response unit for very small positive residual variances. Exact zero RSS keeps its
  separately stated finite convention.
- `GPUProbitRegression` and both operation forms certify convergence per coefficient: every accepted
  step and every full direction must be at most `min(tolerance, 1e-5) · (1 + |coefficient|)`. Loose
  tolerances and fits with one large coefficient can take more iterations, return a corrected
  estimate (a review fixture's slope moves from 2.2803545 to 2.3263478) or return status 3 when the
  budget runs out, where they used to report status 0. Any positive finite `tolerance` is accepted,
  including 1e39 and `Number.MAX_VALUE`, which used to throw. On the GPU, tiny tolerances are floored
  at 2^-126 instead of becoming 0.
- `GPUProbitRegression` on the GPU treats a subnormal response word as nonbinary (status 4), as the
  CPU does, instead of fitting it as 0.
- `GPUSpatialErrorGM`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialLagMaximumLikelihood` and the
  error ML estimator consume `weights.status.complete`: zero suppresses results and one permits them.
- `GPUSpatialWeightsTraces`, `GPUSpatialRegressionDiagnostics`, local collinearity and
  `GPUInstrumentalVariablesCovariance` (IV/HAC) suppress their result on a zero `status.complete`
  certificate and recover when it is restored. HAC loss flags are honoured even without
  `requiredCount`.
- `GPUSpatialRegressionDiagnostics`: a residual Moran test whose projected weight operator vanishes
  (for example K(8,8) with a partition design) reports variance 0 and an undefined z and p instead of
  a spurious significant p. On the GPU, LM tests on weighted graphs center fitted values before
  forming the lag, so an exactly constant fitted lag gives J = T and undefined robust LM tests.
- `GPUSpatialErrorGM` on the GPU finds exact affine fits whose intercept numerator is not a float32
  (such as 7y = x + 16777217) and returns status 3 as the CPU does.
- `GPUInstrumentalVariablesCovariance` makes rank decisions from mean-centered moments, so one
  first-row outlier no longer makes a well-determined column look singular.
- `GPURestrictedRegimeRegression`: incomplete HAC kernels now give robust status 2 with suppressed
  inference, including when the allocation has spare capacity. An unresolved joint contrast rank or
  exact-fit residual roundoff gives robust status 3 instead of false joint significance. Dense HAC
  contrasts that the conservative error bound suppressed can now reach status 0. The classical
  insufficient-row status stays 1 whether or not robust inference is requested, and joint tests
  are stable under row permutations.
- Spatial SUR (`GPUSeeminglyUnrelatedRegression` spatial route) centers predictors throughout
  filtering and GLS. Structural residuals at large predictor origins and intercept covariance across
  equations are corrected. In the status decoder, word 4 now reads `failed: false`.

**Panel regression**

- `GPUSpatialPanelRegression`: pooled LM diagnostics reduce score and rank terms with tiled pairwise
  merges, and rank detection uses centered lag variation, so the rank decision no longer depends on
  period order. Pooled and unit fits honour candidate-only weights loss and positive completeness
  certificates, and recompute when they clear.
- `GPUSpatialPanelLagMaximumLikelihood`: weight counts describe the accessible CSR slots, including
  a leading unused slot, and overstated counts suppress the fit and the Hausman results. Constant
  lag-model predictions keep zero slopes and an undefined (NaN) pseudo R² instead of correlations
  created by lag-fit rounding.
- `GPUSpatialPanelErrorMaximumLikelihood` rejects exact-likelihood poles consistently when predictor
  units change. Likelihood searches count stationary upper boundary maxima even when the last grid
  cell contains an intervening sign change of the derivative.

**Validation, conformal and group statistics**

- `GPUSpatialCrossValidation` (NNDM) keeps the neighbour that defines each adaptive exclusion
  boundary excluded, using its exact squared distance, so a rounded radius no longer admits it. Ties
  in neighbour ordering are resolved from stored coordinates. Training masks change in the recorded
  boundary cases on CPU and GPU.
- `GPUSpatialCrossValidation` buffer membership uses stored coordinates: points strictly outside a
  radius stay eligible for training even when float32 squared distances round to the radius.
  Equality is still included, and a zero buffer compares coordinate bits.
- `GPUSpatialCrossValidation` environmental folds subtract before scaling and update centroids from
  deviations. Close distinct groups, fractional centroid updates and tiny seed separations are kept,
  instead of reporting an infeasible partition (status 2) with all rows discarded or a wrong fold.
- `GPUGeoConformalPredictionIntervals`: supplied neighbour loss suppresses localized intervals to
  NaN with status bit 32, instead of publishing a falsely narrow finite interval. Suppression clears
  the unbounded-endpoint bits, so a previously unbounded row with upstream loss reports word 32
  rather than 35.
- `GPUGeographicalDetector` returns exactly q = 1 (F = +Infinity) for a defined partition with zero
  within-stratum variation. Uniformly tiny responses are normalized before moments, so nonperfect
  partitions keep a finite q and F (a fixture changes from q = 1, F = Infinity to q = 5/7, F = 5).
- `GPUSpatialEntropy` uses pairwise accumulation, which removes spurious mutual information on
  uniform independent category tables.

**Ensemble scores**

- `GPUEnsembleScores` and `CPUEnsembleScoresOperation`: fair CRPS and one-dimensional energy keep
  small observation errors for widely separated members (the two-member examples return 1 instead of
  0). The GPU variogram score uses a common exponent frame, so perfect finite forecasts score 0 and
  extreme representable scores stay finite. High-order perfect CPU forecasts also score 0.
  Negative, nonfinite or negative subnormal pair weights and nonfinite endpoints give NaN on the
  variogram only, and other requested scores stay available.

**Classification, distribution, interaction and transitions**

- `GPUClassificationFit`: opposite finite extreme endpoints whose difference overflows float32 give
  the zero class and pooled median instead of Infinity.
- `GPUGeographicDistribution`: the short axis of the standard-deviational ellipse (`ellipses[1]`,
  `ellipseVertices`) comes from a centered transverse projection. Narrow rotated groups keep a
  nonzero short axis, and exactly collinear groups get exactly 0.
- `GPUSpatialInteraction`, `computeSpatialInteractionOnCPU` and `CPUSpatialInteractionOperation`
  (bounded calibration): an exact fit found by the last allowed refinement reports status 0 and
  calibration `[β, β, 0, 1]` instead of status 4. CPU proportional fitting can detect a genuine cycle
  at a different iteration.
- `GPUTransitionDiagnostics`, its CPU twin and both operations compute Q and LR from exact integer
  count-product residuals. Near-independent tables with counts around 2^27 return their tiny
  positive statistic instead of 0, and regimes with identical proportions give Q = LR = 0 with exact
  pooled counts.

## Input now rejected at construction

**Permutation inference and autocorrelation**

- `GPULocalPermutationTest` and `GPUGlobalPermutationTest` reject loss metadata that is not packed
  uint32 in whole or chunked weights, an empty chunked outer `overflow`, and a wrong-format or empty
  chunked outer `candidateOverflow`. Each loss view must be 1 to 4,294,967,295 words long. The shared
  `validateCapacity` also rejects longer views before device preflight.
- A chunk-root `weights.status.complete` must be an exactly one-word packed uint32 view, as for whole
  and child certificates. A larger allocation can still be passed through a one-word offset view.
- Completeness metadata without the alias that gives its failure polarity is refused rather than
  dropped.
- `GPULocalPermutationTest` counts optional cause writers in its 32,768 command-node budget. For
  example, 32,509 rows with one whole CSR and `rowTileSize: 1` reach 32,769 with an `invalidInput`
  writer and are rejected. Use larger row tiles or a smaller graph.
- `GPUQuantileLISA` requires an explicit `alternative`: `'greater'`, `'two-sided'` (ties inclusive) or
  `'geoda-legacy'` (the former calculation).
- `GPUGlobalSpatialStatistics` rejects disagreeing `analyticAlternative` and `alternative` and an
  explicit `analyticAlternative: null`. Omit the prop to get the default.
- The permutation inference-metadata decoders and rejection-budget inversion reject internally
  inconsistent claims, such as impossible fixed-count readbacks or stopping readbacks below target.
- `GPUSpatialMarkov` rejects invalid transition settings and strided optional `lagValues` or
  `lagClasses` before reserving scratch, so a corrected `graph.add` with the same ID can follow.
- `GPULocalMoran` rejects `quadrantGating` values other than `'analytic'` and `'none'`.
- `GPUMultivariateLocalGearyPermutationTest` rejects conflicting old and new settings aliases,
  unknown or empty tail values, and a family prop without a Bonferroni mode. Decoding inference
  metadata for settings held on the GPU requires the current settings or packed `Uint32Array` words,
  and omitting them throws.
- `GPUColocationQuotient` and `GPULocalCategoryDependence` reject a malformed `status.complete` view
  and outputs that share buffers with status inputs. `requiredCount` without `overflow` is still
  refused.
- `GPULocalOutlierFactor` rejects a density-floor context whose effective distance unit or metric
  parameters differ from the search context (for example implicit metres against a kilometre floor).
  Convert the floor and declare matching units.

**Multiple testing and impacts**

- The CPU adjustment and spatial-lag impacts constructors reject empty input names, empty or duplicate
  output destinations and input/output collisions, including an explicit `status: ''` on CPU impacts.
  `GPUSpatialLagImpactsOperation` now rejects `status: ''` too. Omit the optional input or give a
  nonempty name.
- Adjustment APIs that require `method` reject its omission. `GPUMultipleTestingAdjustmentOperation`
  rejects an unknown method at construction and lists the accepted strings; it used to fail at
  lowering.
- Count aliases of the same quantity must agree on destination name or physical first word, and new
  count views must be exactly one word. A direct legacy `testedCount` keeps its oversized-view
  allowance.
- `GPUMultipleTestingAdjustmentOperation` validates `rowCount` in [1, 2^31] and `minimumFamilySize` in
  [0, 2^31] at construction.
- The raster false-discovery alpha packer rejects an undersized target before writing.

**Spatial weights**

- `GPUContiguityWeights` (ordinary polygon contiguity) rejects `output.distances` and `output.status`
  fields it cannot write. `GPUContiguityWeightsOperation` rejects distance destinations and
  unsupported status ports, including inherited ones. `CPUContiguityWeightsOperation` rejects
  distances, count and converged requests that it used to discard. Pass only `overflow`,
  `requiredCount` and `candidateOverflow` to contiguity, and keep any `invalidCount` on the caller's
  own binding.
- Weights algebra rejects named output status without a writer and unknown operation names.
  Negative-only loss readers reject positive certificates. Region `sourceIds` must be uint32.
- `GPUSpatialWeightsAlgebra` and `GPUChunkedSpatialWeightsAlgebra` reject operands whose metric results
  differ in metric, effective length unit, radius or ellipsoid, including `metric: 'none'` against a
  metric. Kernel transforms reject incompatible metric contexts.
- Chunked symmetrize rejects a destination in the neighbours' physical buffer. Validation rejects
  flags that overlap published `neighborCounts` or distances.
- Weight transforms validate a supplied kernel name even on non-kernel operations, and the pure
  evaluator rejects an unknown name before it considers a nonfinite z.
- `CPUSpatialWeightsTracesOperation` rejects flat completeness and capacity ports (`complete`, `count`,
  `requiredCount`, `overflow`, `candidateOverflow` and prefixed overflow aliases) on roots and children.
  Before, a `candidateOverflow` port could be ignored and an incomplete trace published. Use the pure
  CPU function with concrete certificates, or `GPUSpatialWeightsTraces` with GPU views.
- `GPUSpatialLag` rejects a new status output that aliases any input status metadata.
- `GPUSpatialWeightsValidation` and the CPU validator reject invalid column counts, diagonal policy
  strings and required diagonals that are impossible.

**Regression**

- Regression constructors reject a foreign, aliased or oversized `weights.status.count` supplied
  without `requiredCount`.
- `GPUGeographicallyTemporallyWeightedRegression` (dense) counts its added residual and statistics
  passes against the 32,768 command-node budget. Some large nested searches (notably
  `maximumIterations: 32`) that used to fit now reject.
- `GPUMultiscaleGeographicallyTemporallyWeightedRegression` estimates commands over its complete
  initialization and summary graph and rejects configurations above the unchanged 32,768 ceiling. It
  also rejects inherited unsupported cutoff and route options, as it does own properties.
- Both probit operation forms reject an explicit empty `mask: ''`, `spatialLag.weights.overflow: ''`
  or `spatialDiagnostics.weights.overflow: ''`, which they used to discard. The GPU operation now
  rejects invalid tolerances at construction.
- The `GPUOrdinaryLeastSquares` parameter packer throws, before changing the destination, when a
  positive finite `ridgeLambda` becomes zero, subnormal or infinite in float32. The smallest
  applied penalty is `MINIMUM_GPU_ORDINARY_LEAST_SQUARES_RIDGE_LAMBDA` (2^-126).
- Regression model readers throw an error naming the reader and status path for empty or
  partial-byte primary or covariance status readbacks.
- The GWR and GTWR summary readers reject numeric arrays with a missing or non-numeric word in the
  first 12 slots, where they used to return undefined fields. The `selectedBandwidth` reader rejects
  a missing word in its two-word prefix. Malformed byte lengths are rejected.
- `GPUMultiscaleGeographicallyWeightedRegression` construction errors name the mismatched port and
  its required shape or work limit. Accepted shapes are unchanged.

**Validation and group statistics**

- `GPUGeoConformalPredictionIntervals` rejects any `neighbors.status.requiredCount`, which used to be
  discarded silently. Check the requirement against capacity, omit the port and pass both producer
  loss flags. `decodeConformalStatus` rejects unsupported typed arrays and DataViews.
- `GPUSpatialCrossValidation` and both its operation forms reject an unknown `design.mode`. Accepted
  values are `'groups'`, `'environmental'`, `'nndm'` and `'knndm'`.
- `GPUSegregation` rejects unknown `spatialForm` values.
- `GPUGeographicalDetector` rejects `labelsComplete` input without its paired output, and the reverse.

**Ensemble scores**

- `GPUEnsembleScores` and its operations reject conflicting `estimator` and `fair` settings, unknown
  estimator strings and a non-boolean `fair`. They also reject unknown output keys and an empty GPU
  operation output selection. The GPU operation rejects missing variogram pairs, weights without
  pairs and fair scoring with one member, as the CPU operation does.

**Classification, Markov and transitions**

- `GPUClassAssignment` rejects unknown top-level keys, including those on widened objects.
- `GPUGeographicDistribution`'s output type requires at least one selected terminal view, so an empty
  output selection is now a compile error (it already threw at runtime).
- `GPURankMarkov` rejects empty-string `mode` and `tieOrder` values, which it used to treat as defaults.
- `GPUTransitionDiagnostics`, `GPUTransitionDiagnosticsOperation` and
  `CPUTransitionDiagnosticsOperation` read inherited and getter-defined `minimumTransitions`. An
  inherited out-of-range value now throws instead of falling back to 5.

## Added

**Permutation inference and autocorrelation**

- Local and global permutation tests: optional independent loss-cause ports, a `GPUQuantileLISA`
  tested-row port that separates suppressed inference from a nonsignificant result, and per-row
  `permutationCounts` for fixed-count local tests (untested rows included).
- `decodeGPULocalPermutationTestInferenceMetadata` and `decodeGPUGlobalPermutationTestInferenceMetadata`
  report requested draws separately from actual readback counts in named fields.
- Permutation budget planning reports the required draw count and whether the contributor and
  global row-block budgets can reach it.
- `GPUGlobalSpatialStatistics` analytic route: an optional `invalidCount` that counts the negative
  values that invalidate General G, while Moran and Geary stay available.
- `GPUMultivariateLocalGearyPermutationTest`: `significanceLevel`, `permutationCount`, numeric
  nonnegative safe-integer seeds, `pseudoPValues` and `adjustedPValues` outputs, and an explicit
  choice between the tested and complete Bonferroni family. Metadata publishes the effective
  float32 threshold and the fixed draw count.
- `GPULocalMoran.createCommonMaskNodes` combines values completeness and membership into the
  row mask that analytic and permutation inference share, keeping membership diagnostics.
- `GPUColocationQuotient` and `GPULocalCategoryDependence` accept producer `overflow` and
  `candidateOverflow` views, and `requiredCount` paired with `overflow`. Such loss-bearing calls
  used to throw.

**Multiple testing and impacts**

- All adjustment forms can publish the observed test count separately from the effective family
  size, with documented aliases. `testedCount` keeps its meaning as the family size used for
  correction.
- Named CPU result types for adjustment, lag impacts and weights validation, and an exported
  constant for the 64-term harmonic sum bound that CPU and GPU Benjamini-Yekutieli share.

**Spatial weights**

- `GPUSpatialLag` gains an optional `status: {overflow?, candidateOverflow?}` output that reports
  final-edge and candidate loss separately, chunks included, and resets on each submission.
- `GPUSpatialWeightsValidation`, its operation forms and the pure CPU validator (optional fourth
  argument) accept a declared source-column count for rectangular weights and an allowed or required
  unit-diagonal policy. The defaults stay square and diagonal-free.
- `'quartic'` is accepted as a synonym for `'bisquare'` in the spatial weights kernel, both transform
  constructors and the pure evaluator.
- Six GPU weights producer routes accept an optional count destination for retained counts, chunk-local
  where applicable. CPU contiguity and algebra results gain nested status aliases.
- `decodeGPUAnalysisContiguityStatistics` accepts numeric words, raw `Uint8Array` or `ArrayBuffer`
  reads, an explicit `{bytes}` form and producer-shaped `weights.status`, and honours logical byte
  slices.
- Capacity planners gain `retryStage` and requirement fields that tell unknown, partial and
  saturated requirements apart from exact counts.
- Weights algebra accepts the canonical `'symmetric-difference'`. Weights constructors keep extension
  fields of a declared `spatialContext`, such as `tag`.

**Regression**

- `getGPUGeographicallyTemporallyWeightedRegressionModelResult` reads dense and indexed GTWR summaries.
  It keeps valid likelihoods when the bandwidth-selection flag or corrected AIC is undefined, and
  keeps infinities for exact fits.
- `getGPUSpatialPanelErrorMaximumLikelihoodModelResult` keeps absorbed intercepts and unavailable
  information criteria and counts entity effects in residual degrees of freedom. It requires
  `entityCount` and balanced integer periods of at least 2.
- Named-field regression status decoders (`decodeGPU...StatusFields` for OLS, GWR, GGWR, GTWR, MGWR,
  MGTWR, probit with its spatial-lag and diagnostics results, SUR and spatial SUR, GM, 2SLS, lag and
  error ML, panel, Hausman, regime, restricted-regime, IV covariance and diagnostics) and
  `GPUDecodedRegressionStatus`. The existing numeric words and string decoders are unchanged.
  Estimates kept at the iteration limit are not counted as failed rows.
- GWR and GTWR summary readers and GWR's two-word `selectedBandwidth` reader accept raw float32
  `ArrayBuffer` or `Uint8Array` reads, including padded, unaligned and cross-realm slices.
- `GPUOrdinaryLeastSquares.createDesign` adapts ordered columns into a design.
  `GPUOrdinaryLeastSquares.addJoinedTableFit` fits a joined table and returns its completeness and
  loss status, so an incomplete grouped input cannot publish a truncated fit.
- OLS model results expose the applied ridge regularization, `inferenceMethod` and an original
  summary snapshot. Readers accept `ReadbackInput` byte forms. View-based construction can omit
  the predictor count when the input lengths determine it.
- `GPUSpatialRegressionDiagnostics.estimatedCommandNodeCount`, with a construction-time check against
  the command-node budget.
- `GPUSpatialPanelErrorMaximumLikelihood` exact likelihood accepts N-value entity spectra, including
  panels above 1,024 total rows when the entity spectrum fits the GPU bound. Expanded NT spectra stay
  supported.

**Validation and group statistics**

- `GPUGeoConformalPredictionIntervals` accepts optional producer `overflow` and `candidateOverflow`
  neighbour views (status bit 32 marks incomplete input; bit 16 still means only the internal
  256-entry cap). `decodeConformalStatus` names each status condition and accepts cross-realm
  readbacks.
- `GPUSegregation`: optional `missingCountPolicy` (whole-unit exclusion) and an `invalidUnitCount`
  diagnostic. The default still replaces missing counts with zero.
- `GPUGeographicalDetector`: an optional paired `labelsComplete` input and output that carries
  partition completion certificates alongside the incumbent statistics.

**Ensemble scores**

- `GPUEnsembleScores` and its operations accept `estimator: 'fair' | 'empirical'`; the default is
  still empirical. Literal score selections type the CPU result and GPU `getOutput`. New exported
  types: `EnsembleScore`, `EnsembleScoresEstimator`, `EnsembleScoresEstimatorOptions`,
  `EnsembleScoresOutputSelection`, `CPUEnsembleScoresSelectedResult`, `GPUEnsembleScoresOutput`,
  `GPUEnsembleScoresOperationProps`, `CPUEnsembleScoresOperationProps` and
  `EnsembleScoresOperationOutputs`.

**Interaction and planning**

- `GPUSpatialInteraction` gains an optional `coefficientStatus` output, always returned on the CPU,
  together with `decodeGPUSpatialInteractionCoefficientStatus` and the `GPUSpatialInteractionOutput`
  type. Code 1 proves exact intercept or declared-margin absorption on the observed support. Codes 2
  and 3 are sampled evidence only.
- Static execution planners (for example `GPUIslandAttachment.getExecutionPlan`) on
  `GPUIslandAttachment`, `GPUSpatialWeightsTraces`, `GPURankMarkov`, `GPUGeographicDistribution` and
  `GPUSpatialInteraction` return numeric route and capacity plans and check device bindings.
- `GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS` (4096), `GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS`
  (16384) and `GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS` (4096) tell the automatic crossover
  apart from the forced-route maximum.

## Deprecated spellings

- `GPUEnsembleScores` `fair: true`: use `estimator: 'fair'`.
- `GPUGlobalSpatialStatistics` `alternative`: use `analyticAlternative`.
- `GPUMultivariateLocalGearyPermutationTest` `significance`: use `significanceLevel`.
  `maximumPermutations`: use `permutationCount`. The `pValues` output: use `pseudoPValues` or
  `adjustedPValues`.
- `GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT`: use `GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS`.
  `GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT`: use `GPU_SPATIAL_WEIGHTS_TRACES_EXACT_MAXIMUM_ROWS`.
  `GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT`: use `GPU_ISLAND_ATTACHMENT_AUTO_SCAN_MAXIMUM_ROWS`.
- Weights algebra `symmetricDifference`: use `'symmetric-difference'`. The historical name still works.

## Documentation and tests

- The GTWR guide explains the summary reader, usable-row status, and unavailable versus infinite
  likelihood criteria.
- Regression guides separate the GTWR grid and reference limits from the panel 2SLS, error and lag
  route limits, and describe the main and nested trace tiling limits for lag and error ML.
- The MGWR documentation lists exact buffer shapes and both full-inference limits, states GGWR
  traversal costs separately and corrects the automatic generalized and MGWR grid thresholds.
- The regression status documentation scopes each decoder to its model and output.
- Regression guides describe the probit per-coefficient convergence certificate, completeness
  certificates, point results kept under HAC loss and when restricted robust inference resolves.
- Panel-error guides describe the full Gaussian likelihood, counted fixed effects, entity and
  expanded spectra with their separate GPU limits, and unavailable information criteria.
- The spatial trace documentation describes the all-zero normalization scale and the hierarchical
  tile layout (maximum 16,777,216 tiles).
- The gravity documentation covers exact absorption on disconnected support, evaluated coefficient
  precision and sampled-deviance tolerance.
- Island, trace and Rank Markov guides tell automatic crossovers from forced maxima and show the
  public execution planners, including the 64-state limit on sorted Rank Markov jobs that request
  diagnostics.
- The validation documentation defines boundary membership from stored coordinates and shows both
  neighbour-loss flags in the localized conformal examples.
- Spatial weights examples pass transformed metadata downstream, explain how masked Moran
  re-standardizes, and say which stage a capacity retry repeats.
- Upload guidance aligns with PySAL feature order, reserves a physical slot for empty CSR, and covers
  rectangular and diagonal validation policies. Transform guidance names the `'quartic'` alias.
- The packed chunked-transpose documentation points to the terminal output offset and keeps child
  completeness metadata.
- The chunked validation and spatial lag guides describe host readback and source-row bounds for
  cross weights. The contiguity profile guide gains a hand-counted loss example and documents byte
  decoding.
- Four spatial guides show their public import. Eight statistical recording targets link directly
  to their input, output and limit contracts.
- The rate-mapping examples use public imports and typed inputs and pass strict snippet checks.
