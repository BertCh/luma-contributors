# Changes for callers: The known-issues pass of 2026-11-18 and 2026-11-19 (oracle census, statistics audit, re-review of earlier fix rounds, new regression models, multipart joins)

Part of [the record of caller-visible changes](README.md) from the 2026-11-19 and 2026-11-20 fix passes. Each bullet condenses what the implementers of those passes recorded; [README.md](README.md) says how the changes were tested and what that does and does not establish.

Counts in this file: results 153, rejections 54, added 36, deprecated 1, docs 15.

## Results that change

**Regression: shared distributions and OLS**
- `getChiSquareSurvival` returns 0 at an infinite statistic instead of NaN. Large-df chi-square tails now converge (sf(10000, 10000) is 0.49812, was 0.50053), and the host series throws on exhaustion instead of publishing a partial sum. `getLogGammaOfHalfInteger` uses Stirling's series for arguments of 32 and above.
- The Student-t helpers (CPU and WGSL) return representable Cauchy tails and critical values at extreme arguments, instead of NaN or the old 2^60 bracket ceiling. Large df is now evaluated stably (t = 1.96, df = 200,000 changes from 0.05029414 to 0.04999718), df >= 1e10 uses a normal-density tail expansion, and NaN inputs give NaN.
- `GPUOrdinaryLeastSquares`: when the squared residuals are a positive constant, Breusch-Pagan is 0 with p = 1 instead of NaN/NaN, and Koenker-Bassett stays undefined. Held-out R² of translated outcomes is -1 on the reported fixture, not 0. Kurtosis and Jarque-Bera come from normalized residuals and stay finite at response scales 2^32 and 2^-40. Last-bit rounding of the dimensionless diagnostics can change.
- The OLS auxiliary Breusch-Pagan/Koenker-Bassett response is the centred normalized squared residual, which corrects near-constant Koenker-Bassett decisions and Breusch-Pagan values that were suppressed when raw squares underflowed.
- The CPU singular-value helper terminates on rank-deficient input (it used to run without end), reports only min(rows, columns) values for wide input and keeps small orthogonal components whose squared norms underflow.
- `GPUSpatialCrossValidatedOLS` held-out metrics no longer depend on the outcome origin.

**Regression: GM, 2SLS and instrumental-variables covariance**
- `GPUSpatialErrorGM` on exactly affine data now returns code 3 with lambda inference suppressed, instead of code 0 with finite lambda, SE and p. On the GPU this needs a float32 coefficient proposal, finite normal nonzero terms and a product-exponent span of at most 200; inputs outside that fall through to the ordinary fit. Code 2 (too few rows) still takes precedence, and a one-ulp departure is still fitted.
- `GPUSpatialErrorGM` pseudo-R² at response scale 2^30 is 0.70948776, where it was 0 or NaN.
- 2SLS first-stage F is NaN when the denominator degrees of freedom are zero (a saturated first stage), not 0.
- Moran's I in the regression diagnostics: variances at cancellation scale become 0 with NaN z/p; positive variances that used to be clamped keep finite z/p; exact complete graphs stay undefined. A constant fitted lag gives J = 64, not 100, on the reviewed cycle.
- `GPUInstrumentalVariablesCovariance` keeps real one-ulp column variation and treats exactly constant instruments as invalid. It no longer uses a relative cancellation threshold, which had let a centred norm of 4.4e-16 pass.
- A nonzero HAC kernel `overflow` or `candidateOverflow` flag, or a nonzero first CSR offset, on 2SLS and HAC covariance gives covariance status 4 with NaN covariance and inference, instead of status 0. Point estimates, residuals and classical diagnostics are unchanged.
- A `requiredCount` that proves truncation in the diagnostics, panel or 2SLS HAC weights now yields diagnostics code 3, HAC covariance code 4, or panel fit/diagnostic codes 3/1, with the dependent values unavailable. Trace, GM, 2SLS, lag-ML and error-ML results on truncated weights return NaN with model code 3. Suppression leaves spare output words intact instead of overwriting them.
- Decoded robust models are labelled `'robust'` instead of maximum-likelihood. Classical decoding is unchanged.
- `GPUSpatialTwoStageLeastSquaresDiagnostics.rhoOutsideAdmissibleRange` and `GPUSpatialMaximumLikelihoodDiagnostics.{approximation, converged, atBoundary, seriesLimited, varianceDefined}` decode as `null` when the flag is unavailable or invalid, instead of `false`.
- Robust 2SLS covariance accepts residual capacity sized to the rows (13 rows no longer throws "design length 39").

**Regression: maximum-likelihood lag and error models**
- Error ML on certified exact scalar profiles computes coefficient inference and innovation variance with a stable unfiltered solve. This corrects an intercept p-value drift of about 1.85% for negative c on Metal.
- Error ML with W = cI reports undefined spatial inference (lambda SE, z and p NaN, varianceDefined 0) and picks the lower bound deterministically. It used to report finite SEs that depended on the parity of the series length. Exact scalar lag ML suppresses all joint coefficient and rho inference.
- The tail-extrapolated bound in slot 18 now includes the absolute correction (the published bound grows; the CPU panel-error slot 18 too). Slot 21 is unchanged.
- Pseudo-R² for error ML is the finite squared correlation where products of moments overflowed or underflowed; weak nonconstant predictions keep a finite value and constant fits stay NaN.
- Near-pole exact determinants recover the finite factor (-36.6190 rather than -36.7368 on the fixture). Stationary maxima at either endpoint report boundary and not converged, and a stationary lower-endpoint minimum no longer hides a maximum in the first scan cell. Candidate counts and convergence can change on such profiles.
- Both likelihood shaders remove residual means from fitted moments, so results no longer depend on the predictor origin.

**Regression: panel models**
- `GPUSpatialPanelLagMaximumLikelihood`: the within-lag is formed after demeaning (lagging large levels first lost about 20% of it), and the profile no longer selects biased minima from rounded log-determinants or log-variances. Valid-fit rho, slopes and dependent numbers change.
- `GPUSpatialPanelLagMaximumLikelihood` refuses certified zero-within-variance fits and rational single-column exact poles with status 2, every float output NaN, and Hausman status 1 when requested. The GPU had reported such near-pole fits as successes. Required-edge loss refuses with fit/Hausman status 1. An explicitly converged CPU boundary optimum now succeeds with covariance instead of status 3.
- `GPUSpatialPanelRegression` and the panel diagnostics accumulate normal-equation moments pairwise and centre pooled diagnostics stably, so float32 numbers change where small terms were lost. Exactly zero pooled fits suppress the LM tests, and rank-zero projected LM information gives diagnostic status 2 with NaN values.
- `GPUSpatialPanelErrorMaximumLikelihood` on certified exactly affine data (including constant responses) publishes code 3, convergence 0, NaN table and residuals, and pooled diagnostic status 2, instead of a rounding-created success. CPU exact-fit failures keep their stochastic trace errors (2.2696, not 0).
- `GPUSpatialPanelErrorMaximumLikelihood` CPU changes: scalar-identity exact fits choose lambda = -bound with undefined lambda inference; exact orthogonal fits publish slope 0 with undefined pseudo-R²; exact two-axis design dependence reports singular code 1 rather than code 3.
- A nonzero `weights.status.candidateOverflow` on `GPUSpatialPanelErrorMaximumLikelihood` suppresses the pooled diagnostic tests with status 1.

**Regression: GWR family**
- `GPUGeographicallyWeightedRegression`: local and global R² are NaN for a constant response, and translated singular local designs keep status 2. Default adaptive bounds are capped at the feasible count (8 rather than an impossible 23 on the eight-row fixture). Candidate `observationCount` excludes rows with invalid bandwidths.
- `GPUGeographicallyWeightedRegressionNonstationarityTest`: identical draws give SD 0 and NaN z. Any undefined requested spread gives NaN pseudoPValue, exceedances and zSimulated, where it used to give p = 1/3. An invalid permutation count runs no draws and gives NaN inference and NaN `summary.permutationCount`.
- `GPULocalCollinearity` with incomplete or truncated weights publishes NaN for every diagnostic with status 1, instead of partial values with valid-looking statuses.
- `GPUGeneralizedGeographicallyWeightedRegression`: prediction-only queries have leverage exactly 0. Prediction-only Poisson queries with a nonfinite link or a link outside [-30, 30] return code 4 with NaN values, instead of code 0 with Infinity. Poisson bandwidth selection keeps small deviances near large counts (the review fixture now picks 0.5, not 10). `iterationCount` counts accepted updates, so a singular first system reports 0.
- `GPUGeographicallyTemporallyWeightedRegression` centres the response, so y = 1e6 + x keeps slope 1 and zero residuals. Ordinary GTWR results can change by float32 rounding.
- `CPUMultiscaleGeographicallyWeightedRegressionOperation` reports inference status 2, with NaN dependent outputs, when the published float32 variance overflows.
- Explicit GWR spreads above 4,096 tiles use pairwise grouping and their float32 rounding can change. Layouts at or below 4,096 tiles are unchanged.

**Regression: SUR, probit and restricted regimes**
- `GPUSeeminglyUnrelatedRegression`: equation residuals within 8·width·epsilon of the fitted scale are set to zero before whitening, so an exact fit reports singular status 3. The covariance-overflow fixture gives Sigma 2^125 instead of Infinity. A seeded zero-response fit that used to succeed is now singular status 3.
- Spatial SUR: invalid lambda or island rows suppress the joint outputs with status 1, where they used to report success. A nonzero first CSR offset gives equation status 4 and joint status 1 on the CPU, as on the GPU. Near-unit rows are normalized before every spatial stage and responses are centred, which repairs the slope shift under a +65,536 response offset.
- `GPUProbitRegression` diagnostics with a negative edge on a masked row, or with `requiredCount` above count or capacity, now suppress dependent outputs. They used to return status 0 with finite statistics.
- `GPURestrictedRegimeRegression` and `GPUSpatialRegimeRegression` keep inherited and non-enumerable declared props such as masks.

**Spatial statistics and autocorrelation**
- `GPUGammaStatistic` and `GPUSpatialPearson` publish a NaN statistic when either weights loss flag is nonzero, instead of a finite value from the edges that were kept.
- `GPUSpatialPearson` handles extreme products with exponent-scaled arithmetic: the closed-form ring at (i+1)·2^32 gives 1 instead of 0, an underflow-clamped value becomes 41/42, and constant eligible marginals give NaN.
- `GPUQuantileLISA` now writes `output.analysisCount`, which was accepted but never written. Exact integer ranks are selected, which restores memberships lost on the 21- and 25-row fixtures.
- `GPULocalSpatialHeteroscedasticity` (LOSH) accumulates weighted differences. Constant neighbourhoods give exact zero residuals, entirely constant inputs give NaN H instead of spurious hotspots, and balanced residuals no longer become false hotspots.
- `GPUGlobalSpatialStatistics`: bivariate `varianceRandomization` is NaN for a constant second variable. Uniform full-population Moran/Gi nulls and invariant complete-graph Geary give NaN z/p. Geary separates uniformly scaled invariant weights (variance 0) from near-invariant ones (tiny positive variance). The reference mode and CPU twin now omit nonfinite weights, as the tiled mode does.
- `GPUMultivariateLocalGeary` inference suppresses candidate-only and chunk-local loss, keeps translation invariance (a counterexample went from 2 to 4) and keeps small distinct donors (one row's p changes from .05 to .5). All-tied inference uses class 2.
- Local permutation tests (`GPULocalPermutationTest`, local Geary, bivariate Moran and Lee): significance compares integer ratios against the exact stored float32 cutoff, so boundary false positives can clear. Undefined normalized statistics are NOT_TESTED with NaN inference. Bivariate normalization survives overflow of the direct sum-of-squares product.
- `GPUGlobalPermutationTest`: undefined references no longer count as histogram members or extrema. Sparse General G with fewer than two positive values is unavailable.
- `getPermutationRejectionFloor` drops its tolerance slack, respects adjusted-value caps and returns the unbounded Bonferroni/Holm family size at alpha = 1. With alpha = 1, `minimumRowsAtFloor` changes from 1 to 0.
- `GPUGaussianPredictiveScores` returns correct small tail Brier values where it used to return inaccurate values or zero.
- `GPUSpatialEmpiricalBayesRates` suppresses all outputs on incomplete neighbourhoods.

**Spatial weights**
- `GPUSpatialWeightsAlgebra` and `GPUChunkedSpatialWeightsAlgebra` used to publish `hasSelfWeights: false` unconditionally. Union and xor now report true when either operand permits self-weights, intersection reports false when either operand excludes them, difference follows the left operand, and the result is undefined when unknown.
- `GPUIslandAttachment` carries upstream loss into `outputWeights.status.candidateOverflow`, and writes a caller-supplied candidate destination even when there are no upstream flags.
- `GPUSpatialWeightsSummary` on incomplete input publishes NaN statistics and zero counts, instead of moments of the retained prefix.
- `GPUSpatialWeightsTransform`: `'V'` stays scale-invariant when squares overflow or underflow float32; kernel weights at negative or nonfinite relative distances are zero; `'D'` with an overflowed row sum zeroes the whole result. Double-standardized rows give exact results on tiny normal inputs ([8] rather than [0]), and equal reciprocal 2e38 words stay finite.
- Chunked normalization matches the whole-matrix behaviour for finite extreme weights. This fixes the outputs [0,0,3] and [3e38,3e38,1].
- Fixed kernels report unknown symmetry, so consumers no longer skip a symmetrization they need. Rectangular transpose clears the double-normalization metadata.
- Lattice and algebra CSR stages saturate individual prefixes as well as totals, which corrects wrapped decreasing offsets. Contiguity pair growth caps at uint32.
- Weights metadata: chunked transpose omits inherited count/requiredCount but keeps loss. The correlogram publishes the standardized algebra description. Higher-order `requiredCount` after candidate loss is no longer described as a lower bound.
- A transpose chunk's `candidateOverflow` resets to zero each submission, and the island count is written every submission.

**Dynamics, change points and classification**
- Markov diagnostics (transition matrix, rank, spatial and LISA Markov): reducible active chains publish all-NaN stationary rows with converged 0. GPU diagnostic aggregates use float32 instead of wrapping uint32, so totals above 2^24 can differ by rounding. Nonfinite stationary weights clear `converged`. Q, LR and pooled probabilities can differ in the last float32 bits because the summation order changed.
- Markov LR between near-identical regimes stays positive with the correct tail. A nonzero neighbourhood loss flag suppresses lag, lag-class and transition results in `GPUSpatialMarkov`.
- `GPUEmergingHotSpots`: weighted Gi* on nearly uniform weights keeps valid scores instead of all NaN. A constant detrended signal keeps factor 1, so trendZ, trendP and the categories that depend on them can change. An explicit unadjusted `criticalZ` reports its two-sided normal tail in the legend metadata. Supplied loss flags suppress every Gi* bin.
- `GPULocationChangePoints`: translated Normal costs and partitions match the untranslated ones. Flat or short series below `minimumSegmentLength` give NaN `optimalCost` and zero counts. A lone positive Poisson value reports its stated cost instead of 0, and negative Poisson observations are suppressed on the GPU as on the CPU. The first-segment penalty fix can change near-tie partitions.
- `GPUClassificationFit` computes squared deviations about median offsets, which fixes SS/GVF on skewed distributions.

**Scan statistics and segregation**
- `GPUSegregation` with `spatialForm: 'environment'` leaves out the E-term of populated units whose environments are empty (0.75 becomes 0 on the regression fixture). Atkinson no longer floors the group total at 1e-30. Scales whose weights report loss are suppressed and the other scales stay usable.
- `GPUExpectationBasedSpaceTimeScan` drops non-positive and nonfinite baseline cells from both observed and expected sums (a mixed window goes from 114 cases to 14).
- Scan window-count requests above uint32 saturate and clamp to capacity instead of wrapping.
- Space-time permutation population caps use integer zone margins (a 6/12 boundary cluster is admitted). Capacity-loss reporting covers the whole next tie: overflow goes from 3 to 2, and the winning centre is no longer marked truncated on the symmetric fixture.
- Explicit frame limits are respected (limits of 1, 2 or none give overflow 0, 2 or 3).
- Region and candidate membership loss in the mask statistics summary takes precedence over aggregate overflow. Inherited candidate loss reaches `candidatesTruncated`.

**Regionalization**
- `GPUMaxPRegions`: a fallback singleton at or above the threshold no longer counts as infeasible. Partial-enclave budgets split only disconnected remainders, every pre-enclave growth score ranks starts, and feasibility is recomputed after refinement. Region counts, labels, `selectedStart` and feasibility can change.
- `GPUAZPRegions` ranks destinations after each destination's own cutoff; its review fixture now moves once and converges at SSD 0.5.
- With `standardize: true`, the CPU AZP and Max-P twins keep finite nonconstant columns whose deviation is at most 1e-20 (the AZP fixture objective becomes 0.028328281650734446).
- Shared standardization keeps extreme finite inputs nonconstant and sums each lane pairwise. Results of minimum spanning tree, `GPUSkaterRegions`, `GPUWardRegions`, AZP, Max-P and `GPURedcapSingleLinkage` can change in that regime.
- `GPUWardRegions` accumulates raw sums as offsets from each region's representative. This fixes translation-dependent partitions but doubles raw scratch to rows·(2·columnCount + 2) words. `GPUSkaterRegions` admits finite gains whose intermediate square overflowed, and constant features contribute exactly zero.
- Region evaluator: mixed-sign sums and constant coordinates are kept (means [0,-0.5] become [0.5,-0.5] and between-SSD goes from 0.5 to 1). Constant tables have exactly zero SSDs. Pseudo-F uses exponent-aware ratios. `crossLinkFraction` is NaN on matrix or chunk loss.
- Nonzero AZP and Max-P weights loss flags suppress labels, objective and status.

**Validation, co-location and detectors**
- `GPUGeographicalDetector` keeps small nonzero q and F, computes F from mean squares (finite F and positive p when q rounds to 1), and gives NaN instead of a false perfect tail when F overflows. Inclusive binomial tails are more accurate (0.050010297 becomes 0.049933762 at the decision boundary).
- Environmental cross-validation (kNNDM) centres separately from fitting and keeps distinct constant groups (the shifted objective changes from .0283203125 to .028076171875, and a missing-fold status 2 becomes status 0). Explicit CPU `maximumIterations` now bind, with status 3 on exhaustion.
- `GPUNestedInterpolationValidation` R² on the translated two-outcome fixture changes from -1 to -3. `GPUAreaOfApplicability` queries with no finite squared distance return NaN with applicability 0, and constant packed decimal predictors are flagged.
- `GPUConformalPredictionIntervals` and `GPUGeoConformalPredictionIntervals` (CV+) endpoints equal computed candidates at zero (+0, not a negative subnormal). Endpoints at common scale 2^-149 change from [-7, 7] to [-8, 8], and subnormal localized masses are kept.
- `GPUColocationQuotient` accepts all finite positive weights (it rejected weights of 3e38 and above) and keeps fractions for very large weights.
- Explicit NNDM buffers bind in addition to matched radii.

**Other statistics**
- `GPUNeighborhoodSummary` reserves the category value 0xffffffff as missing, because it collided with NO_MODE. Exact category ties use raw masses. Means equal sum/weightSum again. Translated spread and z-score distances keep relative means. Extrema and median cover finite values beyond ±3e38. A tiny `focalWeight` that rounds to zero no longer gives a finite weighted mean.
- `GPUGeographicDistribution`: the sorted mean length excludes NaN lines, narrow ellipse short axes stay nonzero, and very large axes stay finite.
- `GPURoseStatistic` keeps the descriptive resultant and variance in [0, 1].
- `GPULocalOutlierFactor` skips invalid IDs, never flags islands and suppresses scores from incomplete support.
- `GPUSpatialInteraction` and `computeSpatialInteractionOnCPU`: near-fit Poisson deviances stay positive. Bounded coefficient searches spend the remaining budget after failed probes and need a feasible fit to converge. A positive-deviance incumbent survives exhaustion with status 4. Oversized `topIds` tails are rewritten to NO_RANK.
- `CPUSpatialInteractionOperation` honours explicit `maximumIterations` and `maximumCalibrationIterations` (inner exhaustion gives candidate code 1 and status 3). Without an explicit outer budget, exploratory recovery stops after 32 rounds.

**Point process and space-time**
- `GPUMantelTest`: the CPU rounds correlations to float32 before comparing them, so published ties no longer split (the three-pair p goes from 1/6 to .5). The GPU centres on a data-valued pair anchor, and its partial storage grows from 3 to 7 words per slot and block, so large configurations may use fewer row blocks.
- `GPUPointPatternEnvelope` extreme-rank bounds use the largest attainable global midrank (at alpha .21 the lower bound goes from 0 to 1).
- `GPUNearRepeatKnox` zeros its counts and fills its summary with NaN when the upstream `pairs.status.overflow` or `candidateOverflow` is set, instead of publishing partial tables.
- Variogram, correlogram and space-time variogram fixed-point consumers use scaled unit storage. This corrects zero estimates from flushed adaptive units and false overflow in space-time variograms. Randomization diagnostics and kurtosis at tiny amplitudes are no longer 0/0.
- In the neighbourhood statistics, missing rows publish INVALID_QUERY (1) alone. Finite points outside both the grid and the support give 5. Scalar clears no longer overwrite caller words after index 0 of an oversized status view.
- Clock encounters (`addClockEncounters`) no longer report pairs whose raw longitude separation exceeds 180 degrees, and an exact-distance pair across adjacent cells is no longer missed.
- The CPU reference for `GPUPointDensity` bins against the uploaded float32 bounds and excludes outside or nonfinite coordinates instead of clamping them.
- `GPUPointDensity` bounds given as scientific-notation integer literals no longer produce invalid WGSL.
- Brownian bridge mixed-error fits with zero-error, zero-residual bridges return `motionVariance` 0 and `lowerBoundReached` true.

**Dataframe and raster**
- `GPUInequality` Atkinson near epsilon = 1 keeps the inequality signal (0.133974611790977 at epsilon = 1 + 2^-23 on [1,1,3,3], was 0). Zero incomes are included for epsilon < 1, and [1e-30, 1e30] gives 1 instead of NaN.
- `GPURasterTemporalResample` keeps an exact observation at the start of a gap longer than `maximumGap` ([NaN,NaN,NaN,11,NaN] becomes [NaN,7,NaN,11,NaN]). `getGPURasterCentredFractionalYears` rounds to float32 as the shader does. The kepler.gl raster time-series adapter now passes the last input date as `gridEnd`, so its default 24-point grid ends at 11/status 0 instead of NaN/status 1.
- `CPURasterRegionGraphOperation` honours explicit `maximumIterations` (one round gives [0,0,1] with 1 pending, instead of the full closure). It used to report `iterationLimitReached` 0 regardless.
- `computeRasterWatershedSegmentationOnCPU` and `CPURasterWatershedSegmentationOperation` honour explicit `maximumSteps` and `maximumIterations` and mark unresolved cells; they used to run to completion with [1,0]. Inherited nodata is respected (the three-cell case gives [7,0,0] and [1,0]).
- `computeRasterWalkingCostOnCPU` and `CPURasterWalkingCostOperation` honour an explicit iteration cap ([2,0,1,0] rather than [6,1,0,0]).
- `CPUCostDistanceOperation` honours its budget props, which it used to ignore (budget 1 gave costs [0,1], count 2, converged 1). Exhaustion publishes NaN or GPU_COST_DISTANCE_UNRESOLVED. With an explicit budget, `iterationCount` and `settledAt` count tiled rounds, and `maximumTieIterations` defaults to the cost budget.
- GPU cost-distance and terrain round counts (`iterationCount`, `settledAt`, fill, flat and pull counts) depend on scheduling, with bounded increments; specs no longer require them to be equal across runs. Converged costs are deterministic, but whether a fixed budget is enough to converge can vary between runs. GPU arithmetic is unchanged.

**Terrain**
- `GPUTerrainFlow` publishes the flat-routing statuses on filled-only requests, which it used to leave unwritten. Flat convergence includes upstream fill completion. Unfinished fill or flat stages force `accumulationConverged` to 0 whenever accumulation is suppressed.
- CPU flow with an explicit `maximumFlatIterations` binds all three stages and counts confirming rounds (singleton 0 becomes 3). An explicit `maximumFillIterations` binds tiled relaxation: a cap of one on the bowl suppresses values, and a cap of two completes with 2.25. Explicitly budgeted CPU fill and flat stages run directional sweeps, so the cap-99 corridor now completes.
- `GPUTerrainWatersheds` with an explicit budget reads a stable snapshot each round (the four-cell chain at budget one always gives [UNRESOLVED,3,3,3]). Receiver cycles that collapse to self-pointers stay unresolved with convergence cleared. CPU convergence is 0 for paths that end at an unresolved direction.
- `computeTerrainWatershedsOnCPU` and `CPUTerrainWatershedsOperation` honour an explicit `maximumIterations`, which they used to ignore.
- HAND drainage cycles and unknown drainage publish DRAINAGE_UNRESOLVED/NaN with convergence 0, instead of gaining convergence from stable pointers.
- Spike repair publishes `outputValidity` 0 for unfinished valid pixels, instead of 0xffffffff beside +Infinity heights.

**Network**
- `GPUNetworkNoding` includes split-candidate loss in whole-network overflow, which used to stay 0 on incomplete topology.
- `GPUNetworkPotentialPathArea` gives inactive edge capacity slack -Infinity and membership 0 on every submission, instead of keeping the previous membership.
- `CPUNetworkCostMatrixOperation` honours explicit `maximumIterations` and `localIterations` (the 257-neighbour one-round case gives cost Infinity, not 2). `CPUNetworkCentralityOperation` honours search and dependency budgets (closeness goes from 2/3 to 1 on the one-hop chain). `CPUAlternativeRoutesOperation` forwards `maximumExpansions` and `maximumFrontierSize`, and one expansion gives no route.
- `GPUNetworkServiceAreas`: explicit CPU phase budgets and `activeIterations` keep bounded incumbents (zero active rounds give costs [0, Infinity, Infinity]). A bounded tie hop reports the limit. An unassigned node no longer takes a predecessor from another unassigned node. Large CPU round budgets no longer overflow the hop product.
- Raster triangle facility labels in the isochrone and drive-time recipes rank samples in world units, not pixel indices (the rectangular-pixel fixture gives label 5, not 2). Negative-cost nodes no longer enter the cell table or outline.
- The `isochrone-band-rings` recipe stage publishes `status.candidateOverflow` (ring edge loss), which it used to leave out.
- The landscape recipe's raster `rowCounts` include assigned missing-value cells ([1] becomes [2]). Inherited CPU `featureOffsets` group features as own fields do. Empty multipart value modes publish empty aggregates instead of throwing.

**Joins**
- The CPU feature join treats a distance exactly equal to the threshold (6.5) as inside, and translated disjoint geometry no longer matches at radius 0 or 0.02.
- DE-9IM results are corrected: a narrow-hole matrix goes from 10F0FF212 to 1010FF212, identical one-ulp squares go from FFFF1FFF2 to 2FFF1FFF2, and part order no longer changes the class. Empty pairs get GEOS exterior cells (empty/empty 196608 with equals 1). Nonfinite multipart features are suppressed regardless of order.
- `GPUNearestFeatureJoin` keeps the determinant residual in nearly cancelling point/segment distances, so near hits are no longer reported as exact. Points at tied parallel edges come from one minimizing pair. Neighbours mode writes `status.candidateCount` (zero) every submission instead of leaving caller bytes. CPU nearest search skips nonfinite parts consistently.
- Nearest-feature IDs and distances on disjoint or coincident roads are corrected (ID 33 at distance 0 becomes ID 42, or missing at zero radius). CPU duplicate-road ties choose row 0. Positive-radius boundary matches are found.
- The workgroup `dwithin` `uncertainCount` counts the affected candidate pairs, where it used to report zero.
- `GPUBufferSelection`: an exact collinear hit at zero radius is selected on the CPU.
- `GPULineLengthPerPolygon` gives boundary-coincident pieces to both polygons. Invalid edges no longer consume candidate slots, and all-empty multiline batches clear the outputs.
- Multipart choropleth sum and mean use grouped fixed-point accumulation.

**Geometry, lines and topology**
- `GPULineNoding` keeps the splitter's source row for every atomic traversal, and very large or small finite edges publish edge 0 with parameters [0,1] instead of NONE/[0,0]. Safe projections are exact (0.75 rather than 0.7499999).
- `GPUPolygonize`: an inward-spurred square keeps its area-16 face, a tolerance-collapsed terminal spur no longer removes a face, and the bounded face no longer depends on input order. The middle of a three-edge open chain is a dangle (class 26), not a cut edge.
- `GPUMakeValid`: duplicate-geometry dangle and bridge counts are 2, not 1, and duplicate directed flags inherit their owner's bits. Retraced two-point ring edge IDs change from [0,0] to [0,1].
- `GPUPolygonPrecisionReduction`: `nodedSegments.sourceEdgeIds` refer to the original input rows ([5,6,7,8] becomes [4,5,6,7] with implicit closure). On a non-binary grid, the half-boundary keeps the exact side, so a 0.1-step rectangle keeps four sides and its face.
- `GPUVectorStraightSkeleton` terminal-ridge source pairs are corrected ({0,3} becomes {0,2}, and a clockwise corner becomes {2,3}).
- The CPU convex skeleton, vector skeleton, minimum circle and ratio hull honour explicit limits, which they used to ignore. CPU circle `cellCapacity` 1 to 3 returns a bounded result instead of throwing.
- Offset curve CPU vertices keep the surviving source edge (2 rather than 0).
- `GPULineMerge` writes `GPU_LINE_MERGE_NONE` into `lineOrders` for lines with fewer than two vertices, where it left stale bytes. Ignored lines get `lineReversed` 0. Spare `chainOffsets` rows get the final total.
- `GPULineClipByPolygon` writes `vertexSourceEdges`, which it never wrote. Its unused midpoint rows no longer raise uncertainty. Clip `pieces.status.overflow` and `GPUSharedPaths` `status.overflow` now include uncertifiable predicates (0 becomes 1 on the incomplete fixtures).
- `GPUSharedPaths` source-ID outputs honour caller IDs; single-part supplied IDs used to be ignored ([0,1] rather than [17,88]).
- `GPULineSplit` and `GPULineClipByPolygon` `pieces.sourceIds`, and `GPUBufferSurface` `boundary.sourceFeatureIds`, publish caller feature identity, falling back to the logical row.
- `GPUGeometryPredicates` publishes 0 for `isSimple` and `isRing` when `status.overflow` is set, instead of 1 after dropped crossings.
- Geometry measures: planar minX at row 2 is 0.015625, not a rounded 0 at row 1. A collapsed multiline centroid is 4.5 regardless of repeated vertices. Bounds reductions cover both FLT_MAX signs.
- `GPUMinimumClearance` counts multiline parts as one feature, and clearance between collapsed parts is 9, not Infinity.
- `GPULineLoopRemoval` CPU honours an explicit `intersectionCapacity` (two bowties at budget 1 suppress geometry).
- Buffer: empty or short parts no longer add synthetic origin vertices (the translated boundary count goes from 0 to 8). Nonfinite vertices discarded by compaction set `candidateOverflow`. A stored line that deduplicates to a point emits its square or round cap. A twice-traced square with inherited nonzero fill keeps four edges.
- `GPUSphericalPolygonOverlay` and the planar overlay publish caller `sourceIds` in `boundary.sourceFeatureIds` (for example 101/202 instead of row numbers). Complemented representatives minimize the caller operand and row.
- Spherical polygon containment measures distance along the minor arc, so a short-edge antipode is outside (a closed Kepler result goes from 2 rows to 1). Coincident-square required segments at capacity 3 are 4, not 5.
- The `GPUSphericalArcOverlay` descriptor changes to tier reference, quadratic-source transient memory and multi-pass dispatch. Computed results are unchanged.
- The CPU geodesic pairs operation with an explicit budget that fails to converge publishes the spherical fallback with [0,1,1], instead of an exact distance.
- CPU map colouring with an explicit budget leaves uncoloured sentinels and `converged` 0 on exhaustion.
- CPU S2 polygon and cap traversal honour explicit `maximumVisitedCells` and `frontierCapacity`.

**Cells, tessellation and trajectories**
- `GPUCellLevelSelection` writes `status.requiredCount`, which it left as stale caller bytes.
- `GPUDelaunayTessellation` writes 0 to `candidateOverflow` and `uncertainCount`, and writes `approximation` (1 for a partial result after exhaustion). `GPUVoronoiDiagram` sets `candidateOverflow` when the triangle count exceeds the view, and requires complete input for `converged`.
- `GPUTrajectoryLocations` writes `output.status.iterationCount`. `CPUTrajectoryLocationsOperation` honours `maximumIterations`: one round on the five-point chain gives partial results. Incomplete labels are numbered by first visit ([0,1,1,1,0] becomes [0,1,1,1,2]).
- `CPUTrajectoryRecursionsOperation` honours an explicit `maximumTrackSegments`, which can set `candidateOverflow` and change revisits, residence and passage.

**Construction snapshots**
- Most audited contributors and their GPU, CPU and operation forms copy declared props, port maps and budgets once at construction, including inherited and getter-backed fields. Mutating a caller object afterwards no longer changes topology, budgets or output routing. Buffer contents are still read on each submission. This covers OLS, GM, IV, likelihood impacts, panel, GWR, SUR, probit, regimes, autocorrelation, dynamics, scan, regionalization, validation, weights, network matrices and service areas, raster, terrain, geometry and joins.

## Input now rejected at construction

**Regression**
- The named OLS operations reject a nonpositive `rowCount` and resolved row counts that disagree. `getGPUOrdinaryLeastSquaresModelResult` rejects active coefficient, error and statistic prefixes shorter than `predictorCount`.
- The three GPU 2SLS operations check `rowCount`, which they used to ignore.
- `GPUSpatialPanelRegression` rejects malformed CSR offsets, which it used to clamp silently into a different operator. The panel models reject counts that disagree with the CSR extent, and count/requiredCount outputs that alias other outputs.
- `GPUSpatialLagMaximumLikelihood` (exact) rejects output buffers aliased with the eigenvalues.
- GWR parameter names inherited from `Object.prototype` throw. Aliased local diagnostic outputs throw. `GPUGeneralizedGeographicallyWeightedRegression` refuses overlapping physical output ranges, even through distinct graph imports.
- `GPURestrictedRegimeRegression` requires `robust` and `output.robust` together, with all four robust outputs. Supplied scalar ports must be one uint32.
- `GPUProbitRegression` with `spatialLag` requires exactly one uint32 per supplied status and disjoint writable ranges across the ordinary, diagnostic and lag outputs. It rejects invalid CSR on masked rows.
- Spatial SUR CPU and named forms refuse status-bearing spatial weights instead of discarding their loss flags. A GPU iteration tolerance must round to a normal positive float32.
- Regression layouts that newly fit under 2^24 tiles are still refused when they exceed 32,768 command nodes (`estimatedCommandNodeCount`).

**Spatial statistics, permutation and dynamics**
- A significance level of 1, or one that rounds to 1 or 0 in float32, is refused by the declarative schema, multivariate Geary and the packed `significanceLevel`. `getPermutationRejectionFloor` requires alpha in (0, 1]. The folded `getPermutationPValue` requires an explicit valid count.
- `GPUSpatialPearson` rejects a missing `secondValues`; it used to fall back to univariate correlation. The Gamma adapter rejects unknown operation strings; it used to select Moran. LOSH rejects invalid inference settings at construction. `GPUGammaStatistic` and `GPUSpatialPearson` reject chunk-local loss ports and say how to aggregate them.
- Multivariate Geary inference rejects invalid strategy strings and graph budgets above its `estimatedCommandNodeCount` limit.
- `GPUGaussianPredictiveScores` checks the required `crps` output, and null or omitted output maps give a named error.
- Dynamics: invalid cube layouts fail in every form; spatial and LISA output shapes fail at construction; LISA rejects invalid gating strings and outputs that alias an input loss flag; Markov outputs cannot share a backing buffer.
- In dynamics, GPU `minimumTransitions` above UINT32_MAX is refused, as are emerging hot spot settings outside their float32 domains, an unsupported `multipleTesting` and inherited names such as `cost: 'constructor'`.
- Scan: `GPURegionMask` refuses interleaved masks and malformed rectangle, polygon or pick storage. Explicit window limits must be finite and positive, and fractions must stay positive in float32. Summary `binCount` must be an integer in [0, 65536].
- Regionalization: loss flags misaligned with the whole, outer or chunk-local layout fail. So do empty SKATER count views, empty REDCAP count or convergence views and empty Ward loss-port names.
- `GPUCategoryPairCounts`, `GPUSpatialEntropy`, `GPUColocationQuotient` and `GPULocalCategoryDependence` reject weights that declare `overflow`, `candidateOverflow` or `requiredCount` ports. Check producer loss and pass a complete CSR instead.
- Conformal `calibrationCount` must be a scalar uint32 view. CPU environmental `maximumIterations` must be a positive integer.
- `GPUGeographicDistribution` refuses finite values that become nonfinite in float32 and a `standardDeviations` that rounds to zero. LOF refuses floors that float32 cannot represent. `GPUSimilarLocations` refuses invalid column counts, nonfinite packed weights and a `resultCount` that float32 cannot represent exactly. `GPUNeighborhoodSummary` refuses negative props that underflow.
- `GPUGlobalSpatialStatistics` (Moran) rejects differential quadrants and unstopped `permutationCounts`. Neighbour search refuses its six status fields that are never written.
- `GPUColumnQuantiles` and `GPUSpaceTimeDiggleTest` constructors reject an invalid `rowCount`. `GPURipleyEnvelopeOperation` rejects positions whose row count differs from `rowCount`.
- `GPUNearRepeatKnox` rejects loss buffers that alias its outputs. `GPUCoverageSimplification` checks `maximumIterations` (an integer in [1, 1024]) at construction, under the caller's ID.

**Spatial weights**
- Chunked algebra and packed transpose refuse requested distances they cannot populate. Transforms that read the old weights in place need the exact source byte range.
- Packed transpose refuses every destination status and distance port; use the chunked output for destination diagnostics.
- Named validators refuse colliding graph names, and validation needs exactly one `invalidCount` word. The GPU validation operation refuses a mismatched `rowCount` even without flags.
- Weights refuse summary or transpose outputs that overlap, transform aliases of preserved CSR or status buffers, and duplicate physical output imports.
- Collective controls refuse short, unaligned, foreign-graph, oversized or output-aliased buffers. Loss aggregation refuses a read-only input that shares the result's buffer; use `merge: true`.
- The capacity helper refuses nonfinite headroom and unsafe grown capacities.

**Raster, terrain and cost distance**
- CPU cost and tie budgets must be integers in [1, 1024]. `GPUCostDistancePath` refuses a `candidateOverflow` view with the wrong format, more than one row or writable overlap.
- Terrain: explicit flat budgets must be in [1, 1024], and CPU watershed budgets above 1024 now throw. Flat status words sharing a buffer reject overlapping writable windows.
- Raster watershed and walking-cost contributors refuse missing convergence views and malformed or aliased status views. The CPU operations refuse duplicate or empty output names and input/output collisions.
- `CPURasterRegionGraphOperation` rejects `searchStructure: 'boruvka'` with adaptive merging, as the GPU does.

**Network**
- `GPUNetworkIsochrones` throws at construction for `cellOutline.table.{sums, sumValues, minimums, maximums}`; use `GPUCellAggregation` with values.
- `GPUMapMatchedEdgeVolumes` refuses `output.transitions.zoneOutWeights` and `zoneInWeights` and points to zone counts instead.
- `GPUNetworkNoding` refuses `pieces.geometry.sourceIds` and piece status fields other than count, requiredCount, overflow, candidateOverflow, vertexCount and requiredVertexCount. Neighbourhood and subgraph candidate-loss outputs that alias inputs are refused.
- CPU cost-matrix budgets must be within 1 to 1024 rounds and 1 to 64 local iterations. Isochrone outputs nested in the landscape recipe cannot alias across stages. Packed landscape inputs need STORAGE usage and must not be strided.

**Joins**
- `GPUNearestFeatureJoin` throws when `nearestFeatureIds` is combined with any neighbours-only output.
- `GPUNearestFeatureWeights` accepts only `overflow` and `candidateOverflow` in `output.status`. The predicate-join weights status accepts only count, requiredCount, overflow, candidateOverflow and uncertainCount. Other supplied fields throw.
- The CPU and GPU join operations reject `dwithin` combined with `output.relate`, and distances above 3.4028234663852886e38.
- Joins refuse join-only destination keys, missing-overflow descriptors and candidate flags that alias inputs.
- The multipart buffer selection refuses chunked multipart positions; pack them first. Point-in-polygon membership and counts need separate buffers. `featureValues` cannot alias a join status word.

**Geometry and topology**
- `GPULineNoding`, `GPUSnapRoundedNoding`, `GPUPolygonize`, `GPUMakeValid` and `GPUPolygonPrecisionReduction` (top-level, noded and directed) refuse the status fields `invalidCount`, `iterationCount`, `converged`, `iterationLimitReached` and `approximation`. `GPUTopologyStatusPort` no longer declares them.
- `GPUPolygonOverlay`, `GPUBufferSurface` and `GPUSphericalPolygonOverlay` refuse the same five fields, including boundary `uncertainCount` (planar) and inherited result fields (spherical).
- Final `requiredCount` is refused by make-valid, polygonize, precision reduction and the spherical overlay result; use `requiredRingCount`.
- `GPULineClipByPolygon` and `GPULineSplit` refuse `pieces.geometry.sourceIds` (use `pieces.sourceIds` or `pieces.sourceFeatureIds`) and eight nested status diagnostics (use `output.status.uncertainCount` and `pieces.status.overflow`).
- `GPURectangleClipOutput` no longer has `sourceRows`, `measures` or `status.approximation`. The compact offset-curve output no longer has `status.approximation`. Supplying them is a type error as well as a runtime one.
- Line simplification refuses diagnostics without an IDs selection and unsupported status keys. Diagnostic outputs that alias inputs throw. `GPUSharedPaths` refuses overlapping output words.
- `GPUVoronoiDiagram` `invalidCount` and `converged` need separate buffers. Areal interpolation's nested `weights.status.overflow` must alias `output.status.overflow`. S2 refuses `frontierCapacity` in uniform mode.
- Polygonize refuses zero-row or non-uint32 `uncertainCount` views. Make-valid refuses malformed nested uncertainty views. Topology statuses refuse writable ranges that overlap across meanings or stages.
- Predicates refuse aliased columns and aliased row, diagnostic or vertex provenance outputs. Geometry measures refuse unsupported output keys.
- `GPUTrackSimilarity` sequence metrics require single-part traversals.
- Catalog host packing rejects nonfinite projected float32 coordinates before workspace creation, instead of exhausting capacity retries.

**Trajectories**
- `GPUTrajectoryLocations` and its CPU forms reject negative, fractional, null and above-256 `maximumIterations` (1e9 used to be accepted and ignored). Empty, non-uint32 or non-packed status destinations are refused.

## Added

**Regression**
- `REGRESSION_TILE_MAXIMUM_TILE_COUNT` (2^24) replaces the 4,096-tile refusal in GM, 2SLS, weights traces, GWR and both likelihoods. Traces, GM, both likelihoods and OLS expose `estimatedCommandNodeCount`, with a 32,768-command limit.
- `GPUSpatialPanelErrorMaximumLikelihood` takes `effects: 'unit' | 'time' | 'twoWay' | 'random'` (default `'unit'`). Time effects demean across entities, and twoWay demeans across periods, then entities. GPU time and twoWay need predictorCount + periodCount - 1 <= 15. `'random'` is refused because its covariance needs a second determinant that the trace-series fit cannot compute.
- New `GPUSpatialPanelErrorMaximumLikelihoodOperation`, `CPUSpatialPanelErrorMaximumLikelihoodOperation` and `computeSpatialPanelErrorMaximumLikelihoodOnCPU`, plus three pooled-diagnostic status constants. The operations take `entityCount`.
- `GPURestrictedRegimeRegression`, its operation and the CPU twin take optional `robust: {type: 'white', smallSample?} | {type: 'hac', kernelWeights}`, which gives HC0, HC1 when `smallSample` is set, or Kelejian-Prucha spatial HAC. Restriction tests then use a robust covariance. `output.robust` returns the covariance, the robust joint `chow` chi-square (the existing `output.chow` stays the nested F), the stability test and a status.
- Exported for restricted regimes: `RestrictedRegimeRobustOptions`, `RestrictedRegimeRobustOutput`, `RestrictedRegimeKernelNames`, `CPURestrictedRegimeKernel`, and `RESTRICTED_REGIME_ROBUST_OK`, `_FIT_UNAVAILABLE`, `_INVALID_KERNEL` and `_INVALID_COVARIANCE` (0 to 3). Omitting `robust` leaves results and command nodes unchanged.
- `GPUSeeminglyUnrelatedRegression` and its CPU and operation forms take `iteration: {maximumIterations, tolerance?}` for iterated FGLS. The budget includes the first GLS fit (GPU 1 to 256), the stopping rule is a relative coefficient change below `tolerance` (default 1e-5), and spatial mode holds the first-stage GM parameters fixed. Without it, the one-step behaviour is kept.
- Also on SUR: a runtime `parameters` view of length `GPU_SUR_PARAMETER_LENGTH` = 1, packed by `getGPUSeeminglyUnrelatedRegressionParameterValues`, overrides the tolerance without recompiling. `output.iterationCount` counts completed solves, status 4 means the budget ran out with the last fit kept, and `GPU_SUR_STATUS_INVALID_PARAMETERS` = 5.
- `GPUProbitRegression` and its operation take `spatialLag: {weights, output: {coefficients, standardErrors, status}}`: a one-step weighted linearized GMM estimate of a latent spatial-lag probit about rho = 0. Instruments are [X, WX, W²X], errors are HC0 and islands stay as zero rows.
- The probit `spatialLag` status is `GPU_PROBIT_SPATIAL_LAG_STATUS` (0 to 5). Its weights status carries count and requiredCount. It has not been matched against R `spprobit`. In simulation it estimated rho = 0.2 as 0.217 and rho = 0.6 as 0.855, so do not read it as unbiased.
- `GPUOrdinaryLeastSquaresModelResultOptions.predictorCount` decodes an active coefficient prefix (default `coefficients.length - 1`).
- Decoded robust models carry `covarianceInference`. `SpatialTwoStageLeastSquaresOperationProps.robust.kernel.candidateOverflow` is a new named HAC input. The pure CPU 2SLS input accepts `robust.candidateOverflow` and `robust.kernel.status.{overflow, candidateOverflow}`.
- Named diagnostics and panel forms keep optional `weights.count` and `weights.requiredCount` graph names, and named 2SLS HAC forms keep `robust.kernel.count` and `requiredCount`.

**Multipart geometry and joins**
- Multipart features are now accepted throughout the joins. Every `GPUSpatialPredicate` name works on multipart features with `pattern`, `distance`, `engine` and `how`, and results are reduced per feature. Anti joins return `unmatchedRows`. Opt-in `output.relate` gives the packed DE-9IM (bit 31 marks uncertainty), and ten `output.topological` columns follow it. `CPUFeatureJoinOptions.relate` defaults to false.
- Multipart matrix candidates use 256 lanes per candidate. `GPUSpatialJoinCapacityPlanningOptions.robustPointPolygon` is new, and the one-candidate estimate is corrected from 256 to 192 bytes.
- `GPUNearestFeatureJoin` nearest-feature mode accepts multipart points and lines and line/polygon operands, giving one result per query feature. These routes ignore `candidateCapacity` and publish `candidateCount` 0; BVH leaf loss still marks results incomplete.
- `GPUBufferSelection` and `GPUPointInPolygonJoin` accept `GPUMultipartPointGeometryPort`; outputs and source IDs are packed per feature. `GPUPointInPolygonJoinOutput` adds `pointMatchCounts` (distinct matching polygons per feature, sized by `candidateCapacity`) and an opt-in dense `featurePolygonMembership` matrix, which must fit one storage binding. Operation membership is optional.
- `GPULineLengthPerPolygonProps.featureOffsets` applies one weight row to all parts of a feature. Multipart choropleths work on both backends.
- New pure CPU functions `computeBufferSelectionOnCPU`, `computePointInPolygonJoinOnCPU`, `computeLineLengthPerPolygonOnCPU` and `computePointsInPolygonsChoroplethOnCPU`, with input and option types and matching GPU and CPU operation factories. They do not take the geographic, preparation, palette or class-break options of the contributors and recipe.
- Grouped (multipart) input is accepted by `GPULineNoding`, `GPUSnapRoundedNoding`, `GPUPolygonize`, `GPULineSplit`, `GPULineClipByPolygon`, `GPULineLoopRemoval`, `GPUGeometryValidity`, `GPUBufferSurface`, `GPUSharedPaths`, `GPUSegmentGeometry`, `GPUMinimumClearance`, `GPUOffsetCurveComponents` and `GPUTrackSimilarity`.
- `GPUGeometryPredicates` accepts multipoint and multiline `featureOffsets`. Multipart attributes follow Shapely: numGeometries is the part count, isClosed requires every part to be closed, and numPoints, isRing and isCcw are 0. Equality distinguishes single-part from multipart types and different part splits.
- New provenance outputs: `pieces.geometry.featureOffsets`, `pieces.sourcePartRows`, `pieces.sourceFeatureRows`, `boundary.sourceFeatureRows`, `leftPartRows`/`rightPartRows` on `GPUSharedPaths`, and `output.featureOffsets` on loop removal. Offset curve component word 4 holds the feature row when grouping is supplied.
- `GPUTrackSimilarity` and its search accept `featureOffsets` for Hausdorff, maximum distance and SSPD, with logical top-k.
- Geometry measures can return area for points and lines (exactly zero).

**CPU budgets and options**
- `CPUCostDistanceOptions` takes `maximumIterations`, `maximumTieIterations` and `sweepAfterIteration`.
- `CPUNetworkCostMatrixOptions` and `CPUNetworkCentralityOptions` take `maximumIterations` and `localIterations`. `CPUAlternativeRoutesOptions` takes `maximumExpansions` and `maximumFrontierSize`. `computeNetworkServiceAreasOnCPU(input, options)` and `CPUNetworkServiceAreasOptions` take cost, label, tie and local-hop bounds and can skip predecessors.
- New CPU raster options: `CPURasterWalkingCostOptions.maximumIterations`, `CPURasterRegionGraphOptions.searchStructure` ('auto' | 'dense' | 'boruvka') and `computeTerrainWatershedsOnCPU` `maximumIterations`.
- New CPU trajectory options: `computeTrajectoryLocationsOnCPU` `maximumIterations` with an `iterationCount` result, plus `output.status.iterationCount` on both trajectory-location operations. `computeTrajectoryRecursionsOnCPU(input, {maximumTrackSegments})`.
- New CPU geometry options: `maximumIterations` on CPU convex skeleton and ratio hull, a work-options third argument on CPU minimum circle, a fifth `maximumIterations` argument on CPU map colouring and a third budget argument on the geodesic pairs helper.
- CPU S2 traversal takes `maximumVisitedCells` (6 to 1,048,576) and `frontierCapacity`.

**Other additions**
- `GPURasterTemporalResampleProps.gridEnd` overrides the last grid coordinate.
- `CostDistancePathOperationProps` adds optional `candidateOverflow` and makes `requiredCount` optional. `CPUCostDistancePathResult` adds `candidateOverflow`.
- `LineSimplificationOperationProps.output.status.candidateOverflow` is supported, and `CPULineSimplificationResult` returns it as [0].
- Delaunay operations and the CPU result add `candidateOverflow`, `uncertainCount` and `approximation`. `GPUVoronoiDiagram` takes an optional `triangleStatus` and suppresses dependent segments on upstream failure.
- `GPURegionMask` selections accept `status.overflow`, including through `{kind: 'mask', ...producerOutput}`.
- `GPUMaxPRegions` and `GPUAZPRegions` accept `weights.status` overflow and candidateOverflow names, and their pure CPU inputs accept `input.status`. CPU neighbourhood-summary weights status accepts `candidateOverflow`, `iterationLimitReached` and `converged`.
- Contiguity pair growth exposes `estimatedRequiredCount` and `limited`. Multivariate Geary inference and `GPUTransitionDiagnostics` expose `estimatedCommandNodeCount`. The lens tract weights expose `weightsCandidateOverflow`.

## Deprecated spellings
- Probit operation `spatialLag.weights.overflow` still works; use `spatialLag.weights.status.overflow`. Supplying both with different names is refused.

## Documentation and tests
- New pages: `docs/api/gpu-terrain/output-oracles.md`, `docs/api/gpu-terrain/status-and-count-evidence.md`, `docs/api/gpu-network/status-oracles.md` and `docs/api/gpu-spatial-analysis/spatial-weights-audit.md`, which is linked from operators-weights.md.
- The cost-distance and terrain guides describe GPU round counts as schedule-dependent with bounded increments. The 40-repeat determinism matrix moved to the opt-in `gpu-cost-distance-round-determinism-benchmark.spec.ts`, and correctness specs check convergence and independent Dijkstra costs instead of equal counts.
- The REDCAP docs no longer claim scikit-learn equivalence for arbitrary connectivity. Equivalence needs connected distinct-cost graphs, the same preprocessing and the same edge ordering.
- The CV+ docs drop the universal coverage claim for arbitrary group and spatial folds and state the assumptions behind the theorem.
- The spatial-entropy `mutualInformation` docs drop the incorrect Altieri attribution.
- The GTWR AICc illustration gap at n = 200, trace 10 is corrected from 0.05 to 0.24.
- The regression guides document the GM exact-affine certificate limits, slot 18 (which excludes the stochastic trace error), the `infinityNorm` meaning of the multiplier-bound fourth argument, and the panel-error `effects` limits.
- The joins guide documents multipart semantics, the precision limits of nearest search and the CPU float64 versus GPU float32 differences of the new CPU functions.
- The guides document the binding CPU budgets, how they differ from no budget and the incomplete-result statuses for cost distance, raster, terrain, network, trajectories and geometry refinement.
- operators-cells.md, operators-geometry.md and operators-lines.md list the supported topology status vocabulary and the alternative for each refused field.
- The location change-point guide states how to scale the penalty when units change, and the Max-P scratch formula is corrected to startCount·rows·(2·columnCount + 12)·4 bytes.
- The S2 docs correct the zero-radius cap: a face-centre cap covers one level-0 face, and the four touching cells apply from level 1.
- The Gamma adapter mapping is documented as holding only for identical effective weights and donor populations.
- The analysis README and the point-density binding description state that CPU and GPU interior-boundary cells of the point density grid can differ under WGSL accuracy rules.
- The website `LensSummary` fields are renamed with no aliases: `buffers` to `resourceCount`, `bufferBytes` to `resourceBytes` and `bufferBytesKind` to `resourceBytesKind`, and `WorkflowRecord.buffers` to `resourceCount`. The page histogram adds a zero-compute bin.
