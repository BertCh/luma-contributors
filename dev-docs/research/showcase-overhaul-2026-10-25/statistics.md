# Statistics, weights, regression and interpolation research and delivery ledger

This track owns the showcase's `statistics`, `weights`, `regression` and `interpolation`
chapters. Planning and implementation are iterative. A planned feature is not a delivered
feature, and a constructor call is not evidence of numerical or browser validation.

## Baseline and selection

Read before changing scenes: repository `AGENTS.md`, the showcase README and its builder,
cartography and data guides, and the October 24 sprint roadmap, design and API audit.
The existing stories already contain considerably more than a basic API demonstration:

- `segregation`: genuine ACS counts, five groups, aspatial and five spatial scales,
  environment versus smoothed-population interpretations, self weights, several kernels,
  Atkinson aversion, whole-row shuffle, map swipe, composition and scale profiles.
- `health-regression`: genuine Chicago PLACES/SVI/ACS observations, complete-case connected
  study set, OLS, residual diagnostics, lag 2SLS and error GM, alternative weights and
  predictors. Its automatic interpretation overstated nonsignificant diagnostics and
  selected a model by the smaller robust LM p-value. It had no out-of-fold prediction map.
- `local-relationships`: national county GWR, OLS comparison, fixed/adaptive bandwidth,
  kernel, local uncertainty/condition/influence, nonstationarity permutations.
- `hot-spots`, `global-autocorrelation`, `spatial-weights`, `rate-smoothing`: substantial
  neighborhood, permutation, multiple-testing, empirical-Bayes and graph algebra coverage.
  Recipe-composed contributors must not be misreported as absent solely because a direct
  constructor is missing from the static reporter.
- `rainfall-interpolation`: genuine NOAA Helene gauges, IDW, fitted variograms, ordinary
  kriging and variance, directional variogram inspection, focal statistics, random and
  spatially blocked holdout. The existing blocked design is explicitly one fold, with
  no inter-block exclusion buffer. Adding another copy of that story would add little.
- `change-of-support`: extensive/intensive transfer, ancillary support, multiple grids,
  conservation and pycnophylactic smoothing already deserve preservation.

The initial static snapshot at 2026-10-25 19:32 UTC had 492 public classes/recipes and 235
with direct runtime call evidence across the full showcase. Those totals are provisional
while other tracks and package APIs change; use the global coverage ledger for final counts.

## Research and provenance

### Spatial model diagnostics and validation

[PySAL's specification-test notebook](https://pysal.org/spreg/notebooks/9_specification_tests.html)
and [the LMtests contract](https://pysal.org/spreg/generated/spreg.LMtests.html) describe tests
for particular omitted spatial dependence alternatives. Failure to reject does not establish
model adequacy. Comparing two p-values is not a general model-selection procedure. We retain
the tests, disclose failed/nonfinite diagnostics, and frame them as evidence to investigate.

The local `GPUSpatialCrossValidatedOLS` contract supplies separate training-only OLS fits,
held-out predictions, fold counts, missing scores and metric status. It accepts raw predictor
columns and internally centers each fit using its training mask. The design will compare
row-hash and spatial-block folds, with an explicit exclusion buffer, and aggregate squared
errors by scored row count. It will not average fold RMSEs or call in-sample fit validation.
The contributor is unpenalized; the scene must say so if the separate full-data OLS ridge
control remains visible. Geographic blocking is a stress test of transfer, not proof of
future or causal validity.

### Segregation counterfactuals

[PySAL's decomposition API](https://pysal.org/segregation/generated/segregation.decomposition.DecomposeSegregation.html)
and [worked decomposition](https://pysal.org/segregation/notebooks/07_decomposition_example.html)
motivate composition, share and dual-composition counterfactuals and two-factor Shapley
attribution. The local GPU recipe has a narrower explicit contract: paired focal/complement
counts on the same support, at most 2,048 units, four dissimilarity values, validity flags,
two contributions and an arithmetic closure residual. Its names describe a decomposition
under a chosen construction, not causal demographic mechanisms.

The bundled Chicago ACS observations are genuine. Any redistribution of those observations
is a visibly labeled scenario, not a second observed census, historical trend, or policy
prediction. Paired label-exchange inference requires exchangeability and will not be added
to an invented scenario merely to demonstrate an API.

### Meuse floodplain topsoil: distinctive new prediction case

Primary source: [the sp package's Meuse documentation](https://raw.githubusercontent.com/edzer/sp/master/man/meuse.Rd),
compiled by Edzer Pebesma from field observations by Ruud van Rijn and Mathieu Rikken;
the documentation cites their Utrecht University 1993 field report. The
[gstat tutorial](https://github.com/r-spatial/gstat/blob/main/vignettes/gstat.Rnw) motivates
log zinc versus distance to the river and spatial covariance, and the
[gstat API examples](https://r-spatial.github.io/gstat/reference/gstat.html) show local
ordinary/universal prediction. These are a statistical teaching case, not current exposure
monitoring or an assertion of a regulatory threshold.

Source extraction inspected on 2026-10-25:

- [Maintainer-hosted JSON](https://edzer.r-universe.dev/sp/data/meuse/json): 155 rows,
  14 fields; measured zinc ranges from 113 to 1,839 mg/kg soil.
- [Package metadata](https://edzer.r-universe.dev/api/packages/sp): sp 2.2-4, source
  commit `eae4821e83b8b9eb5fc46c9f05bb9523747c8598`, package distribution GPL (>= 2).
  Data attribution and upstream terms must remain separate from first-party MIT code.
- Coordinates are Dutch RD, EPSG:28992, in meters. Preserve original coordinates for
  reproducibility and transform to WGS84 for map placement. Analysis must use one explicit
  metric frame, consistently for folds, neighborhoods and covariance ranges.
- Metal concentrations come from approximately 15 m by 15 m composite samples. Treating
  them as points is an approximation, and point versus block prediction changes support.
- Cadmium zeros in the supplied dataset were shifted to 0.2; disclose this if cadmium is
  offered. Normalized distance `dist` is not meters; `dist.m` is the field distance in meters.
- The 155-row dataset is a filtered historical subset. The gstat `meuse.all` documentation
  explains that eight pit samples and one zinc outlier were excluded; no fresh filtering
  should be hidden in the showcase.
- A log-concentration prediction and its variance have log units. Exponentiating a mean
  on that scale gives a median under a lognormal model, not an unbiased arithmetic mean.

The [prediction-grid documentation](https://raw.githubusercontent.com/edzer/sp/master/man/meuse.grid.Rd)
warns that the grid flooding-frequency map has unknown construction and that its soil map
may not represent a real mapped soil survey. Prefer grid coordinates and normalized river
distance for prediction; do not present grid soil/flooding categories as validated measured
covariates. The [river outline](https://raw.githubusercontent.com/edzer/sp/master/man/meuse.riv.Rd)
has 176 supplied coordinates and can give the new story geographic context.

The downloaded source is research evidence at this stage. It is not counted as shipped
until a reproducible builder, catalog record, bytes and working scene are present.

### Historical county panels: NCOVR, not fabricated repeated observations

[GeoDa's NCOVR source page](https://geodacenter.github.io/data-and-lab/ncovr/) supplies
3,085 county units and four census-era snapshots, and its downloadable archive contains
GeoJSON, CSV and the original codebook. The codebook explains that counties were combined
to accommodate changing boundaries: preserve those supplied units rather than joining them
to today's county polygons. The archive's `NAT.prj` specifies WGS84.

[PySAL's panel estimation example](https://pysal.org/spreg/_modules/spreg/panel_fe.html)
uses the real HR70/80/90 outcomes with RD and PS covariates. A showcase lab can instead
prefer directly interpretable contemporaneous unemployment and income fields, on a
documented state subset, while comparing pooled and unit effects with the same weights.
The original resource-deprivation component includes racial composition; do not casually
present that composite as a causal mechanism or individual risk score. The outcome is a
three-year-average homicide rate per 100,000 residents; `HCyy` is also a three-year average,
not a raw integer Poisson count. These are historical area associations, not current crime
forecasts. The source states “as is”; no explicit open license was found on the source page,
so do not relabel the source public domain. Attribution must name the NCOVR authors.

Inspected source archive: `/tmp/statistics-ncovr.zip`; text extracted from its bundled
codebook with `pdftotext`. No dataset bytes from this candidate are shipped yet.

### Neighborhood differences, association and variability

[PySAL's local Geary guide](https://pysal.org/esda/stable/user-guide/local_geary.html)
provides the substantive distinction from Moran: neighbor differences versus products of
mean-centered values. [Multivariate Geary](https://pysal.org/esda/stable/generated/esda.Geary_Local_MV.html)
is not a collection of independently shuffled univariate columns. The local GPU permutation
implementation preserves whole observed vectors, with complete-case standardization and
explicit equal-tail versus undoubled minimum-tail conventions. Show that distinction.

[Lee's local spatial Pearson API](https://pysal.org/esda/stable/generated/esda.Spatial_Pearson_Local.html)
describes association of spatially smoothed variables. This must not be called a causal
effect or interchangeable with ordinary Pearson correlation.

[LOSH documentation](https://pysal.org/esda/stable/generated/esda.LOSH.html) distinguishes
squared and absolute local residual deviations. The local `GPULOSHResidualPermutationTest`
has a deliberately different null: it conditionally permutes the fixed normalized residual
field, without refitting local means on each draw. Its own source says this is not esda LOSH
inference. Any showcase comparison must name that null next to its p-values. The
`addLocalVariabilityRecipe` exposes descriptive LOSH and separate analytic Moran tests;
its multiple-testing correction applies only to Moran rows, never a fabricated joint family.

## API and option coverage matrix

The following matrix is a family-level plan informed by public source contracts. “Existing”
means a baseline call path and controls were inspected; it does not assert every overload.
Detailed optional/compile-time settings are listed so whole families are not marked complete
after one default constructor call.

| Family / public APIs | Existing evidence | Important option surface | Delivery plan / remaining limits |
| --- | --- | --- | --- |
| `GPUSegregation` | segregation | 5 groups, 6 scales, spatial form, self weight, Atkinson aversion, kernel, row transform | Preserve; add explicit counterfactual comparison |
| `addSegregationCounterfactualRecipe`, `addSegregationDecompositionRecipe` | No direct baseline calls | composition / dual-composition / share, common support, four indices, validity, arithmetic closure; optional paired exchange | Wave 1: three-method scenario with visible decomposition; randomization not justified by simulated scenario |
| `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM`, `addSpatialRegressionRecipe` | health-regression (some through recipe), local-relationships | response/predictor set, ridge, weights, instrument order, covariance/estimator mode, status, conditioning | Wave 1: correct interpretation and validation; audit covariance/GM choices in later pass |
| `GPUSpatialCrossValidation`, `GPUSpatialCrossValidatedOLS` | First already used in rainfall; second absent | explicit blocks/fold count, seed, buffer, fold masks/status, all-fold fits and score support; validation-design modes | Wave 1: raw predictors, row/block comparison, buffered five-fold out-of-fold map and metrics; do not claim all validation-design modes |
| `GPUGeographicallyWeightedRegression`, nonstationarity test | local-relationships | fixed/adaptive bandwidth, kernel, neighbor cap, local status, significance, permutation seed/count | Preserve; connect collinearity and multiscale comparison |
| `GPUGeneralizedGeographicallyWeightedRegression` | absent | Poisson/logistic family, offset, fixed/adaptive, leave-one-out, excluded prediction, bandwidth candidate selection, covariance, iteration/tolerance | Later lab: genuine count or binary outcome with matching likelihood and explicit failures |
| `GPUMultiscaleGeographicallyWeightedRegression`, `GPULocalCollinearity` | absent | per-variable bandwidth, finite candidate ladders, backfit convergence, conditional versus coupled covariance, VIF/design condition | Later lab: bounded sample; display local stability and bandwidth sensitivity |
| ML spatial lag/error, `GPUSpatialWeightsTraces`, `GPUSpatialLagImpacts` | absent | trace exact/stochastic strategy, power/probes/seed, parameter bounds, convergence, approximation error, direct/indirect effects | Later regression lab: small genuine region and residual/effect comparison; coefficients are not effects |
| Panel lag/error ML, `GPUSpatialPanelRegression` | absent | balanced period-major layout; 2SLS pooled/unit effects and instrument order; lag ML random/unit effects, conditional covariance and optional Hausman (<=32 entities); error ML unit effects and pooled diagnostics | Needs real common-support panel; never manufacture repeated observations from one snapshot |
| `GPUSpatialRegimeRegression`, `GPURestrictedRegimeRegression`, `GPUSeeminglyUnrelatedRegression` | absent | explicit regimes, pooled/varying restrictions, equation covariates, residual covariance and status | Later regional model lab using documented regimes/multiple measured outcomes |
| `GPUProbitRegression`, `GPUInstrumentalVariablesCovariance` | absent | binary response, iteration limits; IV HC0/HC1/HAC with exact structural residuals/design/instruments | Binary outcome must be defensible; HAC kernel PSD is a caller precondition, not arbitrary spatial weights |
| Contiguity/lattice/search, transforms, lag, algebra, transpose, summaries | spatial-weights, hot-spots, several other scenes | queen/rook/kNN/radius, kernels, row/binary/global transforms, set operations, self weight, subgraph, higher order, capacity | Existing broad coverage; validate/export actual graph diagnostics rather than adding a duplicate lesson |
| `GPUSpatialWeightsValidation`, `GPUFuzzyContiguity`, `GPUIslandAttachment` | no direct baseline calls for validation/fuzzy | invalid indices, nonfinite/negative weights, row order, symmetry/islands; tolerance; attach policy | Later weights QA step: show defects and correction, distinguish artificial attachment from observed adjacency |
| Global Moran/Geary/G, local Moran/Gi*, global/local permutations | global-autocorrelation, hot-spots | statistic, weights, null/alternative, seed/count, island/overflow policy | Existing; inspect gate semantics and minimum attainable p |
| `GPUMultipleTestingAdjustment` | used through recipes | BH/FDR, Bonferroni etc., valid hypothesis mask | Existing recipe coverage; explain dependence assumptions and tested family size |
| Local Geary, multivariate Geary, spatial Pearson, Gamma, LOSH and corresponding permutation adapters | absent direct calls | neighborhood, variable standardization, heteroscedasticity residual null, correction and validity | Later multivariate neighborhood lab with visible patterns and one consistent inference contract |
| Assuncao–Reis / empirical-Bayes / spatial EB / excess risk | rate-smoothing (EB through recipe) | raw versus stabilized rate, denominator, weights, zero exposure, prior strength | Add only distinct interpretation; existing birth denominators are real, not invented events |
| Regionalization AZP / MaxP / Ward / MST | regionalization baseline inspected at family level | feature standardization, contiguity, minimum mass, feasible status, seed/iterations | Later broaden algorithm comparisons with one region table and constraint failures |
| IDW / raster kriging / variogram / focal / interpolation diagnostics | rainfall-interpolation; diagnostics recipe absent | neighbor/minimum support, power/radius, variogram model/lag/direction/weighting, variance, focal shape/holes, LOO | Preserve rainfall; Meuse offers model selection and distinct local predictors |
| `GPULocalKriging` | absent | simple/ordinary/universal/external drift/co-kriging, known means, PSD structures, errors, support quadrature, k/radius/status | Wave 2: prediction lab; avoid pretending one covariance model estimates its own parameters |
| `GPULocalRBF` | absent | eight kernels, polynomial degree, epsilon, global/per-row smoothing, index candidates, radius/k | Wave 2: compare selected kernels with rank/status and support diagnostics |
| `GPUNestedInterpolationValidation` | absent | outer masks, inner blocks/folds/buffer, candidate ladder, k, selection scores/status | Wave 2: held-out outer outcomes never tune candidates; all-fold metrics and failed-support counts |
| `GPUNaturalNeighbor` | absent | direct <=64 versus indexed Delaunay, hull policy, vertex/neighbor budget, weights | Later Meuse subset or indexed path; document sampling and outside-hull holes |
| `GPUConditionalGaussianSimulation` | absent | known mean, covariance, measurement error, <=128 samples/queries, seed/replicate offset, retained preparation | Later bounded uncertainty lab, empirical moments versus analytic moments and realization map |
| Areal interpolation / pycnophylactic / dot density | change-of-support, dot-density | extensive/intensive, ancillary, grid/resolution, mass conservation, iteration | Existing substantial story; retain support caveat and conservation evidence |
| Classification / composite score / inequality / similar places / group statistics | respective statistics scenes | class method/count, standardization, weights/ranks, distance metric, group definitions | Audit within coherent existing stories; no claim of every enum covered yet |

## Implementation waves and acceptance

1. **Trust the diagnostics:** fix health-regression interpretation, then add five-fold
   training-only buffered validation and map actual out-of-fold residuals. Expand genuine
   ACS segregation with explicit scenario counterfactuals and decomposition readouts/chart.
2. **Choose a surface without peeking:** ship Meuse source/builder and a local prediction
   lab, including nested validation rather than best-looking in-sample tuning.
3. **Represent uncertainty and alternative models:** bounded conditional simulation,
   natural neighbor, multivariate/local statistical families and regression estimator labs.
   Dataset choices are researched before implementation; broad plans are not delivery claims.

Each visible control must alter a computed result or a clearly labeled display. Compiler
variants use `GraphSwap`; small asynchronous summaries are version/selection-safe; unavailable,
singular, unconverged, unscored and outside-support outputs are distinct from valid zero.
Maps, legends, tooltips and charts agree about units, scaling and sign. New observations
remain tied to provenance and any sampling or transformation is written in the builder.

## Delivery and verification ledger

In progress. The planner has inspected public source contracts and existing scene code,
verified the Meuse source shape against maintainer documentation, and requested the
implementation described above from one Luna implementer. The root agent owns
repository-wide installation, formatting, builds and test gates.

### Independent health-validation numerical reference

The planner ran a separate double-precision OLS solver against the bundled source columns,
using the scene's complete-case/largest-component selection and float32 metric positions.
The reference implements the public contributor's uint32 fold hash and explicit buffer
exclusion, but computes OLS and scores independently of GPU kernels. It is a numerical
oracle for subsequent GPU comparison, not evidence that the GPU has already passed.

- Command: bundle `/tmp/showcase-statistics-review/reference.ts` with local `esbuild`, then
  `node /tmp/showcase-statistics-review/reference.cjs`.
- Output retained as [health-validation-reference.json](health-validation-reference.json).
- Study rows: 788. Design: 3,000 m blocks, seed 31, five folds, 500 m training exclusion.
- Test rows per fold: 106, 249, 135, 159, 139; all 788 rows tested exactly once.
- Buffer-excluded training rows: 13, 24, 6, 13, 9.
- Diabetes income-only aggregate held-out RMSE: 3.1939252809 percentage points.
- Diabetes economic four-predictor aggregate held-out RMSE: 2.6783553260 percentage points.
- Numeric aggregation uses `sqrt(sum(fold MSE * scored count) / sum(scored count))`.

### Review findings resolved before acceptance

First-diff review caught a counterfactual recipe whose internally allocated result views
were not copied to the caller-owned readback buffers, and a two-group dissimilarity result
incorrectly labeled multigroup. These were returned to the implementer for correction,
along with explicit method/scenario controls, a visible counterfactual map, out-of-fold
residual mapping, and stale-readback guards. Do not treat initial code compilation as proof
that these runtime/scientific issues are resolved. Final status will be recorded here.


### Independent segregation-scenario reference

[segregation-scenario-reference.json](segregation-scenario-reference.json) records an
independent double-precision two-group dissimilarity calculation on the scene's float32
scenario inputs, for all five focal groups and strengths 0, 0.1, 0.25, 0.5 and 1.
The source has 791 tracts and 2,716,047 residents across the five supplied groups.
For non-Hispanic Black residents, baseline D is 0.7663358608 and the 25% equalization
scenario gives D 0.5747518969. The analytic identity D(s) = (1 − s)D(0) provides an
independent cross-check; the 25% result differs by 1.3e-9 after float32 input rounding.
Tract totals change by less than 0.0005 person from rounding. These checks validate the
scenario construction and expected baseline/final D, not the GPU quantile counterfactual
construction or its decomposition. Browser readouts have not yet been compared.

### Wave 1 code-review checkpoint

The implementation now includes three decomposition methods, a 0–100% equalization
scenario, a two-map comparison on common aspatial tract support and paired ACS/scenario
shares in tooltips. Health validation includes spatial versus seeded row-hash folds and
0/500/1,000 m exclusion, actual fold-level errors and an out-of-fold residual/fold map.
Raw predictors are passed to training-only OLS; undefined R² is permitted without hiding
otherwise valid RMSE/MAE. Generation stamps guard asynchronous validation/decomposition
summary results.

Review also caught an invalid WGSL `isFinite` call, constant-evaluated NaN construction,
and an incorrect numeric status mask; the implementer corrected those. A subsequent
review found active validation variant lookup using live settings during asynchronous
compilation and an out-of-fold tooltip showing in-sample values. These have been returned
for correction and remain pending acceptance.

Scoped planner verification:

- `node scripts/check-story-steps.mjs statistics regression interpolation weights` from the
  showcase directory: passed all 18 existing scenes at this checkpoint. This validates
  story-control/readout/cartography metadata, not WebGPU execution.
- Independent CPU health and segregation references described above: passed their own
  conservation/fold-support checks; GPU comparison remains pending.
- Root installation, formatter, builds, node/headless tests and browser acceptance remain
  the parent agent's responsibility. No completion is inferred from targeted checks.


### Browser checkpoint from the parent

Parent report `/tmp/luma-showcase-overhaul-20261025/statistics-spherical-browser/report.json`
records seven segregation steps with submitted WebGPU work and no uncaptured GPU, console
or page errors. Planner inspected screenshots 5 and 7. The diversity/profile view is coherent;
the scenario chart's rounded baseline 0.77 and 50% scenario 0.38 agree with the independent
reference at display precision. This is not a full-precision numerical comparison.

Visual inspection found that the scenario step showed an always-present labeled swipe while
its comparison option defaulted off, so both halves displayed the ACS baseline. A fix is
required before accepting this step. The scene's alternative text also still said five-group
dissimilarity despite the two-group calculation; correction requested.

The same parent run failed health-regression initialization: separate imports of one physical
fold buffer violated the graph alias contract. Reuse a graph view for each physical buffer;
this fix and the asynchronous option-switch fixes remain under implementation. No health
WebGPU pass is claimed.

### Concrete panel follow-on constraints

The dense `GPUSpatialPanelLagMaximumLikelihood` contract differs materially from panel
2SLS and panel-error ML: it supports random/unit effects but only up to 32 entities, and
optional Hausman comparison can explicitly fail for an indefinite covariance difference.
A complete natural study unit fitting this bound is New Jersey's 21 counties in NCOVR,
with four genuine 1960/1970/1980/1990 snapshots (84 county-period observations). North
Carolina's 100 counties would not fit this lag-ML implementation. Source inspection found
all 21 NJ units and their supplied common-support polygons. This is a planned case, not
a shipped dataset or accepted model. Avoid nominal income trends without adjusting for
inflation/time effects; direct unemployment/poverty fields are easier to explain. Gaussian
working likelihood for historical rates and the small entity count must remain explicit.


### Meuse nested-validation independent reference

A reproducible independent reference is retained as
[reference-meuse-nested.py](reference-meuse-nested.py), with outputs in
[meuse-nested-reference.json](meuse-nested-reference.json). It solves ordinary kriging with
NumPy's double-precision linear algebra and local RBF models with SciPy's
[RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).
Positions and logged responses are first rounded to float32 to match the GPU inputs.
Nearest-neighbor ties use original source-row order. It does not emulate GPU conditioning
thresholds; numerical agreement and completion status still require execution comparison.

Exact command, from repository root:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/reference-meuse-nested.py /tmp/meuse-source.json dev-docs/research/showcase-overhaul-2026-10-25/meuse-nested-reference.json
```

The isolated temporary environment contains NumPy 2.5.4, SciPy 1.18.1 and pyproj 3.8.0.
No repository dependency or global Python environment was changed.

Accepted planned configuration: 155 samples, translated Dutch RD coordinates using the
sample-bounds midpoint, 16 nearest usable observations, 1 km outer tile hashes (seed 31)
and three outer folds; inner 500 m tiles, two folds and a 250 m exclusion buffer. There is
no outer boundary buffer. Tile hashes do not balance counts: outer tests contain 23, 54
and 78 samples. Inner training counts after exclusion are 20/36, 12/35 and 19/15.

The fixed five-candidate ladder is: exponential ordinary kriging with range 300 or 900 m,
spherical ordinary kriging with range 900 m (all sill 0.35, nugget 0.05); thin-plate RBF
with degree 1 and smoothing 0.02; Gaussian RBF with degree 0 and smoothing 0.08. Both
RBF candidates use epsilon 1/300 per metre. These are authored settings, not estimates
from the complete response. The independent reference chooses zero-based candidates
2, 2 and 4, with outer-fold RMSE 0.68393829, 0.85703860 and 0.95703617; pooled RMSE is
0.886690791 natural-log zinc units across all 155 held-out samples.

Reference review changed the Gaussian candidate from no polynomial augmentation to a
constant: without that term, its distant log-concentration predictions approached zero
(1 mg/kg after exponentiation), making it a poor baseline for this field. That earlier
configuration is not the accepted ladder. The primary
[gstat kriging examples](https://r-spatial.github.io/gstat/reference/krige.html) also provide
a clearly stated assumed mean of 5.9 for a possible later simple-kriging comparison and
a 40 m block example; neither implies the current grid-point predictions are block means.

### Inference resolution limitation for the next neighborhood lab

The public multivariate-Geary equal-tail randomization uses a minimum p-value of
2/(P+1), at most 4,096 replicates. With all 155 Meuse sites, Bonferroni at alpha 0.05
requires p <= 0.05/155 = 0.00032258, below the smallest attainable 0.00048816.
An empty significance map under this combination would be an inference-resolution
limitation, not evidence against spatial structure. A later option comparison must state
this constraint, or use a documented smaller support if familywise rejections are intended
to be possible. Quantile LISA's public source additionally labels its legacy directed
pseudo-p convention anti-conservative; do not portray that mode as two-sided error control.

### Subsequent review checkpoint: health and Meuse

The earlier health graph-import failure was repaired by reusing the existing view for
each physical buffer. Parent's production run in
`/tmp/luma-showcase-overhaul-20261025/wave3-fixes-browser/` completed all seven then-current
steps without GPU errors. Planner inspected the final two screenshots: they showed lag
and error fits, not the new held-out outputs. The story was consequently consolidated to
six steps with a dedicated final OOF residual/fold step. This newer progression still
requires production verification.

`check-health-validation.mjs` is a focused production-browser check for twelve configurations
(income/economic predictors, spatial/random assignment, 0/500/1000 m exclusion), against
`health-validation-options-reference.json`. It checks 788 scored rows, all fold test and
buffer counts, pooled RMSE and per-fold RMSE with 0.006 display tolerance. Syntax check
passed; execution is pending the parent's GPU slot. This is a display-precision comparison,
not a claim about full-precision GPU arrays.

Meuse's four-response independent nested references now include
`meuse-nested-copper-reference.json`, `meuse-nested-lead-reference.json`, and
`meuse-nested-cadmium-reference.json`. Run `reference-meuse-nested.py` as above with an
optional final argument `copper`, `lead`, or `cadmium`. Pooled log RMSE is respectively
0.576074337, 0.896793236, and 1.444621764. Selected zero-based candidates are [2,2,0]
for copper and [2,2,4] for lead and cadmium; zinc remains 0.886690791 and [2,2,4].

Planner review found and Luna repaired these additional Meuse issues before requesting
production acceptance:

- The surface status view previously included two trailing generation words; it now reads
  exactly the 3,103 query statuses.
- The validation generation is checked before handling failures, preventing stale failures
  from replacing a newer metal's summary.
- Pending OOF layers are omitted, rather than coloring observed values as residuals.
- Pending descriptive models are hidden through compilation and execution so a previous
  model is not shown under the new label.
- Response-specific log ranges are shared between layers and legends; variance/conditioning
  maps omit the differently scaled observed markers. RBF is labeled generically.
- The final interpretation step now defaults to the OOF residual map.

Scoped formatting and strict interpolation cartography pass according to the implementer.
Concurrent full checks have unrelated scene errors; no global or Meuse GPU pass is claimed.

### Parallel next waves and exact ownership

The app Luna continues with `meuse-natural-neighbor` and `meuse-conditional-simulation`.
Natural neighbor is planned to use every original site and grid query, composed with
`GPUDelaunayTessellation`, a GPU reciprocal-edge adjacency adapter, and explicit containing
triangle seeds. `GPUNaturalNeighbor` indexed mode exposes finite cavity/polygon budgets,
outside-hull support, degeneracy, and capacity status. The hull is a genuine interpolation
domain; filling holes by extrapolation would change the method.

The independent geometric reference follows the stolen-Voronoi-area definition in the
[CGAL interpolation manual](https://doc.cgal.org/latest/Interpolation/index.html) and uses
[SciPy Voronoi/Qhull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html)
for topology, with separate double-precision half-plane clipping. The retained verification
script is `reference-meuse-natural-neighbor.py`; it checks weight sums and affine coordinate
reconstruction, not just plausible colors. GPU comparison remains pending.

Conditional simulation is planned as a bounded, explicitly selected 128-site conditioning
subset and 128 real grid queries, with joint draws. The known log mean 5.9 and spherical
covariance (sill 0.59, range 874 m, nugget 0.04) are authored teaching assumptions, not
estimates or validated claims about the field. Planned options cover draw seed/count/offset,
mean sensitivity, measurement-error variance and retained-preparation lifecycle. Analytic
conditional moments will be checked independently; Monte Carlo moments need statistical,
not exact-sequence, agreement.

Parent explicitly authorized one additional `gpt-6-luna` CLI lane for the independent county
work. Started through exec session 46471, with prompt/log/result files prefixed
`/tmp/luma-showcase-overhaul-20261025/cli-statistics`. It owns only new
`county-panel`, `county-regionalization`, and `ncovr-new-jersey` scene/dataset/builder files
plus `county-lane.md`. Scene metadata is written last. Planned coverage is panel 2SLS
(pooled/unit), panel lag ML (unit/random/Hausman status), panel error ML (unit/pooled LM),
and coherent cross-sectional ML/traces/impacts; then AZP, MaxP, Ward sequential/mutual-nearest
and Dorling population cartograms. These are assigned plans, not delivered coverage.

The county panel will use 1970/1980/1990 unemployment, family poverty and source log density,
21 actual New Jersey units on NCOVR's supplied common historical support. The earlier
four-snapshot suggestion is narrowed because 1960 family poverty has a different definition.
Neither historical rates nor small-sample working likelihoods establish causal effects.

Further assigned waves remain Meuse neighborhood statistics and model form, population
support alignment/exposure/dasymetric redistribution, and cross/space-time variograms.
Earth's lane confirmed no population/exposure overlap: its Sentinel coverage/remapping
does not justify using NIR or land classes as population locations. A defensible population
ancillary source or explicitly authored allocation sensitivity must precede a dasymetric story.

### Production numerical acceptance, wave 5

On the parent's rebuilt production server `http://127.0.0.1:5313/`, the focused health check
passed all twelve configurations. Report:
`/tmp/luma-showcase-overhaul-20261025/health-numeric/report.json`. All 788 scored rows,
fold test counts and buffered exclusions agree exactly; pooled and fold RMSE agree within
0.006 of independent references. Both page/console errors and uncaptured GPU errors are
empty. OOF residual and fold screenshots were inspected: held-out maps and the five-fold
RMSE chart are visible. The selected economic/spatial/500 m pooled result is 2.678 percentage
points. The fifth fold's gray color resembles omitted tracts; a distinct fifth color was
requested as a subsequent presentation correction.

The focused Meuse check passed all four response references with all 155 held-out rows and
exact selected candidate names in every fold. Report:
`/tmp/luma-showcase-overhaul-20261025/meuse-numeric/report.json`. Displayed pooled RMSE is
0.887 zinc, 0.576 copper, 0.897 lead and 1.445 cadmium, compared with a 0.002 display tolerance.
Three direct kriging modes and eight RBF kernels were also exercised. Every direct variant
reported 3,103 valid queries except quintic RBF, which reported 2,795 valid and 308 rank
failures. These failures remain visible as missing predictions, not zeros. This direct-model
check verifies execution/status accounting and does not establish full-precision agreement
of every surface value. There were no page/console or uncaptured GPU errors.

Exact commands from repository root:

```sh
node dev-docs/research/showcase-overhaul-2026-10-25/check-health-validation.mjs http://127.0.0.1:5313/ /tmp/luma-showcase-overhaul-20261025/health-numeric
node dev-docs/research/showcase-overhaul-2026-10-25/check-meuse-validation.mjs http://127.0.0.1:5313/ /tmp/luma-showcase-overhaul-20261025/meuse-numeric
```

Initial harness attempts needed corrections for the count appended to the All controls
tab name, segmented metal controls, three-decimal health formatting and the revised word
“queries.” They did not reveal product GPU failures. The final reports above are successful
complete runs. Meuse residual and variance screenshots were inspected. Persistent footer
wording that incorrectly generalized the all-sample surface to the residual view, and the
residual's log-ratio unit, were identified for correction after this build.

The independent natural-neighbor reference completed all 3,103 query points: 296 triangles,
12 convex-hull vertices, 2,815 valid queries, 288 outside-hull queries, maximum 13 positive
natural-neighbor weights. Mean log-zinc prediction is 5.6669699113; range
[4.7408506797, 7.4855085026]. All valid weight sums differ from one by less than 1e-7 and
barycentric reconstruction errors are below 1e-4 m. These are CPU geometric reference
results, not yet compared with the forthcoming GPU scene.

### Production numerical acceptance, wave 9: geometry and joint simulation

The final focused production command passed on the parent's wave-9 build:

```sh
node dev-docs/research/showcase-overhaul-2026-10-25/check-meuse-labs.mjs http://127.0.0.1:5313/ /tmp/luma-showcase-overhaul-20261025/meuse-labs-numeric-final
```

Report `meuse-labs-numeric-final/report.json` has empty page/console and GPU-error arrays.
Natural-neighbor results agree with the independent geometry reference at display precision:
296 triangles, 2,815 valid queries, 288 outside the hull, no degenerate/capacity failures,
mean 5.667, range 4.741–7.486, maximum 13 contributors and reported weight-sum error zero.
The verifier checks mean/range to 0.002 and weight-sum error to 1e-4. It does not read back
and compare every individual prediction. Indexed budgets remain fixed at 64 neighbors and
64 clipping vertices; smaller-budget behavior is not covered. Outside-hull queries are
domain exclusions, distinct from computational failures.

Conditional simulation passed 22 configurations: all twelve known-mean (5.5/5.9/6.3) by
observation-error-variance (0/.01/.04/.16) combinations at 128 draws; all nine draw-count
(32/128/256) by replicate-offset (0/256/512) combinations at mean 5.9/error zero; and a final
variance-display change. Analytic mean and variance summaries agree within 0.0015 with
`reference-meuse-conditional.py` / `meuse-conditional-reference.json`, which independently
reconstruct the exact 128-site maximin and 128-query selection and use dense float64
Gaussian conditioning. The default mean is 5.0176986225 and average variance 0.1625582449.
Empirical mean and unbiased sample variance are within four standard errors plus display
rounding, using the full conditional covariance to account for dependence among queries.
This is a distributional moment check, not an exact Philox-sequence oracle. Seed changes,
every mapped replicate, and every per-query moment are not independently certified here.

The earlier wave-8 attempt stopped on a verifier wording mismatch (`outside hull 288`);
it did not reveal a numerical mismatch. A separate reserved WGSL variable `active` was
renamed to `isActive` before wave 9. The scoped WGSL checker subsequently passed 32,655
templates. Natural-neighbor and conditional moment/variance screenshots were inspected.
The simulation's compact query patch occupies too little of its overview; a subsequent
camera-only correction was requested for the detailed draw/moment steps, retaining the
initial geographic overview. Natural-neighbor wording was also requested to distinguish
hull-domain exclusions from numerical failures. These visual corrections postdate the
numerical acceptance and need the parent's next visual sweep.

### Neighborhood inference and county review checkpoint

`meuse-neighborhood-statistics` is implemented but not GPU-accepted. It uses all 155
measured sites, directed eight-nearest binary weights standardized by row, log zinc or
lead as focal variable, and log copper for bivariate comparisons. Current coverage is
local Geary (lesser-tail conditional permutations), joint-vector multivariate Geary
(three metals, equal-tail, no correction), separate descriptive LOSH/analytic Moran via
`addLocalVariabilityRecipe`, explicit fixed-residual LOSH permutations, local Lee L
(two-sided paired-row permutations), directed-edge Spatial Pearson and Gamma. Univariate
local p-values use BY correction; P=999 and seed31 are fixed. Gamma's three operations and
LOSH powers1/2 are implemented sensitivity options, still pending production numerical review.

The [GeoDa multivariate workbook](https://geodacenter.github.io/workbook/6c_local_multi/lab6c.html)
supports whole-vector dissimilarity and the distinction between multivariate and separate
univariate inference. The [LOSH documentation](https://sergerey.org/esda/dev/user-guide/LOSH.html)
describes a different chi-square inferential convention; the scene's residual-label null
must not be presented as that procedure. The separate normalized residual field is held
fixed while labels move; local means are not re-estimated inside each permutation.

Independent descriptive reference:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/reference-meuse-neighborhood.py /tmp/meuse-source.json dev-docs/research/showcase-overhaul-2026-10-25/meuse-neighborhood-reference.json
```

The reference stores all row values, 1,240 directed edges, both LOSH powers and all Gamma
operations. For zinc: mean Geary .93374698, mean Lee L .37139038, mean squared-residual LOSH
.88404929, edge Pearson .44493710, squared-difference Gamma .48344774. For lead these are
.94580770, .37049160, .90506748, .43076875 and .41737565. This reference does not certify
permutation streams. Planner requested explicit BY labels, status counts, comparison of
the two LOSH statistic outputs, stale-readout clearing and descriptive summaries before
GPU acceptance.

The county repair CLI completed scoped formatting, dataset integrity, story/cartography,
isolated strict types, pinned-source rebuild and all-six-region initial connectivity checks;
details are in `county-lane.md`. Parent GPU acceptance remains pending. Review found and
requested two more fixes: 24 golden-section iterations cannot attain a 1e-5 bracket width
from an initial 1.998 interval, so lag-ML search must use 40 iterations; a map helper's
runtime-NaN function referenced `index` outside its scope, so it needs a scoped runtime
variable. The same single CLI lane was interrupted and restarted with those repairs and
a distinct population Dorling story brief (current exec session29619). No concurrent
second CLI lane was introduced. Existing panel/regionalization files are to be frozen
after repairs, before GPU review.

Independent county coefficients are now recorded in `county-panel-reference.json`:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/reference-county-panel.py examples/deck/spatial-analysis-showcase/public/data/ncovr-new-jersey dev-docs/research/showcase-overhaul-2026-10-25/county-panel-reference.json
```

This uses NumPy least-squares instrumental projection and SciPy scalar optimization with
independent exact dense determinants. Pooled 2SLS slopes/rho are 1.067408, -.046086, .551415;
unit 2SLS .655416, 2.034617, .915038; unit lag ML .908544, 2.931871, .792967;
unit error ML .553099, 2.088308, .844751. Cross-sectional 1990 lag slopes/rho are
1.300437, .295172, .095745; error slopes/lambda 1.296656, .336237, .201921.
The production error/cross-sectional models use Q64 trace series, so numerical comparison
must respect truncation and convergence status. Random-effects ML, Hausman covariance,
standard errors and regionalization optimality are not certified by this reference.

Two further Meuse model-form labs are source-ready, pending production numerical acceptance: measured binary lime
with global probit versus local logistic (153 complete cases after two missing organic
matter rows), and aligned log-metal SUR/MGWR/local collinearity using measured covariates.
Different links and spatial supports must not be conflated, authored bandwidths must not
be called optimal, and no causal interpretation is intended. Subsequent priority is a
coherent NCOVR cross/space-time variogram inquiry. Population support/exposure/dasymetric
work remains contingent on defensible ancillary support; existing central-city buildings
do not establish resident locations.

### Wave 10 county acceptance and route correction

`check-county-models.mjs http://127.0.0.1:5313/ /tmp/luma-showcase-overhaul-20261025/county-models-numeric`
passed against frozen production wave 10. The report has empty page/console and GPU-error
arrays. Seven estimators returned valid solver and convergence status: pooled and unit
2SLS, unit and random-effects lag ML, unit error ML, and 1990 cross-sectional lag/error ML.
Six models' displayed coefficients agree with the independent reference within .015;
1990 lag direct/indirect/total impacts agree within .02. Random-effects results were checked
for valid status only. Its Hausman comparison correctly reports an invalid covariance
difference and suppresses interpretation; neither that covariance nor the random-effects
coefficient values have an independent oracle here. All three observed period maps contain
21 finite county units.

Twelve regionalization configurations passed feasibility checks: seeds 17/31 for AZP,
Ward sequential, Ward mutual, and Max-P floors 250,000/500,000/1,000,000. Every configuration
assigns all 21 units, returns finite objectives and no invalid status; independent graph
traversal of the returned labels finds one connected component per region. Fixed-count
methods produce six regions. Max-P returns 15/11/6 regions at seed17 and 15/10/6 at seed31.
All returned regions meet their population floor. These checks certify feasibility of the
GPU-produced partitions, not global optimality or an independent reconstruction of the
heuristic. The two saved county screenshots were inspected; New Jersey occupies too little
of the map, so a camera-only enlargement and question-led titles are queued. Accepted
calculations remain frozen.

The neighborhood verifier stopped before computation: source metadata declared
`meuse-neighborhood-statistics` while its filename was `meuse-neighborhood.scene.ts`.
The registry routes by filename, so the metadata link returned “Story not found”. This was
a source routing defect, not a stale build or a numerical failure. The implementer renamed
the metadata to `meuse-neighborhood-statistics.scene.ts`; the story checker and scoped Biome
pass. The verifier remains pending the next production build. Its failed report and screenshot
are retained in `/tmp/luma-showcase-overhaul-20261025/meuse-neighborhood-numeric/`.

Conditional-simulation presentation now retains a geographic overview in the first step and
fits the actual 128-query bounds in draw/variance/moment steps. The question-led title is
“How much can plausible soil maps differ?” and the control says “Reproducible random seed”.
These are presentation changes; the accepted conditioning and simulation are unchanged.

### Next county inquiry: distance and time support

The same single Luna CLI lane is implementing `county-cross-variogram` on the actual 21
NCOVR New Jersey counties at three decennial observations. This is planned, not yet accepted.
`GPUCrossVariogram` will compare single-period unemployment/poverty differences;
`GPUSpaceTimeVariogram` will use all 63 county-period records and actual 0/10/20-year lags.
Six spatial bins with 100/200/300 km cutoffs and three temporal bins expose actual GPU
semivariances, pair counts and empty-bin statuses. Univariate unemployment and signed
unemployment/poverty cross-semivariance are distinct options. Cross-semivariance is not
correlation, and negative values are retained. Repeated county anchors at different dates
produce legitimate zero-distance temporal pairs. Authored vertex-mean locations approximate
areal support; no station-level or annual trajectory interpretation is justified.

The [NCOVR source](https://geodacenter.github.io/data-and-lab/ncovr/) documents the historical
variables and census periods; preceding-year family-poverty measurements remain explicitly
aligned with their decennial observation. The [gstat space-time variogram documentation](https://r-spatial.github.io/gstat/reference/variogramST.html)
distinguishes temporal lags, spatial intervals and pair counts. Three dates provide limited
temporal support, so this scene will not fit a dynamic process. A positive univariate variogram
model must not be fit naively to signed cross-semivariances; `GPUVariogramModelFit` is therefore
not included merely to add a direct class call.

### Reviewed source and finite closure queue (after wave 10)

The county cross/space-time inquiry is now implemented, source reviewed, and awaiting its
first production numerical check. The typed-array concatenation, immutable readback generation,
all-reader flushing, time-band link filtering and chart metadata were corrected during review.
`reference-county-pairs.py` independently projects the same authored source anchors through PROJ
and enumerates NumPy unordered pairs: 24 cases cover both measures, all periods and all three
cutoffs; a 300 km space-time case contains all 1,953 unordered pairs. The same script checks
12 Dorling relaxation configurations against the documented synchronous force scheme. This
is not a claim of an optimal cartogram. `check-county-pairs.mjs` and `check-county-dorling.mjs`
are prepared and pass Node syntax checks; no GPU acceptance is implied by that check.

`massachusetts-island-weights` now uses all 14 actual Massachusetts county features in the
bundled US counties source. Direct `GPUContiguityWeights` strict Queen is compared with
`GPUFuzzyContiguity` at 0/8/12/16/20 km polygon separation and `GPUIslandAttachment` with
reciprocal/outgoing nearest-centroid links. `GPUSpatialWeightsValidation` checks the actual
before/after CSR. Broad separation tolerances define a different neighborhood relation;
they do not repair shoreline geometry. Nearest-centroid links likewise express an authored
analytical relationship, not a ferry, road or administrative connection. The analysis uses
the showcase spherical Albers frame, not EPSG:5070. The all-ring geometry and source centroid
roles are distinct and explicit.

An independent GEOS/PROJ/NumPy reference finds 38 strict directed slots and one island,
Nantucket. The five fuzzy distances yield 38/48/52/62/64 directed slots and 1/1/1/0/0 islands.
Reciprocal/outgoing bridges yield 40/39 slots; Nantucket's nearest supplied centroid is Dukes,
51,250.78 m away. `reference-massachusetts-weights.py` produced the retained reference JSON;
`check-massachusetts-weights.mjs` is prepared for eight configurations and a rapid cached
A→B→A return. Its Node syntax check passes. Source review corrected bounded variant caching,
status gating, copied request generations, and clearing every changed edge/island readout.
Biome and story checks pass, including the final step's four primary readouts. GPU and visual
acceptance remain pending. Primary method documentation: [libpysal fuzzy contiguity](https://pysal.org/libpysal/stable/generated/libpysal.weights.fuzzy_contiguity.html)
and [island attachment](https://pysal.org/libpysal/stable/generated/libpysal.weights.attach_islands.html).

County regionalization now derives legend entries from the actual visible GPU labels. Fifteen
unique colors cover the maximum 15 returned regions in the accepted configurations; unsupported
labels are withheld. County outlines and tooltips retain geographic support. The larger New
Jersey camera and question-led title, and the conditional-simulation query-window camera,
remain presentation changes awaiting visual acceptance.

Two explicitly authorized disjoint Luna CLI lanes continue. The KDE companion uses actual
2023 Chicago observation records; it is source draft, not accepted. Review found a pending
A→B→A graph race and incomplete final graph destruction, both returned to the implementer.
The transferred `point-patterns` direct pointwise CSR envelope is also assigned to this lane;
it must preserve the already accepted shared-replicate global test. A rose chart in taxi trails
is not evidence that `GPURoseStatistic` executes; movement is being asked to own that precise
remaining directional/axial contributor boundary.

Population support is no longer contingent on inventing ancillary data. The actual ESA
WorldCover 2021 v200 N39W090 COG was read after the CLI sandbox's DNS failure was bypassed by
the planner's normal tool environment. The builder verified the pinned ETag, 74,618,789-byte
source and modification timestamp, then cropped the full Hyde Park analysis extent. The raw
432×578 crop SHA-256 is `04247940e005d2f7997907a57b31f3bf0b8f9d0f41039ff82294e49cdc65471a`.
The bundled 12 whole tracts contain 29,559 people from the actual CDC/ATSDR SVI 2022 ACS
2018–2022 counts. Native class indicators are constructed before area averaging. Built-up class
50 is an explicit allocation support assumption, not residential occupancy; tree-cover class
10 supports a disclosed environmental condition summary, not temperature, health risk or public
park access. The planner flagged exact ellipsoidal-versus-spherical frame agreement for review.
GPU alignment, redistribution, exposure and mass-ledger implementation is in progress; its
metadata remains `.wip`. The [ESA data-access page](https://esa-worldcover.org/en/data-access)
and [2021 product manual](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf)
document source categories, processing and CC BY 4.0 attribution. Global reported accuracy
must not be substituted for local classification validation.

Remaining finite behavioral work is assigned or explicitly reconciled: (1) existing
`rate-smoothing` will add actual observed/expected excess risk and pooled count/exposure rates,
including island policy only on real isolates; (2) measured Meuse groups will compare varying
OLS regimes and common-coefficient restrictions, with classical independence/homoskedastic
assumptions; (3) the actual county 2SLS structural design/residuals can support HC0/HC1 and a
proved-PSD HAC sensitivity, subject to source and numerical review; (4) exact versus approximate
Jenks requires a bounded real subset because exact mode caps rows at 1,024, and integer numeric
versus ordinal color interpretation is not exercised by the current float-valued class story.
No default call receives whole-option-union credit. Existing rainfall CPU `fitVariogramModel`
exposes spherical/exponential/Gaussian and Cressie/pairs/none weighting; the absent direct
`GPUVariogramModelFit` is an execution-placement distinction, not an absent model menu. Signed
cross-semivariances are deliberately not fit with a positive univariate covariance family.

The direct rainfall GPU-fitting backend is now assigned, not waived: compare its parameters,
status and weighted residuals against the existing CPU helper on identical finite empirical
bins, model families, search budget and weighting. `addInterpolationDiagnosticsRecipe` is also
assigned there because its exact leave-one-out residual/positive-variance standardized-residual
contract is distinct from the existing held-out-grid or nested-fold results. The display must
separate LOO diagnostics conditional on an all-data fitted variogram from leakage-resistant
spatial validation; neither automatically estimates deployment error.

`reference-county-iv-covariance.py` prepares an independent NumPy covariance oracle for the
existing pooled 63-row panel 2SLS model, with structural `[1,X,Wy]` and instruments `[1,X,WX]`.
HC0/HC1 standard errors (intercept, poverty, density, rho) are
[1.218536, .163286, .139167, .242563] and
[1.259165, .168730, .143807, .250651]. A full Gaussian space/time product kernel with
50/100 km spatial and ten-year temporal scales is analytically PSD; the rounded f32 kernels
also have positive minimum eigenvalues (1.87e-5 and 1.01e-8). Their standard errors are
[1.504752, .222350, .128778, .285520] and [1.482149, .215359, .105370, .273967].
This establishes a reference for a planned covariance sensitivity, not production acceptance,
valid instruments, causal identification or calibrated small-sample coverage. The
[spreg spatial-lag IV tutorial](https://pysal.org/spreg/notebooks/14_IV_estimation_spatial_lag.html)
and [GM_Lag contract](https://pysal.org/spreg/generated/spreg.GM_Lag.html) distinguish robust
covariance from point estimation and require a separate HAC kernel with unit diagonal.
The [OLS_Regimes contract](https://pysal.org/spreg/generated/spreg.OLS_Regimes.html) identifies
common versus varying coefficients; the planned measured-group comparison must state the
classical covariance and coefficient-stability assumptions explicitly.

### Wave 11 numerical acceptance and remaining repair

The serialized stable-production slot tested the following. Every accepted report has empty
console/page-error and uncaptured-GPU-error arrays; the first attempts and harness corrections
remain separately retained rather than overwritten.

| Story / executable checker | Accepted scenarios and evidence | Limits |
| --- | --- | --- |
| `check-meuse-model-form.mjs` | Four cases: probit coefficients/expected-Fisher SE ±.003 and log likelihood ±.015; all five local-logistic LOO scores ±.003 (750/1000 m unavailable), 2200 m winner and 153 valid local fits; common-design SUR coefficients/correlations ±.011, local condition maximum ±.11, MGWR compact→broad→compact extrema ±.002 | Local logistic candidate failure criteria differ between optimizers; no claim of exact iteration parity. In-sample fits and bandwidth selection do not establish independent predictive accuracy. |
| `check-county-dorling.mjs` | Twelve attraction/budget combinations; maximum overlap/displacement agree with the independent documented-scheme calculation within 3 m, iteration/termination flags match, 21 ordered source IDs are valid, area/radius mapping valid | Checks the specified relaxation scheme, not an optimal cartogram or geographic distances between displaced circles. |
| `check-county-pairs.mjs` | Twenty-four support/period/measure/cutoff combinations plus rapid cutoff return; every bin count/status matches NumPy exactly, each mean within .002 | Descriptive pair averages on authored areal anchors, not a fitted stochastic process. |

Reports are `/tmp/luma-showcase-overhaul-20261025/wave11-model-form-numeric-fixed-harness/report.json`,
`wave11-dorling-numeric-fixed-harness/report.json` and `wave11-county-pairs-numeric-fixed-harness/report.json`.
Initial model-form harness parsing treated an empty unavailable chart cell as numeric zero;
it now treats empty cells as NaN. Initial county control selectors omitted the accessible
“rebuild” badge from exact label matching; prefix selectors fixed this. Neither harness
correction changed production computations. Saved model-form, cartogram and pair-summary
screenshots were reviewed. New Jersey framing and cartogram circles are legible. Lime's
viridis default and blank failed-candidate labels were returned for presentation correction;
metal coefficients use an overly broad fixed legend and were returned for a tighter shared
zero-centered range. These requested presentation fixes do not change accepted calculations.

`meuse-neighborhood-statistics` failed before computation with eleven storage bindings in its
custom map-selection pass, exceeding the production adapter's ten-buffer limit. This was not
an algorithm numerical failure and is not accepted. The source selector is now split into
three passes with five, six and four storage bindings; numerical contributors are untouched.
The failed report remains `wave11-neighborhood-numeric/report.json`; the repaired source needs
a new production build and the full twelve-case numerical check. Massachusetts weights also
remains pending its first GPU check.

The rates extension now directly computes excess-risk ratios and both undefined/own-rate
spatial pooling policies on actual births and woman-year denominators. The returned graph,
not an invented row, identifies Queen isolates Nantucket and San Juan. Review corrected
EB-only tooltip quantities on the new maps, framed Massachusetts for visible island evidence,
and added copied readback generations, pending clears and visible failure status. This is
source-reviewed work pending production execution, not accepted numerical coverage.

### Final finite source freeze (after wave 12b sweep findings)

This section supersedes earlier “assigned”, “in progress”, and draft-metadata status descriptions
for the finite statistics queue. Runtime sources are frozen pending the next parent production
build. The source changes below have scoped formatting and story checks; they do **not** gain
numerical acceptance from source inspection. No GPU tests ran while the parent held the wave12b
all-story sweep. Root build, full lint, full tests, and website gates remain parent-owned.

| Reachable scene | Delivered behavior / options | Current acceptance |
| --- | --- | --- |
| `meuse-neighborhood-statistics` | Local Geary, joint-vector multivariate Geary permutation statistic/p, LOSH and residual diagnostics; repaired selector split uses ≤6 storage bindings per custom pass. Direct `GPUQuantileLISA` now exposes descriptive type-5 quantile membership and raw directed binary local joins for zinc/lead, 4/5/10 bands, lower/middle/upper band. Its legacy pseudo-p output is not displayed or credited as inference. | Prior neighborhood runtime failed on11 bindings; repaired12-case statistics check plus19 quantile cases pending. Independent quantile reference completed18 combinations. |
| `massachusetts-island-weights` | Fuzzy contiguity, actual island attachment and graph validation with real Massachusetts islands; actual added graph links are drawn. | Nine scenarios pending. Wave12b bridge was absent because a reused empty array prevented deck data refresh; publication/clear now replace the PathLayer data array. Numeric graph was not altered. |
| `rate-smoothing` | Observed/expected excess risk and neighbor-pooled rates, Queen/Rook and undefined/own-rate island policies. The empirical-Bayes standardized output already provides documented Assunção–Reis behavior, not a standalone direct call. | Five new cases pending. Independent CPU graph finds actual Nantucket and San Juan isolates under both contiguities. |
| `meuse-spatial-regimes` | Measured lime0/1 groups, pooled/common coefficients, intermediate restrictions, fully varying coefficients, classical covariance, Chow and Wald summaries. | Nine cases pending. Independent153-row/109+44-group reference completed. Wave12b lime group rendering used float interpretation with categorical colors; source now binds the measured uint32 group buffer and uint32 format. |
| `county-panel` | Existing spatial 2SLS design now supplies direct IV HC0/HC1/HAC covariance. HAC is a full Gaussian space-time product kernel at50/100km and10-year scale, with unit diagonal and a PSD reference check. | Existing seven model families retain prior acceptance; five covariance scenarios pending. No instrument validity, causal identification or small-sample coverage claim. |
| `county-population-classification` | Actual21-unit1990 population; exact and128-bin approximate Fisher–Jenks, five classes; GPU class assignment; numeric uint32 population versus ordinal uint32 class-ID color interpretation. Actual21 IDs and both packed color arrays are available in All readouts. | Five scenarios pending; a compact three-step comparison is intentional. No full-union credit for other class-break/color options. |
| `chicago-population-support` | Actual12 Hyde Park tracts, reversed source group rows aligned by tract ID within Cook County; uniform-area or actual WorldCover class50 fractional support; dasymetric64-cell totals; actual class10 tree-cover exposure in four quadrants at10/25/50% thresholds. | Source-ready metadata published last;11-case numerical verifier pending. Independent GEOS/NumPy reference completed. |
| `rainfall-interpolation` | Direct GPU variogram model-fit backend: spherical/exponential/Gaussian × Cressie/pairs/none weights × Matheron/Cressie–Hawkins bins, matching CPU search domain/budget; exact interpolation-diagnostics recipe with all-gauge LOO predictions, positive-variance standardized residuals, counts/bias/MAE/RMSE. | New backend/LOO scenarios pending. LOO is conditional on a variogram fitted on all gauges, distinct from held-out spatial validation. |
| `nature-kernel-density` | Direct KDE methods, kernels/edge corrections/weights/normalizations/bandwidth choices with actual Chicago observations; observed effort limitations explicit. |24-case verifier pending. Source fixes showSurface=true on later steps and preserves probability-scale precision. |
| `point-patterns` | Direct pointwise CSR envelope complements existing simultaneous/global test, with explicit pointwise versus global interpretation and fail-closed support mismatch/jitter behavior. | Six pointwise contract/reproducibility cases and existing seven global cases pending regression run. Independent Philox replicate identity is not claimed. |
| `hot-spots` | Existing cell/tract analysis retains scientific projection tolerance. Cell projection and neighbor index now share the actual four-spacing support domain. | Wave12b failed with arbitrary±2° projection bounds; CPU projection planning passes all Quadbin12–17 and H3 5–9 at unchanged0.01m tolerance. GPU regression pending. |

The multivariate local-Geary observed statistic is produced inside
`GPUMultivariateLocalGearyPermutationTest`, not by a standalone direct
`GPUMultivariateLocalGeary` constructor. The reachable path is scene factory
`createMeuseNeighborhood` → graph.add(permutation contributor) → compile → encode. This is
an explicitly composed observed-statistic behavior, not import-only credit.

#### Final review corrections

Rainfall robust sector bins cannot be averaged as semivariances: the revised reduction recovers
the Cressie–Hawkins root means, pools by pair count, and reapplies the finite-count correction
once to the total. CPU and GPU fitters receive the same Float32 input arrays. The exact input
bins and raw CPU/GPU parameters are visible. LOO waits for a current real fit, idle variogram
readback and no pair overflow; fallback parameters cannot masquerade as fitted diagnostics.
A transient graph-output handle previously reached SummaryReader and caused wave12b's native
copyBufferToBuffer type error; current source copies status into owned luma Buffers before
readback. Execution acceptance remains pending.

Population support is now an explicit option independent of map display. Visiting tree cover
or the built-up input map cannot silently change allocation weights. The current support is
visible during pending and completed states. Fine64² built-up fractions are an actual displayed
input;64 target counts and four quadrant exposure values come from GPU outputs. Uniform means
all valid analysis area, including water—not “land area”. Complete valid support is checked;
zero modeled quadrant denominators retain NaN coverage/share rather than inventing a zero
coverage rate. WorldCover built-up is not housing, and tree cover is not heat/health/access.
The precise source/data/recipe contract is in [population-support-lane.md](population-support-lane.md).

The independent population oracle uses float32 projected rings with GEOS intersections,
NumPy support normalization and target aggregation. It finds1347 positive intersection tuples,
conserves all29,559 residents, and gives peak500m counts2606.710778 (uniform) and3087.391663
(built-up). Every returned cell,12 aligned Black/other source groups, mass ledger and six
support/threshold quadrant summaries will be checked. Display rounding permits0.75 resident
error per cell/quadrant; conservation is checked separately to0.05 resident.

The IV covariance reference now reproduces the documented interior polygon-anchor search
independently with GEOS, replacing the earlier vertex-mean anchor approximation. **Corrected
HAC reference values supersede the earlier paragraph**: Gaussian50km SE =
[1.502319818, .220175229, .130130318, .284740543]; Gaussian100km SE =
[1.496983099, .216240451, .107388722, .276329100]. Rounded-kernel minimum eigenvalues are
4.8075e-5 and1.07739e-7. HC0/HC1 values are unchanged. The oracle calls no production model
or fitting helper.

Independent quantile membership/join references use centered float32 RD coordinates,
eight nearest neighbors with source-index tie order, NumPy Hazen/type5 thresholds and
float32 comparisons. Eighteen combinations are checked, including tied lead values and
half-open internal intervals. For zinc quartile-highest,39 members yield143 summed directed
joins, maximum7; for lead quartile-highest,40 members yield155 joins, maximum8.
These are descriptive adjacency counts, not p-values.

Wave12b visual review accepts the revised conditional-simulation framing (query support now
fills useful map area with narrow-window caveat) and county-regionalization dynamic11-label
legend. It also found the lime probability map still used viridis under a cividis legend;
source now uses cividis. Model numerical calculations are unchanged.

#### Exact prepared verification artifacts

All paths below are relative to this research directory. Browser scripts were syntax-checked
only during the parent's sweep; do not read this list as GPU pass results.

- `check-meuse-neighborhood.mjs` (12 cases), `check-massachusetts-weights.mjs` (9 cases):
  pre-existing prepared verifiers; filenames/commands are checked before the slot.
- `check-rate-recipes.mjs` (5), `reference-rate-recipes.py` / `rate-recipes-reference.json`:
  real contiguity pooled-rate/island/excess-risk reference.
- `check-county-iv-covariance.mjs` (5), `reference-county-iv-covariance.py`:
  SE tolerance.006, full covariance tolerance.012, exact configuration/status.
- `check-meuse-regimes.mjs` (9), `reference-meuse-regimes.py`: coefficient/SE.003,
  Chow F.006, exact degrees of freedom, Wald p.003 and pooled/restricted identity.
- `check-meuse-quantile.mjs` (19), `reference-meuse-quantile.py`:
  exact integer membership/join counts, boundary tolerance.001.
- `check-nature-kernel-density.mjs` (24), `reference-nature-kernel-density.py`:
  documented exact-path tolerances and separate2% binned approximation tolerance.
- `check-pointwise-csr.mjs` (6): pointwise curve/support/seed contract; preserve and rerun
  `scripts/check-point-pattern-global-test.mjs` separately.
- `check-chicago-population-support.mjs` (11), `reference-chicago-population-support.py`
  / `chicago-population-support-reference.json`: all target cells, source alignment and
  modeled tree-cover exposure; CPU reference rerun successfully after adding quadrants.
- Rainfall and county-classification independent verifier artifacts are being finalized by
  the two existing CLI lanes; these lanes are prohibited from runtime source changes or GPU runs.

`statistics-browser-check.mjs` now captures served `build-evidence.json` before launch and
verifies the same `bundleSha256` afterward, saving both evidence and verification next to
each report. Any bundle change invalidates acceptance. Reports retain actual readouts, failure
screenshots, console/page errors and uncaptured GPU errors.

No further substantive statistics functionality is unassigned in this finite closure queue.
Source readiness, production rendering and independent numerical acceptance remain separate
states. Prior accepted models/conditional simulation/nested validation/regionalization/
Dorling/cross-space-time pair results retain only their previously documented scope.

Final verifier review is complete. The county classification CPU oracle enumerated all 4,845
contiguous five-class partitions: exact SSE is 27,607,880,702, interior edges are
224327 / 395066 / 493819 / 671780, frequencies are [6,4,4,4,3]. The independent approximate
reference reaches that same partition on this fixture; this is not a general equality guarantee.
The five-case checker also verifies each actual packed numeric/ordinal color and source FIPS.

The rainfall verifier's independent SciPy NNLS/profile optimizer was exercised before browser
execution on actual-source CPU bins (6,135 gauges, 2,796,525 pairs within 300 km). All 18
model/weight/robust combinations are finite; comparing the production CPU fitting helper with
this independent optimizer gives maximum relative objective difference 2.035e-9. Evidence is
`/tmp/luma-showcase-overhaul-20261025/rainfall-cpu-oracle-check.json`. This validates the reference
and CPU model contract, not GPU binning or fitting. Many profiles hit the upper range boundary,
which is explicitly shown as a candidate rather than an identified range.

Review also identified existing kriging numerical floors: nugget/sill ≥ 1e-4 × sample variance,
range ≥ 1,000 m. The rainfall scene now discloses the exact float32 nugget/sill/range extracted
from the same array written to its GPU parameter buffer and labels these floors. No computation
changed. The LOO oracle conditions on that actual applied tuple; independent fitting is checked
separately. The verifier accounts for seven-significant-digit table rounding and the existing
three-significant-digit summary formatter, and normalizes weighted SSE by total weight before
using it as an RMS scale. Browser numerical acceptance remains pending.

A durable CPU-only hot-spot domain regression now reads the actual Chicago manifest and
reconstructs the shared four-cell domain for all 11 allowed Quadbin/H3 resolutions:
`node dev-docs/research/showcase-overhaul-2026-10-25/check-hotspot-projection-domain.mjs`.
It passed at unchanged 0.01 m tolerance, maximum sampled error 0.008044122 m. The full
bounds/spacing/patch ledger is `hotspot-projection-reference.json`. This is projection-plan
validation only; it does not request a GPU adapter or prove that every downstream scene
pipeline has executed.

Process audit correction: the population final-review CLI unexpectedly invoked `yarn lint fix`
twice despite its no-root-lint instruction. The first invocation checked 8,240 files and
formatted one unspecified file; the second changed none. The planner reported this to the
parent, preserved the shared dirty tree, and did not claim that formatting mutation was scoped
or run any reset. The lane has exited. Parent review of the final shared formatting diff remains
required. This exception does not establish that the eventual final root gates have passed.

The final explicit graph-cast audit removed the two remaining `getCompiledGraphs` casts and
unused imports in Meuse natural-neighbor and conditional-simulation compute files. Both now
return `[compiled]` with inferred types. Scoped formatting and the full showcase package
TypeScript check passed (exit 0) after this type-only cleanup. Root separately reports its full
`yarn lint fix` gate checked 8,242 files with no fixes and a valid lockfile; module build is in
progress, so no final build/test completion is inferred here.

### Coordinated readout-contract corrections after freeze

The parent explicitly authorized two CPU-only source corrections while movement held the GPU
slot; the served production bundle was not changed. Atlas screenshot review established that
Meuse validation's prediction view retained a variance-only interpretation. The readout now
follows the actual selected quantity: log-metal prediction, conditional point variance in
squared log-concentration units, unitless RBF pivot diagnostic, or OOF sample residuals. It
retains support counts only after the existing generation gate, clears them for pending model
changes, and republishes on display-only changes without requiring another GPU dispatch.
Status1 (no usable neighborhood) and status2 (unreliable/rank-deficient solve) are named
separately; no numerical calculation changed.

Rainfall now always reports actual finite-gauge LOO valid/standardized counts. The separate
GPUKriging counters are explicitly described as output-raster invalid/unsupported and uncertain
cell counts, as required by the contributor's documented contract; the LOO summary invalid
count is named separately. Unsupported viewport cells no longer replace the gauge diagnostic
line. This repairs the previously reported source-contract finding; GPU acceptance of the
correction remains pending the rebuilt bundle.

Scoped Biome and interpolation story checks pass. Typecheck at this handoff reports only the
concurrent missing evaluateProjectionPlan symbol in raster/grid-alignment-projection.ts:88.
`check-statistics-readout-contracts.mjs` is syntax-checked and adds six Meuse view transitions
plus one rainfall gauge/raster-status contract check. The existing Meuse numerical verifier's
support parser now sums the separately labeled unsupported and unreliable counts; its numeric
expectations remain unchanged. These are prepared checks, not executed GPU results.

### Wave 14 pinned production acceptance

The serialized statistics GPU slot completed against `http://127.0.0.1:5313/`, bundle SHA-256
`872af62d507f3cae0bed2749a0b9f8c7736739cf2d99cab71ba8658bf5e817ae`. Every accepted report
records identical before/after bundle hashes, zero console/page errors and zero uncaptured GPU
errors. `wave14-statistics-acceptance.json` is the durable index of commands, raw report paths,
report SHA-256 hashes, independent reference methods, tolerances and limitations.

| Suite | Accepted cases | Scope |
| --- | ---: | --- |
| Meuse neighborhood statistics | 12 | Local Geary, LOSH variants, local Lee L, spatial Pearson and Gamma variants |
| Massachusetts island weights | 9 | Strict/fuzzy topology, bridge directionality, graph diagnostics and rapid return |
| Meuse restricted/spatial regimes | 9 | Four independent numerical profiles, covariance, Chow/Wald tests and mapped transitions |
| Rate recipes | 5 | Excess risk, spatial pooled rates and island-policy support |
| Meuse quantile joins | 19 | Zinc/lead, 4/5/10 bins, lowest/middle/highest bands and repeat |
| Interpolation display contracts | 7 | Six Meuse quantity transitions and rainfall gauge-versus-raster status |
| Nature kernel density | 24 | Kernels, edge correction, weighting, normalization, bandwidth rules, exact/auto/binned |
| County classification | 5 | Exact/approximate Jenks, numeric/ordinal integer colors, actual 21-row IDs and packed RGBA |
| Direct pointwise CSR | 6 | Fixed-seed repeat, actual sampled radius, exploratory disclosure, mismatch/jitter clearing |
| Rainfall fitting | 18 | Three models × three weighting rules × classical/robust empirical bins |
| Existing global point-pattern test | 7 | Three global statistics, alpha choices and support/jitter calibration gates |
| Chicago population support | 11 | **Numerical only**: alignment, allocation, exposure, support/display independence |
| County IV covariance | 5 | HC0, HC1, Gaussian space-time HAC at 50/100 km and return |
| Hot-spots initialization | 1 | Production smoke after bounded projection-domain correction |

This totals **138 bounded accepted cases**, plus the rainfall checks of all 6,135 source rows
and 24 independently solved kriging probes. Counts distinguish numerical/contract acceptance
from broad visual story walkthroughs and do not imply every API option is tested.

Rainfall's largest GPU objective deviation from the independent SciPy reference was
`7.77694e-7` relative; the CPU maximum was `1.55538e-7`. Fourteen of eighteen fits correctly
reported boundary/iteration-budget status 3 as a candidate rather than an identified optimum.
Across 24 NumPy ordinary-kriging probes, maximum absolute prediction difference was
`0.000062979 mm` and variance difference `0.000414769 mm²`. Of 6,135 source rows, 6,132 had
valid predictions and 6,128 had positive variance suitable for standardized residuals.
Recomputed bias was `−0.0299072821 mm`, MAE `9.36750602 mm`, RMSE `18.6608078 mm`.
All-row residual and standardized-residual identities passed the seven-significant-digit plus
float32 rounding bounds. These are conditional LOO results under the full-data fitted
variogram; they are not leakage-resistant variogram-refit cross-validation.

Visual review confirmed the Massachusetts centroid bridge is visible; the KDE surface fills
the study window; and the Meuse measured-lime step displays the intended blue/red categories
with matching legend after resetting modified controls. Numerical regime counts are 109/44.

Several first verifier attempts failed on UI/parser assumptions, and their raw reports are
retained alongside the corrected runs. Corrections covered a Local Lee text prefix, the
`intercept-distance` option value, step/progress duplicate button names, a missing literal
`FIPS` readiness marker, the closed Expert controls group, nonbreaking spaces and duplicate
chart instances. The pointwise table formats integer metres while its selected-radius text
uses one decimal; its display equality bound is now the derived `0.5 + 0.05 = 0.55 m`, replacing
an incorrect `0.11 m` parser assumption. No estimator oracle or computational tolerance was
relaxed. The population chart table intentionally summarizes beyond 50 rows, so the numerical
verifier reads all 64 actual SVG bar values from one chart instance.

One **real runtime display defect remains unaccepted on wave 14**: changing population display
alone repainted the map but retained the previously published chart until a GPU readback.
`wave14-population-chart-ready/report.json` preserves that failure. A numerical-only companion
selects the display before forcing support recomputation; its 11 passing cases therefore do
not accept the failed display-only transition. The parent authorized a bounded source fix:
retain copied, generation-valid target totals and exposure arrays; publish the matching chart
on display-only changes; clear cached arrays/charts on support or threshold recomputation.
The implementation was reviewed, scoped Biome and statistics story checks pass, and no math
changed. The main population checker now includes source → allocation → source transitions
(14 cases total) for the next production bundle. **Wave 15 execution remains pending.**

All statistics browser processes were closed and the GPU slot released to the parent/earth
queue. No root build, install, lint, or production-bundle mutation was run during this batch.

### Wave 15 population display correction accepted

The full population suite now passes **14/14 cases** on frozen wave15 bundle SHA-256
`5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65` at the same production
URL. Before/after hashes match; console/page and uncaptured GPU error lists are empty.
`wave15-population-acceptance.json` records the command, raw and partial report paths/hashes,
build evidence, reference tolerances, limitations and screenshots.

The three display-only states were captured and visually reviewed: 12 observed whole-tract
bars → 64 modeled target-cell bars → the identical original 12 observed bars. The final
source values exactly match the initial source values. Map quantity and legend change with
the corresponding display. Both allocation supports and all three tree-cover thresholds also
pass the unchanged independent GEOS/NumPy checks, including exact source ID alignment, mass
conservation and population-weighted quadrant summaries. The cache correction does not
change calculations or reference tolerances.

This closes the real population chart defect recorded under wave14. It supersedes the
pending status in that historical acceptance index; other suites retain their actual wave14
execution provenance. No statistics runtime acceptance check remains pending at this handoff.
All browser processes were closed and the GPU slot explicitly released immediately after
this suite. No runtime source changes or production rebuild occurred during the run.
