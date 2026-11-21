# Regression serial-loop inventory

`single-invocation-data-loop` is a shader heuristic, not a ranking of computational cost.
One invocation can solve a small coefficient system, merge many row tiles, or scan an entire
dataset. These require different reviews. The tables below record the axes, bounds, and execution
routes of the regression entries reviewed for SSR-13. Paths are relative to
`modules/analysis/src/gpu-spatial-analysis/spatial-regression/` unless stated otherwise.

Bounds describe GPU construction, not CPU twin limits or measured execution speed. A helper
inherits its caller's validation unless an independent helper check is stated. An enforced row
capacity bounds an otherwise data-dependent loop; it does not make that loop coefficient work.

## Previously reported regression entries

The 2026-11-16 baseline at `bdc715ea` contained 21 construction sites across 15 spatial-regression
files. Several sites have multiple loop axes. The restricted-regime `nested-models` and `finish`
sites both carry reviewed call-site `contract-allow` comments; their old baseline entries have
been removed.

| File and kernel | Serial axis and bound | Route and classification |
| --- | --- | --- |
| `gpu-geographically-temporally-weighted-regression.ts`, former serial `summary` | Replaced: dense and indexed fits both call `createGTWRIndexedSummaryNodes`, which uses parallel fixed-tree reductions over fit-time centered residuals, hat diagonals, row counts and centered response squares. | Shared GTWR summary route; final publication reads reduced values, with no serial observation scan. The MGTWR component-bandwidth summary below is separate. |
| `gpu-geographically-weighted-regression-nonstationarity-test.ts`, `anchor` | Final copy: at most eight coefficients including the intercept, bounded by `GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT = 7`. | Fixed coefficient publication from a preselected row; row selection uses 256-row tiles followed by a minimum reduction. |
| `gpu-geographically-weighted-regression.ts`, `ladder-init` | Parameter words: header 4 plus ladder capacity at most 32, hence at most 36. | Fixed parameter initialization; no row scan. |
| Same file, `refine-ladder` | Each prior-history scan visits at most `4 × 32 = 128` candidate rows at refinement four; at most 32 next ladder values. Bounds are `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT = 4` and `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH = 32`. | Bounded bandwidth-candidate work, not observation work; the completed table scanned by `choose` has a separate bound. |
| Same file, `choose` | Completed candidate table: at most `(1 + 4) × 32 = 160` rows for finite ladders/refinements, or 65 rows for a 64-step golden-section search (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS = 64`). | Bounded bandwidth selection. |
| Same file, `summarize` | Scalar merge runs only through `MAXIMUM_HISTORICAL_REDUCTION_ROW_COUNT = 1,048,576` input rows. With `TILE_ROWS = 256`, it reads at most `ceil(1,048,576 / 256) = 4,096` partials. | Data-dependent tile merge on the historical reduction route; larger admitted inputs use `createGWRVarianceReductionNodes`, whose final publisher reads one reduced total of squares to publish R². Both routes dispatch one invocation with the same `-summarize` ID suffix and `summarize` variant; only the historical kernel scans tile partials. Local row fitting is separate. |
| `gpu-multiscale-geographically-temporally-weighted-regression.ts`, `convergence-score` | Rows times design columns: at most 32,768 times 12; at most 32 sweeps. | MGTWR exhaustive reference route; data-sized convergence reduction. |
| `gpu-multiscale-geographically-weighted-regression.ts`, `convergence-decide` | At most 256 convergence workgroups, with three partial words each; `CONVERGENCE_MAXIMUM_GROUP_COUNT` fixes the cap. | Bounded merge after parallel contribution checks; reviewed call-site allowance independent of the baseline. |
| `gpu-probit-regression.ts`, `pinkse-kp-pinkse-slade` | Six diagnostic result slots; reads already reduced state scalars. | Fixed output initialization and scalar diagnostics, independent of row count. |
| `gpu-restricted-regime-regression.ts`, `nested-models` | Restricted and unrestricted coefficient axes at most 16; regime axis at most 8; covariance axes at most 16 by 16. Workspace initialization depends only on those axes. Constructor enforces `(predictorCount + 1) * regimeCount <= 16`. | Small dense systems after tiled regime moments. Reviewed allowance; no serial observation scan. |
| Same file, `finish` | At most 16 coefficient outputs, 256 covariance outputs, and 3 Chow outputs. | Fixed coefficient publication; reviewed allowance. |
| `gpu-spatial-panel-error-maximum-likelihood.ts`, `absorbed-intercept` | `(predictorCount + 2) × 4` parallel invocations, using `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE = 4`. The first four clear the intercept fields to NaN; on the period-effects route, remaining invocations map profiled table entries into the remaining slots. Invocation zero also clears two summary fields to NaN. | Unit-/period-effects panel error publication; no serial observation loop. |
| `gpu-spatial-panel-lag-maximum-likelihood.ts`, `unit-ml` / `random-ml` construction site | Up to 32 entities and 32 periods (1,024 observations) on the reference route. Dense entity determinant/inversion loops, observation scans, coefficient systems, at most 128 effect iterations and 64 search steps. | Reference route only. Mixed entity-, row-, and coefficient-axis work; distinct from tiled pivot selection below. |
| Same file, `publish` | Coefficient and covariance layout sizes, from at most 7 predictors and the spatial/intercept slots; six summary fields. | Reference-route publication; fixed coefficient work. |
| Same file, `hausman` | At most 7 slope contrasts and their square covariance system; three test outputs. | Optional reference-route unit/random contrast; bounded coefficient work. |
| `gpu-spatial-panel-regression.ts`, `unit-finish` | Coefficient table rows derived from `predictorCount + 2`, validated by `getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount` (8 or 5 predictors by instrument order); 16 summary slots. | Unit-effects panel 2SLS publication; fixed coefficient work. |
| `gtwr-bandwidth-search.ts`, MGTWR component `summary` | Up to 32,768 rows, plus 12 summary fields; inherited from MGTWR's exhaustive row cap. | Component bandwidth objective; serial row reduction. |
| `gwr-golden-section-nodes.ts`, `search-init` with GWR layout | Candidate rows at most 65, seven fields per candidate. Bounds merge has at most 4,096 partials: the historical reducer uses 256-row tiles through 1,048,576 rows; above that named strategy threshold, parallel reduction feeds one bounds record. | Fixed search-table initialization and capacity-bounded partial merge; reviewed call-site allowance independent of the baseline. |
| Same construction site with MGWR layout (`mgwr-bandwidth-selection.ts`) | At most 66 candidate rows with three fields each: 198 words at 64 search steps. Two interior rows plus the upper-endpoint probe require the extra row. | Fixed search-table initialization; caller supplies distance bounds, so no bounds-tile merge. The allocation boundary is pinned by `stats-gwr-independent-review.node.spec.ts`. |
| `gwr-grid-index.ts`, `grid-bounds` | Exactly four components after upstream reductions. | Fixed bounds publication; row extent reductions occur in earlier nodes. |
| `multiscale-inference.ts`, `full-inference-summary` | Focus and response rows at most 128 each, with at most 25 design columns; construction additionally enforces `rows^3 * columns * maximumIterations <= 268435456`. | Optional full MGWR inference; bounded but data-sized quadratic summary, separate from conditional scalar errors. |
| `panel-diagnostics.ts`, `pooled-panel-lm-sums` / `pooled-panel-lm-tests` | Tiled observation work with pairwise merges produces five reduced sums. The scalar tests kernel reads those sums and fit/projection status, then publishes six diagnostic values and status. Panel-error input is bounded by `GPU_PANEL_ERROR_MAXIMUM_ROW_COUNT = 16,777,216` rows. | Optional pooled panel diagnostics; tiled row reduction followed by scalar publication with no observation scan. GPU accuracy qualification remains pending. |

## Forwarding wrappers exposed by the checker

The checker resolves a local function/arrow that forwards its own `count` and `body` parameters
to `createWGSLKernelNode`, including aliases and generic calls. Every call site receives its own
parameter bindings, default arguments, lexical captures and execution guards. A known count of
one is checked for reachable WGSL loops. Parallel and unknown counts do not become scalar merely
because another caller supplies one. Duplicate declaration and shader-analysis-limit checks use
the same independently resolved shader fragments.

An allowance immediately before a wrapper call applies only to that call. Mutated parameters and
factory aliases are not substituted. Nested wrapper chains, arbitrary function evaluation, spread
arguments and destructured/rest parameters are outside this heuristic; this is not a complete
interprocedural TypeScript analysis. In particular, unresolved shader-generating function calls
can still conceal loops. CPU twins and operation forms are unchanged: this check audits GPU graph
construction, and does not reinterpret their iteration or output contracts.

| File and newly visible sites | Serial axis and bound | Route and classification |
| --- | --- | --- |
| `gpu-instrumental-variables-covariance.ts`, `projection`, `finish` | Projection: up to 15 centered coefficients and 16 instruments (`INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_COEFFICIENT_COUNT = 15`, `INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_INSTRUMENT_COUNT = 16`). Finish: up to 16 coefficients including the intercept, hence up to 256 entries in the full row-major covariance. | Small dense systems and covariance publication after tiled row moments; no row scan in these scalar stages. |
| `gpu-spatial-error-gm.ts`, `weights-norm-merge`, `finish` | Constructor enforces `tileCount <= REGRESSION_TILE_MAXIMUM_TILE_COUNT = 2^24`, including explicit widths and per-chunk tile counts. Above 4096, consecutive groups of at most 4096 partials reduce before the final group merge. Finish uses at most 15 predictors. | Hierarchical tile merge and fixed coefficient publication; device and command budgets also apply. |
| `gpu-spatial-two-stage-least-squares.ts`, `weights-norm-merge`, `solve`, `refine` | Constructor enforces `ceil(rows / tileRowCount) <= REGRESSION_TILE_MAXIMUM_TILE_COUNT = 2^24`. Above 4096, sum/maximum channels reduce within consecutive groups, then over groups. Solve/refinement use at most 16 centered instruments. | Hierarchical tile merge and bounded coefficient systems; device and command budgets also apply. |
| `gpu-spatial-weights-traces.ts`, `norm-merge` | Constructor enforces `tileCount <= REGRESSION_TILE_MAXIMUM_TILE_COUNT = 16,777,216` tiles total, including per-chunk counts. Each merge invocation reads at most `REGRESSION_TILE_MERGE_GROUP_SIZE = 4,096` tile/group partials. Standalone default width is `max(32, ceil(rows / 1024))` rows per tile. | Hierarchical norm/sum reduction on exact and stochastic routes; independent device and command budgets also apply. Nested likelihood trace planning is described below. |
| `ordinary-sur-nodes.ts`, `merge-means`, `solve-ols`, `solve-error-covariance`, `solve-gls` | Row-tile count from `getRegressionTileLayout`, with at most `2^24` rows and default at most 4,096 tiles. Systems have at most 4 equations and 16 total coefficients. | Mixed data-dependent tile merges and small dense equation/coefficient solves; row residual moments run in earlier tiled nodes. |
| `panel-lag-tiled-nodes.ts`, `initialize`, `solve-moments`, `profile-rho`, `information-solve`, `pivot-select`, `publish` | Model/output initialization depends on at most 7 predictors. Moment/information stages merge `ceil(rows / tileRowCount)` partials, default at most 4,096 tiles. Profile uses at most 64 search steps and scans at most 1,024 supplied eigenvalues per objective. Pivot selection scans remaining entities, at most 1,024, once per entity pivot. | `fitMode: 'tiled'` only: at most `2^24` rows. Pivot selection is entity-sized serial work, O(entities²) over all pivots; elimination remains parallel. It is not a coefficient-only loop and not the reference route's repeated dense determinant. The helper-generated residual-pair merge is another data-dependent tile merge; its generator call is opaque to this heuristic. |

Nested likelihood traces start at `max(32, ceil(rows / 1024))` rows per tile, independently of
the main regression tile width. They count tiles separately within each nonempty weights chunk.
When there are at most 4,096 nonempty chunks, the planner doubles the width (up to the row count)
until at most 4,096 tiles remain, aiming for a single merge level. With more nonempty chunks,
it retains the starting width and uses hierarchical merging. The 4,096-partial limit applies per
merge invocation; the total tile limit is 16,777,216. Device binding limits and the independent
32,768-command budget still apply, so an admitted trace layout does not guarantee that the
containing likelihood graph fits those budgets.

The tiled-panel sites now have explicit call-site allowances recording their reviewed axes and
bounds. This contributor is registered as reference tier. Its default tile layout caps each merge
at 4,096 partials; the entity scan remains explicitly reference work, capped at 1,024 entities,
and is not reclassified as coefficient work. Model initialization uses at most 142 words;
publication uses at most 10 coefficient slots, 81 covariance entries, and six summary words.
The opaque helper-generated residual-pair merge has the same 4,096-partial bound recorded too.
These annotations change no numerical code and require no new baseline entry for this file.

Other newly detected sites retain their existing computations. Their owners must review their
axes and bounds before synchronizing counts/reasons with the lead-owned
`scripts/contributor-contract-baseline.json`; a finding alone does not justify an allowance.

## Work the scalar-dispatch heuristic does not inventory

`../spatial-validation/gpu-spatial-cross-validated-ols.ts`, `held-out-metrics`, dispatches one
invocation per fold and scans every observation in each invocation. The nested cross-validation
constructor enforces at most 65,536 rows and 64 folds; these are data capacities, not coefficient
bounds. A count greater than one
does not make a serial inner row scan parallel. The checker deliberately retains its existing
rule name and meaning; the loop must still be considered in algorithm reviews.

## Verification

`modules/analysis/test/analysis-core/contributor-contract-checker.node.spec.ts` pins detection of
two scalar wrapper calls among parallel and unknown-count calls, local aliases, default and
overridden shader helpers, caller/callee guards, explicit allowances, lexical shadowing, mutated
parameters/aliases, and duplicate declarations without mixing callers. These tests run only host
code. No shaders or command graph topology change in this tooling review.

`stats-gwr-review.node.spec.ts` also constructs the reviewed boundary configurations: 16
restricted-regime coefficients and 1,024 tiled-panel entities are accepted, while 18 coefficients
and 1,025 entities are rejected. These checks pin the limits used to justify the classifications.
It also checks the default tiled-panel layout at 262,144/262,145 rows and the `2^24`-row cap,
where the number of partials never exceeds 4,096. This test does not bound explicit widths or
chunked nested trace layouts to 4,096 tiles. A real-source checker regression requires zero
unreviewed tiled-panel sites without relying on the baseline.

Inspect the reviewed region with:

```sh
node scripts/check-contributor-contract.mjs --all --rule single-invocation-data-loop --paths modules/analysis/src/gpu-spatial-analysis/spatial-regression
node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/analysis-core/contributor-contract-checker.node.spec.ts
```
