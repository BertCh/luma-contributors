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

The original regression baseline contained 21 construction sites in 15 files. Several sites have
multiple loop axes. The restricted-regime sites now carry reviewed `contract-allow` comments;
their old baseline entries can be removed by the lead.

| File and kernel | Serial axis and bound | Route and classification |
| --- | --- | --- |
| `gpu-geographically-temporally-weighted-regression.ts`, `summary` | Two row scans: at most 32,768 rows with exhaustive traversal, 16,777,216 on the indexed route (`GPU_GTWR_MAXIMUM_ROW_COUNT`, `GPU_GTWR_MAXIMUM_INDEXED_ROW_COUNT`). | GTWR summary; data-sized reduction, even when local fitting uses an index. |
| `gpu-geographically-weighted-regression-nonstationarity-test.ts`, `anchor` | Design columns, including intercept: at most 12, from GWR's 11-predictor cap. | Fixed coefficient work; tiled row work follows in separate kernels. |
| `gpu-geographically-weighted-regression.ts`, `ladder-init` | Parameter words: header 4 plus ladder capacity at most 32, hence at most 36. | Fixed parameter initialization; no row scan. |
| Same file, `refine-ladder` | Prior candidate comparisons and next ladder values: ladder at most 32, at most 4 refinement passes, hence at most 160 prior candidate rows. | Bounded bandwidth-candidate work, not observation work. |
| Same file, `choose` | Candidate table rows: at most 160 for finite ladders/refinements or 65 for a 64-step golden-section search. | Bounded bandwidth selection. |
| Same file, `summarize` | Row-tile summaries: `ceil(rowCount / 256)`; bounded by the selected GWR traversal's row cap (never more than 16,777,216 rows). | Data-dependent tile merge; local row fitting is separate. |
| `gpu-multiscale-geographically-temporally-weighted-regression.ts`, `convergence-score` | Rows times design columns: at most 32,768 times 12; at most 32 sweeps. | MGTWR exhaustive reference route; data-sized convergence reduction. |
| `gpu-multiscale-geographically-weighted-regression.ts`, `convergence-decide` | At most 256 convergence workgroups, with three partial words each; `CONVERGENCE_MAXIMUM_GROUP_COUNT` fixes the cap. | Bounded merge after parallel contribution checks; reviewed call-site allowance independent of the baseline. |
| `gpu-probit-regression.ts`, `pinkse-kp-pinkse-slade` | Six diagnostic result slots; reads already reduced state scalars. | Fixed output initialization and scalar diagnostics, independent of row count. |
| `gpu-restricted-regime-regression.ts`, `nested-models` | Restricted and unrestricted coefficient axes at most 16; regime axis at most 8; covariance axes at most 16 by 16. Workspace initialization depends only on those axes. Constructor enforces `(predictorCount + 1) * regimeCount <= 16`. | Small dense systems after tiled regime moments. Reviewed allowance; no serial observation scan. |
| Same file, `finish` | At most 16 coefficient outputs, 256 covariance outputs, and 3 Chow outputs. | Fixed coefficient publication; reviewed allowance. |
| `gpu-spatial-panel-error-maximum-likelihood.ts`, `absorbed-intercept` | Four intercept-table slots; reads reduced means and fitted coefficient slots. | Unit-effects panel error publication; no observation loop. |
| `gpu-spatial-panel-lag-maximum-likelihood.ts`, `unit-ml` / `random-ml` construction site | Up to 32 entities and 32 periods (1,024 observations) on the reference route. Dense entity determinant/inversion loops, observation scans, coefficient systems, at most 128 effect iterations and 64 search steps. | Reference route only. Mixed entity-, row-, and coefficient-axis work; distinct from tiled pivot selection below. |
| Same file, `publish` | Coefficient and covariance layout sizes, from at most 7 predictors and the spatial/intercept slots; six summary fields. | Reference-route publication; fixed coefficient work. |
| Same file, `hausman` | At most 7 slope contrasts and their square covariance system; three test outputs. | Optional reference-route unit/random contrast; bounded coefficient work. |
| `gpu-spatial-panel-regression.ts`, `unit-finish` | Coefficient table rows derived from `predictorCount + 2`, validated by `getGPUSpatialTwoStageLeastSquaresMaximumPredictorCount` (8 or 5 predictors by instrument order); 16 summary slots. | Unit-effects panel 2SLS publication; fixed coefficient work. |
| `gtwr-bandwidth-search.ts`, MGTWR component `summary` | Up to 32,768 rows, plus 12 summary fields; inherited from MGTWR's exhaustive row cap. | Component bandwidth objective; serial row reduction. |
| `gwr-golden-section-nodes.ts`, `search-init` with GWR layout | Candidate rows at most 65, seven fields per candidate. Bounds merge has at most 4,096 partials: the historical reducer uses 256-row tiles through 1,048,576 rows; above that named strategy threshold, parallel reduction feeds one bounds record. | Fixed search-table initialization and capacity-bounded partial merge; reviewed call-site allowance independent of the baseline. |
| Same construction site with MGWR layout (`mgwr-bandwidth-selection.ts`) | At most 66 candidate rows with three fields each: 198 words at 64 search steps. Two interior rows plus the upper-endpoint probe require the extra row. | Fixed search-table initialization; caller supplies distance bounds, so no bounds-tile merge. The allocation boundary is pinned by `stats-gwr-independent-review.node.spec.ts`. |
| `gwr-grid-index.ts`, `grid-bounds` | Exactly four components after upstream reductions. | Fixed bounds publication; row extent reductions occur in earlier nodes. |
| `multiscale-inference.ts`, `full-inference-summary` | Focus and response rows at most 128 each, with at most 25 design columns; construction additionally enforces `rows^3 * columns * maximumIterations <= 268435456`. | Optional full MGWR inference; bounded but data-sized quadratic summary, separate from conditional scalar errors. |
| `panel-diagnostics.ts`, `pooled-panel-lm-tests` | Full observation scan, at most `2^24` panel rows through the panel-error contributor; six diagnostic outputs. | Optional pooled panel diagnostics; data-sized reduction despite fixed output length. |

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
| `gpu-instrumental-variables-covariance.ts`, `projection`, `finish` | Up to 16 instruments and 15 coefficients; projection/covariance matrix entries derive only from those caps. | Small dense systems and covariance publication after tiled row moments; no row scan in these scalar stages. |
| `gpu-spatial-error-gm.ts`, `weights-norm-merge`, `finish` | Norm partials: constructor enforces `tileCount <= MAXIMUM_TILE_COUNT = 4096`, including explicit tile widths and the sum of per-chunk tile counts. Finish iterates coefficient/covariance slots from at most 15 predictors. | Bounded tile merge and fixed coefficient publication; default whole-CSR layout still reaches `2^24` rows. |
| `gpu-spatial-two-stage-least-squares.ts`, `weights-norm-merge`, `solve`, `refine` | Norm partials: constructor enforces `ceil(rows / tileRowCount) <= MAXIMUM_TILE_COUNT = 4096`, including explicit tile widths. Solve/refinement use at most 16 centered instruments, with 8 or 5 predictors according to instrument order. | Bounded tile merge and coefficient systems; default layout still reaches `2^24` rows. |
| `gpu-spatial-weights-traces.ts`, `norm-merge` | Constructor enforces `tileCount <= MAXIMUM_TILE_COUNT = 4096`, including explicit tile widths and the sum of per-chunk tile counts. Default tile width remains `max(32, ceil(rows / 1024))`. | Bounded norm-partial reduction on exact and stochastic routes; default whole-CSR stochastic layout still reaches `2^24` rows. |
| `ordinary-sur-nodes.ts`, `merge-means`, `solve-ols`, `solve-error-covariance`, `solve-gls` | Row-tile count from `getRegressionTileLayout`, with at most `2^24` rows and default at most 4,096 tiles. Systems have at most 4 equations and 16 total coefficients. | Mixed data-dependent tile merges and small dense equation/coefficient solves; row residual moments run in earlier tiled nodes. |
| `panel-lag-tiled-nodes.ts`, `initialize`, `solve-moments`, `profile-rho`, `information-solve`, `pivot-select`, `publish` | Model/output initialization depends on at most 7 predictors. Moment/information stages merge `ceil(rows / tileRowCount)` partials, default at most 4,096 tiles. Profile uses at most 64 search steps and scans at most 1,024 supplied eigenvalues per objective. Pivot selection scans remaining entities, at most 1,024, once per entity pivot. | `fitMode: 'tiled'` only: at most `2^24` rows. Pivot selection is entity-sized serial work, O(entities²) over all pivots; elimination remains parallel. It is not a coefficient-only loop and not the reference route's repeated dense determinant. The helper-generated residual-pair merge is another data-dependent tile merge; its generator call is opaque to this heuristic. |

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
It also checks the tile-layout boundary at 262,144/262,145 rows and the `2^24`-row cap, where
the number of partials never exceeds 4,096. A real-source checker regression requires zero
unreviewed tiled-panel sites without relying on the baseline.

Inspect the reviewed region with:

```sh
node scripts/check-contributor-contract.mjs --all --rule single-invocation-data-loop --paths modules/analysis/src/gpu-spatial-analysis/spatial-regression
node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/analysis-core/contributor-contract-checker.node.spec.ts
```
