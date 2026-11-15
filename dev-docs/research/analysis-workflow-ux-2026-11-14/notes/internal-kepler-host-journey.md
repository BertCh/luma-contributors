# Kepler analysis journey review

**Verified:** This is a source-only, read-only review, following the earlier study in `dev-docs/research/kepler-workflows-2026-11-14/README.md`. `git status --short` showed concurrent changes in the host, preparation and analysis runtime files. Findings below describe that working-tree snapshot; they are not assertions that the port is finished.

## Counts and action-count convention

**Verified:** These source commands returned **11 configuration-panel imports, 24 operation options including “off”, 8 gallery samples, and 7 assistant analyses**, respectively:

```sh
rg -c '^import .*Panel.* from' examples/deck/kepler-compute/src/analysis-panel.tsx
sed -n '169,243p' examples/deck/kepler-compute/src/analysis-panel.tsx | rg -c '<option'
sed -n '26,153p' examples/deck/kepler-compute/src/sandbox-samples.ts | rg -c '^    id:'
sed -n '182,190p' examples/deck/kepler-compute/src/app-store.ts | rg -c "^    '"
```

**Inferred:** Action counts below count opening a disclosure, choosing a control value, filling a field or submitting as one action each. They retain defaults and exclude navigation, waiting and corrections. They are not measured browser clicks. For an imported resident dataset, add one dataset-selection action; changing the default value column adds another. The selectors are at `examples/deck/kepler-compute/src/analysis-panel.tsx:85` and `:151`.

## Loaded dataset → mapped result

**Verified:** Layers contains an initially open Analysis disclosure. Add Data gains Prepare data and SQL methods. Native datasets normally prepare automatically under the same ID; manual preparation creates a resident copy. Sources: `examples/deck/kepler-compute/src/native-workflows.tsx:63`, `:110`; `examples/deck/kepler-compute/src/native-auto-prepare.ts:125`; `examples/deck/kepler-compute/src/source-workspace-panel.tsx:217`.

| Path | Minimum actions — inferred | Controls and availability — verified |
|---|---:|---|
| Standardization/classification/risk | 1: choose operation | Runs on changes; configured spatial modes remain disabled until configured. Analysis requires a nonempty prepared Float32 column, otherwise reports an error. `examples/deck/kepler-compute/src/analysis-panel.tsx:169`; `examples/deck/kepler-compute/src/analysis-runtime.ts:371` |
| Lattice hot spots | 4: open, width, height, apply | User declares row-major topology; width × height must equal source rows. Invalid settings produce a notice. Coordinates deliberately do not imply adjacency. `examples/deck/kepler-compute/src/analysis-hotspot-panel.tsx:45`; `examples/deck/kepler-compute/src/analysis-hotspot-settings.ts:20` |
| Point hot spots | 4: open, longitude, latitude, apply | Binary neighbors; defaults include k=8, minimum=8, self-weight=1 and BH correction. Can reuse spatial-rate coordinate/topology settings. `examples/deck/kepler-compute/src/analysis-point-hotspot-panel.tsx:14` |
| Local Moran | 4 | Same coordinate/topology choices; analytic/permutation and uni-/bi-/multivariate controls. Bivariate/multivariate selection forces permutation settings. Permutation option is disabled at the row limit. `examples/deck/kepler-compute/src/analysis-local-moran-panel.tsx:27`, `:123`, `:217` |
| Quantile LISA | 4 | Quantile, membership, seed, permutation and significance defaults supplied; submit disabled below three rows. Warnings explicitly call inference exploratory and anti-conservative. `examples/deck/kepler-compute/src/analysis-quantile-lisa-panel.tsx:42`, `:158` |
| Point weights/spatial rates | 4 | Also review event/population fields, scale and island policy. Coordinate fields start blank. `examples/deck/kepler-compute/src/analysis-point-weights-panel.tsx:14`, `:55` |
| OLS | 3 + predictor count | Open, response, predictors, apply. Response cannot also be a predictor; submit requires predictors and an admitted source-row count. Limits and assumptions are explained. `examples/deck/kepler-compute/src/analysis-ols-panel.tsx:33`, `:95`, `:123` |
| DBSCAN | 4 | Blank longitude/latitude; fixed geographic metric; radius=1,000 m and minimum samples=5. Row-limit failure disables submit with nearby limit text. `examples/deck/kepler-compute/src/analysis-dbscan-panel.tsx:41`, `:135` |
| IDW / kriging | 4 each; fitting adds 1 | Coordinate roles blank; both initially use fixed New York bounds, rather than dataset bounds. Grid, radius and model choices remain visible. Row limits disable submit. `examples/deck/kepler-compute/src/interpolation-panel.tsx:50`, `:170`; `examples/deck/kepler-compute/src/analysis-kriging-panel.tsx:58`, `:227`, `:326` |
| Cartogram | 9 | Open, X/Y, CRS, four bounds, apply; value defaults from the selected analysis field. Row limits disable controls and explain why. Non-3857 display adds four blank anchor fields and an apply action. `examples/deck/kepler-compute/src/analysis-cartogram-panel.tsx:16`; `examples/deck/kepler-compute/src/analysis-cartogram-controls.tsx:71`; `examples/deck/kepler-compute/src/analysis2-cartogram-declaration-form.tsx:15` |
| Polygon statistics | 2 for default Local Moran | First retained polygon field preselected; choose statistic, queen/rook and inference settings as needed. Panel disappears without polygon columns; parent can show “Polygon analysis unavailable: …”. Submit enforces row bounds; vertex validation occurs on apply. `examples/deck/kepler-compute/src/analysis-polygon-panel.tsx:54`, `:158`, `:494`; `examples/deck/kepler-compute/src/analysis-panel.tsx:120` |
| Terrain / zonal | 1 each with defaults | First raster/polygon selected automatically. Terrain chooses slope; zonal chooses band 1. Tools disappear without relevant inputs; zonal explains missing prerequisites. Runtime reports publication/fallback/failure. GPU-resident terrain explicitly produces a table without a map layer. `examples/deck/kepler-compute/src/raster-tools-panel.tsx:26`; `examples/deck/kepler-compute/src/raster-terrain-panel.tsx:42`, `:162`; `examples/deck/kepler-compute/src/raster-zonal-panel.tsx:23`, `:135` |

**Verified:** Applying a hand-written form sets its mode directly; a second operation-menu selection is unnecessary. Runtime updates request rendering; published render bindings use analysis colors, and their encode path runs analysis. A table-only source still needs display geometry to show row colors on the map. Sources: `examples/deck/kepler-compute/src/analysis-runtime.ts:928`, `:992`, `:2548`; `examples/deck/kepler-compute/src/source-publication.ts:823`, `:906`.

## Other entry points

| Entry | Journey and limits |
|---|---|
| Generated catalog | **Inferred:** 5 actions with built-in input; **5 + 3 per input** with native data: operation, source mode/dataset/geometry, run, stage, check, publish; add navigation and parameter edits. **Verified:** Defaults use samples. Geometry candidates are detected but never selected automatically. Run is disabled only while running; geometry/capacity rejection occurs on execution. `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:118`, `:396`, `:477`, `:529`; `modules/kepler/src/analysis/catalog-bindings.ts:598` |
| Catalog dataset eligibility | **Verified:** Host excludes resident datasets, including successfully auto-prepared inputs. Remaining native datasets need a detected GeoJSON field; plain longitude/latitude columns are insufficient. Errors include “Choose a dataset and geometry field” and “Native dataset exceeds catalog capacity”. `examples/deck/kepler-compute/src/app.tsx:498`; `examples/deck/kepler-compute/src/analysis-catalog-input.ts:23`, `:43` |
| Sources / preview | **Inferred:** Candidate → check → publish is two actions, plus role/precision edits. **Verified:** Suggestions prefill roles; manual native preparation supplies none. Unsupported selected fields block publication with “Choose a supported conversion or exclude these fields…”. `examples/deck/kepler-compute/src/source-workspace-panel.tsx:151`, `:197`; `examples/deck/kepler-compute/src/source-preview-panel.tsx:63`, `:182`; `examples/deck/kepler-compute/src/source-preview.ts:103` |
| SQL | **Verified:** App mounts `NativeSqlExplorer`; no source caller of `DuckDBPanel` was found. **Inferred:** Writing SQL + Run is two actions for automatic GPU publication; CPU results add Publish, or Prepare → check → publish. Geometry roles may still be needed. Legacy `DuckDBPanel` needs Enable → Register → Run → Upload. `examples/deck/kepler-compute/src/app.tsx:519`; `examples/deck/kepler-compute/src/native-sql-explorer.tsx:58`, `:81`; `examples/deck/kepler-compute/src/duckdb-panel.tsx:57` |
| Assistant | **Inferred:** JSON entry + approval + run is three actions. **Verified:** This is a structured-operation surface. Analysis supports seven named transformations, executes only against built-in observations, and rejects unsupported recipes/precision/populations. SQL/chart requests stage notices directing users to manual panels. `examples/deck/kepler-compute/src/assistant-panel.tsx:21`; `examples/deck/kepler-compute/src/app-store.ts:182`; `examples/deck/kepler-compute/src/assistant-operation-router.ts:567`; `examples/deck/kepler-compute/src/assistant-host-capabilities.ts:13` |
| Resident charts | **Inferred:** No fixed end-to-end count: native chart creation precedes this component. Existing cards update immediately from dataset/category/measure choices; no Run action. **Verified:** Preparation, unsupported, error and stale states have messages. Crossfilter changes map filters rather than publishing analysis geometry. `examples/deck/kepler-compute/src/resident-chart-panel.tsx:139`, `:284`, `:318`, `:564` |

## What the input already determines

All rows below are **verified**.

| Function | Existing inference/default |
|---|---|
| `makeArrowColumnFromNativeField` | Uses Kepler’s analyzer types; scans integer ranges, retains reals as Float64 before preparation, parses timestamps and serializes geometry/nested values. `examples/deck/kepler-compute/src/native-column-arrow.ts:40` |
| `inferLayerRolesFromFields`, `makeAutoPreparePlan` | GeoJSON/GeoArrow, H3 and point-pair roles; prefers linked roles, visibility and layer-type priority. Supports trip, cell, flow and connection roles; rejected roles degrade visibly to tables. `examples/deck/kepler-compute/src/auto-prepare-plan.ts:140`, `:235`, `:385` |
| `inferTripTimestampEncoding` | Samples first path vertices; chooses ISO, seconds or milliseconds. Preparation tries increasing time-error budgets. `examples/deck/kepler-compute/src/auto-prepare-plan.ts:175`, `:278` |
| `makeSourceTypeRoutes`, `previewArrowSource` | Arrow scalar/category/time routes, sampled WKT detection and derived GeoJSON, measured conversion errors and unsupported-field reasons. Automatic preparation selects supported fields and uses a declared Float64 rounding policy. `examples/deck/kepler-compute/src/source-type-routes.ts:87`, `:116`; `examples/deck/kepler-compute/src/source-preview.ts:38`; `examples/deck/kepler-compute/src/auto-prepare-plan.ts:441` |
| `AnalysisRuntime`, `usePointTopology`, `PolygonPanel` | Selects `value` or first Float32 column, another column as denominator; neighbor capacity derives from row count. Polygon neighbor/pair capacities derive from rows. These are structural defaults, without field-meaning inference. `examples/deck/kepler-compute/src/analysis-runtime.ts:371`; `examples/deck/kepler-compute/src/analysis-point-topology-controls.tsx:20`; `examples/deck/kepler-compute/src/analysis-polygon-panel.tsx:54` |
| `normalizeHotspotSettings`, `suggestKrigingVariogram` | Computes lattice neighbor capacity; fits variogram family/parameters. Empirical maximum lag defaults to one-third of data diagonal. `examples/deck/kepler-compute/src/analysis-hotspot-settings.ts:53`; `examples/deck/kepler-compute/src/kriging-diagnostics.ts:47`; `modules/kepler/src/analysis/kriging-variogram-fit.ts:111` |

**Verified:** Point forms retain explicit coordinate roles; projected mode additionally requires CRS and bounds. Distance-band radius defaults to 1,000 m, rather than a data-derived distance. Sources: `examples/deck/kepler-compute/src/analysis-point-coordinate-settings.ts:83`; `examples/deck/kepler-compute/src/analysis-point-coordinate-controls.tsx:118`; `examples/deck/kepler-compute/src/analysis-point-topology-controls.tsx:20`.

**Inferred:** Known roles could support suggestions. Geometry type alone cannot establish an appropriate value, denominator, CRS or statistical distance.

## Samples, duplication and explanation

- **Verified:** Gallery samples are H3 cells, observations through time, districts, connections, timed routes, grouped trip observations, numeric grid and directed network. Each provides workflow prose and publication roles, **not an analysis recipe**. Numeric-grid metadata includes its shape, but users are instructed to enter 16 × 16 for lattice analysis. `examples/deck/kepler-compute/src/sandbox-samples.ts:26`, `:155`, `:432`.
- **Inferred:** Gallery preparation takes four actions with the default sample: open, prepare, check, publish; another sample adds selection. **Verified:** Catalog samples already populate operation inputs. Lens scenarios are a separate configured execution path: load scenario modules, choose/run a scenario, optionally map numeric fields, then stage outputs. `examples/deck/kepler-compute/src/sample-gallery-panel.tsx:21`; `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:132`, `:194`, `:304`.
- **Verified:** Catalog and hand-written panels overlap in IDW, kriging, Moran and hot spots, but contracts differ. Catalog Moran uses row-standardized weights/permutations without correction; the hand-written form documents binary unnormalized weights and offers corrections. Catalog kriging explicitly lacks fitting/cross-validation. `modules/kepler/src/analysis/catalog-platform3-b-bindings.ts:210`, `:657`, `:995`; `examples/deck/kepler-compute/src/analysis-local-moran-panel.tsx:288`, `:342`.
- **Verified:** Hand-written results explain counts, ranges, missingness, classes and inference. OLS adds coefficients/diagnostics; kriging adds variance/validation; cartogram adds convergence/overlap. `examples/deck/kepler-compute/src/analysis-panel.tsx:418`; `examples/deck/kepler-compute/src/analysis-ols-panel.tsx:158`; `examples/deck/kepler-compute/src/interpolation-panel.tsx:277`; `examples/deck/kepler-compute/src/analysis-cartogram-panel.tsx:164`.
- **Verified:** Analysis legends omit OLS, cartogram and interpolation. Tooltips can read the primary result and bounded companion columns; interpolation needs a position reader. Catalog presents contract text, engine/fallback, raw summary keys, diagnostics and previews, without a result-specific map explanation in that panel. `examples/deck/kepler-compute/src/analysis-legend.ts:62`; `examples/deck/kepler-compute/src/analysis-tooltip-provider.ts:35`, `:60`; `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:367`, `:503`.

## Friction, ordered by expected exposure

Frequency ordering is **inferred**, without usage telemetry.

1. Choosing an operation and meaningful value among structurally eligible fields: every analysis (`analysis-panel.tsx:151`, full path above).
2. Re-entering coordinate roles across point-analysis forms (`examples/deck/kepler-compute/src/analysis-point-coordinate-settings.ts:83`).
3. Discovering that prepared data disappears from catalog inputs (`examples/deck/kepler-compute/src/app.tsx:498`).
4. Crossing stage/check/publish surfaces for catalog results (`examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:529`).
5. Reviewing fixed interpolation bounds, neighbor choices and statistical assumptions (`examples/deck/kepler-compute/src/interpolation-panel.tsx:50`).
6. Declaring projected coordinates and display anchors for cartograms (`examples/deck/kepler-compute/src/analysis2-cartogram-declaration-form.tsx:15`).

## Opportunities

**Inferred**, directly from the findings:

- Carry known publication coordinate roles into analysis drafts.
- Expose dataset compatibility and rejection reasons before Run.
- Offer extent-derived grid suggestions and clearly distinguish fixed defaults.
- Attach reviewable analysis settings to gallery workflow guidance.
- Explain differences between similarly named catalog and resident operations.
- Connect result explanations to the mapped output consistently.

## What I checked

- **Verified:** The earlier study; every cited file, using whole-file reads or selected ranges.
- **Verified:** Additional reads/searches included `sample-data.ts`, `native-hosts.ts`, `assistant-host.ts`, `assistant-published-tools.ts`, `analysis-catalog-lens-mapping.tsx`, `resident-chart-plan.ts`, `resident-channel-legend.tsx`, `resident-tooltip-fields.ts`, `analysis-neighbor-tooltip.ts`, `device-init.ts`, `main-raster-live.ts` under `examples/deck/kepler-compute/src`.
- **Verified:** Commands: `cat`, `rg`, `rg --files`, `wc -l`, `nl -ba`, `sed`, Python static counting, and read-only `git status --short`. Some locator probes failed; counts were successfully rerun.

## Not checked

- Browser interaction, actual click counts, WebGPU execution, visual layout or accessibility.
- Tests, builds, lint, numerical correctness or performance.
- `dist`, complete catalog coverage, website journeys or dataset licensing.
- Whether concurrent edits resolve the observed integration limits.