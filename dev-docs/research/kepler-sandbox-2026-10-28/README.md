> **Note (2026-11-07).** Copied from the luma.gl `map-contributors` branch (`dev-docs/reviews/kepler-sandbox-2026-10-28/`). Paths such as `modules/graph/src/...`, `@luma.gl/graph/...` and `examples/...` refer to that branch; the same code now lives in `@luma-contributors/*` in this repository. This review is current.

# Kepler UI and contributor sandbox review

Date: October 28, 2026. Target: installed Kepler **3.3.0-alpha.18**, Deck **9.4.0**, and this
working tree's luma **10.0.0-alpha.4**. Implementation is changing concurrently; the inventory records
reviewed integration requirements, not a release or a claim of complete runtime parity.

Kepler is a useful second consumer for luma contributors because people can choose datasets, combine
filters, change visual encodings and reuse computed results without writing a story. The strongest
proof will be the same contributor working on two unrelated user datasets through native Kepler
controls, with correct selections, reusable outputs and a saved workspace that opens in a fresh browser.

The existing adapter architecture can support that goal. **Complete functional coverage requires
consumer contracts and UI integration in addition to kernels.** Keep luma for resident interaction and
supported analysis, DuckDB for general relational work and precision-sensitive preparation, and host
services for loading, basemaps and storage. Every visible action needs an executable route or a clear
unavailable state. A contributor export alone does not make the corresponding Kepler control work.

## Scope and evidence

The [capability matrix](./capability-matrix.md) contains **66 entries**, including **all 22 layer IDs**
in the pinned registry. It covers data operations, filters, styling, inspection, analysis, execution,
sharing and usability. The source is the example's
[capability inventory](../../../examples/deck/kepler-compute/src/sandbox-capabilities.json), also used by
the searchable `SandboxCapabilities` panel. A node test compares its layer IDs and version with the
installed upstream registry so dependency upgrades cannot silently omit new layers.

The audit reads installed implementation files, not only the public feature list:

- `@kepler.gl/constants/dist/esm/layers.js`: all 22 layer IDs.
- `@kepler.gl/utils/dist/esm/application-config.js`: charts, dataset operations, sketches, JSON
  editors, column statistics, swipe, globe, remote sources and feature switches.
- `@kepler.gl/components/dist/esm/side-panel.js`, `side-panel/dataset-ops/`, `map/charts/`,
  `modals/` and `editor/`: native workflow and UI surfaces.
- [Host](../../../examples/deck/kepler-compute/src/main.tsx),
  [table bridge](../../../examples/deck/kepler-compute/src/kepler-table-bridge.ts),
  [domain and spatial binding](../../../examples/deck/kepler-compute/src/resident-table-binding.ts),
  [resident layer adapters](../../../examples/deck/kepler-compute/src/resident-layers.ts), and
  [analysis runtime](../../../examples/deck/kepler-compute/src/analysis-runtime.ts).
- [Kepler package](../../../modules/kepler/README.md),
  [aggregation contract](../../../modules/kepler/src/aggregation/README.md), and
  [analysis binding contracts](../../../modules/kepler/src/analysis/README.md).

The official [user guide](https://docs.kepler.gl/docs/user-guides) supplies workflow context, while the
installed registry is the authority for this target. The official
[save/export guide](https://docs.kepler.gl/docs/user-guides/k-save-and-export) describes native image,
data, JSON/HTML and cloud sharing; those paths need separate resident integration checks. Native UI
availability and numerical contributor availability are recorded separately throughout this review.

This complements the architectural [KI01–KI16 integration plan](../../roadmaps/kepler-compute-integration-plan.md).
It does not replace the numerical contracts in the [analysis implementation record](../../roadmaps/kepler-analysis-implementation.md).

## Findings that affect sandbox reliability

| Priority | Finding and consequence | Required resolution |
| --- | --- | --- |
| P0 | Baseline browser smoke initially failed with `process is not defined` in the browser `util` dependency, leaving no usable application. | Initialize the actual browser process shim before evaluating Kepler's dependency graph; verify development and production entry points. Bootstrap remediation is now owned by the integration chat. |
| P0 | After startup remediation, the baseline 10,000-row smoke reported only 5,000 selected rows before user filtering. A wrong initial selection makes every exploration misleading. | The integration owner identified pinned Kepler categorical domain merging as the cause. Preserve every category, and test all/none/subset, numeric labels and clear semantics at the native reducer boundary. Require a passing rerun before claiming this fixed. |
| P0 | Compact controls originally dispatched positional filter indices 0/1/2. Native removal or reordering could direct a control at a different field. | Resolve stable filter identity plus dataset, field and type; disable or offer restoration when its binding is absent. Test deletion, reassignment and reordering through the actual UI. An identity-based change is being integrated. |
| P0 | Most native consumers still assume CPU rows or filtered index arrays. The bridge intentionally clears those arrays and rejects synchronous export. | Give each consumer bounded asynchronous access or an explicit materialization route. Never let an empty CPU index array imply an empty resident dataset. |
| P0 | The initial point wrapper bypassed native channel formatting and ignored several native controls. Overriding the point class for every dataset also risked swallowing normal imported datasets. | Resident-aware wrappers must delegate normal datasets to native classes, validate resident geometry/channel bindings and show unsupported reasons. New adapters are under implementation; require mixed CPU/resident tests. |
| P0 | Recipe JSON stores dataset and selection revisions; it does not reconstruct a workspace or a selection in another browser. | Save source identity/content references, selection construction, weights definitions, operation versions and native map configuration. Rebuild resources and obtain new runtime revisions on restore. |
| P1 | The native histogram is not the whole Charts panel, a small readback preview is not the native Data Table, and a minute slider is not general timestamp playback. | Track and test each surface independently. Avoid marking whole workflow families complete from the demonstration controls. |
| P1 | Split/swipe view controls can create additional render consumers while the host currently uses a global runtime/effect. | Prove device identity and ownership per view; a second Deck device must not receive buffers created by the first. Test closing either view with pending readback. |
| P1 | Custom panels can display stale inspection or diagnostics while newer work is pending. | Tag every result with source/control revisions; clear or label stale values, suppress late completions, and distinguish empty, pending, unsupported and failed states. |

These are observed code paths and baseline results. Ongoing changes must be verified by the integration
owner; the ledger's conservative statuses do not certify that a new file is wired or working.

## UI structure for a reusable sandbox

Keep Kepler's map, layer manager, filter panel, interactions and dataset actions as the main interface.
Use one collapsible **Compute** area with three functions:

1. **Explore:** normal data, filters, linked charts and table inspection. Show the active dataset and
   selection count; expose original and computed fields together. A filter or layer control has one
   source of truth in Kepler state even when a compact shortcut also exists.
2. **Analyze:** choose a compatible operation, bind fields/geometry/weights, choose all versus selected
   population, set numerical parameters, and run/cancel. Publish a named field, table, geometry or raster
   result that other controls can consume. Keep result provenance and diagnostics inspectable.
3. **Coverage and execution:** searchable feature guide, execution route, supported input formats,
   precision/capacity limits and measured transfers. This is a guide to what is available, not another
   operator catalog that implies every item is executable.

Do not generate arbitrary runnable forms directly from ontology metadata. Add an executable binding
registry whose entries supply parameter validation, compatible input requirements, output schema,
capacity estimates, numerical policy, cancellation/cleanup, recipe migration and verification evidence.
Metadata and lenses can provide names, explanations, examples and links. Only a binding with a declared
runtime contract enables Run. Unsupported modes remain discoverable with the specific missing contract.

Offer a small sample gallery: points and time; polygons with counts/population; origin/destination
connections; trips; a numeric raster; and a directed network. Each should also accept user data with
the same schema. Presets configure reusable operations and visual encodings. They must not carry
story-specific computation implementations.

Improve the current local background with optional geographic context, while retaining a token-free,
offline fallback. Collapse the sidecar on small screens; test keyboard focus, label associations,
status announcements, contrast and access to native map controls at both desktop and 390px widths.
Do not infer accessibility from a successful GPU smoke test.

## Extending contributor coverage

| Family | Reusable luma foundation | Native sandbox contract to add |
| --- | --- | --- |
| Dataframes and statistics | Expressions, masks, grouping, profiles, quantiles, classifications, standardization | Bind arbitrary compatible columns; preserve nulls and units; publish derived fields; connect native domains, palettes, legends and charts. |
| Time | Exact timestamp words, window selection and calendar buckets | Native timestamp/interval semantics, zones, bins, playback and per-vertex trips. Keep display formatting distinct from exact comparisons. |
| Spatial statistics | Weights, local/global statistics, hot spots, rates and quantile LISA | Weights construction/cache, masks, islands, population policy, inference settings, corrections and diagnostics. Preserve documented quantile LISA limits and p-value caveats. |
| Geometry | Selection, spatial joins, refinement and cartograms | Geometry/CRS/units schema, feature identity after topology changes, output bounds and reusable result layers. Dorling remains a bounded heuristic, not a guaranteed converged layout. |
| Clustering and interpolation | Spatial-analysis contributor families | Operation-specific controls and independent numerical fixtures; viewport clustering is a separate host contract. |
| Raster and terrain | Raster grids, reprojection, statistics, band math, neighborhood and terrain operations | Numeric source bands, nodata, tile seams, CRS, texture/geometry outputs and explicit viewport versus complete-population semantics. |
| Networks | Graph construction and movement/accessibility contributors | Directed costs, snapping, source-edge identity, unreachable status, path and coverage outputs. |
| SQL and preparation | Bounded GPU SQL plus explicit DuckDB execution adapter | General SQL uses a real connection; declare transfers and result schemas. Do not equate DuckDB with a statistical backend. |

Keep each family in its existing package owner. Host registration belongs in the Kepler adapter or
example, not in `@luma.gl/gpgpu` primitives. Geometry uses source semantics, Arrow conversion stays in
`@luma.gl/arrow`, batches remain intact unless packing is explicitly requested, and borrowed buffers
remain owned by their source.

## Saving and sharing

Treat three deliverables separately:

- **Fixture link:** a small, versioned URL for a known sample, layer/filter/analysis settings and view.
  Replay generates the same fixture and re-executes supported recipes. It should say what it contains.
- **Portable workspace:** native Kepler configuration plus source manifests or included Arrow/data,
  contributor/version IDs, selection definitions, weights recipes, numerical policies and output
  provenance. A reference to a private/local source prompts reattachment; it must not silently use a
  different dataset with the same name. Do not serialize device handles, compiled graphs or buffer IDs.
- **Published sandbox:** a deployed, versioned runtime with required assets and an explicit storage/auth
  choice where needed. A localhost URL is not public hosting. Native Kepler HTML export does not by
  itself embed the resident adapter, contributors or DuckDB assets.

Restoration should validate schema and versions before changing the active workspace, reconstruct the
source and selections, resolve field identities, then execute recipes and attach layers. Export should
freeze source/selection revisions, report estimated size, stream bounded chunks and permit cancellation.
Any GPU-to-CPU materialization is deliberate and visible. Test selected recipes after a fresh launch,
not only by pasting JSON back into the originating session.

## Delivery sequence and exit gates

| Stage | Scope and dependency | Exit gate | Existing plan |
| --- | --- | --- | --- |
| 1. Trustworthy baseline | Startup, native filter semantics, resident/nonresident dispatch, capability reasons and stale-state handling | Import two datasets; native filters, visible map, count and bounded tooltip agree; no silent unsupported controls; dev and production smoke pass. | KI01–KI06 |
| 2. Daily exploration | Native table, domains, legends, charts, complete styling, drawing and main geometry layers | Paging/sort, crossfilter and legend match source IDs and revision; interaction does not download all selected indices or upload the source again. | KI07–KI10 |
| 3. Reusable analysis | Executable bindings, arbitrary compatible fields, weights, classification, standardization, rates and hot spots | Each operation runs on two unrelated datasets, publishes reusable results and passes independent numerical fixtures. | KI09, KI11 |
| 4. Reproducible sharing | Source manifests, workspace replay, filtered export, sample links and hosted runtime | Fresh browser reconstructs a saved selection and derived result; exported rows equal the map selection; missing sources/errors are explicit. | KI12–KI14 |
| 5. Broad coverage | Remaining layer matrix, trips, raster/network workflows, split views, optional services and recovery | Every agreed capability has a declared route and a pinned native behavior test; memory/device-loss cases pass. | KI08, KI12, KI14–KI16 |

Stage 4 can begin alongside stage 2 once source identity and selection serialization are stable.
Complete support is a row-by-row claim against the capability matrix. Approving this plan does not
require replacing every CPU service or numerical backend with a GPU implementation.

## Verification requirements

Use differential native-host fixtures as well as contributor unit tests. Cover empty/all-null data,
category clear/inversion, inclusive range endpoints, tied class boundaries, sliced batches, dictionary
changes, large IDs, Float64 conversion, signed timestamps and DST, polygon holes/dateline geometry,
one-to-many joins, selected/global weights, quick successive controls, replacement during reads,
export cancellation, memory admission and forced device loss.

For every supported warm workflow, record upload and readback bytes, encode/GPU/readback time, peak
accounted memory and p50/p95 interaction latency on a named browser/device. Registry accounting is
not total browser or driver memory. Kernel speed alone is insufficient. Compare cold and warm paths
with identical inputs and policies; smaller or cold datasets may favor CPU/DuckDB execution.

The existing smoke script exercises native actions through a development hook. Add actual UI clicks
for the native panels and production smoke without the hook. Include data table, layer styling,
analysis result publication, fresh-session restore, export and mixed CPU/resident datasets. Each
new adapter should fail closed for unsupported modes and retain ordinary host behavior elsewhere.

Review-specific results and commands are recorded in [verification](./verification.md). Full repository
and integrated host gates belong to the integration run and must pass after its final changes before
merge. This review does not imply those gates are green.

## Native Charts adapter requirements

The installed optional `@kepler.gl/components/charts` entrypoint exports chart factories, but the
pinned main container does not mount its manager/control. Registering a factory override or enabling
`enableChartsPanel` alone therefore does not expose a working editor. The host now explicitly
registers the complete factory dependency tree and mounts the manager, retaining Kepler's chart
configuration and add/update/remove actions.

`ChartPanelContentFactory` synchronously invokes `computeChart` after converting the dataset through
`toChartableDataset` or `toMapFilteredChartDataset`. Those functions consume CPU `allIndexes`,
`filteredIndex` and `getValue`. They cannot safely consume resident datasets whose CPU index arrays
are deliberately empty. The separate exported `ChartRenderer` accepts bounded `ChartViewData`,
providing a rendering boundary for asynchronous resident summaries. Ordinary host datasets should
retain the native path.

The pinned registry includes big number, vertical/horizontal bars, line, heatmap, pivot and layer
charts. Its 15 aggregation choices include count/sum/average/minimum/maximum, dispersion, quantiles,
mode and distinct counts. These require explicit execution contracts; a numeric histogram does not
cover the entire registry. Implementation should preserve these native semantics:

- `applyFilters` chooses filtered versus complete population. Active crossfilter charts exclude their
  owned filters when computing stable bins, while retaining other applicable source/layer filters.
- Numeric/time bins carry actual range endpoints separately from display labels. Category bins carry
  original values. Selection must update native filters and preserve endpoint/tie behavior.
- Source, selection, chart configuration and derived-field revisions identify a result. Pending or
  obsolete summaries must not appear current; deletion/replacement must release borrowed reads.
- Sort order, group limits, missing values, palettes and saved configuration must agree with native
  behavior. Heatmaps/pivots need bounded two-dimensional cardinality; layer charts additionally need
  layer scope and hover identity.
- Source arrays remain resident. Only bounded aggregates reach the renderer; unsupported chart types,
  measures, storage formats and budgets receive explicit reasons rather than empty plots.

Start with a native-configuration planner and independently verified scalar summaries, then add
bounded grouped bins and crossfilter interaction. Verify mixed resident/host datasets, multiple
charts, filter removal, empty/null populations and saved replay before widening the supported set.
The five scalar measures and bounded string-category bars now pass actual native UI creation,
filtering and saved replay. Category bars retain native first-occurrence ordering/colors, top-N and
Other, and self-excluding crossfilters while preserving unrelated filters. Numeric interval bars
also pass native range selection, empty bins, exact integer endpoints and horizontal replay.
Automatic integer grouping and single-series lines also pass CPU/GPU summaries and the native
mixed-source browser, including exact Uint32 singleton selections and line-point replay. The host
now preserves prepared Arrow field metadata instead of guessing large integer IDs are timestamps.
String-dictionary series now pass native oracle, GPU and actual UI checks. Bars/lines preserve native
sum-of-series totals with an accessible breakdown table; a bounded SVG renderer now draws separate
vertical/horizontal bars and lines, with stable colors, negative/zero values, missing-value gaps and
keyboard parent-group selections. The cross-product is bounded to 4,096 cells including a missing-label
slot per group. UTC hour/day/Sunday-week/month/year lines pass CPU/GPU and native browser checks for
second/millisecond Timestamp/Date sources, including chronological periods, exact inclusive selections and chart-owned
null epochs. Same-ID replay waits for the previous reader's selection-mask retirement before replacement.
Automatic calendar intervals, microsecond/nanosecond chart sources, numeric series
domains, remaining statistics and broader chart families remain work.
