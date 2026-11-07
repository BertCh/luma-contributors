> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/kepler-compute-integration-implementation.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Superseded by the roadmap; kept as the full historical log.

# Kepler compute integration implementation

Updated October 30, 2026. This records the implementation of the
[resident integration plan](./kepler-compute-integration-plan.md). The plan remains the target;
this document records delivered contracts and their limits. It does not assert complete Kepler parity.

## Implemented architecture

The private [`@luma.gl/kepler`](../../modules/kepler) package owns the application adapter. The
[`kepler-compute` example](../../examples/deck/kepler-compute) mounts actual Kepler
3.3.0-alpha.18 with the local luma 10 sources and patched Deck 9.4. Deck owns the WebGPU device
and submission; the compute effect encodes selections and derived outputs before rendering.

GPU data, graphs and resource leases stay outside Redux. Native filters and layer configuration
remain in Kepler state. A pinned source transform intercepts Kepler's synchronous plot/domain
consumers; it checks the expected source shape and fails on upstream drift. This patch is an
integration seam to replace with an upstream asynchronous contract, not a general Kepler fork.

The host uses native Kepler entry points: Layers contains analysis controls; Add Data contains
preparation and optional SQL; Export Data and Map expose bounded exports and portable replay.
Native Charts lives in the map controls. The separate Compute panel has been removed, and tooltip
fields follow the saved native configuration while reading only the bounded clicked source row.

| Plan area | Implemented contract | Remaining boundary |
| --- | --- | --- |
| KI01 shared host | Actual Kepler shell, shared Deck device, compute-before-render, resident rendering and original-row picking | WebGPU only; pinned compatibility adapters; native editor dashed strokes disabled because their extension is GLSL-only |
| KI02 registry | Dataset revisions, explicit CPU/GPU reservations, leases, completion fences, cancellation and stale publication guards | Driver/JavaScript and executable-private allocations are not a complete process-memory measurement |
| KI03 Arrow bridge | Narrow integer/Boolean widening, explicit Float64 error policy, category dictionaries, signed 64-bit timestamp sidecars, batch preservation | General decimals/wide arithmetic and unsupported types require explicit complementary execution |
| KI04 native consumers | Table factory, native filter translation, separate domain explorer, resident layer descriptors, guarded unsupported consumers | Original source rows remain for ingestion compatibility; arbitrary synchronous CPU consumers cannot transparently consume GPU-only columns |
| KI05 rendering | Native point/line/arc/GeoJSON/direct Trips/Icon/H3 adapters, multipart geometry, independent primitive masks/styles/picking, embedded property/default styles, point outlines and XYZ points, source/target Line gradients, exact SVG icon meshes, native H3 topology branches, independent GeoJSON height/base-offset channels and GPU extrusion/wireframe primitives, path joins/caps and explicit native Trips timing, bounded one-line scalar point labels, explicit tabular point altitude, borrowed buffers | Grouped tabular Trips, scenegraph markers, non-point altitude roles, portable custom icon catalogs, native 3D hover highlighting, rich labels, globe and remaining layer families need separate adapters; optional path antialiasing has a bounded approximation contract; unsupported property values report a reason |
| KI06 selection | Per-filter scalar/time masks, filter-specific histograms, independent display/domain masks, per-layer point/endpoint/MultiPoint/JSON, homogeneous Arrow and explicit WKB representative predicates and native targeted-layer union, native CPU-topology masks and chart-specific consumer exclusions | Antimeridian selection, unlabelled mixed-union representatives, screen-space brushes and interval UI need more adapters |
| KI07 summaries | Global GPU quantiles/domains, resident color/scalar scales, exact custom thresholds, fixed/dynamic log domains, category remapping, bounded legends, native Big Number, category/numeric/integer charts, UTC hour/day/week/month/year second/millisecond-time charts, separate-series bars/lines and bounded grouped summaries with consumer-specific selections, native time-filter histograms with compiled interval metadata | Dynamic native Jenks, distinct NaN/null colors, additional calendar intervals/units, remaining chart families and broad high-cardinality workflows remain |
| KI08 aggregation | Resident grid/hex count/sum/mean and exact footprints; planar native heatmap SUM/MEAN with ordered GPU weights/masks, radius and linear/sqrt/log normalization | Grid/hex retain explicit centroid Mercator meter projection; heatmap excludes native brushing/globe/picking. Member tables, aggregation extrusion, DGGS and full style parity remain separate work |
| KI09 analysis | Typed bindings and native controls for standardization/classification/rates/hotspots, geographic and explicitly declared projected-meter point weights, exact original-XY polygon queen/whole-edge rook weights, analytic and conditional-permutation Local Moran, point/polygon Quantile LISA, bounded OLS/ridge, geographic IDW/DBSCAN, bounded ordinary kriging with prediction/variance inspection and export, bounded Dorling cartograms with an explicit EPSG:3857 overlay and source-row results | The complete contributor catalog and GeoDa tool surface remain broader than these adapters; arbitrary projected CRS cartograms remain table-only; polygon rook does not implicitly node partial edges |
| KI10 inspection | Revision-aware bounded pages, original-row tooltips, derived-value inspection, stable GPU sorting across batches, native table modes for prepared and exact original/derived values, bounded rich-cell previews | Original rich-type sorting/statistics and joins/group membership need additional contracts; full retained values remain available through explicit source export |
| KI11 DuckDB | Optional real WASM worker, explicit Arrow registration, generation-guarded SQL results, explicit preparation and native dataset publication, transfer counters | Worker/WASM memory is separately owned; no automatic hybrid SQL splitting or hidden GPU materialization |
| KI12 lifecycle | Admission checks, leases/fences, explicit device recovery, chunk-preserving Arrow stream publisher and native ingestion UI with bounded admission, pause/cancel, explicit incomplete previews, completed snapshot handoff and recovery | Partial-prefix persistence, remote streaming sources, background retry and eviction policies need further integration |
| KI13 replay/export | Versioned multi-source state, bounded portable Arrow bundles with integrity/schema/role checks, deterministic fixture links, explicit prepared GPU data, original-source Arrow/CSV, derived analysis results and bounded georeferenced IDW GeoTIFF export | Original-source export materializes retained CPU rows using bounded GPU-selected IDs; derived export retains its result snapshot and recipe. URL links remain fixture-only; remote-source credentials and atomic device-allocation rollback are separate work |
| KI14 verification | Unit, real WebGPU, native Deck reference, and actual Kepler browser workflow tests | Full-suite results and performance scale must be recorded separately from focused correctness |

The [independent feature review](../research/kepler-sandbox-2026-10-28/README.md) inventories all
22 pinned layer types and 44 additional workflows. The sandbox's searchable guide displays that
static inventory and its acceptance criteria; it is explicitly not a live compatibility guarantee.

## Numerical and ownership contracts

Source batches remain distinct. An operation that needs packed scratch allocates that intermediate
explicitly without rewriting source chunks. Geometry selection acts at source-feature granularity.
All borrowed source resources survive until the last lease and submitted work complete.

Float64 upload requires an exact or bounded-error conversion policy. Absolute timestamps use signed
64-bit words and explicit units, while native slider endpoints use exactly validated milliseconds.
Display selection and dynamic-domain selection have separate masks. Analysis exposes all versus
selected population as an explicit choice. Unsupported variants do not silently change algorithms.

DuckDB queries the retained CPU Arrow representation, independently of map visibility. Registration,
query materialization and GPU upload are separate actions. General SQL is complementary execution;
it does not imply replacement of luma's resident interactive path or vice versa.

GPU picking currently enforces the 24-bit source identity limit for pickable vector layers. Path joins
and caps have native reference coverage; optional path antialiasing has a bounded approximation
contract. Some native arc/globe modes remain unsupported. The adapters keep these boundaries visible.

## Second implementation tranche

The source workspace now accepts Arrow IPC, retained native datasets and SQL results through a shared
schema/precision preview. Coordinate roles are explicit, original names and source batches survive,
and publication creates native tables and point or GeoJSON layers. Multiple published datasets coexist
with the fixture. Registration waits for Kepler's asynchronous ingestion before deletion reconciliation
is allowed to retire a consumer; source deletion retires style/geometry consumers before registry data.

Geometry preparation separates source rows from points, line parts, rings and tessellated vertices.
GPU gathers expand selection and style columns into each primitive space. JSON selection uses the
pinned bounds-center recipe, including Feature bbox overrides and the native rectangle boundary rule;
it retains null source slots rather than reproducing upstream's observed null-center index shift.
Homogeneous native GeoArrow uses its distinct vertex-mean selection recipe, including the pinned
polygon rule that excludes the first coordinate of every ring. It honors logical Arrow row bounds,
slices and validity rather than reproducing the pinned converter's padding-derived ghost centers.
Raw WKB uses its distinct all-vertex mean recipe, including polygon closure vertices. Decoded unions
may use that recipe only with explicit WKB provenance; unlabelled unions have no inferred selection
recipe. WKB GeometryCollection and measure-coordinate variants are diagnosed per source row rather
than silently adopting another source mode's semantics. XYZ point projection, hollow point outlines
and source-to-target Line color gradients are now implemented.

Scalar predicates are keyed by native filter ID. Layer masks intersect the common scalar population
with only their targeted polygons. Native table/analysis masks union polygon-targeted layer results,
including hidden targets, while dynamic-domain masks exclude polygons. Each histogram excludes only
its own filter, including when two filters use the same column. Grid/hex programs borrow their own
layer masks. Geometry and time controls are captured in encoder order so two recorded revisions before
one submission do not accidentally execute with the final control values twice.

Resident style plans preserve original chunks while deriving a global domain in explicitly allocated
GPU workspace. Palette edits reuse outputs and avoid unnecessary domain computation. Numeric mapping,
category ordering, fill/stroke separation and point radius versus path width remain distinct contracts.
Bounded legends carry style/source/selection revisions and replace native quantile recomputation from
incomplete CPU metadata. Empty selected populations produce no-data output and matching empty legends.
Custom numeric thresholds retain explicit fixed controls when selection domains change; their
internal padded classification edges do not appear as artificial legend boundaries. Log domains can
be fixed or GPU-derived from source/selection populations. Derived domains retain the complete finite
extent and apply the pinned zero-minimum floor; they do not drop negative observations to manufacture
a positive domain. Same-sign negative and descending positive domains are supported. Invalid domains
produce no-data outputs and an explicit bounded legend status. This does not mean the pinned native
UI offers logarithmic scaling for every channel.

Embedded GeoJSON colors, widths and radii retain immutable source values plus validity. A cached GPU
selection stage applies the current native fallback without reuploading source data; field channels
and analysis overrides take precedence. Disabled channels do not reject a layer because of unused
source properties. Fallback controls are captured in encoder order and their output buffers retire
behind consumer fences.

Direct timed GeoJSON LineStrings now publish as native Trips layers. Preparation retains original
timestamp values and exact integral epoch milliseconds, with an explicit encoding, retained origin
and Float32 error budget. GPU path and timestamp chunks preserve source batch boundaries. Null or
invalid source rows never renumber valid trips. Source-aligned masks and line properties are gathered
onto valid trip primitives on the GPU, with accounted output storage and fenced cleanup.

Native animation registers the absolute source-time domain and subtracts the prepared origin exactly
once for rendering. Trail seconds convert to relative milliseconds. Playback, seeking, filtering and
picking use the existing resident allocations; the native per-frame CPU row callback is bypassed.
Portable source manifests retain timing policy and validate it before replacing a workspace. This
path supports one timed LineString per row; grouped tabular trips and scenegraph markers remain
explicitly unsupported.

Native Icon uses its supplied SVG triangle meshes directly through `GPUMeshIconLayer`; it does not
substitute texture-atlas glyphs. Source strings are grouped once per source/catalog revision with
original row maps and batch boundaries. GPU gathers update positions, colors, radii and masks for
each group without source reuploads. Catalog loading retains the native redraw callback, and missing
marker definitions report a diagnostic. The adapter preserves the native fixed-radius and unclamped
high-zoom scaling rules. Current host coordinate roles supply XY; explicit XYZ callers are supported
by the primitive/runtime, but tabular altitude and portable custom catalog assets remain host work.

Resident rendering now separates native layer identity from the physical Deck renderer ID. The
pinned Deck manager transfers state solely by ID, even between constructors; type-specific render
IDs prevent a native point-to-Line switch from inheriting a scatterplot model. Picking and diagnostics
retain the native ID explicitly.

Device loss immediately hides stale counts, inspection and analysis results. A healthy in-memory
checkpoint retains source Arrow references, native state and recipes. The explicit recovery action
creates a new Deck device, reimports original dataset IDs and batches, then replays controls and
recipes before publishing the candidate. Creation has a 30-second timeout and supports cancellation;
late candidates and retired-device callbacks cannot replace the active workspace. No source payload
is saved to browser storage. Reloading the page still requires reattaching or importing saved sources,
and native external assets retain their own loading requirements.

Published dataset analysis now selects an eligible prepared Float32 field explicitly. Computation may
use packed scratch, while result views preserve the original source batch boundaries for styles and
native table inspection. Derived columns avoid collisions with original field names and retain
analysis/source/selection revisions. Each source owns its analysis lifecycle and saved recipe.

Hot-spot controls accept an explicit row-major lattice shape, rook/queen neighborhood, radius,
minimum neighbors, self weight and inference correction. Point spatial rates instead take explicit
longitude/latitude columns and great-circle meter distances for k-nearest-neighbor weights. The
point topology uses all source observations independently of the selected statistical population;
changing a filter does not silently construct a new nearest-neighbor graph. Recipes retain those
coordinate, metric and population choices. Point hot spots share those cached source-wide weights
with rates, with explicit self weight, minimum neighbors, islands and analytic inference correction.
Filtering changes the statistical population without rebuilding the graph. The cache reuses only
acknowledged GPU submissions and retains bounded in-flight entries through their fences. Initial
dense-neighborhood construction may still be quadratic. Subsequent adapters add distance bands,
exact original-XY polygon contiguity and bounded conditional permutations; their limits and evidence
are recorded below. General weights editing remains separate work.

The native table now offers prepared values or original source values plus computed columns. Bounded
GPU source IDs select retained Arrow cells without a bulk source scan. Float64, Int64 and timestamp
ticks/units remain exact; geometry and long text use explicitly bounded previews with full source
export available. Derived/source name collisions remain distinguishable. Sorting is enabled only
for unchanged Float32 source fields and supported computed fields; rich original types do not
silently sort their normalized GPU representation.

Selected original-source export is separate from prepared GPU export. Bounded GPU row-ID pages select
retained CPU Arrow rows only during the explicit export action. Arrow preserves original Float64,
nanosecond timestamps, nulls, dictionaries and geometry; CSV formats exact timestamp text. Prepared
GPU export reports its normalized values and excludes columns that were never uploaded. The export
workspace has explicit memory admission, cancellation and stale-selection guards.
Derived analysis export uses a separate bounded inspector so it does not supersede native table
requests. Result buffers remain leased through export completion. Arrow/JSON/CSV retain original
source IDs, validity and recipe/revision metadata; a selected export records its row selection
separately from the statistical population used to compute the saved analysis.
Combined export can include original and computed columns in the same Arrow or CSV artifact while
retaining original wide-number/time precision, nullable computed values and result recipe metadata.

The native publication workflow and focused numerical/lifecycle checks have passed. Portable bundles
now round-trip multiple datasets through a fresh host session, retaining IDs, roles, Arrow batches,
filters and native table source rows. These are resident-workspace bundles: native-only original
datasets are not embedded, and source-coverage validation rejects unresolved native references.
All CPU source preparation is validated before replacement; this is not a guarantee of atomic GPU
allocation rollback. Actual native field-control and legend browser checks also pass: quantile color,
square-root radius, dynamic domain revisions, categorical colors, analysis overrides and hollow
outlines preserve source upload counts. The pinned native layer validates its own available scales;
for example, its numeric color channel coerces an unsupported linear selection to quantize.
The earlier full-suite
verification below applies to the first resident baseline; it is not a full-suite claim for these new
changes.

## Remaining implementation

The five tranches listed in the earlier version of this section are delivered; see the
2026-10-30 rows under "Subsequent implementation checks" for evidence and per-row limits:

1. Native sequential IPC ingestion covers remote `fetch()` streams, explicitly persisted partial
   populations with manifest coverage, and an explicit refetch-as-new-dataset recovery policy.
2. Native styles cover dynamic Jenks generation and editor seeding, custom breaks and Int32/Uint32
   sources behind a ±2^24 range check, with legends on the displayed revision.
3. Trips cover grouped tabular trips with native per-vertex filtering, table-mode hover, and
   scenegraph markers with textures, PBR and animations. Native has no multipart timed trajectories.
4. Spatial selection covers mixed-union representative points, native linear-degree antimeridian
   semantics and render-only hover brushing for every layer native brushes.
5. Analysis and charts cover Join, Spatial Join, Suitability, linked multi-dataset filters, native
   tick edges, `auto` and equal-width time intervals, all timestamp units, median/deviation
   measures and grouped series. Layer coverage adds S2/A5/geohash, cluster, flow, globe for every
   resident layer family and single-Deck Dual/Swipe split maps.

Open boundaries, each surfaced in the capability catalog rather than silently approximated:

- Measured acceptance: delivered and passing. Targets in `scripts/acceptance-targets.json` were ratified
  by the project owner on 2026-10-31 (scaled from the plan; accounted memory is max(512 MiB, 96 bytes
  per row)). On an Apple M-class Mac (Metal 3, headless Chromium 151, dev server, `--tier reference-desktop`,
  `?memoryBudgetMebibytes=32768`) every target passed at 1M, 5M and 10M rows: 1M filter p95 22.8 ms (plan
  < 50), linked summary p95 18.8 ms (plan < 100), frame p95 12.7 ms (plan < 16.7), zero warm upload; 10M
  filter p95 19.3 ms, brush 157.8 ms, histogram 131.9 ms, style 172.7 ms, linked summary 124.8 ms, frame
  12.7 ms, upload 1.16 s, accounted memory 880 MB of 960 MB. Timings are host-submitted-frame observations,
  and one machine was measured; other hardware tiers have targets but no runs.
- Automatic policies are opt-in (`?automatic=recovery,retry,evict` or `all`; all off by default,
  because recovery creates a device and eviction drops resident data without a user action).
  Delivered: bounded automatic device recovery with backoff and status, and request-level remote
  retry before the explicit refetch policy. Eviction is host-wired: `KeplerDatasetRegistry.setMemoryReserver` routes every
  `reserveMemory` call site (publication, resident tables, accounted readbacks) through
  `reserveMemoryWithEviction` when `?automatic=evict` is set, planning from layer visibility, active
  filters/charts and in-flight publications, with last-use touched on bind and publish. Evicted
  tables rebuild on demand from the recovery checkpoint when a layer or table binding needs them,
  with status in the recovery area. Narrower limits: only datasets with a retained checkpoint source
  are evictable; eviction frees bytes immediately only when no explorer lease holds them; retries
  cover the request, not mid-body failures (a partial body would duplicate rows). `automatic-recovery.mjs`
  verifies eviction, rebuild and automatic device-loss recovery in the browser; retry exhaustion is
  node-tested only.
- Jenks, Join, Spatial Join and Suitability no longer have 262,144-row caps; Join, Spatial Join
  and Suitability results at or below 262,144 rows are read back, validated and adopted as Arrow
  (`adoptArrowDataset`); larger results stay GPU-resident with no full readback. Join matches and orders on the GPU (stable sort, range probe, scans; a 16-byte summary
  is read back before output allocation). Spatial Join joins points in 1,048,576-row chunks on one
  prepared polygon index with fused GPU aggregation, no point, polygon or pair cap. Suitability
  scores on the GPU with per-column GPU statistics (kernel limit 16,777,216 rows). Above 262,144 rows all three adopt
  their output columns with `adoptResidentDataset` (GPU vectors plus a schema, no retained Arrow
  copy); Spatial Join still refuses non-finite aggregates through a four byte GPU flag. The over-cap dataset's `source` is a lazy Arrow view whose cells are null until
  `resident.ensureRows` pages the containing 1,024-row chunk-bounded pages from the GPU for
  tooltips and original-value table pages; the CPU reservation is the ceiling and a GPU min/max
  reduction supplies field domains. Over-cap resident results have no CPU source and no CPU validation beyond the GPU flag, so
  recovery checkpoints record their lineage (operation, input dataset IDs and revisions, request
  parameters, row count) and device recovery and eviction rebuild recompute them from the recovered
  inputs under the same dataset ID; they show null cells in native views that do not call `ensureRows`; source export,
  analysis export, the portable sandbox bundle and DuckDB registration stream the needed rows from
  GPU pages (`materializeResidentRows`, bounded steps over `ensureRows`) into the accounted lazy
  view, while point labels stay limited to 100,000 rows and recovery capture omits resident tables
  (lineage only); at or below the cap none of these limits apply. Jenks is exact at every size: the selected values are read back in accounted chunks into a Web Worker that runs native `naturalBreaks` (about 16 bytes per row reserved; refused only if the reservation fails). Spatial Join sums use compensated Float32, not native doubles. Cluster and Flow no longer have
  row-count caps. Cluster is exact native supercluster at every size: positions, mask and aggregation
  field are read back in 8 MiB accounted slices (512 bytes per row reserved; refused only if the
  reservation fails) and a module Web Worker builds the pinned supercluster index with native
  `getClusterer` options; all native aggregations (including min/max/median) run through native
  `aggregate()`. Positions are float32 from the GPU, so parity is against supercluster fed the same
  float32 values. The former GPU grid path is removed. Flow dedupes locations and aggregates
  origin-destination pairs on the GPU (differs from native, see the Subsequent implementation checks
  row). `check-cluster-native.mjs` asserts exact parity at 300,000 rows; `check-flow-gpu.mjs` matches a
  CPU reference up to 1,000,000 rows.
- Touch is verified under Chromium mobile emulation (`views-touch.mjs`); real iOS Safari and
  Android Chrome checks are a manual checklist (`examples/deck/kepler-compute/TOUCH-CHECKS.md`).
  The exact globe wheel pivot and far-side tile occlusion are browser-verified (`views.mjs`,
  `globe-tiles.mjs`).
Specific contracts established by the pinned-source audit:

- Native styles need independent fill/stroke channels. Point radius defaults to square root;
  numeric colors default to quantile. Native category ordering differs from dictionary insertion
  order, and native logarithmic/extrapolation rules differ from the current color-scale contributor.
  Run color conversion per chunk with shared global domains; use an accounted intermediate for
  global quantiles. Legends must use the same thresholds/palette revision as the rendered buffers.
- Polygon fill, line parts, rings and MultiPoint components have different primitive index spaces.
  Preserve explicit primitive-to-source mappings and gather source masks/styles onto each space.
  Existing polygon tessellation preserves feature identity, but its asynchronous wrapper does not
  itself offload work to a worker. Invoke preparation per source batch to preserve boundaries.
- Native point containment is open-boundary; MultiPoint uses any contained component, while
  Line/Arc requires both endpoints inside. Apply inversion after that row-level reduction.
  GeoJSON uses representative points, not polygon intersection. Its rectangle selection has an
  inclusive-bounds exception; JSON, WKB and Arrow derive representative points differently.
- Trips need exact source timestamps plus rebased per-vertex animation values. The pinned TripLayer
  receives absolute current time and subtracts the animation domain start at its rendering boundary;
  a prepared adapter subtracts its retained origin exactly once. Native trail seconds convert to
  milliseconds. Grouped tabular trips additionally need source-row membership semantics.
- Native heatmap modifies the Gaussian, ramp normalization, texture size and zoom response. The
  existing ground-unit KDE is useful machinery, not an equivalent native heatmap implementation.

## Verification entry points

Use the repository's selected Node version and installed Yarn dependencies:

```sh
nvm use
yarn install
yarn test-node --maxWorkers=2 modules/kepler/test
yarn test-headless --no-coverage --maxWorkers=1 modules/kepler/test
yarn workspace luma.gl-examples-kepler-compute start
```

With the example running on port 5318:

```sh
node examples/deck/kepler-compute/scripts/smoke.mjs
node examples/deck/kepler-compute/scripts/workflows.mjs
node examples/deck/kepler-compute/scripts/sandbox.mjs
node examples/deck/kepler-compute/scripts/layers.mjs
node examples/deck/kepler-compute/scripts/native-table.mjs
node examples/deck/kepler-compute/scripts/publications.mjs
node examples/deck/kepler-compute/scripts/styles.mjs
node examples/deck/kepler-compute/scripts/native-bundle.mjs
node examples/deck/kepler-compute/scripts/native-source-export.mjs
node examples/deck/kepler-compute/scripts/trips.mjs
node examples/deck/kepler-compute/scripts/icons.mjs
node examples/deck/kepler-compute/scripts/original-table.mjs
node examples/deck/kepler-compute/scripts/device-recovery.mjs
```

The first harness checks native filters, histogram values, bounded original-row inspection, visible
map pixels and actual Deck picking. The second checks polygon holes, selected-population z-scores
against a CPU oracle, classification, recipe restore and stale rejection. The third exercises the
capability guide, save/replay, explicit exports, source regeneration and the complementary SQL UI.
The fourth restores native Line, Arc, Grid and Hexagon configurations and checks actual visible GPU
output and filter changes. Its updated connection selection fixture compares both endpoints against
the native polygon predicate.
The native table harness opens Kepler's dataset table, compares paging and ascending/descending
sorting to a CPU oracle, changes selections while the modal is open, inspects derived z-scores and
checks explicit routing of unsupported native export workflows.
The publication harness imports batched Arrow and multipart GeoJSON, checks original source rows in
native tables, and removes datasets. The style harness drives the actual field controls and verifies
visible map colors outside the legend, domain revisions, outline pixels and stable upload counters.
The bundle harness restores multiple resident sources into a fresh session and rejects malformed
roles before replacing the existing workspace.
The Trips harness publishes a null-prefixed, two-batch source and checks native animation-domain
registration, actual play/pause, trail pixels, source picking, filtering, stable uploads and saved
timing policy.
These are correctness fixtures, not throughput benchmarks or a claim about all source sizes.

## Resident baseline verification

October 28, 2026. These results cover the resident integration described above. Independently prepared
source-publication helpers and the next style/geometry tranche are not evidence of completed native
UI coverage. The browser fixtures use the pinned Kepler build and the local WebGPU host.

| Check | Result |
| --- | --- |
| `nvm use` | Passed with Node 22.23.4. |
| `yarn install` | Passed; Yarn reports existing peer-dependency warnings. |
| `yarn lint fix` | Passed; lockfile valid. The baseline pass checked 8,357 files without changes; a later pass checked 8,361 files and formatted two new, unwired prototype files. |
| `yarn build` | Passed across modules, including the CommonJS constructor/device identity checks. |
| `yarn test` with `VITEST_MAX_WORKERS=2` | Passed with exit code zero: 12,799 Node tests (439 skipped), and 12,708 headless tests (846 skipped) across all eight fresh-browser shards. |
| `yarn website:build` | Passed, including validation of 1,210 raw documentation pages. |
| `(cd website && yarn build)` | Passed with the existing `DOCUSAURUS_NO_PERSISTENT_CACHE=1` flag to avoid exhausting local disk with compiler cache; 1,210 raw pages validated. |
| Example `typecheck` and production `build` | Passed. Production bundling reports its large-chunk advisory; bundle optimization remains separate work. |
| `scripts/smoke.mjs` | Passed native range/category/time selection, histogram values, source identity, stable upload accounting, visible pixels/picking, disabled-filter behavior and duplicate-field plot guard/recovery. |
| `scripts/workflows.mjs` | Passed polygon holes and disable/restore, selected-population standardization against CPU values, classification, recipe restore/stale rejection and exact native timestamps. |
| `scripts/sandbox.mjs` | Passed capability-guide controls, native save/replay, explicit selected export, deterministic reimport and the real optional DuckDB registration/query/GPU-upload workflow. |
| `scripts/layers.mjs` | Passed native Line/Arc/Grid/Hex restore, visible GPU output, filter changes and invalid polygon-scope rejection. |
| `scripts/native-table.mjs` | Passed native modal paging/sorting against CPU values, selection changes, derived values and explicit export routing. |

Focused module and shared-package tests supplement these gates, including independent d3 hex-bin
fixtures, native Deck footprint comparisons, signed timestamp endpoints, ownership and device-loss
cases. The workflow fixtures establish correctness at their recorded sizes; they do not establish
the roadmap's million-row latency, streaming, hardware-tier or complete layer-coverage targets.

## Subsequent implementation checks

These are separate checkpoints taken while implementation continues. They supplement the baseline;
they do not imply that the entire new tranche has rerun the baseline's full test and website gates.
Counts below are per check and may overlap. Rows preserve their checkpoint scope; later rows can
close a boundary that was still pending at an earlier checkpoint. The opening table records current
delivery status.

| Check | Result |
| --- | --- |
| Root lint and build checkpoint | Passed: final lint checked 8,429 files with lockfile valid; all module builds and CommonJS identity checks passed with the new property and Trips exports. |
| Focused Node checkpoint | 231 tests passed across 37 files covering Kepler plus Arrow geometry and Trips preparation. |
| Example compilation checkpoint | Typecheck and production build passed after property styles, published analysis/exports, render identity and registration fixes. Production bundling retains its existing large-chunk advisory. |
| Histogram snapshot coherence | Targeted GPU race test and actual smoke passed after requiring a current scope snapshot: 4,096 source rows, 392 selected, exact self-excluding histograms, duplicate-filter recovery, bounded inspection and real Deck picking. Pending native plots no longer publish the primary inclusive histogram as a fallback. |
| Native layer type switching | Final browser regression passed Line/Arc/Grid/Hex rendering and filters, plus same-ID point-to-Line restoration with 1,140 endpoint-selected rows and 47,013 visible colored pixels. Type-specific Deck IDs prevent incompatible model state transfer; explicit registration transactions avoid publishing incomplete selector state during restore. |
| Native identity and registration lifecycle | Final smoke reverified actual Deck picking with physical render IDs mapped to native IDs. Two scope GPU cases passed nested registration, pending-state suppression, real missing-selector rejection, malformed-geometry failure, recovery and fenced cleanup. Renderer errors route to the captured native layer; stale instances cannot disable replacement layers. |
| Native style browser | Passed actual field selection, quantile map colors, square-root radius, bounded legend data, domain revision changes, categorical colors, analysis overrides and hollow outlines with stable source upload bytes. Final layout check also passed an actual hit-test confirming the legend is not covered by the compute panel. |
| Line target colors | 5 GPU tests and 7 focused Node tests passed, including chunked gradients, alpha, warm control changes, selection, picking and borrowed ownership. |
| Arrow geometry representatives | 8 Node tests and 4 host geometry GPU tests passed, including independent pinned-source oracle comparisons, logical slices/nulls, all six homogeneous types and real selection masks. |
| Additional style plans | 21 translator/legend Node tests and 7 host-runtime tests passed for explicit custom thresholds and supported fixed-domain log plans. Example typecheck passed. |
| Published analysis runtime and UI | 7 GPU tests passed, including nullable multi-batch fields, original chunk topology over packed scratch/output storage, selected populations, all standardization methods, exact natural breaks, excess risk and sorted derived table values. Actual UI checks passed dataset/field selection, z-scores, derived table, selected quantiles and pooled excess risk, recipe source IDs and removal fallback. |
| Portable resident workspace | Fresh-session multi-source restore and malformed-role rejection passed; 18 focused Node tests, example typecheck and production build passed. Native-only original datasets remain outside the portable source payload. |
| Selected original-source export | Actual native source/filter/export workflow passed for 2 selected rows of 4: exact Float64 and nanosecond timestamps, retained geometry and original row IDs in Arrow/CSV; prepared JSON independently exposes rounded GPU values and uploaded columns only. Codec, reservation and cancellation Node checks also passed. |
| Derived analysis export | Actual native export passed nullable two-batch excess-risk Arrow with original IDs, selected JSON retaining its all-population recipe, CSV plus metadata, and rejection after analysis was disabled. Codec checks, 18 focused Node tests, example typecheck/build and lint passed. |
| Automatic recovery, retry and eviction policies | 12 Node tests with injected timers passed for backoff caps, abort during wait, transient-vs-permanent HTTP classification, controller request retry, bounded automatic device recovery (exhaustion, reset on ready, no checkpoint, disabled inert), LRU eviction planning that skips visible and borrowed tables, and on-demand rebuild sharing. Existing recovery, capability and stream specs still pass; example `tsc --noEmit` and `yarn build` pass. Policies are off by default; host wiring of eviction has 7 further node tests (`resident-eviction-host.node.spec.ts`). Browser verification (2026-10-30): `scripts/automatic-recovery.mjs` passed with `?automatic=all` on WebGPU: hidden-layer eviction under memory pressure and rebuild from the retained source, automatic recovery after forced device loss (generation 0 to 1, new device and buffers, no manual click), and eviction again after recovery. It found and fixed two bugs (hidden layers re-requesting evicted tables; recovery checkpoints dropping evicted sources). Retry/backoff exhaustion with `?automatic=retry` and `scripts/device-recovery.mjs` were not re-run. |
| Embedded geometry properties | 10 preparation Node tests, 9 host runtime Node tests and 6 combined GPU tests passed, covering property validity, chunk/offset preservation, exact RGBA bytes, same-encoder default changes, field/analysis precedence and borrowed ownership. |
| Direct Trips preparation | Initial 7 Node/NullDevice checks passed for explicit timestamp encodings, exact source retention, rebasing/error budgets, chunked uploads and provenance. Subsequent geometry/property checks passed 17 CPU and 6 GPU cases; counts overlap other rows. |
| Native Trips runtime and UI | Trip gather passed 2 Node and 1 real WebGPU test; resident renderer passed 13 Node checks, native style translation 20 and host style resolution 13. Timing controls/persistence passed 29 focused Node checks. Actual browser passed native playback/domain, 0 to 4,716 trail pixels, original source IDs 1 and 2 across batches [2, 1], scalar filtering, stable source allocations/uploads and saved precision policy. Example typecheck passed. |
| Gallery and combined export | Actual gallery passed six source families and a seven-source portable bundle, including Trips. Combined original/derived Arrow and CSV exports passed browser assertions. |
| Point analysis checkpoint | Geographic kNN weights/rates passed 3 real GPU cases and the published-analysis browser, including fixed topology under selection and recipe restore. Lattice hot-spot UI also passed. Root lint/build, 237 Kepler Node checks and example typecheck passed at this checkpoint; subsequent changes require their own validation. |
| Dynamic logarithmic style primitives | 7 real WebGPU style cases, 9 Node checks, example typecheck and scoped formatting passed. Covered source/selection domains, ten populations, nulls, negative/constant domains, zero-floor descending domains, invalid populations, stale legends and switching back to linear. This checkpoint does not establish every native UI scale control. |
| Native logarithmic width | Actual style browser passed native Line log width with 1,191 selected rows, matching GPU domain/revision and unchanged source uploads. The same run passed existing quantile, square-root, categorical, analysis override and hollow-outline checks; 37 focused host/translator/legend Node tests passed. |
| WKB source selection | 15 focused CPU and 1 real GPU selection checks plus TypeScript passed for six XY/XYZ WKB families, endian/ISO/EWKB encodings, explicit decoded-union provenance, slices/nulls/source rows and native all-vertex means. |
| Icon render primitives | Exact mesh icons passed 7 WebGPU cases including scatterplot regressions, 12 Node tests and module TypeScript. Atlas-icon selection independently passed a real GPU case and 10 Node tests, retaining eight vertex-buffer slots with a borrowed STORAGE mask. Primitive verification alone does not establish the native Icon host workflow. |
| Native Icon host | Actual browser passed two native SVG meshes, 12,432 visible pixels, same-ID point-to-icon restoration, null-prefixed source IDs 1 and 2 across batches [2, 1], filtering/picking and unchanged mesh, gather and source buffers. Source/runtime passed 7 Node and 1 real GPU case; renderer 14 Node checks and translator 21 checks include native high-zoom scaling. |
| Replacement retirement | 22 focused style-host checks passed, including waiting for older Icon/GeoJSON/Trips/property generations before dataset removal releases borrowed source buffers. Example typecheck passed. |
| Resumed package checkpoint | 283 Kepler Node tests across 47 files and example typecheck passed before the final browser/checkpoint tranche. Later changes are recorded separately. |
| Original source table | Actual native browser passed exact Float64/Int64/nanosecond/geometry cells, two batches, source/derived name collisions, filtered original IDs, supported Float32/derived sorting, bounded long-text previews, keyboard wrapping and stable upload/buffer identities. Nine focused Node checks passed. |
| Point hot spots and weights reuse | Four focused GPU checks and the actual published-analysis browser passed geographic point hot spots, native table results and recipe restore. Source-wide directed kNN topology remains fixed under filtering; selection/settings/rate-to-hotspot changes reuse acknowledged weights. Eight focused Node checks passed. |
| Explicit device recovery | Actual browser destroyed the active device, verified stale results disappeared, recovered through the native host button and checked a new Deck device/buffers, original IDs and batches [2, 2], restored native filters and selected z-score recipe. Sixteen focused helper Node tests covered cancellation, late candidates, checkpoint/source replay and cleanup failures. |
| Native integration build checkpoint | Root `yarn lint fix` checked 8,477 files without changes and validated the lockfile; root `yarn build` passed all modules and CommonJS identity checks. Later H3, Charts and analysis changes are separate from this checkpoint; the full test and website gates have not been repeated for those changes. |
| Local Moran host | Two host GPU checks, an independent shared reference case and twelve actual native UI checks passed local I, standardized values, spatial lag, raw/adjusted analytic p-values, quadrants and neighbor counts. Directed source-wide kNN topology is reused under selection. Conditional-normal analytic inference is explicit; permutation, polygon and distance-weights parity remain separate work. |
| H3 implementation in progress | Exact-ID preparation, native topology masks, bounded branch summaries, GPU coverage/style gathers, column/polygon render adapters and source publication are wired. Sixteen renderer Node checks and example TypeScript pass, including native depth, flat shading, elevation/coverage controls and original row identity. Subsequent primitive/runtime and native browser results are recorded below. Float32 coordinate precision is distinct from exact cell-ID precision. |
| H3 primitive and runtime | The mesh-column WebGPU oracle passed five native Deck footprint/lighting modes plus masks, chunked picking and borrowed ownership. Three runtime GPU cases passed bounded topology summaries, stale generation suppression, native dateline/coverage snapshots, gathered styles and fixed-six-vertex viewport footprints. Fourteen source/runtime Node checks passed, including whole-source rejection of collapsed Float32 boundaries before allocation. Actual native H3 browser verification remains separate. |
| H3 and Charts package checkpoint | 341 Kepler Node tests across 58 files and the example production build passed. The initial full-package run exposed a stale mocked module and an incorrect collapsed-cell test fixture; both were corrected and the package rerun passed. This checkpoint does not replace full root tests or establish Charts browser coverage. |
| Native H3 host | Actual browser passed exact Uint64 IDs, source batches [3, 3], null/invalid slots, original picked rows 1 and 3, precise polygon versus viewport-column branches, ordinary GPU range filters preserving topology, category filters changing topology, coverage pixels 3,098 to 491, final-pan footprint refresh and sub-10km cache reuse. Immutable source buffer identities stayed stable. Full geographic fp64 precision and native 3D hover highlighting remain outside this contract. |
| Source-only datasets | Six real GPU cases passed session/explorer admission with zero scalar fields, batches [2, 0, 2], null geometry, spatial masks, bounded ID-only inspection, original Arrow export and fenced cleanup. No synthetic field is added to schemas or exports. |
| Source-only native workflow | Fresh browser passed H3-only publication with six rows across batches [3, 3], zero scalar upload bytes, no synthetic prepared fields, exact original Uint64 cells, cross-batch picking, native Big Number Count without a field, saved replay, unchanged source storage, Arrow export and re-publication. Preview row counts now use the original Arrow source, and pending H3 metadata hides stale output without publishing an unsupported-operation error. |
| Native topology and consumer scopes | Five GPU and nine policy/oracle Node checks passed native allocation history, inverted predicates/null ordering, numeric playback intervals, source-only chunks, chart CPU-ID versus GPU-field exclusions, polygon unions, bounded cache capacity and retirement. |
| Polygon extrusion | Native Deck reference comparison passed hole/multipolygon lighting, ring orientation, top/side picking, source batches, GPU height/color/mask updates and borrowed ownership. Three existing geometry-selection GPU cases also passed. This primitive checkpoint does not establish every native GeoJSON extrusion control. |
| Formatting checkpoint | Root `yarn lint fix` passed 8,512 files without changes and validated the lockfile. Implementation continues; later changes retain their own focused validation. |
| Native Charts host | Actual native browser passed Big Number creation across two unrelated sources, nullable batched count/sum/mean/min/max, selected versus all populations, captions/pinning, unsupported-measure stale-result removal, save/replay and deletion with stable source storage. Four focused GPU cases include source-only count. Other chart families and resident hover summaries remain explicitly unsupported. |
| Combined module build | Root `yarn build` passed every module and CommonJS identity checks after H3 rendering, source-only datasets, consumer scopes, extrusion and the initial IDW adapter landed. Example production build and the 341-test package checkpoint passed separately. Full root `yarn test` and website builds have not been repeated for this tranche. |
| Additional analysis native workflow | Combined browser passed Quantile LISA controls, explicit legacy inference, bounded table results and replay; IDW raster pixel changes, bounded cell inspection and recipe/off/restore; and DBSCAN noise/cluster display, source-row tables and replay. The IDW host GPU case passed independently. These bounded workflows retain their declared metrics, capacities and inference policies; they do not establish all spatial-analysis modes. |
| Native operation ownership | Eight focused Node checks passed resident Group By/Join/Spatial Join/Suitability rejection before native CPU execution, ordinary-source preservation and single-view split/swipe/restore guards. These explicit boundaries remain until the corresponding resident/complementary execution adapters or per-view ownership exist. |
| GeoJSON vertical integration | Source preparation, property fallbacks, field styles, GPU gathers and renderer bindings now preserve independent height/base-offset semantics, XYZ, offset-only modes, native draw ordering/depth and wireframe-only rendering. Four primitive GPU cases, eight property/style GPU cases and seventeen renderer Node checks passed; source/host checks additionally cover finite zero/negative offsets, truthy height defaults, chunks and ownership. A full native vertical-control browser workflow remains separate. |
| Fixed-height inversion | Three scalar GPU cases and four native/d3 oracle Node cases passed identity inversion against GPU domains, including nulls, chunks, descending/equal domains, extrapolation and extreme Float32 endpoints. Normalizing both interpolation inputs and endpoints avoids overflow from GPU reassociation without CPU row reads. |
| Geographic DBSCAN host | One focused GPU host case passed antimeridian distances against an independent oracle, cluster/noise/excluded rows, categorical colors, selected-population rebuild, source-row inspection and ownership. The combined browser workflow above supplies separate UI/replay evidence. |
| Native categorical Charts | Eleven actual browser checks passed native bars, nullable multi-batch measures, native category ordering/ties, own-filter exclusion, unrelated range preservation, keyboard toggle, top-N/Other, all-source population, replay, stable source storage and removal. Numeric histograms remain a subsequent implementation slice. |
| Native workflow UI | Actual desktop/mobile browser passed full map width, native Layers analysis, Add Data preparation and SQL, bounded tooltip values honoring saved fields, controlled dataset selection across tabs, original-source export, dialog closing after download, native save/replay and 390px mobile controls. The separate Compute panel is removed. Example typecheck and production build passed; the existing large-chunk advisory remains. Exact Select Geometry is a separate adapter. |
| Vertical and native UI checkpoint | Root `yarn lint fix` checked 8,527 files, formatted nine and validated the lockfile. Root `yarn build` passed all modules and CommonJS identity checks; all 372 Kepler Node tests across 64 files passed. Subsequent UI fixes received scoped formatting/types. Later path, numeric chart, publication aggregation and geometry-selection work is not covered by this checkpoint. Full root tests and website builds have not been repeated. |
| IDW GeoTIFF export | One focused host GPU test and the migrated combined native analysis browser passed. The downloaded raster was independently decoded as EPSG:4326 Float32 with expected 8×8 cells, source/recipe metadata and values. Export retains the result snapshot through file preparation and rejects stale grids; the UI exposes the bounded raster contract. Quantile LISA, IDW and DBSCAN controls/replay also passed in the native layout. |
| H3 native extrusion and restore | Extended browser passed both polygon and viewport-column extrusion, picked source IDs, stable uploads, pan/cache behavior and visible coverage changes (3,099 to 490 pixels). Restoring a configuration with no targeted filters now explicitly refreshes each resident table before registration commits; the fixture verifies all six rows return before switching back to a two-row category subset. This fixes a retained prior mask when native Kepler skips filtering a source with no remaining targeted filters. |
| Native numeric Charts | Two numeric GPU cases and twelve native browser checks passed nullable batches, inclusive range clicks, self-filter exclusion, unrelated-filter domains/empty bins, bounded measures, horizontal replay, unchanged source buffers and exact Uint32 selections despite compact-label collisions. The UI uses explicit Numeric intervals mode; default integer-auto grouping, time buckets and multi-series remain separate work. Category/scalar GPU regressions also passed. |
| Explicit native Jenks thresholds | Sixty-five focused Node tests and module/test types passed revision-bound externally supplied thresholds, ties, precision/capacity rejection and bounded legends. The installed randomized Jenks oracle differs from exact Fisher–Jenks; dynamic native threshold generation remains unsupported. The host API accepts explicit controls but does not yet expose a threshold editor. |
| Heatmap and regression build checkpoint | Root `yarn lint fix` passed 8,554 files, formatted one and validated the lockfile. Root `yarn build` passed all modules and CommonJS identity checks, including the new bounded OLS adapter. This establishes compilation, not pending Heatmap/OLS native GPU workflows. |
| Package import checkpoint | The 70-file Kepler Node run passed 395 tests; two suites timed out during module imports before running their eleven tests. Those eleven passed on a focused serial retry. Moving their module resets to hoisted setup and using static imports then passed a six-suite shared-worker reproduction with 38 tests, without increasing timeouts. The entire package has not yet been rerun after that harness repair. |
| Native geometry metadata | Native GeoJSON initialization/restore now obtains feature types from prepared bindings, exposing the missing Height, Fill and Radius controls without source scans. The browser reached the real Height dropdown; the full vertical workflow remains pending. Other resident layer formatting paths likewise refresh prepared bounds. Renderer metadata/source-access checks pass independently. |
| Published Grid and Hex aggregation | Three runtime GPU cases and the actual native browser passed custom coordinate roles, source batches [2, 0, 3], nullable weights, count/mean/sum, bounded twelve-byte cell inspection, source ranges, native polygon targeting one Grid independently of Hex, replay and stable source/projected buffers. Runtime cases additionally cover nullable coordinates and stale/pending reads. The existing centroid-based Mercator meter projection and supported measure/palette subset remain explicit; this is not full native aggregation parity. |
| Native path joins and Trips timing | Four GPU reference cases, two Node tests and module types passed. All 24 non-antialiased comparisons were byte-identical to native Deck across XY/XYZ/XYZM and pitched geographic rendering. Optional antialiasing retains at most two fringe silhouette differences, at most 64/255 channel error, less than one basis point full-image error and less than five basis points painted-union error in these fixtures; measured painted-union maximum was about 0.0301%. Both Trips fade modes, closed-cycle timing, masks, source picking and borrowed ownership passed without source readback. |
| OLS and Local Moran permutations | Five GPU tests across three files and the combined native analysis browser passed. OLS covers singular-fit publication gating, ordinary full/selected fits, ridge closed-form values with suppressed unsupported inference, source-row tables and replay. Local Moran covers analytic and explicitly seeded conditional two-sided permutation inference, controls and recipe replay, with an independent shared reference. Earlier analysis, GeoTIFF and DBSCAN workflows also passed in that browser run. |
| Source-only cross-dataset filters | Two focused Node tests and example types passed a real session/bridge/provider regression: foreign GeoJSON polygons, own H3 polygons and clearing on a zero-scalar source preserve chunks [2, 1] and independent scopes. Empty scalar plans no longer instantiate a scalar selector that requires a source row-ID vector. Native Select Geometry browser retry remains separate. |
| Resident heatmap primitives and weights | Two GPU oracle cases passed byte-identical native SUM/MEAN density, maximum and final RGBA after aligning the reference viewport, including null weights, batches [2, 0, 2], selection revisions, empty selections, buffer/model/palette reuse and borrowed ownership. A separate host GPU case passed native linear/square-root/log weight normalization over DOMAIN, extrapolation and stable output buffers without source transfers. Five readiness tests cover pending pipelines and draw retries. Native heatmap UI acceptance remains separate. |
| Integer and line Charts | Five focused GPU cases and the metadata-corrected native browser passed sixteen checks: default integer categories and native ranked interval fallback, exact Uint32 singleton/interval clicks, nullable numeric measures/filter scopes, empty bins, horizontal replay, single-series line replay and point selection, stable buffers and removal. Prepared Arrow payloads now preserve authoritative numeric/time/string types through the native validator hook, avoiding sampled Uint32-to-timestamp and Float32-to-integer reclassification. Ordinary native inputs retain their validator. |
| Native Select Geometry and editor | Actual browser passed GeoJSON original properties/geometry, canonical H3 geometry independent of rendered coverage, source rows/chunks, fresh editor IDs, frozen tooltip actions and targeted editor filters without page errors. The pinned editor now uses asynchronous WebGPU picking with native event replay, stale-event guards and bounded queues; nine CPU checks include terminal drag cleanup on overflow. H3 browser/Node boundary comparisons allow 1e-12 degree floating-point differences while retaining exact topology and identity assertions. |
| Geographic distance-band weights | Two GPU cases and native controls/capacity refusal/recovery/replay passed explicit great-circle radius topology, including antimeridian cases and fixed source-wide membership under selection. Source size is capped at 65,536 so uint32 neighbor totals remain representable; explicit neighbor capacity and permutation-degree overflow reject incomplete results. kNN defaults remain unchanged. |
| Native map image export | Actual browser passed real 1200×900 PNG download, basemap and resident pixels, exact preview/download equality, transparent alpha, native legends, read-only chart content, 2× output at 2400×1800, 16:9 output and cancellation. Saved configuration, device identity/generation, original source Buffer object identities and source uploads stayed unchanged. Capture reuses the live map/device, honors the native export legend portal, and prevents export resize events from mutating the live viewport. Explicit capture limits remain 8,192 pixels per side, 16 megapixels and a 30-second timeout. |
| Polygon analysis weights | One focused GPU case and the native polygon browser passed exact original-XY queen/shared-whole-edge rook topology, capacity refusal/recovery, selected populations, cache/chunk/source identity, rates, hot spots and analytic/conditional Local Moran. The native workflow also passed multi-batch publication, bounded derived tables, recipe restore and source removal. Retained Float64 polygon descriptors are separately budgeted. Whole-edge rook semantics do not implicitly node split/overlapping segments, and rounded rendering coordinates are never substituted for original topology. |
| Chunk-preserving stream backend | One focused WebGPU case passed batches [2, 0, 3], unchanged earlier buffers, stable dictionary codes/nulls/exact timestamp sidecars, explicit preview admission, stale-prefix rejection, completed filtering/paging and fenced cleanup. Ten stream and fourteen session Node checks cover bounded admission, completion/cancellation and completed-stream recovery, including failed recovery/retry. Seventy focused registry/preview/save/recovery checks cover complete-population restrictions. Incomplete prefixes cannot be silently saved as finite datasets; the last valid complete checkpoint is retained. Native streaming ingestion remains a separate integration slice. |
| Stream, polygon and image checkpoint | Root `yarn lint fix` checked 8,587 files without changes and validated the lockfile. Root `yarn build` passed every module and CommonJS identity checks. All 466 Kepler Node tests across 78 files passed, including the repaired import suites. This checkpoint precedes native stream ingestion, projected-coordinate analysis and multi-series chart implementation; full root tests and website builds have not been repeated. |
| Heatmap compute ordering | Six CPU checks and two updated strict GPU oracles passed after moving density execution from layer update into the effect after selection and normalized-weight encoding. The oracles begin with zero masks and weights and verify native SUM/MEAN density, maximum and final pixels only after ordered updates. Readiness waits for current density and an actual display draw. The native UI workflow remains pending. |
| Native Heatmap workflow | Actual browser passed constant density pixels, the native Weight dropdown, nullable batches [2, 0, 3], native SUM/MEAN and radius, linear/sqrt/log normalization with reused outputs, independent display/dynamic-domain selections and empty-filter replay. Original source Buffer identities and upload bytes stayed unchanged; no page or GPU errors occurred. Visible density areas were checked in the final screenshot. |
| Polygon Quantile LISA | Three consolidated point/polygon GPU cases and the native polygon browser passed Quantile LISA controls, geometry-valid population handling, selected quantiles, explicit legacy inference settings and recipe restoration. Exact original-XY topology and existing polygon capacity/island policies remain unchanged. |
| Grouped chart summaries | Six focused GPU cases, fifteen focused Node checks and twelve native browser checks passed bounded dictionary series over category, integer and interval bins, native sum-of-per-series-statistics semantics, nullable labels/counts, local series order, self-filter exclusion, saved replay, aggregate lines, horizontal intervals and removal. Source chunks and Buffer identities stayed unchanged. The UI exposes a bounded series breakdown; it does not claim separate-series plotting. |
| Native GeoJSON vertical workflow | Seven native checks passed null-prefixed XYZ/holed polygon picking, the actual Height dropdown independent of line width, mapped/fixed/inverted height pixels, offset-only zero/negative bases with zero extrusion height, wireframe-only rendering, in-place selection masks and exact source rows, and native Save map/replay. Original source Buffer object identities stayed unchanged. Fixture pixel counts distinguished mapped (79,549), fixed (59,650), inverted (56,973), filled (67,887) and wireframe (8,572) output; these are correctness fixtures, not performance measurements. |
| Native Arrow IPC streaming | Six focused controller Node tests and eight native browser checks passed first-batch pause, bounded incomplete preview, preserved batches [2, 0, 3], explicit completion at EOF, completed dataset publication, nullable range filtering and cancellation of a second import. Successful native publication closes the dialog without disposing the transferred dataset. Completion/handoff allocates no new GPU buffers in the controller test; the host attaches the existing complete snapshot, and browser source IDs stay stable through filtering and panel reopening. Preview decoding is bounded before reading rich cells, and reader errors cannot skip publisher cleanup. |
| Stream, series and projected-coordinate checkpoint | Root `yarn lint fix` checked 8,600 files without changes and validated the lockfile; all module builds and CommonJS identity checks passed. All 482 Kepler Node tests across 82 files passed. Later calendar, text labels, cartogram and resource-accounting changes are outside this checkpoint; full root tests and website gates have not been repeated. |
| Projected point analysis | Five focused GPU cases across three files and the native projected workflow passed explicit meter-space XY, CRS/bounds, nullable multi-batch coordinates, source-wide kNN/radius reuse and geographic regressions. Native table-only publication, projected rates, coordinate controls/recipe restore, original X=181 and null-row inspection, and removal passed. Projected coordinates are not inferred to be longitude/latitude, and no automatic reprojection occurs. |
| First resident performance smoke | The 100,000-row report (local research log, not checked in) records five measured interactions after one warmup on Apple M3 Pro / headless Chromium 151 WebGPU. Median dispatch-to-submitted-frame was 35.8 ms; linked Count observation was 54.5 ms. Source buffers were unchanged, registered source upload delta was zero and exposed explorer readback was 20 bytes. These are host/polling observations, not GPU execution or physical presentation, and five samples do not establish a latency target. |
| Calendar and null-origin filtering | Two calendar GPU cases and fifteen native browser checks passed exact millisecond UTC month/year grouping, time-zone-neutral buckets, null-plus-zero epoch grouping and inclusive clicks, arbitrary persisted filter IDs, self-filter exclusion, series, source chunks and unchanged buffers. Removing chart ownership restores ordinary time-slider null exclusion; restoring it includes actual nulls only when the chart interval contains epoch zero. Valid timestamps outside the JavaScript Date range never become epoch values. Six scalar GPU cases and a native scope regression independently cover the trusted filter-origin policy. Native time-filter histogram metadata remains a separate gap. |
| Resource planning and memory admission | The graph dry-run planner passes nine focused CPU cases plus existing alias/description regressions without allocating resources or compiling executables. Thirty-one registry/scope admission checks and six combined session/scope GPU cases pass atomic growth, preallocation checks, rollback and fenced retirement. Explorer reservations retain planned resident storage plus bounded inspection capacity after construction, while their original hard limits remain unchanged. Command-encoder-owned control staging, executable-private allocations and driver memory remain explicitly outside complete device-memory accounting. |
| Label, calendar and resource checkpoint | Root formatting checked 8,639 files and formatted four; lockfile validation and every module/CommonJS identity build passed. The package run exposed a missing time-columns test fixture and shared-worker mock isolation; after fixing those tests, all 526 Kepler Node tests across 94 files passed. Text label image parity and Dorling acceptance remain separate pending work; full root tests and website gates have not been repeated. |
| One-million-row admission and performance | The 1m report (local research log, not checked in) passed under the unchanged 512 MiB budget with 16 original batches, 12 MB source uploads and 297,350,340 accounted bytes after warmup. Device/source buffer identities and batches stayed unchanged; measured source upload delta was zero and explorer readback was 20 bytes. Across five measured interactions, submitted-frame p50/p95 was 93.5/120.7 ms, next-animation-frame observation 94.0/143.5 ms and linked Count 159.0/179.3 ms. These observations exceed the proposed latency targets and do not measure GPU execution or physical presentation. |
| Separate-series and second-time charts | Five focused GPU cases, thirteen native series checks and fifteen calendar checks for each of milliseconds and seconds passed. Actual bars, horizontal bars and lines preserve stable series identities, negative values, zeros, gaps, parent-bin keyboard/click filtering and saved replay without source reuploads. Microsecond/nanosecond product admission remains gated on coherent native grouping and exact click semantics. |
| Dorling native cartogram | The bounded host GPU case and native workflow passed multi-batch preparation, finite exhausted layouts, saved recipe restore, original-row table/hover/click inspection and removal. The EPSG:3857 overlay matched all 89,628 predicted interior pixels, with no changes among 95,386 outside pixels. Arbitrary declared CRS remains table-only; projected coordinates are never silently treated as longitude/latitude. Camera readiness observes submitted frames instead of requiring redundant compute encodes. |
| Point label primitive | Six source/runtime Node checks, atlas cache isolation and two native bitmap GPU reference cases passed, including six anchor/baseline/size settings and pitched geographic output. Private mipmapped atlases, original chunks, masks and fenced ownership are covered. The native browser workflow passed actual field selection, visible glyphs, label-only markers, same-layer size/anchor changes with identical glyph buffers, null/filter hide and restore, field-replacement retirement and label removal. Scope is bounded one-line scalar point labels, without outlines, backgrounds, collision or formatted date labels. |
| Native time histogram computation | Four focused GPU cases passed exact signed timestamp classification, greater-than-2^53 microsecond edge neighbors, empty chunks, nullable epoch grouping, self-filter exclusion, interval changes, stale source retirement and bounded readback. More than 256 intervals reports an explicit unsupported plot while valid selected counts remain available. All nine native browser checks passed actual Month/Year interval controls, the pinned native edge/count oracle including null epoch bins, self-filter exclusion with narrowed map selection, unrelated rank filters, visible native SVG bars, saved interval replay and unchanged source revision, uploads, buffers and batches [3, 5]. |
| Tabular point altitude boundary | Twenty renderer Node checks pass matching XYZ admission, mismatched-column rejection, source-mask borrowing, native altitude depth behavior and restoration to XY when altitude is removed. Source preparation retains null Arrow altitude as render Z=0 and rejects nonfinite altitude through a separate validity mask. Two actual WebGPU helper/mask cases pass chunk preservation, source-mask independence, in-place updates and fenced teardown. All five native workflow checks passed pitched XYZ picking, zero/negative/null and nonfinite altitude behavior, altitude toggles with stable XY/XYZ buffers, filtering, portable replay and completed-stream altitude handoff. The first browser attempt stopped on an exact-label fixture lookup; the corrected native control locator passed without runtime changes. |
| Altitude, labels and time-histogram checkpoint | Root `yarn lint fix` checked 8,654 files without changes and validated the lockfile. Every module build and CommonJS identity check passed, as did example TypeScript. The 548-test package run exposed a legend factory mock leaking between shared-worker suites; scoped module/mock isolation fixed it, and all 548 Kepler Node tests across 97 files passed. Subsequent kriging implementation and queued altitude/calendar/histogram browser acceptance are outside this checkpoint; full root tests and website gates have not been repeated. |
| UTC day/week charts | Five focused GPU cases and twenty native browser checks for each of milliseconds and seconds passed exact calendar grouping, signed epoch boundaries, Sunday weeks, null-origin ownership and filtering, bounded bins and stable source allocations. Thirteen native separate-series checks also passed after the weekly-label layout change; inspected labels retain full distinct dates and missing-value gaps. The initial browser mismatch was a fixture expectation of two decimals where pinned native formatting emits three; no runtime workaround was introduced. |
| October 30 integration checkpoint | Root `yarn lint fix` checked 8,660 files without changes and validated the lockfile. All module builds and CommonJS identity checks passed after the kriging integration and IDW compatibility fix. Example TypeScript and all 554 Kepler Node tests across 99 files passed. Hourly chart admission remains rejected; its unfinished GPU fixture is explicitly outside test discovery and is not verification evidence. Final native retries are recorded separately; the full root test and website gates have not been repeated. |
| Ordinary kriging native workflow | The focused kriging and unchanged IDW GPU cases both passed. The native workflow passed manual variogram configuration, exact-hit prediction and zero variance, positive off-sample variance, separate prediction/variance GeoTIFF downloads with independent recipe/band decoding, visible bitmap output, recipe replay, unchanged source revision/uploads/buffers/chunks and removal. Bounds remain 4,096 source rows, 16,384 grid cells and at most 16 neighbors, with explicit geographic coordinates and manual variograms; there is no automatic fitting or cross-validation claim. Writable variance imports share one graph view, graph construction is inside reservation cleanup, scratch is admitted before compilation, and the existing IDW cell payload is preserved. |
| Final calendar requalification | After the isolated hourly-key preparation changed shared helper ownership, the existing five GPU calendar cases and the default native browser workflows passed again: twenty checks for milliseconds and twenty for seconds. No hourly fixture ran; hourly product admission remains rejected. This establishes current-code day/week/month/year coverage after the final build and formatting checkpoint. |
| Hourly charts and same-ID retirement | Ten focused GPU cases passed hourly/calendar computation and the real scope-mask replay regression. The new provider barrier waits for old same-ID readers and fences before constructing a replacement, with currentness rechecked after retirement. Six focused lifecycle CPU checks pass and fail without the fix. Public native hour workflows passed twenty-five checks each for seconds and milliseconds, including previous intervals, exact null-owned ranges, day-to-saved-hour replay, same-ID removal/readdition and unchanged source buffers. Test-only enablement was removed before these public checks. This supersedes earlier gated-hour status; the shared post-change build checkpoint remains pending. |
| Parallel tranche, 2026-10-30 | Implemented ahead of verification, then requalified together. Final gates: `yarn lint fix` clean (8,719 files, lockfile valid); `yarn build` passed every module and CommonJS identity check; `yarn test-node` 13,394 passed with 3 failures confined to the separate, pre-existing uncommitted `GPUScalarScale` documentation work; `yarn test-headless` for `modules/kepler`, `modules/deck-gpu-layers` and Arrow Trips passed 177 tests in 80 files; 29 example browser workflows passed against the dev server, including the new Join, Suitability, split-view and grouped-Trips scripts. Requalification fixed a reserved WGSL identifier in the globe module, a stale-summary read livelock after polygon re-enable, non-plain dictionary objects in Join/Suitability checkpoints, non-resident grouping role typing, a React import in the time-bar GPU path, a split-view capture loop and Deck-to-Kepler layer-id mapping. Rows below keep their implementation scope. Final gates after the cap-removal, recovery, chart, view and acceptance tranche: `yarn lint fix` clean; `yarn build` passed; `yarn test` node 13,550 passed with no failures; headless shards 1-4, 6 and 7 passed in the full run and shards 5 (1,862 tests) and 8 (1,398 tests) passed when rerun alone after intermittent `VITEST_BROWSER_CONNECTION_CLOSED` browser disconnects on unrelated, individually passing specs (no assertion failures in any run); all 69 example browser workflows passed; `yarn website:build` and `(cd website && yarn build)` passed. |
| Native heatmap UI workflow | `scripts/heatmap.mjs` passed: batches [2, 0, 3], custom coordinate roles, nullable weight normalization through the native dropdown, SUM/MEAN with radius and intensity, linear/sqrt/log reuse, independent display and DOMAIN scopes with empty-filter replay. This closes native heatmap UI acceptance. |
| Dorling native acceptance | `scripts/cartogram-analysis.mjs` passed: multibatch projected publication, zero-iteration overlap layout, recipe replay, 89,628 of 89,628 interior pixels changed and 0 of 95,386 outside, real hover/click pinning, table inspection and removal. This closes Dorling native acceptance. |
| Point label workflow and parity | `scripts/point-labels.mjs` passed before a canvas-reference ink-extent assertion (12% width, 20% height) was added; the new assertion has not run. A same-page native Deck TextLayer reference is impossible under WebGPU Deck. |
| Native spatial selection edges | Native uses turf `booleanWithin` on raw degrees, so the new default `native-linear-degrees` space accepts antimeridian-crossing and out-of-range polygons instead of rejecting them. Unlabelled dense-union GeoArrow follows native no-centroid semantics through a `no-representative` layer kind. 20 Node tests passed; the new GPU case is unrun. |
| Native hover brushing | Render-only WGSL port of Deck BrushingExtension (radius = size km × 1000 m, common-space distance, `source` vs `source_target`) for point, line, arc, icon, H3 column, point labels and arrowheads; GeoJSON, trips, aggregates and heatmap ignore the brush as native does. The H3 polygon branch brushes per vertex (see the H3 polygon brushing row). Node specs and module typecheck pass. |
| Trip markers | `GPUTripMarkerLayer` draws the native Trips scenegraph (gallery or custom glTF/GLB, loaded once by loaders.gl and flattened through node transforms) as one instanced marker per trip. The vertex stage finds the current segment with a binary search over the resident retained-origin timestamps, interpolates the position, and derives the native Web Mercator speed-vector yaw (`interpolateAngle` across the two latest segments); native `sizeScale`, adjust angles, model angles/scale and color are applied. Filtered grouped-trip path copies and the selection mask are consumed unchanged, so there is no CPU per-frame work. `scripts/trip-markers.mjs` passed: markers at both trips, moving with animation time, east-bound heading, hidden after trip end, filtered trip hides its marker; `trips.mjs`, `trips-grouped.mjs`, `icons.mjs` and `smoke.mjs` still pass. Gaps: textures, PBR, model animations, roll/pitch/yaw datum fields, table-mode per-timestamp colors, globe; north-bound heading is unverified in the browser. Native multipart (MultiLineString) timed trips do not exist upstream, so none are provided. |
| H3 polygon brushing | `GPUSolidPolygonLayer` and `GPUPathLayer` write a smooth per-vertex brushing varying from the shared `gpuBrushing` module (radius = size km x 1000 m, render-only) and discard below 0.5, like Deck's `brushing_isVisible`; the resident H3 polygon branch passes the brush to its fill and outline layers and the refusal is removed. `scripts/h3.mjs` passed: only cells with a vertex inside the radius draw, moving the pointer swaps them, selection counts are unchanged. Extruded walls use the same varying; GeoJSON polygons remain unbrushed as native does. |
| Dynamic native Jenks, editor seeding and integer styles | Thresholds come from pinned `@kepler.gl/utils` `naturalBreaks` over one bounded, CPU-reserved readback per source/field/class/selection revision (no row cap; exact at every size via a Worker); the table bridge returns them to Kepler's `getColumnLayerDomain`. Custom-break edits produce a new revision-matched plan. Int32/Uint32 style columns are widened once on the GPU after a chunk-wise ±2^24 extent check. 54 Node tests passed. |
| Calendar units, automatic interval and time bars | Line charts accept second through nanosecond Timestamp/Date sources through the exact Int64 normalizer. Unset intervals run native `auto` (GPU Int64 extent, 200-bucket span rule, key rebuild only on interval change); literal `"auto"` gives native UTC days with date-time labels; bar charts on time axes use 2–512 equal-width bins. 11 calendar specs compare against native `computeDatasetChart`. Over-512-period charts are rejected. |
| Linked filters across datasets | Per-dataset resident plot entries with edges from the merged filter domain; native computes CPU datasets; pending entries clear to undefined. 25 bridge specs passed, including a resident/resident/CPU case. Linked non-histogram plots and the field-mapping UI remain. |
| Native Arrow IPC remote and partial ingestion | Remote `fetch()` streams with abort, credential-free fetch, content-length and chunk caps; confirmed `save-incomplete-prefix` records `population.partial`, which bundle restore validates and preserves. Partial datasets are labelled in pickers and bundle messages. Recovery never refetches silently; an explicit Refetch adds a new dataset. 55 Node tests passed. |
| Grouped tabular Trips | Explicit ID/time/lon/lat grouping across batches with an accounted CSR member index; any selected member shows a trip, and unselected vertices collapse to zero-length segments on the GPU to match native shortened paths. Source panel, saved state, native `columnMode: 'table'` and capabilities are wired. 15 Node tests passed; `host-tabular-trips.spec.ts` and `scripts/trips-grouped.mjs` are unrun. |
| Native Join and Spatial Join | Bounded executors publishing table-only results with lineage in portable bundles. Join: LEFT/INNER/FULL on string-dictionary, Int32/Uint32 or integral Float32 keys, ≤262,144 rows per side. Spatial Join: resident points × polygons, `intersects` (closed) and `within` (open), Count/Sum/Average/Min/Max on the GPU plus the panel's default Merge of the polygon geometry column as a bounded CPU step (matched pairs read back and counted, ≤262,144 left rows and ≤4,194,304 pairs, native `mergeGeometries` semantics, GeoJSON Feature string cells; larger inputs are refused with an explicit uncheck-merge message), ≤4,096 polygons and 1,048,576 pairs, explicit Float32 boundary policy. Native-parity specs including merge pass and `check-spatial-join.mjs` runs the default merge-checked flow; other predicates and aggregations (Median, Count unique) route to SQL guidance. |
| Cap removal: Jenks, Join, Spatial Join, Suitability | Browser verification (2026-10-30): `scripts/check-spatial-join.mjs`, `scripts/check-resident-overcap.mjs` (300,000-row Spatial Join, Join and Suitability adopted resident, `ensureRows` pages match CPU references) and `scripts/check-jenks-gpu.mjs` (300,000 rows, breaks deep-equal native `naturalBreaks`, masked path too) pass on WebGPU. Bugs found and fixed: borrowed DynamicBuffer passed as GPUBuffer in Spatial Join copies, reserved WGSL identifier `target`, duplicate nonfinite flag import, and an undersized domain-reduction memory budget (now `getResidentDomainBudgetBytes`). Not covered: native table/tooltip UI for over-cap rows, the Jenks validity binding. Implemented: Implemented and written: GPU Join (`resident-join-gpu-kernels.ts`), chunked Spatial Join with fused aggregation (`resident-spatial-join-aggregate.ts`), GPU Suitability (`resident-suitability-kernels.ts`) and exact Jenks at every size (`native-jenks-worker-client.ts`, Worker running native `naturalBreaks`). Resident adoption (`adoptResidentDataset`, lazy row paging) has node coverage with fake buffers only. Publication is size-gated at 262,144 result rows (`resident-result-readback.ts`): at or below it Join, Spatial Join and Suitability read results back, validate them on the CPU (validity 0 or 1, dictionary code range, Spatial Join non-finite aggregates) and adopt with `adoptArrowDataset` (full cells, device-recovery checkpoint, DuckDB/export/labels/bundle); above it they adopt GPU-resident (`adoptResidentDataset`) with no result readback, and Spatial Join refuses non-finite aggregates with a four byte GPU flag (`resident-nonfinite-check.ts`). Lazy row paging and the readback table builder have node coverage (fake buffers and pure words) with no GPU/browser run. Node checks only: CPU mirrors compared with the oracles (Join 19 specs, Spatial Join 10, Suitability 13, Jenks 14); the WGSL is unexercised. Browser verification (2026-10-31): `check-resident-overcap.mjs` now covers over-cap lineage recovery for Join, Spatial Join and Suitability (`<kind>-recovery-explicit` button recovery, `<kind>-recovery` with `?automatic=all` plus eviction and rebuild; sampled rows match the CPU references across the 262,144 boundary) and analysis export (`analysis-export`: z-score on a 300,000-row Suitability result, analysis Arrow, combined Arrow and combined CSV stream every row with null scores and source row IDs intact). No app bugs found; the check script's `table` mode ordering was fixed (it inspects the last dataset). Running every over-cap result in one page exceeds the 512 MiB accounted budget (`Dataset memory budget exceeded`), so the default run restarts the page before analysis export and before the automatic-recovery phase. |
| Resident DGGS layers (S2, A5, geohash) | Exact cell IDs decode once at publication into a per-batch `dggs_footprint` WKB column; native `s2`, `a5` and `geohash` layers (`token` column) render through the resident GeoJSON path with styles, masks, picking and selection; the manifest records `cellIndex`. All six S2 faces match `s2-geometry`; A5 above 2^53 is exact. |
| Resident cluster layer (superseded for large sources by the GPU row below) | Explicit CPU-readback adapter using pinned supercluster 7.1.5 with native options, `Math.round(zoom)` queries and `minClusterSize`, capped at 262,144 rows. Count or Kepler `aggregate()` colors with quantize/quantile/ordinal/jenks/custom scales; hover lists up to 8 members. 8 specs match native `ClusterBuilder` at 5 zooms with a filter mask. |
| Arrowheads, hover highlight and globe | `GPUArrowHeadLayer` replaces the arrowhead rejection; fragment-stage highlight follows native `highlightColor` and `autoHighlight = !brush.enabled`, with whole-feature GeoJSON ranges. Globe is enabled for points, great-circle arcs and lines; GeoJSON, trips, icons, H3 and arrowheads are rejected on globe. The horizon point rule is local, not Deck's. |
| Flag-gated Dual split map | `?views=2` renders Dual split in one Deck with two viewports on the shared device; resident layers draw once and `layerFilter` admits them per native per-map visibility. A second Deck cannot share the device's single default canvas context. Unflagged behavior and swipe rejection are unchanged. |
| Tabular Trips gaps | Table-mode hover follows native `_findDatumForFeatureByTime` (`bisectRight` over unfiltered retained timestamps, no readback); `planKeplerTripAnimation` accepts table mode; native table mode has no embedded line style properties, so styles gather from the primary row. A grouped `trip-observations` sample is added. |
| Flow map (readback adapter; superseded by the GPU row below) | Native location identity, clustering, totals and picking come from the pinned FlowmapLayer and `@flowmap.gl/data` (loaded lazily); resident rows and a selection mask read back once per revision (capped at 2,000,000 rows) feed it. Roles are wired through the source panel, saved state and native layer config. 3 Node tests pass with the library stubbed; H3 flow mode and GPU-resident reduction remain open. |
| Native Suitability | Native `RUN_SUITABILITY` on a resident source runs a bounded `GPUCompositeScore` executor (min-max or z-score, normalized or raw weights, ≤16 Float32/Int32/Uint32 fields, ≤262,144 rows) and publishes a separate table-only result with native schema, naming, null scores and lineage in portable bundles. Raw standardization, invalid weights, constant columns and differently-missing fields are rejected. Node tests compare the CPU oracle path to pinned native `suitabilityDataset`; `scripts/check-native-suitability.mjs` is unrun. |
| GPU-resident Cluster and Flow (browser-verified 2026-10-30) | Cluster (superseded: now exact native supercluster at every size via a Worker, GPU grid removed; the description below is historical): sources within 262,144 rows keep the exact native supercluster path; larger sources used WGSL projection plus `GPUKeplerAggregation` over a dense viewport grid at `Math.round(zoom)` (cell = cluster radius in 512-px world pixels), per-cell count, planar-position sums and sum/average/count color fields; only O(cells) summaries are read back for sqrt radius and quantize/quantile/ordinal/threshold colors, and hover members come from a bounded atomic-append kernel. Differences from native supercluster: grid cells instead of greedy merge (membership and counts differ), members are the first rows found sorted ascending, float32 projection error near zoom 20, fixed-point field sums, no antimeridian wrap; min/max/median aggregations fall back to capped supercluster. Flow: chunk-wise float32 coordinate upload, GPU sort/scan location dedupe keyed by float32 bits (first-seen ids like native), masked OD aggregation by sort and segmented sum feeding the pinned `@flowmap.gl/data` clustering; row cap removed, limits now 262,144 unique locations and 4,194,304 aggregated pairs. Differences: float32 location identity, one aggregated datum per OD pair (picking returns the aggregate), float32 unordered sums. Node tests (CPU references, 18 Cluster + 9 Flow) and `tsc --noEmit` pass; Browser results (2026-10-30): `scripts/check-cluster-native.mjs` passed at 300,000 rows (cluster counts equal a CPU reference at five views, hover members checked via the lookup directly, small source keeps the exact path; no bugs). `scripts/check-flow-gpu.mjs` passes against the CPU reference up to 1,000,000 rows after fixing a reserved WGSL identifier (`meta`) and a 9-storage-buffer pair-emit kernel. Still unrun: Flow through the real UI layer, large-table memory (about 3 GB estimated at 10M Flow rows), visual parity, and Cluster min/max/median CPU fallback. |
| Views (dual and swipe) | Split maps are the default single-Deck host (`?views=1` forces single view). `views.mjs` and `views-overlays.mjs` pass: per-view basemaps, hover, click, pan/zoom and touch, unsynchronized viewports, pitched and globe splits, wheel pivot under 0.5 px, editor drawing and Select Geometry in either view, closing map 0 with the device retained, labels and elevated points in view 1, and Swipe/unsynced state through native config and portable bundles. |
| Globe coverage | Every resident layer family draws, culls past the horizon and picks under Deck 9.4 GlobeView, including icons, labels, a sphere-mesh heatmap and grid/hexagon/cluster aggregates. Long edges are cut on Deck's 5° grid with antimeridian unwrapping in an accounted per-source globe variant; a WGSL depth disk replaces Kepler's GLSL one. `globe.mjs` and `gpu-globe.spec.ts` pass. |
| Linked filters, charts and field mapping | Native "Add synced filter" and field selectors work on resident datasets; native default intervals apply to every linked dataset; line charts compose per dataset. `native-linked-filters.mjs` passes with two resident datasets and one CPU dataset. |
| Chart measure and bin-edge parity | Native d3 tick and numeric-interval edges shared exactly with CPU datasets; GPU average/sum/min/max/median/sample deviation measures on time and numeric filters; grouped series by dictionary field in native first-seen order. `resident-numeric-plot-edges.node.spec` matches native on 500 random domains; `native-numeric-histogram.spec`, `native-numeric-charts.mjs` and `smoke.mjs` pass. |
| Trip markers | `GPUTripMarkerLayer` computes per-trip position, orientation and color on the GPU and draws Deck `ScenegraphLayer` models with textures, PBR and glTF animations; yaw/roll fields and table-mode per-timestamp colors gather from resident columns. `trip-markers.mjs` passes. |
| H3 polygon brushing | Per-vertex brushing varyings in the polygon and path shaders; `h3.mjs` passes. |
| Acceptance harness | `scripts/acceptance.mjs` loads 1M/5M/10M rows via `?rows=` (with `?memoryBudgetMebibytes=32768`), records adapter info/limits and a classified hardware tier, measures filter, brush, histogram, style, linked-summary latency, frame time, upload time, accounted memory and warm upload/readback bytes through `window.__keplerComputeAcceptance`, and compares against `acceptance-targets.json`. Fixes needed for the first real run: explorer working-memory reservation raised from 128 to 160 bytes/row (1M needed 144 MB against 136 MB), `memoryBudgetMebibytes` query, `--verbose` page logging, per-interaction normalization of readback, blocked reasons kept in notes. Measured on Apple M-class, headless Chromium 151, dev server, tier `unclassified` (device limits below the 1 GiB reference rule), 20 samples each, p50/p95 in ms (target = provisional p95 maximum): 1M filter 10.2/10.8 (500), brush 7.2/11.9 (500), histogram 16.8/17.0 (1000), style 11.2/59.6 (500), linked summary 15.6/26.8 (1000), frame 8.0/13.6 (167); 5M filter 10.8/25.4 (1500), brush 7.3/59.9 (1500), histogram 66.9/68.8 (3000), style 13.9/76.3 (1500), summary 61.7/63.0 (3000), frame 8.0/15.9 (501); 10M filter 10.9/21.5 (3000), brush 7.2/91.5 (3000), histogram 127/128 (6000), style 29.2/155.5 (3000), summary 116.6/118 (6000), frame 8.0/13.5 (1002). Upload 140 ms / 617 ms / 1102 ms (limits 20/60/120 s); warm upload 0 B at all sizes; warm readback about 101-117 B per interaction (limit 4096). Accounted memory 88 MB / 440 MB / 880 MB against 512 MiB: pass, pass, **fail at 10M**. Overall verdict fail, solely from 10M memory. With the default 512 MiB budget and 8 GiB, 10M is blocked (registry total reached 8.5 GB); it loads at 32 GiB. Frame and latency are host-submitted-frame timings, not GPU completion; all targets except warm upload are provisional or derived, so this is a measurement, not ratified acceptance. 10M linked summary (118 ms) would exceed the plan's 100 ms compact-summary target if it applied to that size. Only one machine/tier was run. |
| GPUScalarScale registration | The scalar-scale contributor used by resident styles is registered in the operator catalog, kernel audit, lens scenarios and table of contents; the three documentation contract tests pass. |
| Integer chart measures, median and deviation on chart cards | Chart cards and filter line-chart y measures accept Int32/Uint32 fields, widened to Float32 on the GPU behind the ±2^24 exact-range check (chart cards via a provider-cached copy, plot measures via a per-measure widening pass certified before the first summary); values beyond the range reject clearly. Chart cards now offer exact median and sample deviation for scalar, category, numeric, time, calendar and series cards through the ordered-statistics adapter. Verified: `tsc --noEmit`, `yarn build`, and node specs `resident-chart-plan`, `resident-chart-provider` and `resident-chart-provider-widening` (13 tests). GPU specs (`resident-chart-group-runtime`, `resident-chart-runtime`) and `native-charts.mjs` were extended but not yet run; line-chart plot-measure integer widening has no dedicated spec; GPU/browser verification is pending. |

The first scale audit projected about 867 MB of accounted capacity for one million rows because
each owner retained its entire coarse maximum allowance. Planned live-resource charging now admits
that workload at about 297 MB with preallocation checks and fenced retirement. This resolves the
observed admission blocker. The latency targets, larger workloads, broader consumer combinations
and complete device-transfer/memory instrumentation still need measured acceptance.

The instrumented 1m report (local research log, not checked in) and
CPU profile (local research log, not checked in) clarify the latency workload:
the baseline calls native create-or-update, which updates dataset, field and value even for an
existing filter. Its synchronous dispatch includes intermediate redraw/encode work. It is not a
measurement of the slider's value-only action. Profiling adds overhead and its latest-encode
snapshot does not sum those intermediate encodes. A separate value-only measurement and
transaction-coalesced redraw are now measured below; existing baseline artifacts remain unchanged.

Browser fixtures now retain their initial module snapshot while unrelated implementation files change;
they suppress only development hot-reload messages, while runtime errors remain test failures.
GPU execution is serialized to avoid shared-driver contention. CPU implementation and focused checks
continue alongside it, without a global repository freeze.

The separate coalesced create-or-update report (local research log, not checked in)
and value-only report (local research log, not checked in) retain the same
one-million-row source, native point/Count workflow, 16 batches and 512 MiB budget. Redraw invalidation
now waits until the current Redux transaction completes, avoiding intermediate synchronous encodes.
For the original create-or-update workload, submitted-frame p50/p95 improved from 93.5/120.7 ms to
62.8/87.2 ms, and Count observation from 159.0/179.3 ms to 95.5/140.6 ms. Separately, the existing
filter's value-only action measured 39.7/61.9 ms to submitted frame and 83.5/105.5 ms to Count.
Both runs retained 297,350,340 accounted bytes, identical source/device objects and zero source-upload
delta, with 20 bytes of exposed explorer readback. Five samples and concurrent CPU checks make these
smoke comparisons, not latency-target acceptance or GPU timing. The original workload was not
silently replaced by the faster action.
| View boundaries (globe pivot, live tile occlusion, touch) | Browser-verified: `views.mjs` measures about 1e-11 px wheel-pivot drift in single, swipe and dual globe views after the exact controller was rebuilt as a subclass of Kepler's own globe controller and the top-level Deck `controller` prop was synchronized with view 0; `views-touch.mjs` passes pan, pinch, native two-finger rotate and divider drag under Chromium Pixel 7 emulation; `globe-tiles.mjs` sees 13,538/14,400 near-side tile pixels and no far-side leak over live CARTO vector tiles; `globe.mjs` and `views-overlays.mjs` still pass. Physical iOS/Android devices remain a manual checklist (`TOUCH-CHECKS.md`). Implemented: ray-sphere wheel pivot solver and controller wired into single, dual and swipe globes with `views.mjs` tightened to 0.5 px; tile-layer depth-test parameters and depth-disk ordering (`insertGlobeDepthDisk`, now applied to the single and split layer lists when a GlobeView is present); Deck 9.4 native `touchRotate` enabled for map and globe views through `withTouchRotate` (GlobeState extends MapState, so no custom rotate code; `touch-gestures.ts` pinch/rotate helpers remain unused reference math); Pointer Event Swipe divider with capture and `touch-action: none`; `globe-tiles.mjs` live-tile far-side pixel check and `views-touch.mjs` touch-emulation script (Chromium CDP, WebKit divider only); manual device checklist. Node specs cover the pivot math, tile occlusion helpers and gesture math; scripts are syntax-checked only and not run. |
