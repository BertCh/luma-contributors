# GPU compute adapters for kepler.gl

This private integration package connects application-owned datasets to luma compute. It keeps
GPU resources outside serializable UI state, preserves source identity and makes CPU readback
and complementary execution explicit. The Kepler example (`examples/deck/kepler-compute`) is not
ported to published luma.gl and is deferred. The presence of these adapters does not establish
complete Kepler layer or analysis parity.

## Dataset and execution ownership

`KeplerComputeSession` composes the Arrow bridge, registry and explorer. Import a table with
`importArrowDataset({id, table, normalization})`, then call `openExplorer` with dimensions and a working
memory budget. Source revisions upload once; each explorer holds a lease. Closing an explorer,
removing a dataset or destroying the session accepts a completion fence and waits for pending reads.
The session accounts for retained ArrayBuffer capacities and GPU upload allocations. Explorer
reservations cover reported output/transient storage and bounded inspection; JavaScript objects,
driver overhead and executable-private allocations remain outside those measurements.

`KeplerDatasetRegistry` registers versioned datasets and explicit CPU/GPU allocations. Leases retain
retired revisions until submitted work or readback finishes. Requests record their source revisions;
superseded or cancelled requests cannot publish results. Memory reservations precede allocation,
and device loss retains only explicit recovery sources. Borrowed resources are never destroyed by
the registry. Cleanup failures remain visible through its cleanup-error API.

Callers retain command submission ownership. Cancellation prevents future scheduling and result
publication; it does not preempt an already submitted GPU command buffer. Pass completion fences
when releasing resources still referenced by submitted work.

## Resident exploration

`GPUResidentExplorer` borrows scalar GPU columns and composes numeric, categorical, rectangular and
external selections. It exposes a source-aligned `selectionMask` usable by GPU layers, stable selected
row IDs on the GPU, and linked histograms. Source data and selections remain resident during control
updates. External masks can come from time or spatial contributors.

Geometry-only sources can supply `sourceChunkLengths` without scalar columns. The session derives
this topology from retained Arrow batches, including empty batches. Internal selection storage is
accounted separately and does not create a user-visible field or change exported source schemas.

The host encodes the explorer before Deck renders its output and submits the frame once. Explicit
`readSummary`, `readPage` and `readRows` operations return bounded CPU results for charts, tables and
tooltips. Readback results carry dataset/control revisions; stale reads return no result. Call
`discardEncoding` if an encoded frame is abandoned. Keep borrowed columns alive through
`destroyAsync` when inspection is pending.

`GPUKeplerTableInspector` adds bounded all-source or selected-row pages and stable GPU sorting across
source batches. Sort ties preserve source order, and missing values sort last. It accepts published
analysis columns with explicit validity and revision metadata. Its source lease and reserved sort
memory must be released before destroying borrowed analysis/explorer buffers. The host's native
table modal uses this interface instead of Kepler's synchronous CPU filtering hook.

The example's original-value mode joins bounded GPU source IDs to retained Arrow cells and computed
columns. It preserves exact wide numbers and timestamp units, with bounded geometry/text previews
and explicit source export for complete values. Rich original types have disabled sort controls
until an exact comparator is available; supported Float32 and computed sorts remain on the GPU.

Numeric validity is separate from the source values. The caller chooses which selections affect
display, domains and analysis populations. Hiding observations is not implicitly a request to
recompute statistical weights or inference on a different population.

## Native selection scopes and style domains

`GPUKeplerScalarSelection` keeps a mask per native filter ID, including independent filters on the
same field. Numeric endpoints respect the stored integer or Float32 format; exact time predicates
use the timestamp sidecars. `planKeplerSelectionScopes` and `GPUKeplerSelectionScopes` combine
those masks with geometry predicates. Each layer receives its own polygon intersections. Native
table and analysis consumers receive the union of polygon-targeted layers, including hidden targets.
Dynamic style domains exclude polygon filters, and each filter histogram excludes only its own
scalar predicate. The native host supplies these distinct masks explicitly.

`GPUKeplerSpatialSelection` preserves source chunk lengths in its public mask, including empty
Arrow/WKB batches. Its internal containment and range-selection bindings omit empty chunks together;
nonempty buffers are borrowed without copying, and global source-row order stays unchanged. Empty
position chunks may use four-byte placeholder buffers because they are never bound as XY rows.
Positions and output masks must have packed layouts; interleaved vector layouts and negative
chunk lengths are rejected rather than normalized when empty chunks are omitted.

`addFilter` and `removeFilter` add or drop one filter slot in place (other masks keep their identity;
a removed slot is destroyed immediately or after a supplied completion), and `updateControls` sets
controls for a subset. Allocation failures raise `KeplerAllocationError` (luma `OutOfMemoryError`
for device out-of-memory); test them with `isKeplerAllocationError`.

H3 topology uses an additional mask matching native CPU-filter membership, computed on the GPU.
Ordinary native GPU range filters affect display without changing the native H3 geometry branch;
CPU-assigned filters, numeric playback intervals and layer polygons can change that branch. The
host retains the pinned reducer's filter allocation history when classifying those predicates.

Native Charts supports Big Number count, sum, mean, min and max over resident data, plus vertical and
horizontal string-category bars with the same measures. Category charts preserve native ordering,
top-N/Other behavior and crossfilter selections, with at most 512 dictionary entries and Float32
numeric measures. Numeric interval bars support Float32, Int32 and Uint32 grouping, exact source-format
domain/classification, empty bins and native inclusive interval crossfilters. They use at most 512 bins
and two GPU phases over one source/selection revision; only extrema and summaries return to the CPU.
Integer automatic grouping shows unique values within the native group budget and switches to ranked
intervals above it. Distinct values are discovered on the GPU with bounded readback; unlimited unique
grouping rejects cardinality beyond the 512-summary limit. Lines share the same summaries with native
category ordering, numeric bins and point crossfilters. String-dictionary series are supported for
bars and lines within 4,096 cells, counting one additional missing-label slot per group. Composite
GPU keys preserve source chunks; bounded cell summaries preserve first-row order, null measures and
native sums of per-series statistics (including averages/minima/maxima). Grouped vertical/horizontal
bars and separate lines show individual series values; an accessible breakdown table retains
individual values and row counts. Series colors and line styles stay stable
across filtering; missing observations break lines, supplied zero remains a point, and negative values
share a zero baseline. Mouse/Enter/Space selection filters the entire parent group. Other remains an
aggregate-only, nonselectable mark. Missing series labels remain in the group row count but do not
contribute to its value. Numeric series fields and additional chart families remain unsupported.
UTC hour, day, Sunday-start week, month and year lines accept prepared second or millisecond Arrow
Timestamp/Date fields. They preserve chronological order, exact inclusive period filters and native
null-to-epoch grouping. Chart-owned period filters preserve that null policy; ordinary time filters exclude nulls.
Calendar lines support the same five measures and dictionary series within 512 observed periods and
4,096 cells. Second timestamps normalize with exact integer arithmetic in graph-owned scratch per
original chunk. Hour grouping discovers observed days first, then bounded day/hour keys, preserving
the full supported Date range without overflowing a signed epoch-hour key. Same-ID chart replay waits
for the previous reader and its selection mask to retire before allocating a replacement.
Automatic intervals and microsecond/nanosecond chart sources remain unsupported.
These charts use explicit selected/all populations and bounded summary readback.
Native chart configuration, captions, pinning and replay stay in Kepler. Unsupported chart families
and measures show their availability reason; resident tooltip charts are guarded separately. Consumer
selection scopes can omit owned CPU filters by ID and native GPU filters by axis field independently,
while preserving other CPU filters and targeted-layer polygon unions.

`KeplerResidentStyle` derives global finite domains and quantiles in GPU scratch without changing
source chunks. Outputs retain source-row topology and may be borrowed by renderers or gathered onto
multipart geometry. Scalar and color channels have separate plans, and categorical plans declare
native label order rather than assuming dictionary-code order. Log domains can be fixed or derived
from a source/selection population. Derived domains use the pinned native zero-minimum adjustment
and require finite endpoints with the same nonzero sign. Invalid log populations produce no-data
outputs and an explicit legend status; negative domains remain supported. Unsupported formats or
scale modes remain explicit errors.

`KeplerResidentStyleCache` reuses compatible outputs across control changes and requires a completion
fence for topology replacement. Bounded `readLegend` results carry source, style and selection
revisions; stale legends cannot be published. Empty domain populations produce no-data outputs and
empty legends. The example replaces native synchronous domain/legend consumers for resident styles,
so missing CPU quantile metadata does not invalidate otherwise valid GPU filter selections.

The Arrow package prepares multipart geometry once and records primitive-to-source row mappings.
The host expands source styles and masks with GPU gathers; picking maps back to original source
rows. Geometry-specific predicates retain their native semantics: a MultiPoint uses any component,
connections require both endpoints, and GeoJSON uses its prepared representative point.

## Native H3

The example's H3 source role retains exact Uint64/string cell IDs and original Arrow row/batch
identity. Invalid or missing IDs keep their source slots, with invalid-cell diagnostics and no
geometry. Unsafe numeric IDs fail admission. Native CPU-filter membership selects the column or
polygon branch through a bounded GPU topology summary; display filtering remains independent.
Coverage and style updates reuse uploaded geometry, and picking returns original source rows.

Current render positions use Float32 geographic coordinates. Preparation retains Float64 canonical
coordinates and measures center/boundary rounding separately from exact ID preservation. A cell
whose boundary collapses, loses a distinct edge or reverses orientation under the Float32 upload
policy rejects the whole publication before replacement or GPU allocation. Other measured rounding
is admitted. Full high-resolution geographic precision requires a future high/low position adapter;
exact IDs alone do not establish that rendering guarantee. End-to-end verification status is recorded
in the implementation roadmap.

## Native Trips

The example publishes one timed GeoJSON LineString per source row as a native Trips layer.
`prepareGeoJSONTrips` in `@luma.gl/arrow` requires an explicit timestamp encoding and retains exact
source times, source-row mappings and batch boundaries. Relative Float32 animation values must meet
the chosen error budget. Original timestamps are not replaced by those rendering values.

`planKeplerTripAnimation` maps the native absolute animation time to the prepared origin once and
converts trail seconds to milliseconds. GPU gathers apply source masks and styles to valid paths;
playback and filtering reuse path/time allocations. Picking returns original source rows even when
null rows precede valid trips. Saved source manifests retain the encoding, origin and error budget.
Grouped tabular trips and scenegraph markers require additional adapters.

## Native mesh icons

Native Icon layers retain Kepler's SVG triangle geometry. The example groups source rows by marker
once, uploads immutable meshes and row maps, then gathers changing selections and style columns on
the GPU. Each group preserves source batch boundaries and maps picking back to original row IDs.
Native radius/fixed-radius, billboard and high-zoom scaling remain distinct from texture-atlas icon
sizing. Catalog errors are visible; unknown markers do not silently become squares. Current source
coordinate roles provide XY, with tabular altitude and portable custom catalog assets still pending.

## Analysis and complementary execution

The analysis bindings append existing contributors to a caller-owned command graph. They cover
standardization, classification, excess risk, spatial rates, hot spots and additional bounded
statistical and geometry operations without introducing another numerical implementation. The
[analysis API guide](src/analysis/README.md) documents explicit geographic/projected point weights
and source-aligned Dorling cartograms, including coordinate contracts, limits and output ownership.
It also documents bounded geographic ordinary kriging with explicit manual variograms, separate
prediction/variance rasters and finite-result diagnostics; model fitting and confidence intervals
are not inferred. Result metadata records the dataset revision,
population and numerical policy. Selected-population recipes require an explicit selection revision.

`selectKeplerExecutionRoute` checks formats, capacity, memory admission and backend policy before
dispatch. General SQL goes to an available DuckDB executor. Unsupported statistical operations need
an actual statistical backend; they do not silently become SQL. Execution failures stay on the
selected route instead of triggering an unannounced rerun.

`createKeplerDuckDBExecutor` accepts a caller-owned connection with `query` and `insertArrowTable`.
It returns CPU Arrow results and registers explicitly materialized Arrow tables with source metadata.
It neither starts DuckDB nor performs a hidden GPU transfer. Connection ownership remains with the
application.

Analysis recipes serialize field names, source and selection revisions, numerical choices and backend
policy. Restoring a recipe checks the referenced dataset, selection and weights revisions. Buffer
handles, physical row positions and devices never enter serialized recipes.

## Analysis experience: profile, descriptors, suggestions, results

These modules sit beside the catalog bindings and are joined to them by binding id and parameter name.
They are plain CPU code. The design and its limits are in
`dev-docs/research/analysis-workflow-ux-2026-11-14/README.md`; the vocabulary they use is documented in
`docs/api/gpu-spatial-analysis/analysis-experience.md`.

- `profile/`: `profileKeplerFeatureCollection` and `profileKeplerColumnTable` build a
  `KeplerDatasetProfile` (kind, column statistics, guessed measurement levels, coordinate model, row
  semantics, extent and spacing). `createKeplerDatasetProfileCache` keeps one per dataset revision and
  the user's declarations per dataset; `applyKeplerDatasetProfileDeclarations` replaces a guess and
  recomputes spacing when the coordinate model changes.
- `descriptors/`: one `KeplerOperationDescriptor` per binding (`getKeplerOperationDescriptor`): the
  questions it answers, field roles tied to its column parameters, requirements, default policies,
  basic and advanced parameters, output presentation, interpretation and notes. A binding without a
  hand-written descriptor (81 of the 217 at landing) gets a conservative derived one and is not
  suggested.
  `resolveKeplerOperationParameters` resolves defaults from a profile (data rules, presets, grid caps);
  `evaluateKeplerOperationApplicability` says whether a binding can run on the loaded datasets and with
  which columns; `suggestKeplerAnalyses` ranks analyses for the loaded data and returns the questions a
  host should ask (what rows stand for, the coordinate system, what a column measures) and the analyses
  that are one step away. `workbench-adapters.ts` exposes the same in the shapes a workflow UI reads.
- `presentation/`: `planKeplerResultPresentation` chooses a scale, classes, legend entries with counts
  and colours for an output table; `describeKeplerCatalogResult` returns the sentence, what was chosen
  and why, notes, cautions and next questions; `formatKeplerResultReport` renders it as text.
- `curated/`: `KEPLER_CURATED_ANALYSES` is the starting set of question-named analyses, each one
  binding with typed dataset slots, presets and a synthetic sample (`createKeplerCuratedAnalysisSample`);
  those without a binding yet say what is missing. `createKeplerCuratedAnalysisMapping` binds an
  analysis to the user's datasets and columns and `buildKeplerCuratedAnalysisRequest` builds the run.
  `curated/bindings/` adds six CPU-reference bindings that fuse existing steps (rate cluster map,
  excess risk, clusters and footprints, regression with diagnostics, kriging with diagnostics,
  contiguity correlogram).

A status of `ready` means the checks that can be made from a profile passed. `needs-confirmation` and
`needs-mapping` carry the question or the choice; a guess never blocks and never fills a role unasked.
When a coordinate system chosen for a run contradicts what a layer was declared as, the choice stands
and a warning says what it means for the distances.

## Arrow preparation and precision

The separate `@luma.gl/arrow` package supplies `prepareArrowTableForGPUAnalytics` and the existing
analytics upload helpers. Preparation widens Boolean and narrow integers exactly, reconciles category
labels explicitly and retains source batches. Float64 conversion requires an explicit exact or
bounded-error policy. Timestamp words preserve signed 64-bit values and units; interpreting a general
Int64 column as time requires an explicit unit.

Uploading CPU Arrow to GPU storage and materializing GPU results for DuckDB are real transfers. Account
for them, reuse resident inputs between interactions, and preserve source batches unless an explicit
operation requests packing.

## Verification

Run the repository's stable interfaces, with Node selected through `nvm use`:

```sh
yarn test-node --maxWorkers=1 modules/kepler/test
yarn test-headless --no-coverage --maxWorkers=1 modules/kepler/test
yarn build
```

Focused tests cover resource lifetime, revisions, routing and recipes, with real WebGPU tests for
resident filtering, bounded inspection, spatial selection and analysis outputs. The pinned host also
needs browser integration checks; passing an isolated contributor test is insufficient to establish
the frontend path.
