# Kepler sandbox review verification

Date: October 28, 2026. This records checks performed for the coverage guide and review. The integration
chat owns final runtime, build and full-suite verification after its concurrent changes settle.

| Check | Result |
| --- | --- |
| `nvm use` | Passed with Node 22.23.4. |
| `yarn install` | Passed with existing peer-dependency warnings. |
| `yarn lint fix` | Passed after guide/tool additions: 8,344 files checked, three formatted; lockfile valid. |
| `yarn test-node --maxWorkers=1 modules/kepler/test/sandbox-capabilities.node.spec.ts` | Passed: one file, three tests. Checks all pinned upstream layer IDs/version, ledger completeness and combined search filters. |
| `node examples/deck/kepler-compute/scripts/capability-matrix.mjs` | Passed: generated documentation matches all 66 inventory entries. Use `--write` after changing the JSON source. |
| Guide-only TypeScript check | Passed using the example compiler options with the two guide source files and Vite client declarations. This excludes concurrently edited runtime files and is not a substitute for the full example or repository build. |
| Local review links | Validated, including the implementation files and existing roadmaps. |
| `yarn workspace luma.gl-examples-kepler-compute typecheck` | Ran and failed on concurrently edited resident layer overrides, timestamp fixture, saved-state types and session/export APIs. No errors named the capability guide files. Diagnostics sent to the integration owner; this is not an overall typecheck pass. |
| `yarn build` | Not run in this review. Shared implementation is still changing; disk space was about 2 GiB. Remains a required integration/pre-merge gate. |
| `yarn test` | Not run to completion by this review. Integration owner runs the full suites; no passing claim is inherited here. |
| `yarn website:build` | Not run in this review. Required for final integration/pre-merge verification. |
| `(cd website && yarn build)` | Not run in this review. Required for final integration/pre-merge verification. |

An initial `yarn test-node <path>` invocation unintentionally selected the whole node project because
the shared runner parses positional arguments before options as project names. It was stopped through
its own execution session; it is not a passing test run. Place an option before the path as documented
in [the test setup](../../../docs/developer/dev-tools/llm-friendly-test-setup.md#focused-test-filters).

Two baseline browser smoke attempts ran before coordination of GPU tests with the integration owner:

1. Startup failed in the browser `util` dependency with `process is not defined`.
2. After bootstrap remediation, the initial selected count was 5,000 instead of the fixture's 10,000.

Those failures informed the review and were handed to the runtime owners. Later integration-owner
reports of successful compute workflows do not replace a final independently captured full host gate.
The in-app browser could not load the local page reliably, so this review does not claim a completed
visual/accessibility audit. The owner was asked to check guide expansion, the 22-layer filter, raster
search, details and the 66-entry reset during its next host UI check.

The [source manifest](./source-manifest.json) fingerprints installed upstream files used for the static
inventory. It is not a build manifest, runtime capability detector or performance certificate. The
coverage panel reads reviewed metadata only and never authorizes an execution route.

## Implementation evidence added after the initial review

The table above records the initial audit, including failures that informed the implementation. It is
not the latest integration gate. See the [implementation record](../../../dev-docs/roadmaps/kepler-compute-integration-implementation.md)
for the coordinated full build, test and website results.

This review chat subsequently implemented and checked:

- The native asynchronous data table: bounded all/selected paging, stable GPU sorting across batches,
  source-row identities and analysis columns with validity. Four GPU inspector cases plus three
  host-analysis cases passed. The actual native modal was checked against a CPU sorting and
  selected-population z-score reference; disabling a sorted derived column also recovered correctly.
- Native export/save modal routing to explicit sandbox controls. The optional chart guard is prepared,
  but the pinned application does not mount the native chart editor; no native-chart support is claimed.
- Saved-source coverage checks. Missing native dataset references and renamed logical sources fail
  before reimport or configuration application. Ten focused source/save tests passed.
- Native source preparation: real Kepler row and Arrow containers, original field names, timestamp
  nulls, capacity admission, stale-source detection and borrowed Arrow batches. The CPU browser test
  passed. The adapter itself preserves batches; native ingestion must also disable the pinned
  `maxArrowBatches: 1` compaction default before loading Arrow.
- Explicit schema/precision preview: unsupported field exclusion, measured Float64 rounding policy,
  invalidation after changing conversion choices, and publication only after a valid preview. The
  standalone React browser check passed.
- Generic published-dataset consumers: arbitrary field names, dictionary labels, sorting, source
  replacement, selection invalidation and borrowed ownership passed a real WebGPU harness. These
  source/publication helpers remain unconnected to the main native application.
- Share links: bounded versioned URL fragments, Unicode state, source coverage, reproducible fixture
  admission, clean destination parameters, clipboard fallback and deferred loading while busy passed
  standalone browser checks. The link stores configuration and a deterministic fixture reference;
  it does not upload source rows or provision hosting.

Focused harnesses live under `examples/deck/kepler-compute/scripts`: `native-table.mjs`,
`native-source.mjs`, `source-preview.mjs`, `published-datasets.mjs`, `sandbox-share.mjs`,
`sandbox-share-panel.mjs`, and `native-share.mjs`. GPU harnesses must run sequentially with the full
headless suite to avoid competing for the device. Standalone component checks do not establish
end-to-end arbitrary-data publication.


The connected share workflow also passed in the actual host: a fresh session rebuilt 4,096 fixture
rows and restored exactly 1,576 selected rows after explicit replay. The sandbox regression passed
capability search, native replay, selected export, regeneration, DuckDB registration/query and result
upload. A later replay fix rejects incompatible fixture schema/normalization before destroying the
running source; all 14 focused source/save/capability tests passed. Final example typecheck,
production build and `yarn lint fix` passed for this addition. Production output still reports large
chunks. The full coordinated baseline test run predates these UI/share changes; the new changes have
these separate focused and browser checks, not a second full-suite claim.


The independent portable-bundle codec (`sandbox-bundle.ts`, exercised by `scripts/sandbox-bundle.mjs`)
also passed CPU-browser and staged TypeScript checks. It round-trips multiple retained Arrow sources,
original field names, 2+1 batch boundaries, Float64 values, validity, dictionary labels and exact
microsecond timestamp words with UTC metadata. It validates source revisions/schema/coverage,
checksums, stale-source export and explicit document/Arrow/row/source capacities before returning a
restored document. Limits describe payload sizes, not peak JSON/Arrow parser memory. It performs no
GPU readback/upload or native edits. Bundle UI and host restoration are not yet connected.


## Connected multi-source state and portable workspace controls

The later workspace addition supersedes the earlier "bundle UI not connected" limitation. Save state
now captures the fixture and all resident publications, including original dataset IDs, source schema,
normalization and coordinate/geometry roles. Reference-only replay requires those sources to remain
available at their saved revisions; rebuilding uploaded data uses the portable bundle.

`SandboxBundlePanel` saves full retained Arrow sources, validates uploaded bundles and previews their
source/row/batch/field counts before explicit restoration. Limits are 16 sources, 1,000,000 rows per
source, 64 MiB Arrow payload and 96 MiB JSON. CPU preflight validates every manifest, source schema,
normalization, coordinate role/value, geometry representation and supported analysis before the host
replaces the workspace. Recipe acceptance is shared with the analysis runtime. GPU allocation during
restoration is not transactional: a later device/allocation failure can leave a partial workspace.

The actual-host `scripts/native-bundle.mjs` check passed a complete download → fresh-session load →
explicit restore cycle. It restored 503 selected fixture rows out of 1,024 and two selected uploaded
rows out of four. Original IDs, coordinate roles and 2+2 Arrow batches survived; the native selected
row table returned source IDs 1 and 2 with values 20 and 30. A valid-checksum bundle with a nonexistent
coordinate field failed preflight without changing the current selections or publications. No browser
errors were reported. Twelve focused sandbox-host tests also passed, including all-source save/replay,
revision rebinding and preflight failures before any callback mutation.


Final focused verification for this addition: 18 Node tests across sandbox-host, native-config-sources
and capability inventory passed; full example typecheck and `yarn lint fix` passed. These checks are
separate from the earlier full baseline suite. Portable capture currently includes the fixture and
resident publications. Native-only originals retained by Add Data are not embedded; references to
such sources are rejected by the source-coverage guard rather than silently omitted.


## Selected original-source exports and per-source analysis persistence

The sandbox export controls now select any resident dataset and one of prepared GPU JSON, original
Arrow, or original CSV. Original-source export obtains bounded selected row identities from the GPU,
then explicitly serializes retained Arrow source rows. It does not filter retained CPU rows or read
whole GPU columns. Arrow slices contiguous source runs without repacking the source and may split
record batches; it preserves raw storage, field/schema metadata, validity, dictionary domains and
geometry. An added collision-safe source-row column and export metadata identify the source and
selection revisions. CSV is a text representation with quoting, JSON structured cells and exact UTC
timestamp fractions where representable; Arrow is the lossless typed format.

`scripts/source-export.mjs` passed CPU-browser checks for original Float64 values, nulls, dictionaries
(including different dictionaries across source batches), structured geometry, signed nanosecond
instants, field-name collisions, metadata, empty selections, CSV quoting, output budgets and
cancellation. Two focused Node tests verified identity-only readback, serialization reservation
ownership and cleanup on failure/cancellation/stale selection. Actual-host
`scripts/native-source-export.mjs` passed published-dataset selection and all three download formats:
original Arrow/CSV preserved precision, exact nanosecond timestamps and excluded source geometry;
prepared JSON deliberately returned rounded GPU values and omitted nonuploaded fields.

Saved workspace state now supports one active analysis per resident dataset. Shared recipe parsing
validates the operation, source field and eligible nonempty prepared Float32 storage before applying
native state or rebuilding bundle sources. Replay maps each recipe to its dataset, rebinds selected
population revisions and disables existing analyses absent from the saved state. The focused
sandbox-host/export Node group passed 16 tests, including multiple active analyses, changed selection
revision, missing/duplicate recipes and incompatible normalized column types. The example typecheck
also passed. These are separate focused results, not a new full-suite claim.


The final combined source-export and sandbox-host group passed 17 tests after adding shared exact
natural-breaks capacity validation. A 1,025-row exact recipe is rejected before current config replay,
fixture reimport or bundle replacement. Final example typecheck and formatting passed; the production
example build passed during this tranche with the existing large-chunk warning. The analysis chat
owns the expanded method/runtime tests; this record does not substitute persistence tests for those
numerical or native UI checks.

## Computed analysis export and multi-input replay

The export selector now offers analysis JSON, Arrow and CSV for the selected resident dataset.
A dedicated inspector reads bounded selected pages from a captured analysis snapshot, including every
computed scalar column and its validity. It does not supersede native table requests. The analysis
owner retains result buffers until the export and inspector cleanup finish. Source, selection and
analysis revisions are checked throughout and once more before download publication; cancellation
or invalidation discards partial output. Readback budgets include both count queries per page,
source identities, computed values and validity. CPU reservations cover retained/page arrays, Arrow
validity bitmaps and bounded serialization buffers; ordinary JavaScript metadata remains additional.

Arrow preserves Float32, Int32 and Uint32 storage, null validity and nonfinite values. JSON represents
nonfinite numbers as explicit strings and distinguishes them from nulls. All three formats preserve
source-row identities using a collision-safe field. Arrow schema metadata and the JSON document carry
the recipe, analysis revision, source/selection revisions, selected count and page offset; CSV emits a
companion metadata JSON download. The export population is the current selected rows even when the
analysis recipe was calculated over all rows. Original source and computed results remain separate
export choices; combined exports, native-only sources and streaming downloads are still gaps.

CPU-browser codec checks passed all scalar types, null/nonfinite values, identity collisions, recipe
metadata, empty selections, output limits and cancellation. The focused analysis-export and
sandbox-host Node group passed 18 tests, including readback/CPU/row budget rejection, stale/canceled
page cleanup, output-reservation ownership and both-field excess-risk replay validation. Excess-risk
replay rejects missing or non-Float32 denominator fields before native configuration or bundle
replacement. Example typecheck and production build passed, with the existing large-chunk warning. Formatting
passed. These focused results do not imply a fresh full repository build/test run.

The actual-host `scripts/native-analysis-export.mjs` passed a nullable, two-batch excess-risk source:
Arrow preserved the null and computed values plus source IDs; filtering changed JSON output to rows
1 and 3 while retaining the all-population recipe; CSV and its companion metadata both downloaded;
disabling the analysis rejected a subsequent export without downloading stale results. The initial
test fixture incorrectly concatenated a non-nullable first-batch schema with later nulls; ingestion
correctly rejected it. The verified fixture now slices one consistent nullable schema into batches.

## Combined source and analysis output

The export panel now also offers original and computed fields in one Arrow or CSV download. The
analysis inspector supplies selected source identities and computed values/validity once; retained
source Arrow is sliced by those identities, so original Float64, timestamps, geometry and dictionary
storage are not reconstructed from prepared GPU columns. Computed fields are appended in the same
selected-row order across source batches. Collisions receive repeated `_analysis` suffixes until
unique, and metadata records the original-to-exported computed name map. The identity field is unique
across both source and computed names. Calculation metadata uses that final identity field name.

The combined codec passed selections spanning three physical source batches, exact Float64 and
nanosecond values, dictionary labels, nullable and nonfinite computed fields, repeated name collisions,
and CSV metadata. Four focused source/analysis-export Node tests passed, including identical GPU
readback cost for computed-only and combined output, budget admission, stale/canceled reads and
reservation cleanup. Original-source codec regression checks also passed. Full example typecheck
passed at this checkpoint; actual browser download checks remain queued with shared GPU use.

Actual-host combined downloads subsequently passed: filtered original measurement fields and the
computed excess-risk result appeared together in Arrow and CSV, with original source IDs and recipe
metadata. The source export codec regression remained green. This closes the separate-download gap;
rich original fields in the native inspection table and native-only source exports remain separate work.

## Reusable sample gallery and Compute disclosure

The gallery generates six deterministic synthetic Arrow sources: observations/time, districts with
population/events, origin/destination flows, timed routes, a 16×16 numeric grid and a directed edge
network. Each source uses the same preview, normalization and publication route as uploaded user data.
All retain two source batches and original row IDs. The gallery supplies suggested roles, explains
units and current workflow boundaries, and downloads the original Arrow with versioned sample metadata.
Grid cell tables do not imply raster rendering, and displaying directed edges does not execute routing.

Eight source-contract Node tests passed Arrow transport, deterministic values, exact timestamps,
polygon topology, grid order/nodata and asymmetric directed costs. Trips-specific validation was then
extended with explicit timestamp settings, exact relative-time error, accepted vertices and per-batch
source mapping. The gallery's initial static-path attempt exposed rejection of four-coordinate timed
geometry; the host integration owner wired the explicit Trips route, source preview and saved options.
The final actual-host gallery check passed all six downloads/publications, suggested roles, native
rendered layers, preserved batches, and a portable bundle containing the fixture plus all six samples.
The timed source saved `unix-second` with zero time error. This is sample workflow evidence, not full
coverage of raster/network algorithms or every Trips variant.

The Compute area now uses a native disclosure. It starts closed at viewport widths up to 900px,
retains mounted workflow state when hidden, supports keyboard toggling and reopens for native menu
shortcuts before focus moves. CPU-browser checks passed keyboard activation, preserved edits/counts,
native shortcut focus and 390px control bounds. Visual inspection of the first actual 390px gallery
screenshot found the native sidebar above the compute controls; a responsive stacking fix now places
the disclosure above the pinned sidebar (99) and below native modal overlays (1000). A blocking-sidebar
fixture verifies hit-testing at the corrected layer. The final actual-host mobile check passed initial collapse, keyboard opening/closing, sample
preparation, control hit-testing above the native sidebar and Add Data modal precedence. Final example
typecheck, production build and formatting passed; the existing large-chunk build warning remains.
These checks do not constitute a complete accessibility audit.

## Original values in the native table

The native resident table now offers prepared/derived or original/derived values. Original cells
come from retained Arrow using only the bounded page of GPU-selected source row identities. Float64
and Int64 values retain their exact source representation; timestamps display their exact signed
64-bit words, units and timezone. Nulls, source batches and geometry text remain aligned after filters.
Computed columns receive collision-safe names when an excluded original field has the same name.
Original Float32 and computed scalar sorting use the GPU inspector; unsupported original-type sorting
is explicitly disabled rather than sorting a lossy prepared representation under an original label.

Pages admit at most 16,384 cells and a 1,048,576-character display budget, with at most 4,096 characters
per rich-cell preview. UTF-8 and dictionary string previews decode bounded byte ranges; nested arrays
are traversed with depth/node limits rather than full `toJSON`/`toArray` materialization. These are
display bounds, not a claim about all browser or Arrow allocation. Shortened cells are disclosed and
complete values remain available through original Arrow export. A keyboard-operable wrapping control
exposes the displayed preview without relying on hover text.

Nine focused Node checks passed exact values, sliced/nullable chunks, signed nanoseconds, dictionaries,
Unicode, bounded text/nesting, page admission before readback, stale responses and repeated name
collisions. The actual-host `scripts/original-table.mjs` passed a two-batch source with excluded original
fields and a colliding computed name: original/prepared switching, exact Float64/Int64/timestamps,
geometry, selection, both sort directions, computed sorting, all-source rows, refresh, a 20,000-character
source cell shortened to a bounded preview, and keyboard wrapping. Source revisions, buffer IDs,
batches and upload bytes were unchanged. The screenshot was visually inspected. The first browser
attempt selected population while the operation was off; the fixture now enables the operation first.

Scoped formatting and all three capability-inventory Node checks passed. The subsequent full example
typecheck encountered a separate concurrent `analysis-local-moran-outputs.ts` buffer-type error; its
owner fixed the packed-buffer ownership and reported a passing strict example typecheck. The host
owner is running the integrated formatting/build checkpoint. These focused checks do not replace
full repository gates.

## Native Icon coverage reconciliation

The host integration owner reported passing `scripts/icons.mjs` against the native SVG mesh path:
two meshes, 12,432 rendered pixels, original source IDs 1/2, filtered masks/picking and unchanged mesh,
source and gather buffers across filtering. Fixed/scaled radius and zoom above 14 were verified.
The capability inventory now records partial native Icon support rather than only a reusable
`GPUIconLayer` foundation. Altitude ingestion, text/brushing/globe and portable custom catalog assets
remain unqualified. Native Icons use SVG meshes rather than an atlas in this path.

## Native scalar Charts adapter

The host now explicitly registers the complete optional native chart factory dependency tree and
mounts its manager. Resident chart cards retain native chart configuration/actions, the type picker,
JSON editor and `ChartRenderer`; ordinary host charts retain the upstream panel. The initial resident
execution path supports Big Number count, sum, average, minimum and maximum. Source and selected
populations use borrowed, chunk-preserving vectors and the current explorer mask. GPU dataframe
reductions return only four bytes for count or eight bytes for a value plus validity. Scalar counts
include null rows, matching native behavior; other supported measures exclude missing/nonfinite
measurements, and empty results display native zero with an explicit note. Sum/average currently
require Float32 to avoid silently introducing integer overflow semantics.

Programs retain source leases, reserve conservative working memory before compilation, serialize
reads, suppress superseded results and await retirement before explorer destruction. Geometry-only
sources use the existing selection-mask row/chunk topology for count, without a visible synthetic
column. Original-table pages likewise now request IDs alone when no derived values are needed.
Unsupported resident tooltip/layer charts are intercepted at `MapPopoverContentFactory`, separately
from the panel, so enabling Charts cannot send them into the synchronous native hover aggregator.

Four focused GPU checks passed scalar measures on nullable multi-batch data, source versus selected
population, cancellation, stale revisions, memory admission/cleanup and geometry-only count with no
uploaded scalar column. Four planner/hover-policy Node tests and three provider-lifetime Node tests
passed. Ten original-table Node checks passed the new source-only route. The complete Kepler Node
checkpoint then passed **341 tests in 58 files**. Its first combined run exposed the non-isolated
Node project's preloaded provider module; a hoisted module-cache reset, matching existing host-mock
fixtures, makes the mock deterministic in the full package run.

Strict example typecheck and production build passed at this checkpoint, with existing large-chunk
warnings. The first production attempt crossed an unrelated in-progress `point-idw` barrel export
before its source file existed; the repeat passed after the owner completed that file. The first
actual browser attempt stopped before application initialization: the optional ChartManager had
not yet been registered in the injector. A CPU browser probe with `navigator.gpu` disabled identified
the failure, and after dependency registration it confirmed module initialization without page errors.
After the registration fix, `scripts/native-charts.mjs` passed actual native creation, two resident
datasets, nullable multi-batch measures, map-selected versus source populations, unsupported-statistic
stale-value removal, caption/pin controls, saved configuration replay and chart removal. Source
buffer identities, revisions and upload totals remained unchanged. Screenshot inspection exposed
low-contrast card text/controls, which received scoped dark-panel styles. This evidence does not
establish native grouped/layer chart parity or a fresh full repository build/headless gate.

A CPU-only component browser probe reused the host's native theme context and renderer, with a
mocked scalar response, and confirmed the shared Vite styling instance now supplies the native
text color rather than black. Card labels/inputs use `#edf2f7` on `#242730`; the resident renderer
also overrides its value/caption colors to `#edf2f7`/`#c4cfdd` for readable dark-panel contrast.
The repeated CPU probe verified those final value/caption colors. A source-only browser run that
overlapped dependency optimization failed in React context handling; a fresh run after optimization
settled passed native Big Number creation/count/replay without React errors, alongside six-row,
two-batch original table/export/republication checks with zero scalar uploads. The failure is not
suppressed by the chart adapter.
The latest example typecheck crossed active analysis/DBSCAN edits and reports errors in those
owners' files; it is not a new green combined checkpoint.

## Reconciled host coverage

The H3 row now reflects the host owner's actual native-browser evidence: exact Uint64 IDs, source
batches `[3, 3]`, preserved null/invalid source slots, picked rows 1 and 3, native polygon versus
viewport-column branches, distinct range/category topology behavior, coverage pixels 3,098 to 491,
final-pan refresh and native cache reuse with immutable source buffers. Exact IDs do not imply fp64
coordinates; full geographic fp64 precision and native 3D hover highlighting remain gaps.

Device recovery is also implemented rather than merely planned. The host owner's browser destroyed
the active device, verified stale results disappeared, recovered through the host control and
confirmed a new device/buffers with original IDs, batches `[2, 2]`, native filters and the selected
z-score recipe. Background retry, remote recovery and broader streaming/eviction remain separate.
Original-value table and Arrow-export evidence above replaces the old precision row's already
completed time/geometry gaps; wide arithmetic and additional encodings remain explicitly open.

## Native categorical bar charts

Vertical/horizontal string-category bars now consume dense GPU dataframe reductions over borrowed
source chunks. Count, sum, average, minimum and maximum retain native null semantics; numeric
measures require Float32. The complete dictionary is capped at 512 entries, with explicit admission
against a working-memory ceiling before compilation. Only the bounded group counts, values,
validity and first-row indices return to the CPU (12 bytes per dictionary entry for count, 16 for
other measures). Original data is neither repacked nor read back for plotting.

An additional GPU reduction computes each group's exact first selected source row. This preserves
native insertion-order palette assignments and equal-value sorting instead of substituting dictionary
order. Bounded presentation preserves native top-N, Other (including native aggregation of the
remaining group summaries), custom/default colors and category filter values. A 40-combination
Node oracle compares the output with the pinned native chart implementation across orientation,
measure, grouping and color choices.

Each selected chart borrows a named consumer mask. Active chart-owned CPU filter IDs are excluded,
while other CPU filters on the same field remain active; native GPU-field exclusions and polygon
scopes retain the shared scope policy. The runtime encodes the scope first on the same encoder,
requires its `canConsume` eligibility before encoding chart work, and verifies source/selection
revisions and eligibility again after readback. Pending/failed registration cannot publish a stale
mask. Removal cancels reads, fences dependent work, retires the consumer mask and releases its
source lease/reservation.

Verification at this checkpoint:

- Focused GPU test passed all five measures over nullable batches `[3, 6]`, exact first rows,
  self-exclusion with an unrelated same-field CPU filter retained, source population, cancellation,
  preallocation admission, unchanged source buffers and exact registry-memory restoration.
  Explicit `canConsume: false` returned no result and transferred zero bytes.
- `scripts/native-category-charts.mjs` passed actual bar creation through the native Charts picker,
  source publication through native Add Data, nullable measures, map selection with stable bins,
  unrelated range retention, keyboard toggle, top-N/Other, all-source mode, saved replay and removal.
  Its screenshot was visually inspected. Source revisions, upload totals, batches and buffer IDs
  remained unchanged.
- Nine focused chart Node checks passed, then the full Kepler Node suite passed **372 tests in
  64 files**. The example TypeScript check passed. The coordinated root `yarn lint fix` checked
  8,527 files, formatted nine and validated the lockfile. The final example TypeScript check and
  production build passed after formatting, with the existing large-chunk warning.

Numeric/time bins, multiple grouping series, heatmap, pivot, layer charts, remaining statistics and
analysis-derived fields are still separate work; these checks do not establish full chart parity.

## Numeric interval Charts

Numeric vertical/horizontal bars now share the native Charts manager, renderer and saved configuration.
Float32, Int32 and Uint32 grouping preserves exact source-format domains and bin membership, including
Float32 subnormals and integer extremes. Two GPU phases capture one source/selection revision; stale
results are discarded. Readback consists of 12 bytes of extrema/validity plus bounded group summaries,
with at most 512 bins. Original source buffers and physical batches remain unchanged. Native integer
metadata requires the explicit **Numeric intervals** choice; integer automatic groups, time bins,
multiple series and other chart families remain availability gaps.

Verification for this addition:

- Focused Node checks passed: 12 tests in four files, including 56 comparisons against the pinned
  native numeric-bin implementation, exact thresholds, planner admission and chart provider behavior.
- Two numeric WebGPU tests passed: nullable multi-batch data, all five measures, empty bins,
  self-excluding selection scopes, selection changes between phases, cancellation, memory admission,
  exact source-format edges, nonfinite-only input and ownership cleanup. The categorical GPU test
  passed after the shared grouping change; four scalar GPU tests also passed.
- `scripts/native-numeric-charts.mjs` passed 12 actual-host checks: publication, nullable counts,
  inclusive native range selection, owned-filter exclusion, unrelated-filter domains, empty bins,
  means over all source rows, numeric Other semantics, horizontal saved replay, unchanged source
  buffers, removal and exact Uint32 interval selection despite colliding compact labels. The final
  browser had no page errors; existing React 19 ref console warnings remain. The screenshot was
  visually inspected for readable native bars and controls.
- Initial GPU failures exposed a React dependency in a pure helper and absent validity entries being
  passed as undefined. The pure domain module and omission of absent validity keys fixed both; the
  passing GPU rerun includes those fixes. An initial blank browser startup and a harness assumption
  about native integer inference were corrected before the final passing browser run.
- Final example typecheck and production build passed after the helper fixes and the host owner's
  correction of a nullable Layer.type mismatch. Production still reports large chunks. The host
  owner reports the coordinated `yarn lint fix` checkpoint passed (8,554 files, one formatted,
  lockfile valid), followed by a passing root `yarn build` including CommonJS identity checks.
  The generated matrix matches all 66 reviewed features. These results do not claim a fresh full
  repository test or website build.

## Automatic integer grouping and single-series lines

The resident chart planner/runtime now supports native integer automatic grouping: unique categories
within the configured group budget, then numeric intervals above it. Fallback intervals retain the
native descending-measure ranking, while explicit numeric intervals stay in domain order. A bounded
GPU hash index discovers selected distinct values, with a separate reserved-key flag for full-width
integer extrema. Discovery allows 1,024 keys to distinguish the 512-summary limit from a larger source;
when the native group limit is positive, proven excess cardinality takes the interval path. Unlimited
unique grouping rejects excess/uncertain cardinality explicitly. No source rows are copied to the CPU.

Single-series line charts reuse these summaries. Integer/string series preserve native alphabetical
ordering, first-occurrence palettes and exact singleton range selections; real axes use native numeric
bins. Line charts ignore bar top-N/Other controls, which are hidden in their editor. Time axes and
multiple series remain unsupported.

The focused GPU run passed five tests in three files: two new integer-auto tests, both numeric tests,
and the categorical regression. It covers selected category/interval transitions, nullable batches,
canonical Float32 nonfinite/zero categories, Int32/Uint32 extremes, 513 and 1,537 distinct values,
stale selections, unchanged source ownership and memory restoration. CPU oracles cover 300 integer
configurations plus native single-series string/integer/real line results.

The native browser initially exposed ingestion re-inferring Uint32 values near 2^32 as timestamps.
The host owner added an explicit prepared-Arrow validation path; its focused tests and example types
passed. The final mixed-source browser then passed all 16 checks, including automatic integer groups,
ranked interval fallback, exact singleton and interval Uint32 selections, all previous numeric bar
checks, native line rendering/replay/point selection, unchanged source buffers and removal. No page
errors occurred; existing React 19 ref warnings remain. The native line screenshot was inspected.
A flat SVG polyline has zero bounding-box height, so the harness checks its containing SVG rather
than using Playwright's visibility test on that polyline alone.

The final focused Node run passed 12 tests in four files, including the 300 integer cases, three
line cases and 56 earlier numeric-bin comparisons. Read-only chart cards now also omit the manual
refresh control and GPU diagnostics; map-image export qualification belongs to the export owner.
Final example typecheck and production build passed after these changes; the production build
retains the existing large-chunk warning. The generated capability matrix matches all 66 entries.

## Geographic distance bands and exact polygon contiguity

Distance-band weights now share the revision-keyed point cache with kNN. The radius is inclusive
and uses spherical great-circle meters, including antimeridian neighbors. Explicit directed-slot
capacity is bounded to 1,048,576 and source rows to 65,536. Overflow blocks result publication and
reports the required capacity; conditional inference rejects active degree above 64 without truncation.
The focused GPU run passed both cases, including overflow/recovery, selected populations, chunks,
cache reuse and dense inference refusal. The native workflow passed after explicitly choosing kNN
for its unrelated hot-spot fixture. Logs: `/tmp/kepler-distance-band-gpu.log` and
`/tmp/kepler-distance-band-native-retry.log`.

Polygon spatial rates, hot spots and Local Moran now use original Float64 XY equality for queen
(shared vertex) or rook (whole shared undirected edge) adjacency. Bounded Arrow preparation interns
coordinates to exact integer tokens before GPU upload. It preserves source rows/chunks, holes and
polygon parts in collections, while excluding nonpolygon/empty/malformed rows from statistics.
Valid isolated polygons retain the selected island policy. No snapping, partial-edge equivalence,
noding or coordinate normalization is claimed. Retention and preparation enforce 4,096 source rows
and 65,536 inspected vertices; oversized geometry remains renderable. The topology cache reserves
CPU/GPU memory and separately reports candidate overflow (a lower bound) and exact neighbor overflow.

The focused polygon GPU fixture passed: queen/rook results, both overflow paths and recovery,
selected-population reuse, original source identities/chunks, analytic and seeded conditional Moran,
hot-spot table outputs and ownership cleanup. The actual native browser passed multibatch Prepare
data publication, queen rates 15–40 versus rook rates 15–15, saved recipe restoration, derived table
rows, conditional Moran configuration and source removal. No page or WebGPU errors occurred;
existing React 19 ref console warnings remain. Logs: `/tmp/kepler-polygon-focused-gpu.log` and
`/tmp/kepler-polygon-native.log`. The integrated example typecheck and 25 focused replay Node checks
also passed. Replay validates geometry as a source role separately from Float32 measures and rejects
unavailable topology before native configuration changes. These are focused integration checks,
not a new full-repository or website-build claim. Polygon Quantile LISA is the next implementation
slice and is not covered by these passing polygon results.

Polygon Quantile LISA subsequently passed the consolidated three-case GPU run, including the
existing point-analysis regression after extracting their shared computation. The polygon case
checks source geometry validity before quantiles, valid islands included in boundaries but excluded
from inference, original chunks and numeric ports, selected-population boundaries, cached topology
and deterministic recipe replay. The native workflow also passed actual polygon Quantile LISA
configuration and recipe restoration with its explicit legacy policy. Logs:
`/tmp/kepler-polygon-quantile-focused-gpu.log` and `/tmp/kepler-polygon-quantile-native.log`.
Whole-example types and focused shared/replay Node checks passed. The coordinator's subsequent
shared checkpoint passed `yarn lint fix` (8,587 files, no fixes, lockfile valid), root `yarn build`
(all modules and CommonJS identity), and 466 Node tests in 78 files. These results precede the next
projected-coordinate implementation and do not certify those later changes.

### Dictionary series for resident charts

The chart planner now accepts a second prepared string dictionary for bars and lines. The product
of primary groups and (series dictionary size + one missing-label slot) must not exceed 4,096 cells.
Positive integer-auto limits allocate only the maximum of their unique limit and interval count;
unlimited unique charts retain their 512-group capacity. A GPU pass creates composite keys without
combining source chunks or reading source rows. GPU count/sum/mean/min/max and first-row reductions
return bounded cells. Folding preserves native sums of per-series statistics, per-group series order,
missing-label row counts, empty numeric bins, native palettes and category ranking/Other behavior.
The native renderer plots the aggregate. The new Chart series control and accessible breakdown table
expose individual values and row counts without claiming stacked bars or distinct series lines.
Numeric series fields remain explicitly unsupported.

Verification after implementation:

- `nvm use`: Node 22.23.4.
- Five focused Node files: 15 tests passed, including 60 category-series and 15 numeric-series native
  oracle comparisons, planner capacity/identity and existing numeric/integer formatting checks.
  Log: `/tmp/kepler-series-node.log`.
- Four changed GPU runtime files: six tests passed in 7.77 seconds, covering categorical, interval and
  unique-integer series, null/empty labels, all five measures, first rows, selected/owned-filter scopes,
  same-field grouping, chunk offsets, cancellation and ownership cleanup. Existing category/numeric/
  integer regressions also passed. Log: `/tmp/kepler-series-gpu.log`.
- The actual native browser passed all 12 checks: control/table, sum of means, missing labels, series
  order, self-filter exclusion, saved replay, aggregate line, integer groups, horizontal interval bars,
  unsupported numeric series, series removal and unchanged source buffers/uploads/revisions/chunks.
  The first page remained blank at startup; a fresh page passed without HTTP/request or page errors.
  Existing React 19 ref console warnings remain. Log: `/tmp/kepler-native-series-charts.log`.
  Screenshot: `/tmp/kepler-native-series-charts.png`.
- Final example typecheck and production build passed after the concurrent planar type seam settled
  (build 1.51 seconds; existing large-chunk warning). Logs: `/tmp/kepler-series-types.log` and
  `/tmp/kepler-series-build.log`. The capability matrix check passed for all 66 features.
- Scoped Biome formatting applied. The coordinator's subsequent `yarn lint fix` checkpoint passed
  8,600 files with no fixes and a valid lockfile; its root `yarn build` passed all modules and CommonJS
  identity. Those shared checks include this series implementation.

These are focused chart checks, not a new full `yarn test` or website-build claim.

## Explicit projected-meter point analysis

Point rates, hot spots, Local Moran and Quantile LISA now accept either the unchanged geographic
coordinate recipe or explicit already-projected Float32 X/Y, meter units, a declared CRS and saved
bounds. The shared builder, borrowed-weight validation, cache identity, routes, controls and replay
resolve the active roles. Planar distance uses the core Euclidean metric; no coordinate reprojection
or geographic map position is inferred. Bounds and radius validate Float32 distance capacity; the
packing pass uses a nonfinite invalid marker so valid projected X=181 is preserved.

The consolidated focused GPU run passed all five tests in three files: the new projected case plus
existing geographic point and distance-band regressions. It covers capacity rejection/recovery,
nullable [2,3] batches, valid X=181, selected-population cache reuse, exact saved CRS/bounds/roles,
Moran, Quantile LISA and projected kNN. The final native browser passed table-only publication,
projected distance-band controls, recipe restoration, original/derived table rows and source removal.
Logs: `/tmp/kepler-projected-focused-gpu.log` and `/tmp/kepler-projected-native-final.log`.
No page/WebGPU errors occurred; existing React 19 ref warnings remain. The first native fixture
incorrectly declared X non-nullable in its first batch despite a later null, which ingestion correctly
rejected. A second harness locator expected a generic heading rather than the actual conditional
geographic/projected heading. Both fixture issues were fixed without changing runtime behavior.

Whole-example/module types and 14 focused shared Node tests passed. The coordinator also reran
`yarn lint fix` (8,600 files, no fixes, lockfile valid) and root `yarn build` successfully before GPU
acceptance. Later Dorling work is outside this checkpoint. Full repository tests and website builds
were not rerun for this addition.


## UTC calendar Charts

Resident line charts now group prepared millisecond Arrow Timestamp/Date fields by explicit UTC
month or year. Each original chunk feeds GPU calendar extraction and exact signed-word Date-range
checks. Bounded distinct keys are sorted chronologically, then mapped to category reductions for
count, sum, mean, minimum and maximum, optionally split by a string dictionary. No source timestamp
array is read back. Limits remain 512 observed periods and 4,096 period/series cells. Periods that
cross JavaScript's Date boundary are rejected explicitly; automatic/hour/day/week intervals and
microsecond/nanosecond sources are not admitted by this milestone.

Native compatibility includes years 0–99, negative years, pre-epoch instants and Date(null)'s epoch
bucket. Chart-owned exact time filters can include original nulls when their inclusive range
contains epoch zero. Ownership is resolved from current chart configuration, dataset, axis field and
filter ID, including arbitrary persisted IDs. Removing the owning chart restores ordinary time-filter
null exclusion. Valid but out-of-Date-range timestamps never become synthetic epoch rows. Period
clicks retain the pinned native inclusive endpoints, including a row at the next period's exact start.

Verified before the subsequent separate-series plot and seconds-source changes:

- Node 22.23.4; 11 focused tests in three files passed, including ten pinned native month/year ×
  measure combinations, chronology, period endpoints, admission and trusted filter ownership.
  Log: `/tmp/kepler-calendar-node.log`.
- Two GPU runtime tests passed, covering original three-chunk inputs, all five measures, dictionary
  series, selected scopes, own-filter exclusion, cancellation, source immutability, null epochs,
  out-of-range exclusion, 512/513-period capacity and resource cleanup.
  Log: `/tmp/kepler-calendar-gpu.log`.
- Three provider lifecycle regressions passed after their mock dataset gained the required empty
  `timeColumns` metadata. Log: `/tmp/kepler-calendar-provider-node.log`.
- The actual native browser passed 15 checks: UTC month/year creation, count/average summaries,
  dictionary series, epoch/null selection, persisted arbitrary filter IDs, ordinary null exclusion,
  inclusive March selection, unsupported day interval recovery, removal and unchanged source
  revision/uploads/buffer identities/chunk boundaries. No page, GPU or request errors occurred;
  existing React 19 ref warnings remain. Log: `/tmp/kepler-native-calendar-charts.log`;
  screenshot: `/tmp/kepler-native-calendar-charts.png`.
- Scoped Biome and example types passed. The coordinator's shared `yarn lint fix` checkpoint checked
  8,639 files, formatted four and validated the lockfile; root `yarn build` then passed all modules
  and CommonJS identity. Later series-renderer integration is outside that checkpoint.

The native time-filter histogram is a separate surface: the browser still reported its missing
compiled-interval metadata while these chart filters operated correctly. Its implementation is
tracked separately. These results do not establish full calendar/zone/playback or chart parity and
are not a fresh full repository test or website-build gate.

## Dorling host implementation and native rendering acceptance

The bounded Dorling adapter, saved recipes, source-aligned table fields, projected-coordinate
controls, Web Mercator result overlay and bounded hover/pinned inspector are implemented.
The host retains finite exhausted layouts and suppresses only the overlay when full circle extents
cross the finite Web Mercator world. Arbitrary projected CRSs remain table-only.

- The focused host GPU fixture passed source-chunk preservation, validity and selected masks,
  zero-radius rows, explicit exhaustion, recipe replay, world-bounds suppression/recovery and
  resource cleanup: `/tmp/kepler-dorling-focused-gpu.log` (one test).
- Replay and capacity Node checks passed 26 tests:
  `/tmp/kepler-cartogram-replay-node-final.log`. The Deck projection oracle passed three cases;
  bounded picking guards passed four cases after the disposal fix.
- The first native attempt found inaccessible field labels, now explicitly labelled. The second
  found missing vertex-buffer usage and synchronous redraw reentry during picking teardown:
  `/tmp/kepler-dorling-native.log` and `/tmp/kepler-dorling-native-retry.log` are failures, not
  acceptance evidence. Owned outputs now request vertex usage, disposal detaches callbacks
  before clearing, and internal inspection resets do not notify during encoding.
- Scoped formatting and the example typecheck passed after those fixes:
  `/tmp/kepler-cartogram-lifecycle-types.log`. Native rendering acceptance was still pending at
  that checkpoint; the final bounded workflow below covers actual pixels and inspection.
- A subsequent lifecycle review found that clearing inspection could be undone by an in-flight
  row read. Clearing now disposes its handlers before resetting the UI; reads retain their fences
  but cannot publish after disposal. Scoped formatting and the example typecheck passed again:
  `/tmp/kepler-cartogram-clear-types.log`.
- The next native attempt passed publication, layout and recipe restoration without the earlier
  GPU/reentry errors, then timed out at the camera gate:
  `/tmp/kepler-dorling-native-lifecycle.log`. That gate incorrectly required compute executions
  after a camera-only change. It now uses the submitted rendering timestamp and source revision,
  followed by an animation frame; the actual pixel and picking assertions remain unchanged.
- The corrected native workflow passed with exit 0:
  `/tmp/kepler-dorling-native-render-frame.log`. An explicitly projected NYC fixture retains source
  chunks `[2, 3]`, with two overlapping circles, one stationary zero-radius row and null/negative
  exclusions. All 89,628 sampled predicted interior pixels changed blue when compared with the
  disabled overlay; none of the 95,386 sampled outside pixels changed. Actual mouse hover reports
  original row/source ID 0 and its radius, click pins inspection, mouseout preserves the pin, and
  clearing removes it. Saved recipe restoration, native center/radius/identity table columns,
  source batches and dataset/overlay removal also passed. No page or GPU errors occurred; existing
  React 19 ref warnings remain. This is bounded native evidence, not general cartogram parity.

This checkpoint does not include a new full repository test or website-build gate.

## Rate result diagnostics

Excess risk and point/polygon spatial rates now display existing compact summary counts separately:
rows included in the result summary, defined outputs, NaN undefined outputs and source rows excluded
by the combined mask. Shared-adapter and host reviews confirmed the mask combines prepared input
validity, selected population where requested, and spatial eligibility for spatial rates. Included
rows can still produce NaN; defined outputs include infinity. The UI warns when nonempty result
min/max contain infinity and does not infer how many outputs are infinite. Whole-source topology
counts remain separate and may overlap the combined exclusions.

This change adds no GPU work, readback, input-validation policy or cause-specific counts. Scoped
formatting and the example typecheck passed (`/tmp/kepler-rate-diagnostics-types.log`). Existing
projected/polygon browser selectors were updated to the labelled result count and syntax-checked;
their browser workflows were not rerun for this UI-only change.


## Separate series plots and exact second calendar sources

The resident chart card now renders bounded per-series vertical bars, horizontal bars and lines.
Stable name-based colors and line styles identify series across filtering and category order changes;
a finite palette can share colors, so legends and line styles remain available. Negative and zero
values share the zero baseline. Missing observations break line segments instead of inventing zeros.
Mouse, Enter and Space select the exact parent category or numeric/time interval, and every mark in
that parent shows the selected state. Native top-N Other remains an outlined, aggregate-only mark
without a fabricated series breakdown or selectable filter. Read-only cards remove mark interaction
and expand clipping; standalone export capture remains a separate unqualified workflow.

Calendar lines now also admit Arrow TimestampSecond. Exact signed-word multiplication checks Date
range before converting each original chunk into graph-owned millisecond scratch; the runtime
reserves an additional eight bytes per source row. Original validity remains separate from time
words, so a null row can become the native epoch bucket while a valid Int64 extreme or out-of-Date
instant cannot. Microsecond/nanosecond conversion primitives are tested but their chart admission
remains gated: Arrow's fractional Number rounding and exact raw-source click semantics need a
coherent policy before enabling them.

Final focused verification:

- `nvm use`: Node 22.23.4. Calendar CPU: **12 tests in three files passed**; series layout/accessibility
  CPU/SSR: **six tests passed**. Logs: `/tmp/kepler-second-calendar-node.log` and
  `/tmp/kepler-series-view-node-final.log`.
- GPU: **five tests in two files passed** in 6.42 seconds. The primitive tests all four time units
  against exact BigInt references at word, Date and fractional boundaries, original chunk offsets,
  repeated encodes and immutable inputs. Integrated month/year charts run for both admitted units;
  raw-second selections verify null epochs, invalid clipping and inclusive December/January and
  February/March boundaries. Log: `/tmp/kepler-calendar-seconds-gpu.log`.
- Actual native separate-series browser: **13 checks passed**, including negative/zero grouped bars,
  Enter category selection and self-filter exclusion, unselectable Other, missing line segments,
  integer groups, horizontal interval bars, Space selection of exact [1.5,2], saved replay,
  unsupported numeric series, removal and unchanged source uploads/revisions/buffers/chunks.
  Log: `/tmp/kepler-native-separate-series.log`;
  screenshot inspected: `/tmp/kepler-native-series-charts.png`.
- Actual calendar browser: **15 checks passed for milliseconds and 15 for seconds**, including
  separate-series marks/legends/segments, persisted arbitrary chart filter IDs, ordinary null
  exclusion without the owning chart, exact period selection and unchanged [2,5] batches.
  Logs: `/tmp/kepler-native-calendar-millisecond.log` and `/tmp/kepler-native-calendar-second.log`.
  Screenshot inspected: `/tmp/kepler-native-calendar-millisecond-charts.png`.
- No page, request or GPU errors in these native runs; existing React 19 ref warnings remain.
- Whole-example typecheck and production build passed (1.56 seconds; existing large-chunk warning).
  Logs: `/tmp/kepler-calendar-series-types-final.log` and
  `/tmp/kepler-calendar-series-build-final.log`. Scoped Biome passed and the generated matrix still
  matches all 66 reviewed features. A new shared root formatting/build checkpoint is coordinated
  separately; these are not fresh full repository test or website-build results.

The next bounded calendar extension is explicit UTC day and Sunday-start week intervals. Native
weeks use utcWeek, not ISO week numbering. Signed epoch-day keys fit the existing domain, while
full Date-range hour ordinals exceed Int32 and need a wider or staged representation. Numeric
series, remaining measures, heatmap/pivot/layer charts and derived-field lifetimes remain explicit
review gaps.


## Shared checkpoint and capability ledger reconciliation

The coordinator's next `yarn lint fix` checked **8,654 files** cleanly and validated the lockfile;
root `yarn build` passed all modules and CommonJS identity. This checkpoint includes separate-series
rendering, exact seconds and the authored day/week extension. Day/week GPU/native acceptance was
still queued at this checkpoint; compilation alone does not qualify those new intervals.

The Heatmap layer and media-export matrix rows now reflect the existing accepted evidence in the
[implementation verification ledger](../../roadmaps/kepler-compute-integration-implementation.md),
under **Native Heatmap workflow** and **Native map image export**. This is a ledger correction,
not a new execution claim. The former includes native planar WebGPU SUM/MEAN, radius, nullable
weights, linear/sqrt/log normalization, selection/domain scopes and stable source buffers. The
latter includes real PNG pixels, preview equality, transparency, legends/read-only chart content,
resolution/aspect changes, cancellation and retained device/source identity. Both remain partial:
heatmap projection/scale/picking constraints and PNG limits/CORS apply, while additional export
combinations, animation/video and standalone HTML remain unqualified or unimplemented.


## UTC day and Sunday-start week Charts

Day and week lines now use exact signed epoch-day keys derived from GPU calendar year/day-of-year
outputs, with explicit floor division for negative years. Sunday-start weeks match the pinned native
utcWeek convention; source nulls belong to the week beginning 1969-12-28. Calendar extraction still
uses each original source chunk, and day-of-year replaces month as the one additional output, so the
existing memory allowance remains sufficient. Period endpoints retain native inclusive selections.
A valid minimum-Date daily period is admitted; periods whose start/end cross Date limits are rejected.

Series plots now reserve bounded axis-label space for up to 24 characters, keeping complete weekly
dates readable without changing exact keys, accessibility labels or selected filters. Longer labels
retain their full titles and accessible descriptions.

Verification:

- Calendar CPU **six tests passed**, including **20 interval × measure native comparisons**, UTC
  midnight/Sunday boundaries, leap/DST dates, null epochs, negative years and Date boundaries.
  Planner/time-normalization CPU **seven tests passed**. Series layout/SSR **seven tests passed**,
  including weekly label visibility in vertical/horizontal/line and read-only rendering.
  Logs: `/tmp/kepler-calendar-day-week-node.log`, `/tmp/kepler-day-week-node.log` and
  `/tmp/kepler-series-label-node.log`.
- Calendar GPU **five tests passed** in 5.59 seconds, covering both second/millisecond sources,
  pre-epoch/year 0/year 99/negative years, Sunday ±1ms, null epoch, invalid Date exclusion,
  minimum/maximum Date period boundaries, 512 accepted/513 refused periods, preserved source
  chunks/buffers and cleanup. Log: `/tmp/kepler-calendar-day-week-gpu.log`.
- Native calendar browser **20 checks passed for each timestamp unit**, including exact leap-day
  [Feb29,Mar1] selection, the null-inclusive [Dec28,Jan4] week, Sunday [Mar10,Mar17] keyboard
  selection, own-filter exclusion, saved month/year replay and unchanged source [2,5] batches.
  Logs: `/tmp/kepler-native-day-week-millisecond.log` and `/tmp/kepler-native-day-week-second.log`.
- Native series regression **13 checks passed** after the label-layout change, including negative
  and zero bars, missing line gaps, Enter/Space filtering, Other and unchanged source identity.
  Log: `/tmp/kepler-native-series-labels.log`. Weekly, vertical-bar and horizontal-bar screenshots
  were inspected: `/tmp/kepler-native-calendar-millisecond-week.png`,
  `/tmp/kepler-native-series-bars.png`, `/tmp/kepler-native-series-horizontal.png`.
- The first week browser assertion incorrectly expected 4.33 where the pinned native formatter
  returns 4.333. Correcting only the fixture resolved it. Final runs had no page/request/GPU errors;
  existing React 19 ref warnings remain.
- Final whole-example typecheck passed after the concurrent interpolation seam settled; production
  build passed in 1.96 seconds with the existing chunk warning. The shared 8,654-file formatting and root build
  checkpoint above includes this code. Logs: `/tmp/kepler-day-week-types-final.log`,
  `/tmp/kepler-series-label-types.log`, `/tmp/kepler-day-week-build.log`.

Hour admission remains disabled. An isolated `makeResidentHourPeriods` helper and three Node tests
now decode bounded dense-day/hour keys across the full Date range without narrowing epoch hours to
Int32. This helper is not connected to chart execution; its required second GPU domain and lifecycle
integration remain future work. Log: `/tmp/kepler-hour-periods-node.log`. Automatic intervals,
submillisecond chart-source semantics and the other chart-family gaps remain separate work.

## Ordinary kriging implementation and host acceptance

The shared adapter, saved geographic recipe, manual variogram controls, prediction map, bounded
cell prediction/variance inspector and separate numeric GeoTIFF exports are implemented. Existing
GPUKriging performs the solve; no new solver, implicit reprojection, automatic fit or cross-validation
was added. Source/grid caps match IDW, with at most 16 retained neighbors. Final cell validity
requires finite prediction and nonnegative finite variance; core invalid/uncertain counters remain
separate from authoritative final nodata counts.

Five focused shared parser/routing and IDW compatibility Node checks passed, as did module types.
Two independent CPU GeoTIFF decode checks passed, including explicit metric/unit-exponent metadata
and legacy IDW compatibility. Scoped formatting and consolidated example types passed; after a
review found a constructor-failure reservation leak, graph creation moved into guarded cleanup and
example types passed again (`/tmp/kepler-kriging-cleanup-types.log`). Graph scratch is planned and
admitted before compilation; reservations include owned outputs, texture and exact parameter bytes.
Driver-private shader allocations remain outside this accounting.

The focused host GPU fixture and native workflow were prepared but had not run at that intermediate checkpoint; final acceptance follows below.
Earlier shared integration before the new kriging edits passed `yarn lint fix` (8,654 files), root
module/CommonJS build, example types and 548 Kepler Node tests across 97 files. That earlier checkpoint
does not establish GPU/native kriging acceptance. No fresh full repository or website gate is claimed.

Final ordinary kriging acceptance:

- The first host attempt rejected a duplicate writable variance-buffer import before solver work
  (`/tmp/kepler-kriging-focused-gpu.log`). Output and profile now share one graph view.
- The next run passed kriging and exposed an added discriminator in the existing IDW cell payload
  (`/tmp/kepler-kriging-idw-focused-gpu.log`). The legacy IDW payload was restored rather than
  weakening its regression assertion. Both focused GPU fixtures then passed, 2/2:
  `/tmp/kepler-kriging-idw-final-gpu.log`. They cover independent symmetric spherical-variogram
  variance, exact hits, invalid samples, selected neighborhoods, insufficient and uncertain solves,
  recipe restoration, immutable source buffers, pending export lifetime and zero-leak cleanup.
- The initial native attempt found coordinate selects without clean accessible names
  (`/tmp/kepler-kriging-native.log`); explicit labels fixed them. The next passed computation,
  inspection, downloads and pixels, then stopped because the fixture switched analysis off and
  hid its own Restore button (`/tmp/kepler-kriging-native-final.log`). The fixture now switches
  to z-score for the overlay comparison, preserving every assertion.
- The final native workflow passed with exit 0: `/tmp/kepler-kriging-native-accepted.log`. A nullable
  four-row `[2, 2]` source configured through Prepare data produces three exact hits with prediction
  4 and variance 0, plus a fourth prediction of 4 with finite positive variance. The cell inspector
  reports prediction/variance without fabricated sample counts. Both downloaded single-band
  GeoTIFFs independently decode with numeric values, geographic placement, complete recipe and
  explicit metric/unit-exponent provenance. All 4,096 sampled bitmap interior pixels change when
  the overlay is removed at a fixed camera. Recipe restoration, unchanged source revision/uploads/
  buffer identities/batches, and source/overlay removal pass. No page or GPU errors occurred;
  existing React 19 ref warnings remain.
- The coordinator's final shared checkpoint passes `yarn lint fix` (8,660 files clean), all-module
  build and CommonJS identity, example types, and 554 Kepler Node tests across 99 files. The final
  fixture-only replay correction passed syntax checking and the native workflow above.

These checks establish the bounded ordinary-kriging workflow, not automatic model fitting, calibrated
uncertainty, global spatial-model parity or a fresh full repository/website gate.


## Staged hourly keys and final calendar requalification

At this earlier checkpoint the hour extension had isolated GPU key infrastructure in
`resident-calendar-keys.ts`, alongside the previously tested bounded CPU hour decoder.
In hour mode the helper extracts a sanitized hour,
discovers epoch days first, and can map them into dense day/hour keys using a separate day-domain
buffer. Its optional vectors and compiled graph resources are accounted for; the shared class now
releases each owned raw buffer once while vectors own their wrappers. The second domain and chart
runtime lifecycle were not yet connected, and planner/runtime admission rejected hour charts.
The subsequent qualification below supersedes that staged status.

At this checkpoint the future GPU runtime fixture was an authored, unexecuted draft outside test
discovery. It has since moved to
[resident-chart-hour-runtime.spec.ts](../../../modules/kepler/test/resident-chart-hour-runtime.spec.ts)
for the subsequent runtime qualification. The earlier draft is not verification evidence.
The native calendar fixture keeps its 20 accepted checks by default; future hour assertions require
an explicit `--hour` argument and must not be run as accepted coverage until runtime admission is
implemented and qualified. No test was skipped to imply hour support.

Because the key infrastructure also changed shared allocation/cleanup paths, the earlier calendar
GPU/native results were treated as preceding that change. Current-code requalification passed:

- **Five existing calendar GPU tests passed** in 5.44 seconds, covering both admitted timestamp
  units, all current intervals, nulls, signed/Date boundaries, selection scopes, series, capacity,
  source chunks and resource cleanup. Log: `/tmp/kepler-calendar-final-regression-gpu.log`.
- Default native calendar **20 checks passed for milliseconds and 20 for seconds**, with
  `includeHour:false`, exact filter replay, unsupported hour behavior and unchanged source identity.
  Logs: `/tmp/kepler-native-calendar-final-millisecond.log` and
  `/tmp/kepler-native-calendar-final-second.log`. No page/request/GPU errors; existing React 19
  ref warnings remain.
- The shared final checkpoint passed `yarn lint fix` across **8,660 files**, lockfile validation,
  all-module `yarn build`, CommonJS identity, example TypeScript and **554 Kepler Node tests across
  99 files**. Full repository `yarn test` and both website build gates were not rerun for this tranche.

The point and time-filter matrix rows also now reflect the coordinator's separate accepted altitude
and native-histogram evidence. Point altitude covers prepared XYZ, pitched picking, XY/XYZ toggles,
null-zero/nonfinite-hide policy, filtering, portable replay and completed-stream handoff; icon/line/arc
altitude and richer label/brushing contracts remain gaps. Native time histograms cover bounded
compiled edges, exact classification, nullable epochs, own-filter exclusion and Month/Year controls;
the 256-bin limit and wider time-zone/playback contracts remain explicit. These are ledger updates
from the [implementation evidence](../../roadmaps/kepler-compute-integration-implementation.md),
not additional runs performed by this chart worker.

## Hourly charts and saved-chart resource lifetime

The hourly runtime now connects two bounded GPU domain passes: observed epoch days, followed by
dense day/hour keys. This avoids overflowing signed epoch-hour keys at the ends of the supported
Date range. Second and millisecond sources retain their original chunks and source buffers. Each
readback phase checks the captured source/selection revision, cancellation and consumer eligibility;
both domains and their transient buffers count against the reservation. Hourly charts retain the
512-observed-period and 4,096-cell limits, exact UTC boundaries and chart-owned null epochs.

Initial GPU qualification passed, but the native saved-hour replay exposed a real lifetime error:
a newly mounted chart could borrow its predecessor's named selection mask before that predecessor
finished retiring. The later cleanup destroyed a buffer still referenced by the new reader's compiled
bindings. `ResidentChartProvider` now keeps a per-chart retirement barrier. A replacement waits for
the previous read and fenced destruction, then rechecks currentness before allocating. Other chart
IDs remain independent. Public hourly admission was gated again while this was fixed and tested;
the bind-group error was not suppressed and replay assertions were retained.

- **Six provider CPU tests pass**, including pending-read/fence retirement, repeated same-ID
  remounts, unrelated charts and disposal during replacement. Removing only the barrier makes two
  regression tests fail. Logs: `/tmp/kepler-chart-replay-node.log` and
  `/tmp/kepler-chart-replay-before-fix-node.log`.
- **Sixteen focused CPU tests across three files pass** after restored hourly admission: provider
  lifecycle, calendar planning/ownership/capacity, and full-Date-range hour decoding.
  Log: `/tmp/kepler-hour-qualified-node.log`.
- **Ten GPU tests across three files pass** in 8.01 seconds. The real replay regression retains a
  pending read, removes/recreates the same chart ID, explicitly collects retired selection masks,
  then executes the restored chart again. It verifies exact summaries, unchanged source bytes and
  chunks, and reservation cleanup. Hour fixtures cover both timestamp units, pre-epoch/leap/DST
  and full-Date-range boundaries, nulls, dictionary series, 512/513-period limits, and cancellation
  after either domain readback. Existing calendar regressions also pass.
  Log: `/tmp/kepler-chart-replay-gpu.log`.
- With public admission still gated, **25 native checks for milliseconds and 25 for seconds pass**
  through an explicitly test-only served-planner override. Both include the original failing
  day-to-saved-hour replay, same-ID removal/readdition, persisted filter identity, exact keyboard
  selections, ordinary-versus-chart-owned null behavior and unchanged source identity. The first
  override attempt checked interception before module load; moving that assertion after app
  readiness repaired the fixture without relaxing it. Logs:
  `/tmp/kepler-native-hour-replay-millisecond.log` and `/tmp/kepler-native-hour-replay-second.log`.
  The temporary override has been removed and public hour admission restored for final smoke.
- **Final public-path native smoke passes 25 checks for milliseconds and 25 for seconds**, with
  no served-source override. Both repeat the failing replay and explicit same-ID removal/readdition,
  exact ranges, source identity and automatic-interval rejection. No page/request/GPU errors occur;
  existing React 19 ref warnings remain. Logs: `/tmp/kepler-native-hour-public-millisecond.log`
  and `/tmp/kepler-native-hour-public-second.log`.
- **Whole-example TypeScript passes** with restored public admission and the concurrent Group By
  seams present. Log: `/tmp/kepler-hour-qualified-types.log`.

The hourly screenshot was inspected for controls, UTC labels, distinct series and accessible
horizontal overflow. Automatic intervals, finer timestamp-source units, numeric series, additional
chart families and statistics remain explicit gaps. Full repository tests and website gates are not
established by these focused checks; the shared formatter/build checkpoint is tracked separately.

## Native Group By integration in progress

The original Group By panel now feeds a resident all-row planner and GPU aggregation adapter.
The bounded contract admits one unordered string dictionary key, up to 16 count/sum/average/minimum/
maximum measures, 1,000,000 source rows, 4,096 dictionary slots including null, 1 MiB key text and
4 MiB materialized results. Count includes null measure rows; all-null numeric groups remain null.
Groups retain first source occurrence. Numeric aggregation uses explicit Float32 semantics and rejects
nonfinite requested numeric inputs/results. Unsupported configurations remain editable with SQL
guidance; no implicit SQL execution occurs.

Completed output is a separate table snapshot with bounded result Arrow and adopted GPU columns.
Saved metadata retains native field/analyzer types and historical source revision. The pinned native
reducer explicitly preserves completed derived snapshots when their parent is deleted; only pending
operations referencing the removed dataset are discarded. Portable replay therefore validates the
result's own identity/schema and does not require its historical parent to remain present.

- Planner and pinned-native CPU oracle: **4 tests passed**.
- Host operation controller: **15 CPU tests passed**, as reported by the host implementer.
- Sandbox host: **29 tests passed**, including completed-result restore with an absent historical
  parent and rejection of mismatched result ID, field order and invented geometry roles.
  Log: `/tmp/kepler-group-by-bundle-node.log`.
- Integrated example TypeScript passed after the provenance/publication seams landed.
  Log: `/tmp/kepler-group-by-provenance-types.log`.
- Scoped Biome checks passed for the parent publication, persistence and test changes.

Atomic CPU/GPU reservation transfer and the adoption runtime are still being finalized. Shared and
host GPU fixtures are authored but **unrun**; the native workflow script is being prepared. The guide
marks this lane partial pending those checks. No full repository tests or website gates are implied.
