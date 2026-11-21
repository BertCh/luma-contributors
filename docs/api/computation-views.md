---
title: Computation views authoring guide
description: Embed recorded, explorable views of a computation in the docs with the lens directive.
---

# Computation views authoring guide

:::caution Experimental
`@luma-contributors/lenses/computation-views` and `@luma-contributors/lenses/computation-lenses` are
experimental entry points. The directive syntax, record files and views may change between releases.
:::

A **lens** is a view of one real computation at one level of abstraction: the formula an operator
computes, one datum's receipt, the iterations a loop actually ran, the kernel dispatch, or the command
graph. Every lens is drawn from a recorded run of the real GPU computation, never from a CPU replay. The
same Markdown renders on this site and in the Reference section of the spatial analysis showcase
(`examples/deck/spatial-analysis-showcase`).

This page is for people writing the operator reference. The record format is documented in
[GPU computation records](/docs/api/computation-record).

## The lens directive

A lens is a Markdown leaf directive on a line of its own:

```md
::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=17 caption="Local Moran I of tract 17"}
```

| Attribute | Required | Meaning |
| --- | --- | --- |
| `scenario` | yes | Registered lens scenario, `<entry>.<element>[.<variant>]` in kebab case, for example `gpu-spatial-analysis.local-moran` |
| `rung` | yes | Level of abstraction to show; see [Rungs](#rungs) |
| `bind` | no | Page coordinator the lens links through; lenses with the same name share selections. Defaults to `page` |
| `datum` | no | Non-negative integer: the entity (row, cell, node) the view opens bound to |
| `iteration` | no | Non-negative integer: the loop iteration the view opens at |
| `caption` | no | Quoted caption. Without it the generated description is the caption |
| `view` | no | Id of a registered view to use for the rung instead of the preferred one, for example `neighbourhood-view` |

Unknown attributes, an invalid scenario id or an unknown rung fail the website build with the file and
line. The showcase prints the same error in place instead of failing.

## Rungs

| Rung | View | Use it for |
| --- | --- | --- |
| `formula` | FormulaCard | The statement the operator computes, with symbols bound to buffers and kernels |
| `datum` | DatumReceipt | One entity's worked example: inputs, neighbourhood and result, in analysis-stage order |
| `iterations` | SkeletonView and RealizedView | The declared step structure and what this run actually did, iteration by iteration |
| `kernel` | DispatchView | Real dispatch geometry, workgroups and lanes, with the kernel source |
| `graph` | CommandGraphView | The command graph at node or contributor grain, folded by degree of interest |
| `timing` | TimingView | Per-node GPU time with the observer effects of timing stated |
| `compare` | CompareView | Two runs or two references as an explicit delta |
| `scope` | ScopeDiagram | Which inputs an output reads, generated from the operator's `scope × reduction` |
| `table` | Table | The typed table view every record has, for anything without a custom view |

A rung the record cannot satisfy is shown disabled with its reason and the table view, never faked.

Line and polygon results select features before sampling vertices. The renderer reserves drawable
pinned and authored-highlight features, then prioritizes linked selections and hover rows within
the remaining feature budget before sampling other features evenly in source order. Each
selected open line retains both endpoints; each selected polygon retains at least three vertices,
excluding a repeated closing vertex before adding SVG closure. Closed lines retain an intermediate
vertex as well as their endpoints. Vertex sampling illustrates the result and does not preserve its
topology or support area measurements. If even sampling flattens a polygon, the renderer keeps
non-collinear source corners instead. Features smaller than the figure's coordinate resolution are
omitted and counted instead of being reported as drawn.

The shared extent uses only valid offset-delimited feature spans from the panels and animation
frames actually displayed. Spare coordinate capacity cannot change that extent. All displayed
animation positions share it, and each panel or key-frame SVG describes its own capture and count.

Line and polygon capture windows have independent feature and vertex origins. For example, an
offsets capture with `meaning: 'first:2'` and values `[5, 8]` describes feature key 2, while a
positions capture with `meaning: 'first:5'`, `components: 2` and values `[0, 0, 1, 0, 0, 1]`
contains vertex keys 5, 6 and 7. Offsets remain absolute vertex keys; subtract the positions origin
only to address its captured array. Declare `geometry.featureKeySpace` for the feature domain and
the positions capture's `keySpace` for the vertex domain. A feature-value capture `[10, 20, 30]`
starting at feature key zero supplies value 30 for feature key 2. Displayed values, picks and linked
selections all use that feature key, independently of the first captured vertex; they never treat
the first captured feature as feature zero.

Base geometry targets 400 vertices in a full figure and 64 in a thumbnail, divided across panels and
animation frames. Each panel has a minimum of three vertices; its allocation can rise to six to
retain distinct pinned and highlighted features. Selection and highlight overlays reuse the samples
and add drawing vertices. Captions and SVG descriptions report how many source features are shown.
Invalid offsets, insufficient vertices and non-finite coordinates omit a feature and are counted
explicitly. Empty or wholly undrawable captures show a labelled fallback in both view sizes.

### Result rows and the primary output

The `result-view` uses captures tagged `role: 'input'` or `role: 'output'`. Array columns share a
table only when both their role and declared `keySpace` match. Each table names its row domain; columns without
a key space get individual tables labelled “row domain unknown”. Equal lengths or a `column`
meaning do not establish a relation. Cross-space data must first be explicitly aligned through a
recorded relation and captured in the resulting common key space; the renderer never guesses a join.
Run-level diagnostics therefore remain separate from station or cell results. A geometric result
keeps its map and shows its array tables below it. Full-size tables show at most six entity rows per
domain, with omitted rows counted. Vector captures count complete rows as `values.length / components`
and show every component in each table cell. Input and output columns remain separate even when
they use the same key space. Table thumbnails retain captured input and output role groups in the
primary output's key space. Thumbnail keys come from usable nonempty captured windows: an input
with `first:9` and two values alongside an empty output at origin zero shows both input rows.
Empty outputs contribute no keys or marks. Thumbnails do not invent output readbacks for input rows.
The GPU Core array renderer also groups `column:` captures by both role and declared key space and
isolates each unkeyed column. Raw vector entities and explicit primary outputs use `result-view`. A single changing keyed
array retains its strip frames and playback control. Strips show one cell per complete entity,
including every vector component. Thumbnails draw readback marks only for captured entity keys;
missing cells remain absent.
Point playback also retains changes to the published active count and visible capture window,
even when the stored coordinates are identical; points outside that frame's active prefix stay hidden.
A `first:<key>` annotation offsets a captured window in both tables and
strips, so array element zero is associated with that key rather than key zero.
Comparisons require matching key spaces, element types and component counts. Across runs, both
captures must also declare matching domain sizes and grid layouts; a shared capture name alone
is insufficient. The effective raster shape comes from `capture.geometry.shape` when present,
otherwise the declared key-space shape. Both effective shapes must match before rows can be
subtracted. Raster origins, cell sizes, row directions, coordinate models and units must also
match, using the renderer's defaults when omitted. Incompatible references make comparison unavailable; buffer
inspectors also suppress difference summaries and envelopes for incompatible iteration captures.
Comparisons use the intersection of the two absolute capture windows. Disjoint windows compare zero
rows; missing keys never count as equal values. Buffer inspectors, receipts and iteration key frames
also translate entity keys to capture-local component offsets.
Buffer difference summaries publish `comparedCount` alongside `changedCount`, so a disjoint window
cannot be mistaken for an unchanged one.
Vector equality and maximum absolute delta cover every component in each compared row. Scalar
delta and juxtaposition strips display component 0 and label it explicitly for vector captures.
Shaped iteration frames use absolute row-major grid keys; uncaptured cells stay no-data and pins
retain their full-grid coordinates. Unshaped strips display the captured window locally.

Result marks carry their capture's `data-key-space` into pick and hover actions. The controller
resolves selection images independently for each domain. A station key and a cell key with the same
integer remain distinct; selections cross domains only through a recorded relation. Line and polygon
paths use `geometry.featureKeySpace` for feature rows, independently of the vertex capture's
`keySpace`. Without a recorded feature domain, paths are drawn but cannot be picked or receive linked
selection; the renderer never treats a track row as an AIS fix. Direct action
callers can supply `keySpace`; legacy single-domain actions default to the view's datum key space.
Raster result panels draw at most 64 selected-cell outlines per panel.
Captured index and CSR relations respect `first:<key>` on mapping rows, CSR offsets and target-list
slots in both directions. Keys outside the recorded windows have no captured mapping; missing slots
never imply a mapping to row zero.
If a requested CSR span is only partly captured, or a requested source row is missing, linked
selection returns `unknown` with no definite highlighted/off-view counts. Inverse lookups require
complete source-domain coverage, including every target span. An explicitly captured empty row
remains a known empty mapping. Direct image/preimage calls refuse incomplete evidence; coordinator
lookups propagate it as `unknown`.

Point geometry can bind `countCapture` to the captured scalar active count. Its headline uses that
published count and rendering/sampling uses only the active prefix; unused allocation slots never
inflate the figure. The count is bounded by the declared full point capacity when available,
including a logical source-buffer size when the domain size is absent; captured windows and
observed resource lower bounds never cap it. The map draws only captured keys in the active prefix.
A missing or invalid bound count makes the panel unavailable. Result rasters
read pixels by absolute grid key, and point picks add the capture window's first key. Shared S2
polygon inputs and the authored park-validity/tract-join outputs declare their feature domains.
Strip thumbnails retain at most four panels, including the primary output and one recorded input;
the full view retains every panel.
Strip thumbnails show at most four panels, retaining the primary output and an input when present,
with at most eight entity cells per panel.

Capture authors must record component width when several scalar values belong to one entity. S2
scenarios can set `S2CaptureSpec.components` independently of the graph view's storage format; the
spatial-regression scenario records two predictors per tract. If a capture's row range exceeds its
declared key-space size, the generic result view withholds its values and states that its component
layout is unknown. It never infers the missing width from array length.

Scenarios can select the headline and result grammar through presentation metadata:

```ts
presentation: {
  primaryOutputCaptureId: 'gpu-spatial-analysis-kriging-snapshots-rainfall-0'
}
```

Built-in scenarios declare it in the side-effect-free table
`modules/lenses/src/computation-lenses/scenario-presentations.ts`, and newly captured recordings
store it in the envelope as `presentation`. A static twin uses the recording's `presentation`
first, then the registered scenario's, then that table, so reading a saved record without
importing any scenario module selects the same headline (the saved kriging record headlines
rainfall, 4,662 cells, rather than its trailing one-row diagnostic).
Static twins and `mountLadderLens({scenarioId, ...})` pass this metadata to the view. Direct view
callers can set `context.options.primaryOutputCaptureId`; live hosts can override it through
`params.primaryOutputCaptureId`. `LensControllerOptions.presentation` retains the defaults when URL
parameters change. The ID must identify a final capture tagged `role: 'output'`, or
the rung is disabled with a reason. Without an explicit ID the last tagged output remains the
default. Kriging explicitly selects rainfall, so its headline counts raster cells rather than a
trailing run diagnostic.
Segment intersection selects `intersections`, retaining points colored by `kinds`; predicate join
selects `joinedTracts`, retaining polygons shaded by `tractMatches`. Counts, overflow flags and other
diagnostics remain in separate tables below the full figure. A trailing diagnostic or color-value
capture must not displace the geometric output promised by a map's caption.
When the headline output draws nothing (all features undrawable, empty geometry or unknown
layout), the view's `display` fact states it (`geometry not drawn: no drawable captured features
(0 of 1 polygons shown)`), and the alt text, live-region text and thumbnail caption all carry it
ahead of the key number.
Suppressed inputs are named even when the output is drawable. For example, a valid polygon output
`geometry` beside an undrawable line input `roads` starts its concise display statement with
`geometry drawn; roads not drawn: no drawable captured features (0 of 1 lines shown)`.
Accessible descriptions and live-region text carry both clauses, and thumbnail captions retain
that display statement as their first sentence. This describes what is drawn without implying
that the input was absent from the recorded computation.
An empty point output named `trees` instead says
`trees not drawn: empty captured geometry (0 of 0 points shown)`; a nonempty point capture with no
finite drawable coordinates says `no drawable captured features`. If an input `trail` is also
suppressed, its `trail not drawn` clause follows the output clause in the same descriptions and
captions. A headline count alone is not evidence that any output points are visible.
Point drawability text uses the same stride sample as the figure, including any explicit
highlight mark. If no sampled point is drawable, the caption, accessible description and live
announcement say `not drawn` and `0 of N points shown`, even when an unsampled source row is finite.
Full figures and thumbnails resolve their own sampling budgets independently.
The showcase loads the scenario registration entry point when exploration is activated, preserving
the same primary-output metadata when opening an existing recorded run.

## Page templates

Each operator section in the reference ends with a **How it computes** block. Its order depends on the
audience of the entry:

- **Analyst entries** (spatial analysis, network, raster, terrain, dataframe analysis) go concrete first:
  `datum` with a substituted formula, then `formula`, then `iterations` if the operator is iterative,
  then `kernel`, then `graph`.
- **Engineer entries** (GPU Core, gpgpu) open at `kernel` or `graph`, with the other rungs one level up.

Each entry's Concepts page has a **GPU execution model** section drawn from the real `describe()` output
of a representative graph: the notional machine made explicit.

### Lens slots

The operator pages carry one marker per exported contributor at the end of the section that documents
it:

```md
<!-- lens:GPULocalMoran -->
```

To fill a slot, keep the marker and add the heading and directives around it:

```md
#### How it computes

<!-- lens:GPULocalMoran -->

::lens{scenario=gpu-spatial-analysis.local-moran rung=datum datum=17 caption="Tract 17, worked example"}

::lens{scenario=gpu-spatial-analysis.local-moran rung=formula}

::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel}
```

The coverage contract finds a contributor's section by its marker. The Concepts pages use the marker
`<!-- lens:execution-model -->`.

## Static twins

Every lens has a static twin printed at build time: a `<figure>` with the recorded view, a caption, an
inline key number, the receipt as a table, the formula as LaTeX with MathML, and the adapter, date and
observer effects of the recorded run. Search engines, print, the llms text export and readers without
WebGPU read the twin. A twin needs no JavaScript.

Twins are produced in three steps:

1. A scenario in `@luma-contributors/lenses/computation-lenses` runs the real computation on docs-sized open
   data (`modules/lenses/src/computation-lenses/scenarios/<entry>/<element>.lens.ts`).
2. A headless capture run on a real GPU writes `docs/lens-records/<scenario>.json`, labelled with
   adapter, variant and date. Only maintainers run it; it is not part of the docs build. Records are
   compacted losslessly on disk (repeated captures and word runs, WGSL source lines, interned graph
   strings, object shapes and repeated arrays); read them with `fromRecordedLensFile`. Each record has a
   512 KB budget (`record-budget.node.spec.ts`): a scenario shrinks by capturing key frames rather than
   every iteration, only the captures its views read, or a smaller docs input, and says so in the
   capture's meaning.
3. A node spec turns each record into one twin per directive key, using the same description generator
   that feeds alt text and the live region, and checks them in:

   ```sh
   UPDATE_LENS_TWINS=1 npx vitest run --project node modules/lenses/test/computation-docs/lens-twins.node.spec.ts
   ```

   The files land in `docs/lens-records/twins/<scenario>~<rung>[~d<datum>][~i<iteration>][~v<view>].json`. Without
   the variable the spec fails when a twin is stale, and it fails when a twin has no caption or no key
   number.

Renderer or primary-output presentation changes use existing source records: regenerate the twins,
then run the same spec without `UPDATE_LENS_TWINS` to verify deterministic output and the artifact
size budgets. The result-table regression spec also checks the committed kriging headline and its
separate station, cell and run tables. A renderer-only change needs no GPU recapture.

`registerLensScenario` returns an idempotent disposal callback for temporary registrations.
Retain it and call it during teardown: it removes only that registration and preserves all other
scenario descriptors, including later replacements. Authored module registrations may ignore it.

`scenario-declaration.node.spec.ts` runs every registered scenario on a null WebGPU device through
record assembly, with executable callbacks disabled. It checks that graphs declare successfully
and every declared analyst choice has a value, including the owning spatial, raster, terrain or
dataframe pack's choice templates (both named templates and string-name lists).
For scenarios without a hand-written `describeStructure`, it also compares the ordered node IDs,
node kinds, read/write resource names and snapshot capture names with the decoded committed record.
Named `.geometry` aliases participate in this signature. Their names are derived from polygon tags
before readback, using the same naming function as the actual capture producer. Removing an alias
from a recorded sidecar makes parity fail even when all graph nodes and numeric values agree.
Every compiled observation graph is independently described from its preflight and reconciled
one-to-one with the graphs assembled into the record. An unrecorded graph or node fails declaration,
as does an assembled `compiled` graph that was never compiled during the scenario run.
The intercepted live record also passes the production schema validator before it is accepted,
so invalid key-space references fail before any readback.
Changed graph declarations require GPU recapture; this check does not verify numeric GPU results.
The construction check uses core WebGPU limits. Structure parity replays the storage-buffer binding
limit required by each record's retained bindings, because a higher binding limit can fuse Moran
epilogues into a different graph. Uniform and sampled-texture bindings do not count toward that
limit. Legacy records do not retain the complete device limits, so this replay uses the minimum
core-or-higher storage-buffer limit supported by the recorded graph.
All registered scenarios, including dataframe scenarios, declare without GPU execution; the gate
has no device-dependent exceptions. Dataframe capture preparation declares the snapshot tap and
graph description before record assembly. Submission and readback follow record assembly.
Scenarios that reshape readbacks declare their final capture names and check those names against
the actual derived sidecar after execution. No synthetic numeric captures are used for declaration.

S2 scenarios must supply every choice template from their spatial operator entry before building
their graph. Choice receipts include defaults the example uses and explicitly mark disabled
optional features, such as bandwidth selection when the example supplies fixed bandwidths.
Supplied-intensity point patterns disable both bandwidth estimation and self-contribution choices;
estimated-intensity receipts retain both controls. Point-process receipts distinguish event rows,
radius/curve bins, simulation or sampling rows, geometry offset boundaries, ordering ranks and status
words. Choosing an event never indexes a radius, boundary or diagnostic word as though it were that event.
Missing choices throw with the scenario ID and choice name before allocation or submission.
Disabled choice strings are never sweepable. Coordinate captures use a vertex domain; polygon
`geometry.featureKeySpace` identifies the feature rows selected by paths and used by scalar receipts.
S2 park/tract display coordinates have a separate display-vertex domain, so pinning feature 1 cannot
read vertex 1 of another feature as its datum. Spatial polygon aliases retain their scalar capture's
feature domain for picking while keeping coordinates in the vertex domain.
The shared Midwest county outlines and painted maps likewise capture coordinates in `county-vertex`
and pick feature rows in `county`; scalar county receipts never index a ring's vertex array.
Point maps resolve `geometry.countCapture` before fitting bounds, sampling coordinates or sampling
associated values. Spare slots cannot change the extent, colour range or sampling stride. A zero
count draws no points; a missing or invalid count withholds them. A captured point window intersects
the active prefix, with count validation against the full declared domain when available. The
dot-density result uses `points` as its primary output and its published `count` as the headline
in both full and thumbnail twins.
Automatic ordered point scales and legends use only value keys in the final active captured window,
intersected with the value capture's absolute window. Explicit host-authored scales remain authoritative.
The packed spatial-weights transpose lens captures the source's inherited overflow and candidate-loss
flags. It supplies only CSR destination buffers: packed transpose does not write destination status
ports. Either source loss flag refuses record publication rather than presenting incomplete topology.
All tract-weights recordings forward both flags directly from the producer at the shared capture
boundary, including contiguity and summary lenses; suppressed zero counts cannot establish zero islands.
The staleness proofs mutate committed signatures in memory for pair-gather and nearest-feature-join:
removed, added or reordered nodes, changed node kinds, renamed read/write resources and capture names.
The same comparison used by registry parity reports the scenario and the drift. Unmodified records
still pass. These tests require neither Git nor history objects or extra record fixtures.

The website and the showcase only read twin files, so a docs build never imports view or GPU code.

Each displayed `#region example` is limited to 25 source lines. Keep allocation and histogram setup
above the region or in a named helper in the example file; keep the operator construction and its
required status ports visible. Invoke graph-building helpers at the original stage so shortening a
snippet does not change execution order or require recapturing its computation record.
Checked dynamic imports and `require` calls must supply a single literal module specifier; concatenated
or computed expressions are rejected even when their first string names a public entry.

`captureLensScenarioRecord` and `toRecordedLensFile` return a `RecordedLensFile`: its `record`
payload is opaque because graphs and capture values are encoded. Call `fromRecordedLensFile(file)`
to obtain a `DecodedLensFile` with graph objects and numeric capture arrays before inspecting the
result. The reader accepts compact files with their encoding dictionaries and older uncompressed
envelopes containing graph and node objects with IDs. A compact graph missing its dictionary is
rejected before returning a decoded record. Writer and reader share the same metadata requirements:
a valid scenario ID, a parseable `recordedAt` timestamp, a
nonblank `adapterLabel`, and a string `variant` when supplied. `buildLensTwin` accepts either form.

A twin also prints a verdict chip for each guard that applies to the element: a domain pack returns
them from `guardsFor({record, element})`, and the same guards run in the live lens. A chip says
*pass*, *fail* with the reason, or *unknown* with the missing metadata.

### Record pending

A directive whose scenario has no recorded run yet prints a hatched **Record pending** twin with its
caption and a pending key number. To stage such a page, list the scenario ID and the capture reason in
`modules/lenses/test/computation-docs/pending-lens-records.ts`. Coverage rejects unlisted gaps and
requires removing the entry when its capture lands. Every checked-in twin must keep its source
record, including during opt-in regeneration; deleting a capture cannot silently make it pending.
The twin regeneration command runs this pending-manifest check itself before destination writes
or pruning. A mistyped, unlisted scenario ID aborts the update and preserves existing twins, even
when the separate coverage spec is not run.

## Live mode

On a browser with WebGPU each twin offers **Run live on this GPU**. Nothing runs during render or
hydration. On the reader's click the page loads `computation-views` and `computation-lenses`, mounts the
view into the twin's `<ladder-lens>` element, and can re-run the same scenario on the reader's adapter.
On failure the twin comes back with the error stated.

- One device per page, using the `webgpu-analysis` profile; a page scheduler serializes runs.
- Lenses far offscreen revert to their twin, and views pause when the page is hidden.
- Leaving the page disposes every live lens.

## Linking lenses

Lenses on one page link by default through the coordinator named by `bind`: selecting a datum in one
lens highlights it in every lens bound to the same name, through declared key-space relations rather than
geometry. Give unrelated lenses different `bind` names to keep them independent.

Each mounted controller retains its record's relations until it is destroyed or receives a replacement
record. Removing one lens preserves mappings used by other lenses on the same run; the final release
removes the relation. Programmatic hosts can use `LadderCoordinator.retainRelations` and call its
idempotent release handle on teardown. Supply source-domain metadata as the second argument,
`retainRelations(relations, [{id: 'external', keyCount: 16}])`; sizes and optional field-value readers
follow the same owner as the relations. When owners supply different mappings for the same relation ID,
the latest retained mapping takes precedence; releasing it restores the surviving owner's mapping.
Any relation installed with `addRelations` is restored after the final retained owner releases it,
unless it was removed with `removeRelations`.
Effective mapping or domain-metadata changes emit a `mapping` event after the whole update, so
surviving controllers refresh their selections immediately without requiring another pick.

## Coverage contract

`modules/lenses/test/computation-docs/lens-coverage.node.spec.ts` checks the docs, packs and
scenarios together. The coverage unit is a named family scenario. For each entry listed in its
`COVERED_ENTRIES`, every exported contributor needs:

1. an operator descriptor `scope × reduction` in the entry's domain pack, plus `variableKinds` for spatial
   analysis;
2. a formula reference, or an explicit `{formula: 'none', reason}` with an `iterations` or `graph` lens;
3. at least one registered lens scenario whose `element` is the exported name and whose id is
   `<entry>.<element-kebab>[.<variant>]`, or a `familyScenarios` coverage record explicitly naming
   the family scenario and why it demonstrates this variant;
4. a `::lens` directive for that scenario in its section;
5. a recorded run, or an explained entry in the pending capture manifest (set
   `LENS_COVERAGE_REQUIRE_RECORDS=1` to require all captures).

A variant's named family scenario must satisfy the same pack, formula, registration, directive and
capture checks. A sibling scenario from the same package does not establish coverage by itself.
The demonstrated operation must also be supported by the variant: chunked spatial-weights algebra
cannot inherit the whole-CSR `higherOrder` scenario. A formula exemption's graph or iterations
directive must belong to the selected element or family scenario, even in a shared marked section.
Contributors intentionally outside lens coverage have a separate documented exemption.

The review resume lists in the coverage spec name contributors whose scenarios are still unfinished
and state the missing parts. When discovery starts recognizing an existing adapter, explicitly add
its missing work to that list or supply coverage; it does not inherit coverage merely by sharing a
package. A completed contributor must leave the resume list.

Twin verification groups directives by scenario, decodes one source record, and releases its arrays
after rendering that scenario's variants. For the same pattern in a host, call
`fromRecordedLensFile` once and pass its `DecodedLensFile` to `buildDecodedLensTwin` for each rung.
Use `buildLensTwin` at the serialized input boundary. Verification retains filenames and hashes
between scenarios; opt-in updates stage rendered text until all scans and renders succeed.

Independent of `COVERED_ENTRIES`, the spec checks that every lens slot marker is present, that every
directive is well formed, and that every record file names its scenario.

## Writing good lenses

- Draw from a real run. Folded state recovered by a GPU replay is labelled as a replay; a value is never
  re-simulated on the CPU for display.
- State observer effects. A timing lens says that timestamps disabled pass coalescing.
- Choose captions that say what the reader should see, and make the key number the one fact a reader
  should take away.
- Keep data small and real: each scenario dataset is at most 150 KB, with its source and licence listed in
  `modules/lenses/src/computation-lenses/data/README.md`.

## See also

- [GPU computation records](/docs/api/computation-record)
- [GPU Spatial Analysis concepts](/docs/api/gpu-spatial-analysis/concepts)
- [GPU Network concepts](/docs/api/gpu-network/concepts)
- [GPU Terrain concepts](/docs/api/gpu-terrain/concepts)
