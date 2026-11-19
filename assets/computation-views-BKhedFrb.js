var e={slug:`api/computation-views`,title:`Computation views authoring guide`,html:`<h1 id="computation-views-authoring-guide">Computation views authoring guide</h1>
<p>:::caution Experimental
<code>@luma-contributors/lenses/computation-views</code> and <code>@luma-contributors/lenses/computation-lenses</code> are
experimental entry points. The directive syntax, record files and views may change between releases.
:::</p>
<p>A <strong>lens</strong> is a view of one real computation at one level of abstraction: the formula an operator
computes, one datum's receipt, the iterations a loop actually ran, the kernel dispatch, or the command
graph. Every lens is drawn from a recorded run of the real GPU computation, never from a CPU replay. The
same Markdown renders on this site and in the Reference section of the spatial analysis showcase
(<code>examples/deck/spatial-analysis-showcase</code>).</p>
<p>This page is for people writing the operator reference. The record format is documented in
<a href="/luma-contributors/docs/api/computation-record">GPU computation records</a>.</p>
<h2 id="the-lens-directive">The lens directive</h2>
<p>A lens is a Markdown leaf directive on a line of its own:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=17 caption="Local Moran I of tract 17"}</span></span></code></pre>
<table>
<thead>
<tr>
<th>Attribute</th>
<th>Required</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>scenario</code></td>
<td>yes</td>
<td>Registered lens scenario, <code>&#x3C;entry>.&#x3C;element>[.&#x3C;variant>]</code> in kebab case, for example <code>gpu-spatial-analysis.local-moran</code></td>
</tr>
<tr>
<td><code>rung</code></td>
<td>yes</td>
<td>Level of abstraction to show; see <a href="#rungs">Rungs</a></td>
</tr>
<tr>
<td><code>bind</code></td>
<td>no</td>
<td>Page coordinator the lens links through; lenses with the same name share selections. Defaults to <code>page</code></td>
</tr>
<tr>
<td><code>datum</code></td>
<td>no</td>
<td>Non-negative integer: the entity (row, cell, node) the view opens bound to</td>
</tr>
<tr>
<td><code>iteration</code></td>
<td>no</td>
<td>Non-negative integer: the loop iteration the view opens at</td>
</tr>
<tr>
<td><code>caption</code></td>
<td>no</td>
<td>Quoted caption. Without it the generated description is the caption</td>
</tr>
<tr>
<td><code>view</code></td>
<td>no</td>
<td>Id of a registered view to use for the rung instead of the preferred one, for example <code>neighbourhood-view</code></td>
</tr>
</tbody>
</table>
<p>Unknown attributes, an invalid scenario id or an unknown rung fail the website build with the file and
line. The showcase prints the same error in place instead of failing.</p>
<h2 id="rungs">Rungs</h2>
<table>
<thead>
<tr>
<th>Rung</th>
<th>View</th>
<th>Use it for</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>formula</code></td>
<td>FormulaCard</td>
<td>The statement the operator computes, with symbols bound to buffers and kernels</td>
</tr>
<tr>
<td><code>datum</code></td>
<td>DatumReceipt</td>
<td>One entity's worked example: inputs, neighbourhood and result, in analysis-stage order</td>
</tr>
<tr>
<td><code>iterations</code></td>
<td>SkeletonView and RealizedView</td>
<td>The declared step structure and what this run actually did, iteration by iteration</td>
</tr>
<tr>
<td><code>kernel</code></td>
<td>DispatchView</td>
<td>Real dispatch geometry, workgroups and lanes, with the kernel source</td>
</tr>
<tr>
<td><code>graph</code></td>
<td>CommandGraphView</td>
<td>The command graph at node or contributor grain, folded by degree of interest</td>
</tr>
<tr>
<td><code>timing</code></td>
<td>TimingView</td>
<td>Per-node GPU time with the observer effects of timing stated</td>
</tr>
<tr>
<td><code>compare</code></td>
<td>CompareView</td>
<td>Two runs or two references as an explicit delta</td>
</tr>
<tr>
<td><code>scope</code></td>
<td>ScopeDiagram</td>
<td>Which inputs an output reads, generated from the operator's <code>scope × reduction</code></td>
</tr>
<tr>
<td><code>table</code></td>
<td>Table</td>
<td>The typed table view every record has, for anything without a custom view</td>
</tr>
</tbody>
</table>
<p>A rung the record cannot satisfy is shown disabled with its reason and the table view, never faked.</p>
<p>Line and polygon results select features before sampling vertices. The renderer reserves drawable
pinned and authored-highlight features, then prioritizes linked selections and hover rows within
the remaining feature budget before sampling other features evenly in source order. Each
selected open line retains both endpoints; each selected polygon retains at least three vertices,
excluding a repeated closing vertex before adding SVG closure. Closed lines retain an intermediate
vertex as well as their endpoints. Vertex sampling illustrates the result and does not preserve its
topology or support area measurements. If even sampling flattens a polygon, the renderer keeps
non-collinear source corners instead. Features smaller than the figure's coordinate resolution are
omitted and counted instead of being reported as drawn.</p>
<p>The shared extent uses only valid offset-delimited feature spans from the panels and animation
frames actually displayed. Spare coordinate capacity cannot change that extent. All displayed
animation positions share it, and each panel or key-frame SVG describes its own capture and count.</p>
<p>Base geometry targets 400 vertices in a full figure and 64 in a thumbnail, divided across panels and
animation frames. Each panel has a minimum of three vertices; its allocation can rise to six to
retain distinct pinned and highlighted features. Selection and highlight overlays reuse the samples
and add drawing vertices. Captions and SVG descriptions report how many source features are shown.
Invalid offsets, insufficient vertices and non-finite coordinates omit a feature and are counted
explicitly. Empty or wholly undrawable captures show a labelled fallback in both view sizes.</p>
<h3 id="result-rows-and-the-primary-output">Result rows and the primary output</h3>
<p>The <code>result-view</code> uses captures tagged <code>role: 'input'</code> or <code>role: 'output'</code>. Array columns share a
table only when their declared <code>keySpace</code> matches. Each table names its row domain; columns without
a key space get individual tables labelled “row domain unknown”. Equal lengths or a <code>column</code>
meaning do not establish a relation. Cross-space data must first be explicitly aligned through a
recorded relation and captured in the resulting common key space; the renderer never guesses a join.
Run-level diagnostics therefore remain separate from station or cell results. A geometric result
keeps its map and shows its array tables below it. Full-size tables show at most six entity rows per
domain, with omitted rows counted. Vector captures count complete rows as <code>values.length / components</code>
and show every component in each table cell. Table thumbnails show only the primary output's domain.
The GPU Core array renderer also groups <code>column:</code> captures by key space and isolates unkeyed
columns. Raw vector entities and explicit primary outputs use <code>result-view</code>. A single changing keyed
array retains its strip frames and playback control. Strips show one cell per complete entity,
including every vector component. Thumbnails draw readback marks only for captured entity keys;
missing cells remain absent. A <code>first:&#x3C;key></code> annotation offsets a captured window in both tables and
strips, so array element zero is associated with that key rather than key zero.
Comparisons require matching key spaces, element types and component counts. Across runs, both
captures must also declare matching domain sizes and grid layouts; a shared capture name alone
is insufficient. The effective raster shape comes from <code>capture.geometry.shape</code> when present,
otherwise the declared key-space shape. Both effective shapes must match before rows can be
subtracted. Raster origins, cell sizes, row directions, coordinate models and units must also
match, using the renderer's defaults when omitted. Incompatible references make comparison unavailable; buffer
inspectors also suppress difference summaries and envelopes for incompatible iteration captures.
Comparisons use the intersection of the two absolute capture windows. Disjoint windows compare zero
rows; missing keys never count as equal values. Buffer inspectors, receipts and iteration key frames
also translate entity keys to capture-local component offsets.
Buffer difference summaries publish <code>comparedCount</code> alongside <code>changedCount</code>, so a disjoint window
cannot be mistaken for an unchanged one.
Vector equality and maximum absolute delta cover every component in each compared row. Scalar
delta and juxtaposition strips display component 0 and label it explicitly for vector captures.
Shaped iteration frames use absolute row-major grid keys; uncaptured cells stay no-data and pins
retain their full-grid coordinates. Unshaped strips display the captured window locally.</p>
<p>Result marks carry their capture's <code>data-key-space</code> into pick and hover actions. The controller
resolves selection images independently for each domain. A station key and a cell key with the same
integer remain distinct; selections cross domains only through a recorded relation. Line and polygon
paths use <code>geometry.featureKeySpace</code> for feature rows, independently of the vertex capture's
<code>keySpace</code>. Without a recorded feature domain, paths are drawn but cannot be picked or receive linked
selection; the renderer never treats a track row as an AIS fix. Direct action
callers can supply <code>keySpace</code>; legacy single-domain actions default to the view's datum key space.
Raster result panels draw at most 64 selected-cell outlines per panel.
Captured index and CSR relations respect <code>first:&#x3C;key></code> on mapping rows, CSR offsets and target-list
slots in both directions. Keys outside the recorded windows have no captured mapping; missing slots
never imply a mapping to row zero.
If a requested CSR span is only partly captured, or a requested source row is missing, linked
selection returns <code>unknown</code> with no definite highlighted/off-view counts. Inverse lookups require
complete source-domain coverage, including every target span. An explicitly captured empty row
remains a known empty mapping. Direct image/preimage calls refuse incomplete evidence; coordinator
lookups propagate it as <code>unknown</code>.</p>
<p>Point geometry can bind <code>countCapture</code> to the captured scalar active count. Its headline uses that
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
with at most eight entity cells per panel.</p>
<p>Capture authors must record component width when several scalar values belong to one entity. S2
scenarios can set <code>S2CaptureSpec.components</code> independently of the graph view's storage format; the
spatial-regression scenario records two predictors per tract. If a capture's row range exceeds its
declared key-space size, the generic result view withholds its values and states that its component
layout is unknown. It never infers the missing width from array length.</p>
<p>Scenarios can select the headline and result grammar through presentation metadata:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">presentation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: {</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  primaryOutputCaptureId</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'gpu-spatial-analysis-kriging-snapshots-rainfall-0'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span></code></pre>
<p>Static twins and <code>mountLadderLens({scenarioId, ...})</code> pass this metadata to the view. Direct view
callers can set <code>context.options.primaryOutputCaptureId</code>; live hosts can override it through
<code>params.primaryOutputCaptureId</code>. <code>LensControllerOptions.presentation</code> retains the defaults when URL
parameters change. The ID must identify a final capture tagged <code>role: 'output'</code>, or
the rung is disabled with a reason. Without an explicit ID the last tagged output remains the
default. Kriging explicitly selects rainfall, so its headline counts raster cells rather than a
trailing run diagnostic.
Segment intersection selects <code>intersections</code>, retaining points colored by <code>kinds</code>; predicate join
selects <code>joinedTracts</code>, retaining polygons shaded by <code>tractMatches</code>. Counts, overflow flags and other
diagnostics remain in separate tables below the full figure. A trailing diagnostic or color-value
capture must not displace the geometric output promised by a map's caption.
The showcase loads the scenario registration entry point when exploration is activated, preserving
the same primary-output metadata when opening an existing recorded run.</p>
<h2 id="page-templates">Page templates</h2>
<p>Each operator section in the reference ends with a <strong>How it computes</strong> block. Its order depends on the
audience of the entry:</p>
<ul>
<li><strong>Analyst entries</strong> (spatial analysis, network, raster, terrain, dataframe analysis) go concrete first:
<code>datum</code> with a substituted formula, then <code>formula</code>, then <code>iterations</code> if the operator is iterative,
then <code>kernel</code>, then <code>graph</code>.</li>
<li><strong>Engineer entries</strong> (GPU Core, gpgpu) open at <code>kernel</code> or <code>graph</code>, with the other rungs one level up.</li>
</ul>
<p>Each entry's Concepts page has a <strong>GPU execution model</strong> section drawn from the real <code>describe()</code> output
of a representative graph: the notional machine made explicit.</p>
<h3 id="lens-slots">Lens slots</h3>
<p>The operator pages carry one marker per exported contributor at the end of the section that documents
it:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">&#x3C;!-- lens:GPULocalMoran --></span></span></code></pre>
<p>To fill a slot, keep the marker and add the heading and directives around it:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-light-font-weight:bold;--shiki-dark:#79B8FF;--shiki-dark-font-weight:bold">#### How it computes</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">&#x3C;!-- lens:GPULocalMoran --></span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">::lens{scenario=gpu-spatial-analysis.local-moran rung=datum datum=17 caption="Tract 17, worked example"}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">::lens{scenario=gpu-spatial-analysis.local-moran rung=formula}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel}</span></span></code></pre>
<p>The coverage contract finds a contributor's section by its marker. The Concepts pages use the marker
<code>&#x3C;!-- lens:execution-model --></code>.</p>
<h2 id="static-twins">Static twins</h2>
<p>Every lens has a static twin printed at build time: a <code>&#x3C;figure></code> with the recorded view, a caption, an
inline key number, the receipt as a table, the formula as LaTeX with MathML, and the adapter, date and
observer effects of the recorded run. Search engines, print, the llms text export and readers without
WebGPU read the twin. A twin needs no JavaScript.</p>
<p>Twins are produced in three steps:</p>
<ol>
<li>
<p>A scenario in <code>@luma-contributors/lenses/computation-lenses</code> runs the real computation on docs-sized open
data (<code>modules/lenses/src/computation-lenses/scenarios/&#x3C;entry>/&#x3C;element>.lens.ts</code>).</p>
</li>
<li>
<p>A headless capture run on a real GPU writes <code>docs/lens-records/&#x3C;scenario>.json</code>, labelled with
adapter, variant and date. Only maintainers run it; it is not part of the docs build. Records are
compacted losslessly on disk (repeated captures and word runs, WGSL source lines, interned graph
strings, object shapes and repeated arrays); read them with <code>fromRecordedLensFile</code>. Each record has a
512 KB budget (<code>record-budget.node.spec.ts</code>): a scenario shrinks by capturing key frames rather than
every iteration, only the captures its views read, or a smaller docs input, and says so in the
capture's meaning.</p>
</li>
<li>
<p>A node spec turns each record into one twin per directive key, using the same description generator
that feeds alt text and the live region, and checks them in:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">UPDATE_LENS_TWINS</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">1</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> npx</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> vitest</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> run</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> --project</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> node</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> modules/lenses/test/computation-docs/lens-twins.node.spec.ts</span></span></code></pre>
<p>The files land in <code>docs/lens-records/twins/&#x3C;scenario>~&#x3C;rung>[~d&#x3C;datum>][~i&#x3C;iteration>][~v&#x3C;view>].json</code>. Without
the variable the spec fails when a twin is stale, and it fails when a twin has no caption or no key
number.</p>
</li>
</ol>
<p>Renderer or primary-output presentation changes use existing source records: regenerate the twins,
then run the same spec without <code>UPDATE_LENS_TWINS</code> to verify deterministic output and the artifact
size budgets. The result-table regression spec also checks the committed kriging headline and its
separate station, cell and run tables. A renderer-only change needs no GPU recapture.</p>
<p><code>registerLensScenario</code> returns an idempotent disposal callback for temporary registrations.
Retain it and call it during teardown: it removes only that registration and preserves all other
scenario descriptors, including later replacements. Authored module registrations may ignore it.</p>
<p><code>scenario-declaration.node.spec.ts</code> runs every registered scenario on a null WebGPU device through
record assembly, with executable callbacks disabled. It checks that graphs declare successfully
and every declared analyst choice has a value, including the owning spatial, raster, terrain or
dataframe pack's choice templates (both named templates and string-name lists).
For scenarios without a hand-written <code>describeStructure</code>, it also compares the ordered node IDs,
node kinds, read/write resource names and snapshot capture names with the decoded committed record.
Named <code>.geometry</code> aliases participate in this signature. Their names are derived from polygon tags
before readback, using the same naming function as the actual capture producer. Removing an alias
from a recorded sidecar makes parity fail even when all graph nodes and numeric values agree.
Every compiled observation graph is independently described from its preflight and reconciled
one-to-one with the graphs assembled into the record. An unrecorded graph or node fails declaration,
as does an assembled <code>compiled</code> graph that was never compiled during the scenario run.
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
the actual derived sidecar after execution. No synthetic numeric captures are used for declaration.</p>
<p>S2 scenarios must supply every choice template from their spatial operator entry before building
their graph. Choice receipts include defaults the example uses and explicitly mark disabled
optional features, such as bandwidth selection when the example supplies fixed bandwidths.
Missing choices throw with the scenario ID and choice name before allocation or submission.
Disabled choice strings are never sweepable. Coordinate captures use a vertex domain; polygon
<code>geometry.featureKeySpace</code> identifies the feature rows selected by paths and used by scalar receipts.
S2 park/tract display coordinates have a separate display-vertex domain, so pinning feature 1 cannot
read vertex 1 of another feature as its datum. Spatial polygon aliases retain their scalar capture's
feature domain for picking while keeping coordinates in the vertex domain.
Point maps resolve <code>geometry.countCapture</code> before fitting bounds, sampling coordinates or sampling
associated values. Spare slots cannot change the extent, colour range or sampling stride. A zero
count draws no points; a missing or invalid count withholds them. A captured point window intersects
the active prefix, with count validation against the full declared domain when available. The
dot-density result uses <code>points</code> as its primary output and its published <code>count</code> as the headline
in both full and thumbnail twins.
The staleness proofs mutate committed signatures in memory for pair-gather and nearest-feature-join:
removed, added or reordered nodes, changed node kinds, renamed read/write resources and capture names.
The same comparison used by registry parity reports the scenario and the drift. Unmodified records
still pass. These tests require neither Git nor history objects or extra record fixtures.</p>
<p>The website and the showcase only read twin files, so a docs build never imports view or GPU code.</p>
<p>Each displayed <code>#region example</code> is limited to 25 source lines. Keep allocation and histogram setup
above the region or in a named helper in the example file; keep the operator construction and its
required status ports visible. Invoke graph-building helpers at the original stage so shortening a
snippet does not change execution order or require recapturing its computation record.</p>
<p><code>captureLensScenarioRecord</code> and <code>toRecordedLensFile</code> return a <code>RecordedLensFile</code>: its <code>record</code>
payload is opaque because graphs and capture values are encoded. Call <code>fromRecordedLensFile(file)</code>
to obtain a <code>DecodedLensFile</code> with graph objects and numeric capture arrays before inspecting the
result. The reader accepts compact files with their encoding dictionaries and older uncompressed
envelopes containing graph and node objects with IDs. A compact graph missing its dictionary is
rejected before returning a decoded record. Writer and reader share the same metadata requirements:
a valid scenario ID, a parseable <code>recordedAt</code> timestamp, a
nonblank <code>adapterLabel</code>, and a string <code>variant</code> when supplied. <code>buildLensTwin</code> accepts either form.</p>
<p>A twin also prints a verdict chip for each guard that applies to the element: a domain pack returns
them from <code>guardsFor({record, element})</code>, and the same guards run in the live lens. A chip says
<em>pass</em>, <em>fail</em> with the reason, or <em>unknown</em> with the missing metadata.</p>
<h3 id="record-pending">Record pending</h3>
<p>A directive whose scenario has no recorded run yet prints a hatched <strong>Record pending</strong> twin with its
caption and a pending key number. To stage such a page, list the scenario ID and the capture reason in
<code>modules/lenses/test/computation-docs/pending-lens-records.ts</code>. Coverage rejects unlisted gaps and
requires removing the entry when its capture lands. Every checked-in twin must keep its source
record, including during opt-in regeneration; deleting a capture cannot silently make it pending.
The twin regeneration command runs this pending-manifest check itself before destination writes
or pruning. A mistyped, unlisted scenario ID aborts the update and preserves existing twins, even
when the separate coverage spec is not run.</p>
<h2 id="live-mode">Live mode</h2>
<p>On a browser with WebGPU each twin offers <strong>Run live on this GPU</strong>. Nothing runs during render or
hydration. On the reader's click the page loads <code>computation-views</code> and <code>computation-lenses</code>, mounts the
view into the twin's <code>&#x3C;ladder-lens></code> element, and can re-run the same scenario on the reader's adapter.
On failure the twin comes back with the error stated.</p>
<ul>
<li>One device per page, using the <code>webgpu-analysis</code> profile; a page scheduler serializes runs.</li>
<li>Lenses far offscreen revert to their twin, and views pause when the page is hidden.</li>
<li>Leaving the page disposes every live lens.</li>
</ul>
<h2 id="linking-lenses">Linking lenses</h2>
<p>Lenses on one page link by default through the coordinator named by <code>bind</code>: selecting a datum in one
lens highlights it in every lens bound to the same name, through declared key-space relations rather than
geometry. Give unrelated lenses different <code>bind</code> names to keep them independent.</p>
<p>Each mounted controller retains its record's relations until it is destroyed or receives a replacement
record. Removing one lens preserves mappings used by other lenses on the same run; the final release
removes the relation. Programmatic hosts can use <code>LadderCoordinator.retainRelations</code> and call its
idempotent release handle on teardown. Supply source-domain metadata as the second argument,
<code>retainRelations(relations, [{id: 'external', keyCount: 16}])</code>; sizes and optional field-value readers
follow the same owner as the relations. When owners supply different mappings for the same relation ID,
the latest retained mapping takes precedence; releasing it restores the surviving owner's mapping.
Any relation installed with <code>addRelations</code> is restored after the final retained owner releases it,
unless it was removed with <code>removeRelations</code>.
Effective mapping or domain-metadata changes emit a <code>mapping</code> event after the whole update, so
surviving controllers refresh their selections immediately without requiring another pick.</p>
<h2 id="coverage-contract">Coverage contract</h2>
<p><code>modules/lenses/test/computation-docs/lens-coverage.node.spec.ts</code> checks the docs, packs and
scenarios together. The coverage unit is a named family scenario. For each entry listed in its
<code>COVERED_ENTRIES</code>, every exported contributor needs:</p>
<ol>
<li>an operator descriptor <code>scope × reduction</code> in the entry's domain pack, plus <code>variableKinds</code> for spatial
analysis;</li>
<li>a formula reference, or an explicit <code>{formula: 'none', reason}</code> with an <code>iterations</code> or <code>graph</code> lens;</li>
<li>at least one registered lens scenario whose <code>element</code> is the exported name and whose id is
<code>&#x3C;entry>.&#x3C;element-kebab>[.&#x3C;variant>]</code>, or a <code>familyScenarios</code> coverage record explicitly naming
the family scenario and why it demonstrates this variant;</li>
<li>a <code>::lens</code> directive for that scenario in its section;</li>
<li>a recorded run, or an explained entry in the pending capture manifest (set
<code>LENS_COVERAGE_REQUIRE_RECORDS=1</code> to require all captures).</li>
</ol>
<p>A variant's named family scenario must satisfy the same pack, formula, registration, directive and
capture checks. A sibling scenario from the same package does not establish coverage by itself.
The demonstrated operation must also be supported by the variant: chunked spatial-weights algebra
cannot inherit the whole-CSR <code>higherOrder</code> scenario. A formula exemption's graph or iterations
directive must belong to the selected element or family scenario, even in a shared marked section.
Contributors intentionally outside lens coverage have a separate documented exemption.</p>
<p>The review resume lists in the coverage spec name contributors whose scenarios are still unfinished
and state the missing parts. When discovery starts recognizing an existing adapter, explicitly add
its missing work to that list or supply coverage; it does not inherit coverage merely by sharing a
package. A completed contributor must leave the resume list.</p>
<p>Twin verification groups directives by scenario, decodes one source record, and releases its arrays
after rendering that scenario's variants. For the same pattern in a host, call
<code>fromRecordedLensFile</code> once and pass its <code>DecodedLensFile</code> to <code>buildDecodedLensTwin</code> for each rung.
Use <code>buildLensTwin</code> at the serialized input boundary. Verification retains filenames and hashes
between scenarios; opt-in updates stage rendered text until all scans and renders succeed.</p>
<p>Independent of <code>COVERED_ENTRIES</code>, the spec checks that every lens slot marker is present, that every
directive is well formed, and that every record file names its scenario.</p>
<h2 id="writing-good-lenses">Writing good lenses</h2>
<ul>
<li>Draw from a real run. Folded state recovered by a GPU replay is labelled as a replay; a value is never
re-simulated on the CPU for display.</li>
<li>State observer effects. A timing lens says that timestamps disabled pass coalescing.</li>
<li>Choose captions that say what the reader should see, and make the key number the one fact a reader
should take away.</li>
<li>Keep data small and real: each scenario dataset is at most 150 KB, with its source and licence listed in
<code>modules/lenses/src/computation-lenses/data/README.md</code>.</li>
</ul>
<h2 id="see-also">See also</h2>
<ul>
<li><a href="/luma-contributors/docs/api/computation-record">GPU computation records</a></li>
<li><a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts">GPU Spatial Analysis concepts</a></li>
<li><a href="/luma-contributors/docs/api/gpu-network/concepts">GPU Network concepts</a></li>
<li><a href="/luma-contributors/docs/api/gpu-terrain/concepts">GPU Terrain concepts</a></li>
</ul>`};export{e as default};