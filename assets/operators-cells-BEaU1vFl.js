var e={slug:`api/gpu-spatial-analysis/operators-cells`,title:`GPU Spatial Analysis: Cells (DGGS)`,html:`<h1 id="cells-dggs">Cells (DGGS)</h1>
<h2 id="cell-system-support">Cell system support</h2>
<p>Shared cell operators support S2 topology, compaction, grid paths, center covering and outlines.
The separate S2 compatibility API retains its own status words and unit-sphere units
(steradians and radians, little-endian keys only). Aggregation/rollup still support H3 and Quadbin.</p>
<p>Use <code>GPUCellAggregationFamily</code> (<code>'quadbin' | 'h3'</code>) for aggregation, rollup and pyramid inputs,
and <code>GPUCellTopologyFamily</code> (<code>'quadbin' | 'h3' | 's2'</code>) for topology, compaction, grid paths,
covering and outlines. Both types are exported from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.
<code>GPUCellIndexFamily</code> additionally includes <code>'quadkey'</code> and <code>'geohash'</code> for indexing.
These capability types replace the two different types formerly named <code>GPUCellFamily</code>.</p>
<table>
<thead>
<tr>
<th>Family</th>
<th>Indexing / topology resolution</th>
<th><code>GPUCellCover</code> resolution and containment</th>
<th><code>GPUCellCompaction</code> expansion depth</th>
<th><code>GPUCellSetOutline</code> condition</th>
</tr>
</thead>
<tbody>
<tr>
<td>H3</td>
<td>0–15</td>
<td>0–15, center only</td>
<td>At most 11 levels per input cell</td>
<td>Sorted distinct keys</td>
</tr>
<tr>
<td>Quadbin</td>
<td>0–26</td>
<td>0–26, center, full or intersects</td>
<td>At most 15 levels per input cell</td>
<td>Sorted distinct keys</td>
</tr>
<tr>
<td>S2</td>
<td>0–30</td>
<td>0–15, center only</td>
<td>At most 15 levels per input cell</td>
<td>Sorted distinct keys, all at one level</td>
</tr>
<tr>
<td>quadkey / geohash</td>
<td>Indexing only: zoom 1–29 / length 1–12</td>
<td>Unsupported</td>
<td>Unsupported</td>
<td>Unsupported</td>
</tr>
</tbody>
</table>
<p>Topology children are additionally bounded by a 4096-cell output stride; disks and rings accept
a radius at most 8. The expansion depth is <code>target resolution - input cell resolution</code>, bounded
by <code>maximumDepth</code> (default 8) during uncompaction. S2 aggregation and non-center shared covering
remain unsupported.</p>
<table>
<thead>
<tr>
<th>Operation</th>
<th>H3</th>
<th>Quadbin</th>
<th>S2</th>
<th>quadkey / geohash</th>
<th>A5</th>
</tr>
</thead>
<tbody>
<tr>
<td>Point to cell</td>
<td><code>GPUPointToCell</code>; extended precision exact</td>
<td><code>GPUPointToCell</code>, also inside <code>GPUCellAggregation</code></td>
<td><code>GPUPointToCell</code>, f32 only, exact to about level 20</td>
<td><code>GPUPointToCell</code></td>
<td>none</td>
</tr>
<tr>
<td>Cell to center / boundary</td>
<td><code>GPUCellGeometry</code> (6 to 10 vertices)</td>
<td><code>GPUCellGeometry</code></td>
<td><code>GPUCellGeometry</code></td>
<td><code>GPUCellGeometry</code></td>
<td><code>GPUCellGeometry</code></td>
</tr>
<tr>
<td>Parent</td>
<td><code>GPUCellTopology</code></td>
<td><code>GPUCellTopology</code></td>
<td><code>GPUCellTopology</code>; <code>GPUS2CellHierarchy</code> compatibility</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Children</td>
<td><code>GPUCellTopology</code>, any depth to stride 4096</td>
<td>same</td>
<td><code>GPUCellTopology</code>, any depth to stride 4096; compatibility <code>GPUS2CellHierarchy</code> also provides leaf ranges</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>k-ring / disk</td>
<td><code>GPUCellTopology</code>, k at most 8</td>
<td><code>GPUCellTopology</code>, k at most 8 (Chebyshev)</td>
<td><code>GPUCellTopology</code>, k at most 8 (edge steps)</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Grid distance / path</td>
<td><code>GPUCellGridPath</code></td>
<td><code>GPUCellGridPath</code></td>
<td><code>GPUCellGridPath</code>, levels 0–30</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Compact / uncompact</td>
<td><code>GPUCellCompaction</code></td>
<td><code>GPUCellCompaction</code></td>
<td><code>GPUCellCompaction</code>, expansion depth ≤15</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Polyfill / cover</td>
<td><code>GPUCellCover</code> (scales)</td>
<td><code>GPUCellCover</code> (scales)</td>
<td><code>GPUCellCover</code> (center, level ≤15); <code>GPUS2PolygonCovering</code> (one polygon; <code>refinement: 'level-synchronous'</code> is parallel), <code>GPUS2CapCovering</code> (level at most 8, or 15 with <code>refinement: 'level-synchronous'</code>)</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Aggregation, rollup, pyramid</td>
<td><code>GPUCellAggregation</code> (pre-keyed only), rollup, pyramid</td>
<td>all, including points</td>
<td>none</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Cell set outline</td>
<td><code>GPUCellSetOutline</code></td>
<td><code>GPUCellSetOutline</code></td>
<td><code>GPUCellSetOutline</code></td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Cell area, perimeter</td>
<td><code>GPUCellMeasures</code>, metres or km</td>
<td><code>GPUCellMeasures</code>, metres or km</td>
<td><code>GPUCellMeasures</code>, metres or km; <code>GPUS2CellMeasures</code> for unit-sphere compatibility</td>
<td>none</td>
<td>none</td>
</tr>
<tr>
<td>Edge / vertex identities</td>
<td><code>GPUH3CellIdentities</code></td>
<td>not applicable</td>
<td>none</td>
<td>not applicable</td>
<td>not applicable</td>
</tr>
</tbody>
</table>
<p>The separate <code>GPUS2PolygonCovering</code> and <code>GPUS2CapCovering</code> take unit xyz vectors and an optional <code>spatialContext</code> (longitude-latitude, great-circle
edges); see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/global-coordinates">Global coordinates</a> for their limits and status words.</p>
<h3 id="gpucellaggregation-gpucellrollup-gpucellpyramid-and-gpucelllevelselection"><code>GPUCellAggregation</code>, <code>GPUCellRollup</code>, <code>GPUCellPyramid</code>, and <code>GPUCellLevelSelection</code></h3>
<p><code>GPUCellAggregation</code>: Aligned chunked inputs use fixed tile sorting scratch controlled by
scratchTileSize. Aggregate padded tile rows are at most 4,294,967,295; packed global aggregate
outputs remain device-binding bounded.</p>
<p><strong><code>GPUCellAggregation</code></strong> — Counts and sums rows per Quadbin or H3 cell into a sorted, capacity-bounded cell table.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=result caption="Blue: Quadbin zoom-18 cells shaded by place count · grey: 1,275 Wicker Park places"}</p>
<p><strong><code>GPUCellRollup</code></strong> — Reduces a cell table to a coarser resolution by summing child counts and sums, without re-reading the rows.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-rollup rung=result caption="Grey: 115 zoom-18 cells · blue: 12 zoom-16 parents shaded by summed place count"}</p>
<p><strong><code>GPUCellPyramid</code></strong> — Aggregates rows once at the finest resolution, then rolls each level into the next coarser table.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=result caption="Grey: zoom-18 level · blue: zoom-16 level of the same pyramid, shaded by place count"}</p>
<p><strong><code>GPUCellLevelSelection</code></strong> — Publishes the cell count and first row of a pyramid level chosen by one per-frame index.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=result caption="Blue: the selected zoom-17 level, shaded by place count · grey: places"}</p>
<p>Aggregates rows into discrete global grid cells (Quadbin or H3) and builds a zoom pyramid of
cell tables. 64-bit cell keys are stored as two <code>uint32</code> words in little-endian <code>(low, high)</code>
order, which is the layout of Arrow <code>Uint64</code> columns and of <code>GPUH3CellProjection</code> input. An H3
table therefore feeds the projection directly.</p>
<ul>
<li><strong>Inputs.</strong> <code>positions</code> (lng/lat degrees, Quadbin only) are keyed in WGSL. <code>cells</code> are
pre-keyed rows, for example CARTO Quadbin or H3 columns, as <code>uint32x2</code> (<code>wordOrder: 'high-low'</code> is also accepted). Pre-keyed rows finer than <code>resolution</code> are truncated to it.
Invalid rows, and rows coarser than <code>resolution</code>, are skipped. H3 validity comes from
gpu-dggs <code>dggs_h3_is_valid_cell_id</code>. Optional <code>values</code> and a <code>mask</code> can change every frame.
Rows with a NaN coordinate or a non-finite value are skipped.</li>
<li><strong>Coordinate space.</strong> <code>spatialContext</code> is checked at construction and exposed as a resolved
field. <code>positions</code> are longitude/latitude degrees (default <code>{coordinateSpace: 'longitude-latitude', metric: 'none'}</code>); a planar context throws, because Quadbin keying is
defined on degrees. Pre-keyed <code>cells</code>, <code>GPUCellRollup</code> and <code>GPUCellPyramid</code> use a <code>discrete-grid</code>
context (default <code>gridFamily</code> is <code>family</code>): a given <code>gridFamily</code> must equal <code>family</code>, and a given
<code>gridResolution</code> (the finest resolution of the input keys) must be at least <code>resolution</code>. Nothing
is measured, so only metric <code>'none'</code> is accepted.</li>
<li><strong>Cell table</strong> (<code>GPUCellTable</code>). Capacity is <code>cells.length</code>. Rows <code>[0, status.count)</code> hold occupied
cells in ascending key order. Later rows hold the empty key <code>0xffffffff</code> in both words, count 0
and NaN extremes. Columns: <code>counts</code>, fixed-point <code>sums</code> (signed 64-bit words of
<code>roundHalfEven(fround(value * sumScale))</code>, default scale 65536), <code>sumValues</code> (f32
<code>sums / sumScale</code>), <code>minimums</code> and <code>maximums</code>. On overflow the table keeps the smallest keys
and sets <code>status.overflow</code>. That flag describes only this destination. Optional
<code>status.candidateOverflow</code> reports source-table loss and is cleared to zero by raw aggregation.
Rollup destinations and every coarser pyramid level must provide that scalar; a missing scalar
is rejected at construction. Forward the whole status when reusing a table as input.
Final <code>overflow</code> and <code>candidateOverflow</code> publish in separate dispatches, so their one-word
views may occupy adjacent words (for example byte offsets 0 and 4) in one buffer. All output
logical byte ranges must remain disjoint. Fields bound together, such as <code>count</code>, <code>overflow</code>
and <code>requiredCount</code>, also need disjoint storage binding ranges after device-offset alignment.
Rollup accumulation also binds <code>counts</code> and <code>sums</code> writable together: their aligned ranges
must be disjoint, including at every coarser pyramid level. For one count at byte 0 and one
sum at byte 8, the logical ranges are disjoint but a 256-byte device alignment makes the
sum binding start at 0, overlapping the count binding; construction rejects that layout.
Raw aggregation writes these columns in separate dispatches and accepts that layout.
Aggregation, rollup, pyramid and comparison constructors capture declared props, table ports
and status ports once, including inherited or getter-backed fields; unrelated metadata is
ignored. Replacing caller fields afterwards
does not change the declared graph. Buffer contents remain per-frame inputs.</li>
<li><strong>Accuracy of sums.</strong> The integer sum is exact, order independent and equal under roll-up; the
values are not: each row is rounded to the nearest multiple of <code>1 / sumScale</code> before it is added.
With the default 65536 that is an absolute error of up to <code>2^-17</code> (7.6e-6) per row, a value
below it adds 0, and a column of constant <code>1e-4</code> sums 6.8% high (<code>7 / 65536</code> per row). The
default suits values of about 0.01 and above. For small or very large values pass
<code>sumScale: getGPUCellSumScale(maximumMagnitude, maximumRowCount)</code> (exported beside
<code>GPU_CELL_DEFAULT_SUM_SCALE</code>): the largest normal float32 power of two, clamped to
<code>[2^-126, 2^127]</code>, that keeps <code>maximumRowCount</code> rows of that magnitude at most <code>2^61</code> when
that bound is representable. Positive finite magnitudes, including <code>Number.MIN_VALUE</code>, are
accepted; tiny magnitudes select <code>2^127</code>. When that upper cap does not constrain the scale,
float32 rows within a factor of 32 of the maximum are added without rounding at the default
row count of <code>2^31</code>. When it does, even the maximum can be quantized: the normal float32 value
<code>1.25 * 2^-126</code> scales to 2.5, rounds to even at 2, and decodes as <code>2^-126</code> (20% low).
Use one <code>sumScale</code> for every table that is rolled
up or compared together (<code>GPUCellAggregation</code>, <code>GPUCellRollup</code>, <code>GPUCellPyramid</code>,
<code>GPUCellTableCompare</code>).</li>
<li><strong>Roll-up.</strong> <code>GPUCellRollup</code> reduces a table to a coarser resolution without re-reading rows
or sorting. Parent keys truncate the cell path (<code>cellToParent</code>), which preserves key order.
Counts and sums are integer sums, so a roll-up equals direct aggregation bit for bit. The
source table's <code>overflow</code> and inherited <code>candidateOverflow</code> are combined into the parent's
<code>candidateOverflow</code>; its own <code>overflow</code> is set only when parent rows exceed its capacity.
<code>requiredCount</code> counts parents formed from retained source cells, so it is a lower bound on
the full parent requirement after source loss. Growing only the parent destination cannot
recover cells already dropped by a finer stage.</li>
<li><strong>Pyramid.</strong> <code>GPUCellPyramid</code> aggregates once at the finest of <code>resolutions</code> (listed finest first),
rolls each level into the next, writes one table per level to <code>output.levels</code> and can write
per-level counts to <code>output.levelCounts</code>. Intermediate
sums and extremes that a coarser level needs become transients. Each level keeps final and
inherited loss separate, and inherited loss propagates through all remaining coarser levels.
Read both loss flags for every displayed level. Either flag means that level is incomplete;
its <code>levelCounts</code> entry still reports the number of retained rows.</li>
<li><strong>Level selection.</strong> <code>GPUCellLevelSelection</code> reads a per-frame <code>activeLevel</code> (one <code>uint32</code>) and
writes that level's <code>status.count</code>, first row, and the <code>instanceCount</code>/<code>firstInstance</code> words of indirect
draw arguments. Keep it in a small per-frame graph: switching levels rewrites one word and never
re-runs the pyramid.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Allocate every table with status: {count, overflow, candidateOverflow}.</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Read both loss flags after execution; either makes that level incomplete.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellPyramid</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'quadbin'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, values, mask,                    </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// mask and values can change every frame</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  resolutions: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">18</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">14</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {levels: tables, levelCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">selectionGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellLevelSelection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  levelCounts, activeLevel: activeLevel.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(selectionGraph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {status: {count}, drawArguments}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">activeLevel.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(Uint32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">levelForZoom</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(zoom)));   </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li><strong>Determinism.</strong> Point keys use only integer arithmetic. The longitude column is exact. The
latitude row is a fixed-point Mercator (<code>ln(cos t) - ln(sin t)</code>, Q0.32 polynomials, exact long
division), so every adapter and the CPU reference agree bit for bit. f32 polynomials would not
agree, because WebGPU compilers may fuse <code>a * b + c</code> (Metal does). Sorting uses one stable
<code>GPUSort</code> when the key path fits 31 bits (Quadbin ≤ 15, H3 ≤ 8) and two otherwise. Rows
accumulate with u32 atomics only: counts, 64-bit sums as two words with a carry, and extremes
as order-preserving keys.</li>
<li><strong>Accuracy.</strong> Compared with the f64 quadbin-py formula, the fixed-point row differs for 0% of
random points up to resolution 8, 0.01% at 12, 0.1% at 16, 1.1% at 20, 17% at 24 and 55% at
26. It is never off by more than 1 row up to resolution 22, 2 rows at 24 and 5 rows at 26
(f32 latitudes are not that precise). The column always matches.</li>
<li>Aligned chunked inputs use fixed tile scratch; packed aggregate outputs remain device-bound. Forward keying of points is Quadbin only.</li>
</ul>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUCellAggregation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=datum datum=0 caption="The first occupied cell: its key and how many places fell inside"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=formula caption="Cell count is the number of accepted rows keyed into the cell"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=kernel caption="Key, sort, scan segment heads, then atomic adds into table rows"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=graph caption="Keying, radix sort, head scan and the per-cell reduction nodes"}</p>
<p>{/* lens:GPUCellRollup */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-rollup rung=datum datum=0 caption="The first parent cell: its count is the sum of its children"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-rollup rung=formula caption="Parent counts are integer sums of child counts, so totals are conserved"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-rollup rung=kernel caption="Truncate each key to its parent, scan heads, reduce children per parent"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-rollup rung=graph caption="One roll-up node reads the source table and writes the parent table"}</p>
<p>{/* lens:GPUCellPyramid */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=datum datum=0 caption="The first zoom-16 cell: its count rolled up through zoom 17"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=formula caption="Each level sums the counts of the finer level below it"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=kernel caption="Aggregation kernels at level 0, roll-up kernels for each coarser level"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=graph caption="One aggregation, two chained roll-ups and the level-count packing node"}</p>
<p>{/* lens:GPUCellLevelSelection */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=datum datum=0 caption="The first cell of the selected zoom-17 level and its count"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=formula caption="The level is the active index clamped to the last level"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=kernel caption="One thread reads the active level and writes its count and first row"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=graph caption="A single selection node beside the pyramid it reads"}</p>
<h3 id="gpupointtocell-and-gpucellgeometry"><code>GPUPointToCell</code> and <code>GPUCellGeometry</code></h3>
<p><strong><code>GPUPointToCell</code></strong> — Keys longitude/latitude points to 64-bit cell keys of a discrete global grid family at one resolution.</p>
<p>::lens{scenario=gpu-spatial-analysis.point-to-cell rung=result caption="Grey: every 4th Wicker Park place · blue: the H3 resolution-9 cell each falls in"}</p>
<p><strong><code>GPUCellGeometry</code></strong> — Decodes cell keys to centres and boundary polygons in longitude/latitude degrees.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-geometry rung=result caption="Blue: H3 resolution-9 cell boundaries · grey: the places they were keyed from"}</p>
<p>Keys points to discrete global grid cells and turns cells back into centres and boundary
polygons. This covers the most common Studio/CARTO preprocessing step ("lng/lat column to H3")
and the geometry needed to draw any DGGS cell column. Keys are 64-bit <code>uint32x2</code> little-endian
<code>(low, high)</code> words, which is the Arrow <code>Uint64</code> layout that <code>GPUCellAggregation</code> pre-keyed rows
and <code>GPUH3CellProjection</code> read. The zero key means "no cell".</p>
<ul>
<li>
<p><strong>Families</strong> (<code>GPUCellIndexFamily</code>) and their resolutions:</p>
<table>
<thead>
<tr>
<th>Family</th>
<th>Resolutions</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'quadbin'</code></td>
<td>0-26</td>
<td>integer keys, exact</td>
</tr>
<tr>
<td><code>'h3'</code></td>
<td>0-15</td>
<td>f32 input: exact to res 4</td>
</tr>
<tr>
<td><code>'quadkey'</code></td>
<td>1-29</td>
<td>packed: length in bits 58-63, base-4 digits</td>
</tr>
<tr>
<td><code>'geohash'</code></td>
<td>1-12</td>
<td>packed: length in bits 60-63, base-32 characters, i.e. the lng/lat bit interleave</td>
</tr>
<tr>
<td><code>'s2'</code></td>
<td>levels 0-30</td>
<td>standard S2CellId</td>
</tr>
</tbody>
</table>
<p>The quadkey and geohash packings are the layouts the shadertools <code>dggs</code> decoders read.</p>
</li>
<li>
<p><strong><code>GPUPointToCell</code>.</strong></p>
<ul>
<li>Input is <code>positions</code> (lng/lat <code>float32x2</code>) and an optional <code>mask</code>. Output is <code>cells</code> and an
optional <code>validity</code>, one row per input row.</li>
<li>Masked and non-finite rows get the zero key. The resolution is compile-time.</li>
<li>Rewriting positions or the mask re-encodes without recompiling.</li>
<li><code>spatialContext</code> (optional) must be longitude/latitude without a metric, which is also the
default; planar contexts throw. <code>GPUCellGeometry</code> takes cell ids, so it has no <code>spatialContext</code>
prop; its decoded <code>centers</code> and <code>boundaries</code> are described by <code>outputSpatialContext</code>.</li>
</ul>
</li>
<li>
<p><strong><code>GPUCellGeometry</code>.</strong></p>
<ul>
<li>Input is <code>cells</code> (<code>wordOrder</code> option).</li>
<li>Outputs are <code>centers</code> (lng/lat) and optional <code>boundaries</code> with a fixed <code>maximumVertexCount</code>
stride and <code>vertexCounts</code>. Families are the five above plus <code>'a5'</code>.</li>
<li>An invalid key (the zero key included) gets a <code>NaN</code> centre, vertex count 0 and zero boundary
rows. The optional <code>validity</code> output (<code>uint32</code>, 1 for a valid cell, 0 otherwise) follows luma.gl
<code>GPUDGGSCellProjection</code>, which decodes H3 and A5 centres with the same shader functions but
writes a zero centre for an invalid cell.</li>
<li>H3 boundaries follow h3-js <code>cellToBoundary</code> exactly in vertex count and order. That includes
face-crossing distortion vertices (up to 10, so use stride 10) and pentagons.</li>
<li>Quadbin, quadkey, geohash and S2 have 4 vertices; A5 has 5.</li>
<li><code>edgeModel</code> and <code>outputSpatialContext.edgeModel</code> are <code>'great-circle'</code> for H3, S2 and A5,
and <code>'linear'</code> for quadbin, quadkey and geohash. Before linear polygon consumers, prepare
great-circle boundaries with <code>GPULonLatPolygonPreparation</code>; see the
<a href="#gpucellsetoutline">H3 edge-deviation table</a> below.</li>
</ul>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointToCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, output: {cells}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellGeometry</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, cells, maximumVertexCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {centers, boundaries, vertexCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><strong>Exactness and accuracy.</strong>
<ul>
<li>
<p>Quadbin, quadkey and geohash keying is integer-only and equals the BigInt oracles bit for bit
on every device.</p>
</li>
<li>
<p>S2 and H3 use f32 sphere math (polynomial sine/cosine, no trig builtins for H3) and are exact
at coarse resolutions. A mismatch is an edge neighbour of the f64 answer:</p>
<table>
<thead>
<tr>
<th>Family</th>
<th>Uniform-point mismatch rate vs the f64 reference (f32 inputs)</th>
<th>Usable limit</th>
</tr>
</thead>
<tbody>
<tr>
<td>H3 vs h3-js</td>
<td>0 at res 0-4, 1e-4-4e-4 at 5-8, 1e-3 at 9, 2e-3 at 10, 1.8% at 12, 28% at 15</td>
<td>res ≤ 12</td>
</tr>
<tr>
<td>S2</td>
<td>0 at levels 0-10, 1e-4-3e-3 at 11-15, 1.8% at 18, 13% at 21</td>
<td>level ≤ 20</td>
</tr>
</tbody>
</table>
</li>
<li>
<p>Geometry matches h3-js / oracles within about 3e-5 degrees.</p>
</li>
</ul>
</li>
</ul>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUPointToCell */}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-to-cell rung=datum datum=0 caption="Place 0: its longitude/latitude and the H3 key written for it"}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-to-cell rung=kernel caption="One thread per place runs the H3 face, IJK and digit encoder"}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-to-cell rung=graph caption="A single keying node: positions in, 64-bit cell keys out"}</p>
<p>{/* lens:GPUCellGeometry */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-geometry rung=datum datum=0 caption="Cell 0: its centre and the six boundary vertices decoded from its key"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-geometry rung=kernel caption="One thread per key decodes the centre and boundary vertices into fixed strides"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-geometry rung=graph caption="A single decode node: cell keys in, centres, boundaries and counts out"}</p>
<h3 id="gpucelltopology-and-gpucellcompaction"><code>GPUCellTopology</code> and <code>GPUCellCompaction</code></h3>
<p><code>GPUCellCompaction</code>: Supports H3, Quadbin and S2. S2 levels are 0–30; compaction expansion depth is
at most 15. S2 covering uses level-15 center containment, rather than exact polygon intersection.</p>
<p><strong><code>GPUCellTopology</code></strong> — Expands each cell key to its grid disk, ring, parent or children as a fixed-stride row.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-topology rung=result caption="Blue: the radius-4 disk around the central place, shaded by grid steps from its cell"}</p>
<p><strong><code>GPUCellCompaction</code></strong> — Merges complete sibling sets of cells into parents (compact) or expands cells to descendants (uncompact).</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-compaction rung=result caption="Grey: 49 resolution-9 cells, two masked out · blue: the merged set, resolutions 8 and 9"}</p>
<p>Cell topology for Quadbin, H3 and S2 keys stored as <code>uint32x2</code> little-endian <code>(low, high)</code> words.
<code>GPUCellCompaction</code> supports Quadbin, H3 and S2 (levels 0–30; expansion depth at most 15).
This is the Arrow <code>Uint64</code> layout used by <code>GPUCellAggregation</code>. Inputs also accept
<code>wordOrder: 'high-low'</code>. A zero key means "no cell" in every output.</p>
<p>Both consume cell keys, not positions, so neither takes a <code>spatialContext</code>.</p>
<ul>
<li><strong><code>GPUCellTopology</code></strong> writes a fixed fan-out row per input cell, with stride from
<code>getCellTopologyStride(family, operation)</code>:
<ul>
<li><code>{type: 'disk', k}</code>: <code>gridDisk</code> / <code>gridDiskDistances</code>, sorted by (distance, key). Optional
<code>distances</code> column.</li>
<li><code>{type: 'ring', k}</code>: <code>gridRing</code>, sorted by key.</li>
<li><code>{type: 'parent', resolution}</code>: <code>cellToParent</code>.</li>
<li><code>{type: 'children', resolution, inputResolution}</code>: <code>cellToChildren</code>, in ascending key order.</li>
<li>Rows are zero padded and an optional <code>counts</code> column gives the valid entries per row. <code>k</code> is at
most 8 and the children stride at most 4096 (H3 depth 4, Quadbin/S2 depth 6). Invalid, masked
or wrong-resolution rows give zero rows.</li>
</ul>
</li>
<li><strong>H3 neighbors are exact.</strong> <code>H3_NEIGHBOR_WGSL</code> ports H3's <code>h3NeighborRotations</code>: base-cell
neighbor and rotation tables, class II/III digit carry tables and pentagon clockwise-offset
faces. Disks and rings are a bounded breadth-first search per thread over the six neighbor
steps. The search is integer-only and exact across icosahedron faces and pentagons. A pentagon's
disk has fewer cells, and the rest of the row is zero padded.</li>
<li><strong>Quadbin neighborhoods</strong> use Chebyshev distance on tile column and row (<code>QUADBIN_KRING</code>
style). Columns wrap across the antimeridian. Rows are clipped at the poles. A tile that the wrap
reaches twice at small zoom levels appears once.</li>
<li><strong>S2 neighborhoods</strong> count steps across the four edge neighbors, including exact integer
cube-face transitions through level 30. A diagonal contact is not a step. Disks have stride
<code>2k(k+1)+1</code>, rings <code>4k</code> (both 1 at <code>k = 0</code>); these are flat square-grid upper bounds, so
cube corners and small levels leave zero padding. Output order is (distance, key) for disks
and key for rings. Parents accept mixed levels; children require the declared <code>inputResolution</code>
and enumerate every descendant in ascending Hilbert-key order. <code>GPUS2CellHierarchy</code> remains
available for its compatibility statuses, immediate children, edge-neighbor slots and leaf ranges.</li>
<li><strong><code>GPUCellCompaction</code></strong> handles sets of cells:
<ul>
<li><code>{type: 'compact', minimumResolution?}</code> (<code>compactCells</code>): input is cells at one resolution,
which are sorted unless <code>sorted: true</code> and then deduplicated. It replaces every complete
sibling set by its parent, recursively: 4 children for Quadbin, 7 for an H3 hexagon, and for an
H3 pentagon 6 at depth 1, <code>(5 * 7^d + 1) / 6</code> in general.</li>
<li><code>{type: 'uncompact', resolution, maximumDepth?}</code> (<code>uncompactCells</code>): count, scan, then emit
the descendants at <code>resolution</code>.</li>
<li>Both write a capacity-bounded <code>{cells, status: {count, requiredCount?, overflow, droppedCount?}}</code>.
Compact output is in ascending canonical key order. Uncompact output is ascending when the
input is in path order.</li>
</ul>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellTopology</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, operation: {type: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'disk'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, k: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cells, output: {cells: diskCells, distances, counts}   </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// diskCells.length = rows * 19</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellCompaction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, operation: {type: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'compact'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cells: polyfillCells, output: {cells: compacted, status: {count, overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><strong>Determinism.</strong> All operations are integer-only, with fixed sort orders. GPU results equal the
CPU oracles (h3-js and BigInt Quadbin) bit for bit, including padding.</li>
</ul>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUCellTopology */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-topology rung=datum datum=0 caption="Disk cell 0: the input cell itself at grid distance 0"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-topology rung=kernel caption="One thread per input cell walks the six neighbour steps breadth first"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-topology rung=graph caption="A single topology node: cell keys in, disk cells and distances out"}</p>
<p>{/* lens:GPUCellCompaction */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-compaction rung=datum datum=0 caption="Compacted cell 0: the lowest key after sorting and merging"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-compaction rung=kernel caption="Sort, deduplicate, then ancestor-completeness checks merge sibling sets"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-compaction rung=graph caption="Sort, scan and compaction passes between input cells and the bounded table"}</p>
<h3 id="gpucellgridpath-and-gpucellmeasures"><code>GPUCellGridPath</code> and <code>GPUCellMeasures</code></h3>
<p><code>GPUCellGridPath</code>: Supports H3, Quadbin and S2 paths. Aligned GraphVectorView cell-pair and mask
rows support chunked input; complete fixed-stride paths stay within one output chunk.</p>
<p><strong><code>GPUCellGridPath</code></strong> — Computes H3, Quadbin or S2 grid distance and the cells of the grid path between origin and destination cells.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=result caption="Grey: 7 origin places · blue: grid-path cell centres to the central place"}</p>
<p><strong><code>GPUCellMeasures</code></strong> — Computes the spherical area, perimeter, ordered edge lengths and pentagon flag of H3, S2 and Quadbin cells.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-measures rung=result caption="Blue: 160 H3 resolution-9 cells by spherical area, 106,248–106,278 m², near-equal here"}</p>
<p>H3/S2/Quadbin grid paths and cell measures, integer-exact where possible.</p>
<p><code>GPUCellGridPath</code> consumes cell keys, so it takes no <code>spatialContext</code>.</p>
<ul>
<li><strong><code>GPUCellGridPath</code></strong> (h3 <code>grid_distance</code>, <code>grid_path_cells</code>, <code>are_neighbor_cells</code>): per pair,
<code>origins</code> and <code>destinations</code> (<code>uint32x2</code>) give <code>output.distances</code> (the H3 <code>gridDistance</code>, or
<code>GPU_CELL_GRID_DISTANCE_UNDEFINED</code> = <code>0xffffffff</code> where H3 fails: invalid input, resolution mismatch,
non-adjacent base cells, behind a pentagon's deleted sector, masked), <code>output.cells</code> (<code>gridPathCells</code>,
stride <code>maximumPathLength</code> up to 256, zero padded) and <code>output.counts</code> (the true length <code>distance + 1</code>;
above <code>maximumPathLength</code> means overflow, 0 means no path). Omit <code>cells</code> for distance only;
<code>are_neighbor_cells</code> is <code>distance == 1</code>. It is <code>cellToLocalIjk</code> with pentagon rotations and an
integer cube-round. Known deviations: about 0.2% of random pairs differ from H3 by one adjacent cell at
exact rounding ties (H3 rounds with double noise), and paths crossing pentagon base cells fail
(<code>counts = 0</code>) in about 6% of those cases. Distances matched h3 4.5.0 on 355 of 355 pairs, failures
included.</li>
<li><strong><code>GPUCellMeasures</code></strong> (h3 <code>cell_area</code>, <code>edge_length</code>, <code>is_pentagon</code>): <code>areas</code> (square meters),
<code>perimeters</code>, <code>edgeLengths</code> (stride <code>maximumEdgeCount</code> 6 to 10 for H3, 4 to 10 for S2/Quadbin;
defaults to 6 for H3 and 4 otherwise), <code>edgeCounts</code> and <code>pentagons</code>. H3 edges follow
<code>cellToBoundary</code>; S2 follows geometry face-UV order <code>(00, 01, 11, 10)</code>. Quadbin follows north, east, south,
west: north/south run along parallels and east/west along meridians. Invalid keys give all-zero
rows, valid S2/Quadbin cells have four edges and a zero pentagon flag, and edges beyond the
cell count are zero. <code>maximumEdgeCount</code> never truncates <code>edgeCounts</code>, areas or perimeters. <strong>Breaking:</strong> outputs used to
be kilometers; meters are now the default and kilometers need an explicit <code>units: 'kilometers'</code>
(then areas are square kilometers). The sphere is the WGS84 authalic sphere
(<code>GPU_EARTH_AUTHALIC_RADIUS_METERS</code>, replacing the removed <code>GPU_CELL_MEASURES_EARTH_RADIUS_KM</code>), and
the instance's <code>measures</code> field is the machine-readable description (<code>lengthModel: 'great-circle'</code>,
<code>areaModel: 'sphere-polygon'</code>, units, for H3/S2; <code>lengthModel: 'sphere-parallel-meridian'</code> and
<code>areaModel: 'sphere-rectangle'</code> for Quadbin). Cell identifiers carry no coordinates, so the class has no
<code>spatialContext</code>. Single-face hexagons are exact from lattice
offsets, within 1e-6 relative at resolutions 0 to 15. Pentagons and cells straddling a face edge use f32
boundary vectors, which resolve about 0.4 m on the ground: measured relative error 6e-7 at resolution 0,
5e-5 at 3 and 4, at most 5e-4 up to resolution 8 and 2e-2 at 9 and 10 (areas alone: 2e-5 at 4, 1.5e-4 at 8,
5e-3 at 9 and 10), so treat them as unreliable beyond resolution 10. S2 uses stable local quadratic
increments through level 30. Quadbin uses hyperbolic difference identities through zoom 26;
its root cell covers the full Mercator band and excludes the polar caps. Use a relative tolerance
of 3e-6 for S2/Quadbin GPU measures. The CPU twin <code>computeCellMeasuresOnCPU</code> and
<code>CPUCellMeasuresOperation</code> support the same families, units, word orders and output layouts;
binary64 vector differences serve as an independent S2 evaluation.
Set <code>precision: 'fp64'</code> to use the published luma.gl double-single shader module for H3
boundary projection, face-edge intersections and spherical-vector differences, including
pentagons and face-crossing cells above resolution 10. S2 uses double-single quadratic-map
coordinates and local increments, and Quadbin uses double-single Mercator coordinates and
hyperbolic functions. The option is compile-time; it changes neither ports nor float32 output
formats. It defaults to <code>'float32'</code>, preserving the existing evaluation. GPU/CPU parity specs
require relative tolerances of 2e-5 for H3 and 3e-6 for S2/Quadbin. The CPU twin accepts either option and computes
in binary64 for both. The fp64 path costs more arithmetic per row and should be selected when
the precision is needed. Both graph operation forms
infer the same default edge stride from <code>family</code>. Open: directed edges and vertexes.</li>
</ul>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUCellGridPath */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=datum datum=0 caption="Pair 0: origin place, grid distance and the cells on its path"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=formula caption="Grid distance is the spread of the cube-coordinate differences"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=kernel caption="Local IJK conversion, distance, then cube-rounded interpolation of path cells"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=graph caption="A single grid-path node: origin and destination keys in, paths out"}</p>
<p>{/* lens:GPUCellMeasures */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-measures rung=datum datum=0 caption="Cell 0: area and perimeter, with the hexagon flag"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-measures rung=formula caption="Area sums spherical triangles from the cell centre to each edge"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-measures rung=kernel caption="One thread per cell decodes the boundary and sums edge lengths and areas"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-measures rung=graph caption="A single measures node: cell keys in, areas and perimeters out"}</p>
<h3 id="gpucellcover"><code>GPUCellCover</code></h3>
<p>Supports H3, Quadbin and S2. S2 compaction/outline levels are 0–30 and compaction expansion
depth is at most 15. S2 covering is limited to level-15 center containment, rather than exact
polygon intersection.</p>
<p>Polyfills polygons into H3, Quadbin or S2 cells, emitting (feature id, cell) pairs.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-cover rung=result caption="Grey: 17 community areas · blue: 163 resolution-8 H3 cells whose centres fall inside"}</p>
<p>Polyfills polygons (with holes) into Quadbin, H3 or S2 cells, like CARTO <code>QUADBIN_POLYFILL</code> /
<code>H3_POLYFILL</code> and h3-js <code>polygonToCells</code>. The result is <code>(feature id, cell)</code> pairs that feed
<code>GPUCellAggregation</code> pre-keyed rows, <code>GPUKeyJoin</code> and <code>GPUCellCompaction</code>.</p>
<p>Maximum: <code>candidateCapacity</code> is at most <code>getGPUCellCoverMaximumCandidateCapacity(device)</code> (2^24 on a default device), and with the edge-slab index <code>positions.length + edgeSlabEntryCapacity</code> is at most <code>getGPUCellCoverMaximumEdgeSlabEntryCount(device)</code> (2^24).</p>
<ul>
<li><strong>Inputs.</strong> Polygon features use the four-level layout: <code>positions</code>
(lng/lat <code>float32x2</code>), <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>, and optional
<code>featureIds</code>. Geometry is planar in lng/lat degrees, as in h3-js and turf. Containment is
even-odd over all rings, decided exactly (exact orientation signs, a cell centre exactly on an edge
resolved by the shared Simulation-of-Simplicity tie rule of <code>utils/exact-geometry-wgsl.ts</code>).
<code>spatialContext</code> defaults to, and must be, longitude/latitude without a
metric and with linear edges (planar contexts and great-circle edges throw).</li>
<li><strong>Antimeridian and poles.</strong> A feature with a non-pole edge whose raw longitude difference exceeds
180 degrees gets no cells and increments <code>output.status.invalidCount</code>; prepare it with
<code>GPULonLatPolygonPreparation</code> first. Pre-split multipolygons and same-pole edges are accepted,
including split Fiji and Natural-Earth Antarctica. Empty rings contribute no bounds.
Full-width features enumerate full-width lattices; with <code>candidateEnumeration: 'bounds'</code>,
provision <code>candidateCapacity</code> accordingly and inspect <code>candidateCount</code> and <code>candidateOverflow</code>.
A feature reaching past the cover's latitude limit
(89 degrees for H3, 85.051129 for Quadbin) loses its polar cells and is counted in the optional
<code>output.status.uncertainCount</code>. Both scalars are cleared on every encoding.</li>
<li><strong>Containment.</strong>
<ul>
<li><code>'center'</code>: the cell centre is inside the polygon.</li>
<li><code>'full'</code>: the cell lies inside the polygon.</li>
<li><code>'intersects'</code>: the cell interior meets the polygon.</li>
<li>Quadbin supports all three modes. A Quadbin tile is an axis-aligned lng/lat rectangle, so
<code>full</code> and <code>intersects</code> are exact edge-versus-open-rectangle tests. H3 supports <code>'center'</code>
only.</li>
</ul>
</li>
<li><strong>Candidates.</strong> Quadbin tests the integer-exact tile range of each feature's bounding box. H3
lays a lng/lat lattice over the bounding box, with spacing below half the minimum H3 inradius at
that resolution (measured from h3-js; the minimum sits next to a pentagon). The longitude spacing
is taken at the feature's most equatorward latitude (0 when it spans the equator), so tall
features are oversampled toward the pole rather than missing cells nearer the equator. Each lattice point
indexes its cell with <code>cellIndexH3FromLngLat</code>. The cell is emitted only from the lattice point
nearest its centre, so every cell whose centre lies in the polygon is emitted exactly once with
no sort.</li>
<li><strong>Sparse polygons (<code>candidateEnumeration</code>).</strong> A thin diagonal polygon (a river, a road buffer, a
coastline strip) or a sparse multipolygon has a bounding box that is mostly empty, so testing every
box cell is wasteful and can overflow <code>candidateCapacity</code>. Both lattices are axis aligned in
degree space (Quadbin tile rows are latitude bands, H3 sample rows are constant-latitude lines),
so per lattice row the cells that can be accepted lie in x intervals that are found by a boundary
trace plus a scanline fill, and only cells in those intervals are candidates:
<ul>
<li>The boundary trace emits, for every edge and every row its y range touches, the x span of the
edge part inside the row band. The scanline fill takes the crossings of every edge with the
row's test line (half-open in y, so a vertex on the line counts once), sorts them by row and x
(stable radix sorts) and pairs them even-odd over all rings, which gives the interior intervals.</li>
<li>Spans and interior intervals are swept per row, grown by a proven f32 slack (a few ulps of
the crossing, plus half a column for H3, whose accepted centre rounds to the lattice point) and
converted to disjoint column runs. The exact test, the cell order and the output are unchanged,
bit for bit, because the intervals contain every cell the test can accept.</li>
<li><code>candidateEnumeration: 'bounds' | 'scanline' | 'auto'</code> (compile-time, default <code>'auto'</code>). With
<code>'auto'</code> the stage is built when <code>candidateCapacity</code> is at least 65536, and each feature picks
the scanline runs on the GPU when its box has more than four times as many cells as the
scanline work (twice the rows plus the edge-row slots). A feature that does not fit
<code>scanlineSlotCapacity</code> (default <code>min(candidateCapacity, 2^20, max(16 * vertices, 16384))</code>),
has a non-finite vertex or would enumerate 2^32 candidates keeps its bounding box. The built
stage runs a fixed set of passes (two sorts and several scans over three events per slot)
whether or not a feature uses it, so size <code>scanlineSlotCapacity</code> to the polygons at hand or use
<code>'bounds'</code> for compact polygons only.</li>
<li><code>candidateCapacity</code> counts enumerated candidates, so with the stage it covers far larger
bounding boxes of thin or sparse polygons. <code>output.status.candidateCount</code> (optional) reports the
unclamped enumerated total.</li>
</ul>
</li>
<li><strong>Output.</strong> <code>{featureIds, cells (uint32x2 little-endian), status: {count, requiredCount?, overflow, candidateOverflow, candidateCount?, invalidCount?, uncertainCount?}}</code>. <code>status.overflow</code> means only that accepted cells exceeded the final
cell capacity; <code>status.candidateOverflow</code> means <code>candidateCapacity</code> truncated enumeration, in which case
<code>requiredCount</code> is not an exact full-cover requirement. Order is by feature, then candidate order.
The pipeline is count, scan, test, scan, write, with no atomic append, so it is deterministic.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellCover</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, resolution: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, containment: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'center'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, featureOffsets, polygonOffsets, ringOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 22</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {featureIds, cells, status: {count, overflow, candidateOverflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><strong>Accuracy.</strong>
<ul>
<li>Quadbin results equal an f32-faithful CPU oracle at resolutions 3-10 in all modes. Compared
with an independent f64 centre test there were 0 disagreements.</li>
<li>H3 <code>center</code> matches h3-js <code>polygonToCells</code> exactly at resolutions 3-7. At 8-10 it differs on
12 of about 75,000 cells. Each of those cells has its centre within 5e-5 degrees of a polygon
edge, where the f32 centre differs from the f64 one.</li>
<li>An f64 simulation of the lattice rule matches h3-js exactly, so the spacing never misses a
cell.</li>
</ul>
</li>
</ul>
<p><code>output.core</code> is an optional <code>uint32</code> column parallel to <code>output.featureIds</code> (the same length): 1 for a core
cell (a Mosaic chip, provably inside the feature) and 0 for a border cell. It works for every containment
and both families. It is conservative: a cell is core only when its center is inside and no polygon edge
enters the cell (for H3, its boundary bounding box) grown by 2e-4 degrees plus 0.1% of the cell size.
Border cells can still lie entirely inside. A join can skip the exact point-in-polygon test for points whose
cell is core.</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUCellCover */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-cover rung=datum datum=0 caption="Cell 0: its H3 key and the area it covers"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-cover rung=kernel caption="Count, scan, test, scan and write passes enumerate and test candidate cells"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-cover rung=graph caption="Candidate counting, exact containment and bounded emission as one graph"}</p>
<h3 id="gpucellsetoutline"><code>GPUCellSetOutline</code></h3>
<p>Supports H3, Quadbin and S2. S2 compaction/outline levels are 0–30 and compaction expansion
depth is at most 15. S2 covering is limited to level-15 center containment, rather than exact
polygon intersection.</p>
<p>Finds the boundary edges of a sorted H3, Quadbin or S2 cell set and chains them into closed rings.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=result caption="Grey: 163 covering H3 cells · blue: one closed ring chained from 140 exposed edges"}</p>
<p>The boundary edges of a set of H3, Quadbin or S2 cells, the <code>cellsToMultiPolygon</code> outline. Set <code>rings</code> to chain
the edges into closed rings with shells and holes on the GPU (<a href="#gpusegmentringassembly"><code>GPUSegmentRingAssembly</code></a>).</p>
<p>It consumes cell keys, so it takes no <code>spatialContext</code>; the segments and rings it produces are
longitude/latitude degrees. <code>edgeModel</code> and <code>outputSpatialContext.edgeModel</code> are <code>'great-circle'</code>
for H3 and <code>'linear'</code> for Quadbin; <code>outputSpatialContext.metric</code> is <code>'none'</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellSetOutline</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  family: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'h3'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cells, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32x2 little-endian keys, sorted ascending and distinct, e.g. a GPUCellAggregation table</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  count: tableCount, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional one-row valid row count</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  groups, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32 label per row: also emit edges between different groups</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {rows, cells: segmentCells, edgeIndices, endpoints, groups: segmentGroups, status: {count, overflow, requiredCount}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  rings: {normalizeWinding: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: ringOutput} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, see below</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li>The output columns share one capacity: <code>rows</code> (the input row, also the compact ID column), <code>cells</code>,
<code>edgeIndices</code>, <code>endpoints</code> (<code>float32x4</code>: lng0, lat0, lng1, lat1), optional <code>groups</code> and <code>status</code> with <code>count</code> (clamped),
<code>overflow</code> and optional <code>requiredCount</code>.</li>
<li>An edge is emitted when the neighbor across it is absent, or has a different group label. With <code>groups</code>,
a border between two groups is emitted from both sides, tagged with each side's group.</li>
<li>Order is by <code>(input row, edge index)</code>: classify, inclusive scan, write, with no atomics, so it is
deterministic.</li>
<li>H3: edge <code>e</code> is the <code>e</code>-th edge of <code>cellToBoundary</code> (up to 10 slots with distortion vertices, so such an
edge appears as two segments). The neighbor across it is the H3 neighbor whose center is nearest the
edge midpoint, a geometric rule validated against h3-js on hexagons, a pentagon disk and distortion
cells. Quadbin: 0 north, 1 east, 2 south, 3 west; columns wrap, and rows beyond the poles are absent, so
the map edge is an outline edge.</li>
<li>Membership is a binary search, so the input must be sorted and distinct. Invalid keys never emit.</li>
<li>Endpoints are f32. H3 cells that straddle the antimeridian keep their <code>cellToBoundary</code> longitudes, so a
segment can span more than 180 degrees. Other families (quadkey, geohash, S2, A5) are not supported.</li>
<li>Segments are directed with the cell interior on a fixed side: <code>outline.interiorSide</code> is <code>'left'</code> for H3
(counter-clockwise shells) and <code>'right'</code> for Quadbin (clockwise shells).</li>
<li><code>rings: {vertexTolerance?, normalizeWinding?, output: GPUSegmentRingAssemblyOutput}</code> composes
<code>GPUSegmentRingAssembly</code> after the outline, with <code>output.endpoints</code>, <code>output.status.count</code> and <code>output.groups</code>
as its inputs, <code>outputSpatialContext</code> as its context and <code>interiorSide</code> taken from the family. Use <code>normalizeWinding: true</code> for
counter-clockwise shells and clockwise holes in both families. Without <code>rings</code>, hole detection and ring
orientation are left to the caller.</li>
</ul>
<p>For an east-west edge of ground length <code>L</code> at latitude <code>phi</code>, the great-circle arc deviates from
the degree-straight edge by approximately <code>tan(phi) * L² / (8R)</code>. For average H3 edge lengths at 45°N:</p>
<table>
<thead>
<tr>
<th>H3 resolution</th>
<th>Average edge length</th>
<th>Approximate deviation</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>1281 km</td>
<td>32 km</td>
</tr>
<tr>
<td>1</td>
<td>483 km</td>
<td>4.6 km</td>
</tr>
<tr>
<td>2</td>
<td>183 km</td>
<td>660 m</td>
</tr>
<tr>
<td>3</td>
<td>69 km</td>
<td>93 m</td>
</tr>
<tr>
<td>4</td>
<td>26 km</td>
<td>13 m</td>
</tr>
<tr>
<td>5</td>
<td>9.9 km</td>
<td>1.9 m</td>
</tr>
</tbody>
</table>
<p>Run outlines of H3 resolution &#x3C;= 4 through <code>GPULonLatPolygonPreparation</code> with
<code>spatialContext.edgeModel: 'great-circle'</code> before linear consumers. At resolution >= 5, the
average-edge approximation at 45°N is within 2 m; this is not a global bound (multiply the table's
deviations by about 1.7 at 60°). Preparing preserves the declared great-circle meaning while
producing a linear output context. Global or antimeridian-crossing rings also need preparation.</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUCellSetOutline */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=datum datum=0 caption="Exposed edge 0: its owning table row and edge index"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=kernel caption="Neighbour lookup, scan and write passes keep the edges with no neighbour"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=graph caption="Outline classification, edge packing and ring assembly as one graph"}</p>
<h3 id="gpusegmentringassembly"><code>GPUSegmentRingAssembly</code></h3>
<p>vertexTolerance must round to positive finite float32 (at least 2^-149); caller and scratch bindings
are checked before graph declarations.</p>
<p>Chains directed boundary segments into closed rings with shell and hole classification.</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=result caption="Grey: 17 community areas · blue: one outer shell from 3,184 edges once shared borders cancel"}</p>
<p>Chains directed boundary segments into closed rings on the GPU, the ring-assembly step of
<code>cellsToMultiPolygon</code>. It consumes <code>GPUCellSetOutline</code> output, or any <code>float32x4</code> segment list whose
vertices meet within a tolerance.</p>
<p><code>spatialContext</code> replaces the former <code>geographic</code> flag. It defaults to native planar coordinates (<strong>behavior change:</strong> <code>geographic</code> defaulted to true). A <code>longitude-latitude</code> context with metric <code>'none'</code> scales x differences by <code>cos(latitude)</code> (clamped at 0.01) when comparing turns at shared vertices; great-circle and other metrics are rejected because ring assembly measures nothing, and <code>vertexTolerance</code> stays in position units. Hole detection uses the exact orientation of the turn at each ring's lowest vertex (smallest y, then x; the f32 area sign only when that turn is zero), so a sliver ring whose f32 shoelace sum cancels is still classified correctly.</p>
<p>The resolved <code>spatialContext</code> describes both input and output. For longitude/latitude it preserves
the input <code>edgeModel</code>, including omission; both <code>'linear'</code> and <code>'great-circle'</code> edges are accepted
with metric <code>'none'</code>. Pass <code>outline.outputSpatialContext</code> when assembling cell outline segments.
Prepare global or antimeridian-crossing rings with <code>GPULonLatPolygonPreparation</code> before polygon
consumers, and prepare great-circle rings before consumers that require linear edges.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSegmentRingAssembly</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  endpoints, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32x4 (x0, y0, x1, y1), e.g. GPUCellSetOutline output.endpoints</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  count, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional one-row valid segment count</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  groups, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional uint32 per segment: rings never mix groups</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  vertexTolerance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5e-5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// compile-time Chebyshev matching distance, input units</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  interiorSide: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'left'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// side of each segment that is filled ('left': counter-clockwise shells)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  normalizeWinding: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// shells counter-clockwise, holes clockwise (RFC 7946)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  splitTouchingRings: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// default</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialContext: {coordinateSpace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'longitude-latitude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'none'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// cos(lat) in turn comparisons; omit for planar</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ringOffsets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32, ringCapacity + 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32x2, vertex capacity</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ringAreas, ringIsHole, ringShells, ringGroups, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, ringCapacity rows</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    segmentRings, segmentVertices, segmentFlags, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional, one row per input segment</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    polygons: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: polygonPositions, ringOffsets: polygonRingOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      polygonOffsets, featureOffsets}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional four-level polygon port</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {count, overflow, requiredCount, openSegmentCount, touchingSegmentCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><strong>Layout.</strong> GeoArrow style: ring <code>r</code> is <code>positions[ringOffsets[r] .. ringOffsets[r + 1])</code>, closed (the last
vertex repeats the first). Rings are ordered by their lowest input segment index; a ring starts at that
segment and follows its direction (or is reversed by <code>normalizeWinding</code>). Offsets past <code>status.count</code> repeat the
final offset.</li>
<li><strong>Matching.</strong> Each segment end is matched to segments that start within <code>vertexTolerance</code> in the same
group, through a sort of quantized vertex hashes and a 3 x 3 neighborhood lookup. Quadbin vertices are
bit-exact; H3 vertices of neighboring cells agree within f32 round-off (below 5e-5 degrees). Pick a
tolerance of at least four times the f32 spacing of the coordinates and well below the shortest segment.
Without <code>vertexTolerance</code> the tolerance is four f32 ulps of the frame extent (<code>extent * 2^-21</code>, measured
on the GPU), so only separately rounded copies of one vertex match (<strong>behavior change:</strong> the default was
an absolute 5e-5). <code>GPUCellSetOutline</code> keeps 5e-5 degrees as its own preset
(<code>GPU_CELL_SET_OUTLINE_VERTEX_TOLERANCE</code>), valid up to H3 resolution 12 and Quadbin zoom 22; pass about
a quarter of the shortest edge for finer cells. Vertices are hashed relative to the first segment's start vertex into unsigned cells of size
<code>max(vertexTolerance, extent * 2^-22)</code> (with a small slack), where <code>extent</code> is the largest coordinate
offset from that origin, measured on the GPU. Projected coordinates (UTM, Web Mercator) therefore never
collide at any tolerance, and candidates are still compared by their true distance against
<code>vertexTolerance</code>. <code>status.overflow</code> is also set when a valid segment has a non-finite coordinate.</li>
<li><strong>Shells and holes.</strong> The signed area (shoelace, relative to the first vertex) is compared with
<code>interiorSide</code>: with <code>'left'</code> shells are counter-clockwise. <code>ringIsHole</code> flags holes. <code>ringShells</code>
assigns every hole to its innermost enclosing shell (same group), a
shell to itself, and a hole without a shell to <code>GPU_SEGMENT_RING_ASSEMBLY_NONE</code>. Of two shells that
contain the hole, the inner one is the one whose bounding box lies inside the other's, which is exact
for a planar boundary; float32 areas order only shells whose boxes are equal or do not nest (smaller
first, lowest index on ties). Until 2026-11-11 the choice was the smallest float32 area among shells
larger than the hole, which left a hole unowned, or gave it to an outer shell, when a thin shell and its
hole had areas that rounded to the wrong order. A BVH over shell boxes rejects whole
subtrees whose bounds cannot contain the hole box. Its two-word Morton key interleaves every
bit of the ordered float32 centers, avoiding global-extent quantization: a distant shell cannot
collapse nearby shells into an unresolved key tie. The index reuses the four-float ring bounds
and stores area, ring kind and group in a separate
three-word record. At 8,388,608 rings these bindings use 128 MiB and 96 MiB respectively, retaining
the default device's ring capacity and eight-storage-binding traversal limit.
Only candidate shells receive a vertex walk; equal-area ties still select the lowest ring index,
independently of BVH order. Containment is exact: the probe is the symbolic point infinitesimally
beside the midpoint of the hole's first edge, on the side away from the hole, classified with the exact
point-in-ring rule of <code>utils/exact-geometry-wgsl.ts</code>, so a hole whose edge runs along another ring is
still assigned correctly. A ring of area 0 is an empty walk and is left out of <code>polygons</code>; a face
thinner than one float32 step, whose float32 area sums to exactly 0 although it has an exact non-zero
turn at a lowest vertex it passes once, reports <code>1.17549435e-38</code> with the sign of that turn in
<code>ringAreas</code> and is kept.</li>
<li><strong>Touching vertices.</strong> When more than one segment starts at a segment's end vertex (Quadbin cells that
meet at a corner; H3 hexagons never do), the continuation is the outgoing segment that turns tightest
toward the interior, so regions that touch at a point stay separate rings. The turns
are ordered exactly: at the shared vertex when every candidate starts on it, and otherwise (a candidate
matched within <code>vertexTolerance</code> but not equal) by the exact signs of the cross products of the segment
directions <code>end - start</code>, computed from the four endpoints. Until 2026-11-11 those signs were taken
from the float32 differences, which round when the two ends of a segment differ in magnitude; with
edges about 1e-7 radians apart at a matched vertex that lost rings. Longitude/latitude input is ordered
by the same exact tests: its turns are measured with longitude differences scaled by the cosine of the
latitude, a positive scale of one axis, which changes no orientation sign and so no order. It used a
float32 turn key before, which sent near-parallel candidates to the lowest index. Only non-finite
coordinates still fall back to that key. A candidate that exactly reverses the incoming segment (and is
not removed as its paired or cancelled twin) is the tightest turn for either <code>interiorSide</code>, so a scene
and its mirror image with the interior on the other side give the same rings; until 2026-11-11 it was
the first choice for <code>'left'</code> and the last for <code>'right'</code>, and a square with a zero-width spike came back
as two rings with the interior on the left and as one ring with two open segments when mirrored. With <code>splitTouchingRings</code> a
second tracing pass swaps the continuations of segment pairs that end at one vertex and lie on one ring,
so two holes meeting at a corner, or a hole pinched to its shell, become simple rings. Choices taken
twice resolve to the lowest segment index. <code>status.touchingSegmentCount</code> and <code>segmentFlags</code>
(<code>GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING</code> 1, <code>_DANGLING</code> 2, <code>_CONFLICT</code> 4) expose the ambiguity.</li>
<li><strong>Open chains</strong> (dangling ends, tolerance misses, truncated input) emit no ring and are counted in
<code>status.openSegmentCount</code>.</li>
<li><strong>Cancelling pairs.</strong> <code>cancelOpposingSegments</code> removes exact <code>a -> b</code> and <code>b -> a</code> pairs within a group
before tracing (they are flagged <code>GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED</code>, 8, and not counted as open).
<code>GPUIsobandRings</code> uses it for the edges shared by two cells. <code>output.polygons.sourceIds</code> gives the
group of each polygon's shell.</li>
<li><strong>H3 pentagons.</strong> Rings of three mid-latitude pentagons (resolutions 2, 3 and 5, alone, in disks, as holes
and open rings) equal h3-js <code>cellsToMultiPolygon</code>. The two polar pentagons enclose a pole, so their rings
have no planar lng/lat orientation.</li>
<li><strong>Bounded output.</strong> Rings are written whole, as a prefix: <code>status.count</code> is the number written, <code>status.overflow</code> is 1
when ring or vertex capacity dropped rings, and <code>status.requiredCount</code> is the full ring count.</li>
<li><strong>Polygon layout.</strong> <code>output.polygons</code> regroups the rings as GeoArrow polygons (a shell, then its holes;
the closing vertex dropped; holes without a shell dropped) with GPU-written, flat-padded offsets over
fixed-length views. Every polygon is one feature, <code>featureOffsets</code> is an identity prefix padded with
<code>polygonCount</code>, and optional <code>sourceIds</code> carries its shell group. It plugs into
the <code>polygons</code> port of <code>GPUPointInPolygonJoin</code> (<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>): the join's
compile-time topology is the ring capacity and unused polygons and rings are empty.</li>
<li>Deterministic (integer atomics only). The cost is two pointer-jumping phases of <code>ceil(log2(rows))</code>
dispatches each (plus the second tracing pass), a sort and scans. Rings that cross the antimeridian keep
the input longitudes; use <code>GPULonLatPolygonPreparation</code> for global or antimeridian-crossing rings.
Turn comparison uses a monotone pseudo angle, not <code>atan2</code>. Ring areas are f32.</li>
</ul>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUSegmentRingAssembly */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=datum datum=0 caption="Ring 0: its signed area and whether it is a hole"}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=kernel caption="Vertex matching, pointer jumping and scans link segments into rings"}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=graph caption="Cancel, match, trace, split touching rings and write as one graph"}</p>
<h3 id="gpulinenoding-gpupolygonize-and-gpumakevalid"><code>GPULineNoding</code>, <code>GPUPolygonize</code> and <code>GPUMakeValid</code></h3>
<p><strong><code>GPULineNoding</code></strong> — Splits planar linework at crossings, touches and overlaps into atomic segments with source provenance.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-noding rung=result caption="Grey: 223 Wicker Park streets · blue: 294 segments split at every crossing"}</p>
<p><strong><code>GPUPolygonize</code></strong> — Builds every bounded polygon face from planar linework, as shapely <code>polygonize</code> does.</p>
<p>::lens{scenario=gpu-spatial-analysis.polygonize rung=result caption="Grey: 223 Wicker Park streets · blue: 73 bounded faces, the city blocks"}</p>
<p><strong><code>GPUMakeValid</code></strong> — Repairs each polygon feature into valid faces, keeping those covered by an odd number of source rings.</p>
<p>::lens{scenario=gpu-spatial-analysis.make-valid rung=result caption="Blue: 51 kept faces coloured by source park; all 51 parks were already valid"}</p>
<p>These contributors form the bounded planar geometry-topology substrate:</p>
<ul>
<li><code>GPULineNoding</code> splits arbitrary finite linework at crossings, touches, both ends of collinear
overlaps, and repeated vertices. Each atomic <code>GPUNodedSegmentPort</code> row carries source feature,
line/ring, original edge, and normalized start/end parameter provenance. Segment pairs the exact
predicates cannot certify (no split is made for them) and non-finite input vertices set <code>candidateOverflow</code>
whether or not <code>output.status.uncertainCount</code> is wired. Such input sets <code>candidateOverflow</code>;
increasing final segment capacity cannot repair it. <code>GPULineSplit</code> reports the same incomplete
split input in <code>pieces.status.overflow</code>.</li>
<li><code>GPUPolygonize</code> expands each noded segment into two directed edges, assigns deterministic left-face
ownership, omits the unbounded and zero-area faces, and emits canonical polygon features. It
emits every bounded face (shapely <code>polygonize</code> semantics): the hole of a holed ring comes back as
its own polygon. Its diagnostics classify source edges as closed-ring, open-chain, cut, dangle,
and unused.</li>
<li><code>GPUMakeValid</code> applies the same pipeline independently to every input polygon feature, then keeps a
face only when the feature's source rings cover it an odd number of times. The fill rule is source
parity (even-odd), the GEOS <code>make_valid</code> linework method that shapely uses: a valid polygon with a
hole returns the annulus only (no second polygon filling the hole), an island inside a hole is kept
as its own polygon, and the doubly covered lens of mutually crossing rings is dropped. It repairs
self-crossing and mutually crossing rings into bounded polygon faces; dimensional collapse remains
available through diagnostics rather than being converted to mixed geometry. Parity is probed
exactly at the symbolic point infinitesimally left of the midpoint of the first edge of each face
shell (the shared probe of <code>utils/exact-geometry-wgsl.ts</code>; strictly inside the face, no rounded
sample point: the ray's height is the exact midpoint of the edge's two heights, so an edge one
float32 step tall is handled), one thread per face. Stable sorts order segments by height, then
exact uint32 source group. Each face binary-searches its group's contiguous leaf range before
traversing the height BVH; node ranges outside that group are pruned before testing segments.
Group ID gaps, outliers and float32 collisions therefore do not broaden the candidate search.
Inclusive height bounds retain edges on the symbolic ray. Candidates on both sides of
the probe are retained to preserve the uncertainty fallback. A face whose
orientation cannot be certified is kept. <code>status.count</code> is the number of kept polygons. Rings close
implicitly: the edge from a ring's last vertex back to its first is always part of the boundary,
whether or not the ring repeats its first vertex. <code>GPULineNoding</code> emits no zero-length row for a
repeated vertex.</li>
</ul>
<p>All three require an explicit <code>GPUTopologyPrecisionPolicy</code>. The current policy uses the shared exact
orientation predicates with an uncertainty counter, float32 coordinate storage, and a caller-chosen
Chebyshev vertex tolerance. For <code>GPULineNoding</code>, <code>requiredCount</code> is exact when <code>candidateOverflow</code> is
zero. For <code>GPUPolygonize</code> and <code>GPUMakeValid</code>, <code>status.requiredRingCount</code> is the unclamped ring-assembly
requirement (every directed face ring, including holes and unbounded walks), for sizing ring capacity
when noding is complete. <code>status.count</code> counts written features, after parity filtering for make-valid.
There is no <code>status.requiredCount</code>: an exact feature requirement is unknown after ring loss.
<strong>Breaking (2026-11-17):</strong> rename the former ring-valued <code>status.requiredCount</code> to <code>requiredRingCount</code>.
Every supplied final status view is packed <code>uint32</code> with at least one row; final status ranges
must be disjoint, including <code>count</code> and <code>requiredRingCount</code>.
<code>overflow</code> always describes the current port's final capacity: segments for noding, directed edges
for the directed port, and rings/vertices for polygon publication. Directed requirements are twice
the noded segment requirement, so a full directed buffer can still overflow. <code>candidateOverflow</code>
reports intermediate loss or uncertainty separately. At polygon publication it includes noded
segment truncation as well as incomplete intersection/snap-rounding candidates. Check both flags
before using geometry or trusting requirements. Every supplied <code>uncertainCount</code> destination
(top-level, noded and directed) receives the same count on every submission, including zero.
Make-valid groups internally by feature row and maps retained rows to caller <code>sourceIds</code> only at
publication; all uint32 identities, including <code>0xffffffff</code>, are accepted in the active prefix.
No CPU readback or compatibility adapter is involved.</p>
<p>Maximum: <code>positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows, and the line (or ring) count plus four times <code>intersectionCapacity</code> is at most <code>getGPULineSplitMaximumPieceBound(device)</code> (4194304 on a default device, the 32-byte piece-table row). Each contributor validates both under its own ID.</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>). The resolved context is passed on to the noding and ring-assembly stages.</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPULineNoding */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-noding rung=datum datum=0 caption="Noded segment 0: the road and edge it came from"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-noding rung=kernel caption="Pair intersections, piece scans and provenance reconstruction emit atomic segments"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-noding rung=graph caption="Line splitting, segment flags, scan and writing as one graph"}</p>
<p>{/* lens:GPUPolygonize */}</p>
<p>::lens{scenario=gpu-spatial-analysis.polygonize rung=datum datum=0 caption="Noded segment 0: its closed-ring, cut or dangle class bits"}</p>
<p>::lens{scenario=gpu-spatial-analysis.polygonize rung=kernel caption="Twin edges, left-face traversal and ring writing build the faces"}</p>
<p>::lens{scenario=gpu-spatial-analysis.polygonize rung=graph caption="Noding, directed edges, ring assembly and diagnostics as one graph"}</p>
<p>{/* lens:GPUMakeValid */}</p>
<p>::lens{scenario=gpu-spatial-analysis.make-valid rung=datum datum=0 caption="Face 0: the source park it was kept for"}</p>
<p>::lens{scenario=gpu-spatial-analysis.make-valid rung=kernel caption="One thread per face probes parity against every noded segment"}</p>
<p>::lens{scenario=gpu-spatial-analysis.make-valid rung=graph caption="Noding, polygonize, parity filter and repacking as one graph"}</p>
<h3 id="gpucelltablecompare"><code>GPUCellTableCompare</code></h3>
<p>Outer-joins two sorted cell tables into one union table of before, after, delta, ratio and z-score columns.</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=result caption="35 zoom-17 union cells coloured by shops minus restaurants · grey: the places counted"}</p>
<p><code>GPUCellTableCompare</code> outer-joins two sorted cell tables (the outputs of <code>GPUCellAggregation</code> or <code>GPUCellRollup</code>) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the contributor cannot check that.</p>
<p>It compares tables keyed by cell, so it takes no <code>spatialContext</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCellTableCompare</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  before: lastMonth, after: thisMonth,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  measure: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'sum'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sumScale: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">65536</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {cells, presence, before, after, delta, ratio, percentChange, zScore, status: {count, overflow, candidateOverflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><code>output.cells</code> holds the union keys ascending (<code>uint32x2</code>, little-endian <code>(low, high)</code>); its length is the capacity. When the retained-input union is larger, the smallest keys are kept, <code>status.count</code> is clamped and <code>status.overflow</code> is 1. Required <code>status.candidateOverflow</code> receives the OR of both inputs' final and inherited loss; it is separate from destination overflow and is rewritten on every submission. Optional <code>status.requiredCount</code> receives the retained-input union requirement, a lower bound on the full union when input cells were lost. For one retained before-cell, an empty after-table and four union slots, source loss gives <code>count = requiredCount = 1</code>, <code>overflow = 0</code>, <code>candidateOverflow = 1</code>. Increase the source capacity to recover the missing cell.</li>
<li><code>presence</code>: bit 1 (<code>GPU_CELL_COMPARE_PRESENT_BEFORE</code>) when the cell is in <code>before</code>, bit 2 (<code>GPU_CELL_COMPARE_PRESENT_AFTER</code>) when in <code>after</code>.</li>
<li>The two loss flags publish separately and may share a buffer at adjacent word offsets.
Mapping both flags to the same word is rejected. <code>count</code>, <code>overflow</code> and <code>requiredCount</code>
publish together and must have nonoverlapping aligned storage binding ranges.</li>
<li><code>before</code> / <code>after</code>: the measure (<code>'count'</code>, or <code>'sum'</code> decoded from the fixed-point <code>sums</code>), 0 when absent.</li>
<li><code>delta = after - before</code>, computed with an exact sign and unsigned 64-bit magnitude from the signed integer measures, then rounded once to nearest f32 (ties to even) and divided by <code>sumScale</code> for sums. Guard and sticky bits from both integer words decide rounding. Differences outside the signed 64-bit range retain their sign, and large sums with small differences are not lost to f32 cancellation. For example, sums of <code>-2^62</code> and <code>+2^62</code> at scale 1 produce <code>+2^63</code>.</li>
<li><code>ratio = after / before</code> and <code>percentChange = 100 * (after - before) / before</code>, NaN when <code>before</code> is 0.</li>
<li><code>zScore</code>: <code>'poisson'</code> (default for counts) is <code>delta / sqrt(before + after)</code>, 0 when both are 0. It is a statistic of counts: with <code>measure: 'sum'</code> use it only for sums of non-negative values, and a row where <code>before + after</code> is negative, or is 0 while the sums differ, gets NaN. <code>'standardized'</code> (default for sums) is <code>(delta - mean) / sd</code> over the union rows with the population standard deviation, 0 when sd is 0. A published extent reduction supplies one common power-of-two normalization for all finite deltas before computing mean, squared deviations and scores. This prevents squared-delta overflow and preserves scores across sum scales. The three subsequent fixed-order stages compute partial sums, partial squared deviations and the combined statistics, so a re-encode is bitwise identical.</li>
<li>Rows past <code>status.count</code> hold the empty key (all ones), presence 0, <code>before</code>/<code>after</code> 0, and NaN in <code>delta</code>, <code>ratio</code>, <code>percentChange</code> and <code>zScore</code>.</li>
</ul>
<p>The merge is a binary-search merge path with an exclusive scan of matched rows, so every union row is written by exactly one invocation and no atomics are used. Cost is O((n + m) log(n + m)). Table contents are per-frame (re-encode without recompiling); capacities, columns, <code>measure</code> and <code>zScore</code> are topology.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUCellTableCompare */}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=datum datum=0 caption="Union cell 0: restaurants, shops, difference and z-score"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=formula caption="Delta from exact integer difference, ratio, percent change and the Poisson z-score"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=kernel caption="A merge path pairs the two tables and writes each union row once"}</p>
<p>::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=graph caption="Merge, scan, statistics reduction and column writes as one graph"}</p>`};export{e as default};