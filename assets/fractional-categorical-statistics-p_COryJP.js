var e={slug:`api/gpu-raster/fractional-categorical-statistics`,title:`GPURasterFractionalCategoricalStatistics`,html:`<h1 id="gpurasterfractionalcategoricalstatistics">GPURasterFractionalCategoricalStatistics</h1>
<p><code>GPURasterFractionalCategoricalStatistics</code> summarizes categorical raster values over fractional
zone coverage. It consumes <code>GPURasterFractionalCoverage</code> tuples, groups them by zone and category,
and returns class coverage, planar class area, composition and majority/minority categories.
It does not count intersected cells or apply observation weights.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPURasterFractionalCategoricalStatistics} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterFractionalCategoricalStatistics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  input: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'land-cover'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    storage: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'buffer'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, values: categories},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validity: categoryValidity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    noDataValue: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0xffffffff</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zoneCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  categoryCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coverage,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    coverageSums,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    classAreas,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    proportions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    coveredArea,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validArea,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    missingArea,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    majority,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    minority,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validity</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>All arguments representing data are graph views over caller-owned buffers. <code>coverage</code> contains
<code>zoneIds</code>, <code>cellIds</code>, <code>fractions</code>, <code>zoneValidity</code>, and <code>status: {count, requiredCount, overflow}</code>.
The active tuples must contain unique <code>(zone, cell)</code> pairs with fractions in <code>[0, 1]</code>. Unused
capacity after <code>status.count</code> is ignored. Duplicate pairs are not deduplicated: the producer's
unique-pair contract is required. Disjoint tile cores can be combined only after their cell IDs
are mapped into the common input grid; overlapping halo coverage must not be included twice.</p>
<h2 id="category-and-output-contract">Category and output contract</h2>
<p>Categories are uncalibrated <code>uint32</code> identifiers in <code>[0, categoryCapacity)</code>. Their numeric order
is the stable tie order. Noncontiguous external codes must be explicitly encoded into a dense
ordered dictionary before upload. Zero is a valid class. A validity mask of zero or the explicit
<code>noDataValue</code> marks missing support; a valid out-of-range code is an error. Calibration other than
<code>scale: 1</code> and <code>offset: 0</code> is rejected.</p>
<p>The three category columns have <code>zoneCount * categoryCapacity</code> rows, in zone-major order:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>coverageSums</code></td>
<td>Sum of covered cell fractions for a category; this is fractional support, not an integer count</td>
</tr>
<tr>
<td><code>classAreas</code></td>
<td>Category coverage multiplied by the absolute affine determinant</td>
</tr>
<tr>
<td><code>proportions</code></td>
<td>Category coverage divided by total valid category coverage in the zone</td>
</tr>
</tbody>
</table>
<p>The remaining columns have <code>zoneCount</code> rows:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>coveredArea</code></td>
<td>All represented coverage, including nodata cells</td>
</tr>
<tr>
<td><code>validArea</code></td>
<td>Represented area with valid category values</td>
</tr>
<tr>
<td><code>missingArea</code></td>
<td>Represented area whose categories are masked or nodata</td>
</tr>
<tr>
<td><code>majority</code></td>
<td>Present category with greatest fractional support; greatest category wins exact ties</td>
</tr>
<tr>
<td><code>minority</code></td>
<td>Present category with least fractional support; lowest category wins exact ties</td>
</tr>
<tr>
<td><code>validity</code></td>
<td>One when coverage is complete and valid category support is positive</td>
</tr>
</tbody>
</table>
<p>For complete zones, class areas sum to valid area, and valid plus missing area equals covered
area, within float32 rounding. Proportions sum to one when valid support is positive. Absent
categories have zero coverage and area and never win minority. Complete empty or fully missing
zones have zero class areas, NaN proportions, <code>0xffffffff</code> winners and zero validity. Their area
outputs still distinguish an empty zone from a zone containing only missing categories.</p>
<p>Overflow, inconsistent active/required counts, malformed active tuples, and covered unmasked
out-of-range category IDs invalidate every zone. Failed polygon triangulation invalidates its
zone. Float32 overflow of a zone's covered area also invalidates that zone. Invalid zones return
NaN in every floating output, <code>0xffffffff</code> winners and zero validity. This separates incomplete
or numerically unrepresentable results from explicit missing category area. Input and output
buffers must be distinct; input updates are read on every execution of the same compiled graph.</p>
<h2 id="bounds-and-units">Bounds and units</h2>
<ul>
<li><code>categoryCapacity</code>: 1 through 65534.</li>
<li><code>zoneCount</code>: positive, with <code>zoneCount * (categoryCapacity + 1) &#x3C;= 65535</code>.</li>
<li>Coverage tuple capacity: 1 through 1,048,576, including unused capacity.</li>
<li>Cell area must be positive and representable in float32.</li>
</ul>
<p>The extra group per zone stores missing support. Stable sorted segmented sums avoid a dense
zone/category/tuple scan. These are bounded experimental workloads, not measured throughput
claims. Category, coverage and area calculations use float32 support sums; tie decisions apply
to these sums. No tolerance merges nearly equal categories.</p>
<p>Areas are squared coordinate units from the affine determinant. By default that is the
determinant of <code>grid.affine</code>, fixed at compile time. Pass the optional <code>parameters</code> prop (the
per-frame georeference view given to <code>GPURasterFractionalCoverage</code>) to read the determinant on
every encoding instead, so <code>classAreas</code>, <code>coveredArea</code>, <code>validArea</code> and <code>missingArea</code> follow a
georeference repacked for another cell size. By default geographic degree grids do not
produce square meters; set <code>cellSizeMode: 'geographic'</code> (degrees) or <code>'web-mercator'</code> (EPSG:3857
meters) for WGS84 ground square meters. Each coverage fraction is then weighted by the ground area
per world area of its row (row edges from <code>grid.affine</code>, evaluated at the row center, so
<code>coverageSums</code> holds weighted coverage and <code>proportions</code> are area proportions). The row edges are
read from <code>parameters</code> (origin y and row step y, slots 1 and 5) on every encoding when it is given,
so the ground scale follows a repacked georeference together with the cell area; without
<code>parameters</code> they are baked from <code>grid.affine</code>. The grid must be axis-aligned. Nonzero packed
cross-axis terms suppress every dependent result for that frame (NaN areas/proportions and
validity zero), and an aligned later submission restores validity. The default <code>'uniform'</code> keeps planar coordinate units. Missing area describes nodata inside the supplied raster footprint; it
does not estimate a polygon's area outside that footprint. This API does not add observation
weights, spherical areas, category dictionary discovery, or implicit tile stitching.</p>
<h2 id="reference-evidence">Reference evidence</h2>
<p>The checked-in generator <code>modules/analysis/test/oracles/rs02-fractional-categories.py</code> pins
exactextract 0.3.0, NumPy 2.2.6 and Shapely 2.1.2. It calls exactextract <code>frac</code>, <code>majority</code>,
<code>minority</code>, <code>count</code>, and Cartesian area-weighted count directly. A separate all-valid raster
retains missing-cell coverage. Shapely constructs partial-cell multipart and hole geometries;
it does not compute the expected statistics. Fixtures retain versions and generator hash.</p>
<p>The majority/minority tie rules match the <a href="https://github.com/isciences/exactextract/blob/v0.3.0/src/raster_stats.h">pinned exactextract implementation</a>.
Coverage and observation-weight meanings follow the <a href="https://isciences.github.io/exactextract/operations.html">official operations reference</a>.
The tests compare every category, retain empty classes, verify area conservation, compose actual
GPU polygon coverage with the reducer, and update categories, masks and failure status on one
compiled graph. Tests are executable evidence only after the corresponding GPU run succeeds.</p>
<h2 id="executable-examples">Executable examples</h2>
<p>These examples use explicit synthetic inputs and expose real GPU graph captures.</p>
<h3 id="gpurasterfractionalcategoricalstatistics-1">GPURasterFractionalCategoricalStatistics</h3>
<p>{/* lens:GPURasterFractionalCategoricalStatistics */}</p>
<p>::lens{scenario=gpu-raster.raster-fractional-categorical-statistics rung=graph}</p>
<p>The polygon-composition test compares floating outputs and input coverage tuples with an absolute
plus relative bound of <code>4 * 2^-23 + 4 * 2^-23 * abs(reference)</code>. Triangle clipping can introduce
nonbinary intersection ratios even for these dyadic polygon coordinates. The observed hole-zone
class area was <code>11.999998092651367</code> versus <code>12</code> (two float32 ULP). Direct tuple-fed comparisons
retain an absolute <code>5e-7</code> bound; category winners, tuple identities and validity flags remain exact.</p>
<p>Discrete majority/minority ties apply to <strong>represented float32 coverage sums</strong>, not ideal polygon
areas. Clipping roundoff can split an ideal geometric tie: in the updated hole-zone fixture,
exactextract chooses minority category 0 while the GPU composition chose category 3. The composed
test therefore validates each captured GPU coverage tuple against exactextract first, then computes
expected winners independently from those captured input weights and asserts exact category
equality. Each fixture zone/class has at most two contributing cells, allowing an independent
double sum followed by one float32 rounding without copying the GPU reduction tree. The test logs
the captured fractions and class supports when these winners differ from ideal geometry. This is
composition validation under represented weights; direct exactextract tie parity remains covered
by the separate reference-tuple test. No near-tie epsilon changes the API's tie rule.</p>`};export{e as default};