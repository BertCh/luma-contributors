var e={slug:`api/gpu-spatial-analysis/operators-joins`,title:`GPU Spatial Analysis: Spatial joins`,html:`<h1 id="spatial-joins">Spatial joins</h1>
<h3 id="gpuspatialpredicatejoin"><code>GPUSpatialPredicateJoin</code></h3>
<p>candidateTileCapacity bounds paged candidate scratch independently of the unchanged global
candidateCapacity. Final pairs and weights remain globally capacity bounded; inspect candidate and
output overflow separately.</p>
<p>Finds feature pairs that satisfy a spatial predicate such as intersects, contains or within, and counts matches per feature.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=result caption="102 parks against 199 tracts; tracts shaded by the number of parks they intersect."}</p>
<p>Joins two sets of features by a spatial predicate, like GeoPandas <code>sjoin</code> or PostGIS
<code>ST_Intersects</code>, <code>ST_Contains</code>, <code>ST_Within</code> and <code>ST_DWithin</code>. Coordinates are planar by default; a
longitude/latitude <code>spatialContext</code> selects great-circle <code>dwithin</code> in meters (see
<a href="#spatial-context-of-gpuspatialpredicatejoin">Spatial context</a>). Either side is
<code>{kind: 'points', positions}</code>, <code>{kind: 'lines', positions, lineOffsets}</code> (one linestring per
feature) or <code>{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}</code>
(multipolygons, holes as later rings, GeoArrow layout); all nine combinations are supported.</p>
<p>Multipart points and lines use <code>GPUMultipartPointGeometryPort</code> and
<code>GPUMultipartLineGeometryPort</code> from <code>@luma-contributors/analysis/analysis-core</code>.
Their <code>featureOffsets</code> map feature rows to point or linestring parts; a line part's
<code>lineOffsets</code> then map to vertices. Offsets start at zero, are nondecreasing and terminate at
the child count. <code>sourceIds</code> has one row per feature. Repeated IDs do not merge features.
Measures, affine transform, prepared join indices and the following joins share that hierarchy.</p>
<p>For <code>GPUSpatialPredicateJoin</code>, multipart operands support <code>intersects</code>, <code>within</code> and <code>coveredBy</code>
with <code>engine: 'auto'</code> or <code>'fast'</code>, without a <code>relate</code> output. Both sides may be multipart.
<code>leftRows</code> and <code>rightRows</code> always identify FEATURE rows, with one pair per matching feature pair:</p>
<ul>
<li><code>intersects</code> accepts any matching part pair, including a match in a later part.</li>
<li><code>within</code> requires all usable input parts to lie in the target's closure and at least one
interior intersection. A feature with all parts on the target boundary is not within it.</li>
<li><code>coveredBy</code> requires all usable parts to lie in the target's closure, allowing boundary-only
contact. Containment is against the complete target feature, including the union of line parts.</li>
</ul>
<p>Empty features never match. Empty line parts contribute no edges; short line parts with fewer
than two vertices are empty. Multipart line boundary points follow the mod-2 endpoint rule:
an endpoint shared by two open line parts belongs to the feature interior.
Line coverage merges intervals on a supplied coordinate axis, preserving distinct endpoints
even when projecting them onto a long query would round them to the same fraction.
The BVH indexes feature bounds spanning all parts, so separate parts do not occupy separate leaf
slots. Bounds may overlap across a gap, but exact refinement never draws a line across that gap.
<code>candidateCapacity</code> budgets logical feature pairs, including bounding-box false positives.
If a feature loses a candidate reached through any of its parts, its <code>candidateTruncated</code> row
and <code>status.candidateOverflow</code> are set. A short BVH <code>leafCapacity</code> also sets candidate loss.
Final pair capacity loss stays in <code>overflow</code>; increasing final capacity cannot recover candidates.</p>
<p><code>GPUNearestFeatureJoin</code> supports multipart queries and targets in neighbors mode (<code>neighborIds</code>
and <code>neighborCounts</code>), with minimum distance over every usable part pair. Query and target rows
are feature rows; <code>neighborRows</code> retains the target row separately from <code>neighborIds</code>.
Containment of any later part in a polygon gives distance zero. The existing <code>k</code>, tie, radius,
attribute and exclusion rules apply to feature rows. Each topology and output binding must fit
the device's storage-binding limit; sorted multipart line queries pack their topology into one
binding to stay within the eight-buffer traversal budget.
Ranking and pruning use the same normalized squared-distance significand and biased exponent
on CPU and GPU, including squares outside float32 range. Normal small distances therefore remain
distinct from an exact hit without reducing ordinary squared-distance precision. Subnormal
coordinates or distances are outside the GPU precision contract and may flush or clamp. At the
largest finite distances, the final two squared-rank bins coalesce with their predecessor and
publish the maximum float32 distance; row order breaks those ties. For normal distances or exact hits, a radius of either positive or negative zero
admits exact hits only, including great-circle searches. Endpoint distances use the endpoint
directly, and cancellation-prone segment determinants retain integer product residuals before
distance conversion.
Targets whose published distances round alike may still have different ranks. Finite linear
distances are supported when their squares exceed the float32 range. Combined sorted-query
topology binding sizes are checked before graph resources are declared when query sorting actually
packs metadata (at least two query features), permitting an unsorted retry after a metadata-size
rejection. Prepared consumers require identical position and
topology views, including the presence or absence of feature grouping.</p>
<p>Other multipart predicates, DE-9IM output/<code>engine: 'relate'</code>, and nearest-feature mode remain
rejected with explicit messages. <code>GPUPointInPolygonJoin</code> and <code>GPUBufferSelection</code> still reject
multipart point query ports; ordinary point ports have one query per position row.</p>
<p><code>GPUPointInPolygonJoin</code> and <code>GPUBufferSelection</code> infer the point input when a helper returns
the broader <code>GPUSpatialJoinGeometry</code> union and the caller narrows it with <code>kind === 'points'</code>.
This preserves valid single-point callers without a cast. A known multipart point port remains
a type error; if grouping arrives through a broader union, construction rejects <code>featureOffsets</code>
before flattening or declaring graph resources. Their props types keep single-point defaults
and expose an optional point-geometry type parameter for explicitly typed union inputs.</p>
<p>Pure <code>computeSpatialPredicateJoinOnCPU</code> and <code>computeNearestFeatureJoinOnCPU</code> take packed typed
arrays with the same hierarchy. The CPU predicate supports the three predicates above, excluding
polygon/polygon containment, and takes explicit <code>candidateCapacity</code> and <code>pairCapacity</code> limits.
CPU predicate candidates use a sorted bounds index and obey the explicit candidate budget;
<code>requiredCount</code> is a lower bound after candidate loss. CPU nearest selects one unbounded planar
neighbor per query feature, breaks distance ties by lowest target row, uses a sorted bounds index
with range pruning and emits float32 distances. Empty queries produce empty per-query columns;
queries with no targets produce count 0, row/ID <code>0xffffffff</code> and distance -1.
The CPU predicate result's <code>candidateBoundsTestCount</code> counts index bounds tested, separately
from result status; entire ranges ending before or starting after a query are pruned together.</p>
<p><code>predicate</code> is <code>'intersects'</code>, <code>'contains'</code>, <code>'within'</code>, <code>'dwithin'</code> (with an inclusive
<code>distance</code>: planar in coordinate units, or meters with a great-circle <code>spatialContext</code>), <code>'covers'</code>, <code>'coveredBy'</code>, <code>'touches'</code>, <code>'crosses'</code>, <code>'overlaps'</code>, <code>'equals'</code>,
<code>'containsProperly'</code> or <code>'relate'</code> (with a DE-9IM <code>pattern</code>). Boundaries follow OGC / DE-9IM: a point has no boundary, a linestring's boundary is its
endpoints unless it is closed, and a polygon's boundary is its rings.</p>
<ul>
<li><code>intersects</code>: the closed geometries share a point, so polygons touching at an edge or corner
intersect, and so does a point on a polygon boundary.</li>
<li><code>contains(a, b)</code>: <code>b</code> lies in the closure of <code>a</code> and the interiors intersect. A polygon does not
contain a point or line on its boundary, and a line does not contain its own endpoints.</li>
<li><code>within(a, b)</code> is <code>contains(b, a)</code>.</li>
<li><code>dwithin</code>: the minimum distance between the closed geometries is at most <code>distance</code>.</li>
<li><code>covers</code> and <code>coveredBy</code>: <code>contains</code> and <code>within</code> that allow boundary contact.</li>
<li><code>touches</code>: the boundaries meet and the interiors do not (never true for two points).</li>
<li><code>crosses</code>: the interiors meet and the intersection has a lower dimension than the larger input, by the
OGC dimension rules (false for polygon/polygon and point/point).</li>
<li><code>overlaps</code>: the same dimension, the interiors meet and neither contains the other.</li>
<li><code>equals</code>: the same point set. <code>containsProperly</code>: <code>b</code> lies in the interior of <code>a</code>.</li>
<li><code>relate</code> with <code>pattern</code>: any DE-9IM pattern (characters <code>T F * 0 1 2</code>), as a string or an any-of list.</li>
</ul>
<p>The pipeline is upstream primitives end to end: per-feature bounds into a <code>GPUBVH</code> over the right
side, a count pass, <code>GPUScan</code> and an ordered write of candidate pairs, the exact predicate per
candidate (single-point/polygon pairs use <code>GPUPairwisePointInPolygon</code>; points on an edge it reports as
uncertain are decided by the f32 kernel and counted in <code>uncertainCount</code>), then a scan and a
capacity-bounded scatter. Outputs go in <code>output</code>; <code>output.pairs</code> (a <code>GPUCompactPairPort</code>: <code>leftRows</code>,
<code>rightRows</code>, <code>status</code> with <code>count</code>, <code>overflow</code>, optional <code>requiredCount</code>) are unique and sorted by <code>(left, right)</code>, and
scalars go in <code>output.status</code> (<code>overflow</code>, <code>candidateCount</code>, <code>uncertainCount</code>). The sides are geometry
ports (<code>GPUPackedGeometryPort</code> of the point, line or four-level polygon port). On overflow the output is a
sorted subset; <code>join.getCapacityPlan({candidateCount, requiredCount})</code> turns the read-back counts into the
next capacities (see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#shared-ports-status-and-planning">Shared ports, status and planning</a>).
<code>candidateCapacity</code> defaults to that plan's estimate, <code>min(rightCount, 8)</code> per left feature.</p>
<p>Pass <code>output.weights</code> to also write the matches as a <code>GPUSpatialWeights</code> with left features as rows and
weight 1; set <code>excludeSameRow</code> for a self join. A predicate join is then a weights producer, for
example <code>dwithin</code> between polygons. Optional <code>output.weights.distances</code> contains the minimum
closed-geometry distance aligned with <code>neighbors</code>: planar coordinate units, or meters for
great-circle <code>dwithin</code>. Intersecting or contained geometries have distance zero; holes contribute
their boundaries. Planar distance output uses scaled norms, so representable distances such as
<code>1e20</code> do not saturate when their square exceeds float32. Normal gaps beside long, asymmetric
segments remain measurable even when a projection fraction would underflow. Large <code>dwithin</code>
thresholds use the same linear distance in both engines; ordinary thresholds retain squared
comparisons. Unused distance slots are NaN on every encoding, including empty results.
Distance output requires a metric (<code>native</code> or <code>great-circle</code>); longitude/latitude topology with
<code>metric: 'none'</code> cannot produce distances. The optional <code>weights.status</code> fields <code>count</code>,
<code>requiredCount</code>, <code>overflow</code>, <code>candidateOverflow</code> and <code>uncertainCount</code> are written even when no
pair output is requested. <code>overflow</code> reports final pair capacity, and <code>candidateOverflow</code> reports
an incomplete candidate or BVH stage. Distance evaluation adds one packed pair scratch buffer
(eight bytes per output slot), included by the instance's <code>getCapacityPlan()</code>.</p>
<p>Declared status destinations, including inherited and non-enumerable <code>requiredCount</code> getters,
are captured at construction and retained when implicit completeness destinations are added.
The pair-gather example preserves the caller's pair requirement destination in the same way.
Geometry descriptors, options and the nested pair, unmatched and weights maps are captured too.
Replacing <code>unmatched.ids</code> after construction cannot bypass the original output-alias checks.</p>
<p>For downstream statistics, pass <code>join.outputWeights</code>, which always includes both overflow and
candidateOverflow. Missing caller scalar views are allocated in the graph and written on every
submission. The caller's <code>output.weights</code> is an allocation description; rebuilding an input from
only that description can omit completeness signals.</p>
<p>The new predicates are masks over one DE-9IM matrix per candidate pair, computed from structure (vertex
locations, edge crossings, collinear overlaps, and edge pieces located against the other geometry, with
polygon areas decided from edge pieces) for all nine point, line and polygon combinations. Semantics
follow GEOS and Shapely. <code>intersects</code>, <code>contains</code>, <code>within</code> and <code>dwithin</code> keep their short-circuiting
kernels and use the matrix only when <code>relate</code> output is requested; the specs assert both paths agree.
Only bounding-box candidates are enumerated, so a pattern that admits disjoint geometries (all of II, IB,
BI and BB allow <code>F</code>) is rejected at construction; use <code>how: 'anti'</code> with <code>intersects</code> for disjoint.
Lines are single linestrings with the OGC endpoint boundary (multilinestrings and mod-2 boundaries are not
modeled). The matrix kernel is <code>O(n * m)</code> per pair, which suits small features and is not tuned for large
polygons.</p>
<p>Orientation tests are exact. Edge pieces between crossings are located from an exactly located endpoint
and the exact crossings between it and the piece, not from their rounded midpoint, so edges within a few
ulps of each other (near-parallel or nearly shared) classify like Shapely. Distances and the parameters
that order crossings along an edge run in f32, so recenter or project coordinates before measuring
near-degenerate geometry. Polygons must be valid.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialPredicateJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  left: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'points'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: points},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  right: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, featureOffsets, polygonOffsets, ringOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  predicate: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'within'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pairs: {leftRows, rightRows, status: {count, overflow}}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><strong><code>relate</code> output.</strong> <code>output.relate</code> is an optional <code>GraphDataView&#x3C;'uint32'></code> as long as <code>pairs.leftRows</code>; slot <code>k</code>
holds the matrix of <code>(leftRows[k], rightRows[k])</code>. Each of the nine cells (order II IB IE BI BB BE EI EB EE,
row-major, left parts then right parts) takes 2 bits at <code>[2k, 2k + 1]</code> with value <code>dimension + 1</code> (0
empty, 1 point, 2 line, 3 area). <code>formatGPUSpatialRelate(value)</code> decodes it on the CPU to a string such as
<code>'FF2F11212'</code>. Bit 31 (<code>GPU_SPATIAL_RELATE_UNCERTAIN_BIT</code>) of a slot is set when an orientation of that pair
had no sign (see <strong>Exactness</strong>); <code>formatGPUSpatialRelate</code> ignores it, and <code>value &#x26; 0x3ffff</code> is the matrix alone.
It is not available for <code>dwithin</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialPredicateJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  left, right, predicate: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'touches'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pairs: {leftRows, rightRows, status: {count, overflow}}, relate, status: {uncertainCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><strong>Anti joins.</strong> <code>how: 'anti'</code> with <code>output.unmatched: GPUCompactOutput</code> emits the left rows that have no match, in
ascending order (<code>pairs</code>, <code>weights</code> and <code>relate</code> are not allowed with it). When the candidate capacity overflows,
the left rows whose candidates were dropped cannot be classified (they may have a match among the dropped
candidates), so they are kept out of <code>output.unmatched</code> and counted in <code>output.status.candidateTruncatedCount</code>;
the anti output then never lists a row that may have a match, and may omit true unmatched rows among the
truncated ones. (Before this change those rows were emitted as unmatched.) A BVH overflow (more right features
than <code>leafCapacity</code>) drops right features that any row may match, so since 2026-11-10 it marks every left row and
the anti output is empty; before, it listed rows whose only matches were among the dropped features.</p>
<p><strong>Candidate overflow.</strong> Candidates are written in left-row order, so the slots past <code>candidateCapacity</code> are the
tail of that layout: the row whose candidates cross the capacity keeps its first ones, and every later row that
had a candidate keeps none; earlier rows are complete. A BVH leaf overflow (more right features than
<code>leafCapacity</code>, in the join's own build or in a <code>GPUSpatialJoinPrepared</code>) drops features that any row may match,
so it marks every row. <code>output.status.candidateOverflow</code> is 1 when either happened (<code>output.status.overflow</code> keeps
its combined meaning), and the optional <code>output.candidateTruncated</code> (one <code>uint32</code> per left feature) is 1 for exactly
the rows described above. <code>output.status.candidateTruncatedCount</code> counts them. <code>GPUPointInPolygonJoin</code> and every
mode of <code>GPUNearestFeatureJoin</code> (nearest-feature, planar or great-circle, and neighbors) take the same
<code>output.candidateTruncated</code> and <code>output.status.candidateOverflow</code>, one flag per query; a flagged query may show no
feature, or a farther one. Neighbors mode and the great-circle search have no candidate table, so a BVH overflow is
their only cause. <code>GPUSpatialJoinPrepared</code> holds only the right-hand BVH and has no candidate stage. The join
consumers pass the flags through: <code>GPUZonalStatistics</code> (<code>output.candidateTruncated</code> per point and
<code>output.status.candidateOverflow</code>), <code>GPUBufferSelection</code> (the same pair) and <code>GPULineLengthPerPolygon</code>
(<code>output.status.candidateOverflow</code> only; its output is per polygon).</p>
<p><strong>Exactness.</strong> Every orientation sign is exact, in the short-circuiting kernels (<code>engine: 'fast'</code>) and in
the relate engine alike: an f32 filter decides the clear cases and the rest fall back to exact integer
arithmetic (256-bit), with one inlined copy per kernel. So touches, crosses, covers, <code>intersects</code>,
<code>contains</code> and the other predicates do not depend on f32 rounding of the orientation test, and the result
does not depend on the engine or on the feature size that <code>engine: 'auto'</code> looks at, including
shared-edge and collinear configurations. Polygon <code>contains</code> / <code>within</code> in the fast kernel decides on
which side of a shared boundary run the two interiors lie with a symbolic probe (a point just beyond an
outer vertex and just off its edge, never formed in f32), so an inner polygon that fills a hole of the
outer one, or has a hole of the outer one inscribed in it, is not contained for any vertex coordinates.
The exactness is with respect to the f32 vertices; edge piece
midpoints, break parameters and distances are computed in f32: the location of a piece shorter than about
one ulp is not exact, so a line starting one ulp outside a polygon edge can have its <code>crosses</code> piece
classified as boundary.</p>
<p><strong>Uncertain pairs.</strong> A pair is uncertain when an orientation has no sign: a non-finite coordinate, or
product exponents spanning more than 200 bits. It is never dropped silently. The undecidable orientation is
treated as collinear (sign 0) and the predicate is evaluated with that value, so the pair can come out
wrong in either direction, and it is counted in <code>uncertainCount</code> (zero means every decision is certified).
Matched uncertain pairs are identifiable per pair through bit 31 of the <code>relate</code> output. Point/polygon
pairs use the robust double-single classifier (they do not get the exact re-decision of
<code>GPUPointInPolygonJoin</code>); pairs it cannot certify are decided by the f32 kernel and counted.
Scalar exact kernels with eight bindings carry uncertainty in candidate flags and count it in a
later pass, so adding an identity multipart grouping preserves the diagnostic. The workgroup
<code>dwithin</code> kernel still does not track uncertainty (reported as 0).</p>
<p><strong>Per-frame distance and pattern.</strong> <code>distance</code> is a <code>number</code> or a <code>GraphDataView&#x3C;'float32'></code>, and <code>pattern</code>
is a <code>GPUSpatialRelatePattern</code> or a <code>GraphDataView&#x3C;'uint32'></code>. A view is read from a parameter buffer, so
it changes without recompiling and gives bit-identical results to the compile-time value (a negative, NaN
or infinite distance selects nothing). Fill a pattern view with <code>packGPUSpatialRelatePattern(pattern, slotCount)</code>, which writes <code>GPU_SPATIAL_RELATE_PATTERN_WORDS</code> (2) words per slot; unused slots are zero and
match nothing, and the helper rejects patterns that admit disjoint geometries. Size <code>candidateCapacity</code>
for the largest distance. The exact kernels dispatch by the live candidate count, but the cooperative
workgroup kernels (<code>dwithin</code> with <code>engine: 'relate'</code> and the relate engine, 64 and 256 lanes per
candidate) index their lanes with a u32, so the constructor rejects a <code>candidateCapacity</code> above
<code>floor((2^32 - 1) / lanes)</code> for them, naming the prop. <code>predicate</code> stays compile-time because it selects the kernel family; for a
per-frame predicate choice use <code>predicate: 'relate'</code> with patterns.</p>
<p><strong>Relate performance.</strong> Per-edge break parameters are collected in one pass (up to 24 per edge) and the
covers scan runs only when a collinear edge exists. The workgroup relate kernel also builds a per-pair
y-slab index of each polygon side in workgroup memory (32 slabs, 1280 entries per side), so point location
scans the closing edges and the query's slab instead of every edge; features under 48 edges, with a
degenerate y extent or over 1280 entries use the plain scan. Results are identical. On 100 polygon pairs
the batched breaks and the slab index took relate from about 7.5 ms to 4.3 to 5 ms, against about 1 ms for
GEOS: the break collection, covers scan and vertex loops are still <code>O(n * m)</code>, and about 2 ms of the
remaining time is fixed graph and readback cost.</p>
<p><strong><code>onAttribute: {left, right}</code></strong> (GeoPandas <code>sjoin(on_attribute=)</code>). Two <code>uint32</code> key views, one per left
row and one per right row. A pair survives only when the keys are equal. The filter empties candidate
slots before the exact kernels, so <code>candidateCount</code> stays the bounding-box count before filtering, and
<code>how: 'anti'</code> then reports left rows with no key-equal match. Its presence is compile-time and its
contents are per-frame. Defaults are unchanged.</p>
<h4 id="engine-selection">Engine selection</h4>
<p><code>engine?: 'auto' | 'fast' | 'relate'</code> is compile-time and defaults to <code>'auto'</code>. It selects the kernel for
<code>intersects</code>, <code>contains</code>, <code>within</code> and <code>dwithin</code>; every other predicate always uses the relate engine.</p>
<ul>
<li><code>'fast'</code> runs a short-circuiting kernel with one invocation per candidate. It is quickest for small features.</li>
<li><code>'relate'</code> runs the DE-9IM engine with one workgroup per candidate and bounding-box pruning. For <code>dwithin</code> it
runs a workgroup distance kernel instead.</li>
<li><code>'auto'</code> picks <code>'relate'</code> when neither side is points and the product of the two sides' average vertices per
feature is at least 1024.</li>
</ul>
<p>Requesting <code>relate</code> output always uses the relate engine. Every engine uses the exact orientation sign, so
shared boundaries give the same answers on all three. Measured cost is about 10 ms for 100 pairs of 600 x 175
vertices. A pair with hundreds of shared edges costs 5–10 ms. A great-circle <code>dwithin</code> always uses the
short-circuiting kernel (<code>'relate'</code> throws, <code>'auto'</code> picks <code>'fast'</code>).</p>
<h4 id="spatial-context-of-gpuspatialpredicatejoin">Spatial context of <code>GPUSpatialPredicateJoin</code></h4>
<p><code>spatialContext</code> (omitted: native planar) selects how positions are read. It is validated at construction and
exposed as the readonly field <code>spatialContext</code>; unsupported combinations throw.</p>
<table>
<thead>
<tr>
<th>Context</th>
<th>Predicates</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>planar (default)</td>
<td>all</td>
<td>Euclidean plane; <code>dwithin</code> <code>distance</code> in coordinate units.</td>
</tr>
<tr>
<td><code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code></td>
<td><code>dwithin</code> only</td>
<td>positions are longitude/latitude degrees; <code>distance</code> is meters on the mean Earth sphere (<code>sphereRadius</code> overrides the radius; <code>units: 'kilometers'</code> throws).</td>
</tr>
<tr>
<td><code>{coordinateSpace: 'longitude-latitude', metric: 'none'}</code></td>
<td>every predicate except <code>dwithin</code></td>
<td>topology in the degree plane, as GEOS and PostGIS <code>geometry</code> do for coordinates in degrees; no distances.</td>
</tr>
</tbody>
</table>
<p>Great-circle <code>dwithin</code>: the candidate stage widens each left box by the angular radius in latitude and by the
longitude reach of that radius at the box's far latitude (<code>2 asin(sin(r/2) / cos(lat))</code>, every longitude when
the radius reaches a pole), and a box that crosses the antimeridian also probes its copy shifted by 360
degrees, with a 0.1% plus 1e-5 degree margin so rounding never drops a candidate. The exact kernel measures
the central angle from a vertex to an edge with <code>geodesicSegmentFraction</code> and <code>geodesicSegmentDistance</code>
(<code>GEODESIC_WGSL</code>): edges are straight in longitude/latitude with the longitude difference wrapped at +-180,
the fraction of the nearest point is found in the local equirectangular frame of the point, and the
distance is the haversine of coordinate differences, which keeps meter-scale offsets well conditioned in f32.
The same edge model is used by the intersection and containment tests (exact orientation in degrees, as the
planar kernels) and by the network snapping contributors, so there is one convention rather than two. The
foot is exact to first order for edges of road length, so densify long edges. Distances are accurate to f32
rounding across the antimeridian and over a pole, up to about 8000 km (a larger compile-time distance throws,
a larger per-frame value selects nothing). Pairs whose geometries intersect in the degree plane count as
distance 0. Prepare polygons that cross ±180 or enclose a pole with <code>GPULonLatPolygonPreparation</code>.
Longitude/latitude contexts require <code>edgeModel: 'linear'</code> (the default); <code>'great-circle'</code> edges throw
with a preparation hint. Optional <code>output.status.antimeridianUncertainCount</code> and
<code>output.status.poleUncertainCount</code> report unprepared rings, accumulated over both polygon inputs.
They are cleared each encoding in longitude/latitude contexts; planar contexts leave these fields
untouched and build no ring check. A numeric great-circle <code>dwithin</code> distance below ten float32
longitude/latitude quanta (about 17 m on the default sphere) warns to use <code>createGPULocalFrame</code>.
Oracle: numpy haversine and
<code>pyproj.Geod(a=R, f=0)</code> for point pairs, shapely distances in the azimuthal equidistant projection for
polygons (see the spec fixtures).</p>
<p>The topological predicates take <code>metric: 'none'</code>; <code>metric: 'great-circle'</code> with them throws, and <code>dwithin</code>
with <code>metric: 'none'</code> throws, so the intent is explicit. <code>GPUSpatialJoinCandidates</code> is planar only;
<code>GPUPointInPolygonJoin</code> also accepts longitude/latitude with <code>metric: 'none'</code>. <code>GPUNearestFeatureJoin</code> and
<code>GPUBufferSelection</code> also take <code>metric: 'great-circle'</code> (see their section); <code>GPUSpatialJoinPrepared</code> and <code>GPUNearestFeatureWeights</code> take no positions with a metric
(a prepared tree is a bounding-box tree and may serve a great-circle join) and have no context.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUSpatialPredicateJoin */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=datum datum=46 caption="Tract 17031031400 (row 46): its candidate parks and how many intersect it."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=formula caption="Intersects means some interior or boundary cell of the DE-9IM matrix is non-empty."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=kernel caption="The mask kernel reads one DE-9IM matrix per candidate pair and keeps intersecting pairs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=graph caption="Candidate pairs from bounds, an exact relate per pair, a match mask, then scan and scatter."}</p>
<h3 id="gpuspatialjoinprepared"><code>GPUSpatialJoinPrepared</code></h3>
<p>Retains a static geometry snapshot for one side of a join.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=result caption="One retained tract snapshot serves both the candidate and the exact park join."}</p>
<p>A static right-hand side for repeated joins. Geometry refresh is gated on the GPU by requested
and applied generations. The applied generation changes only after the complete build executes,
so discarded encodings and abandoned execution prefixes retry automatically.</p>
<p>The published bounds/BVH/sort stages replay against the retained geometry every encoding. This
conservative implementation repeats build work because published nested nodes lack the dispatch
metadata needed for whole-build gating. Resident overhead is one copy of every geometry view plus
24 generation-control bytes; the snapshot uses the same view lengths and binding limits.</p>
<p>The first encoding and first encoding after explicit invalidation bypass <code>rebuildWhen</code>; other
encodings call it once. Public join props preserve the handle's frame type: use
<code>GPUSpatialJoinCandidatesProps&#x3C;Frame></code> or <code>GPUSpatialPredicateJoinProps&#x3C;Frame></code> when a helper
registers <code>props.prepared</code> on a <code>GPUCommandGraph&#x3C;Frame></code>. Point-in-polygon and nearest-feature
props take the frame type after their existing row/geometry type parameters.</p>
<p>The props and handle accept a frame type parameter, for example
<code>GPUSpatialJoinPrepared&#x3C;{geometryChanged: boolean}></code>; use the same type on <code>GPUCommandGraph</code>.
The default is <code>unknown</code>, suitable for <code>invalidate()</code> without a frame predicate. See the
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts">execution guide</a> for a typed predicate example. A recorded refresh request remains
pending until its complete preparation executes. <code>encodedBuildCount</code> counts CPU refresh-request
encodings, including discarded requests, and excludes automatic GPU retries and replayed builds.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> prepared</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialJoinPrepared</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({geometry: polygons});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(prepared); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// before the joins that use it</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialPredicateJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  left: movingPoints, right: polygons, prepared, predicate: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'within'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pairs: {leftRows, rightRows, status: {count, overflow}}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointInPolygonJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  points: movingPoints, polygons, prepared, candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {pointFeatureIds, status: {overflow: pointOverflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">prepared.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">invalidate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// request a fresh geometry snapshot; discarded encodings retry it</span></span></code></pre>
<p>It reuses geometry: every join still evaluates all candidates against the current left and
right geometry each encoding. A right side that changed without <code>invalidate()</code> keeps the old tree, so
candidates come from the old boxes and a feature that moved outside its old box is missed. Storage is
persistent buffers, allocated by the handle and freed by <code>destroy()</code>, or passed as <code>storage</code>. One handle
belongs to one graph, and several joins may share it. A sorted prepared handle is accepted by
<code>GPUPointInPolygonJoin</code>, <code>GPUSpatialPredicateJoin</code>, <code>GPUSpatialJoinCandidates</code> and
<code>GPUNearestFeatureJoin</code> (points, lines and polygons, not segments). Sorting preserves logical
feature rows, including multipart grouping.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUSpatialJoinPrepared */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=datum datum=46 caption="Tract 17031031400 (row 46): matched through the retained geometry snapshot."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=kernel caption="Geometry refresh is GPU-gated; bounds and edge builders replay against the snapshot."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=graph caption="Refresh the geometry, replay public builders, then commit the applied generation on the GPU."}</p>
<h3 id="gpuspatialjoincandidates"><code>GPUSpatialJoinCandidates</code></h3>
<p>Lists feature pairs whose bounding boxes overlap: the filter step before exact predicates.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=result caption="Park and tract pairs with overlapping boxes; boxes can overlap where polygons do not."}</p>
<p>Maximum: <code>output.leftRows.length</code> is at most <code>getGPUSpatialJoinCandidatesMaximumPairCapacity(device)</code> (2^24 on a default device).</p>
<p>The bounding-box candidate stage on its own: <code>(left, right)</code> pairs sorted by <code>(left, right)</code> whose bounding
boxes overlap, with the left box optionally expanded by <code>distance</code>. The <code>output</code> is a <code>GPUCompactPairPort</code>
(capacity <code>output.leftRows.length</code>, <code>status.count</code>, <code>status.overflow</code>, <code>status.candidateOverflow?</code>, <code>status.requiredCount</code>); slots past <code>status.count</code> hold <code>0xffffffff</code>.
<code>overflow</code> reports final pair-capacity loss only; <code>candidateOverflow</code> independently reports BVH
leaf loss. Both supplied words are overwritten with 0 or 1 on every encoding, including clean
reuse. When BVH leaves are lost, <code>requiredCount</code> counts only enumerated pairs and is a lower bound.
A <code>leafCapacity</code> below the right feature count requires <code>candidateOverflow</code>, including
when using a prepared index. Consumers that omit that diagnostic must use enough BVH slots;
otherwise construction rejects the configuration. <code>GPUPairGather</code> propagates the two causes separately.
The constructor captures geometry descriptors, options, pair views and nested status destinations;
editing the caller's maps later cannot retarget a declared binding. Buffer contents remain dynamic.
The separate flag passes permit adjacent <code>overflow</code> and <code>candidateOverflow</code> words in one buffer;
the words must not logically overlap. Outputs bound together must also have disjoint aligned ranges.
With both flags zero it is conservative: no intersecting pair is missed, and some candidates do not intersect. It has no
<code>excludeSameRow</code>.</p>
<p><code>distance</code> is planar in coordinate units (<code>spatialContext</code> defaults to native planar coordinates and a
<code>longitude-latitude</code> context throws; use <code>GPUSpatialPredicateJoin</code> <code>dwithin</code> for great-circle candidates).
It may be a compile-time number or a one-row <code>GraphDataView&#x3C;'float32'></code>. The view is read every
encoding, so an interactive distance join can reuse one compiled graph. Size the pair capacity for the
largest distance that the application permits. Negative, NaN and infinite view values act as zero; invalid
compile-time numbers throw.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> candidateDistance</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'candidate-distance'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialJoinCandidates</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  left,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  right,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  distance: candidateDistance.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {leftRows, rightRows, status: {count, overflow, requiredCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> compiled</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> await</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">compileAsync</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">();</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">candidateDistance.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(Float32Array.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">of</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">compiled.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">encode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(commandEncoder, {parameters: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">undefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUSpatialJoinCandidates */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=datum datum=46 caption="Tract 17031031400 (row 46): every park whose box overlaps the tract's box."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=kernel caption="Count and write kernels test box overlap; the write pass fills each row's scanned slots."}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=graph caption="Count candidates per row, prefix-scan to offsets, then write each row's pairs in place."}</p>
<h3 id="gpupointinpolygonjoin-and-gpunearestfeaturejoin"><code>GPUPointInPolygonJoin</code> and <code>GPUNearestFeatureJoin</code></h3>
<p><code>GPUPointInPolygonJoin</code>: Chunked queries keep global source order and shared polygon geometry.
candidateTileCapacity bounds paged candidate scratch independently of the unchanged global
candidateCapacity; global overflow/status semantics remain.</p>
<p><strong><code>GPUPointInPolygonJoin</code></strong>: Assigns each point the polygon that contains it, under an explicit boundary rule, and counts points per polygon.</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=result caption="2,344 observations (grey) assigned to the containing tract; colour is the tract."}</p>
<p><strong><code>GPUNearestFeatureJoin</code></strong>: Finds each point's nearest point or segment feature within a radius and reports its id and distance.</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=result caption="Observations coloured by planar distance to the nearest library."}</p>
<p><strong><code>GPUNearestFeatureWeights</code></strong>: Turns each point's k nearest features into a CSR spatial-weights matrix with inverse-distance weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=result caption="Each observation linked to its three nearest libraries, weighted by inverse distance."}</p>
<p>Maximum (<code>GPUNearestFeatureJoin</code>, neighbors mode and great-circle): the node table <code>5 * (2 * leafCapacity - 1) + 2 * queries</code> (plus <code>2 * features</code> with a filter) is at most <code>getGPUNearestFeatureJoinMaximumNodeTableWordCount(device)</code> (2^25 words on a default device), which allows 2^21 leaves without queries; <code>leafCapacity</code> is at most <code>getGPUNearestFeatureJoinMaximumLeafCapacity(device)</code> (2^23).</p>
<p>Spatial joins over a <code>GPUBVH</code> built from feature bounds every encoding. <code>GPUPointInPolygonJoin</code>
classifies each (point, feature) candidate with the shared exact orientation sign (even-odd rings, union
polygons) and assigns each point the containing feature with the smallest row (see the boundary rule
below); <code>GPUNearestFeatureJoin</code> measures candidates with
<code>GPUPairwisePointSegmentDistance</code> and keeps the nearest point or segment feature within a per-frame
radius. Both take the points as a <code>GPUPointGeometryPort</code> (<code>points</code> and <code>queries</code>; chunked positions and
<code>sourceIds</code> allowed) and the features as a geometry port whose optional <code>sourceIds</code> replace feature rows in
the outputs. Both write, in <code>output</code>, per-point feature IDs, per-feature counts, optional compact matches,
and one <code>output.status.overflow</code> flag covering BVH leaf, candidate, and match capacity. <code>GPUPointInPolygonJoin</code> takes planar
coordinates (the default) or <code>{coordinateSpace: 'longitude-latitude', metric: 'none'}</code>. Containment is
topological with straight edges in degrees, as in GeoJSON, and gives the same result as planar. A
longitude/latitude context with a metric such as <code>'great-circle'</code> throws at construction, because the join
has no geodesic edges ("project positions to a planar frame first"). <code>GPUNearestFeatureJoin</code> also takes the
great-circle context below.</p>
<p>Both joins require linear longitude/latitude edges (omitted <code>edgeModel</code> means <code>'linear'</code>);
<code>edgeModel: 'great-circle'</code> throws with a preparation hint. Prepare polygons that cross ±180 or enclose
a pole with <code>GPULonLatPolygonPreparation</code>. Optional <code>output.status.antimeridianUncertainCount</code> and
<code>output.status.poleUncertainCount</code> report unprepared rings; nearest joins count both polygon inputs.
These counters reset each encoding for longitude/latitude and remain untouched for planar contexts,
which build no ring check. Empty rings and polygons in the preparation's fixed 2x topology are accepted.
<code>getPointInPolygonContainmentContext</code> preserves the input edge model and datum when selecting
<code>metric: 'none'</code>, so it cannot bypass edge-model validation.</p>
<p><strong>Great-circle nearest features (<code>GPUNearestFeatureJoin</code>, <code>GPUBufferSelection</code>).</strong> With
<code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> positions are longitude and
latitude in degrees and every distance (<code>radius</code>, <code>nearestDistances</code>, <code>neighborDistances</code>,
<code>GPUBufferSelection</code> <code>distance</code> and <code>output.distances</code>) is in the context's length unit (meters by default;
kilometers with <code>units: 'kilometers'</code>) on the mean Earth sphere (<code>sphereRadius</code> overrides the radius in
that unit). A <code>longitude-latitude</code> context with another metric and planar contexts with <code>great-circle</code>
throw at construction. Both distance operators reject <code>local-lambert-equal-area</code> frames. Both output modes use the branch-and-bound BVH traversal of the neighbors mode (nearest-feature
mode is the <code>k = 1</code> traversal followed by the nearest-feature columns, counts and matches), so
<code>candidateCapacity</code> is not used on this path (it is accepted and ignored, <code>candidateCount</code> reads 0, and
<code>GPUBufferSelection</code> still validates it). The traversal orders and bounds by central angle. A node is pruned
with a lower bound of the great-circle distance between its box and the query box,
<code>hav(d) = hav(dlat) + cos^2(max |lat|) hav(dlon)</code>, with the latitude gap, the circular longitude gap (through
the antimeridian when that is shorter) and the cosine of the latitude farthest from the equator over both
boxes, which never overestimates; a box 180 degrees or wider in longitude (a feature across the antimeridian)
counts as covering every longitude, and a box touching a pole has cosine 0. Edges are straight in
longitude/latitude with wrapped longitude differences, so the vertex boxes bound them exactly. Point to edge
distance uses <code>geodesicSegmentFraction</code> and <code>geodesicSegmentDistance</code>; a crossing or containment is decided
in the degree plane with the exact orientation sign after the second edge is shifted by a multiple of 360
degrees to overlap the first, and counts as distance 0. Foot points and query points are on the edges with
longitude wrapped to +-180. The mixed convention of the planar-in-degrees tests is therefore one edge model;
densify edges longer than a few kilometers, because the foot is exact to first order in the local frame of the
query. Prepare crossing and pole-enclosing polygons as described above.
No warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.
The planar kernels are unchanged and compile
none of this code. Oracle: <code>geodesic-nearest-fixtures.ts</code> (numpy 2.5.4 haversine of the edge densified in
longitude/latitude, <code>pyproj</code> 3.8.0 <code>Geod(a=R, f=0)</code> cross-check), scenes at the equator, latitude 60, across
the antimeridian and next to the pole. <code>GPUNearestFeatureWeights</code> adapts the join's outputs; pass the
join's <code>spatialContext</code> so its distances and <code>distanceFloor</code> share the context's length unit and
inverse-distance weights use those distances.</p>
<p><strong>Exact orientation.</strong> The classifier and the nearest-feature kernels use the same exact orientation sign
(an f32 filter with an exact 256-bit integer fallback), so "on the edge" means the same thing in the
point-in-polygon join, the predicate join and the nearest-feature join: a point exactly on an edge is
distance 0 in all three, and polygon containment (distance 0 inside a polygon) is decided with the exact
sign too. Distances and foot points stay f32. Only candidates that no arithmetic can decide stay
<code>uncertain</code>: non-finite input or malformed offsets. In <code>GPUPointInPolygonJoin</code> an empty polygon part
or a ring with fewer than three vertices contains nothing and is skipped, as in shapely, so the
feature's other parts still classify the point. Uncertain
pairs are never assigned. They are counted in the optional <code>uncertainCount</code>, so such a point stays
unassigned visibly rather than silently.</p>
<p><strong>Boundary rule (<code>GPUPointInPolygonJoin</code>).</strong> <code>boundaryRule</code> decides a point exactly on a polygon
boundary:</p>
<table>
<thead>
<tr>
<th>Rule</th>
<th>Boundary point</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'half-open'</code> (default)</td>
<td>belongs to exactly one feature of a valid coverage, independent of row order</td>
<td>The owner is the feature containing the point after an infinitesimal move toward +x and +y (+y dominates): left and bottom boundaries are included, right and top boundaries are excluded. A horizontal edge belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the feature whose interior touches its upper right. Points on the outer right or top boundary of a coverage belong to no feature.</td>
</tr>
<tr>
<td><code>'closed'</code></td>
<td>contained by every feature it touches, then the smallest row wins</td>
<td>Nothing is lost on the outer boundary; the owner of a shared-boundary point depends on the row order.</td>
</tr>
<tr>
<td><code>'open'</code></td>
<td>belongs to no feature</td>
<td>Only strict interior counts; boundary points are lost, not half-assigned.</td>
</tr>
</tbody>
</table>
<p>The half-open decision is the crossing-number convention made exact: a
supporting-line tie is broken by the sign of dx (the epsilon term of the move), or of -dy for a vertical
edge, so the result follows from the exact orientation signs and the vertex order does not matter. Overlapping
features (an invalid coverage) still resolve to the smallest containing row. Oracle: shapely
<code>Polygon.contains(Point(x + 1e-9, y + 1e-6))</code> on integer scenes, 24 row permutations, in the specs.
<code>GPUZonalStatistics</code> and the recipes that pass no <code>boundaryRule</code> use the half-open default, so a coverage's
counts partition the points.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNearestFeatureJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  queries: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'points'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: points},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  features: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'segments'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, starts: roadStarts, ends: roadEnds},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  radius: snapRadius.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), candidateCapacity: points.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {nearestFeatureIds, nearestDistances, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Both joins accept a <a href="#gpuspatialjoinprepared"><code>prepared</code></a> handle, so a static polygon or line set builds its
<code>GPUBVH</code> once. The handle must be added to the graph first, and <code>spatialSort</code> and <code>leafCapacity</code> then come
from the handle (conflicting props throw).</p>
<p><strong>k nearest features.</strong> Set <code>output.neighborIds</code> and <code>output.neighborCounts</code> on <code>GPUNearestFeatureJoin</code> to get the <code>k</code>
nearest features per query (PostGIS <code>a &#x3C;-> b ORDER BY ... LIMIT k</code>, GeoPandas <code>sjoin_nearest</code>). The
original mode above is unchanged. Each query walks the BVH nearest child first and prunes boxes farther than
the current <code>k</code>-th distance, which is exact and deterministic.</p>
<ul>
<li>
<p><code>queries</code> are points (possibly chunked), lines or polygons (one packed view each); features are <code>points</code>,
<code>segments</code>, <code>lines</code> or <code>polygons</code> (multipolygons, holes, even-odd). Distance is the minimum planar
distance between the geometries and 0 when a polygon contains the other geometry. Polygons must be valid.</p>
</li>
<li>
<p><code>k</code> and <code>neighborCapacity</code> are positive integers, bounded by the packed-index maximum
<code>0x7ffffffe</code> (2147483646); actual usable capacity is lower when device storage bindings cannot
hold the scratch/output views. <code>neighborCapacity >= k</code>. The 64-slot private-storage threshold
is not the public maximum: 65 and larger capacities use storage-backed lists. The interleaved
top-k scratch needs <code>3 + 2 * neighborCapacity</code> uint32 words per scratch query plus three shared
control words; each output column needs <code>queryCount * neighborCapacity</code> rows in one binding.
Foot-point scratch adds five words per slot. Each must fit <code>maxStorageBufferBindingSize</code>, and
all packed indices must fit uint32. <code>ties</code> is <code>'lowest-id'</code> (default) or <code>'all'</code>, which keeps every feature tied with the
<code>k</code>-th. <code>neighborCapacity</code> can exceed <code>k</code> only with <code>ties: 'all'</code>.
<code>radius</code> (per-frame, inclusive; omitted means unbounded) bounds the search. Negative, NaN and
infinite radius values yield no neighbors for that encoding; a later finite nonnegative radius
resumes matching on the same compiled graph.</p>
</li>
<li>
<p>Outputs are dense, <code>query * neighborCapacity + slot</code>, ordered by distance and then feature row:
<code>neighborIds</code> (remapped by <code>features.sourceIds</code>), <code>neighborCounts</code>, optional <code>neighborRows</code>
(the original feature rows for attribute lookup), optional <code>neighborDistances</code>,
<code>neighborFootPoints</code> (the nearest point on the feature, or the crossing point when geometries cross)
and <code>neighborSegmentIndices</code> (the start-vertex index of the nearest edge; <code>GPU_NEAREST_NO_SEGMENT</code> for
points and containment). Unused slots hold <code>GPU_SPATIAL_JOIN_NO_FEATURE</code>, -1, NaN and
<code>GPU_NEAREST_NO_SEGMENT</code>. <code>neighborRows</code> uses <code>GPU_SPATIAL_JOIN_NO_FEATURE</code> in unused slots,
requires <code>neighborIds</code>, and has the same length. Allocate disjoint views for IDs and rows even
when no <code>sourceIds</code> are supplied.</p>
</li>
<li>
<p><code>output.status.overflow</code> is 1 when <code>ties: 'all'</code> had more ties than <code>neighborCapacity</code> or the BVH leaf capacity was
exceeded. Distances are f32, so near-equal distances can order differently from a double reference, and
tie order and the foot point at exact vertex ties depend on evaluation order.</p>
</li>
<li>
<p>The per-query scratch (result slots, and the foot-point scratch when a foot-point output is requested)
is one binding while it fits <code>maxStorageBufferBindingSize</code> (128 MiB by default). Beyond that, for
example 838k queries with <code>k = 8</code> and foot points, the scratch holds a fixed chunk of queries (a multiple
of 64) that every chunk of queries reuses in turn, so the query count is not limited by the scratch. The
caller-owned output views still have to fit their own bindings.</p>
</li>
<li>
<p><code>exclusive</code> skips features whose ID equals the query's ID (<code>features.sourceIds[row]</code> or the row, and optional
<code>queryIds</code>, which require <code>exclusive</code>), so the <code>k</code> neighbors are the nearest other features
(<code>STRtree.query_nearest(exclusive=True)</code>, <code>sjoin_nearest</code> self-joins). It compares IDs, not geometry
equality, which is the same when IDs are unique per geometry. Both modes support it.</p>
</li>
<li>
<p><code>neighborQueryPoints</code> (<code>float32x2</code>, the layout of <code>neighborIds</code>) is the point of the query geometry
nearest to the feature, NaN in unused slots. <code>(neighborQueryPoints, neighborFootPoints)</code> is
<code>shapely.shortest_line</code> or <code>ops.nearest_points</code> for every point, line and polygon pair. Ties between
parallel edges return the first in vertex order, which can differ from GEOS.</p>
</li>
<li>
<p><code>onAttribute: {left, right}</code> (as on <code>GPUSpatialPredicateJoin</code>) finds the nearest key-equal feature:
the filter runs inside the traversal, and nearest-feature mode filters candidates in the reduce passes.</p>
</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNearestFeatureJoin</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  queries: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'points'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: points},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  features: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  k: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, ties: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'all'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, neighborCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, radius: radiusView, spatialSort: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    neighborIds, neighborCounts, neighborDistances, neighborFootPoints, neighborSegmentIndices,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {overflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>spatialSort: true</code> reorders features along a Morton curve of their bound centers before the
<code>GPUBVH</code> build. Results are identical, and traversal is 30x to 75x cheaper when features are not
spatially coherent in row order.</p>
<h4 id="gpunearestfeatureweights"><code>GPUNearestFeatureWeights</code></h4>
<p>Adapter from the k-nearest output of <code>GPUNearestFeatureJoin</code> to a <code>GPUSpatialWeights</code> CSR (libpysal 4.15 <code>KNN</code>
and <code>Graph.build_knn</code>, for points, lines or polygons, using the join's exact geometry distances). Pass the
join's <code>neighborRows</code>, <code>neighborCounts</code>, optional <code>neighborDistances</code> and its <code>neighborCapacity</code> as
<code>slotCapacity</code>. Pass the join's resolved <code>spatialContext</code> to the adapter (omitted means native planar);
<code>distanceFloor</code> uses its length unit and <code>outputWeights.spatialContext</code> carries it to consumers. Per row it keeps the first <code>k</code> valid slots in the join's distance order, drops padding and
(with <code>excludeSelf</code>, for self-joins run with <code>k + 1</code>) the query's own feature row, then sorts the feature rows ascending. Short rows
stay short. <code>weightType</code> is <code>'binary'</code> (default) or <code>'inverse-distance'</code> (<code>1 / max(d, distanceFloor)^power</code>;
<code>distanceFloor</code> defaults to <code>1e-6</code>, the same default as <code>GPUNeighborSearch</code>, so coincident or touching features weigh <code>1e6^power</code>
and dominate a row once it is standardized; pass a floor on the scale of the data when that is not wanted),
and <code>output.distances</code> is filled when present (in the join's context length unit, also used by <code>distanceFloor</code>). The output is directed, so for <code>'symmetrize'</code> or symmetric
statistics take the union of the weights and their <code>GPUSpatialWeightsTranspose</code>. <code>excludeSelf</code> compares feature rows to
the query row. Request the join's <code>neighborRows</code> alongside its <code>neighborIds</code>; application IDs may
repeat across segment rows and must never become CSR column indices. Two winning segments with
source IDs <code>[7, 7]</code> produce columns <code>[0, 1]</code> when their distinct rows are used. The adapter's old
<code>neighborIds</code> input is rejected; precomputed input must likewise contain distinct feature rows per
query. <code>output</code> is a
<code>GPUSpatialWeightsOutput</code>. Pass the producing join's status as <code>neighborStatus</code> and provide
<code>output.status.candidateOverflow</code>: it carries any upstream search loss, including truncated tie
slots. <code>output.status.overflow</code> reports only the adapter's final CSR capacity loss. Both flags
are rewritten on each submission. Downstream statistics such as local Moran suppress inference
when either is nonzero. For precomputed complete slots, omit <code>neighborStatus</code>; an optional
candidate flag then receives zero.
The adapter likewise captures its input/output maps at construction. Its final and candidate flags
are written in separate passes and may occupy adjacent words of one buffer; using the same word
for both is rejected.</p>
<p>The spec pins libpysal on 24 random points with <code>k = 4</code> (identical neighbor sets). Deviations: ties at the
<code>k</code>-th distance keep the lowest feature rows (libpysal breaks them in KDTree order, and <code>KNN</code> and
<code>Graph.build_knn</code> disagree with each other on grids); inverse-distance floors the distance instead of
dividing by zero.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNearestFeatureWeights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  neighborRows, neighborCounts, neighborDistances, neighborStatus: joinStatus,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  slotCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, k: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, excludeSelf: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">knnWeights, status: {overflow, candidateOverflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="spatial-sort-default">Spatial sort default</h4>
<p><code>spatialSort</code> of <code>GPUPointInPolygonJoin</code>, <code>GPUNearestFeatureJoin</code> and <code>GPUBufferSelection</code> defaults to <code>true</code>
from 256 features and <code>false</code> below; pass <code>false</code> to turn it off. With shuffled features each encoding is
50–125x faster; with spatially coherent features it is about 1 ms slower. <code>GPUSpatialJoinPrepared</code>
uses the same threshold: sorting defaults on from 256 logical feature rows and off below it
(255 logical features: <code>false</code>; 256: <code>true</code>), including multipart points and lines.</p>
<p><code>spatialSort</code> orders the features along a Hilbert curve (the default) or a Morton curve
(<code>spatialSortCurve: 'morton'</code>, a prop of the three joins above, type <code>SpatialSortCurve</code>) before the BVH build.
Results are identical; in paired runs the Hilbert joins were 10 to 15 percent faster (point-in-polygon,
nearest, buffer selection). The prepared join and <code>GPUZoneEvents</code> have no curve prop and use Hilbert.</p>
<p><code>querySort</code> orders queries the same way: each point chunk, or the bounds centers of line/polygon queries, is
processed along a Hilbert curve over the root bounds of the feature BVH (non-finite points and empty bounds last,
equal keys in row order), so neighboring invocations walk
neighboring subtrees. Every query still reads and writes its own row, so all outputs are identical, including which
candidate pairs a <code>candidateCapacity</code> overflow drops. It is a prop of <code>GPUPointInPolygonJoin</code>, <code>GPUNearestFeatureJoin</code>
and <code>GPUBufferSelection</code> and defaults to <code>true</code> for point queries from 4096 queries and 256 features; line and polygon queries
default to <code>false</code> because the measured sort cost exceeds the traversal saving.
Lines default off: the measured 8192-line workload took 0.80 ms without sorting and 1.20 ms with sorting;
explicit <code>querySort: true</code> remains available. Point sorting stays enabled; polygon defaults are unchanged.
Line/polygon queries use the same source-row output addressing, including
chunked scratch and foot-point outputs; polygon query topology is packed to keep every pass within eight storage
bindings. Geometry sorting adds bounds and center passes. With a planar <code>radius</code> the second candidate pass keeps
input order.</p>
<p><code>edgeSlabs</code> (<code>GPUPointInPolygonJoin</code>) makes the exact refine test only the edges of the y-slab at the point's height
in each polygon part, instead of every edge of the candidate feature: O(slab) per candidate rather than O(vertices).
Results are identical (all boundary rules, the half-open owner, malformed offsets as UNCERTAIN); the one exception is
a pair that the full walk reports UNCERTAIN only because an unrelated far edge defeats the exact orientation test
(terms more than 200 binary orders apart). It defaults on when <code>polygons.positions.length * candidateCapacity</code> is at
least 4,000,000 and features average 8 vertices. <code>new GPUSpatialJoinPrepared({geometry, edgeSlabs: true})</code> builds the
index once with the BVH, and joins given that handle reuse it. Slab counts are limited by GPU-computed
total vertical edge travel and the entry budget, including ring closure. This prevents jagged long rings
from overflowing the index and reverting to a full walk through indirect edge references.</p>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUPointInPolygonJoin */}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=datum datum=0 caption="Observation 0: its candidate tracts and the one whose rings contain it."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=formula caption="Crossing parity with the half-open boundary rule; rings XOR within a part, parts OR."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=kernel caption="The classify kernel walks each candidate tract's edges with exact orientation signs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="Bounding-box candidates first, then one exact classify pass per candidate pair."}</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUNearestFeatureJoin */}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=datum datum=0 caption="Observation 0: its nearest library row and the planar distance in metres."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=formula caption="Argmin of squared distance, ties to the lower feature row; distance published as metres."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=kernel caption="The traverse kernel descends the library BVH, pruning boxes farther than the best so far."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=graph caption="BVH build over libraries, then one traversal per observation."}</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUNearestFeatureWeights */}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=datum datum=0 caption="Observation 0: its CSR row of three library neighbours and their weights."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=formula caption="Weight is one over distance (floored at 1 m) to the power; no row standardisation."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=kernel caption="The emit kernel writes each row's neighbour ids in ascending order with their weights."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=graph caption="Nearest search, per-row counts, offset scan, then the CSR emit pass."}</p>
<h3 id="gpupairgather-gpuoffsetexpansion-and-gpuboundsfilter"><code>GPUPairGather</code>, <code>GPUOffsetExpansion</code>, and <code>GPUBoundsFilter</code></h3>
<p><code>GPUBoundsFilter</code>: Chunked feature bounds use chunk-local bit masks and global block-count scratch.
Compact IDs retain global source-row order; output bindings and global block counts must fit the
device.</p>
<p><strong><code>GPUPairGather</code></strong>: Copies attributes from the source rows of matched pairs onto the pair rows.</p>
<p>The reusable pair-gather example allocates a producer <code>candidateOverflow</code> word when the caller omits
it and passes that completed pair port to both join and gather. Final <code>overflow</code> reports final output
capacity loss; <code>candidateOverflow</code> independently reports lost intermediate candidates. The pair-gather,
predicate-join and prepared-join examples mask histogram membership by <code>pairs.status.count</code> on every
submission, so unused allocation slots and stale IDs cannot contribute to tract bins after reuse.</p>
<p>::lens{scenario=gpu-spatial-analysis.pair-gather rung=result caption="Park area copied onto each park and tract pair, without clipping or apportioning."}</p>
<p><strong><code>GPUOffsetExpansion</code></strong>: Expands CSR offsets into one row per member, with its owner index and local index.</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-expansion rung=result caption="Observations grouped by tract receive their tract index and position within it."}</p>
<p><strong><code>GPUBoundsFilter</code></strong>: Flags features whose bounds intersect, lie within, or contain an axis-aligned box: a bounds test, not exact geometry.</p>
<p>::lens{scenario=gpu-spatial-analysis.bounds-filter rung=result caption="Observations inside the Lincoln Park bounding box (border included) are selected."}</p>
<p>Dataframe joints around the joins. <code>GPUPairGather</code> consumes join row indices and columns, so it takes no
<code>spatialContext</code>.</p>
<ul>
<li><strong><code>GPUPairGather</code></strong> merges join output with attribute columns (GeoPandas <code>sjoin</code> and <code>sjoin_nearest</code>
merges, <code>distance_col</code>). The input is <code>pairs</code> (a <code>GPUSpatialPredicateJoin</code> pair output) or <code>neighbors</code>
(the dense <code>[query, slot]</code> lists of <code>GPUNearestFeatureJoin</code>, flattened by stable compaction).
Compact pair ports use <code>pairs.leftRows</code> and <code>pairs.rightRows</code> as input row offsets. The old
<code>leftIds</code> and <code>rightIds</code> keys are rejected: application identities belong in attribute columns.
With rows <code>[0, 1]</code>, attributes <code>[10, 20]</code> and caller IDs <code>[1, 0]</code>, gathering returns attributes
<code>[10, 20]</code> and IDs <code>[1, 0]</code>; caller identity never chooses an attribute row.
For nearest joins request <code>output.neighborRows</code> alongside <code>neighborIds</code> and wire
<code>neighbors: {rows: neighborRows, counts: neighborCounts, capacity: neighborCapacity}</code>. Gather's
<code>rows</code> are feature row indices, while <code>neighborIds</code> may contain application IDs. For feature
attributes <code>[10, 20]</code>, a winning row 0 must gather 10 with either source IDs <code>[1, 0]</code> or
<code>[100, 200]</code>. Using IDs as rows would instead gather 20 or a missing-value fill. The old
<code>neighbors.ids</code> spelling is rejected, including in cast objects and spreads; use <code>neighbors.rows</code>
for precomputed row-valued lists too. View types cannot detect an application-ID buffer
deliberately assigned to <code>rows</code>, so preserve the producer's named row output when composing.
<code>how: 'inner'</code> emits matched rows, and <code>'left'</code> appends the unmatched left rows (<code>unmatchedLeft</code>, the
anti-join output, derived automatically for neighbors) with <code>rightRows = GPU_PAIR_GATHER_NO_ROW</code> and
NaN or <code>0xffffffff</code> in the right and slot columns. <code>leftColumns</code> and <code>rightColumns</code> gather <code>float32</code> or
<code>uint32</code> columns by row, and <code>slotColumns</code> carry slot-aligned columns such as <code>neighborDistances</code>. The
output order is matched pairs then unmatched, which is not GeoPandas' left order (sort by
<code>(left, right)</code> to compare). It is bounded by <code>output.leftRows.length</code>. <code>output.status.overflow</code>
reports final row capacity loss, including input final-output flags and <code>overflowSources</code>.
Supplied <code>pairs.status.candidateOverflow</code>, <code>unmatchedLeft.status.candidateOverflow</code> and
<code>candidateOverflowSources</code> propagate separately to <code>output.status.candidateOverflow</code>; that
destination is required when any candidate flag is supplied. Both flags are rewritten on every
submission. Growing the gathered output cannot recover candidate loss.
Gather captures the declared mode, port maps, columns and status fields once at construction,
including inherited getters. Later caller mutations cannot change graph declarations; GPU
buffer contents remain per-frame inputs. The recorded pair example preserves optional
<code>requiredCount</code> and candidate status views when completing its pair port. The catalog example
allocates separate final-overflow and candidate-overflow destinations for this composition.
When consuming nearest lists, pass the join's aggregate <code>output.status.overflow</code> through
<code>candidateOverflowSources</code> and request gather's candidate flag: that aggregate includes tie
truncation and BVH loss, which a larger gathered output cannot recover.
Predicate anti-join <code>unmatched</code> ports and point-in-polygon/nearest compact <code>matches</code> ports use
the same split: compact <code>overflow</code> is final capacity only, and <code>candidateOverflow</code> reports
candidate/BVH loss and clears on every submission. Their separate whole-join aggregate diagnostic
retains its documented meaning. Gather permits adjacent status words written by separate passes;
outputs written in one pass must have disjoint aligned binding ranges.
Chunked <code>GraphVectorView</code> inputs are unsupported.</li>
<li><strong><code>GPUOffsetExpansion</code></strong> gives the owner index and local index of every child row from an offsets view
(<code>explode</code>, <code>get_coordinates(return_index=True)</code>, <code>get_parts(return_index=True)</code>) by binary search per row.
Empty ranges yield no rows, <code>ownerMap</code> chains levels (vertex to ring to polygon to feature), and the
output is bounded with <code>status.count</code>, <code>status.overflow</code> and <code>status.requiredCount</code>. Offsets monotonicity is not checked.</li>
<li><strong><code>GPUBoundsFilter</code></strong> (<code>spatialContext</code>: planar, or longitude/latitude with metric <code>'none'</code>, treated
as plain coordinates with no antimeridian wrap, so a box needs <code>minX &#x3C;= maxX</code>) is a <code>.cx</code>-style mask and compaction over <code>float32x4</code> feature bounds with a
per-frame box (<code>parameters</code>, written with <code>getGPUBoundsFilterParameterValues</code>; the optional 0/1 mask is
<code>output.outputMask</code>). Modes are <code>'intersects'</code>, <code>'within'</code> and
<code>'contains'</code>, with closed comparisons and non-finite bounds rejected. It is bounds-level only, and OGC
<code>envelope.within(box)</code> rejects bounds that only touch the border where this accepts them, so refine
with <code>GPUSpatialPredicateJoin</code>.
<code>GPUGeometryMeasures</code> planar bounds can be connected directly. Its geographic bounds unwrap
longitude from each feature's first vertex, so normalize features and query to a common branch
or test all relevant query intervals shifted by whole turns and union their masks. For example,
bounds <code>[179,181]</code>, reversed-line bounds <code>[-181,-179]</code>, and one-turn-shifted bounds <code>[539,541]</code>
need equivalent viewport intervals <code>[180.25,180.75]</code>, <code>[-179.75,-179.25]</code>, and <code>[540.25,540.75]</code>.
Testing only <code>[-179.75,-179.25]</code> retains only the reversed line. <code>wrapX</code> splits a wrapped query
into two rays; it does not move feature bounds onto the query's longitude branch. Use the
filter's supported metric <code>'none'</code>, rather than passing a geographic measures metric context.</li>
</ul>
<p>Matched GeoPandas 1.2.0 <code>sjoin</code> (inner 11 pairs, left 32 rows) and <code>sjoin_nearest</code> with ties and
<code>distance_col</code>, and <code>explode</code> and <code>get_coordinates</code> indices, exactly.</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUPairGather */}</p>
<p>::lens{scenario=gpu-spatial-analysis.pair-gather rung=datum datum=46 caption="Tract 17031031400 (row 46): the park areas gathered onto its pairs."}</p>
<p>::lens{scenario=gpu-spatial-analysis.pair-gather rung=kernel caption="Each pair row reads its source row and copies the attribute into its slot column."}</p>
<p>::lens{scenario=gpu-spatial-analysis.pair-gather rung=graph caption="Slot assignment, one gather pass per attribute column, then publication of the gathered rows."}</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUOffsetExpansion */}</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-expansion rung=datum datum=0 caption="The first observation in tract order: its tract and its local index within it."}</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-expansion rung=kernel caption="One invocation per member binary-searches the offsets for its owner, then subtracts the start."}</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-expansion rung=graph caption="A single expansion pass over the CSR offsets; no sort or scan is needed."}</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUBoundsFilter */}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounds-filter rung=datum datum=0 caption="Observation 0: its position against the Lincoln Park box, and the mask bit."}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounds-filter rung=formula caption="Within: every bound inside the box, borders included, compared in float32."}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounds-filter rung=kernel caption="One invocation per feature compares its bounds with the box and writes one mask word."}</p>
<p>::lens{scenario=gpu-spatial-analysis.bounds-filter rung=graph caption="One mask pass, then a visibility workflow compacts and publishes the selected rows."}</p>
<p>{/* parameters: GPUBoundsFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuboundsfilter-parameters"><code>GPUBoundsFilter</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUBoundsFilterParameterValues(settings)</code>, writes <code>GPU_BOUNDS_FILTER_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUBoundsFilterParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>minX</code></td>
<td><code>number</code></td>
<td>Left edge.</td>
</tr>
<tr>
<td><code>minY</code></td>
<td><code>number</code></td>
<td>Bottom edge.</td>
</tr>
<tr>
<td><code>maxX</code></td>
<td><code>number</code></td>
<td>Right edge, at least <code>minX</code> unless <code>wrapX</code> is set.</td>
</tr>
<tr>
<td><code>maxY</code></td>
<td><code>number</code></td>
<td>Top edge, at least <code>minY</code>.</td>
</tr>
<tr>
<td><code>wrapX</code> (optional)</td>
<td><code>boolean</code></td>
<td>Allows <code>minX > maxX</code>, meaning the box wraps through the longitude seam: it covers <code>x >= minX</code> and <code>x &#x3C;= maxX</code> (for example <code>minX = 170, maxX = -170</code>). Intended for longitude/latitude contexts. Feature bounds must not wrap themselves. A wrapped box never satisfies <code>'contains'</code>, because a finite feature box cannot cover both sides of the seam.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpubufferselection"><code>GPUBufferSelection</code></h3>
<p>Keep the observation and reference views distinct: <code>points.positions</code> contains rows to select;
<code>features.positions</code> contains station or bus-stop rows. For example, an observation at <code>(0, 0)</code>
is outside a 400 m buffer around a station at <code>(1000, 0)</code>. Binding the observation view to both
sides would create a self-match at zero distance. Size candidate capacity from the query view's
row count. The Near Transit snippet uses <code>observationPositions</code>, <code>stationPositions</code> or
<code>busPositions</code>, and separate <code>transitStarts</code>/<code>transitEnds</code> for line features.</p>
<p>Selects features within a distance of any reference feature, without building buffer polygons.</p>
<p>The compact <code>output.matches.status.overflow</code> reports final selection capacity loss. Optional
<code>output.matches.status.candidateOverflow</code> reports join candidate loss separately. <code>requiredCount</code>
counts matches reaching compaction and is a lower bound when candidates were lost.</p>
<p>::lens{scenario=gpu-spatial-analysis.buffer-selection rung=result caption="Observations within 300 m of a library are selected; distance is straight-line, not walking."}</p>
<p>No warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Selects points within a per-frame distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The contributor composes <code>GPUNearestFeatureJoin</code>. Unlike
the join's unordered <code>matches</code>, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order (<code>output.outputMask</code>, <code>output.matches</code>), with a clamped count. <code>output.matches.status.overflow</code>
reports final output capacity loss; optional <code>output.matches.status.candidateOverflow</code> reports
join candidate loss separately. <code>points</code> is a <code>GPUPointGeometryPort</code> whose <code>sourceIds</code> become the match IDs.
Optional <code>output.nearestFeatureRows</code> contains the original feature row before application-ID
remapping; it replaces the misleading <code>nearestFeatureIds</code> spelling, which is rejected. Selection
uses those rows, so a feature ID of <code>0xffffffff</code> still selects a genuinely matched point. Feature
<code>sourceIds</code> are ignored by this selection operation; query <code>points.sourceIds</code> still identify the
selected query points in <code>matches.ids</code>.
<code>output.drawInstanceCount</code> (one <code>uint32</code> row, requires <code>output.matches</code>) receives the clamped selected count
from the publish node, so it can be bound as an indirect draw instance count. The distance is planar in
coordinate units by default; with <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> it is a great-circle distance in the context's length unit (meters by default; kilometers
with <code>units: 'kilometers'</code>). The context is forwarded to <code>GPUNearestFeatureJoin</code>, whose section lists the
edge model, sphere radius and rejected combinations; <code>candidateCapacity</code> is ignored.
It measures distance, so <code>local-lambert-equal-area</code> frames throw. Optional <code>output.candidateTruncated</code> (one
packed <code>uint32</code> per point) and <code>output.status.candidateOverflow</code> pass the join's candidate status through: a
flagged point may be left out although a feature lies within the distance (see <strong>Candidate overflow</strong> above).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUBufferSelection</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  points: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'points'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: points},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  features: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'segments'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, starts: roadStarts, ends: roadEnds},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  distance: bufferDistance.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), candidateCapacity: points.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialSort: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, output: {outputMask, matches}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUBufferSelection */}</p>
<p>::lens{scenario=gpu-spatial-analysis.buffer-selection rung=datum datum=0 caption="Observation 0: distance to its nearest library and whether it is within 300 m."}</p>
<p>::lens{scenario=gpu-spatial-analysis.buffer-selection rung=formula caption="Selected when the nearest-library distance is between 0 and r, here 300 m."}</p>
<p>::lens{scenario=gpu-spatial-analysis.buffer-selection rung=kernel caption="The mask kernel turns each nearest distance into a selection bit."}</p>
<p>::lens{scenario=gpu-spatial-analysis.buffer-selection rung=graph caption="A nearest-library join within 300 m, a mask pass, then compaction of the selected rows."}</p>`};export{e as default};