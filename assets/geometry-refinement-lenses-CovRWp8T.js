var e={slug:`api/gpu-spatial-analysis/geometry-refinement-lenses`,title:`Geometry refinement computation lenses`,html:`<h1 id="geometry-refinement-computation-lenses">Geometry refinement computation lenses</h1>
<p>These executable examples use small, explicitly synthetic coordinates in arbitrary planar units. Each lens captures the actual GPU command graph, input buffers and bounded outputs. Read active counts, validity masks and status before interpreting unused capacity. See <a href="/luma-contributors/docs/api/gpu-geometry-refinement">Geometry refinement</a> for supported domains and limitations; these examples are not independent numerical validation or performance benchmarks.</p>
<h2 id="gpunormalizepolygons"><code>GPUNormalizePolygons</code></h2>
<p>A rotated, counterclockwise rectangle becomes a clockwise ring starting at its smallest coordinate. Offset captures retain the polygon hierarchy.</p>
<p>{/* lens:GPUNormalizePolygons */}</p>
<p>::lens{scenario=gpu-spatial-analysis.normalize-polygons rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.normalize-polygons rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpucanonicalpolygonequality"><code>GPUCanonicalPolygonEquality</code></h2>
<p>Both inputs are normalized on the GPU before equality compares their structural coordinates. The equality code is one for the two differently ordered rectangle inputs.</p>
<p>{/* lens:GPUCanonicalPolygonEquality */}</p>
<p>::lens{scenario=gpu-spatial-analysis.canonical-polygon-equality rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.canonical-polygon-equality rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpusubdividepolygons"><code>GPUSubdividePolygons</code></h2>
<p>A rectangle is triangulated and merged into convex pieces with at most four vertices. Piece metadata provides the active vertex counts and original feature, polygon and seed triangle.</p>
<p>{/* lens:GPUSubdividePolygons */}</p>
<p>::lens{scenario=gpu-spatial-analysis.subdivide-polygons rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.subdivide-polygons rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpupolygonprecisionreduction"><code>GPUPolygonPrecisionReduction</code></h2>
<p>An implicitly closed rectangle with fractional coordinates is reconstructed on a power-of-two unit grid. Inspect the actual lattice and topology counts before reading the bounded polygon storage.</p>
<p>{/* lens:GPUPolygonPrecisionReduction */}</p>
<p>::lens{scenario=gpu-spatial-analysis.polygon-precision-reduction rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.polygon-precision-reduction rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpulineprecisionreduction"><code>GPULinePrecisionReduction</code></h2>
<p>Two nearby line vertices round to the same grid point. The keep-collapsed policy retains two equal endpoints and reports status two.</p>
<p>{/* lens:GPULinePrecisionReduction */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-precision-reduction rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-precision-reduction rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpusegmentsnap"><code>GPUSegmentSnap</code></h2>
<p>Default <code>searchStructure: 'grid'</code> reuses candidate scratch in batches of at most 256 lines.
<code>candidateCapacity</code> defaults to 1,024 and is at most 4,294,967,290. Overflow uses bounded exhaustive
fallback or reports incomplete geometry; <code>candidateOverflowPolicy: 'incomplete'</code> disables fallback.
If this can suppress a line, supply <code>issueCode</code>, <code>validity</code> or <code>status.candidateOverflow</code>; required
geometry/count outputs alone are rejected because suppressed and genuinely empty lines have the
same zero counts. This check applies to GPU and CPU operation forms too.
Explicit brute force and fallback bound <code>(positions + lines) * (references + 1)^3</code> by 16,777,216.</p>
<p>Two reference points are inserted along a six-unit segment in projection order. Counts delimit the fixed output slots; issue codes and source edges are captured. Optional validity is nonzero for usable lines.</p>
<p>{/* lens:GPUSegmentSnap */}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-snap rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.segment-snap rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpulineloopremoval"><code>GPULineLoopRemoval</code></h2>
<p>A four-vertex crossing line is noded, then its chronological loop is removed. The remaining path and source edges retain traversal order.</p>
<p>{/* lens:GPULineLoopRemoval */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-loop-removal rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-loop-removal rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpuratioconcavehull"><code>GPURatioConcaveHull</code></h2>
<p>Five points with unequal exposed edge lengths use ratio 0.1 and no holes. Triangle and edge validity masks delimit the active hull; convergence reports exhausted work.</p>
<p>{/* lens:GPURatioConcaveHull */}</p>
<p>::lens{scenario=gpu-spatial-analysis.ratio-concave-hull rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.ratio-concave-hull rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpumaximuminscribedcircle"><code>GPUMaximumInscribedCircle</code></h2>
<p>A four-by-two rectangle uses tolerance 0.01, 128 live cells and 128 splits. Circle rows include the remaining geometric search error; a witness and status accompany the result.</p>
<p>{/* lens:GPUMaximumInscribedCircle */}</p>
<p>::lens{scenario=gpu-spatial-analysis.maximum-inscribed-circle rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.maximum-inscribed-circle rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpuconvexstraightskeleton"><code>GPUConvexStraightSkeleton</code></h2>
<p>A strictly convex four-by-two rectangle produces corner arcs and a terminal ridge. An inward offset at distance 0.5 is captured alongside active counts and unsupported/budget flags.</p>
<p>{/* lens:GPUConvexStraightSkeleton */}</p>
<p>::lens{scenario=gpu-spatial-analysis.convex-straight-skeleton rung=result caption="Inspect the synthetic input, GPU result and status buffers."}</p>
<p>::lens{scenario=gpu-spatial-analysis.convex-straight-skeleton rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}</p>
<h2 id="gpunormalizegeometry"><code>GPUNormalizeGeometry</code></h2>
<p>Normalize a mixed collection without losing its structure. This small synthetic example captures status and all bounded geometry outputs.</p>
<p>{/* lens:GPUNormalizeGeometry */}</p>
<p>::lens{scenario=gpu-spatial-analysis.normalize-geometry rung=result caption="Inspect the actual GPU output and completion status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.normalize-geometry rung=graph caption="Follow the geometry stages and provenance."}</p>
<h2 id="gpucanonicalgeometryequality"><code>GPUCanonicalGeometryEquality</code></h2>
<p>Compare normalized mixed geometry roots. This small synthetic example captures status and all bounded geometry outputs.</p>
<p>{/* lens:GPUCanonicalGeometryEquality */}</p>
<p>::lens{scenario=gpu-spatial-analysis.canonical-geometry-equality rung=result caption="Inspect the actual GPU output and completion status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.canonical-geometry-equality rung=graph caption="Follow the geometry stages and provenance."}</p>
<h2 id="gpuvectorstraightskeleton"><code>GPUVectorStraightSkeleton</code></h2>
<p>Hole-aware event wavefront and inward offset. This small synthetic example captures status and all bounded geometry outputs.</p>
<p>{/* lens:GPUVectorStraightSkeleton */}</p>
<p>::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=result caption="Inspect the actual GPU output and completion status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=graph caption="Follow the geometry stages and provenance."}</p>
<h2 id="gpuoffsetcurvecomponents"><code>GPUOffsetCurveComponents</code></h2>
<p>Select connected offset components by side and distance. This small synthetic example captures status and all bounded geometry outputs.</p>
<p>{/* lens:GPUOffsetCurveComponents */}</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=result caption="Inspect the actual GPU output and completion status."}</p>
<p>::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=graph caption="Follow the geometry stages and provenance."}</p>`};export{e as default};