# Geometry refinement computation lenses

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

These executable examples use small, explicitly synthetic coordinates in arbitrary planar units. Each lens captures the actual GPU command graph, input buffers and bounded outputs. Read active counts, validity masks and status before interpreting unused capacity. See [Geometry refinement](../gpu-geometry-refinement.md) for supported domains and limitations; these examples are not independent numerical validation or performance benchmarks.

## `GPUNormalizePolygons`

A rotated, counterclockwise rectangle becomes a clockwise ring starting at its smallest coordinate. Offset captures retain the polygon hierarchy.

{/* lens:GPUNormalizePolygons */}

::lens{scenario=gpu-spatial-analysis.normalize-polygons rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.normalize-polygons rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUCanonicalPolygonEquality`

Both inputs are normalized on the GPU before equality compares their structural coordinates. The equality code is one for the two differently ordered rectangle inputs.

{/* lens:GPUCanonicalPolygonEquality */}

::lens{scenario=gpu-spatial-analysis.canonical-polygon-equality rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.canonical-polygon-equality rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUSubdividePolygons`

A rectangle is triangulated and merged into convex pieces with at most four vertices. Piece metadata provides the active vertex counts and original feature, polygon and seed triangle.

{/* lens:GPUSubdividePolygons */}

::lens{scenario=gpu-spatial-analysis.subdivide-polygons rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.subdivide-polygons rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUPolygonPrecisionReduction`

An implicitly closed rectangle with fractional coordinates is reconstructed on a power-of-two unit grid. Inspect the actual lattice and topology counts before reading the bounded polygon storage.

{/* lens:GPUPolygonPrecisionReduction */}

::lens{scenario=gpu-spatial-analysis.polygon-precision-reduction rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.polygon-precision-reduction rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPULinePrecisionReduction`

Two nearby line vertices round to the same grid point. The keep-collapsed policy retains two equal endpoints and reports status two.

{/* lens:GPULinePrecisionReduction */}

::lens{scenario=gpu-spatial-analysis.line-precision-reduction rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.line-precision-reduction rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUSegmentSnap`

Default `searchStructure: 'grid'` reuses candidate scratch in batches of at most 256 lines.
`candidateCapacity` defaults to 1,024 and is at most 4,294,967,290. Overflow uses bounded exhaustive
fallback or reports incomplete geometry; `candidateOverflowPolicy: 'incomplete'` disables fallback.
If this can suppress a line, supply `issueCode`, `validity` or `status.candidateOverflow`; required
geometry/count outputs alone are rejected because suppressed and genuinely empty lines have the
same zero counts. This check applies to GPU and CPU operation forms too.
Explicit brute force and fallback bound `(positions + lines) * (references + 1)^3` by 16,777,216.

Two reference points are inserted along a six-unit segment in projection order. Counts delimit the fixed output slots; issue codes and source edges are captured. Optional validity is nonzero for usable lines.

{/* lens:GPUSegmentSnap */}

::lens{scenario=gpu-spatial-analysis.segment-snap rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.segment-snap rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPULineLoopRemoval`

A four-vertex crossing line is noded, then its chronological loop is removed. The remaining path and source edges retain traversal order.

{/* lens:GPULineLoopRemoval */}

::lens{scenario=gpu-spatial-analysis.line-loop-removal rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.line-loop-removal rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPURatioConcaveHull`

Five points with unequal exposed edge lengths use ratio 0.1 and no holes. Triangle and edge validity masks delimit the active hull; convergence reports exhausted work.

{/* lens:GPURatioConcaveHull */}

::lens{scenario=gpu-spatial-analysis.ratio-concave-hull rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.ratio-concave-hull rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUMaximumInscribedCircle`

A four-by-two rectangle uses tolerance 0.01, 128 live cells and 128 splits. Circle rows include the remaining geometric search error; a witness and status accompany the result.

{/* lens:GPUMaximumInscribedCircle */}

::lens{scenario=gpu-spatial-analysis.maximum-inscribed-circle rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.maximum-inscribed-circle rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}

## `GPUConvexStraightSkeleton`

A strictly convex four-by-two rectangle produces corner arcs and a terminal ridge. An inward offset at distance 0.5 is captured alongside active counts and unsupported/budget flags.

{/* lens:GPUConvexStraightSkeleton */}

::lens{scenario=gpu-spatial-analysis.convex-straight-skeleton rung=result caption="Inspect the synthetic input, GPU result and status buffers."}

::lens{scenario=gpu-spatial-analysis.convex-straight-skeleton rung=graph caption="Follow the actual bounded GPU computation and its intermediate stages."}


## `GPUNormalizeGeometry`

Normalize a mixed collection without losing its structure. This small synthetic example captures status and all bounded geometry outputs.

{/* lens:GPUNormalizeGeometry */}

::lens{scenario=gpu-spatial-analysis.normalize-geometry rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.normalize-geometry rung=graph caption="Follow the geometry stages and provenance."}

## `GPUCanonicalGeometryEquality`

Compare normalized mixed geometry roots. This small synthetic example captures status and all bounded geometry outputs.

{/* lens:GPUCanonicalGeometryEquality */}

::lens{scenario=gpu-spatial-analysis.canonical-geometry-equality rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.canonical-geometry-equality rung=graph caption="Follow the geometry stages and provenance."}

## `GPUVectorStraightSkeleton`

Hole-aware event wavefront and inward offset. This small synthetic example captures status and all bounded geometry outputs.

The ten-by-ten shell with square hole `[3,7] × [3,7]`, offset 0.25 and event budget 16 has
12 live arcs and status `[12,12,8,0]` in the current CPU reference. The verified CPU provenance
for terminal rows 8–11 is `[7,0], [1,6], [2,5], [3,4]` (original bounding-edge rows).
The historical capture instead records `[6,7], [0,1], [1,2], [2,3]` for those arcs.
Matching geometry and a zero flags word do not validate those source edges. The live GPU
provenance regression and lens recapture remain pending; the embedded historical recording
below is not evidence that its source-edge metadata matches the current implementation.

{/* lens:GPUVectorStraightSkeleton */}

::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=graph caption="Follow the geometry stages and provenance."}

## `GPUOffsetCurveComponents`

Select connected offset components by side and distance. This small synthetic example captures status and all bounded geometry outputs.

{/* lens:GPUOffsetCurveComponents */}

::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=graph caption="Follow the geometry stages and provenance."}
