# Geometry refinement computation lenses

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

Two reference points are inserted along a six-unit segment in projection order. Counts delimit the fixed output slots; validity and source edges are captured.

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

{/* lens:GPUVectorStraightSkeleton */}

::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.vector-straight-skeleton rung=graph caption="Follow the geometry stages and provenance."}

## `GPUOffsetCurveComponents`

Select connected offset components by side and distance. This small synthetic example captures status and all bounded geometry outputs.

{/* lens:GPUOffsetCurveComponents */}

::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=result caption="Inspect the actual GPU output and completion status."}

::lens{scenario=gpu-spatial-analysis.offset-curve-components rung=graph caption="Follow the geometry stages and provenance."}
