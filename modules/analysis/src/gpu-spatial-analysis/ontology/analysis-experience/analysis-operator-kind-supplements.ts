// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisOperatorCatalog} from '../all-entry-points/analysis-operator-catalog';
import {GPU_SPATIAL_ANALYSIS_DATA_KINDS} from '../spatial-analysis-ontology';
import type {GPUAnalysisOperator, GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';
import {getGPUAnalysisOperatorQuestions} from './analysis-operator-questions';

/** Most operators a path may chain. Longer chains are rarely meaningful and the count grows fast. */
export const GPU_ANALYSIS_MAXIMUM_PATH_STEPS = 3;

/** Paths {@link findGPUAnalysisOperatorPaths} returns when `maximumPaths` is not given. */
export const GPU_ANALYSIS_MAXIMUM_PATH_COUNT = 50;

/** Largest `maximumPaths` accepted, so one answer stays small enough to render. */
export const GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT = 500;

/**
 * Most distinct chains one search counts, in final ranking order. This bounds work and retained
 * candidates at hub kinds while exceeding the largest requested answer, so truncation keeps the best paths.
 */
export const GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES = 10000;

/** Input and output data kinds of one operator. */
export type GPUAnalysisOperatorKinds = {
  inputs: readonly GPUSpatialAnalysisDataKind[];
  outputs: readonly GPUSpatialAnalysisDataKind[];
};

/** Where one side (inputs or outputs) of an operator's kinds came from. */
export type GPUAnalysisOperatorKindSideSource = 'catalogue' | 'supplement' | 'unknown';

/** Where {@link resolveGPUAnalysisOperatorKinds} got its answer: one source, `mixed`, or `unknown`. */
export type GPUAnalysisOperatorKindSource = GPUAnalysisOperatorKindSideSource | 'mixed';

/** Resolved kinds of one operator and where each side came from. */
export type GPUAnalysisResolvedOperatorKinds = GPUAnalysisOperatorKinds & {
  /**
   * `catalogue` or `supplement` when every known side came from there, `mixed` when one side came
   * from each, `unknown` when neither side is known.
   */
  source: GPUAnalysisOperatorKindSource;
  inputsSource: GPUAnalysisOperatorKindSideSource;
  outputsSource: GPUAnalysisOperatorKindSideSource;
};

/** The part of a catalogue record that kind resolution reads. */
export type GPUAnalysisOperatorKindRecord = Pick<
  GPUAnalysisOperator,
  'exportName' | 'inputs' | 'outputs'
>;

/**
 * Operators that produce no table: they build a reusable handle (an index, a prepared geometry) that
 * other operators take through a prop, so their supplement row has empty outputs. Only these
 * operators may have empty outputs in {@link GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS}.
 */
export const GPU_ANALYSIS_HANDLE_OPERATORS: readonly string[] = Object.freeze([
  // Builds bounds and a BVH over a feature set; it has no output port, only `storage`.
  'GPUSpatialJoinPrepared'
]);

/**
 * Input and output kinds for the operators whose catalogue record lists none on one side or both.
 * Each kind was read from the operator's own props and outputs: a view of positions is `points`,
 * line offsets are `lines`, ring offsets are `polygons`, a `GPUSpatialWeights` port is
 * `spatial-weights`, a raster `grid` is `surface`, a column of per-row values is `attributes`, a
 * per-frame `parameters` view is `parameters` and a statistics or test report is `summary`.
 * Label kind convention (taken from the catalogue): the catalogue gives cluster labels (OPTICS,
 * HDBSCAN) and masks `attributes`, and `regions` only to contiguous regionalization and
 * segmentation (AZP, Max-P, Ward, region graphs). The rows here follow that: cluster labels,
 * selection masks and map colours are `attributes`, regionalization output is `regions`, and
 * operators that read a label column list both `attributes` and `regions`.
 * Inputs list `parameters` only when the operator has a per-frame `parameters` port, as populated
 * catalogue records do. A generator with no data input lists `parameters` alone. Each row names the
 * source file (relative to `modules/analysis/src`) and the ports that decided it.
 *
 * A row is stale when the catalogue describes both sides of its operator, or when the operator is
 * gone. The spec for this file fails on both, so delete a row when the catalogue catches up. When
 * the catalogue describes only one side, {@link resolveGPUAnalysisOperatorKinds} takes the other
 * side from the row.
 */
export const GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS: Readonly<
  Record<string, GPUAnalysisOperatorKinds>
> = freezeSupplementRows({
  // gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts: values, weights, quantileParameters; membership, classes, pValues
  GPUQuantileLISA: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts: positions, values; positions, radii
  GPUDorlingCartogram: {inputs: ['points', 'attributes'], outputs: ['points', 'attributes']},
  // gpu-spatial-analysis/recipes/rate-mapping-recipes.ts: events, populations, mask; excess risk column and summary
  addExcessRiskRecipe: {inputs: ['attributes'], outputs: ['attributes', 'summary']},
  // gpu-spatial-analysis/recipes/rate-mapping-recipes.ts: events, populations, weights; spatial rates
  addSpatialRateRecipe: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts: positions, response, fold masks; scores, predictions, metrics
  GPUNestedInterpolationValidation: {
    inputs: ['points', 'attributes'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/geometry-measures/gpu-geometry-measures.ts: geometryType points, lines or polygons; lengths, areas, centroids
  GPUGeometryMeasures: {inputs: ['points', 'lines', 'polygons'], outputs: ['attributes']},
  // gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors.ts: polygon geometry fields; compactness and orientation columns
  GPUShapeDescriptors: {inputs: ['polygons'], outputs: ['attributes']},
  // gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds.ts: polygon fields or positions plus labels; rectangle corners, circles, longest lines
  GPUMinimumBounds: {
    inputs: ['polygons', 'points', 'attributes', 'regions'],
    outputs: ['polygons', 'lines', 'attributes']
  },
  // gpu-spatial-analysis/minimum-clearance/gpu-minimum-clearance.ts: geometry GPUSegmentGeometry; clearances, lines
  GPUMinimumClearance: {inputs: ['lines', 'polygons'], outputs: ['attributes', 'lines']},
  // gpu-spatial-analysis/label-point/gpu-label-point.ts: polygon geometry; points, distances
  GPULabelPoint: {inputs: ['polygons'], outputs: ['points', 'attributes']},
  // gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution.ts: positions, weights, groupIds, parameters; centers, ellipse vertices
  GPUGeographicDistribution: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['points', 'polygons', 'attributes']
  },
  // gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs.ts: origins, targets; distances, bearings, midpoints
  GPUGeodesicPairs: {inputs: ['points', 'parameters'], outputs: ['attributes', 'points']},
  // gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination.ts: origins, bearings, distances; destinations
  GPUGeodesicDestination: {inputs: ['points', 'attributes'], outputs: ['points']},
  // gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts: positions GPUGeospatialPositions; projected positions
  GPULocalFrameProjection: {
    inputs: ['points', 'lines', 'polygons'],
    outputs: ['points', 'lines', 'polygons']
  },
  // gpu-spatial-analysis/geometry-edit/gpu-affine-transform.ts: positions, per-frame matrix; transformed positions
  GPUAffineTransform: {
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['points', 'lines', 'polygons']
  },
  // gpu-spatial-analysis/geometry-edit/gpu-geometry-orientation.ts: ring offsets; reversed positions
  GPUGeometryOrientation: {inputs: ['lines', 'polygons'], outputs: ['lines', 'polygons']},
  // gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup.ts: point, line or polygon output; cleaned positions
  GPUGeometryCleanup: {
    inputs: ['points', 'lines', 'polygons'],
    outputs: ['points', 'lines', 'polygons']
  },
  // gpu-spatial-analysis/vertex-snap/gpu-vertex-snap.ts: positions, referencePositions, parameters; snapped positions
  GPUVertexSnap: {
    inputs: ['points', 'lines', 'polygons', 'parameters'],
    outputs: ['points', 'lines', 'polygons']
  },
  // gpu-spatial-analysis/outline-geometry/gpu-offset-curve.ts: geometryType lines or polygons; offset positions
  GPUOffsetCurve: {inputs: ['lines', 'polygons'], outputs: ['lines']},
  // gpu-spatial-analysis/grid-generators/gpu-grid-generator.ts: gridType, columns, rows, parameters (generator); cell rings, centers
  GPUGridGenerator: {inputs: ['parameters'], outputs: ['polygons', 'points']},
  // gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts: centers, radii, parameters; rings with offsets
  GPUShapeGenerator: {inputs: ['points', 'attributes', 'parameters'], outputs: ['polygons']},
  // gpu-spatial-analysis/dot-density/gpu-dot-density.ts: polygon fields, values, parameters; dot positions
  GPUDotDensity: {inputs: ['polygons', 'attributes', 'parameters'], outputs: ['points']},
  // gpu-spatial-analysis/dot-density/gpu-random-points-in-polygon.ts: polygon fields, counts, parameters; point positions
  GPURandomPointsInPolygon: {inputs: ['polygons', 'attributes', 'parameters'], outputs: ['points']},
  // gpu-spatial-analysis/dot-density/gpu-random-points-on-line.ts: counts, parameters, line fields; positions
  GPURandomPointsOnLine: {inputs: ['lines', 'attributes', 'parameters'], outputs: ['points']},
  // analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts: polygon fields; triangle indices
  GPUPolygonTriangulation: {inputs: ['polygons'], outputs: ['render-geometry']},
  // gpu-spatial-analysis/outline-geometry/gpu-outline-geometry.ts: geometryType points, lines, rings; triangle positions
  GPUOutlineGeometry: {inputs: ['points', 'lines', 'polygons'], outputs: ['render-geometry']},
  // gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates.ts: geometry GPUSpatialJoinGeometry; predicate columns, unique positions
  GPUGeometryPredicates: {
    inputs: ['points', 'lines', 'polygons'],
    outputs: ['attributes', 'points']
  },
  // gpu-spatial-analysis/geometry-validity/gpu-geometry-validity.ts: polygons, lines, points; validity mask
  GPUGeometryValidity: {inputs: ['points', 'lines', 'polygons'], outputs: ['attributes']},
  // gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity.ts: polygon rows of one coverage; segmentFlags, isValid
  GPUCoverageValidity: {inputs: ['polygons'], outputs: ['attributes', 'summary']},
  // gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs.ts: sources, targets, parameters; GPULineGeometryOutput
  GPUGreatCircleArcs: {inputs: ['points', 'parameters'], outputs: ['lines']},
  // gpu-spatial-analysis/line-segmentize/gpu-line-chunk.ts: positions, lineOffsets, parameters; line pieces
  GPULineChunk: {inputs: ['lines', 'parameters'], outputs: ['lines']},
  // gpu-spatial-analysis/line-segmentize/gpu-line-segmentize.ts: positions, lineOffsets, parameters; densified lines
  GPULineSegmentize: {inputs: ['lines', 'parameters'], outputs: ['lines']},
  // gpu-spatial-analysis/line-segmentize/gpu-line-smooth.ts: positions, lineOffsets, parameters; smoothed lines
  GPULineSmooth: {inputs: ['lines', 'parameters'], outputs: ['lines']},
  // gpu-spatial-analysis/line-simplification/gpu-line-simplification.ts: positions, importance, timestamps; importance column
  GPULineSimplification: {inputs: ['lines', 'trajectories', 'attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/line-merge/gpu-line-merge.ts: line fields; chains and chain offsets
  GPULineMerge: {inputs: ['lines'], outputs: ['lines', 'attributes']},
  // gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon.ts: lines, polygons; pieces
  GPULineClipByPolygon: {inputs: ['lines', 'polygons'], outputs: ['lines']},
  // gpu-spatial-analysis/line-clip/gpu-shared-paths.ts: left, right lines; shared runs
  GPUSharedPaths: {inputs: ['lines'], outputs: ['lines']},
  // gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip.ts: geometryType lines or polygons, parameters; clipped geometry
  GPURectangleClip: {inputs: ['lines', 'polygons', 'parameters'], outputs: ['lines', 'polygons']},
  // gpu-spatial-analysis/linear-referencing/gpu-linear-referencing.ts: points and polylines; line index, measure, foot point
  GPULinearReferencing: {inputs: ['points', 'lines'], outputs: ['attributes', 'points']},
  // gpu-spatial-analysis/linear-referencing/gpu-line-locate.ts: lines and event measures; positions, tangents
  GPULineLocate: {inputs: ['lines', 'attributes'], outputs: ['points', 'attributes']},
  // gpu-spatial-analysis/segment-intersection/gpu-segment-intersection.ts: left and right segment geometry; intersecting segment pairs
  GPUSegmentIntersection: {inputs: ['lines', 'polygons'], outputs: ['pairs', 'points']},
  // gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve.ts: polygons, labels; dissolved polygons
  GPUCoverageDissolve: {inputs: ['polygons', 'attributes', 'regions'], outputs: ['polygons']},
  // gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification.ts: polygon rings; simplified positions and ring offsets
  GPUCoverageSimplification: {inputs: ['polygons'], outputs: ['polygons']},
  // gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys.ts: positions or minima and maxima, bounds; keys, sorted rows
  GPUHilbertKeys: {inputs: ['points', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates.ts: left, right GPUSpatialJoinGeometry; GPUCompactPairPort
  GPUSpatialJoinCandidates: {inputs: ['points', 'lines', 'polygons'], outputs: ['pairs']},
  // gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts: geometry; bounds and BVH reused by joins (not a table; pairs chosen as the join product)
  GPUSpatialJoinPrepared: {inputs: ['points', 'lines', 'polygons'], outputs: []},
  // gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts: points, polygons; pointFeatureIds, matches
  GPUPointInPolygonJoin: {inputs: ['points', 'polygons'], outputs: ['attributes', 'pairs']},
  // gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join.ts: queries, features; nearestFeatureIds, neighbor tables, matches
  GPUNearestFeatureJoin: {
    inputs: ['points', 'lines', 'polygons'],
    outputs: ['attributes', 'pairs']
  },
  // gpu-spatial-analysis/spatial-join/gpu-buffer-selection.ts: points, features, distance; outputMask, distances
  GPUBufferSelection: {
    inputs: ['points', 'lines', 'polygons', 'attributes'],
    outputs: ['attributes']
  },
  // gpu-spatial-analysis/offset-expansion/gpu-offset-expansion.ts: offsets view; owners and local index
  GPUOffsetExpansion: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/bounds-filter/gpu-bounds-filter.ts: bounds float32x4, parameters; selection mask
  GPUBoundsFilter: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts: positions, parameters, mode knn or radius; spatial weights CSR
  GPUNeighborSearch: {inputs: ['points', 'parameters'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights.ts: polygons; spatial weights CSR
  GPUContiguityWeights: {inputs: ['polygons'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-lattice-weights.ts: width, height, criterion (generator, mask optional); spatial weights CSR
  GPULatticeWeights: {inputs: ['parameters'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights.ts: neighborIds, neighborCounts, neighborDistances; spatial weights CSR
  GPUNearestFeatureWeights: {inputs: ['attributes'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform.ts: weights; transformed weights
  GPUSpatialWeightsTransform: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra.ts: weights; combined weights
  GPUSpatialWeightsAlgebra: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary.ts: weights; statistics, counts
  GPUSpatialWeightsSummary: {inputs: ['spatial-weights'], outputs: ['summary']},
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose.ts: weights; transposed weights
  GPUSpatialWeightsTranspose: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation.ts: weights; flags, invalidCount
  GPUSpatialWeightsValidation: {inputs: ['spatial-weights'], outputs: ['attributes', 'summary']},
  // gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform.ts: weights chunks; transformed weights
  GPUChunkedSpatialWeightsTransform: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra.ts: weights chunks; combined weights
  GPUChunkedSpatialWeightsAlgebra: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose.ts: weights chunks; transposed chunks
  GPUChunkedSpatialWeightsTranspose: {inputs: ['spatial-weights'], outputs: ['spatial-weights']},
  // gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation.ts: weights chunks; flags, invalidCount
  GPUChunkedSpatialWeightsValidation: {
    inputs: ['spatial-weights'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/spatial-weights/gpu-spatial-lag.ts: values, weights; lag column
  GPUSpatialLag: {inputs: ['attributes', 'spatial-weights'], outputs: ['attributes']},
  // gpu-spatial-analysis/map-coloring/gpu-map-coloring.ts: weights, parameters; colors per polygon
  GPUMapColoring: {inputs: ['spatial-weights', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/cell-indexing/gpu-point-to-cell.ts: positions; cells
  GPUPointToCell: {inputs: ['points'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts: cells; boundaries, centers
  GPUCellGeometry: {inputs: ['cells'], outputs: ['polygons', 'points']},
  // gpu-spatial-analysis/cell-topology/gpu-cell-topology.ts: cells; neighbor, parent or child cells
  GPUCellTopology: {inputs: ['cells'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts: cells; compacted or expanded cells
  GPUCellCompaction: {inputs: ['cells'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-topology/gpu-cell-grid-path.ts: origins, destinations; path cells, distances
  GPUCellGridPath: {inputs: ['cells'], outputs: ['cells', 'attributes']},
  // gpu-spatial-analysis/cell-indexing/gpu-cell-measures.ts: cells; areas, perimeters, edge lengths
  GPUCellMeasures: {inputs: ['cells'], outputs: ['attributes']},
  // gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts: polygons; cells, featureIds
  GPUCellCover: {inputs: ['polygons'], outputs: ['cells', 'attributes']},
  // gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation.ts: positions or cells, values; GPUCellTable
  GPUCellAggregation: {inputs: ['points', 'cells', 'attributes'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-aggregation/gpu-cell-rollup.ts: source GPUCellTable; coarser GPUCellTable
  GPUCellRollup: {inputs: ['cells'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-aggregation/gpu-cell-pyramid.ts: resolutions; levels of GPUCellTable
  GPUCellPyramid: {inputs: ['points', 'cells', 'attributes'], outputs: ['cells']},
  // gpu-spatial-analysis/cell-aggregation/gpu-cell-level-selection.ts: levelCounts, activeLevel; draw arguments
  GPUCellLevelSelection: {inputs: ['attributes', 'parameters'], outputs: ['render-geometry']},
  // gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline.ts: cells, groups; boundary edge endpoints
  GPUCellSetOutline: {inputs: ['cells'], outputs: ['lines']},
  // gpu-spatial-analysis/cell-table-compare/gpu-cell-table-compare.ts: before, after GPUCellTable; delta, ratio, zScore
  GPUCellTableCompare: {inputs: ['cells'], outputs: ['cells', 'attributes']},
  // gpu-spatial-analysis/point-density/gpu-point-density.ts: positions, weights, gridSize; per-cell values
  GPUPointDensity: {inputs: ['points', 'attributes', 'parameters'], outputs: ['surface']},
  // gpu-spatial-analysis/point-density/gpu-kernel-density.ts: positions, weights, grid GPURasterMetadata; values
  GPUKernelDensity: {inputs: ['points', 'attributes', 'parameters'], outputs: ['surface']},
  // gpu-spatial-analysis/line-density/gpu-line-density.ts: columns, rows, line fields, parameters; lengths, densities per cell
  GPULineDensity: {inputs: ['lines', 'parameters'], outputs: ['surface', 'attributes']},
  // gpu-spatial-analysis/line-density/gpu-line-length-per-polygon.ts: lines, polygons, weights; lengths per polygon
  GPULineLengthPerPolygon: {inputs: ['lines', 'polygons', 'attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts: positions, values, grid; values
  GPUInverseDistanceWeighting: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // gpu-spatial-analysis/spatial-interpolation/gpu-kriging.ts: positions, values, grid; values, variance
  GPUKriging: {inputs: ['points', 'attributes', 'parameters'], outputs: ['surface']},
  // gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts: values, grid, parameters; mean, min, max
  GPUFocalStatistics: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics.ts: features, positions, values; counts, sums, means
  GPUZonalStatistics: {inputs: ['points', 'polygons', 'attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/region-statistics/gpu-region-mask.ts: positions, region; 0/1 mask
  GPURegionMask: {inputs: ['points', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/region-statistics/gpu-pick-region-mask.ts: picks; 0/1 mask
  GPUPickRegionMask: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/region-statistics/gpu-region-statistics.ts: positions, values, region; summary, selected ids
  GPURegionStatistics: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering.ts: positions, times, parameters; labels, centroids
  GPUSpatialClustering: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'points']
  },
  // gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans.ts: positions, k, parameters; labels, centers
  GPUSpatialKMeans: {inputs: ['points', 'parameters'], outputs: ['attributes', 'points']},
  // gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts: weights, values; componentLabels, treeEdgeFlags
  GPUSpatialWeightsMinimumSpanningTree: {
    inputs: ['spatial-weights', 'attributes'],
    outputs: ['regions', 'attributes']
  },
  // gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions.ts: weights, values, treeEdgeFlags, componentLabels; labels
  GPUSkaterRegions: {inputs: ['spatial-weights', 'attributes', 'regions'], outputs: ['regions']},
  // gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation.ts: values, labels, weights; summary
  GPURegionPartitionEvaluation: {
    inputs: ['attributes', 'regions', 'spatial-weights'],
    outputs: ['summary']
  },
  // gpu-spatial-analysis/group-geometry/gpu-group-geometry.ts: positions, labels, weights; centers, bounds, ellipses
  GPUGroupGeometry: {
    inputs: ['points', 'attributes', 'regions', 'parameters'],
    outputs: ['points', 'attributes']
  },
  // gpu-spatial-analysis/group-geometry/gpu-group-convex-hull.ts: positions, labels; hull vertices and offsets
  GPUGroupConvexHull: {inputs: ['points', 'attributes', 'regions'], outputs: ['polygons']},
  // gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor.ts: neighbors GPUSpatialWeights, parameters; lof, outlier
  GPULocalOutlierFactor: {inputs: ['spatial-weights', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/similar-locations/gpu-similar-locations.ts: values, referenceMask, parameters; ranks, distances
  GPUSimilarLocations: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts: weights, values; summary
  GPUGlobalSpatialStatistics: {inputs: ['attributes', 'spatial-weights'], outputs: ['summary']},
  // gpu-spatial-analysis/spatial-autocorrelation/gpu-hot-spot-analysis.ts: weights, values, parameters; zScores, bins, globalStatistics
  GPUHotSpotAnalysis: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts: weights, values, parameters; zScores, quadrants, globalStatistics
  GPULocalMoran: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test.ts: weights, values, parameters; pseudoPValues, classes
  GPULocalPermutationTest: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes']
  },
  // gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test.ts: weights, values, parameters; summary, referenceDistribution
  GPUGlobalPermutationTest: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['summary']
  },
  // gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary.ts: weights, values, categories; per-row table
  GPUNeighborhoodSummary: {inputs: ['attributes', 'spatial-weights'], outputs: ['attributes']},
  // gpu-spatial-analysis/rate-smoothing/gpu-empirical-bayes-rates.ts: events, populations; smoothedRates, summary
  GPUEmpiricalBayesRates: {inputs: ['attributes'], outputs: ['attributes', 'summary']},
  // gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates.ts: events, populations, weights; spatialRates
  GPUSpatialEmpiricalBayesRates: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['attributes']
  },
  // gpu-spatial-analysis/spatial-autocorrelation/gpu-rate-standardization.ts: events, populations; standardizedRates
  GPUAssuncaoReisStandardization: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment.ts: pValues; adjustedPValues
  GPUMultipleTestingAdjustment: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts: positions, cases, timeBuckets, parameters; clusterIndices, statistics
  GPUSpatialScanStatistic: {
    inputs: ['points', 'attributes', 'time-series', 'parameters'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/space-time-tests/gpu-knox-test.ts: pairs GPUSpatialWeights, times, parameters; summary
  GPUKnoxTest: {inputs: ['spatial-weights', 'attributes', 'parameters'], outputs: ['summary']},
  // gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts: pairs GPUSpatialWeights, times, parameters; summary
  GPUMantelTest: {inputs: ['spatial-weights', 'attributes', 'parameters'], outputs: ['summary']},
  // gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts: predictors, response; coefficients, residuals
  GPUOrdinaryLeastSquares: {inputs: ['attributes'], outputs: ['summary', 'attributes']},
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics.ts: weights, predictors, response, residuals; tests, summary
  GPUSpatialRegressionDiagnostics: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts: weights, predictors, response; table, residuals
  GPUSpatialTwoStageLeastSquares: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm.ts: weights, predictors, response; table, residuals
  GPUSpatialErrorGM: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood.ts: weights, predictors, response; table, residuals
  GPUSpatialLagMaximumLikelihood: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood.ts: weights, predictors, response; table, residuals
  GPUSpatialErrorMaximumLikelihood: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces.ts: weights, maximumPower; powerTraces, powerSums
  GPUSpatialWeightsTraces: {inputs: ['spatial-weights'], outputs: ['summary']},
  // gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts.ts: coefficients, traces; impacts
  GPUSpatialLagImpacts: {inputs: ['summary'], outputs: ['summary']},
  // gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression.ts: positions, predictors, response, parameters; coefficients, localR2
  GPUGeographicallyWeightedRegression: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test.ts: positions, predictors, response, coefficients; table, summary
  GPUGeographicallyWeightedRegressionNonstationarityTest: {
    inputs: ['points', 'attributes'],
    outputs: ['summary']
  },
  // gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment.ts: values, breaks; class per row
  GPUClassAssignment: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit.ts: values, classes; summary
  GPUClassificationFit: {inputs: ['attributes'], outputs: ['summary']},
  // gpu-spatial-analysis/distribution-dynamics/gpu-transition-matrix.ts: classes over periodCount; transition counts
  GPUTransitionMatrix: {inputs: ['time-series'], outputs: ['summary']},
  // gpu-spatial-analysis/distribution-dynamics/gpu-spatial-markov.ts: values, classes, weights, periodCount; transition counts
  GPUSpatialMarkov: {
    inputs: ['time-series', 'spatial-weights', 'attributes', 'parameters'],
    outputs: ['summary']
  },
  // gpu-spatial-analysis/distribution-dynamics/gpu-lisa-markov.ts: values, weights, periodCount; transition counts
  GPULISAMarkov: {inputs: ['time-series', 'spatial-weights', 'parameters'], outputs: ['summary']},
  // gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics.ts: counts; homogeneity, steadyStates
  GPUTransitionDiagnostics: {inputs: ['summary'], outputs: ['summary']},
  // gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots.ts: values, sliceCount, weights; category, trendZ
  GPUEmergingHotSpots: {
    inputs: ['time-series', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts: supply, demand, facilityWeights; accessibility
  GPUCatchmentAccessibility: {inputs: ['attributes', 'spatial-weights'], outputs: ['attributes']},
  // gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts: attractiveness, demandWeights, parameters; tradeArea, probabilities
  GPUHuffTradeAreas: {
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes']
  },
  // gpu-spatial-analysis/segregation/gpu-segregation.ts: groupCounts, scales; indices, local indices
  GPUSegregation: {inputs: ['attributes', 'spatial-weights'], outputs: ['summary', 'attributes']},
  // gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation.ts: grid, sourceZones, targetZones, cellWeights; weights, areas
  GPUArealInterpolation: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['spatial-weights', 'attributes']
  },
  // gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic.ts: grid, zones, totals, parameters; mass, density
  GPUPycnophylactic: {inputs: ['surface', 'attributes', 'parameters'], outputs: ['surface']},
  // gpu-spatial-analysis/trajectory-analysis/gpu-trajectory-metrics.ts: trajectory geometry fields, parameters; track lengths, stops
  GPUTrajectoryMetrics: {inputs: ['trajectories', 'parameters'], outputs: ['attributes', 'points']},
  // gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-playhead.ts: trajectory fields, parameters; currentPositions, speeds
  GPUTrajectoryPlayhead: {
    inputs: ['trajectories', 'parameters'],
    outputs: ['points', 'attributes']
  },
  // gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-resample.ts: trajectory fields, sampleCount; samples
  GPUTrajectoryResample: {inputs: ['trajectories'], outputs: ['trajectories']},
  // gpu-spatial-analysis/trajectory-zones/gpu-zone-events.ts: tracks, zones; events, dwellTimes
  GPUZoneEvents: {inputs: ['trajectories', 'polygons'], outputs: ['pairs', 'attributes']},
  // gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-encounters.ts: samples, distance, bucketCount; encounter output
  GPUTrajectoryEncounters: {inputs: ['trajectories', 'attributes'], outputs: ['pairs']},
  // gpu-spatial-analysis/trajectory-encounters/gpu-trajectory-closest-approach.ts: trajectory fields, distance; pairs
  GPUTrajectoryClosestApproach: {inputs: ['trajectories'], outputs: ['pairs']},
  // gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts: linesA, linesB, pairA, pairB; hausdorff, frechet
  GPUTrackSimilarity: {inputs: ['lines', 'pairs'], outputs: ['attributes']},
  // gpu-spatial-analysis/trajectory-encounters/clock-encounters.ts: tracks resampled to a clock, distance; encounter pairs
  addClockEncounters: {inputs: ['trajectories'], outputs: ['pairs']},
  // gpu-spatial-analysis/recipes/change-of-support-recipe.ts: source and target zones, values; transferred values, weights
  addChangeOfSupportRecipe: {
    inputs: ['polygons', 'attributes'],
    outputs: ['attributes', 'spatial-weights']
  },
  // gpu-spatial-analysis/recipes/cluster-and-outline-recipe.ts: positions; labels, hull polygons, measures
  addClusterAndOutlineRecipe: {inputs: ['points'], outputs: ['attributes', 'polygons']},
  // gpu-spatial-analysis/recipes/fleet-dwell-recipe.ts: tracks, zones; events, dwell per zone
  addFleetDwellZoneEventsRecipe: {
    inputs: ['trajectories', 'polygons'],
    outputs: ['attributes', 'pairs']
  },
  // gpu-spatial-analysis/recipes/hot-spot-analysis-recipe.ts: points with study area or raster; cell z-scores, classes
  addHotSpotAnalysisRecipe: {
    inputs: ['points', 'polygons', 'surface'],
    outputs: ['cells', 'attributes']
  },
  // gpu-spatial-analysis/recipes/period-comparison-recipe.ts: two periods of events; compare table, classes
  addPeriodComparisonRecipe: {inputs: ['points', 'attributes'], outputs: ['cells', 'attributes']},
  // gpu-spatial-analysis/recipes/rate-cluster-map-recipe.ts: counts, populations, polygons; cluster class colors
  addRateClusterMapRecipe: {inputs: ['polygons', 'attributes'], outputs: ['attributes']},
  // gpu-spatial-analysis/recipes/space-time-hot-spots-recipe.ts: timestamped events; category per cell
  addSpaceTimeHotSpotsRecipe: {inputs: ['points', 'time-series'], outputs: ['attributes']},
  // gpu-spatial-analysis/recipes/spatial-regression-recipe.ts: predictors, response, weights; decision, residual map
  addSpatialRegressionRecipe: {
    inputs: ['attributes', 'spatial-weights'],
    outputs: ['summary', 'attributes']
  },
  // gpu-spatial-analysis/recipes/straight-line-catchments-recipe.ts: facilities, value raster; allocation raster, zonal statistics
  addStraightLineCatchmentsRecipe: {
    inputs: ['points', 'surface'],
    outputs: ['surface', 'attributes']
  }
});

/** Freezes every row and its kind lists, so a caller cannot change what later calls see. */
function freezeSupplementRows(
  rows: Record<string, GPUAnalysisOperatorKinds>
): Readonly<Record<string, GPUAnalysisOperatorKinds>> {
  for (const row of Object.values(rows)) {
    Object.freeze(row.inputs);
    Object.freeze(row.outputs);
    Object.freeze(row);
  }
  return Object.freeze(rows);
}

function validateDataKind(functionName: string, argumentName: string, value: unknown): void {
  if (!GPU_SPATIAL_ANALYSIS_DATA_KINDS.includes(value as GPUSpatialAnalysisDataKind)) {
    throw new Error(
      `${functionName} ${argumentName} ${JSON.stringify(value)} is not a data kind; accepted: ${GPU_SPATIAL_ANALYSIS_DATA_KINDS.join(', ')}`
    );
  }
}

/**
 * The kinds of one operator, resolved one side at a time: the catalogue record's array when it is
 * not empty, otherwise the supplement row's array, otherwise empty. The result says where each
 * side came from, and `source` is `mixed` when the two sides differ (for example a record that
 * lists outputs only, with the inputs taken from the row). A supplement row may declare empty
 * outputs for the operators in {@link GPU_ANALYSIS_HANDLE_OPERATORS}; that counts as a known answer
 * (`supplement`). An operator the catalogue describes on neither side and no row covers is
 * `unknown` with empty arrays, never an error: operators are added to the catalogue often.
 */
export function resolveGPUAnalysisOperatorKinds(
  operator: GPUAnalysisOperatorKindRecord
): GPUAnalysisResolvedOperatorKinds {
  const row = Object.hasOwn(GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS, operator.exportName)
    ? GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS[operator.exportName]
    : undefined;
  const inputs = resolveSide(operator.inputs, row?.inputs);
  const outputs = resolveSide(operator.outputs, row?.outputs);
  const known = new Set([inputs.source, outputs.source].filter(source => source !== 'unknown'));
  const source: GPUAnalysisOperatorKindSource =
    known.size === 0 ? 'unknown' : known.size === 1 ? [...known][0] : 'mixed';
  return {
    inputs: inputs.kinds,
    outputs: outputs.kinds,
    source,
    inputsSource: inputs.source,
    outputsSource: outputs.source
  };
}

function resolveSide(
  catalogueKinds: readonly GPUSpatialAnalysisDataKind[],
  supplementKinds: readonly GPUSpatialAnalysisDataKind[] | undefined
): {kinds: readonly GPUSpatialAnalysisDataKind[]; source: GPUAnalysisOperatorKindSideSource} {
  if (catalogueKinds.length > 0) return {kinds: catalogueKinds, source: 'catalogue'};
  if (supplementKinds) return {kinds: supplementKinds, source: 'supplement'};
  return {kinds: [], source: 'unknown'};
}

type IndexedOperator = {
  operator: GPUAnalysisOperator;
  kinds: GPUAnalysisResolvedOperatorKinds;
  /** Primary question is `prepare-data`: a step that gets data ready rather than answers anything. */
  isPlumbing: boolean;
  /** A workflow recipe (`add...Recipe`), which chains several operators already. */
  isRecipe: boolean;
};

type OperatorIndex = {
  /** The catalogue array this index was built from; a different array means a rebuild. */
  catalogue: readonly GPUAnalysisOperator[];
  /** Non-helper operators in catalogue order. */
  inCatalogueOrder: readonly IndexedOperator[];
  /** Non-helper operators sorted by name. */
  byName: readonly IndexedOperator[];
  indexedByName: ReadonlyMap<string, IndexedOperator>;
  /** For each input kind and then output kind, the operators making that step, sorted by name. */
  transitions: ReadonlyMap<
    GPUSpatialAnalysisDataKind,
    ReadonlyMap<GPUSpatialAnalysisDataKind, readonly IndexedOperator[]>
  >;
};

let cachedIndex: OperatorIndex | undefined;

function compareNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/**
 * The operator index, rebuilt only when the catalogue array is a different one (the catalogue array
 * is replaced when the catalogue changes), so repeated queries do not resolve and sort again.
 */
function getOperatorIndex(): OperatorIndex {
  const catalogue = getGPUAnalysisOperatorCatalog();
  if (cachedIndex?.catalogue === catalogue) return cachedIndex;
  const inCatalogueOrder = catalogue
    .filter(operator => operator.role !== 'helper')
    .map(operator => ({
      operator,
      kinds: resolveGPUAnalysisOperatorKinds(operator),
      isPlumbing: getGPUAnalysisOperatorQuestions(operator)[0] === 'prepare-data',
      isRecipe: /^add[A-Z]/.test(operator.exportName)
    }));
  const byName = [...inCatalogueOrder].sort((left, right) =>
    compareNames(left.operator.exportName, right.operator.exportName)
  );
  const transitions = new Map<
    GPUSpatialAnalysisDataKind,
    Map<GPUSpatialAnalysisDataKind, IndexedOperator[]>
  >();
  for (const indexed of byName) {
    for (const inputKind of indexed.kinds.inputs) {
      const byOutput = transitions.get(inputKind) ?? new Map();
      transitions.set(inputKind, byOutput);
      for (const outputKind of indexed.kinds.outputs) {
        const operators = byOutput.get(outputKind) ?? [];
        byOutput.set(outputKind, operators);
        operators.push(indexed);
      }
    }
  }
  cachedIndex = {
    catalogue,
    inCatalogueOrder,
    byName,
    indexedByName: new Map(byName.map(indexed => [indexed.operator.exportName, indexed])),
    transitions
  };
  return cachedIndex;
}

/**
 * Operators that take `input` and give `output`, by resolved kinds (an operator matches when its
 * input list contains `input` and its output list contains `output`). With neither, every operator
 * is returned. Helpers are left out. Catalogue order. Throws for a kind that is not an ontology kind.
 *
 * Limit of the answer: kinds are only as consistent as the catalogue. Label-like results are
 * `attributes` for cluster labels, masks and colours (as OPTICS and HDBSCAN are in the catalogue)
 * and `regions` only for contiguous regionalization and segmentation, so a query by output
 * `regions` is deliberately narrow and misses cluster labels; ask for `attributes` as well.
 */
export function queryGPUAnalysisOperatorsByKind(query: {
  input?: GPUSpatialAnalysisDataKind;
  output?: GPUSpatialAnalysisDataKind;
}): readonly GPUAnalysisOperator[] {
  if (query.input !== undefined)
    validateDataKind('queryGPUAnalysisOperatorsByKind', 'input', query.input);
  if (query.output !== undefined)
    validateDataKind('queryGPUAnalysisOperatorsByKind', 'output', query.output);
  return getOperatorIndex()
    .inCatalogueOrder.filter(
      ({kinds}) =>
        (query.input === undefined || kinds.inputs.includes(query.input)) &&
        (query.output === undefined || kinds.outputs.includes(query.output))
    )
    .map(({operator}) => operator);
}

/**
 * Names of the operators that can take data of `outputKind` as an input, sorted by name: "what can
 * I do next with this result". Uses resolved kinds, so it includes supplemented operators. Throws
 * for a kind that is not an ontology kind.
 */
export function getGPUAnalysisNextOperators(
  outputKind: GPUSpatialAnalysisDataKind
): readonly string[] {
  validateDataKind('getGPUAnalysisNextOperators', 'outputKind', outputKind);
  return getOperatorIndex()
    .byName.filter(({kinds}) => kinds.inputs.includes(outputKind))
    .map(({operator}) => operator.exportName);
}

/** Result of {@link findGPUAnalysisOperatorPaths}. */
export type GPUAnalysisOperatorPathResult = {
  /** Chains of operator names, best first, at most `maximumPaths`. */
  paths: readonly (readonly string[])[];
  /** True when more chains were found (or may exist) than `paths` holds. */
  truncated: boolean;
  /**
   * Number of distinct chains (of 1 to `maximumSteps` operators) the search counted. The search
   * stops counting at {@link GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES}; a `totalFound` equal to that
   * bound is a lower bound, and then `truncated` is true even when `paths` is full.
   */
  totalFound: number;
};

/**
 * Chains of operators that turn data of `fromKind` into data of `toKind`, as lists of operator
 * names. A chain has between 1 and `maximumSteps` operators (capped at
 * {@link GPU_ANALYSIS_MAXIMUM_PATH_STEPS}), never visits a kind twice and never repeats an
 * operator. Shorter chains come first (breadth-first: all chains of one length are found before the
 * next length is searched). Within a length, chains with fewer plumbing operators (primary question
 * `prepare-data`) come first, then chains with more recipes (`add...Recipe`, which already bundle
 * several steps), then operator names, so a capped answer keeps the useful chains. `maximumPaths`
 * (default {@link GPU_ANALYSIS_MAXIMUM_PATH_COUNT}, at most
 * {@link GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT}) bounds `paths`; `truncated` and `totalFound` say
 * what the cap left out. `fromKind` equal to `toKind`, and an unreachable pair, give no paths. A
 * kind that is not an ontology kind, or a `maximumSteps` or `maximumPaths` that is not an integer
 * in range, throws.
 *
 * A path is a discovery hint, not a guarantee that the chain executes. Kinds do not carry the
 * companions an operator needs (spatial weights next to a statistic, a time column, a mask), nor
 * layouts, capacities or coordinate models, so a chain can still need extra inputs or an adapter
 * before it runs. `attributes` is a hub kind: most operators take or give a column, so many
 * kind-valid chains through it (a statistic fed by an unrelated column) are meaningless; prefer
 * the chains that stay on spatial kinds.
 */
export function findGPUAnalysisOperatorPaths(
  fromKind: GPUSpatialAnalysisDataKind,
  toKind: GPUSpatialAnalysisDataKind,
  maximumSteps: number,
  maximumPaths: number = GPU_ANALYSIS_MAXIMUM_PATH_COUNT
): GPUAnalysisOperatorPathResult {
  validateDataKind('findGPUAnalysisOperatorPaths', 'fromKind', fromKind);
  validateDataKind('findGPUAnalysisOperatorPaths', 'toKind', toKind);
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1) {
    throw new Error(
      `findGPUAnalysisOperatorPaths maximumSteps ${String(maximumSteps)} must be an integer from 1 to ${GPU_ANALYSIS_MAXIMUM_PATH_STEPS} (larger values are capped)`
    );
  }
  if (
    !Number.isInteger(maximumPaths) ||
    maximumPaths < 1 ||
    maximumPaths > GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT
  ) {
    throw new Error(
      `findGPUAnalysisOperatorPaths maximumPaths ${String(maximumPaths)} must be an integer from 1 to ${GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT}`
    );
  }
  const empty: GPUAnalysisOperatorPathResult = {paths: [], truncated: false, totalFound: 0};
  if (fromKind === toKind) return empty;
  const stepLimit = Math.min(maximumSteps, GPU_ANALYSIS_MAXIMUM_PATH_STEPS);
  const index = getOperatorIndex();
  const distances = getDistancesToKind(index, toKind);
  const fromDistance = distances.get(fromKind);
  if (fromDistance === undefined || fromDistance > stepLimit) return empty;

  // Enumerate rank buckets before names: the cutoff can only discard worse-ranked chains.
  const found = new Map<string, readonly string[]>();
  for (let length = fromDistance; length <= stepLimit; length++) {
    for (let plumbing = 0; plumbing <= length; plumbing++) {
      for (let recipes = length; recipes >= 0; recipes--) {
        collectChains(
          index,
          distances,
          [{kind: fromKind, visitedKinds: new Set([fromKind])}],
          toKind,
          length,
          plumbing,
          recipes,
          [],
          found
        );
        if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
      }
      if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
    }
    if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
  }
  const paths = [...found.values()].slice(0, maximumPaths);
  return {
    paths,
    truncated: found.size > paths.length || found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES,
    totalFound: found.size
  };
}

/**
 * Fewest steps from each kind to `toKind` over the kind graph, ignoring the visited-kind rule (so
 * it never overstates the distance). Kinds that cannot reach `toKind` are absent. Used to cut
 * branches that cannot finish in the steps left, which bounds the search for an unreachable pair
 * to a single pass over 15 kinds.
 */
function getDistancesToKind(
  index: OperatorIndex,
  toKind: GPUSpatialAnalysisDataKind
): Map<GPUSpatialAnalysisDataKind, number> {
  const distances = new Map<GPUSpatialAnalysisDataKind, number>([[toKind, 0]]);
  for (let round = 0; round < GPU_SPATIAL_ANALYSIS_DATA_KINDS.length; round++) {
    let changed = false;
    for (const [inputKind, byOutput] of index.transitions) {
      for (const outputKind of byOutput.keys()) {
        const next = distances.get(outputKind);
        if (next !== undefined && next + 1 < (distances.get(inputKind) ?? Infinity)) {
          distances.set(inputKind, next + 1);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return distances;
}

/** A kind route for the current operator prefix; several routes can name the same chain. */
type ChainState = {
  kind: GPUSpatialAnalysisDataKind;
  visitedKinds: ReadonlySet<GPUSpatialAnalysisDataKind>;
};

/**
 * Adds one rank bucket in name order. All kind routes for an operator prefix travel together,
 * so a chain with multiple possible intermediate kinds is emitted once, in its proper order.
 */
function collectChains(
  index: OperatorIndex,
  distances: ReadonlyMap<GPUSpatialAnalysisDataKind, number>,
  states: readonly ChainState[],
  toKind: GPUSpatialAnalysisDataKind,
  remainingSteps: number,
  remainingPlumbing: number,
  remainingRecipes: number,
  chain: string[],
  found: Map<string, readonly string[]>
): void {
  for (const indexed of index.byName) {
    if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) return;
    const plumbing = remainingPlumbing - Number(indexed.isPlumbing);
    const recipes = remainingRecipes - Number(indexed.isRecipe);
    const nextSteps = remainingSteps - 1;
    if (plumbing < 0 || plumbing > nextSteps || recipes < 0 || recipes > nextSteps) continue;
    const name = indexed.operator.exportName;
    if (chain.includes(name)) continue;
    const nextStates: ChainState[] = [];
    for (const state of states) {
      if (!indexed.kinds.inputs.includes(state.kind)) continue;
      for (const nextKind of indexed.kinds.outputs) {
        if (state.visitedKinds.has(nextKind)) continue;
        if (nextSteps === 0 ? nextKind !== toKind : nextKind === toKind) continue;
        const nextDistance = distances.get(nextKind);
        if (nextDistance === undefined || nextDistance > nextSteps) continue;
        nextStates.push({kind: nextKind, visitedKinds: new Set([...state.visitedKinds, nextKind])});
      }
    }
    if (nextStates.length === 0) continue;
    chain.push(name);
    if (nextSteps === 0) {
      found.set(chain.join('\u0000'), [...chain]);
    } else {
      collectChains(
        index,
        distances,
        nextStates,
        toKind,
        nextSteps,
        plumbing,
        recipes,
        chain,
        found
      );
    }
    chain.pop();
  }
}
