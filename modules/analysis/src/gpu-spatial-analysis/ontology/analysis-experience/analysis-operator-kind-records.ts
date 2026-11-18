// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';

/** Input and output data kinds of one operator. */
export type GPUAnalysisOperatorKinds = {
  inputs: readonly GPUSpatialAnalysisDataKind[];
  outputs: readonly GPUSpatialAnalysisDataKind[];
};

/** An intentional empty side, with the source-level reason it is not a dataset port. */
export type GPUAnalysisOperatorEmptyKindException = {
  role: 'helper' | 'handle';
  inputs?: string;
  outputs?: string;
};

/**
 * Only these named sides may be empty. A helper role alone never exempts an operator from the
 * completeness gate. Remove an exception when the operator gains a port on that side.
 */
export const GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS: Readonly<
  Record<string, GPUAnalysisOperatorEmptyKindException>
> = Object.freeze({
  GPUParameterBuffer: Object.freeze({
    role: 'helper',
    inputs:
      'analysis-core/kernels/gpu-contributor-utils.ts: fills a parameters view from host values; it reads no analysis data port.'
  }),
  GPUSpatialJoinPrepared: Object.freeze({
    role: 'handle',
    outputs:
      'gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts: exposes reusable bounds and BVH storage through a handle, with no output port.'
  })
});

/** Operators that expose a reusable handle rather than a data output. */
export const GPU_ANALYSIS_HANDLE_OPERATORS: readonly string[] = Object.freeze(
  Object.entries(GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS)
    .filter(([, exception]) => exception.role === 'handle' && exception.outputs)
    .map(([exportName]) => exportName)
);

/**
 * Authoritative input and output kinds for individual operators, never capability-family unions.
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
 * These static records populate the raw catalogue as well as the resolver. This module imports
 * types only: catalogue construction must never import the resolver, which queries the catalogue.
 * The completeness spec rejects absent operators, invalid kinds and unexplained empty sides.
 */
export const GPU_ANALYSIS_OPERATOR_KIND_RECORDS: Readonly<
  Record<string, GPUAnalysisOperatorKinds>
> = freezeKindRecords({
  // dataframe-analysis/forest-inference/gpu-forest-inference.ts
  GPUForestInference: {inputs: ['attributes', 'surface', 'parameters'], outputs: ['attributes']},
  // computation-record/computation-recorder.ts
  GPUComputationRecorder: {inputs: ['parameters'], outputs: ['summary']},
  // computation-record/gpu-snapshot-arena.ts
  GPUSnapshotArena: {inputs: ['parameters'], outputs: ['summary']},
  // computation-record/gpu-snapshot-tap.ts
  GPUSnapshotTap: {inputs: ['summary'], outputs: ['summary']},
  // computation-record/gpu-transfer-mask.ts
  GPUTransferMask: {inputs: ['attributes', 'pairs'], outputs: ['attributes']},
  // gpu-network/network-noding/gpu-network-noding.ts
  GPUNetworkNoding: {inputs: ['lines', 'parameters'], outputs: ['network', 'lines', 'attributes']},
  // gpu-network/network-accessibility/gpu-network-snapping.ts
  GPUNetworkSnapping: {
    inputs: ['points', 'network', 'parameters'],
    outputs: ['points', 'attributes']
  },
  // gpu-network/network-line-graph/gpu-network-line-graph.ts
  GPUNetworkLineGraph: {inputs: ['network', 'attributes', 'parameters'], outputs: ['network']},
  // gpu-network/network-subgraph-filter/gpu-network-subgraph-filter.ts
  GPUNetworkSubgraphFilter: {
    inputs: ['network', 'attributes', 'time-series', 'parameters'],
    outputs: ['network', 'attributes', 'summary']
  },
  // gpu-network/network-coarsening/gpu-network-coarsening.ts
  GPUNetworkCoarsening: {
    inputs: ['network', 'attributes'],
    outputs: ['network', 'points', 'attributes', 'summary']
  },
  // gpu-network/network-reachability/gpu-network-reachability.ts
  GPUNetworkReachability: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-analysis/gpu-network-path-extraction.ts
  GPUNetworkPathExtraction: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-network/k-shortest-paths/gpu-k-shortest-paths.ts
  GPUKShortestPaths: {inputs: ['network', 'attributes'], outputs: ['attributes', 'summary']},
  // gpu-network/alternative-routes/gpu-alternative-routes.ts
  GPUAlternativeRoutes: {inputs: ['network', 'attributes'], outputs: ['attributes', 'summary']},
  // gpu-network/network-optimization/gpu-network-point-routing.ts
  GPUNetworkPointRouting: {inputs: ['network', 'attributes'], outputs: ['attributes']},
  // gpu-network/network-optimization/gpu-network-time-routing.ts
  GPUNetworkTimeRouting: {
    inputs: ['network', 'time-series', 'attributes'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/recipes/network-scenarios-recipe.ts
  addNetworkDisruptionRecipe: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-accessibility/gpu-network-cost-matrix.ts
  GPUNetworkCostMatrix: {inputs: ['network', 'attributes', 'parameters'], outputs: ['attributes']},
  // gpu-network/network-cost-table/gpu-network-cost-table.ts
  GPUNetworkCostTable: {inputs: ['network', 'parameters'], outputs: ['attributes']},
  // gpu-network/network-accessibility/gpu-network-accessibility.ts
  GPUNetworkAccessibility: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // gpu-network/network-cost-weights/gpu-network-cost-weights.ts
  GPUNetworkCostWeights: {inputs: ['attributes', 'parameters'], outputs: ['spatial-weights']},
  // gpu-network/network-analysis/gpu-network-service-areas.ts
  GPUNetworkServiceAreas: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-isochrones/gpu-network-isochrones.ts
  GPUNetworkIsochrones: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['surface', 'cells', 'polygons', 'render-geometry']
  },
  // gpu-network/network-optimization/gpu-network-rational-access.ts
  GPUNetworkRationalAccess: {
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/recipes/drive-time-catchment-recipe.ts
  addDriveTimeCatchmentRecipe: {
    inputs: ['network', 'points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary', 'polygons']
  },
  // gpu-network/network-structure/gpu-network-strongly-connected-components.ts
  GPUNetworkStronglyConnectedComponents: {inputs: ['network'], outputs: ['attributes']},
  // gpu-network/network-structure/gpu-network-structure.ts
  GPUNetworkStructure: {inputs: ['network'], outputs: ['attributes']},
  // gpu-network/network-statistics/gpu-network-statistics.ts
  GPUNetworkStatistics: {inputs: ['network', 'attributes', 'parameters'], outputs: ['summary']},
  // gpu-network/network-analysis/gpu-network-analytics-columns.ts
  GPUNetworkAnalyticsColumns: {inputs: ['network'], outputs: ['attributes', 'summary']},
  // gpu-network/network-analysis/gpu-network-neighborhood.ts
  GPUNetworkNeighborhood: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // gpu-network/network-centrality/gpu-network-centrality.ts
  GPUNetworkCentrality: {inputs: ['network', 'attributes', 'parameters'], outputs: ['attributes']},
  GPUNetworkCentralityScheduleStep: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // gpu-network/network-optimization/gpu-network-louvain.ts
  GPUNetworkLouvain: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-optimization/gpu-network-flow.ts
  GPUNetworkFlow: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-optimization/gpu-network-facility-optimization.ts
  GPUNetworkFacilityOptimization: {
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/recipes/network-scenarios-recipe.ts
  addFacilityScenarioRecipe: {
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-k-function/gpu-network-k-function.ts
  GPUNetworkKFunction: {inputs: ['points', 'network', 'parameters'], outputs: ['summary']},
  // gpu-network/gpu-network-inhomogeneous-k.ts
  GPUNetworkInhomogeneousK: {inputs: ['network', 'attributes', 'parameters'], outputs: ['summary']},
  // gpu-network/gpu-network-kernel-density.ts
  GPUNetworkKernelDensity: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-network/map-matching/gpu-map-matching.ts
  GPUMapMatching: {
    inputs: ['trajectories', 'network', 'parameters'],
    outputs: ['points', 'attributes', 'summary']
  },
  // gpu-network/flow-aggregation/gpu-map-matched-edge-volumes.ts
  GPUMapMatchedEdgeVolumes: {inputs: ['attributes'], outputs: ['pairs', 'attributes']},
  // gpu-network/network-time-geography/gpu-network-potential-path-area.ts
  GPUNetworkPotentialPathArea: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // gpu-network/network-time-geography/gpu-network-travel-time-percentiles.ts
  GPUNetworkTravelTimePercentiles: {
    inputs: ['network', 'time-series', 'parameters'],
    outputs: ['attributes']
  },
  // gpu-network/flow-aggregation/gpu-flow-aggregation.ts
  GPUFlowAggregation: {
    inputs: ['points', 'attributes', 'time-series', 'parameters'],
    outputs: ['pairs', 'attributes']
  },
  // gpu-network/flow-aggregation/gpu-radiation-flow-model.ts
  GPURadiationFlowModel: {inputs: ['points', 'attributes'], outputs: ['attributes']},
  // gpu-network/flow-aggregation/gpu-kernel-flow-smoothing.ts
  GPUKernelFlowSmoothing: {inputs: ['points', 'attributes', 'parameters'], outputs: ['attributes']},
  // gpu-network/recipes/track-quality-recipe.ts
  addTrackQualityRecipe: {
    inputs: ['trajectories', 'network', 'parameters'],
    outputs: ['points', 'attributes', 'summary']
  },
  // gpu-network/adjacency-matrix/gpu-adjacency-matrix.ts
  GPUAdjacencyMatrix: {
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['surface', 'summary']
  },
  // gpu-network/adjacency-matrix/adjacency-matrix-order.ts
  GPUAdjacencyMatrixOrder: {inputs: ['attributes'], outputs: ['attributes']},
  // gpu-network/edge-bundling/gpu-edge-bundling.ts
  GPUEdgeBundling: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['lines', 'render-geometry']
  },
  // gpu-terrain/hydrology/gpu-terrain-flow.ts
  GPUTerrainFlow: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-processing/gpu-terrain-flow-fractions.ts
  GPUTerrainFlowFractions: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts
  GPUTerrainBranchingFlow: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts
  GPUTerrainTiledFlow: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/hydrology/gpu-terrain-flow-distance.ts
  GPUTerrainFlowDistance: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/hydrology/gpu-terrain-watersheds.ts
  GPUTerrainWatersheds: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/hydrology/gpu-terrain-stream-order.ts
  GPUTerrainStreamOrder: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/hydrology/gpu-terrain-stream-network.ts
  GPUTerrainStreamNetwork: {inputs: ['surface'], outputs: ['lines', 'attributes']},
  // gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts
  GPUTerrainHeightAboveDrainage: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/hydrology/gpu-terrain-hydrologic-indices.ts
  GPUTerrainHydrologicIndices: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts
  GPUTerrainCatchmentSummary: {inputs: ['surface'], outputs: ['summary']},
  // gpu-terrain/terrain-processing/gpu-terrain-breach.ts
  GPUTerrainBreach: {
    inputs: ['surface', 'attributes'],
    outputs: ['surface', 'attributes', 'summary']
  },
  // gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts
  GPUTerrainDepressionHierarchy: {
    inputs: ['surface', 'attributes'],
    outputs: ['surface', 'attributes']
  },
  // gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts
  GPUTerrainSpikeRepair: {inputs: ['surface'], outputs: ['surface', 'summary']},
  // gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts
  GPUTerrainRGBDecode: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts
  GPUTerrainDEMCoregistration: {inputs: ['surface'], outputs: ['surface', 'summary']},
  // gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts
  GPUTerrainDerivatives: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts
  GPUTerrainCurvature: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/topographic-position/gpu-terrain-ruggedness.ts
  GPUTerrainRuggedness: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness.ts
  GPUTerrainVectorRuggedness: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-analysis/gpu-terrain-contours.ts
  GPUTerrainContours: {inputs: ['surface', 'parameters'], outputs: ['lines']},
  // gpu-terrain/topographic-position/gpu-terrain-topographic-position.ts
  GPUTerrainTopographicPosition: {inputs: ['surface'], outputs: ['surface']},
  // gpu-terrain/topographic-position/gpu-terrain-weiss-landforms.ts
  GPUTerrainWeissLandforms: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/geomorphons/gpu-geomorphons.ts
  GPUGeomorphons: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-features/gpu-terrain-critical-points.ts
  GPUTerrainCriticalPoints: {inputs: ['surface'], outputs: ['surface', 'summary']},
  // gpu-terrain/terrain-features/gpu-terrain-summits.ts
  GPUTerrainSummits: {inputs: ['surface', 'parameters'], outputs: ['surface', 'attributes']},
  // gpu-terrain/terrain-features/gpu-terrain-peak-snap.ts
  GPUTerrainPeakSnap: {
    inputs: ['surface', 'points', 'attributes', 'parameters'],
    outputs: ['points', 'attributes']
  },
  // gpu-terrain/terrain-features/gpu-profile-peaks.ts
  GPUProfilePeaks: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // gpu-terrain/terrain-analysis/gpu-terrain-viewshed.ts
  GPUTerrainViewshed: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight.ts
  GPUTerrainLineOfSight: {inputs: ['surface', 'points', 'parameters'], outputs: ['attributes']},
  // gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed.ts
  GPUTerrainCumulativeViewshed: {inputs: ['surface', 'points', 'parameters'], outputs: ['surface']},
  // gpu-terrain/point-horizon/gpu-point-horizon-profile.ts
  GPUPointHorizonProfile: {inputs: ['surface', 'points', 'parameters'], outputs: ['attributes']},
  // gpu-terrain/point-horizon/gpu-point-horizon-visibility.ts
  GPUPointHorizonVisibility: {inputs: ['surface', 'points', 'parameters'], outputs: ['attributes']},
  // gpu-terrain/terrain-illumination/gpu-solar-position.ts
  GPUSolarPosition: {inputs: ['points', 'parameters'], outputs: ['attributes']},
  // gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts
  GPUTerrainHorizon: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow.ts
  GPUTerrainCastShadow: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-illumination/gpu-solar-shadow-mask.ts
  GPUSolarShadowMask: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-illumination/gpu-solar-irradiance.ts
  GPUSolarIrradiance: {inputs: ['surface', 'attributes', 'parameters'], outputs: ['surface']},
  // gpu-terrain/relief-visualization/gpu-simple-local-relief.ts
  GPUSimpleLocalRelief: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts
  GPUMultiScaleRelief: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/relief-visualization/gpu-local-dominance.ts
  GPULocalDominance: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/relief-visualization/gpu-relief-blend.ts
  GPUReliefBlend: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-illumination/gpu-relief-shading.ts
  GPUReliefShading: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-illumination/gpu-texture-shading.ts
  GPUTextureShading: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts
  GPUTerrainFlowField: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-algebra/gpu-raster-arithmetic.ts
  GPURasterArithmetic: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-algebra/gpu-raster-conditional.ts
  GPURasterConditional: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-algebra/gpu-raster-reclassify.ts
  GPURasterReclassify: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['surface', 'summary']
  },
  // raster-analysis/raster-algebra/gpu-weighted-overlay.ts
  GPUWeightedOverlay: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['surface', 'summary']
  },
  // raster-analysis/raster-algebra/gpu-raster-cell-statistics.ts
  GPURasterCellStatistics: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization.ts
  GPURasterContrastEqualization: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-stretch/gpu-raster-stretch.ts
  GPURasterStretch: {
    inputs: ['surface', 'regions', 'attributes', 'parameters'],
    outputs: ['surface', 'attributes', 'summary']
  },
  // raster-analysis/gpu-raster-spectral-indices.ts
  GPURasterNDWI: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-spectral-indices.ts
  GPURasterNBR: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-spectral-indices.ts
  GPURasterEVI: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-spectral-indices.ts
  GPURasterSAVI: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-spectral-unmixing.ts
  GPURasterSpectralUnmixing: {inputs: ['surface', 'attributes'], outputs: ['surface']},
  // raster-analysis/focal-pairs/gpu-raster-focal-pairs.ts
  GPURasterFocalPairs: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-rank-filter.ts
  GPURasterRankFilter: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-texture.ts
  GPURasterTexture: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-fill-nodata.ts
  GPURasterFillNoData: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-reconstruction.ts
  GPURasterReconstruction: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-reconstruction.ts
  GPURasterFillHoles: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-skeleton.ts
  GPURasterSkeleton: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-skeleton.ts
  GPURasterMedialAxis: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/gpu-raster-watershed-segmentation.ts
  GPURasterWatershedSegmentation: {inputs: ['surface', 'regions'], outputs: ['regions']},
  // raster-analysis/remote-sensing/gpu-raster-region-graph.ts
  GPURasterRegionGraph: {inputs: ['surface', 'regions'], outputs: ['network', 'regions']},
  // raster-analysis/raster-patches/gpu-raster-patch-metrics.ts
  GPURasterPatchMetrics: {inputs: ['surface', 'regions', 'parameters'], outputs: ['attributes']},
  // raster-analysis/raster-patches/gpu-raster-sieve.ts
  GPURasterSieve: {
    inputs: ['surface', 'regions', 'parameters'],
    outputs: ['surface', 'regions', 'summary']
  },
  // raster-analysis/distance-field/gpu-distance-field.ts
  GPUDistanceField: {
    inputs: ['surface', 'points', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/cost-distance/gpu-cost-distance.ts
  GPUCostDistance: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['surface', 'summary']
  },
  // raster-analysis/cost-distance/gpu-cost-distance-path.ts
  GPUCostDistancePath: {inputs: ['surface', 'attributes'], outputs: ['attributes']},
  // raster-analysis/gpu-raster-walking-cost.ts
  GPURasterWalkingCost: {inputs: ['surface', 'attributes', 'parameters'], outputs: ['surface']},
  // raster-analysis/polygon-rasterization/gpu-polygon-rasterization.ts
  GPUPolygonRasterization: {inputs: ['polygons', 'parameters'], outputs: ['surface']},
  // raster-analysis/polygon-rasterization/gpu-raster-join.ts
  GPURasterJoin: {
    inputs: ['surface', 'points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts
  GPURasterZonalStatistics: {
    inputs: ['surface', 'regions', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // raster-analysis/gpu-raster-fractional-coverage.ts
  GPURasterFractionalCoverage: {
    inputs: ['polygons', 'parameters'],
    outputs: ['pairs', 'attributes']
  },
  // raster-analysis/gpu-raster-fractional-coverage.ts
  GPURasterFractionalZonalStatistics: {
    inputs: ['surface', 'pairs', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // raster-analysis/gpu-raster-fractional-categorical-statistics.ts
  GPURasterFractionalCategoricalStatistics: {
    inputs: ['surface', 'pairs', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // raster-analysis/gpu-raster-fractional-zonal-distribution.ts
  GPURasterFractionalZonalDistribution: {
    inputs: ['surface', 'pairs', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // raster-analysis/isolines/gpu-isolines.ts
  GPUIsolines: {inputs: ['surface', 'parameters'], outputs: ['lines', 'attributes']},
  // raster-analysis/isolines/gpu-isobands.ts
  GPUIsobands: {
    inputs: ['surface', 'parameters'],
    outputs: ['surface', 'lines', 'render-geometry']
  },
  // raster-analysis/isolines/gpu-isoband-rings.ts
  GPUIsobandRings: {inputs: ['surface', 'parameters'], outputs: ['polygons', 'attributes']},
  // raster-analysis/particle-advection/gpu-particle-advection.ts
  GPUParticleAdvection: {
    inputs: ['surface', 'points', 'parameters'],
    outputs: ['points', 'lines', 'attributes']
  },
  // raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map.ts
  GPULagrangianFlowMap: {
    inputs: ['surface', 'points', 'regions', 'parameters'],
    outputs: ['points', 'attributes']
  },
  // raster-analysis/lagrangian-diagnostics/gpu-scalar-field-ridges.ts
  GPUScalarFieldRidges: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts
  GPUFiniteTimeLyapunovExponent: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/flow-texture/gpu-line-integral-convolution.ts
  GPULineIntegralConvolution: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/flow-texture/gpu-streamlines.ts
  GPUStreamlines: {inputs: ['surface', 'parameters'], outputs: ['lines', 'attributes']},
  // raster-analysis/gpu-raster-reprojection.ts
  GPURasterReprojection: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/gpu-raster-tile-reprojection.ts
  GPURasterTileReprojection: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-conservative-remapping.ts
  GPURasterConservativeRemapping: {inputs: ['surface', 'spatial-weights'], outputs: ['surface']},
  // raster-analysis/spectral-filter/gpu-raster-spectral-filter.ts
  GPURasterSpectralFilter: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-registration.ts
  GPURasterRegistration: {inputs: ['surface'], outputs: ['summary']},
  // raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid.ts
  GPURasterExtremaPyramid: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/raster-sampling/gpu-raster-sampling.ts
  GPURasterSampling: {inputs: ['surface', 'points', 'parameters'], outputs: ['attributes']},
  // raster-analysis/raster-sampling/gpu-raster-profile.ts
  GPURasterProfile: {
    inputs: ['surface', 'lines', 'parameters'],
    outputs: ['points', 'attributes', 'summary']
  },
  // raster-analysis/change-detection/gpu-change-detection.ts
  GPUChangeDetection: {inputs: ['surface', 'time-series', 'parameters'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-temporal.ts
  GPURasterTemporal: {inputs: ['surface', 'time-series'], outputs: ['surface']},
  // raster-analysis/remote-sensing/gpu-raster-seasonal-break.ts
  GPURasterSeasonalBreak: {inputs: ['surface', 'time-series'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts
  GPURasterEmbeddingDequantize: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-quantize.ts
  GPURasterEmbeddingQuantize: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-similarity.ts
  GPURasterEmbeddingSimilarity: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-embeddings/gpu-raster-embedding-nearest-pixels.ts
  GPURasterEmbeddingNearestPixels: {inputs: ['surface', 'attributes'], outputs: ['attributes']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-change.ts
  GPURasterEmbeddingChange: {inputs: ['surface', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts
  GPURasterEmbeddingKMeans: {inputs: ['surface'], outputs: ['surface', 'attributes', 'summary']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts
  GPURasterEmbeddingKNearestClassifier: {inputs: ['surface', 'attributes'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-linear-probe.ts
  GPURasterEmbeddingLinearProbe: {inputs: ['surface', 'attributes'], outputs: ['surface']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-covariance.ts
  GPURasterEmbeddingCovariance: {inputs: ['surface'], outputs: ['summary']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-projection.ts
  GPURasterEmbeddingProjection: {
    inputs: ['surface', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean.ts
  GPURasterEmbeddingZoneMean: {inputs: ['surface', 'regions', 'pairs'], outputs: ['attributes']},
  // raster-analysis/raster-embeddings/gpu-raster-embedding-pyramid-level.ts
  GPURasterEmbeddingPyramidLevel: {inputs: ['surface'], outputs: ['surface']},
  // raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts
  GPURasterStreamingAccumulator: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-rolling-mean.ts
  GPURasterRollingMean: {inputs: ['surface', 'time-series', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts
  GPURasterWhittakerSmoother: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-hants.ts
  GPURasterHANTS: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface', 'attributes']
  },
  // raster-analysis/raster-time-series/gpu-raster-temporal-resample.ts
  GPURasterTemporalResample: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-medoid-composite.ts
  GPURasterMedoidComposite: {inputs: ['surface', 'time-series'], outputs: ['surface']},
  // raster-analysis/raster-time-series/gpu-raster-geometric-median-composite.ts
  GPURasterGeometricMedianComposite: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-best-available-pixel.ts
  GPURasterBestAvailablePixel: {
    inputs: ['surface', 'time-series', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-phenology-metrics.ts
  GPURasterPhenologyMetrics: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration.ts
  GPURasterStandardizedIndexCalibration: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-standardized-index.ts
  GPURasterStandardizedIndex: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator.ts
  GPURasterClimateRunAccumulator: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-harmonic-regression.ts
  GPURasterHarmonicRegression: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts
  GPURasterCOLDMonitor: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['surface', 'attributes']
  },
  // raster-analysis/raster-time-series/gpu-raster-bfast-monitor.ts
  GPURasterBFASTMonitor: {inputs: ['surface', 'time-series', 'parameters'], outputs: ['surface']},
  // raster-analysis/raster-time-series/gpu-raster-snapshot-covariance.ts
  GPURasterSnapshotCovariance: {
    inputs: ['surface', 'time-series', 'parameters'],
    outputs: ['summary']
  },
  // raster-analysis/raster-time-series/gpu-raster-eof-projection.ts
  GPURasterEOFProjection: {
    inputs: ['surface', 'time-series', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-lagged-correlation.ts
  GPURasterLaggedCorrelation: {
    inputs: ['surface', 'time-series', 'attributes', 'parameters'],
    outputs: ['surface']
  },
  // raster-analysis/raster-time-series/gpu-raster-climate-anomaly.ts
  GPURasterClimateAnomaly: {inputs: ['surface', 'time-series', 'parameters'], outputs: ['surface']},
  // dataframe-analysis/column-classification/gpu-column-quantiles.ts
  GPUColumnQuantiles: {inputs: ['attributes', 'parameters'], outputs: ['summary', 'attributes']},
  // dataframe-analysis/column-classification/gpu-class-breaks.ts
  GPUClassBreaks: {inputs: ['attributes', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/column-classification/gpu-color-scale.ts
  GPUColorScale: {inputs: ['attributes', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/column-classification/gpu-bivariate-classification.ts
  GPUBivariateClassification: {
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // dataframe-analysis/column-classification/gpu-scalar-scale.ts
  GPUScalarScale: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // dataframe-analysis/variable-standardization/gpu-variable-standardization.ts
  GPUVariableStandardization: {inputs: ['attributes'], outputs: ['attributes']},
  // dataframe-analysis/group-statistics/gpu-group-statistics.ts
  GPUGroupStatistics: {inputs: ['attributes'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/key-join/gpu-key-join.ts
  GPUKeyJoin: {inputs: ['attributes'], outputs: ['pairs', 'attributes']},
  // dataframe-analysis/column-profile/gpu-column-profile.ts
  GPUColumnProfile: {inputs: ['attributes', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/composite-indicators/gpu-composite-score.ts
  GPUCompositeScore: {inputs: ['attributes', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/composite-indicators/gpu-inequality.ts
  GPUInequality: {inputs: ['attributes', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts
  GPUCalendarBuckets: {
    inputs: ['time-series', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts
  GPUTemporalReduction: {
    inputs: ['time-series', 'attributes', 'parameters'],
    outputs: ['time-series']
  },
  // dataframe-analysis/time-window-filter/gpu-time-window-filter.ts
  GPUTimeWindowFilter: {
    inputs: ['time-series', 'attributes', 'parameters'],
    outputs: ['attributes']
  },
  // dataframe-analysis/cube-layout/gpu-cube-layout-transpose.ts
  GPUCubeLayoutTranspose: {inputs: ['time-series'], outputs: ['time-series']},
  // dataframe-analysis/period-index/gpu-period-index.ts
  GPUPeriodIndex: {inputs: ['time-series', 'attributes'], outputs: ['attributes']},
  // dataframe-analysis/time-words-relative/gpu-time-words-relative.ts
  GPUTimeWordsRelative: {inputs: ['time-series'], outputs: ['time-series']},
  // dataframe-analysis/value-cube/gpu-value-cube.ts
  GPUValueCube: {inputs: ['attributes', 'time-series'], outputs: ['time-series']},
  // dataframe-analysis/value-cube/gpu-cell-period-keys.ts
  GPUCellPeriodKeys: {inputs: ['attributes', 'time-series'], outputs: ['attributes']},
  // dataframe-analysis/pair-statistics/gpu-variogram.ts
  GPUVariogram: {inputs: ['points', 'attributes', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts
  GPUCoregionalizationFit: {inputs: ['attributes', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts
  GPUVariogramModelFit: {inputs: ['attributes'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-cross-variogram.ts
  GPUCrossVariogram: {
    inputs: ['points', 'attributes', 'time-series', 'parameters'],
    outputs: ['summary']
  },
  // dataframe-analysis/pair-statistics/gpu-space-time-variogram.ts
  GPUSpaceTimeVariogram: {
    inputs: ['points', 'attributes', 'time-series', 'parameters'],
    outputs: ['summary']
  },
  // dataframe-analysis/pair-statistics/gpu-spatial-correlogram.ts
  GPUSpatialCorrelogram: {inputs: ['points', 'attributes', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-ripley.ts
  GPURipley: {inputs: ['points', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-ripley-envelope.ts
  GPURipleyEnvelope: {inputs: ['points', 'attributes', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-ripley-distance-functions.ts
  GPURipleyDistanceFunctions: {inputs: ['points', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts
  GPUPointPatternIndices: {inputs: ['points', 'parameters'], outputs: ['attributes', 'summary']},
  // dataframe-analysis/pair-statistics/gpu-marked-point-statistics.ts
  GPUMarkedPointStatistics: {inputs: ['points', 'attributes', 'parameters'], outputs: ['summary']},
  // dataframe-analysis/pair-statistics/gpu-space-time-k-function.ts
  GPUSpaceTimeKFunction: {
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['summary', 'attributes']
  },
  // dataframe-analysis/pair-statistics/gpu-space-time-diggle-test.ts
  GPUSpaceTimeDiggleTest: {inputs: ['points', 'time-series', 'parameters'], outputs: ['summary']},
  // pointcloud-analysis/operations/gpu-pointcloud-neighborhoods.ts
  GPUPointCloudNeighborhoods: {inputs: ['points', 'attributes'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-features.ts
  GPUPointCloudFeatures: {inputs: ['points', 'attributes', 'parameters'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-statistical-filter.ts
  GPUPointCloudStatisticalFilter: {inputs: ['attributes', 'parameters'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-ground.ts
  GPUPointCloudGround: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['surface', 'attributes']
  },
  // pointcloud-analysis/operations/gpu-pointcloud-smrf.ts
  GPUPointCloudSMRF: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['surface', 'attributes']
  },
  // pointcloud-analysis/operations/gpu-pointcloud-surfaces.ts
  GPUPointCloudSurfaces: {inputs: ['points', 'attributes'], outputs: ['surface']},
  // pointcloud-analysis/operations/gpu-pointcloud-height.ts
  GPUPointCloudHeight: {inputs: ['points', 'surface'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-icp.ts
  GPUPointCloudICP: {
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary']
  },
  // pointcloud-analysis/operations/gpu-pointcloud-normal-scale.ts
  GPUPointCloudNormalScale: {inputs: ['points', 'parameters'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-change.ts
  GPUPointCloudChange: {inputs: ['points', 'attributes', 'parameters'], outputs: ['attributes']},
  // pointcloud-analysis/operations/gpu-pointcloud-voxel-merge.ts
  GPUPointCloudVoxelMerge: {inputs: ['points', 'attributes'], outputs: ['points', 'attributes']},
  // geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts
  GPUDatumGridTransform: {inputs: ['points', 'surface'], outputs: ['points', 'attributes']},
  // geospatial-analysis/point-buffer/gpu-point-buffer.ts
  GPUPointBuffer: {inputs: ['points', 'parameters'], outputs: ['polygons']},

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
  // gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join.ts: queries, features; nearestFeatureIds, neighborIds plus original neighborRows, matches
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
  // gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights.ts: neighborRows, neighborCounts, neighborDistances; spatial weights CSR
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
function freezeKindRecords(
  rows: Record<string, GPUAnalysisOperatorKinds>
): Readonly<Record<string, GPUAnalysisOperatorKinds>> {
  for (const row of Object.values(rows)) {
    Object.freeze(row.inputs);
    Object.freeze(row.outputs);
    Object.freeze(row);
  }
  return Object.freeze(rows);
}

/** Compatibility name for the static records formerly used only as resolver supplements. */
export const GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS = GPU_ANALYSIS_OPERATOR_KIND_RECORDS;
