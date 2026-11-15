// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialAnalysisCapability} from '../spatial-analysis-ontology';

/**
 * Capability families for the contributors and recipes of the entry points other than
 * `gpu-spatial-analysis`. They are kept out of `GPU_SPATIAL_ANALYSIS_CAPABILITIES` so that the
 * consumers of that list (the website inventory generator and the kepler catalog bindings) see
 * what they saw before. Use `getGPUAnalysisCapabilityCatalog()` for the whole set.
 */
export const ENTRY_POINT_CAPABILITIES: readonly GPUSpatialAnalysisCapability[] = [
  {
    id: 'forest-inference',
    title: 'Imported forest inference',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'analyze',
    summary:
      'Applies caller-supplied sparse random forests and boosted numerical trees to rows or calibrated raster stacks.',
    inputs: ['attributes', 'parameters', 'surface'],
    outputs: ['attributes'],
    coordinateModels: ['none', 'grid'],
    outputBound: 'source-sized',
    operators: ['GPUForestInference']
  },
  {
    id: 'computation-recording',
    title: 'Computation recording',
    status: 'available',
    domain: 'workflow',
    stage: 'orchestrate',
    summary:
      'Records a command graph and captures bounded snapshots of its views for inspection and replay.',
    inputs: ['parameters', 'summary'],
    outputs: ['summary'],
    coordinateModels: ['none'],
    outputBound: 'fixed-sized',
    operators: ['GPUComputationRecorder', 'GPUSnapshotArena', 'GPUSnapshotTap']
  },
  {
    id: 'selection-transfer',
    title: 'Selection transfer',
    status: 'available',
    domain: 'workflow',
    stage: 'transform',
    summary: 'Clears and scatters a selection mask across a GPU-resident relation.',
    inputs: ['attributes'],
    outputs: ['attributes'],
    coordinateModels: ['none'],
    outputBound: 'source-sized',
    operators: ['GPUTransferMask']
  },
  {
    id: 'network-construction',
    title: 'Network construction and transformation',
    status: 'available',
    domain: 'network-analysis',
    stage: 'construct',
    summary:
      'Builds routable networks from lines, snaps points to them, and derives line graphs, subgraphs and coarsened graphs.',
    inputs: ['lines', 'points', 'network', 'attributes'],
    outputs: ['network', 'attributes'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUNetworkNoding',
      'GPUNetworkSnapping',
      'GPUNetworkLineGraph',
      'GPUNetworkSubgraphFilter',
      'GPUNetworkCoarsening'
    ]
  },
  {
    id: 'network-routing',
    title: 'Network routing',
    status: 'available',
    domain: 'network-analysis',
    stage: 'relate',
    summary:
      'Shortest-path costs, predecessor extraction, alternative routes, turn-aware and time-dependent routing on directed networks.',
    inputs: ['network', 'points', 'attributes', 'parameters'],
    outputs: ['attributes', 'network', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUNetworkReachability',
      'GPUNetworkPathExtraction',
      'GPUAlternativeRoutes',
      'GPUKShortestPaths',
      'GPUNetworkPointRouting',
      'GPUNetworkTimeRouting',
      'GPUNetworkPotentialPathArea',
      'GPUNetworkTravelTimePercentiles',
      'addNetworkDisruptionRecipe'
    ]
  },
  {
    id: 'network-accessibility',
    title: 'Network accessibility',
    status: 'available',
    domain: 'network-analysis',
    stage: 'aggregate',
    summary:
      'Cost matrices and tables, accessibility scores, service areas, isochrones and catchments from network travel costs.',
    inputs: ['network', 'points', 'attributes', 'parameters'],
    outputs: ['attributes', 'spatial-weights', 'polygons', 'summary'],
    coordinateModels: ['topological', 'planar'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUNetworkCostMatrix',
      'GPUNetworkCostTable',
      'GPUNetworkAccessibility',
      'GPUNetworkCostWeights',
      'GPUNetworkServiceAreas',
      'GPUNetworkIsochrones',
      'GPUNetworkRationalAccess',
      'addDriveTimeCatchmentRecipe'
    ]
  },
  {
    id: 'network-structure',
    title: 'Network structure and centrality',
    status: 'available',
    domain: 'network-analysis',
    stage: 'analyze',
    summary:
      'Degree, components, strongly connected components, k-hop neighborhoods, centrality and community detection on a network.',
    inputs: ['network', 'attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'source-sized',
    operators: [
      'GPUNetworkStructure',
      'GPUNetworkStronglyConnectedComponents',
      'GPUNetworkStatistics',
      'GPUNetworkAnalyticsColumns',
      'GPUNetworkNeighborhood',
      'GPUNetworkCentrality',
      'GPUNetworkLouvain'
    ]
  },
  {
    id: 'network-optimization',
    title: 'Network flow and facility optimization',
    status: 'available',
    domain: 'network-analysis',
    stage: 'analyze',
    summary: 'Maximum flow and exhaustive facility selection over network costs.',
    inputs: ['network', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'fixed-sized',
    operators: ['GPUNetworkFlow', 'GPUNetworkFacilityOptimization', 'addFacilityScenarioRecipe']
  },
  {
    id: 'network-point-patterns',
    title: 'Network point patterns',
    status: 'available',
    domain: 'network-analysis',
    stage: 'infer',
    summary: 'Network-constrained K functions and kernel density of events on a road network.',
    inputs: ['network', 'points', 'attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    outputBound: 'fixed-sized',
    operators: ['GPUNetworkKFunction', 'GPUNetworkInhomogeneousK', 'GPUNetworkKernelDensity']
  },
  {
    id: 'network-movement',
    title: 'Network movement',
    status: 'available',
    domain: 'network-analysis',
    stage: 'analyze',
    summary:
      'Map matching of tracks to a network, origin-destination flow aggregation and track quality.',
    inputs: ['trajectories', 'network', 'attributes'],
    outputs: ['attributes', 'lines', 'summary'],
    coordinateModels: ['planar', 'topological'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUMapMatching',
      'GPUFlowAggregation',
      'GPUMapMatchedEdgeVolumes',
      'GPURadiationFlowModel',
      'GPUKernelFlowSmoothing',
      'addTrackQualityRecipe'
    ]
  },
  {
    id: 'network-visualization',
    title: 'Network visualization layouts',
    status: 'available',
    domain: 'network-analysis',
    stage: 'transform',
    summary: 'Adjacency-matrix ordering and binning, and edge bundling into renderable polylines.',
    inputs: ['network', 'attributes'],
    outputs: ['render-geometry', 'attributes'],
    coordinateModels: ['topological'],
    outputBound: 'capacity-bounded',
    operators: ['GPUAdjacencyMatrix', 'GPUAdjacencyMatrixOrder', 'GPUEdgeBundling']
  },
  {
    id: 'terrain-hydrology',
    title: 'Terrain hydrology',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'analyze',
    summary:
      'Flow direction and accumulation, watersheds, stream order and networks, flow distance and height above drainage on elevation rasters.',
    inputs: ['surface'],
    outputs: ['surface', 'attributes', 'lines', 'summary'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUTerrainFlow',
      'GPUTerrainFlowFractions',
      'GPUTerrainBranchingFlow',
      'GPUTerrainTiledFlow',
      'GPUTerrainFlowDistance',
      'GPUTerrainWatersheds',
      'GPUTerrainStreamOrder',
      'GPUTerrainStreamNetwork',
      'GPUTerrainHeightAboveDrainage',
      'GPUTerrainHydrologicIndices',
      'GPUTerrainCatchmentSummary'
    ]
  },
  {
    id: 'terrain-conditioning',
    title: 'Terrain conditioning',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'transform',
    summary: 'Decodes, repairs, breaches and co-registers elevation rasters before analysis.',
    inputs: ['surface', 'points'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUTerrainBreach',
      'GPUTerrainDepressionHierarchy',
      'GPUTerrainSpikeRepair',
      'GPUTerrainRGBDecode',
      'GPUTerrainDEMCoregistration'
    ]
  },
  {
    id: 'terrain-morphometry',
    title: 'Terrain morphometry',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'analyze',
    summary: 'Slope, aspect, curvature, ruggedness and contour lines of elevation rasters.',
    inputs: ['surface'],
    outputs: ['surface', 'lines'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUTerrainDerivatives',
      'GPUTerrainCurvature',
      'GPUTerrainRuggedness',
      'GPUTerrainVectorRuggedness',
      'GPUTerrainContours'
    ]
  },
  {
    id: 'terrain-landforms',
    title: 'Terrain landforms and features',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'analyze',
    summary:
      'Topographic position, landform classification, geomorphons, critical points, summits and profile peaks.',
    inputs: ['surface', 'points', 'attributes'],
    outputs: ['surface', 'attributes', 'points'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUTerrainTopographicPosition',
      'GPUTerrainWeissLandforms',
      'GPUGeomorphons',
      'GPUTerrainCriticalPoints',
      'GPUTerrainSummits',
      'GPUTerrainPeakSnap',
      'GPUProfilePeaks'
    ]
  },
  {
    id: 'terrain-visibility',
    title: 'Terrain visibility',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'relate',
    summary:
      'Viewsheds, line of sight, cumulative viewsheds and point horizon profiles over elevation rasters.',
    inputs: ['surface', 'points'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUTerrainViewshed',
      'GPUTerrainLineOfSight',
      'GPUTerrainCumulativeViewshed',
      'GPUPointHorizonProfile',
      'GPUPointHorizonVisibility'
    ]
  },
  {
    id: 'terrain-illumination',
    title: 'Terrain illumination',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'analyze',
    summary:
      'Solar position, horizon maps, cast shadows, shadow masks and irradiance over a day or date range.',
    inputs: ['surface', 'points', 'parameters'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid', 'spherical'],
    outputBound: 'source-sized',
    operators: [
      'GPUSolarPosition',
      'GPUTerrainHorizon',
      'GPUTerrainCastShadow',
      'GPUSolarShadowMask',
      'GPUSolarIrradiance'
    ]
  },
  {
    id: 'terrain-relief-visualization',
    title: 'Terrain relief visualization',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'transform',
    summary:
      'Local relief models, local dominance, hillshade blends and texture shading for relief display.',
    inputs: ['surface'],
    outputs: ['surface'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUSimpleLocalRelief',
      'GPUMultiScaleRelief',
      'GPULocalDominance',
      'GPUReliefBlend',
      'GPUReliefShading',
      'GPUTextureShading'
    ]
  },
  {
    id: 'terrain-wind',
    title: 'Terrain-following wind',
    status: 'available',
    domain: 'terrain-analysis',
    stage: 'analyze',
    summary: 'A uniform wind deflected by the terrain so that it flows around and over relief.',
    inputs: ['surface', 'parameters'],
    outputs: ['surface'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: ['GPUTerrainFlowField']
  },
  {
    id: 'raster-algebra',
    title: 'Raster algebra',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Per-cell arithmetic, conditionals, reclassification, weighted overlay, cell statistics and contrast stretch.',
    inputs: ['surface', 'parameters'],
    outputs: ['surface'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPURasterArithmetic',
      'GPURasterConditional',
      'GPURasterReclassify',
      'GPUWeightedOverlay',
      'GPURasterCellStatistics',
      'GPURasterStretch',
      'GPURasterContrastEqualization'
    ]
  },
  {
    id: 'raster-spectral',
    title: 'Raster spectral indices and unmixing',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Normalized-difference, vegetation and burn indices and least-squares spectral unmixing.',
    inputs: ['surface'],
    outputs: ['surface'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPURasterNDWI',
      'GPURasterNBR',
      'GPURasterEVI',
      'GPURasterSAVI',
      'GPURasterSpectralUnmixing'
    ]
  },
  {
    id: 'raster-focal',
    title: 'Raster focal operations',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Neighborhood statistics on rasters: focal pairs, rank filters, texture and nodata filling.',
    inputs: ['surface', 'parameters'],
    outputs: ['surface'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPURasterFocalPairs',
      'GPURasterRankFilter',
      'GPURasterTexture',
      'GPURasterFillNoData'
    ]
  },
  {
    id: 'raster-morphology',
    title: 'Raster morphology and segmentation',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Geodesic reconstruction, hole filling, skeletons, medial axes, watershed segmentation and region graphs.',
    inputs: ['surface', 'regions'],
    outputs: ['surface', 'regions'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPURasterReconstruction',
      'GPURasterFillHoles',
      'GPURasterSkeleton',
      'GPURasterMedialAxis',
      'GPURasterWatershedSegmentation',
      'GPURasterRegionGraph'
    ]
  },
  {
    id: 'raster-patches',
    title: 'Raster patches',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'analyze',
    summary:
      'Per-patch area, perimeter and bounds of label rasters, and sieve filtering of small patches.',
    inputs: ['surface', 'regions'],
    outputs: ['surface', 'attributes', 'summary'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: ['GPURasterPatchMetrics', 'GPURasterSieve']
  },
  {
    id: 'raster-distance',
    title: 'Raster distance and cost surfaces',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'relate',
    summary:
      'Euclidean distance fields, least-cost distance, least-cost paths and walking cost over rasters.',
    inputs: ['surface', 'points'],
    outputs: ['surface', 'lines'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUDistanceField',
      'GPUCostDistance',
      'GPUCostDistancePath',
      'GPURasterWalkingCost'
    ]
  },
  {
    id: 'raster-zonal',
    title: 'Raster zonal statistics',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'aggregate',
    summary:
      'Rasterizes polygons and summarizes rasters per zone, with exact fractional coverage and raster joins of points.',
    inputs: ['surface', 'polygons', 'points', 'regions'],
    outputs: ['attributes', 'summary', 'surface'],
    coordinateModels: ['grid', 'planar'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUPolygonRasterization',
      'GPURasterJoin',
      'GPURasterZonalStatistics',
      'GPURasterFractionalCoverage',
      'GPURasterFractionalZonalStatistics',
      'GPURasterFractionalCategoricalStatistics',
      'GPURasterFractionalZonalDistribution'
    ]
  },
  {
    id: 'raster-contours',
    title: 'Raster contours and isobands',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'construct',
    summary: 'Marching-squares isolines, filled isobands and closed isoband rings from rasters.',
    inputs: ['surface', 'parameters'],
    outputs: ['lines', 'polygons', 'surface'],
    coordinateModels: ['grid'],
    outputBound: 'capacity-bounded',
    operators: ['GPUIsolines', 'GPUIsobands', 'GPUIsobandRings']
  },
  {
    id: 'raster-flow-visualization',
    title: 'Raster flow visualization',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Particle advection, line integral convolution and evenly spaced streamlines through vector fields.',
    inputs: ['surface', 'points'],
    outputs: ['surface', 'points', 'lines'],
    coordinateModels: ['grid', 'planar', 'mixed'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUParticleAdvection',
      'GPULagrangianFlowMap',
      'GPUFiniteTimeLyapunovExponent',
      'GPUScalarFieldRidges',
      'GPULineIntegralConvolution',
      'GPUStreamlines'
    ]
  },
  {
    id: 'raster-resampling',
    title: 'Raster resampling and registration',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'transform',
    summary:
      'Reprojection, tiled reprojection, conservative remapping, phase-correlation registration and min/max pyramids.',
    inputs: ['surface', 'parameters'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPURasterReprojection',
      'GPURasterTileReprojection',
      'GPURasterConservativeRemapping',
      'GPURasterRegistration',
      'GPURasterSpectralFilter',
      'GPURasterExtremaPyramid'
    ]
  },
  {
    id: 'raster-sampling',
    title: 'Raster sampling',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'relate',
    summary: 'Samples rasters at points and along polylines.',
    inputs: ['surface', 'points', 'lines'],
    outputs: ['attributes'],
    coordinateModels: ['grid', 'planar'],
    outputBound: 'source-sized',
    operators: ['GPURasterSampling', 'GPURasterProfile']
  },
  {
    id: 'raster-time-series',
    title: 'Raster time series',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'analyze',
    summary:
      'Change detection, temporal composites and smoothing, and seasonal break detection over stacks of raster time slices.',
    inputs: ['surface', 'time-series'],
    outputs: ['surface', 'attributes'],
    coordinateModels: ['grid'],
    outputBound: 'source-sized',
    operators: [
      'GPUChangeDetection',
      'GPURasterTemporal',
      'GPURasterSeasonalBreak',
      'GPURasterStreamingAccumulator',
      'GPURasterRollingMean',
      'GPURasterWhittakerSmoother',
      'GPURasterHANTS',
      'GPURasterTemporalResample',
      'GPURasterMedoidComposite',
      'GPURasterGeometricMedianComposite',
      'GPURasterBestAvailablePixel',
      'GPURasterPhenologyMetrics',
      'GPURasterStandardizedIndexCalibration',
      'GPURasterStandardizedIndex',
      'GPURasterClimateRunAccumulator',
      'GPURasterClimateAnomaly',
      'GPURasterHarmonicRegression',
      'GPURasterCOLDMonitor',
      'GPURasterBFASTMonitor',
      'GPURasterSnapshotCovariance',
      'GPURasterEOFProjection',
      'GPURasterLaggedCorrelation'
    ]
  },
  {
    id: 'raster-embeddings',
    title: 'Embedding rasters',
    status: 'available',
    domain: 'raster-analysis',
    stage: 'analyze',
    summary:
      'Decodes int8 embedding rasters and computes similarity, change, clustering, classification, principal components, zone means and unit-vector overviews.',
    inputs: ['surface', 'parameters'],
    outputs: ['surface', 'attributes', 'summary'],
    coordinateModels: ['grid'],
    outputBound: 'mixed',
    operators: [
      'GPURasterEmbeddingDequantize',
      'GPURasterEmbeddingQuantize',
      'GPURasterEmbeddingSimilarity',
      'GPURasterEmbeddingNearestPixels',
      'GPURasterEmbeddingChange',
      'GPURasterEmbeddingKMeans',
      'GPURasterEmbeddingKNearestClassifier',
      'GPURasterEmbeddingLinearProbe',
      'GPURasterEmbeddingCovariance',
      'GPURasterEmbeddingProjection',
      'GPURasterEmbeddingZoneMean',
      'GPURasterEmbeddingPyramidLevel'
    ]
  },
  {
    id: 'dataframe-classification',
    title: 'Column classification and scales',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'transform',
    summary:
      'Class breaks, quantiles, colour and scalar scales, bivariate classes and standardization of columns.',
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    outputBound: 'source-sized',
    operators: [
      'GPUColumnQuantiles',
      'GPUClassBreaks',
      'GPUColorScale',
      'GPUBivariateClassification',
      'GPUScalarScale',
      'GPUVariableStandardization'
    ]
  },
  {
    id: 'dataframe-aggregation',
    title: 'Column aggregation and joins',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'aggregate',
    summary:
      'Per-group statistics, column profiles, composite scores, inequality indices and key joins.',
    inputs: ['attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUGroupStatistics',
      'GPUKeyJoin',
      'GPUColumnProfile',
      'GPUCompositeScore',
      'GPUInequality'
    ]
  },
  {
    id: 'dataframe-temporal',
    title: 'Temporal columns',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'transform',
    summary:
      'Calendar bucketing, period indices, relative Int64 time conversion, per-bucket temporal reduction, time-window filtering and cube layout changes.',
    inputs: ['attributes', 'time-series', 'parameters'],
    outputs: ['attributes', 'time-series', 'summary'],
    coordinateModels: ['none'],
    outputBound: 'source-sized',
    operators: [
      'GPUCalendarBuckets',
      'GPUTemporalReduction',
      'GPUTimeWindowFilter',
      'GPUCubeLayoutTranspose',
      'GPUPeriodIndex',
      'GPUTimeWordsRelative',
      'GPUValueCube',
      'GPUCellPeriodKeys'
    ]
  },
  {
    id: 'dataframe-geostatistics',
    title: 'Empirical variograms',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'infer',
    summary:
      'Empirical variograms, cross and space-time variograms, model fitting and spatial correlograms.',
    inputs: ['points', 'attributes'],
    outputs: ['summary'],
    coordinateModels: ['planar'],
    outputBound: 'fixed-sized',
    operators: [
      'GPUVariogram',
      'GPUVariogramModelFit',
      'GPUCoregionalizationFit',
      'GPUCrossVariogram',
      'GPUSpaceTimeVariogram',
      'GPUSpatialCorrelogram'
    ]
  },
  {
    id: 'dataframe-point-patterns',
    title: 'Point pattern statistics',
    status: 'available',
    domain: 'dataframe-analysis',
    stage: 'infer',
    summary:
      'Ripley K and L, simulation envelopes, distance functions, nearest-neighbor indices and marked point statistics.',
    inputs: ['points', 'attributes'],
    outputs: ['summary'],
    coordinateModels: ['planar'],
    outputBound: 'fixed-sized',
    operators: [
      'GPURipley',
      'GPURipleyEnvelope',
      'GPURipleyDistanceFunctions',
      'GPUPointPatternIndices',
      'GPUMarkedPointStatistics',
      'GPUSpaceTimeKFunction',
      'GPUSpaceTimeDiggleTest'
    ]
  },
  {
    id: 'pointcloud-neighborhoods',
    title: 'Point cloud neighborhoods and filters',
    status: 'available',
    domain: 'point-cloud-analysis',
    stage: 'relate',
    summary:
      'Exact 3D neighborhoods, covariance features and statistical outlier filtering of point clouds.',
    inputs: ['points'],
    outputs: ['attributes', 'spatial-weights'],
    coordinateModels: ['planar'],
    outputBound: 'capacity-bounded',
    operators: [
      'GPUPointCloudNeighborhoods',
      'GPUPointCloudFeatures',
      'GPUPointCloudStatisticalFilter'
    ]
  },
  {
    id: 'pointcloud-ground',
    title: 'Point cloud ground and surfaces',
    status: 'available',
    domain: 'point-cloud-analysis',
    stage: 'transform',
    summary:
      'Ground classification (progressive morphology and SMRF), terrain and surface models, and height above ground.',
    inputs: ['points', 'surface'],
    outputs: ['attributes', 'surface'],
    coordinateModels: ['planar'],
    outputBound: 'source-sized',
    operators: [
      'GPUPointCloudGround',
      'GPUPointCloudSMRF',
      'GPUPointCloudSurfaces',
      'GPUPointCloudHeight'
    ]
  },
  {
    id: 'pointcloud-registration',
    title: 'Point cloud registration and change',
    status: 'available',
    domain: 'point-cloud-analysis',
    stage: 'relate',
    summary:
      'Iterative closest point alignment and cylindrical change detection between point clouds.',
    inputs: ['points'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    outputBound: 'source-sized',
    operators: ['GPUPointCloudICP', 'GPUPointCloudChange', 'GPUPointCloudNormalScale']
  },
  {
    id: 'pointcloud-merge',
    title: 'Point cloud voxel merging',
    status: 'available',
    domain: 'point-cloud-analysis',
    stage: 'aggregate',
    summary: 'Merges per-voxel sufficient statistics of disjoint point-cloud parts.',
    inputs: ['points', 'attributes'],
    outputs: ['points', 'attributes'],
    coordinateModels: ['planar'],
    outputBound: 'capacity-bounded',
    operators: ['GPUPointCloudVoxelMerge']
  },
  {
    id: 'geospatial-transformation',
    title: 'Geospatial transformation',
    status: 'available',
    domain: 'geometry',
    stage: 'transform',
    summary: 'Datum grid shifts and inscribed circle buffers around points.',
    inputs: ['points', 'parameters'],
    outputs: ['points', 'polygons'],
    coordinateModels: ['ellipsoidal', 'planar'],
    outputBound: 'source-sized',
    operators: ['GPUDatumGridTransform', 'GPUPointBuffer']
  }
];
