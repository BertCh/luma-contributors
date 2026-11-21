// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {
  getGPUAnalysisOperatorQuestions,
  getGPUAnalysisQuestionCoverage,
  queryGPUAnalysisOperatorsByQuestion
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-questions';
import {
  GPU_ANALYSIS_QUESTION_RECORDS,
  getGPUAnalysisQuestion
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-questions';
import {GPU_ANALYSIS_QUESTIONS} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';
import type {GPUAnalysisQuestion} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-experience-types';
import {GPU_SPATIAL_ANALYSIS_DOMAINS} from '../../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';
import type {GPUAnalysisOperator} from '../../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';

it('W3-01 discovers all qualified building tessellation routes as preparation', () => {
  const matches = queryGPUAnalysisOperatorsByQuestion('prepare-data', {primaryOnly: true});
  for (const exportName of [
    'addBuildingTessellationRecipe',
    'addBuildingTessellationReferenceRecipe',
    'addLocalBuildingTessellationRecipe'
  ]) {
    expect(
      matches.some(
        operator =>
          operator.package === '@luma-contributors/analysis/gpu-spatial-analysis' &&
          operator.exportName === exportName
      ),
      exportName
    ).toBe(true);
  }
});

/**
 * Operators labelled by hand from their documentation and source, across all sixteen domains, before
 * the rules were run against them. The value is the full expected list, primary first.
 */
const LABELLED_OPERATORS: readonly (readonly [string, readonly GPUAnalysisQuestion[]])[] = [
  // where-concentrated
  ['GPUKernelDensity', ['where-concentrated']],
  ['GPUPointDensity', ['where-concentrated']],
  ['GPUAdaptiveBandwidthKernelDensity', ['where-concentrated']],
  ['GPUNetworkKernelDensity', ['where-concentrated']],
  ['GPURipley', ['where-concentrated']],
  // A dot-density map builds render points from counts: preparation, not an answer.
  ['GPUDotDensity', ['prepare-data']],
  ['GPUKeyLookup', ['prepare-data']],
  // where-significant
  ['GPUHotSpotAnalysis', ['where-significant']],
  ['GPULocalMoran', ['where-significant']],
  ['GPUQuantileLISA', ['where-significant']],
  ['GPUGlobalSpatialStatistics', ['where-significant']],
  ['GPUKnoxTest', ['where-significant']],
  ['GPULocalOutlierFactor', ['where-significant']],
  ['GPUMultipleTestingAdjustment', ['where-significant']],
  ['GPUSpatialScanStatistic', ['where-significant', 'where-concentrated']],
  // how-much-per-area
  ['GPUZonalStatistics', ['how-much-per-area']],
  ['GPUPointInPolygonJoin', ['how-much-per-area']],
  ['GPUGroupStatistics', ['how-much-per-area']],
  ['GPURasterZonalStatistics', ['how-much-per-area']],
  ['GPUCellAggregation', ['how-much-per-area']],
  ['GPUArealInterpolation', ['how-much-per-area']],
  ['addPointsInPolygonsChoroplethRecipe', ['how-much-per-area']],
  ['GPUDorlingCartogram', ['prepare-data']],
  // what-is-near-or-reachable
  ['GPUBufferSelection', ['what-is-near-or-reachable']],
  ['GPUNearestFeatureJoin', ['what-is-near-or-reachable']],
  ['GPUNetworkServiceAreas', ['what-is-near-or-reachable']],
  ['GPUNetworkIsochrones', ['what-is-near-or-reachable']],
  ['GPUNetworkCostMatrix', ['what-is-near-or-reachable']],
  ['GPUCostDistance', ['what-is-near-or-reachable']],
  ['GPUCatchmentAccessibility', ['what-is-near-or-reachable']],
  ['GPUNeighborSearch', ['what-is-near-or-reachable']],
  // how-related
  ['GPUOrdinaryLeastSquares', ['how-related']],
  ['GPUGeographicallyWeightedRegression', ['how-related']],
  ['GPUMultiscaleGeographicallyWeightedRegression', ['how-related']],
  ['GPUSpatialPearson', ['how-related']],
  ['GPURasterLaggedCorrelation', ['how-related']],
  ['GPUSpatialLagMaximumLikelihood', ['how-related']],
  // value-where-unmeasured
  ['GPUInverseDistanceWeighting', ['value-where-unmeasured']],
  ['GPUKriging', ['value-where-unmeasured']],
  ['GPULocalRBF', ['value-where-unmeasured']],
  ['GPUVariogram', ['value-where-unmeasured']],
  ['GPUConditionalGaussianSimulation', ['value-where-unmeasured']],
  // what-changed
  ['GPUEmergingHotSpots', ['what-changed', 'where-significant']],
  ['GPUChangeDetection', ['what-changed']],
  ['addPeriodComparisonRecipe', ['what-changed']],
  ['addRegionalRasterChangeRecipe', ['what-changed']],
  ['GPURasterEmbeddingChange', ['what-changed']],
  ['GPUPettittTest', ['what-changed']],
  ['GPUPointCloudChange', ['what-changed']],
  // how-it-moves
  ['GPUTrajectoryStaypoints', ['how-it-moves']],
  ['GPUMapMatching', ['how-it-moves']],
  ['GPUFlowAggregation', ['how-it-moves']],
  ['GPUTerrainFlow', ['how-it-moves']],
  ['GPUTerrainStreamNetwork', ['how-it-moves']],
  ['GPUParticleAdvection', ['how-it-moves']],
  ['GPUTrajectoryEncounters', ['what-is-near-or-reachable', 'how-it-moves']],
  // shape-of-the-surface
  ['GPUTerrainDerivatives', ['shape-of-the-surface']],
  ['GPUTerrainCurvature', ['shape-of-the-surface']],
  ['GPUTerrainViewshed', ['shape-of-the-surface']],
  ['GPUReliefShading', ['shape-of-the-surface']],
  ['GPUIsolines', ['shape-of-the-surface']],
  ['GPUPointCloudFeatures', ['shape-of-the-surface']],
  ['GPUGeomorphons', ['shape-of-the-surface']],
  // which-groups-or-regions
  ['GPUDBSCAN', ['which-groups-or-regions', 'where-concentrated']],
  ['GPUSpatialKMeans', ['which-groups-or-regions']],
  ['GPUSkaterRegions', ['which-groups-or-regions']],
  ['GPUAZPRegions', ['which-groups-or-regions']],
  ['GPUNetworkStronglyConnectedComponents', ['which-groups-or-regions']],
  ['GPUNetworkLouvain', ['which-groups-or-regions']],
  ['GPURasterEmbeddingKMeans', ['which-groups-or-regions']],
  // what-is-at-each-place
  ['GPURasterNDWI', ['what-is-at-each-place']],
  ['GPURasterNBR', ['what-is-at-each-place']],
  ['GPURasterEVI', ['what-is-at-each-place']],
  ['GPURasterSAVI', ['what-is-at-each-place']],
  ['GPURasterSpectralUnmixing', ['what-is-at-each-place']],
  ['GPURasterTexture', ['what-is-at-each-place']],
  ['GPUWeightedOverlay', ['what-is-at-each-place']],
  ['GPUCompositeScore', ['what-is-at-each-place']],
  ['GPURasterEmbeddingKNearestClassifier', ['what-is-at-each-place']],
  // Added after an independent read of operator sources.
  ['GPURasterFocalPairs', ['how-related']],
  ['GPURasterCellStatistics', ['what-is-at-each-place']],
  ['GPUSinkhornDivergence', ['what-changed']],
  ['GPUCategoryPairCounts', ['where-significant']],
  ['GPUSpatialCorrelogram', ['where-significant']],
  ['GPUBrownianBridgeUtilization', ['where-concentrated', 'how-it-moves']],
  ['GPUUtilizationVolumeIsopleths', ['where-concentrated', 'how-it-moves']],
  ['GPUNetworkStructure', ['which-groups-or-regions']],
  ['GPUNetworkStatistics', ['which-groups-or-regions']],
  ['GPUNetworkCentrality', ['what-is-near-or-reachable']],
  ['GPUBufferSurface', ['what-is-near-or-reachable']],
  ['GPUPointBuffer', ['what-is-near-or-reachable']],
  ['GPUGeodesicPairs', ['what-is-near-or-reachable']],
  ['GPULocalOutlierFactor', ['where-significant']],
  // Geometry and topology build geometry; they do not answer a question.
  ['GPUPolygonOverlay', ['prepare-data']],
  ['GPUPopulationSupportAlignment', ['prepare-data']],
  ['GPUSphericalPolygonOverlay', ['prepare-data']],
  ['GPUSphericalConvexContains', ['prepare-data']],
  ['GPUSegmentIntersection', ['prepare-data']],
  ['GPUSharedPaths', ['prepare-data']],
  ['GPUS2CapCovering', ['prepare-data']],
  ['GPUS2PolygonCovering', ['prepare-data']],
  ['GPUVoronoiDiagram', ['prepare-data']],
  ['GPUGeometryPredicates', ['prepare-data']],
  ['GPUShapeDescriptors', ['prepare-data']],
  ['GPUGeometryMeasures', ['prepare-data']],
  ['GPUMinimumBounds', ['prepare-data']],
  ['GPUSolarPosition', ['prepare-data']],
  // Discrete grids and selection: only aggregation and comparison are analysis.
  ['GPUCellTableCompare', ['what-changed']],
  ['GPUCellTopology', ['prepare-data']],
  ['GPUCellCover', ['prepare-data']],
  ['GPUPointToCell', ['prepare-data']],
  ['GPUCellLevelSelection', ['prepare-data']],
  ['GPUPickRegionMask', ['prepare-data']],
  ['GPURegionMask', ['prepare-data']],
  ['GPUTransferMask', ['prepare-data']],
  // Plumbing that its own documentation calls a transformation, a fold builder or a lag.
  ['GPUAssuncaoReisStandardization', ['prepare-data']],
  ['GPUNeighborhoodSummary', ['prepare-data']],
  ['GPUSpatialCrossValidation', ['prepare-data']],
  ['GPUSpatialWeightsTraces', ['prepare-data']],
  ['GPUSpatialLag', ['prepare-data']],
  ['GPUAdjacencyMatrix', ['prepare-data']],
  ['GPURasterTemporal', ['prepare-data']],
  ['GPURasterSpectralFilter', ['prepare-data']],
  ['GPURasterSampling', ['prepare-data']],
  // prepare-data
  ['GPUGeometryValidity', ['prepare-data']],
  ['GPUMakeValid', ['prepare-data']],
  ['GPULineSimplification', ['prepare-data']],
  ['GPUDatumGridTransform', ['prepare-data']],
  ['GPUVariableStandardization', ['prepare-data']],
  ['GPUContiguityWeights', ['prepare-data']],
  ['GPUSpatialWeightsTransform', ['prepare-data']],
  ['GPURasterReprojection', ['prepare-data']],
  ['GPUTerrainBreach', ['prepare-data']],
  ['GPUComputationRecorder', ['prepare-data']]
];

/** Operators of the catalogue when this table was written; see the coverage test. */
const BASELINE_OPERATOR_NAMES: readonly string[] = [
  'GPUQuantileLISA',
  'GPUDorlingCartogram',
  'addExcessRiskRecipe',
  'addSpatialRateRecipe',
  'GPUTrackSimilaritySearch',
  'GPUTrajectoryKalman',
  'GPUTrajectoryRestriction',
  'GPUCanonicalGeometryEquality',
  'GPUNormalizeGeometry',
  'GPUOffsetCurveComponents',
  'GPUVectorStraightSkeleton',
  'GPUCanonicalPolygonEquality',
  'GPUConvexStraightSkeleton',
  'GPULineLoopRemoval',
  'GPUNormalizePolygons',
  'GPUMaximumInscribedCircle',
  'GPUSegmentSnap',
  'GPURatioConcaveHull',
  'GPUPolygonPrecisionReduction',
  'GPULinePrecisionReduction',
  'GPUSubdividePolygons',
  'GPUS2CellHierarchy',
  'GPUS2CellMeasures',
  'GPUS2CapCovering',
  'GPUSphericalConvexIntersection',
  'GPUSphericalConvexContains',
  'GPUSphericalPolygonContains',
  'GPUSphericalPolygonOverlay',
  'GPUSphericalArcOverlay',
  'GPUS2PolygonCovering',
  'GPULocalKriging',
  'GPULocalRBF',
  'GPUNaturalNeighbor',
  'GPUConditionalGaussianSimulation',
  'GPUInstrumentalVariablesCovariance',
  'GPURestrictedRegimeRegression',
  'GPUSpatialPanelErrorMaximumLikelihood',
  'GPUSpatialPanelLagMaximumLikelihood',
  'GPUSeeminglyUnrelatedRegression',
  'GPUProbitRegression',
  'GPULocalCollinearity',
  'GPUMultivariateLocalGearyPermutationTest',
  'GPURankMarkov',
  'GPUFuzzyContiguity',
  'GPUIslandAttachment',
  'GPUSpatialCrossValidation',
  'GPUSpatialCrossValidatedOLS',
  'GPUSpatialInteraction',
  'GPUInhomogeneousPointPattern',
  'GPUPointPatternEnvelope',
  'GPUSpatialBalancedSample',
  'GPUContinuousSpatialSample',
  'GPUOPTICS',
  'addBuildingTessellationRecipe',
  'addStreetProfilesRecipe',
  'addBuildingAdjacencyRecipe',
  'addIsovistsRecipe',
  'addVisibilityGraphRecipe',
  'addLandscapeMetricsRecipe',
  'addPatchProximityRecipe',
  'addResistanceConnectivityRecipe',
  'addSparseResistanceConnectivityRecipe',
  'addSegregationDecompositionRecipe',
  'addSegregationCounterfactualRecipe',
  'addPopulationSupportAlignmentRecipe',
  'GPUNestedInterpolationValidation',
  'GPULocalSpatialHeteroscedasticity',
  'GPULOSHResidualPermutationTest',
  'GPUMultivariateLocalGeary',
  'GPUGeneralizedGeographicallyWeightedRegression',
  'GPUMultiscaleGeographicallyWeightedRegression',
  'GPUSpatialPanelRegression',
  'GPUSpatialRegimeRegression',
  'GPUConcaveHull',
  'GPUGroupConcaveHull',
  'GPUConstrainedDelaunayTriangulation',
  'GPUH3CellIdentities',
  'GPUBoundedFacilityAllocation',
  'addZonalDistributionRecipe',
  'addRasterZonalDistributionRecipe',
  'addInterpolationDiagnosticsRecipe',
  'addExposureByZoneRecipe',
  'addRegionalRasterChangeRecipe',
  'addDrainageNetworkRecipe',
  'addLocalVariabilityRecipe',
  'addCellFlowTopologyRecipe',
  'addSurfacePreparationRecipe',
  'addDasymetricRedistributionRecipe',
  'GPUTerrainPeakProminence',
  'GPUGeometryMeasures',
  'GPUShapeDescriptors',
  'GPUMinimumBounds',
  'GPUMinimumClearance',
  'GPULabelPoint',
  'GPUGeographicDistribution',
  'GPUGeodesicPairs',
  'GPUGeodesicDestination',
  'GPULocalFrameProjection',
  'GPUAffineTransform',
  'GPUGeometryOrientation',
  'GPUGeometryCleanup',
  'GPUVertexSnap',
  'GPUOffsetCurve',
  'GPUGridGenerator',
  'GPUShapeGenerator',
  'GPUDotDensity',
  'GPURandomPointsInPolygon',
  'GPURandomPointsOnLine',
  'GPUPolygonTriangulation',
  'GPUOutlineGeometry',
  'GPUGeometryPredicates',
  'GPUGeometryValidity',
  'GPUCoverageValidity',
  'GPUGreatCircleArcs',
  'GPULineChunk',
  'GPULineSegmentize',
  'GPULineSmooth',
  'GPULineSimplification',
  'GPULineSplit',
  'GPULineMerge',
  'GPULineClipByPolygon',
  'GPUSharedPaths',
  'GPURectangleClip',
  'GPULonLatPolygonPreparation',
  'GPULinearReferencing',
  'GPULineLocate',
  'GPUSegmentIntersection',
  'GPULineNoding',
  'GPUSegmentRingAssembly',
  'GPUCoverageDissolve',
  'GPUCoverageSimplification',
  'GPUHilbertKeys',
  'GPUSpatialJoinCandidates',
  'GPUSpatialJoinPrepared',
  'GPUSpatialPredicateJoin',
  'GPUPointInPolygonJoin',
  'GPUNearestFeatureJoin',
  'GPUBufferSelection',
  'GPUPairGather',
  'GPUOffsetExpansion',
  'GPUBoundsFilter',
  'GPUNeighborSearch',
  'GPUContiguityWeights',
  'GPULatticeWeights',
  'GPUNearestFeatureWeights',
  'GPUSpatialWeightsTransform',
  'GPUSpatialWeightsAlgebra',
  'GPUSpatialWeightsSummary',
  'GPUSpatialWeightsTranspose',
  'GPUSpatialWeightsValidation',
  'GPUChunkedSpatialWeightsTransform',
  'GPUChunkedSpatialWeightsAlgebra',
  'GPUChunkedSpatialWeightsTranspose',
  'GPUChunkedSpatialWeightsValidation',
  'GPUSpatialLag',
  'GPUMapColoring',
  'GPUPointToCell',
  'GPUCellGeometry',
  'GPUCellTopology',
  'GPUCellCompaction',
  'GPUCellGridPath',
  'GPUCellMeasures',
  'GPUCellCover',
  'GPUCellAggregation',
  'GPUCellRollup',
  'GPUCellPyramid',
  'GPUCellLevelSelection',
  'GPUCellSetOutline',
  'GPUCellTableCompare',
  'GPUPointDensity',
  'GPUKernelDensity',
  'GPUStudyWindowPreparation',
  'GPULineDensity',
  'GPULineLengthPerPolygon',
  'GPUInverseDistanceWeighting',
  'GPUKriging',
  'GPUFocalStatistics',
  'GPUZonalStatistics',
  'GPURegionMask',
  'GPUPickRegionMask',
  'GPURegionStatistics',
  'GPURegionStatisticsReadback',
  'GPUDBSCAN',
  'GPUHDBSCAN',
  'GPUSpatialKMeans',
  'GPUSpatialWeightsMinimumSpanningTree',
  'GPUSkaterRegions',
  'GPURegionPartitionEvaluation',
  'GPUGroupGeometry',
  'GPUGroupConvexHull',
  'GPULocalOutlierFactor',
  'GPUSimilarLocations',
  'GPUGlobalSpatialStatistics',
  'GPUHotSpotAnalysis',
  'GPULocalMoran',
  'GPULocalPermutationTest',
  'GPUGlobalPermutationTest',
  'GPUNeighborhoodSummary',
  'GPUEmpiricalBayesRates',
  'GPUSpatialEmpiricalBayesRates',
  'GPUAssuncaoReisStandardization',
  'GPUMultipleTestingAdjustment',
  'GPUSpatialScanStatistic',
  'GPUKnoxTest',
  'GPUMantelTest',
  'GPUOrdinaryLeastSquares',
  'GPUSpatialRegressionDiagnostics',
  'GPUSpatialTwoStageLeastSquares',
  'GPUSpatialErrorGM',
  'GPUSpatialLagMaximumLikelihood',
  'GPUSpatialErrorMaximumLikelihood',
  'GPUSpatialWeightsTraces',
  'GPUSpatialLagImpacts',
  'GPUGeographicallyWeightedRegression',
  'GPUGeographicallyWeightedRegressionNonstationarityTest',
  'GPUClassAssignment',
  'GPUClassificationFit',
  'GPUTransitionMatrix',
  'GPUSpatialMarkov',
  'GPULISAMarkov',
  'GPUTransitionDiagnostics',
  'GPUEmergingHotSpots',
  'GPUCatchmentAccessibility',
  'GPUHuffTradeAreas',
  'GPUSegregation',
  'GPUArealInterpolation',
  'GPUPycnophylacticInterpolation',
  'GPUTrajectoryMetrics',
  'GPUTrajectoryPlayhead',
  'GPUTrajectoryResample',
  'GPUZoneEvents',
  'GPUTrajectoryEncounters',
  'GPUTrajectoryClosestApproach',
  'GPUTrackSimilarity',
  'addClockEncounters',
  'addChangeOfSupportRecipe',
  'addClusterAndOutlineRecipe',
  'addFleetDwellRecipe',
  'addFleetDwellZoneEventsRecipe',
  'addHotSpotAnalysisRecipe',
  'addPeriodComparisonRecipe',
  'addPointsInPolygonsChoroplethRecipe',
  'addRateClusterMapRecipe',
  'addSpaceTimeHotSpotsRecipe',
  'addSpatialRegressionRecipe',
  'addStraightLineCatchmentsRecipe',
  'GPUPolygonOverlay',
  'GPUBufferSurface',
  'GPUDelaunayTessellation',
  'GPUVoronoiDiagram',
  'GPUPolygonize',
  'GPUMakeValid',
  'GPULocalGeary',
  'GPUSpatialPearson',
  'GPUGammaStatistic',
  'GPURoseStatistic',
  'GPUAZPRegions',
  'GPUMaxPRegions',
  'GPUWardRegions',
  'GPULocationAllocation',
  'GPUComputationRecorder',
  'GPUSnapshotArena',
  'GPUSnapshotTap',
  'GPUTransferMask',
  'GPUNetworkNoding',
  'GPUNetworkSnapping',
  'GPUNetworkLineGraph',
  'GPUNetworkSubgraphFilter',
  'GPUNetworkCoarsening',
  'GPUNetworkReachability',
  'GPUNetworkPathExtraction',
  'GPUAlternativeRoutes',
  'GPUKShortestPaths',
  'GPUNetworkPointRouting',
  'GPUNetworkTimeRouting',
  'GPUNetworkPotentialPathArea',
  'GPUNetworkTravelTimePercentiles',
  'addNetworkDisruptionRecipe',
  'GPUNetworkCostMatrix',
  'GPUNetworkCostTable',
  'GPUNetworkAccessibility',
  'GPUNetworkCostWeights',
  'GPUNetworkServiceAreas',
  'GPUNetworkIsochrones',
  'GPUNetworkRationalAccess',
  'addDriveTimeCatchmentRecipe',
  'GPUNetworkStructure',
  'GPUNetworkStronglyConnectedComponents',
  'GPUNetworkStatistics',
  'GPUNetworkAnalyticsColumns',
  'GPUNetworkNeighborhood',
  'GPUNetworkCentrality',
  'GPUNetworkLouvain',
  'GPUNetworkFlow',
  'GPUNetworkFacilityOptimization',
  'addFacilityScenarioRecipe',
  'GPUNetworkKFunction',
  'GPUNetworkInhomogeneousK',
  'GPUNetworkKernelDensity',
  'GPUMapMatching',
  'GPUFlowAggregation',
  'GPUMapMatchedEdgeVolumes',
  'GPURadiationFlowModel',
  'GPUKernelFlowSmoothing',
  'addTrackQualityRecipe',
  'GPUAdjacencyMatrix',
  'GPUAdjacencyMatrixOrder',
  'GPUEdgeBundling',
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
  'GPUTerrainCatchmentSummary',
  'GPUTerrainBreach',
  'GPUTerrainDepressionHierarchy',
  'GPUTerrainSpikeRepair',
  'GPUTerrainRGBDecode',
  'GPUTerrainDEMCoregistration',
  'GPUTerrainDerivatives',
  'GPUTerrainCurvature',
  'GPUTerrainRuggedness',
  'GPUTerrainVectorRuggedness',
  'GPUTerrainContours',
  'GPUTerrainTopographicPosition',
  'GPUTerrainWeissLandforms',
  'GPUGeomorphons',
  'GPUTerrainCriticalPoints',
  'GPUTerrainSummits',
  'GPUTerrainPeakSnap',
  'GPUProfilePeaks',
  'GPUTerrainViewshed',
  'GPUTerrainLineOfSight',
  'GPUTerrainCumulativeViewshed',
  'GPUPointHorizonProfile',
  'GPUPointHorizonVisibility',
  'GPUSolarPosition',
  'GPUTerrainHorizon',
  'GPUTerrainCastShadow',
  'GPUSolarShadowMask',
  'GPUSolarIrradiance',
  'GPUSimpleLocalRelief',
  'GPUMultiScaleRelief',
  'GPULocalDominance',
  'GPUReliefBlend',
  'GPUReliefShading',
  'GPUTextureShading',
  'GPUTerrainFlowField',
  'GPURasterArithmetic',
  'GPURasterConditional',
  'GPURasterReclassify',
  'GPUWeightedOverlay',
  'GPURasterCellStatistics',
  'GPURasterStretch',
  'GPURasterNDWI',
  'GPURasterNBR',
  'GPURasterEVI',
  'GPURasterSAVI',
  'GPURasterSpectralUnmixing',
  'GPURasterFocalPairs',
  'GPURasterRankFilter',
  'GPURasterTexture',
  'GPURasterFillNoData',
  'GPURasterReconstruction',
  'GPURasterFillHoles',
  'GPURasterSkeleton',
  'GPURasterMedialAxis',
  'GPURasterWatershedSegmentation',
  'GPURasterRegionGraph',
  'GPURasterPatchMetrics',
  'GPURasterSieve',
  'GPUDistanceField',
  'GPUCostDistance',
  'GPUCostDistancePath',
  'GPURasterWalkingCost',
  'GPUPolygonRasterization',
  'GPURasterJoin',
  'GPURasterZonalStatistics',
  'GPURasterFractionalCoverage',
  'GPURasterFractionalZonalStatistics',
  'GPURasterFractionalCategoricalStatistics',
  'GPURasterFractionalZonalDistribution',
  'GPUIsolines',
  'GPUIsobands',
  'GPUIsobandRings',
  'GPUParticleAdvection',
  'GPULagrangianFlowMap',
  'GPUFiniteTimeLyapunovExponent',
  'GPULineIntegralConvolution',
  'GPUStreamlines',
  'GPURasterReprojection',
  'GPURasterTileReprojection',
  'GPURasterConservativeRemapping',
  'GPURasterRegistration',
  'GPURasterSpectralFilter',
  'GPURasterExtremaPyramid',
  'GPURasterSampling',
  'GPURasterProfile',
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
  'GPURasterLaggedCorrelation',
  'GPURasterEmbeddingDequantize',
  'GPURasterEmbeddingSimilarity',
  'GPURasterEmbeddingNearestPixels',
  'GPURasterEmbeddingChange',
  'GPURasterEmbeddingKMeans',
  'GPURasterEmbeddingKNearestClassifier',
  'GPURasterEmbeddingLinearProbe',
  'GPURasterEmbeddingCovariance',
  'GPURasterEmbeddingProjection',
  'GPURasterEmbeddingZoneMean',
  'GPURasterEmbeddingPyramidLevel',
  'GPUColumnQuantiles',
  'GPUClassBreaks',
  'GPUColorScale',
  'GPUBivariateClassification',
  'GPUScalarScale',
  'GPUVariableStandardization',
  'GPUGroupStatistics',
  'GPUKeyLookup',
  'GPUColumnProfile',
  'GPUCompositeScore',
  'GPUInequality',
  'GPUCalendarBuckets',
  'GPUTemporalReduction',
  'GPUTimeWindowFilter',
  'GPUCubeLayoutTranspose',
  'GPUPeriodIndex',
  'GPUTimeWordsRelative',
  'GPUValueCube',
  'GPUCellPeriodKeys',
  'GPUVariogram',
  'GPUVariogramModelFit',
  'GPUCoregionalizationFit',
  'GPUCrossVariogram',
  'GPUSpaceTimeVariogram',
  'GPUSpatialCorrelogram',
  'GPURipley',
  'GPURipleyEnvelope',
  'GPURipleyDistanceFunctions',
  'GPUPointPatternIndices',
  'GPUMarkedPointStatistics',
  'GPUSpaceTimeKFunction',
  'GPUSpaceTimeDiggleTest',
  'GPUPointCloudNeighborhoods',
  'GPUPointCloudFeatures',
  'GPUPointCloudStatisticalFilter',
  'GPUPointCloudGround',
  'GPUPointCloudSMRF',
  'GPUPointCloudSurfaces',
  'GPUPointCloudHeight',
  'GPUPointCloudICP',
  'GPUPointCloudChange',
  'GPUPointCloudVoxelMerge',
  'GPUDatumGridTransform',
  'GPUPointBuffer',
  'GPULocationChangePoints',
  'GPUPettittTest',
  'GPUExpectationBasedSpaceTimeScan',
  'GPUNearRepeatKnox',
  'GPULocalKnox',
  'GPUJacquezTest',
  'GPUSpaceTimeLocalMoran',
  'GPUSpaceTimeCrossCorrelation',
  'GPUSpaceTimeKernelDensity',
  'GPUSpaceTimeRelativeRisk',
  'GPUSpaceTimeKriging',
  'GPUEnsembleScores',
  'GPUGaussianPredictiveScores',
  'GPUSpaceTimeHawkes',
  'GPUSinkhornDivergence',
  'GPUVecchiaLogLikelihood',
  'GPUVecchiaFisherScoring',
  'GPUVecchiaSpaceTimeLogLikelihood',
  'GPUConformalPredictionIntervals',
  'GPUGeoConformalPredictionIntervals',
  'GPUAreaOfApplicability',
  'GPUGeographicallyTemporallyWeightedRegression',
  'GPUMultiscaleGeographicallyTemporallyWeightedRegression',
  'addTrajectorySortCheck',
  'GPUTrajectoryStaypoints',
  'GPUTrajectoryLocations',
  'GPUTrajectorySegments',
  'GPUMobilityIndicators',
  'GPUBrownianBridgeUtilization',
  'GPUUtilizationVolumeIsopleths',
  'GPUTrajectoryRecursions',
  'GPUModalSplit',
  'GPUTrajectoryOutlierFlags',
  'GPUTrajectoryHampel',
  'GPUTrajectorySplit',
  'GPUTrajectoryThinning',
  'GPUTrajectoryTrackRelativeTimes',
  'GPUGeographicalDetector',
  'GPURedcapSingleLinkage',
  'GPUCategoryPairCounts',
  'GPUSpatialEntropy',
  'GPUColocationQuotient',
  'GPULocalCategoryDependence',
  'GPUAdaptiveBandwidthKernelDensity',
  'GPURasterFalseDiscoveryRateMask',
  'addContiguityOrderCorrelogramRecipe',
  'GPUHealpixPointToCell',
  'GPUHealpixCellAggregation',
  'addRasterResistanceConnectivityRecipe'
];

function makeSyntheticOperator(
  exportName: string,
  primaryCapability: string,
  overrides: Partial<GPUAnalysisOperator> = {}
): GPUAnalysisOperator {
  return {
    exportName,
    stability: 'experimental',
    primaryCapability,
    tags: [],
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    inputs: [],
    outputs: [],
    dimensionality: 'unspecified',
    coordinateModels: [],
    topology: 'unspecified',
    cardinality: 'unspecified',
    semantics: 'unspecified',
    deterministic: 'unspecified',
    status: [],
    readback: 'none',
    evidence: {status: 'opportunity', tests: '', documentation: '', contract: ''},
    ...overrides
  };
}

describe('GPU_ANALYSIS_QUESTION_RECORDS', () => {
  it('covers exactly the question ids, in order', () => {
    expect(GPU_ANALYSIS_QUESTION_RECORDS.map(record => record.id)).toEqual([
      ...GPU_ANALYSIS_QUESTIONS
    ]);
  });

  it('words every question as a question with a summary and two or three examples', () => {
    for (const record of GPU_ANALYSIS_QUESTION_RECORDS) {
      expect(record.label.endsWith('?'), record.id).toBe(true);
      expect(record.summary.length, record.id).toBeGreaterThan(20);
      expect(record.summary.endsWith('.'), record.id).toBe(true);
      expect(record.examples.length, record.id).toBeGreaterThanOrEqual(2);
      expect(record.examples.length, record.id).toBeLessThanOrEqual(3);
      for (const example of record.examples) expect(example.endsWith('?'), example).toBe(true);
    }
    expect(new Set(GPU_ANALYSIS_QUESTION_RECORDS.map(record => record.label)).size).toBe(
      GPU_ANALYSIS_QUESTION_RECORDS.length
    );
  });

  it('keeps the wording the host menus rely on', () => {
    expect(getGPUAnalysisQuestion('where-significant').label).toBe(
      'Where do values cluster or stand out?'
    );
  });
});

describe('getGPUAnalysisQuestion', () => {
  it('returns the record of every id', () => {
    for (const id of GPU_ANALYSIS_QUESTIONS) expect(getGPUAnalysisQuestion(id).id).toBe(id);
  });

  it('throws naming the function and the id for an unknown id', () => {
    expect(() => getGPUAnalysisQuestion('nonsense' as GPUAnalysisQuestion)).toThrow(
      /getGPUAnalysisQuestion.*"nonsense"/
    );
    expect(() => getGPUAnalysisQuestion('' as GPUAnalysisQuestion)).toThrow(
      /getGPUAnalysisQuestion/
    );
  });
});

describe('getGPUAnalysisOperatorQuestions on the hand-labelled operators', () => {
  it('has at least sixty labelled operators that exist in the catalogue and span every domain', () => {
    expect(LABELLED_OPERATORS.length).toBeGreaterThanOrEqual(60);
    const domains = new Set<string>();
    const capabilityById = new Map(
      getGPUAnalysisCapabilityCatalog().map(capability => [capability.id, capability])
    );
    for (const [exportName] of LABELLED_OPERATORS) {
      const operator = getGPUAnalysisOperator(
        exportName === 'addVisibilityGraphRecipe'
          ? {package: '@luma-contributors/analysis/gpu-spatial-analysis', exportName}
          : exportName
      );
      expect(operator, exportName).toBeDefined();
      domains.add(capabilityById.get(operator!.primaryCapability)!.domain);
    }
    // The model-fitting-kernel family is the only one whose operators are all helpers.
    for (const domain of GPU_SPATIAL_ANALYSIS_DOMAINS) {
      if (domain === 'execution' || domain === 'discrete-global-grids') continue;
      expect(domains.has(domain), domain).toBe(true);
    }
  });

  it.each(LABELLED_OPERATORS)('answers %s with the labelled questions', (exportName, expected) => {
    const operator = getGPUAnalysisOperator(
      exportName === 'addVisibilityGraphRecipe'
        ? {package: '@luma-contributors/analysis/gpu-spatial-analysis', exportName}
        : exportName
    )!;
    expect(getGPUAnalysisOperatorQuestions(operator)).toEqual(expected);
  });
});

describe('getGPUAnalysisOperatorQuestions over the whole catalogue', () => {
  const operators = getGPUAnalysisOperatorCatalog();

  it('gives every operator and recipe one to three distinct valid questions', () => {
    for (const operator of operators) {
      const questions = getGPUAnalysisOperatorQuestions(operator);
      expect(questions.length, operator.exportName).toBeGreaterThanOrEqual(1);
      expect(questions.length, operator.exportName).toBeLessThanOrEqual(3);
      expect(new Set(questions).size, operator.exportName).toBe(questions.length);
      for (const question of questions) {
        expect(GPU_ANALYSIS_QUESTIONS, operator.exportName).toContain(question);
      }
    }
  });

  it('is deterministic', () => {
    const first = operators.map(operator => getGPUAnalysisOperatorQuestions(operator));
    const second = [...operators]
      .reverse()
      .map(operator => getGPUAnalysisOperatorQuestions(operator));
    expect(second.reverse()).toEqual(first);
  });

  it('leaves no question empty on the pinned baseline and keeps preparation a minority', () => {
    // Operators other sessions add later do not move these numbers: the check runs on the names
    // pinned in BASELINE_OPERATOR_NAMES, so honest growth of the catalogue cannot turn it red.
    const baseline = new Set(BASELINE_OPERATOR_NAMES);
    const counts = new Map<GPUAnalysisQuestion, number>();
    let baselineCount = 0;
    for (const operator of operators) {
      if (!baseline.has(operator.exportName) || operator.role === 'helper') continue;
      baselineCount++;
      const primary = getGPUAnalysisOperatorQuestions(operator)[0];
      counts.set(primary, (counts.get(primary) ?? 0) + 1);
    }
    expect(baselineCount).toBeGreaterThan(500);
    for (const question of GPU_ANALYSIS_QUESTIONS) {
      expect(counts.get(question) ?? 0, question).toBeGreaterThanOrEqual(10);
    }
    expect((counts.get('prepare-data') ?? 0) / baselineCount).toBeLessThan(0.45);
  });

  it('keeps the coverage table consistent with the catalogue', () => {
    const coverage = getGPUAnalysisQuestionCoverage();
    expect(coverage.map(entry => entry.question)).toEqual([...GPU_ANALYSIS_QUESTIONS]);
    const operatorCount = queryGPUAnalysisOperators().length;
    expect(coverage.reduce((sum, entry) => sum + entry.primaryCount, 0)).toBe(operatorCount);
    for (const entry of coverage) {
      expect(entry.totalCount, entry.question).toBeGreaterThanOrEqual(entry.primaryCount);
    }
  });

  it('reports, without failing, operators that are not in the pinned baseline', () => {
    const baseline = new Set(BASELINE_OPERATOR_NAMES);
    const newer = operators.filter(operator => !baseline.has(operator.exportName));
    // Newer operators only have to resolve structurally; their labels belong in the table above.
    for (const operator of newer) {
      const questions = getGPUAnalysisOperatorQuestions(operator);
      expect(questions.length, operator.exportName).toBeGreaterThanOrEqual(1);
    }
  });

  it('asks a second question only for a minority of operators', () => {
    const withSecond = operators.filter(
      operator => getGPUAnalysisOperatorQuestions(operator).length > 1
    );
    expect(withSecond.length / operators.length).toBeLessThan(0.15);
  });
});

describe('getGPUAnalysisOperatorQuestions on operators the catalogue does not contain', () => {
  const capabilities = getGPUAnalysisCapabilityCatalog();

  it('resolves a synthetic operator in every domain and stage that has a family', () => {
    for (const domain of GPU_SPATIAL_ANALYSIS_DOMAINS) {
      for (const capability of capabilities.filter(candidate => candidate.domain === domain)) {
        const operator = makeSyntheticOperator('GPUNeverSeenBefore', capability.id);
        const questions = getGPUAnalysisOperatorQuestions(operator);
        expect(questions.length, `${domain}/${capability.id}`).toBeGreaterThanOrEqual(1);
        for (const question of questions) expect(GPU_ANALYSIS_QUESTIONS).toContain(question);
      }
    }
  });

  it('follows the subject area for an unrecognised name', () => {
    const byIdentifier = (identifier: string) =>
      getGPUAnalysisOperatorQuestions(makeSyntheticOperator('GPUNeverSeenBefore', identifier));
    expect(byIdentifier('spatial-regression')).toEqual(['how-related']);
    expect(byIdentifier('network-routing')).toEqual(['what-is-near-or-reachable']);
    expect(byIdentifier('terrain-morphometry')).toEqual(['shape-of-the-surface']);
    expect(byIdentifier('geometry-editing')).toEqual(['prepare-data']);
    expect(byIdentifier('zonal-and-region-statistics')).toEqual(['how-much-per-area']);
    expect(byIdentifier('movement-analysis')).toEqual(['how-it-moves']);
  });

  it('answers a new operator from its name before its family', () => {
    expect(
      getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator('GPUWeightedKernelDensity', 'geometry-measurement')
      )
    ).toEqual(['where-concentrated']);
    expect(
      getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator('addTemporalKrigingRecipe', 'workflow-recipes')
      )
    ).toEqual(['value-where-unmeasured']);
  });

  it('uses the data kinds of an unrecognised operator in an analysing family', () => {
    const trackOperator = makeSyntheticOperator('GPUNeverSeenBefore', 'geometry-measurement', {
      inputs: ['trajectories']
    });
    expect(getGPUAnalysisOperatorQuestions(trackOperator)).toEqual(['how-it-moves']);
    const seriesOperator = makeSyntheticOperator('GPUNeverSeenBefore', 'geometry-measurement', {
      inputs: ['time-series']
    });
    expect(getGPUAnalysisOperatorQuestions(seriesOperator)).toEqual(['what-changed']);
  });

  it('does not add a second question from a tagged family that is only context', () => {
    const operator = makeSyntheticOperator('GPUNeverSeenBefore', 'movement-analysis', {
      tags: ['spatial-joins', 'spatial-regression']
    });
    expect(getGPUAnalysisOperatorQuestions(operator)).toEqual(['how-it-moves']);
  });

  it('adds how-much-per-area as the second question of an operator tagged with an aggregation family', () => {
    const operator = makeSyntheticOperator('GPUNeverSeenBefore', 'movement-analysis', {
      tags: ['zonal-and-region-statistics']
    });
    expect(getGPUAnalysisOperatorQuestions(operator)).toEqual([
      'how-it-moves',
      'how-much-per-area'
    ]);
  });

  it('falls back to preparation for an unknown family', () => {
    expect(
      getGPUAnalysisOperatorQuestions(makeSyntheticOperator('GPUNeverSeenBefore', 'no-such-family'))
    ).toEqual(['prepare-data']);
  });

  it('never returns more than three questions even with many tags', () => {
    const operator = makeSyntheticOperator('GPUDBSCAN', 'group-and-regional-analysis', {
      tags: ['movement-analysis', 'spatial-regression', 'terrain-morphometry', 'network-routing']
    });
    expect(getGPUAnalysisOperatorQuestions(operator).length).toBeLessThanOrEqual(3);
  });
});

describe('stage veto', () => {
  const capabilities = getGPUAnalysisCapabilityCatalog();
  const preparationCapabilities = capabilities.filter(capability =>
    ['bridge', 'construct', 'transform', 'index'].includes(capability.stage)
  );
  // Names a reviewer proposed: a preparation step whose name contains an analytical word.
  const PREPARATION_NAMES = [
    'GPUTrajectoryGapFill',
    'GPUSpatialLagRegressionPrep',
    'GPUKernelDensityBandwidthSelector',
    'GPURasterTemporalGapFill',
    'GPUNetworkRenumbering',
    'GPUKernelDensityLite',
    'GPUTrajectorySimplification',
    'GPUPointCloudClassification',
    'addNewHotspotRecipe',
    'GPULocalMoranBivariate'
  ];

  it('has building and converting families to test against', () => {
    expect(preparationCapabilities.length).toBeGreaterThan(20);
  });

  it('resolves analytical-sounding names to prepare-data in a building or converting family', () => {
    for (const capability of preparationCapabilities) {
      for (const name of PREPARATION_NAMES) {
        const operator = makeSyntheticOperator(name, capability.id);
        expect(getGPUAnalysisOperatorQuestions(operator), `${name} in ${capability.id}`).toEqual([
          'prepare-data'
        ]);
      }
    }
  });

  it('keeps preparation-word names as preparation even in an analysing family', () => {
    for (const name of [
      'GPUTrajectoryGapFill',
      'GPUSpatialLagRegressionPrep',
      'GPUKernelDensityBandwidthSelector',
      'GPUNetworkRenumbering'
    ]) {
      for (const identifier of [
        'movement-analysis',
        'spatial-regression',
        'density-and-interpolation'
      ]) {
        expect(
          getGPUAnalysisOperatorQuestions(makeSyntheticOperator(name, identifier)),
          `${name} in ${identifier}`
        ).toEqual(['prepare-data']);
      }
    }
  });

  it('lets the same analytical name answer in an analysing family', () => {
    expect(
      getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator('GPUKernelDensityLite', 'density-and-interpolation')
      )
    ).toEqual(['where-concentrated']);
  });

  it('keeps the explicit exemptions: buffers, flow drawings, spectral indices, watersheds', () => {
    const transformFamily = preparationCapabilities.find(
      capability => capability.stage === 'transform'
    )!;
    const expectations: [string, GPUAnalysisQuestion][] = [
      ['GPUBufferSurface', 'what-is-near-or-reachable'],
      ['GPUPointBuffer', 'what-is-near-or-reachable'],
      ['GPUStreamlines', 'how-it-moves'],
      ['GPURasterNDWI', 'what-is-at-each-place'],
      ['GPUReliefShading', 'shape-of-the-surface'],
      ['GPURasterWatershedSegmentation', 'which-groups-or-regions']
    ];
    for (const [name, expected] of expectations) {
      expect(
        getGPUAnalysisOperatorQuestions(makeSyntheticOperator(name, transformFamily.id))[0],
        name
      ).toBe(expected);
    }
  });

  it('lets an override win over the veto', () => {
    const transformFamily = preparationCapabilities.find(
      capability => capability.stage === 'transform'
    )!;
    expect(
      getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator('GPUWeightedOverlay', transformFamily.id)
      )
    ).toEqual(['what-is-at-each-place']);
  });
});

describe('robustness', () => {
  it('resolves export names that collide with Object.prototype', () => {
    for (const name of ['constructor', 'toString', '__proto__', 'hasOwnProperty', 'valueOf']) {
      const questions = getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator(name, 'no-such-family')
      );
      expect(questions, name).toEqual(['prepare-data']);
      const inFamily = getGPUAnalysisOperatorQuestions(
        makeSyntheticOperator(name, 'spatial-regression')
      );
      expect(inFamily, name).toEqual(['how-related']);
    }
  });

  it('tolerates an operator without tags, inputs or semantics', () => {
    const bare = {
      exportName: 'GPUBare',
      primaryCapability: 'movement-analysis'
    } as unknown as GPUAnalysisOperator;
    expect(getGPUAnalysisOperatorQuestions(bare)).toEqual(['how-it-moves']);
    const unknownFamily = {
      exportName: 'GPUBare',
      primaryCapability: 'nothing'
    } as unknown as GPUAnalysisOperator;
    expect(getGPUAnalysisOperatorQuestions(unknownFamily)).toEqual(['prepare-data']);
  });
});

describe('queryGPUAnalysisOperatorsByQuestion', () => {
  it('finds operators by any question, and by primary question only', () => {
    const all = queryGPUAnalysisOperatorsByQuestion('where-significant');
    const primary = queryGPUAnalysisOperatorsByQuestion('where-significant', {primaryOnly: true});
    expect(primary.length).toBeGreaterThan(0);
    expect(all.length).toBeGreaterThan(primary.length);
    for (const operator of primary) {
      expect(getGPUAnalysisOperatorQuestions(operator)[0]).toBe('where-significant');
    }
    expect(primary.map(operator => operator.exportName)).toContain('GPULocalMoran');
    expect(all.map(operator => operator.exportName)).toContain('GPUEmergingHotSpots');
    expect(primary.map(operator => operator.exportName)).not.toContain('GPUEmergingHotSpots');
  });

  it('leaves helpers out unless asked', () => {
    const helperName = 'GPUParameterBuffer';
    expect(
      queryGPUAnalysisOperatorsByQuestion('prepare-data').map(operator => operator.exportName)
    ).not.toContain(helperName);
    expect(
      queryGPUAnalysisOperatorsByQuestion('prepare-data', {includeHelpers: true}).map(
        operator => operator.exportName
      )
    ).toContain(helperName);
  });

  it('throws for an unknown question', () => {
    expect(() => queryGPUAnalysisOperatorsByQuestion('nonsense' as GPUAnalysisQuestion)).toThrow(
      /queryGPUAnalysisOperatorsByQuestion.*"nonsense"/
    );
  });

  it('agrees with the coverage counts', () => {
    for (const entry of getGPUAnalysisQuestionCoverage()) {
      expect(queryGPUAnalysisOperatorsByQuestion(entry.question).length).toBe(entry.totalCount);
      expect(queryGPUAnalysisOperatorsByQuestion(entry.question, {primaryOnly: true}).length).toBe(
        entry.primaryCount
      );
    }
  });
});

describe('analysis questions after preparation-stage review', () => {
  it.each([
    ['GPURasterRegionGraph', 'which-groups-or-regions'],
    ['GPUPointCloudSurfaces', 'shape-of-the-surface'],
    ['GPUPointCloudHeight', 'shape-of-the-surface'],
    ['GPUHealpixCellAggregation', 'how-much-per-area'],
    ['GPUValueCube', 'how-much-per-area'],
    ['GPURasterCellStatistics', 'what-is-at-each-place']
  ] as const)('keeps %s visible as %s', (name, question) => {
    const operator = getGPUAnalysisOperator(name);
    expect(operator).toBeDefined();
    if (operator) expect(getGPUAnalysisOperatorQuestions(operator)).toEqual([question]);
  });
});
