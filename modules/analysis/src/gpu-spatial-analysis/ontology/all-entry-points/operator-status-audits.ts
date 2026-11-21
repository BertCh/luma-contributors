// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Affirmative output/publisher audits, retained by qualified public identity.
 * Scanner-negative CAT-101 entries were rechecked against publishers during R19 review round 1.
 * Empty means no catalog scalar status kinds, not absence of packed/per-row diagnostics.
 * This closed list is evidence, not a rule for newly encountered declarations.
 */
export const AUDITED_EMPTY_OPERATOR_STATUSES: Readonly<Record<string, string>> = Object.freeze({
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegmentSnap':
    'geometry-refinement/gpu-segment-snap.ts: candidateOverflow is per-line; counts is a packed uint32x4 row per line',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrackSimilaritySearch':
    'status bits per pair',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryKalman':
    'status bits per observation',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryRestriction':
    'status bits per interval slot and a packed compactSummary',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCanonicalGeometryEquality':
    'gpu-spatial-analysis/geometry-refinement/gpu-canonical-geometry-equality.ts: writes one equality/invalid-traversal code per pair to output; no scalar status port',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizeGeometry': 'status code per node',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVectorStraightSkeleton':
    'packed uint32x4 status row per polygon',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCanonicalPolygonEquality':
    'gpu-spatial-analysis/geometry-refinement/gpu-canonical-polygon-equality.ts: writes equality/nonfinite codes per polygon pair to output; no scalar status port',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConvexStraightSkeleton':
    'packed uint32x4 status row per polygon',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineLoopRemoval':
    'packed uint32x4 status row [written, required, overflow, incomplete]',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizePolygons':
    'gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons.ts: publishes only positions, ringOffsets and polygonOffsets; normalization delegates orientation without scalar status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMaximumInscribedCircle':
    'status code per feature',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULinePrecisionReduction':
    'status code per line',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSubdividePolygons':
    'packed uint32x4 status row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2CellHierarchy': 'status code per cell row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2CellMeasures': 'status code per cell row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSphericalConvexContains':
    'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSphericalPolygonContains':
    'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalKriging': 'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalRBF': 'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNaturalNeighbor':
    'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConditionalGaussianSimulation':
    'one status code per replicate',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUInstrumentalVariablesCovariance':
    'solver code only (Pick<GPUSolverStatusPort, code>)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURestrictedRegimeRegression':
    'one status code word',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPanelLagMaximumLikelihood':
    'one status code word with bit flags',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSeeminglyUnrelatedRegression':
    'status code words (per equation and FGLS)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUProbitRegression':
    'model and diagnostic status code words',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalCollinearity': 'status code per row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultivariateLocalGearyPermutationTest':
    'gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test.ts: publishes statistic, pseudoPValues, adjustedPValues, classes and optional simulations per row; incomplete input suppresses rows',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialCrossValidation':
    'status bits per fold',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialCrossValidatedOLS':
    'gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols.ts: publishes fold-major coefficients/predictions/metrics and per-fold fitStatus/metricStatus codes',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialInteraction': 'one status code word',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUInhomogeneousPointPattern':
    'one status code word',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointPatternEnvelope':
    'one status code word',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialBalancedSample':
    'packed [selectedCount, status] per stratum',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUContinuousSpatialSample':
    'packed uint32x4 status row per wave',
  '@luma-contributors/analysis/gpu-spatial-analysis:addStreetProfilesRecipe':
    'result has no status',
  '@luma-contributors/analysis/gpu-spatial-analysis:addIsovistsRecipe': 'result has no status',
  '@luma-contributors/analysis/gpu-spatial-analysis:addLandscapeMetricsRecipe':
    'result has no status',
  '@luma-contributors/analysis/gpu-spatial-analysis:addResistanceConnectivityRecipe':
    'packed uint32x4 status row',
  '@luma-contributors/analysis/gpu-spatial-analysis:addSparseResistanceConnectivityRecipe':
    'packed uint32x4 [converged, iterations, reachable, invalid] row',
  '@luma-contributors/analysis/gpu-spatial-analysis:addSegregationDecompositionRecipe':
    'result has no status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNestedInterpolationValidation':
    'status bits per fold',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUH3CellIdentities':
    'gpu-spatial-analysis/cell-topology/gpu-h3-cell-identities.ts: writes six edge or vertex keys per input cell; invalid keys are zero',
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainPeakProminence':
    'gpu-terrain/terrain-features/gpu-terrain-peak-prominence.ts: publish writes per-cell prominence, keySaddle and parentPeak columns',
  '@luma-contributors/analysis/analysis-core:GPUParameterBuffer':
    'parameter upload; no output status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUShapeDescriptors':
    'gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors.ts: publishes per-feature area, perimeter, compactness/moment/convexity and clockwise/sliver flags only',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULabelPoint':
    'gpu-spatial-analysis/label-point/gpu-label-point.ts: publishes points, distances and degenerate flags per feature',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeographicDistribution':
    'gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution.ts: publishes per-group counts, moments, medianConverged and geometry; no global scalar status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalFrameProjection':
    'gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts: publishes only projected coordinate rows and optional per-position validity',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUAffineTransform':
    'gpu-spatial-analysis/geometry-edit/gpu-affine-transform.ts: publishes only transformed positions; bounds/centroids are internal origin scratch',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeometryOrientation':
    'gpu-spatial-analysis/geometry-edit/gpu-geometry-orientation.ts: publishes positions and optional reversedRings flags per ring',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVertexSnap':
    'gpu-spatial-analysis/vertex-snap/gpu-vertex-snap.ts: publishes positions and referenceRows per input vertex',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGridGenerator':
    'gpu-spatial-analysis/grid-generators/gpu-grid-generator.ts: publishes fixed-size positions, centers, corners and per-cell/per-corner intersection masks only',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPolygonTriangulation':
    'analysis-core/polygon-triangulation/gpu-polygon-triangulation.ts: publishes indices and per-polygon valid, triangleCount and triangleOffsets; triangleCount is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineLocate':
    'gpu-spatial-analysis/linear-referencing/gpu-line-locate.ts: publishes event positions, segmentIndices, tangents, angles and per-event statuses; vertexMeasures is per vertex',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUHilbertKeys':
    'gpu-spatial-analysis/hilbert-keys/gpu-hilbert-keys.ts: publishes keys, sortedRows and sortedKeys per input item',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialJoinPrepared':
    'prepared index handle; its BVH storage words are read by the joins that use it',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsSummary':
    'gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-summary.ts: publishes packed statistics/counts arrays and per-row cardinality; no individually named scalar status views',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLag':
    'gpu-spatial-analysis/spatial-weights/gpu-spatial-lag.ts: publishes only dense lag values; input loss suppresses results rather than producing scalar status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointToCell':
    'gpu-spatial-analysis/cell-indexing/gpu-point-to-cell.ts: publishes cell keys and optional validity per position',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellGeometry':
    'gpu-spatial-analysis/cell-indexing/gpu-cell-geometry.ts: publishes centers, boundaries, vertexCounts and validity per cell; vertexCounts is per row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellTopology':
    'gpu-spatial-analysis/cell-topology/gpu-cell-topology.ts: publishes cells, distances and counts per input cell record',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellGridPath':
    'gpu-spatial-analysis/cell-topology/gpu-cell-grid-path.ts: publishes path cells, distances and counts per origin/destination row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellMeasures':
    'gpu-spatial-analysis/cell-indexing/gpu-cell-measures.ts: publishes areas, perimeters, edgeLengths, edgeCounts and pentagon flags per cell',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUStudyWindowPreparation':
    'gpu-spatial-analysis/study-window/gpu-study-window-preparation.ts: getViews exposes edges, summary, slabs, sampleCells, sampleCellCount and packed data; sampleCellCount is not a catalog kind and raster overflow is internal and recovered',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUInverseDistanceWeighting':
    'gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting.ts: publishes values and neighbor counts per raster cell',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUFocalStatistics':
    'gpu-spatial-analysis/spatial-interpolation/gpu-focal-statistics.ts: publishes neighborhood statistics and counts per raster cell',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURegionStatisticsReadback':
    'readback adapter; host results',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSkaterRegions':
    'gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions.ts: publishes labels, regionCount and cut logs; regionCount is not a catalog kind and tree loss suppresses results',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURegionPartitionEvaluation':
    'gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation.ts: publishes summary and per-region columns; optional scalar invalidRowCount is not the catalog invalidCount kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalOutlierFactor':
    'gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor.ts: publishes row scores/flags and optional scalar outlierCount, which is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialEmpiricalBayesRates':
    'gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates.ts: publishes spatialRates, smoothedRates, priorMeans and priorVariances per input row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultipleTestingAdjustment':
    'gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment.ts: publishes adjustedPValues and optional scalar finiteTestCount and effectiveFamilySize, which are outside the catalog status kinds',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUKnoxTest':
    'gpu-spatial-analysis/space-time-tests/gpu-knox-test.ts: publishes packed statistics/summary and optional permutationCount, which is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMantelTest':
    'gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts: publishes packed statistics/summary and optional permutationCount, which is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOrdinaryLeastSquares':
    'solver code only (Pick<GPUSolverStatusPort, code>)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialRegressionDiagnostics':
    'solver code only (Pick<GPUSolverStatusPort, code>)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialTwoStageLeastSquares':
    'solver code only (Pick<GPUSolverStatusPort, code>)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialErrorGM':
    'solver code only (Pick<GPUSolverStatusPort, code>)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLagImpacts':
    'gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts.ts: publishes only impacts; optional status is a read-only input solver code',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUClassAssignment':
    'gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment.ts: publishes only per-row class IDs; classCount is an input',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUClassificationFit':
    'gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit.ts: publishes per-class counts/medians/deviations and packed summary',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCatchmentAccessibility':
    'gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts: publishes demand/facility attributes and per-demand validity, not global status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUHuffTradeAreas':
    'gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts: publishes trade areas, probabilities, expectedDemand and per-demand validity only',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegregation':
    'gpu-spatial-analysis/segregation/gpu-segregation.ts: publishes indices and local environment/entropy/dissimilarity/Theil columns; weights loss suppresses weight-dependent values only at the affected scale. Environment-form population diversity and output.invalidUnitCount remain readable. Optional scalar uint32 output.invalidUnitCount counts input units with any negative or non-finite group count before the missing-count policy, separately from weights loss; invalidUnitCount is not the catalog invalidCount kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryResample':
    'gpu-spatial-analysis/trajectory-interpolation/gpu-trajectory-resample.ts: publishes fixed-size sample positions, elevations and times only',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrackSimilarity':
    'gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts: publishes distances, decisions and pairStatus codes per pair',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPearson':
    'GPUEdgeAssociationOutput holds only `statistic` (props reach it through an intersection)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGammaStatistic':
    'GPUEdgeAssociationOutput holds only `statistic` (props reach it through an intersection)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURoseStatistic':
    'gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts: publishes packed float summary and histogram bins per group',
  '@luma-contributors/analysis/dataframe-analysis:GPUWideNumericScan':
    'dataframe-analysis/column-profile/gpu-wide-numeric-scan.ts: caller phase bodies write packed result words to output; no separately named scalar status ports',
  '@luma-contributors/analysis/dataframe-analysis:aggregateInteger64OnGpu':
    'host result after readback; no GPU status view',
  '@luma-contributors/analysis/dataframe-analysis:aggregateInteger64Range':
    'host result after readback; no GPU status view',
  '@luma-contributors/analysis/dataframe-analysis:aggregateDecimal128OnGpu':
    'host result after readback; no GPU status view',
  '@luma-contributors/analysis/dataframe-analysis:aggregateDecimal128Column':
    'host result after readback; no GPU status view',
  '@luma-contributors/analysis/dataframe-analysis:scanWideNumericColumnOnGpu':
    'host result after readback; no GPU status view',
  '@luma-contributors/analysis/dataframe-analysis:diagnoseFloat32Narrowing':
    'CPU report; overflowCount is Float32 range overflow, not capacity',
  '@luma-contributors/analysis/gpu-spatial-analysis:preparePolygonContiguityTopology':
    'CPU function; host diagnostics',
  '@luma-contributors/analysis/gpu-spatial-analysis:computePolygonContiguityOnCPU':
    'CPU function; host diagnostics',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPettittTest':
    'gpu-spatial-analysis/change-points/gpu-pettitt-test.ts: publishes statistic, changeSlice and pValues per cell',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUExpectationBasedSpaceTimeScan':
    'gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan.ts: publishes packed cluster/statistic/summary columns; no named scalar status views',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNearRepeatKnox':
    'gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox.ts: publishes counts by replicate/bin and packed summary by bin',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalKnox':
    'gpu-spatial-analysis/space-time-tests/gpu-local-knox.ts: publishes only packed per-row summary statistics',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUJacquezTest':
    'gpu-spatial-analysis/space-time-tests/gpu-jacquez-test.ts: publishes statistics/summary and optional scalar permutationCount, which is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpaceTimeCrossCorrelation':
    'gpu-spatial-analysis/space-time-statistics/gpu-space-time-cross-correlation.ts: publishes per-cell/lag correlations, p-values and pairCounts; scalar testedCount is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpaceTimeRelativeRisk':
    'caseInvalidCount and controlInvalidCount, not catalog kinds',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpaceTimeKriging':
    'status code per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUEnsembleScores':
    'gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.ts: publishes CRPS and scalar floating-point energy/variogram scores only',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGaussianPredictiveScores':
    'gpu-spatial-analysis/scoring-rules/gpu-gaussian-predictive-scores.ts: publishes only CRPS per observation',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaLogLikelihood':
    'packed [failed, excluded, used] site counts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaFisherScoring':
    'packed [failed, excluded, used] site counts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaSpaceTimeLogLikelihood':
    'packed [failed, excluded, used] site counts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConformalPredictionIntervals':
    'status bits per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeoConformalPredictionIntervals':
    'status bits per query row',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUAreaOfApplicability':
    'one status code word (GPU_AREA_OF_APPLICABILITY_STATUS_*)',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUBrownianBridgeUtilization':
    'gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization.ts: publishes only density per raster cell',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUUtilizationVolumeIsopleths':
    'gpu-spatial-analysis/trajectory-home-range/gpu-utilization-volume-isopleths.ts: publishes per-cell volume/inside and scalar float totalMass, which is not a status kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUModalSplit':
    'gpu-spatial-analysis/trajectory-mobility/gpu-modal-split.ts: publishes per-track/mode shares and per-tripleg predictedModes; input tripleg loss suppresses outputs',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryOutlierFlags':
    'gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-outlier-flags.ts: publishes per-position flags and per-track trackFlaggedCounts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryHampel':
    'gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-hampel.ts: publishes filtered values/flags and optional scalar outlierCount, which is not a catalog kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryTrackRelativeTimes':
    'gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-track-relative-times.ts: publishes only relative times per timestamp',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeographicalDetector':
    'gpu-spatial-analysis/geographical-detector/gpu-geographical-detector.ts: publishes packed floating-point output.summary and optional scalar uint32 output.labelsComplete paired with input labelsComplete. The output copies the supplied partition certificate as 0/1; statistics still describe the incumbent partition. labelsComplete is not the catalog complete kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCategoryPairCounts':
    'gpu-spatial-analysis/spatial-entropy/gpu-category-pair-counts.ts: publishes category-pair matrix and categoryRowCounts, not a global scalar status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialEntropy':
    'gpu-spatial-analysis/spatial-entropy/gpu-spatial-entropy.ts: publishes packed summary, two-word row/link counts and optional category-pair matrix',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUColocationQuotient':
    'gpu-spatial-analysis/categorical-association/gpu-colocation-quotient.ts: publishes global/local quotients and per-category row counts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalCategoryDependence':
    'gpu-spatial-analysis/categorical-association/gpu-local-category-dependence.ts: publishes per-row statistics and per-category row counts',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUAdaptiveBandwidthKernelDensity':
    'gpu-spatial-analysis/point-density/gpu-adaptive-bandwidth-kernel-density.ts: publishes raster values, per-point bandwidth/pilot and scalar float weightSum, which is not a status kind',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURasterFalseDiscoveryRateMask':
    'gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask.ts: publishes per-cell significance and scalar finiteTestCount, effectiveFamilySize and significantCount, which are outside the catalog status kinds',
  '@luma-contributors/analysis/gpu-spatial-analysis:getHealpixCellBoundary':
    'CPU function; no status',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUHealpixPointToCell':
    'gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell.ts: publishes cell keys and optional validity per position',
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPopulationSupportAlignment':
    'packed uint32x4 alignment counts',
  '@luma-contributors/analysis/analysis-core:GPUFittingGradientSums':
    'analysis-core/kernels/gpu-fitting-gradient-sums.ts: publishes only the parameter-indexed sums vector',
  '@luma-contributors/analysis/analysis-core:GPUFittingMinibatchOrder':
    'analysis-core/kernels/gpu-fitting-minibatch-order.ts: publishes only row permutation order',
  '@luma-contributors/analysis/analysis-core:GPUFittingCoordinateEncoder':
    'analysis-core/kernels/gpu-fitting-coordinate-encoder.ts: publishes feature vectors and per-position validity'
});
