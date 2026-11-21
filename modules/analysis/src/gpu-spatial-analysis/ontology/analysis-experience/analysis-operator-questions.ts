// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisCapabilityCatalog,
  queryGPUAnalysisOperators
} from '../all-entry-points/analysis-operator-catalog';
import type {
  GPUAnalysisOperator,
  GPUSpatialAnalysisCapability,
  GPUSpatialAnalysisDomain,
  GPUSpatialAnalysisStage
} from '../spatial-analysis-ontology';
import {GPU_ANALYSIS_QUESTIONS} from './analysis-experience-types';
import type {GPUAnalysisQuestion} from './analysis-experience-types';

/**
 * Most questions offered for one operator. A fourth question would stop being a recommendation and
 * start being a list of everything the operator could loosely be used for.
 */
const MAXIMUM_QUESTIONS_PER_OPERATOR = 3;

type QuestionList = readonly GPUAnalysisQuestion[];

/**
 * Operators whose question the rules below get wrong or cannot see. The first question is the
 * primary one. Overrides also bypass the stage veto, so an analysis that lives in a transforming
 * family is named here. A second question is listed only where a user asking it would be glad to
 * be offered the operator.
 */
const OPERATOR_QUESTION_OVERRIDES: ReadonlyMap<string, QuestionList> = new Map([
  // Methods where the statistic is a test but the user wants a map of places.
  ['GPUEmergingHotSpots', ['what-changed', 'where-significant']],
  ['GPUSpaceTimeLocalMoran', ['where-significant', 'what-changed']],
  ['GPUSpaceTimeRelativeRisk', ['where-concentrated', 'where-significant']],
  ['GPUExpectationBasedSpaceTimeScan', ['where-significant', 'what-changed']],
  ['addSpaceTimeHotSpotsRecipe', ['where-significant', 'what-changed']],
  ['addRateClusterMapRecipe', ['where-significant', 'how-much-per-area']],
  ['addExcessRiskRecipe', ['how-much-per-area', 'where-significant']],
  ['GPUSpatialScanStatistic', ['where-significant', 'where-concentrated']],
  ['GPUDBSCAN', ['which-groups-or-regions', 'where-concentrated']],
  ['GPUHDBSCAN', ['which-groups-or-regions', 'where-concentrated']],
  ['GPUOPTICS', ['which-groups-or-regions', 'where-concentrated']],
  ['GPUSpaceTimeHawkes', ['where-concentrated', 'what-changed']],
  // A utilisation distribution is a density of where a tracked animal or vehicle spends time.
  ['GPUBrownianBridgeUtilization', ['where-concentrated', 'how-it-moves']],
  ['GPUUtilizationVolumeIsopleths', ['where-concentrated', 'how-it-moves']],
  // Joins: only the polygon join is an aggregation; the rest relate or prepare.
  ['GPUSpatialJoinCandidates', ['prepare-data']],
  ['GPUSpatialJoinPrepared', ['prepare-data']],
  // A key join that lines population rows up with their support units, not a per-area measure.
  ['GPUPopulationSupportAlignment', ['prepare-data']],
  ['GPUPairGather', ['prepare-data']],
  ['GPUOffsetExpansion', ['prepare-data']],
  ['GPUBoundsFilter', ['prepare-data']],
  ['GPUSpatialPredicateJoin', ['what-is-near-or-reachable']],
  ['GPUPointInPolygonJoin', ['how-much-per-area']],
  // Change of support moves totals between areas; it lives in a transforming family.
  ['GPUArealInterpolation', ['how-much-per-area']],
  ['GPUPycnophylacticInterpolation', ['how-much-per-area', 'value-where-unmeasured']],
  ['GPUSpatialCrossValidatedOLS', ['how-related']],
  ['GPUSpatialInteraction', ['how-it-moves', 'how-related']],
  // Raster layers that say what is on the ground, or how a place scores, sit in transforming families.
  ['GPUWeightedOverlay', ['what-is-at-each-place']],
  ['GPUCompositeScore', ['what-is-at-each-place']],
  // Reductions across arbitrary bands describe each cell; dates are not part of the input contract.
  ['GPURasterCellStatistics', ['what-is-at-each-place']],
  // Mean-difference merging produces groups of adjacent raster regions.
  ['GPURasterRegionGraph', ['which-groups-or-regions']],
  // Cell minima/maxima measure terrain, surface and canopy; height measures distance above ground.
  ['GPUPointCloudSurfaces', ['shape-of-the-surface']],
  ['GPUPointCloudHeight', ['shape-of-the-surface']],
  // These contributors reduce rows to cell counts, sums and means, rather than construct cell keys.
  ['GPUHealpixCellAggregation', ['how-much-per-area']],
  ['GPUValueCube', ['how-much-per-area']],
  ['GPURasterFocalPairs', ['how-related']],
  // A transport distance between two mass fields compares them, as a period comparison does.
  ['GPUSinkhornDivergence', ['what-changed']],
  ['GPURasterSnapshotCovariance', ['how-related', 'what-changed']],
  ['GPURasterEOFProjection', ['what-changed', 'how-related']],
  ['GPUGeographicDistribution', ['where-concentrated']],
  ['GPUNetworkCentrality', ['what-is-near-or-reachable']],
  ['GPUNetworkLouvain', ['which-groups-or-regions']],
  ['GPUTerrainWatersheds', ['how-it-moves', 'which-groups-or-regions']],
  ['GPUTerrainCatchmentSummary', ['how-much-per-area']],
  ['GPUPointCloudChange', ['what-changed']],
  ['GPUSolarPosition', ['prepare-data']],
  ['addLandscapeMetricsRecipe', ['how-much-per-area']],
  ['addClusterAndOutlineRecipe', ['which-groups-or-regions']],
  ['addBuildingTessellationRecipe', ['prepare-data']],
  ['addBuildingTessellationReferenceRecipe', ['prepare-data']],
  ['addLocalBuildingTessellationRecipe', ['prepare-data']],
  ['addFleetDwellRecipe', ['how-it-moves', 'how-much-per-area']],
  ['addPointsInPolygonsChoroplethRecipe', ['how-much-per-area']]
] as [string, QuestionList][]);

/** One name rule. An exempt rule is trusted even in a family that builds or converts data. */
type NameRule = {
  pattern: RegExp;
  questions: QuestionList;
  /** True only where the analysis is known to live in a building or converting family. */
  stageExempt?: boolean;
};

/**
 * Name rules, tried in order against the export name without its `GPU` or `add` prefix and its
 * `Recipe` suffix. The first rule that matches gives the questions. Order matters: a narrow rule
 * ahead of a broad one is how "preparation" words are kept from swallowing analyses and the other
 * way round. Names are the best signal because capability families mix purposes.
 */
const NAME_RULES: readonly NameRule[] = [
  // Preparation that shares words with analyses, and plumbing that only looks like analysis.
  {
    pattern:
      /GapFill|Prep$|Preparation|Selector$|Renumbering|Export$|Encoder|Decoder|StandardizedIndexCalibration|TerrainBreach|DepressionHierarchy|SpikeRepair|RGBDecode|DEMCoregistration|EmbeddingDequantize|EmbeddingCovariance|EmbeddingProjection|EmbeddingPyramidLevel|SurfacePreparation|StudyWindowPreparation|PolygonRasterization|MapColoring|RasterSieve|ClassAssignment|ClassificationFit|FuzzyContiguity|IslandAttachment|FocalStatistics|RasterSkeleton|RasterMedialAxis|KeyLookup|ColumnProfile|NetworkCostWeights|RollingMean|Whittaker|HANTS|TemporalResample|Composite|BestAvailablePixel|StreamingAccumulator|TrajectoryRestriction|TrackQuality|TrajectorySortCheck|^CellLevelSelection$|^PickRegionMask$|^RegionMask$|^AssuncaoReisStandardization$|^ShapeDescriptors$|^GeometryMeasures$|^NeighborhoodSummary$|^SpatialCrossValidation$|^SpatialWeightsTraces$|^AdjacencyMatrix(Order)?$|^NetworkAnalyticsColumns$|^RasterTemporal$/,
    questions: ['prepare-data']
  },
  {
    pattern:
      /Trajectory(Kalman|Hampel|OutlierFlags|Thinning|Resample|Split|Playhead|TrackRelativeTimes)/,
    questions: ['prepare-data']
  },
  {
    pattern: /PointCloud(Ground|SMRF|ICP|StatisticalFilter|VoxelMerge|Neighborhoods)/,
    questions: ['prepare-data']
  },
  {
    pattern: /SpatialBalancedSample|ContinuousSpatialSample|RandomPointsIn|RandomPointsOn/,
    questions: ['prepare-data']
  },
  // Groups and regions.
  {
    pattern:
      /SpatialKMeans|Skater|Redcap|AZPRegions|MaxPRegions|WardRegions|MinimumSpanningTree|RegionPartition|SimilarLocations|Louvain|ConnectedComponents|NetworkStructure|NetworkStatistics|GroupGeometry|GroupConvexHull|GroupConcaveHull|EmbeddingKMeans/,
    questions: ['which-groups-or-regions']
  },
  {
    pattern: /WatershedSegmentation/,
    questions: ['which-groups-or-regions'],
    stageExempt: true
  },
  {pattern: /SpaceTimeHotSpots/, questions: ['where-significant', 'what-changed']},
  // What is on the ground, or how a place scores.
  {
    pattern: /RasterEmbedding(KNearestClassifier|LinearProbe|Similarity|NearestPixels)/,
    questions: ['what-is-at-each-place']
  },
  {
    pattern: /Raster(NDWI|NBR|EVI|SAVI|SpectralUnmixing|Texture)/,
    questions: ['what-is-at-each-place'],
    stageExempt: true
  },
  // Changes.
  {
    pattern:
      /EmergingHotSpots|PeriodComparison|ChangeDetection|RegionalRasterChange|EmbeddingChange|PointCloudChange|Pettitt|ChangePoints|SeasonalBreak|COLDMonitor|BFASTMonitor|Markov|TransitionMatrix|TransitionDiagnostics|ClimateAnomaly|ClimateRunAccumulator|StandardizedIndex|Phenology|HarmonicRegression/,
    questions: ['what-changed']
  },
  {pattern: /CellTableCompare/, questions: ['what-changed'], stageExempt: true},
  // Movement, flow and drainage.
  {
    pattern: /TrajectoryEncounters|TrajectoryClosestApproach|ClockEncounters/,
    questions: ['what-is-near-or-reachable', 'how-it-moves']
  },
  {pattern: /TerrainCatchmentSummary|DrainageNetwork/, questions: ['how-it-moves']},
  {
    pattern: /ParticleAdvection|Streamlines|LineIntegralConvolution|LagrangianFlowMap|Lyapunov/,
    questions: ['how-it-moves'],
    stageExempt: true
  },
  {
    pattern:
      /Trajectory|Track|MapMatching|FlowAggregation|MapMatchedEdgeVolumes|RadiationFlow|KernelFlowSmoothing|Staypoint|Mobility|Brownian|UtilizationVolume|ModalSplit|ZoneEvents|FleetDwell|NetworkFlow|EdgeBundling|TerrainFlow|TerrainBranchingFlow|TerrainTiledFlow|TerrainStreamOrder|TerrainStreamNetwork|TerrainHeightAboveDrainage|TerrainHydrologicIndices|CellFlowTopology/,
    questions: ['how-it-moves']
  },
  // Relationships and models.
  {pattern: /GeographicalDetector/, questions: ['how-related']},
  {
    pattern:
      /Regression|OrdinaryLeastSquares|TwoStageLeastSquares|Probit|InstrumentalVariables|SeeminglyUnrelated|SpatialErrorGM|LagMaximumLikelihood|ErrorMaximumLikelihood|LagImpacts|Collinearity|SpatialPearson|CrossCorrelation|LaggedCorrelation|SpatialInteraction|PanelRegression/,
    questions: ['how-related']
  },
  {pattern: /RasterEmbeddingZoneMean/, questions: ['how-much-per-area']},
  // Interpolation, prediction and variograms.
  {
    pattern:
      /Kriging|LocalRBF|NaturalNeighbor|InverseDistanceWeighting|ConditionalGaussianSimulation|EnsembleScores|PredictiveScores|Vecchia|ConformalPrediction|AreaOfApplicability|Variogram|NestedInterpolation|InterpolationDiagnostics|CoregionalizationFit/,
    questions: ['value-where-unmeasured']
  },
  // Density and point patterns.
  {
    pattern: /PointPatternEnvelope|RipleyEnvelope/,
    questions: ['where-concentrated', 'where-significant']
  },
  {
    pattern:
      /KernelDensity|PointDensity|LineDensity|DotDensity|Ripley|PointPatternIndices|MarkedPointStatistics|KFunction|InhomogeneousK|InhomogeneousPointPattern/,
    questions: ['where-concentrated']
  },
  // Clusters, outliers and association: exploratory scores and tests alike.
  {
    pattern:
      /Knox|Mantel|Jacquez|GlobalSpatialStatistics|GlobalPermutationTest|GammaStatistic|RoseStatistic|DiggleTest|HotSpot|LocalMoran|LocalGeary|LISA|LocalPermutationTest|LocalOutlierFactor|LocalSpatialHeteroscedasticity|LOSH|LocalVariability|FalseDiscovery|MultipleTesting|ColocationQuotient|LocalCategoryDependence|LocalCollinearity|Correlogram|CategoryPairCounts|SpatialEntropy/,
    questions: ['where-significant']
  },
  // Reachability and proximity. A buffer is the standard "within a distance" step, so it keeps
  // this question although it is built by a constructing or transforming family.
  {
    pattern: /^(Point)?Buffer(Selection|Surface)?$/,
    questions: ['what-is-near-or-reachable'],
    stageExempt: true
  },
  {
    pattern:
      /NearestFeatureJoin|NeighborSearch|Reachab|Routing|PathExtraction|AlternativeRoutes|KShortestPaths|ServiceAreas|Isochrones|CostMatrix|CostTable|NetworkAccessibility|RationalAccess|PotentialPathArea|TravelTimePercentiles|DriveTimeCatchment|StraightLineCatchments|CatchmentAccessibility|HuffTradeAreas|DistanceField|CostDistance|WalkingCost|GeodesicPairs|PatchProximity|ResistanceConnectivity|VisibilityGraph|BuildingAdjacency|Facility|LocationAllocation|NetworkNeighborhood|NetworkDisruption|NetworkCentrality|NetworkTimeRouting/,
    questions: ['what-is-near-or-reachable']
  },
  // Per-area totals, rates and shares.
  {
    pattern:
      /ExcessRisk|SpatialRate|EmpiricalBayes|ZonalStatistics|ZonalDistribution|RegionStatistics|ExposureByZone|GroupStatistics|PointsInPolygons|LineLengthPerPolygon|Fractional(Zonal|Categorical|Coverage)|Cell(Aggregation|Rollup|Pyramid)|ChangeOfSupport|Dasymetric|Segregation|Inequality|PatchMetrics|PopulationSupport|RasterJoin/,
    questions: ['how-much-per-area']
  },
  // The form of a surface: terrain, relief, visibility, profiles and contours.
  {
    pattern: /Relief|LocalDominance|TextureShading|Isolines|Isoband/,
    questions: ['shape-of-the-surface'],
    stageExempt: true
  },
  {
    pattern:
      /Terrain(Derivatives|Curvature|Ruggedness|VectorRuggedness|Contours|TopographicPosition|WeissLandforms|CriticalPoints|Summits|PeakSnap|PeakProminence|Viewshed|LineOfSight|CumulativeViewshed|Horizon|CastShadow)|Geomorphons|ProfilePeaks|PointHorizon|Solar|Isovist|StreetProfiles|RasterProfile|PointCloud(Features|Surfaces|Height)/,
    questions: ['shape-of-the-surface']
  }
];

/** Capability family text that decides a question for an operator that no name rule recognised. */
const CAPABILITY_RULES: readonly (readonly [RegExp, QuestionList])[] = [
  [/hydrology|flow-visualization|movement|trajector/, ['how-it-moves']],
  [/point-patterns|local-variability|autocorrelation/, ['where-significant']],
  [/raster-time-series/, ['what-changed']]
];

/**
 * Stages of a family that builds or converts data. An operator of such a family answers
 * `prepare-data` unless an override or an exempt name rule says otherwise, so that a new
 * preparation operator whose name contains an analytical word does not pass for an analysis.
 */
const PREPARATION_STAGES: ReadonlySet<GPUSpatialAnalysisStage> = new Set([
  'bridge',
  'construct',
  'transform',
  'index'
]);

let capabilityById: ReadonlyMap<string, GPUSpatialAnalysisCapability> | undefined;

function getCapabilityById(identifier: string): GPUSpatialAnalysisCapability | undefined {
  capabilityById ??= new Map(
    getGPUAnalysisCapabilityCatalog().map(capability => [capability.id, capability])
  );
  return capabilityById.get(identifier);
}

/** The export name without the `GPU` or `add` prefix and the `Recipe` suffix. */
function getBaseName(exportName: string): string {
  return exportName.replace(/^(GPU|add)/, '').replace(/Recipe$/, '');
}

/**
 * The question a whole family answers when nothing more specific is known. Stages that build or
 * convert data lead to preparation; the analysing stages follow the subject area.
 */
function getDefaultQuestion(
  domain: GPUSpatialAnalysisDomain,
  stage: GPUSpatialAnalysisStage
): GPUAnalysisQuestion {
  const isAnalysing = stage === 'analyze' || stage === 'infer';
  switch (domain) {
    case 'density-and-interpolation':
      if (stage === 'infer') return 'value-where-unmeasured';
      return stage === 'aggregate' || isAnalysing ? 'where-concentrated' : 'prepare-data';
    case 'spatial-statistics':
      return isAnalysing ? 'where-significant' : 'prepare-data';
    case 'spatial-models':
      return isAnalysing ? 'how-related' : 'prepare-data';
    case 'movement':
      return isAnalysing || stage === 'relate' ? 'how-it-moves' : 'prepare-data';
    case 'regional-analysis':
      if (stage === 'aggregate') return 'how-much-per-area';
      return isAnalysing ? 'which-groups-or-regions' : 'prepare-data';
    case 'proximity':
      return stage === 'relate' ? 'what-is-near-or-reachable' : 'prepare-data';
    case 'network-analysis':
      return isAnalysing || stage === 'relate' || stage === 'aggregate'
        ? 'what-is-near-or-reachable'
        : 'prepare-data';
    case 'terrain-analysis':
      return isAnalysing || stage === 'relate' ? 'shape-of-the-surface' : 'prepare-data';
    case 'point-cloud-analysis':
      return isAnalysing ? 'shape-of-the-surface' : 'prepare-data';
    case 'raster-analysis':
      if (stage === 'aggregate') return 'how-much-per-area';
      return isAnalysing ? 'shape-of-the-surface' : 'prepare-data';
    case 'dataframe-analysis':
      if (stage === 'aggregate') return 'how-much-per-area';
      return stage === 'infer' ? 'value-where-unmeasured' : 'prepare-data';
    case 'discrete-global-grids':
      return stage === 'aggregate' ? 'how-much-per-area' : 'prepare-data';
    case 'execution':
    case 'geometry':
    case 'topology':
    case 'workflow':
      return 'prepare-data';
  }
}

/** Removes repeats and keeps the first `MAXIMUM_QUESTIONS_PER_OPERATOR` questions in order. */
function limitQuestions(questions: QuestionList): readonly GPUAnalysisQuestion[] {
  return [...new Set(questions)].slice(0, MAXIMUM_QUESTIONS_PER_OPERATOR);
}

/** Questions from the capability family and data kinds, for an operator no name rule recognised. */
function resolveFromCapability(operator: GPUAnalysisOperator): QuestionList {
  const capability = getCapabilityById(operator.primaryCapability);
  const identifier = operator.primaryCapability;
  if (!capability) return ['prepare-data'];
  if (PREPARATION_STAGES.has(capability.stage)) return ['prepare-data'];
  for (const [pattern, questions] of CAPABILITY_RULES) {
    if (pattern.test(identifier)) return questions;
  }
  const isAnalysing = capability.stage === 'analyze' || capability.stage === 'infer';
  const inputs = operator.inputs ?? [];
  // Rows of tracks are about movement whatever the family says.
  if (inputs.includes('trajectories') && isAnalysing) return ['how-it-moves'];
  if (inputs.includes('time-series') && isAnalysing) return ['what-changed'];
  // Stochastic or model-based results of the statistics families are tests and fits.
  if (capability.domain === 'spatial-statistics' && operator.semantics === 'stochastic') {
    return ['where-significant'];
  }
  if (capability.domain === 'regional-analysis' && operator.semantics === 'model-based') {
    return ['which-groups-or-regions'];
  }
  return [getDefaultQuestion(capability.domain, capability.stage)];
}

/**
 * A second question from the families named in the operator's tags: an operator tagged with a
 * per-area aggregation family also tells how much there is in each area, which is the one
 * second question that is always welcome. Other tagged families are context, not a question.
 */
function resolveFromTags(operator: GPUAnalysisOperator): QuestionList {
  for (const tag of operator.tags ?? []) {
    const capability = getCapabilityById(tag);
    if (
      capability &&
      getDefaultQuestion(capability.domain, capability.stage) === 'how-much-per-area'
    ) {
      return ['how-much-per-area'];
    }
  }
  return [];
}

/**
 * Returns the questions an operator answers, primary first, between one and three. An explicit
 * override wins; otherwise name rules (vetoed to preparation in a building or converting family
 * unless the rule is exempt), then the family and data kinds, then the subject area. Never empty,
 * so a new operator that no rule has seen still lands somewhere sensible.
 */
export function getGPUAnalysisOperatorQuestions(
  operator: GPUAnalysisOperator
): readonly GPUAnalysisQuestion[] {
  const override = OPERATOR_QUESTION_OVERRIDES.get(operator.exportName);
  if (override) return limitQuestions(override);

  const capability = getCapabilityById(operator.primaryCapability);
  const isPreparationFamily = capability !== undefined && PREPARATION_STAGES.has(capability.stage);
  const baseName = getBaseName(String(operator.exportName));
  for (const rule of NAME_RULES) {
    if (!rule.pattern.test(baseName)) continue;
    // The first rule is preparation itself and needs no veto; any other rule yields to the stage.
    if (isPreparationFamily && !rule.stageExempt) return ['prepare-data'];
    return limitQuestions([...rule.questions, ...resolveFromTags(operator)]);
  }
  return limitQuestions([...resolveFromCapability(operator), ...resolveFromTags(operator)]);
}

/** Options of {@link queryGPUAnalysisOperatorsByQuestion}. */
export type GPUAnalysisOperatorQuestionQueryOptions = {
  /** Only operators whose primary (first) question is the one asked. */
  primaryOnly?: boolean;
  /** Includes helpers, which answer no question of their own and are left out by default. */
  includeHelpers?: boolean;
};

/** Finds the operators of any entry point that answer a question. */
export function queryGPUAnalysisOperatorsByQuestion(
  question: GPUAnalysisQuestion,
  options: GPUAnalysisOperatorQuestionQueryOptions = {}
): readonly GPUAnalysisOperator[] {
  if (!GPU_ANALYSIS_QUESTIONS.includes(question)) {
    throw new Error(
      `queryGPUAnalysisOperatorsByQuestion: unknown question "${String(question)}"; accepted ids are ${GPU_ANALYSIS_QUESTIONS.join(', ')}`
    );
  }
  return queryGPUAnalysisOperators({includeHelpers: options.includeHelpers === true}).filter(
    operator => {
      const questions = getGPUAnalysisOperatorQuestions(operator);
      return options.primaryOnly ? questions[0] === question : questions.includes(question);
    }
  );
}

/** How many operators answer one question. */
export type GPUAnalysisQuestionCoverage = {
  question: GPUAnalysisQuestion;
  /** Operators with this as their first question. */
  primaryCount: number;
  /** Operators with this question anywhere in their list; includes the primary ones. */
  totalCount: number;
};

/** Counts operators per question, in the order of `GPU_ANALYSIS_QUESTIONS`. Helpers are excluded. */
export function getGPUAnalysisQuestionCoverage(): readonly GPUAnalysisQuestionCoverage[] {
  const operators = queryGPUAnalysisOperators();
  const primaryCounts = new Map<GPUAnalysisQuestion, number>();
  const totalCounts = new Map<GPUAnalysisQuestion, number>();
  for (const operator of operators) {
    const questions = getGPUAnalysisOperatorQuestions(operator);
    primaryCounts.set(questions[0], (primaryCounts.get(questions[0]) ?? 0) + 1);
    for (const question of questions) {
      totalCounts.set(question, (totalCounts.get(question) ?? 0) + 1);
    }
  }
  return GPU_ANALYSIS_QUESTIONS.map(question => ({
    question,
    primaryCount: primaryCounts.get(question) ?? 0,
    totalCount: totalCounts.get(question) ?? 0
  }));
}
