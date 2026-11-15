// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {OPERATOR_ANNOTATIONS} from './all-entry-points/operator-annotations';
import type {GPUAnalysisOperator} from './spatial-analysis-ontology';

/**
 * Descriptor of an operator added to the catalog after the original 261. It names the original
 * capability family that owns it, which is not changed, so the original lists stay as consumers
 * saw them.
 */
type SpatiotemporalReviewOperatorOverride = Pick<
  GPUAnalysisOperator,
  'primaryCapability' | 'inputs' | 'outputs' | 'coordinateModels' | 'cardinality' | 'summary'
> &
  Partial<
    Pick<
      GPUAnalysisOperator,
      'dimensionality' | 'topology' | 'semantics' | 'deterministic' | 'status' | 'readback'
    >
  > & {
    contract: string;
    evidence: GPUAnalysisOperator['evidence']['status'];
    tests: string;
    documentation: string;
  };

const TEST_ROOT = 'modules/analysis/test/gpu-spatial-analysis';
const DOCUMENTATION_ROOT = 'docs/api/gpu-spatial-analysis';

/**
 * Operators of `gpu-spatial-analysis` that read or produce times, sequences or tracks. The
 * original operator table tags each `dimensionality: '2d+time'` (roadmap TC-08), so a query for
 * time-aware operators finds them. Operators of the other entry points carry the tag in
 * `ENTRY_POINT_OPERATOR_DETAILS`.
 */
export const SPATIOTEMPORAL_OPERATOR_NAMES: ReadonlySet<string> = new Set([
  // Space-time tests and statistics.
  'GPUKnoxTest',
  'GPUMantelTest',
  'GPUNearRepeatKnox',
  'GPULocalKnox',
  'GPUJacquezTest',
  'GPUSpaceTimeLocalMoran',
  'GPUSpaceTimeCrossCorrelation',
  'GPUSpaceTimeKriging',
  'GPUSpaceTimeHawkes',
  'GPUSpaceTimeKernelDensity',
  'GPUSpaceTimeRelativeRisk',
  // Distribution dynamics and panels.
  'GPUEmergingHotSpots',
  'GPUTransitionMatrix',
  'GPUSpatialMarkov',
  'GPULISAMarkov',
  'GPURankMarkov',
  'GPUTransitionDiagnostics',
  'GPUSpatialPanelRegression',
  'GPUSpatialPanelErrorMaximumLikelihood',
  'GPUSpatialPanelLagMaximumLikelihood',
  // Trajectories.
  'GPUTrajectoryMetrics',
  'GPUTrajectoryPlayhead',
  'GPUTrajectoryResample',
  'GPUZoneEvents',
  'GPUTrajectoryEncounters',
  'GPUTrajectoryClosestApproach',
  'GPUTrackSimilarity',
  'GPUTrackSimilaritySearch',
  'GPUTrajectoryKalman',
  'GPUTrajectoryRestriction',
  'GPUTrajectoryStaypoints',
  'GPUTrajectoryLocations',
  'GPUTrajectorySegments',
  'GPUMobilityIndicators',
  'GPUModalSplit',
  'GPUBrownianBridgeUtilization',
  'GPUUtilizationVolumeIsopleths',
  'GPUTrajectoryRecursions',
  'GPUTrajectoryOutlierFlags',
  'GPUTrajectoryHampel',
  'GPUTrajectorySplit',
  'GPUTrajectoryThinning',
  'GPUTrajectoryTrackRelativeTimes',
  'addTrajectorySortCheck',
  'addClockEncounters',
  // Temporal recipes.
  'addSpaceTimeHotSpotsRecipe',
  'addPeriodComparisonRecipe',
  'addFleetDwellRecipe',
  'addFleetDwellZoneEventsRecipe'
]);

const PACKAGE = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
const SPACE_TIME_FAMILY = 'spatial-autocorrelation-and-inference';
const PREDICTION_FAMILY = 'coverage-local-prediction';
const MOVEMENT_FAMILY = 'movement-analysis';
const REGRESSION_FAMILY = 'expansion-regression';

const STX08_DESCRIPTOR = {
  primaryCapability: SPACE_TIME_FAMILY,
  dimensionality: '2d',
  topology: 'fixed',
  semantics: 'approximate',
  deterministic: true,
  evidence: 'partial',
  documentation: `${DOCUMENTATION_ROOT}/geographical-detector-redcap-entropy.md`
} as const;

const RTS08_DESCRIPTOR = {
  primaryCapability: 'dggs-indexing-and-topology',
  dimensionality: '2d',
  topology: 'fixed',
  semantics: 'approximate',
  deterministic: true,
  evidence: 'partial',
  documentation: `${DOCUMENTATION_ROOT}/healpix-cells-dggs-json.md`
} as const;

const SPACE_TIME_TEST_DESCRIPTOR = {
  primaryCapability: SPACE_TIME_FAMILY,
  inputs: ['spatial-weights', 'time-series', 'parameters'],
  outputs: ['summary'],
  coordinateModels: ['topological'],
  dimensionality: '2d+time',
  topology: 'fixed',
  cardinality: 'fixed-sized',
  semantics: 'stochastic',
  evidence: 'partial',
  tests: `${TEST_ROOT}/space-time-tests/space-time-statistics.node.spec.ts`,
  documentation: `${DOCUMENTATION_ROOT}/operators-dynamics.md`
} as const;

const MOVEMENT_SEMANTICS_DOCUMENTATION = `${DOCUMENTATION_ROOT}/movement-semantics.md`;
const CLEANING_TESTS = `${TEST_ROOT}/trajectory-cleaning/trajectory-cleaning.spec.ts`;
const MOBILITY_TESTS = `${TEST_ROOT}/trajectory-mobility/mobility-indicators.spec.ts`;

const MOB07_HOME_RANGE_DOCUMENTATION = `${DOCUMENTATION_ROOT}/home-range-and-recursion-mob07.md`;
const MOB07_TESTS = `${TEST_ROOT}/trajectory-home-range/home-range-recursion-mob07.spec.ts`;

const TRAJECTORY_ROW_DESCRIPTOR = {
  primaryCapability: MOVEMENT_FAMILY,
  inputs: ['trajectories', 'parameters'],
  coordinateModels: ['planar', 'spherical'],
  dimensionality: '2d+time',
  topology: 'fixed',
  evidence: 'partial',
  documentation: MOVEMENT_SEMANTICS_DOCUMENTATION
} as const;

const SPATIOTEMPORAL_REVIEW_OPERATOR_OVERRIDES: Readonly<
  Record<string, SpatiotemporalReviewOperatorOverride>
> = {
  GPUGaussianRandomField: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['approximation'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-interpolation/gaussian-random-field.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/gaussian-random-field.md`,
    summary:
      'Stationary Gaussian fields by published GPUFFT2D circulant embedding or finite spectral modes, with indexed local kriging of residuals for hard conditioning (Dietrich and Newsam; Hesse et al.).',
    contract: 'GPUGaussianRandomFieldProps'
  },
  GPULocationChangePoints: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['time-series', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    evidence: 'partial',
    tests: `${TEST_ROOT}/change-points/stx06-location-change-points.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/change-points-and-expectation-based-scan.md`,
    summary:
      'Per-location optimal partitioning of a space-time cube into segments under a mean, variance, Poisson or slope cost.',
    contract: 'GPULocationChangePointsProps'
  },
  GPUPettittTest: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['time-series', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['none'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    evidence: 'partial',
    tests: `${TEST_ROOT}/change-points/stx06-location-change-points.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/change-points-and-expectation-based-scan.md`,
    summary:
      'Per-location Pettitt rank test for one change in a series, with its approximate p-value.',
    contract: 'GPUPettittTestProps'
  },
  GPUExpectationBasedSpaceTimeScan: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['points', 'time-series', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar', 'spherical'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    evidence: 'partial',
    tests: `${TEST_ROOT}/scan-statistics/stx07-expectation-based-scan.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/change-points-and-expectation-based-scan.md`,
    summary:
      'Expectation-based Poisson scan over space-time cylinders that end at the latest period, against a caller-supplied baseline.',
    contract: 'GPUExpectationBasedSpaceTimeScanProps'
  },
  GPUNearRepeatKnox: {
    ...SPACE_TIME_TEST_DESCRIPTOR,
    summary:
      'Knox pair counts over a table of distance bands by time bands with a permutation null per cell (near-repeat analysis).',
    contract: 'GPUNearRepeatKnoxProps'
  },
  GPULocalKnox: {
    ...SPACE_TIME_TEST_DESCRIPTOR,
    cardinality: 'source-sized',
    summary:
      'Per-event local Knox count of spatial neighbors that are also close in time, with conditional permutation pseudo p-values.',
    contract: 'GPULocalKnoxProps'
  },
  GPUJacquezTest: {
    ...SPACE_TIME_TEST_DESCRIPTOR,
    summary:
      'Jacquez k-nearest-neighbor space-time interaction test: events whose spatial and temporal neighbors coincide, tested by time permutation.',
    contract: 'GPUJacquezTestProps'
  },
  GPUSpaceTimeLocalMoran: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['time-series', 'spatial-weights', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'stochastic',
    evidence: 'partial',
    tests: `${TEST_ROOT}/space-time-statistics/space-time-statistics.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/space-time-statistics.md`,
    summary:
      'Lagged and differential local Moran of one period of a space-time cube, tested by conditional permutation (GeoDa recipes).',
    contract: 'GPUSpaceTimeLocalMoranProps'
  },
  GPUSpaceTimeCrossCorrelation: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['time-series', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/space-time-statistics/space-time-cross-correlation.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/space-time-statistics.md`,
    summary:
      'Time-series cross-correlation per location over a set of lags, with Fisher-z p-values and false discovery control across locations and lags.',
    contract: 'GPUSpaceTimeCrossCorrelationProps'
  },
  GPUSpaceTimeKernelDensity: {
    primaryCapability: 'density-and-interpolation',
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['surface', 'summary'],
    coordinateModels: ['planar', 'grid'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/point-density/space-time-kernel-density.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/space-time-density-and-k-function.md`,
    summary:
      'Space-time kernel density on a voxel cube with fixed bandwidths and uniform edge correction (sparr spattemp.density), one thread per voxel over a grid index with a time-sorted run per cell.',
    contract: 'GPUSpaceTimeKernelDensityProps'
  },
  GPUSpaceTimeRelativeRisk: {
    primaryCapability: 'density-and-interpolation',
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['surface', 'summary'],
    coordinateModels: ['planar', 'grid'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/point-density/space-time-kernel-density.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/space-time-density-and-k-function.md`,
    summary:
      'Log relative risk of two space-time kernel densities, cases over time-varying controls (sparr spattemp.risk).',
    contract: 'GPUSpaceTimeRelativeRiskProps'
  },
  GPUSpaceTimeKriging: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-interpolation/space-time-kriging.node.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/interpolation-expansion.md`,
    summary:
      'Local space-time kriging with one neighbor search per query over a per-frame space-time covariance model.',
    contract: 'GPUSpaceTimeKrigingProps'
  },
  GPUEnsembleScores: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/scoring-rules/gpu-ensemble-scores.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/scoring-rules.md`,
    summary:
      'Proper scoring rules of an ensemble forecast against an observed field: CRPS per site, energy score and variogram score.',
    contract: 'GPUEnsembleScoresProps'
  },
  GPUGaussianPredictiveScores: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['attributes'],
    outputs: ['attributes'],
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/scoring-rules/gpu-ensemble-scores.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/scoring-rules.md`,
    summary:
      'Closed-form continuous ranked probability score of a Gaussian predictive distribution, for kriging predictions.',
    contract: 'GPUGaussianPredictiveScoresProps'
  },
  GPUSpaceTimeHawkes: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['points', 'time-series', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    evidence: 'partial',
    tests: `${TEST_ROOT}/space-time-hawkes/stm08-space-time-hawkes.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/space-time-hawkes.md`,
    summary:
      'Space-time Hawkes process (exponential time kernel, Gaussian offspring displacement, constant background) fitted by expectation-maximization with bounded parent windows over a time-sorted event table, log-sum-exp responsibilities, unrolled gated iterations and per-event background probability.',
    contract: 'GPUSpaceTimeHawkesProps'
  },
  GPUSinkhornDivergence: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['surface', 'parameters'],
    outputs: ['summary'],
    coordinateModels: ['grid'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/optimal-transport/stm09-sinkhorn-divergence.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/sinkhorn-divergence.md`,
    summary:
      'Debiased entropic optimal transport (Sinkhorn) divergence between two mass fields on a grid, in the log domain with separable 1D log-sum-exp blurs, an epsilon schedule and a convergence status.',
    contract: 'GPUSinkhornDivergenceProps'
  },
  GPUVecchiaLogLikelihood: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-interpolation/vecchia-log-likelihood.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/vecchia-likelihood.md`,
    summary:
      'Vecchia (nearest-neighbour Gaussian process) log-likelihood over ordered conditioning sets, solved with per-frame covariance parameters through local kriging, with a binary64-tested float32 failure and nugget-floor rule.',
    contract: 'GPUVecchiaLogLikelihoodProps'
  },
  GPUVecchiaFisherScoring: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-interpolation/vecchia-fisher-scoring.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/vecchia-likelihood.md`,
    summary:
      'Score and expected information of a Vecchia likelihood in log sill, log range and log nugget (Guinness 2019), for a CPU-driven Fisher scoring loop.',
    contract: 'GPUVecchiaFisherScoringProps'
  },
  GPUVecchiaSpaceTimeLogLikelihood: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-interpolation/vecchia-space-time-log-likelihood.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/vecchia-likelihood.md`,
    summary:
      'Space-time Vecchia log-likelihood over ordered conditioning sets with the per-frame covariance of GPUSpaceTimeKriging.',
    contract: 'GPUVecchiaSpaceTimeLogLikelihoodProps'
  },
  GPUConformalPredictionIntervals: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-validation/conformal-prediction-intervals.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/spatial-validation-advanced.md`,
    summary:
      'Split-conformal and CV+ prediction intervals from held-out predictions, with exact order statistics found by a per-query bisection over a once-sorted calibration set.',
    contract: 'GPUConformalPredictionIntervalsProps'
  },
  GPUGeoConformalPredictionIntervals: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['attributes', 'spatial-weights', 'parameters'],
    outputs: ['attributes'],
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-validation/conformal-prediction-intervals.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/spatial-validation-advanced.md`,
    summary:
      'Spatially weighted (GeoConformal) split-conformal intervals: the kernel-weighted quantile of neighbouring calibration scores with the query as a point at infinity.',
    contract: 'GPUGeoConformalPredictionIntervalsProps'
  },
  GPUAreaOfApplicability: {
    primaryCapability: PREDICTION_FAMILY,
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-validation/area-of-applicability.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/spatial-validation-advanced.md`,
    summary:
      'Dissimilarity index and area of applicability (Meyer and Pebesma 2021, CAST): distance in standardized predictor space to the nearest training point, thresholded by the training distribution under cross-validation folds.',
    contract: 'GPUAreaOfApplicabilityProps'
  },
  GPUGeographicallyTemporallyWeightedRegression: {
    primaryCapability: REGRESSION_FAMILY,
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    status: ['converged', 'iterationLimitReached'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-regression/gtwr-bandwidth-search.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/operators-regression.md`,
    summary:
      'Geographically and temporally weighted regression with fixed or adaptive bandwidth and optional nested golden-section bandwidth/time-scale selection: local weighted least squares with kernel weights on the space-time distance, with per-row conditioning status.',
    contract: 'GPUGeographicallyTemporallyWeightedRegressionProps'
  },
  GPUMultiscaleGeographicallyTemporallyWeightedRegression: {
    primaryCapability: REGRESSION_FAMILY,
    inputs: ['points', 'time-series', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    status: ['converged', 'iterationLimitReached'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/spatial-regression/gtwr-bandwidth-search.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/operators-regression.md`,
    summary:
      'Multiscale GTWR by backfitting: fixed or adaptive bandwidth and time scale per covariate, optional partial-residual AICc profile search, a fixed number of Gauss-Seidel iterations, per-iteration convergence scores.',
    contract: 'GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps'
  },
  addTrajectorySortCheck: {
    primaryCapability: MOVEMENT_FAMILY,
    inputs: ['trajectories'],
    outputs: ['summary'],
    coordinateModels: ['none'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    status: ['invalidCount'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/trajectory-analysis/trajectory-sort-check.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/operators-lines.md`,
    summary:
      'Optional GPU check of the sort-by-track-then-time precondition, reporting the count of rows out of order.',
    contract: 'AddTrajectorySortCheckProps'
  },
  GPUTrajectoryStaypoints: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    outputs: ['points', 'attributes', 'summary'],
    cardinality: 'capacity-bounded',
    status: ['count', 'requiredCount', 'overflow', 'candidateOverflow'],
    tests: `${TEST_ROOT}/trajectory-staypoints/gpu-trajectory-staypoints.spec.ts`,
    summary:
      'Staypoints of each track with the trackintel sliding-window rule, found by a bounded forward search per row and pointer doubling.',
    contract: 'GPUTrajectoryStaypointsProps'
  },
  GPUTrajectoryLocations: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes'],
    cardinality: 'source-sized',
    tests: `${TEST_ROOT}/trajectory-staypoints/gpu-trajectory-locations.spec.ts`,
    summary:
      'Staypoint locations per track as connected components of staypoints within a distance (trackintel generate_locations).',
    contract: 'GPUTrajectoryLocationsProps'
  },
  GPUTrajectorySegments: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    outputs: ['attributes', 'summary'],
    cardinality: 'capacity-bounded',
    status: ['count', 'requiredCount', 'overflow'],
    tests: `${TEST_ROOT}/trajectory-staypoints/trackintel-parity.spec.ts`,
    summary:
      'Triplegs and trips of each track from its staypoints, following the trackintel tripleg rule and trip state machine.',
    contract: 'GPUTrajectorySegmentsProps'
  },
  GPUMobilityIndicators: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    cardinality: 'fixed-sized',
    status: ['candidateOverflow'],
    tests: MOBILITY_TESTS,
    summary:
      'Per-track mobility indicators of staypoints: jump lengths, waiting times, location entropies, home and work, and tracking quality.',
    contract: 'GPUMobilityIndicatorsProps'
  },
  GPUBrownianBridgeUtilization: {
    primaryCapability: MOVEMENT_FAMILY,
    inputs: ['trajectories', 'attributes', 'parameters'],
    outputs: ['surface'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: MOB07_TESTS,
    documentation: MOB07_HOME_RANGE_DOCUMENTATION,
    summary:
      'Brownian bridge utilization distribution of a track (Horne et al. 2007) on a grid, with per-segment motion variance for the dynamic form.',
    contract: 'GPUBrownianBridgeUtilizationProps'
  },
  GPUUtilizationVolumeIsopleths: {
    primaryCapability: MOVEMENT_FAMILY,
    inputs: ['surface', 'parameters'],
    outputs: ['surface', 'attributes', 'summary'],
    coordinateModels: ['grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    evidence: 'partial',
    tests: MOB07_TESTS,
    documentation: MOB07_HOME_RANGE_DOCUMENTATION,
    summary:
      'Volume isopleths of a utilization distribution: the cumulative probability volume of every cell and the mask of the home range at a level.',
    contract: 'GPUUtilizationVolumeIsoplethsProps'
  },
  GPUTrajectoryRecursions: {
    primaryCapability: MOVEMENT_FAMILY,
    inputs: ['trajectories', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['planar'],
    dimensionality: '2d+time',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    status: ['candidateOverflow'],
    evidence: 'partial',
    tests: MOB07_TESTS,
    documentation: MOB07_HOME_RANGE_DOCUMENTATION,
    summary:
      'Revisits, residence time and forward-only first passage time of the circles around each fix, with interpolated circle crossings (recurse definitions; first passage time after Fauchald and Tveraa, forward leg only).',
    contract: 'GPUTrajectoryRecursionsProps'
  },
  GPUModalSplit: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    inputs: ['attributes', 'parameters'],
    outputs: ['attributes'],
    cardinality: 'fixed-sized',
    tests: MOBILITY_TESTS,
    summary:
      'Modal split of triplegs per track by count, length or duration, optionally predicting the mode from speed classes (trackintel calculate_modal_split).',
    contract: 'GPUModalSplitProps'
  },
  GPUTrajectoryOutlierFlags: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    outputs: ['attributes', 'summary'],
    cardinality: 'source-sized',
    tests: CLEANING_TESTS,
    summary:
      'Point-centred speed and acceleration outlier flags: a fix is flagged only when both adjacent steps exceed the limit.',
    contract: 'GPUTrajectoryOutlierFlagsProps'
  },
  GPUTrajectoryHampel: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    inputs: ['trajectories', 'attributes', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['none'],
    cardinality: 'source-sized',
    tests: CLEANING_TESTS,
    summary:
      'Hampel filter per track: replaces values further than a multiple of the scaled median absolute deviation from the window median.',
    contract: 'GPUTrajectoryHampelProps'
  },
  GPUTrajectorySplit: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    outputs: ['trajectories', 'attributes'],
    cardinality: 'capacity-bounded',
    tests: CLEANING_TESTS,
    summary:
      'Splits tracks at observation gaps, at staypoints or where a value changes (MovingPandas splitters), as a new trajectory port.',
    contract: 'GPUTrajectorySplitProps'
  },
  GPUTrajectoryThinning: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    outputs: ['points', 'attributes', 'summary'],
    cardinality: 'capacity-bounded',
    status: ['count', 'requiredCount', 'overflow', 'candidateOverflow'],
    tests: CLEANING_TESTS,
    summary:
      'Minimum-distance and minimum-time generalization of tracks with MovingPandas sequential semantics, in a bounded number of passes.',
    contract: 'GPUTrajectoryThinningProps'
  },
  GPUTrajectoryTrackRelativeTimes: {
    ...TRAJECTORY_ROW_DESCRIPTOR,
    inputs: ['trajectories'],
    outputs: ['time-series'],
    coordinateModels: ['none'],
    cardinality: 'source-sized',
    semantics: 'exact',
    tests: CLEANING_TESTS,
    summary:
      'Int64 word timestamps to float32 ticks after the first row of each track, differenced exactly on the words.',
    contract: 'GPUTrajectoryTrackRelativeTimesProps'
  },
  GPUGeographicalDetector: {
    ...STX08_DESCRIPTOR,
    inputs: ['attributes', 'parameters'],
    outputs: ['summary'],
    coordinateModels: ['none'],
    cardinality: 'fixed-sized',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-geographical-detector.spec.ts`,
    summary:
      'Geographical detector factor statistic: q of a stratification, its non-central F statistic and non-centrality (geodetector 0.2.0).',
    contract: 'GPUGeographicalDetectorProps'
  },
  GPURedcapSingleLinkage: {
    ...STX08_DESCRIPTOR,
    primaryCapability: 'group-and-regional-analysis',
    inputs: ['spatial-weights', 'attributes', 'parameters'],
    outputs: ['regions', 'summary'],
    coordinateModels: ['topological'],
    cardinality: 'source-sized',
    status: ['approximation'],
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-redcap-single-linkage.spec.ts`,
    summary:
      'Contiguity-constrained single-linkage clustering cut at a per-frame region count, from the minimum spanning tree: the tree of first-order single-linkage REDCAP with the heaviest edges removed, not its SSD partition (that is GPUSkaterRegions).',
    contract: 'GPURedcapSingleLinkageProps'
  },
  GPUCategoryPairCounts: {
    ...STX08_DESCRIPTOR,
    inputs: ['spatial-weights', 'attributes'],
    outputs: ['summary'],
    coordinateModels: ['topological'],
    cardinality: 'fixed-sized',
    semantics: 'exact',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-spatial-entropy.spec.ts`,
    summary:
      'Ordered category-pair link counts and rows per category of a categorical map over a weights graph (exact integer atomics).',
    contract: 'GPUCategoryPairCountsProps'
  },
  GPUSpatialEntropy: {
    ...STX08_DESCRIPTOR,
    inputs: ['spatial-weights', 'attributes'],
    outputs: ['summary'],
    coordinateModels: ['topological'],
    cardinality: 'fixed-sized',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-spatial-entropy.spec.ts`,
    summary:
      'Shannon, co-occurrence (Leibovici) entropy and spatial mutual information (Altieri) of a categorical map over a weights graph.',
    contract: 'GPUSpatialEntropyProps'
  },
  GPUColocationQuotient: {
    ...STX08_DESCRIPTOR,
    inputs: ['spatial-weights', 'attributes'],
    outputs: ['summary', 'attributes'],
    coordinateModels: ['topological'],
    cardinality: 'fixed-sized',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-colocation-quotient.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/colocation-category-dependence-adaptive-density.md`,
    summary:
      'Global and local colocation quotient of a categorical map over a weights graph (Leslie and Kronenfeld 2011), with deterministic fixed-point sums of the neighbour shares.',
    contract: 'GPUColocationQuotientProps'
  },
  GPULocalCategoryDependence: {
    ...STX08_DESCRIPTOR,
    inputs: ['spatial-weights', 'attributes'],
    outputs: ['attributes'],
    coordinateModels: ['topological'],
    cardinality: 'source-sized',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-local-category-dependence.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/colocation-category-dependence-adaptive-density.md`,
    summary:
      'Local composition measure for categorical data after Boots (2003), focal row excluded: own-category neighbour count, local share, binomial upper tail against the global share and z score per row.',
    contract: 'GPULocalCategoryDependenceProps'
  },
  GPUAdaptiveBandwidthKernelDensity: {
    primaryCapability: 'density-and-interpolation',
    inputs: ['points', 'attributes', 'parameters'],
    outputs: ['surface', 'attributes', 'summary'],
    coordinateModels: ['planar', 'grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-adaptive-bandwidth-kernel-density.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/colocation-category-dependence-adaptive-density.md`,
    summary:
      'Abramson variable-bandwidth kernel density on a raster: a pilot density at the points sets each point bandwidth, clamped to a factor limit; reuses the GPUKernelDensity kernels and grid index.',
    contract: 'GPUAdaptiveBandwidthKernelDensityProps'
  },
  GPURasterFalseDiscoveryRateMask: {
    primaryCapability: SPACE_TIME_FAMILY,
    inputs: ['surface', 'parameters'],
    outputs: ['surface', 'summary'],
    coordinateModels: ['grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'approximate',
    evidence: 'partial',
    tests: `${TEST_ROOT}/multiple-testing/rts01-raster-false-discovery-rate-mask.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/operators-statistics.md`,
    summary:
      'Field significance of a p-value raster: Bonferroni, Holm, Benjamini-Hochberg or Benjamini-Yekutieli adjustment, a significance mask, the critical p-value and a p-value histogram.',
    contract: 'GPURasterFalseDiscoveryRateMaskProps'
  },
  addContiguityOrderCorrelogramRecipe: {
    ...STX08_DESCRIPTOR,
    inputs: ['spatial-weights', 'attributes', 'parameters'],
    outputs: ['summary'],
    coordinateModels: ['topological'],
    cardinality: 'fixed-sized',
    status: ['overflow'],
    tests: `${TEST_ROOT}/stx08-spatial-methods/stx08-contiguity-order-correlogram.spec.ts`,
    summary:
      'Moran correlogram over exactly-k-step contiguity orders: higher-order rings, row standardisation and global Moran per order.',
    contract: 'GPUContiguityOrderCorrelogramRecipeProps'
  },
  getHealpixCellBoundary: {
    ...RTS08_DESCRIPTOR,
    inputs: ['cells', 'parameters'],
    outputs: ['polygons'],
    coordinateModels: ['spherical'],
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    tests: `${TEST_ROOT}/healpix-cells/healpix-cell-boundary.node.spec.ts`,
    summary:
      'Samples curved spherical HEALPix nested cell edges as a closed longitude/latitude display ring on a continuous longitude branch.',
    contract: 'getHealpixCellBoundary(cell, order, edgeSegments)'
  },
  GPUHealpixPointToCell: {
    ...RTS08_DESCRIPTOR,
    inputs: ['points', 'parameters'],
    outputs: ['cells'],
    coordinateModels: ['spherical'],
    cardinality: 'source-sized',
    tests: `${TEST_ROOT}/rts08-healpix-a5/rts08-healpix-point-to-cell.spec.ts`,
    summary:
      'Longitude/latitude points to HEALPix nested (or UNIQ) 64-bit cell keys on the sphere, orders 0 to 29, checked against healpix-geo.',
    contract: 'GPUHealpixPointToCellProps'
  },
  GPUHealpixCellAggregation: {
    ...RTS08_DESCRIPTOR,
    inputs: ['cells', 'attributes', 'parameters'],
    outputs: ['cells', 'attributes', 'summary'],
    coordinateModels: ['spherical'],
    cardinality: 'capacity-bounded',
    status: ['count', 'requiredCount', 'overflow', 'invalidCount'],
    semantics: 'exact',
    tests: `${TEST_ROOT}/rts08-healpix-a5/rts08-healpix-cell-aggregation.spec.ts`,
    summary:
      'Distinct HEALPix cells in key order with row counts and per-cell sums, minima and maxima of a value column.',
    contract: 'GPUHealpixCellAggregationProps'
  },
  GPUPopulationSupportAlignment: {
    primaryCapability: 'coverage-landscape-ecology',
    inputs: ['attributes'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['topological'],
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'exact',
    deterministic: true,
    evidence: 'partial',
    tests: `${TEST_ROOT}/urban-ecology/population-support-alignment-operation.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/landscape-ecology.md`,
    summary:
      'Exact population-support alignment through indexed key joins, preserving target order and reporting missing, ambiguous and excluded rows; CPU map operation available for mixed programs.',
    contract: 'GPUPopulationSupportAlignmentProps'
  },
  addRasterResistanceConnectivityRecipe: {
    primaryCapability: 'coverage-landscape-ecology',
    inputs: ['surface', 'parameters'],
    outputs: ['attributes', 'summary'],
    coordinateModels: ['grid'],
    dimensionality: '2d',
    topology: 'fixed',
    cardinality: 'source-sized',
    semantics: 'model-based',
    deterministic: true,
    status: ['converged', 'iterationLimitReached', 'invalidCount'],
    evidence: 'partial',
    tests: `${TEST_ROOT}/urban-ecology/raster-resistance-connectivity.spec.ts`,
    documentation: `${DOCUMENTATION_ROOT}/landscape-ecology.md`,
    summary:
      'Circuit potentials and current map across a resistance raster: 4- or 8-neighbour conductances 2 / (R(a) + R(b)) / edgeLength built on the GPU, then the sparse Jacobi-PCG solve between one source and one sink cell.',
    contract: 'GPURasterResistanceConnectivityProps'
  }
};

/**
 * Operators of `gpu-spatial-analysis` added after the original 261. Each names an original
 * capability family as its primary one, but that family is not edited, so
 * `GPU_SPATIAL_ANALYSIS_OPERATORS` and `GPU_SPATIAL_ANALYSIS_CAPABILITIES` stay as consumers saw
 * them. `getGPUAnalysisOperatorCatalog()` appends these after the other entry points.
 */
export const SPATIOTEMPORAL_REVIEW_OPERATORS: readonly GPUAnalysisOperator[] = Object.entries(
  SPATIOTEMPORAL_REVIEW_OPERATOR_OVERRIDES
).map(([exportName, override]) => {
  const annotation = OPERATOR_ANNOTATIONS[exportName] ?? {};
  return {
    exportName,
    primaryCapability: override.primaryCapability,
    tags: [],
    package: PACKAGE,
    inputs: override.inputs,
    outputs: override.outputs,
    dimensionality: override.dimensionality ?? 'unspecified',
    coordinateModels: override.coordinateModels,
    topology: override.topology ?? 'unspecified',
    cardinality: override.cardinality,
    semantics: override.semantics ?? 'unspecified',
    deterministic: override.deterministic ?? 'unspecified',
    status: override.status ?? [],
    readback: override.readback ?? 'none',
    summary: override.summary,
    role: annotation.role ?? 'operator',
    tier: annotation.tier ?? 'scale',
    ...(annotation.limits ? {limits: annotation.limits} : {}),
    ...(annotation.limitation ? {limitation: annotation.limitation} : {}),
    evidence: {
      status: override.evidence,
      tests: override.tests,
      documentation: override.documentation,
      contract: override.contract
    }
  };
});
