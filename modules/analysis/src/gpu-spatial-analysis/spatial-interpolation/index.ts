// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {decodeGPUVecchiaLogLikelihoodResult} from './gpu-vecchia-log-likelihood';
import type {decodeGPUVecchiaFisherScoringResult} from './gpu-vecchia-fisher-scoring';

export {GPULocalKriging, decodeGPULocalKrigingStatus} from './gpu-local-kriging';
export type {GPULocalKrigingProps, GPUPredictionCovariance} from './gpu-local-kriging';
export {
  getGPULocalKrigingParameterLength,
  getGPULocalKrigingParameterValues,
  getGPUSpaceTimeKrigingParameterValues,
  GPU_LOCAL_KRIGING_PARAMETER_LENGTH,
  GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH
} from './prediction-covariance';
export type {
  GPULocalKrigingSettings,
  GPUSpaceTimeKrigingSettings,
  GPUPredictionCovarianceModel,
  GPUPredictionCovarianceSmoothness
} from './prediction-covariance';
export {GPUSpaceTimeKriging} from './gpu-space-time-kriging';
export type {
  GPUSpaceTimeKrigingNeighborSearch,
  GPUSpaceTimeKrigingProps
} from './gpu-space-time-kriging';
export {
  evaluateSpaceTimeVariogramModel,
  fitSpaceTimeVariogramModel,
  getGPUSpaceTimeVariogramBinLags
} from '../../dataframe-analysis/pair-statistics/space-time-variogram-model';
export type {
  SpaceTimeVariogramFit,
  SpaceTimeVariogramFitOptions,
  SpaceTimeVariogramInput,
  SpaceTimeVariogramModel,
  SpaceTimeVariogramModelType,
  SpaceTimeVariogramStructure
} from '../../dataframe-analysis/pair-statistics/space-time-variogram-model';
export {
  GPUVecchiaLogLikelihood,
  getVecchiaLogLikelihoodResult,
  decodeGPUVecchiaLogLikelihoodStatus,
  decodeGPUVecchiaLogLikelihoodResult
} from './gpu-vecchia-log-likelihood';
export type {
  GPUVecchiaLogLikelihoodOutput,
  GPUVecchiaLogLikelihoodProps
} from './gpu-vecchia-log-likelihood';
export {
  GPUVecchiaFisherScoring,
  GPU_VECCHIA_FISHER_CELL_COUNT,
  getVecchiaFisherEvaluation,
  decodeGPUVecchiaFisherScoringStatus,
  decodeGPUVecchiaFisherScoringResult
} from './gpu-vecchia-fisher-scoring';
export type {
  GPUVecchiaFisherScoringOutput,
  GPUVecchiaFisherScoringProps
} from './gpu-vecchia-fisher-scoring';
export {
  GPUVecchiaSpaceTimeLogLikelihood,
  GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,
  decodeGPUVecchiaSpaceTimeLogLikelihoodResult,
  decodeGPUVecchiaSpaceTimeLogLikelihoodStatus
} from './gpu-vecchia-space-time-log-likelihood';
export type {
  GPUVecchiaSpaceTimeLogLikelihoodProps,
  GPUVecchiaSpaceTimeLogLikelihoodOutput
} from './gpu-vecchia-space-time-log-likelihood';
export {
  buildVecchiaConditioningSets,
  buildVecchiaPredictionNeighborhoods,
  getVecchiaMaxMinOrdering,
  uploadVecchiaNeighborhoods,
  MAXIMUM_VECCHIA_NEIGHBOR_COUNT,
  MAXIMUM_VECCHIA_ENTRY_COUNT
} from './vecchia-conditioning-sets';
export type {
  VecchiaConditioningSetProps,
  VecchiaConditioningSets,
  PreparedVecchiaConditioningSets
} from './vecchia-conditioning-sets';
export {
  fitVecchiaCovarianceFisherScoring,
  maximizeVecchiaLogLikelihood
} from './vecchia-fisher-fit';
export type {
  VecchiaFisherEvaluation,
  VecchiaFisherFit,
  VecchiaFisherFitOptions,
  VecchiaNelderMeadOptions,
  VecchiaNelderMeadResult
} from './vecchia-fisher-fit';
export {
  getGPUVecchiaLogLikelihoodParameterValues,
  getVecchiaNuggetFloor,
  GPU_VECCHIA_PARAMETER_LENGTH,
  VECCHIA_NUGGET_FLOOR_FRACTION
} from './vecchia-parameters';
export type {GPUVecchiaCovariance, GPUVecchiaCovarianceModel} from './vecchia-parameters';
export type {GPUVecchiaLikelihoodOutput} from './vecchia-likelihood-nodes';
export {GPULocalRBF, decodeGPULocalRBFStatus} from './gpu-local-rbf';
export type {GPULocalRBFProps} from './gpu-local-rbf';
export {GPUNaturalNeighbor} from './gpu-natural-neighbor';
export type {GPUNaturalNeighborProps} from './gpu-natural-neighbor';
export {GPUConditionalGaussianSimulation} from './gpu-conditional-gaussian-simulation';
export type {GPUConditionalGaussianSimulationProps} from './gpu-conditional-gaussian-simulation';

export {GPUInverseDistanceWeighting} from './gpu-inverse-distance-weighting';
export type {
  GPUInverseDistanceWeightingOutput,
  GPUInverseDistanceWeightingProps
} from './gpu-inverse-distance-weighting';
export {
  GPUKriging,
  KRIGING_NEGATIVE_VARIANCE_TOLERANCE,
  KRIGING_PIVOT_RATIO_TOLERANCE,
  KRIGING_RELATIVE_PIVOT_TOLERANCE
} from './gpu-kriging';
export type {GPUKrigingOutput, GPUKrigingProps} from './gpu-kriging';
export {GPUFocalStatistics} from './gpu-focal-statistics';
export type {
  GPUFocalStatisticsOutput,
  GPUFocalStatisticsProps
} from './gpu-focal-statistics';
export {
  getGPUKrigingParameterValues,
  GPU_KRIGING_PARAMETER_LENGTH
} from './kriging-parameters';
export type {GPUKrigingSettings} from './kriging-parameters';
export {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH
} from './spatial-interpolation-parameters';
export type {
  GPUFocalStatisticsSettings,
  GPUFocalStatisticsShape,
  GPUInverseDistanceWeightingSettings,
  GPUInverseDistanceWeightingTaper
} from './spatial-interpolation-parameters';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeConditionalGaussianSimulationOnCPU,
  CPUConditionalGaussianSimulationOperation
} from './cpu-conditional-gaussian-simulation-operation';
export type {
  CPUConditionalGaussianSimulationInput,
  CPUConditionalGaussianSimulationOptions,
  CPUConditionalGaussianSimulationResult
} from './cpu-conditional-gaussian-simulation-operation';
export {
  computeVecchiaFisherScoringOnCPU,
  CPUVecchiaFisherScoringOperation
} from './cpu-vecchia-fisher-scoring-operation';
export type {
  CPUVecchiaFisherScoringInput,
  CPUVecchiaFisherScoringResult,
  PreparedCPUVecchiaFisherScoringResult
} from './cpu-vecchia-fisher-scoring-operation';
export {
  computeVecchiaLogLikelihoodOnCPU,
  CPUVecchiaLogLikelihoodOperation
} from './cpu-vecchia-log-likelihood-operation';
export type {
  CPUVecchiaLogLikelihoodInput,
  CPUVecchiaLogLikelihoodResult,
  PreparedCPUVecchiaLogLikelihoodResult
} from './cpu-vecchia-log-likelihood-operation';
export {
  computeVecchiaSpaceTimeLogLikelihoodOnCPU,
  CPUVecchiaSpaceTimeLogLikelihoodOperation
} from './cpu-vecchia-space-time-log-likelihood-operation';
export type {
  CPUVecchiaSpaceTimeLogLikelihoodInput,
  CPUVecchiaSpaceTimeLogLikelihoodResult,
  PreparedCPUVecchiaSpaceTimeLogLikelihoodResult
} from './cpu-vecchia-space-time-log-likelihood-operation';
export {GPUConditionalGaussianSimulationOperation} from './gpu-conditional-gaussian-simulation-operation';
export type {ConditionalGaussianSimulationOperationProps} from './gpu-conditional-gaussian-simulation-operation';
export {GPUVecchiaFisherScoringOperation} from './gpu-vecchia-fisher-scoring-operation';
export type {VecchiaFisherScoringOperationProps} from './gpu-vecchia-fisher-scoring-operation';
export {GPUVecchiaLogLikelihoodOperation} from './gpu-vecchia-log-likelihood-operation';
export type {VecchiaLogLikelihoodOperationProps} from './gpu-vecchia-log-likelihood-operation';
export {GPUVecchiaSpaceTimeLogLikelihoodOperation} from './gpu-vecchia-space-time-log-likelihood-operation';
export type {VecchiaSpaceTimeLogLikelihoodOperationProps} from './gpu-vecchia-space-time-log-likelihood-operation';

export {
  GPUGaussianRandomField,
  GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_FFT_SIDE,
  GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_ROW_COUNT,
  GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_MODE_COUNT,
  GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_SPECTRAL_WORK,
  GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_NEIGHBOR_COUNT
} from './gpu-gaussian-random-field';
export type {
  GPUGaussianRandomFieldProps,
  GPUGaussianRandomFieldOutput
} from './gpu-gaussian-random-field';
export {
  getGPUGaussianRandomFieldParameterValues,
  GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH
} from './gaussian-random-field-parameters';
export type {GPUGaussianRandomFieldSettings} from './gaussian-random-field-parameters';

export {
  GPU_SPACE_TIME_KRIGING_MAXIMUM_SAMPLE_COUNT,
  GPU_SPACE_TIME_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT,
  GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT
} from './gpu-space-time-kriging';

export {
  GPU_LOCAL_KRIGING_MAXIMUM_NEIGHBOR_COUNT,
  GPU_LOCAL_KRIGING_MAXIMUM_SUPPORT_COUNT,
  GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT,
  GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK
} from './gpu-local-kriging';
export {GPU_LOCAL_RBF_MAXIMUM_NEIGHBOR_COUNT} from './gpu-local-rbf';

export {GPU_CONDITIONAL_GAUSSIAN_SIMULATION_MAXIMUM_JOINT_ROW_COUNT} from './gpu-conditional-gaussian-simulation';

/** Typed ports accepted by the checked Vecchia likelihood readback decoder. */
export type GPUVecchiaLogLikelihoodReadback = Parameters<
  typeof decodeGPUVecchiaLogLikelihoodResult
>[0];
/** Typed ports accepted by the checked Vecchia Fisher-scoring readback decoder. */
export type GPUVecchiaFisherScoringReadback = Parameters<
  typeof decodeGPUVecchiaFisherScoringResult
>[0];

export type {VecchiaNeighborhoodStatus} from './vecchia-neighborhood-status';
