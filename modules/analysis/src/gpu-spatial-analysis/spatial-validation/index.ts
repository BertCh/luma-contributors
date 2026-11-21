// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPU_NESTED_VALIDATION_MAXIMUM_CANDIDATE_COUNT} from './nested-validation-limits';

export {GPUSpatialCrossValidation} from './gpu-spatial-cross-validation';
export type {GPUSpatialCrossValidationProps} from './gpu-spatial-cross-validation';
export {GPUSpatialCrossValidatedOLS} from './gpu-spatial-cross-validated-ols';
export type {GPUSpatialCrossValidatedOLSProps} from './gpu-spatial-cross-validated-ols';
export type {GPUSpatialValidationDesign} from './spatial-validation-design';
export {GPUNestedInterpolationValidation} from './gpu-nested-interpolation-validation';
export type {
  GPUNestedInterpolationValidationProps,
  GPUNestedInterpolationCandidate
} from './gpu-nested-interpolation-validation';
export {
  GPUConformalPredictionIntervals,
  GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED,
  GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED,
  GPU_CONFORMAL_STATUS_INVALID_QUERY,
  GPU_CONFORMAL_STATUS_NO_CALIBRATION,
  GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED,
  GPU_CONFORMAL_STATUS_INPUT_INCOMPLETE,
  decodeConformalStatus
} from './gpu-conformal-prediction-intervals';
export {
  GPUGeoConformalPredictionIntervals,
  GPU_GEO_CONFORMAL_MAXIMUM_NEIGHBORS
} from './gpu-geo-conformal-prediction-intervals';
export type {
  GPUConformalPredictionIntervalsProps,
  GPUConformalPredictionIntervalsOutput
} from './gpu-conformal-prediction-intervals';
export type {
  GPUGeoConformalPredictionIntervalsProps,
  GPUGeoConformalPredictionIntervalsOutput,
  GPUGeoConformalNeighbors
} from './gpu-geo-conformal-prediction-intervals';
export {
  getGPUConformalPredictionParameterValues,
  GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH
} from './conformal-prediction-nodes';
export {
  getConformalPredictionRanks,
  GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT
} from './conformal-prediction-ranks';
export type {
  GPUConformalCalibration,
  GPUConformalQuery,
  GPUConformalPredictionSettings
} from './conformal-prediction-nodes';
export {
  GPUAreaOfApplicability,
  GPU_AREA_OF_APPLICABILITY_MAXIMUM_PREDICTOR_COUNT,
  GPU_AREA_OF_APPLICABILITY_MAXIMUM_TRAINING_ROWS,
  GPU_AREA_OF_APPLICABILITY_STATUS_CONSTANT_PREDICTOR,
  GPU_AREA_OF_APPLICABILITY_STATUS_NO_THRESHOLD,
  GPU_AREA_OF_APPLICABILITY_STATUS_TOO_FEW_ROWS,
  GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH
} from './gpu-area-of-applicability';
export type {
  GPUAreaOfApplicabilityFolds,
  GPUAreaOfApplicabilityOutput,
  GPUAreaOfApplicabilityProps
} from './gpu-area-of-applicability';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  CPU_SPATIAL_CROSS_VALIDATION_MAXIMUM_CANDIDATE_COUNT,
  computeSpatialCrossValidationOnCPU,
  CPUSpatialCrossValidationOperation
} from './cpu-spatial-cross-validation-operation';
export type {
  CPUSpatialCrossValidationInput,
  CPUSpatialCrossValidationOptions,
  CPUSpatialCrossValidationResult
} from './cpu-spatial-cross-validation-operation';
export {GPUSpatialCrossValidationOperation} from './gpu-spatial-cross-validation-operation';
export type {SpatialCrossValidationOperationProps} from './gpu-spatial-cross-validation-operation';
export {
  CPU_NESTED_INTERPOLATION_VALIDATION_MAXIMUM_ROW_COUNT,
  computeNestedInterpolationValidationOnCPU,
  CPUNestedInterpolationValidationOperation
} from './cpu-nested-interpolation-validation-operation';
export type {
  CPUNestedInterpolationValidationInput,
  CPUNestedInterpolationValidationOptions,
  CPUNestedInterpolationValidationResult
} from './cpu-nested-interpolation-validation-operation';
export {GPUNestedInterpolationValidationOperation} from './gpu-nested-interpolation-validation-operation';
export type {NestedInterpolationValidationOperationProps} from './gpu-nested-interpolation-validation-operation';
export {
  GPU_NESTED_VALIDATION_MAXIMUM_ROW_COUNT,
  GPU_NESTED_VALIDATION_MAXIMUM_NEIGHBOR_COUNT
} from './gpu-nested-interpolation-validation';

export {GPU_NESTED_VALIDATION_REFERENCE_MAXIMUM_ROW_COUNT} from './gpu-nested-interpolation-validation';
