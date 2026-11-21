// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUVariogramModelFit} from './gpu-variogram-model-fit';
export type {GPUVariogramModelFitProps} from './gpu-variogram-model-fit';

export {getGPUPairStatisticsMaximumRowCount} from './pair-statistics-grid';
export {GPUVariogram} from './gpu-variogram';
export type {GPUVariogramOutput, GPUVariogramProps} from './gpu-variogram';
export {
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH
} from './variogram-parameters';
export type {GPUVariogramSettings} from './variogram-parameters';
export {
  evaluateVariogramModel,
  fitVariogramModel,
  fitVariogramModelWithDiagnostics,
  getVariogramModelShape
} from './variogram-model';
export type {
  VariogramModel,
  VariogramModelFitDiagnostics,
  VariogramModelInput,
  VariogramModelOptions,
  VariogramModelType,
  VariogramModelWeighting
} from './variogram-model';
export {GPUSpatialCorrelogram} from './gpu-spatial-correlogram';
export type {
  GPUSpatialCorrelogramBandMode,
  GPUSpatialCorrelogramOutput,
  GPUSpatialCorrelogramProps
} from './gpu-spatial-correlogram';
export {
  getGPUSpatialCorrelogramParameterValues,
  GPU_SPATIAL_CORRELOGRAM_NO_BAND,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
} from './spatial-correlogram-parameters';
export type {
  GPUSpatialCorrelogramSettings,
  GPUSpatialCorrelogramVarianceAssumption
} from './spatial-correlogram-parameters';
export {GPURipley, GPU_RIPLEY_WEIGHT_CAP} from './gpu-ripley';
export type {GPURipleyOutput, GPURipleyProps} from './gpu-ripley';
export {
  GPURipleyEnvelope,
  GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT
} from './gpu-ripley-envelope';
export type {
  GPURipleyEnvelopeOutput,
  GPURipleyEnvelopeProps,
  GPURipleyEnvelopeStatistic
} from './gpu-ripley-envelope';
export {
  getGPURipleyParameterValues,
  GPU_RIPLEY_EDGE_CORRECTION,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_RIPLEY_SEED_SLOT
} from './ripley-parameters';
export type {GPURipleyEdgeCorrection, GPURipleySettings} from './ripley-parameters';
export {GPUPointPatternIndices, GPU_NO_NEAREST_NEIGHBOR} from './gpu-point-pattern-indices';
export type {
  GPUPointPatternIndicesOutput,
  GPUPointPatternIndicesProps
} from './gpu-point-pattern-indices';
export {
  getGPUPointPatternIndicesParameterValues,
  GPU_CLARK_EVANS_CORRECTION,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_ALTERNATIVE,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_MAXIMUM_COUNT,
  GPU_QUADRAT_STATISTICS_LENGTH
} from './point-pattern-indices-parameters';
export type {
  GPUClarkEvansCorrection,
  GPUPointPatternIndicesHintSettings,
  GPUPointPatternIndicesSettings
} from './point-pattern-indices-parameters';
export {GPURipleyDistanceFunctions} from './gpu-ripley-distance-functions';
export type {
  GPURipleyDistanceFunctionsOutput,
  GPURipleyDistanceFunctionsProps
} from './gpu-ripley-distance-functions';
export {
  getGPURipleyDistanceFunctionsParameterValues,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
} from './ripley-distance-parameters';
export type {
  GPURipleyDistanceEdgeCorrection,
  GPURipleyDistanceFunctionsSettings
} from './ripley-distance-parameters';

export type {GPUPairStatisticsBounds} from './pair-statistics-parameters';
export {GPUCrossVariogram} from './gpu-cross-variogram';
export type {GPUCrossVariogramProps} from './gpu-cross-variogram';
export {
  GPUSpaceTimeVariogram,
  GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH,
  getGPUSpaceTimeVariogramParameterValues
} from './gpu-space-time-variogram';
export type {
  GPUSpaceTimeVariogramProps,
  GPUSpaceTimeVariogramSettings
} from './gpu-space-time-variogram';
export {
  getGPUSpaceTimeVariogramBinLags,
  fitSpaceTimeVariogramModel,
  evaluateSpaceTimeVariogramModel
} from './space-time-variogram-model';
export type {
  SpaceTimeVariogramFit,
  SpaceTimeVariogramFitOptions,
  SpaceTimeVariogramInput,
  SpaceTimeVariogramModel,
  SpaceTimeVariogramModelType,
  SpaceTimeVariogramStructure
} from './space-time-variogram-model';
export {
  GPUMarkedPointStatistics,
  GPU_MARKED_POINT_STATISTICS_COLUMNS
} from './gpu-marked-point-statistics';
export type {GPUMarkedPointStatisticsProps} from './gpu-marked-point-statistics';
export {GPUSpaceTimeKFunction} from './gpu-space-time-k-function';
export type {
  GPUSpaceTimeKFunctionOutput,
  GPUSpaceTimeKFunctionProps
} from './gpu-space-time-k-function';
export {
  getGPUSpaceTimeKFunctionParameterValues,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
  GPU_SPACE_TIME_K_MAXIMUM_CLASS_COUNT,
  GPU_SPACE_TIME_K_MAXIMUM_LOCAL_BINS
} from './space-time-k-function-nodes';
export type {
  GPUSpaceTimeKEdgeCorrection,
  GPUSpaceTimeKFunctionSettings,
  GPUSpaceTimePairCorrelationKernel
} from './space-time-k-function-nodes';
export {GPU_SPACE_TIME_K_WEIGHT_CAP} from './space-time-k-function-wgsl';
export {
  getGPUSpaceTimeDiggleRandomParameterValues,
  GPUSpaceTimeDiggleTest,
  GPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS,
  GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH
} from './gpu-space-time-diggle-test';
export type {
  GPUSpaceTimeDiggleTestOutput,
  GPUSpaceTimeDiggleTestProps
} from './gpu-space-time-diggle-test';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeVariogramModelFitOnCPU,
  CPUVariogramModelFitOperation
} from './cpu-variogram-model-fit-operation';
export type {
  CPUVariogramModelFitOptions,
  CPUVariogramModelFitResult
} from './cpu-variogram-model-fit-operation';
export {
  VARIOGRAM_MODEL_FIT_DEFAULT_MODELS,
  GPUVariogramModelFitOperation
} from './gpu-variogram-model-fit-operation';
export type {VariogramModelFitOperationProps} from './gpu-variogram-model-fit-operation';
export {
  computeRipleyEnvelopeOnCPU,
  CPURipleyEnvelopeOperation
} from './cpu-ripley-envelope-operation';
export type {
  CPURipleyEnvelopeInput,
  CPURipleyEnvelopeOptions,
  CPURipleyEnvelopeResult
} from './cpu-ripley-envelope-operation';
export {
  CPU_SPACE_TIME_DIGGLE_MAXIMUM_PERMUTATIONS,
  CPU_SPACE_TIME_DIGGLE_MAXIMUM_BIN_COUNT,
  computeSpaceTimeDiggleTestOnCPU,
  CPUSpaceTimeDiggleTestOperation
} from './cpu-space-time-diggle-test-operation';
export type {
  CPUSpaceTimeDiggleTestInput,
  CPUSpaceTimeDiggleTestOptions,
  CPUSpaceTimeDiggleTestResult
} from './cpu-space-time-diggle-test-operation';
export {GPURipleyEnvelopeOperation} from './gpu-ripley-envelope-operation';
export type {RipleyEnvelopeOperationProps} from './gpu-ripley-envelope-operation';
export {GPUSpaceTimeDiggleTestOperation} from './gpu-space-time-diggle-test-operation';
export type {SpaceTimeDiggleTestOperationProps} from './gpu-space-time-diggle-test-operation';

export {GPUCoregionalizationFit} from './gpu-coregionalization-fit';
export type {
  GPUCoregionalizationFitProps,
  GPUCoregionalizationFitOutput,
  GPUCoregionalizationFitStatus
} from './gpu-coregionalization-fit';
export {GPUCoregionalizationFitOperation} from './gpu-coregionalization-fit-operation';
export type {CoregionalizationFitOperationProps} from './gpu-coregionalization-fit-operation';
export {
  CPUCoregionalizationFitOperation,
  computeCoregionalizationFitOnCPU,
  fitLinearModelOfCoregionalization
} from './cpu-coregionalization-fit-operation';
export type {
  CPUCoregionalizationFitInput,
  CPUCoregionalizationFitResult
} from './cpu-coregionalization-fit-operation';
export {
  getGPUCoregionalizationFitParameterValues,
  COREGIONALIZATION_DEFAULT_TOLERANCE,
  GPU_COREGIONALIZATION_MAXIMUM_VARIABLE_COUNT,
  GPU_COREGIONALIZATION_MAXIMUM_STRUCTURE_COUNT,
  GPU_COREGIONALIZATION_MAXIMUM_BIN_COUNT,
  GPU_COREGIONALIZATION_MAXIMUM_ITERATIONS
} from './coregionalization-parameters';
export type {
  CoregionalizationFitOptions,
  CoregionalizationStructure
} from './coregionalization-parameters';
