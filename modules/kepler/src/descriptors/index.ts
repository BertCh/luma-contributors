// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationInputDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationUniformStyleException,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';
export {
  MAXIMUM_OPERATION_ACTION_LENGTH,
  validateKeplerOperationDescriptor
} from './validate-operation-descriptor';
export {AGGREGATION_OPERATION_DESCRIPTORS} from './descriptors-aggregation';
export {CLUSTERING_OPERATION_DESCRIPTORS} from './descriptors-clustering';
export {CURATED_OPERATION_DESCRIPTORS} from './descriptors-curated';
export * from './descriptors-curated-dasymetric-redistribution';
export {GEOMETRY_A_OPERATION_DESCRIPTORS} from './descriptors-geometry-a';
export {GEOMETRY_B_OPERATION_DESCRIPTORS} from './descriptors-geometry-b';
export {INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS} from './descriptors-interpolation-density';
export {NETWORK_OPERATION_DESCRIPTORS} from './descriptors-network';
export {POINT_CLOUD_OPERATION_DESCRIPTORS} from './descriptors-pointcloud';
export {NETWORK_CATEGORICAL_OPERATION_DESCRIPTORS} from './descriptors-network-categorical';
export {RASTER_OPERATION_DESCRIPTORS} from './descriptors-raster';
export {RASTER_EMBEDDING_OPERATION_DESCRIPTORS} from './descriptors-raster-embedding';
export {STATISTICS_OPERATION_DESCRIPTORS} from './descriptors-statistics';
export {VALIDATION_OPERATION_DESCRIPTORS} from './descriptors-validation';
export {TRAJECTORY_OPERATION_DESCRIPTORS} from './descriptors-trajectory';
export {SPACETIME_OPERATION_DESCRIPTORS} from './descriptors-spacetime';
export {
  KEPLER_OPERATION_DESCRIPTORS,
  deriveKeplerOperationDescriptor,
  getKeplerOperationDescriptor,
  getKeplerOperationDescriptorCoverage,
  resolveKeplerOperationDescriptor,
  type KeplerOperationDescriptorCoverage,
  type KeplerResolvedOperationDescriptor
} from './operation-descriptors';
export {
  DATA_DEFAULT_SIGNIFICANT_DIGITS,
  KEPLER_DEFAULT_RULE_ROUNDING,
  type KeplerDefaultRuleRounding,
  KEPLER_CPU_WORK_CAPS,
  KEPLER_GRID_CELL_CAPS,
  getKeplerEffectiveEngine,
  getKeplerGridFrameExtent,
  type KeplerCpuWorkCap,
  resolveKeplerOperationParameters,
  type KeplerParameterResolution,
  type KeplerParameterResolutionOptions,
  type KeplerParameterSource,
  type KeplerResolvedParameters
} from './default-parameters';
export {
  KEPLER_BLOCKED_SUGGESTION_COUNT,
  KEPLER_CURATED_ANALYSIS_ORDER,
  KEPLER_GUESSED_SEMANTICS_MATCH_BONUS,
  KEPLER_GUESSED_SEMANTICS_MISMATCH_PENALTY,
  KEPLER_MAXIMUM_CANDIDATES_PER_BINDING,
  KEPLER_MAXIMUM_DATASETS_PER_INPUT,
  KEPLER_QUESTION_WINDOW,
  getKeplerDatasetGeometries,
  type KeplerGuessedFact,
  type KeplerSuggestionQuestion,
  KEPLER_CURATED_BONUS,
  KEPLER_DATASET_KIND_INPUT_KINDS,
  KEPLER_DEFAULT_SUGGESTION_COUNT,
  KEPLER_EXTRA_DATASET_BONUS,
  KEPLER_FAILED_WARNING_PENALTY,
  KEPLER_FAR_BELOW_MINIMUM_PENALTY,
  KEPLER_HIDDEN_PROMINENCE,
  KEPLER_MEASUREMENT_WARNING_PENALTY,
  KEPLER_NEEDS_MAPPING_PENALTY,
  KEPLER_OPTIONAL_ROLE_BONUS,
  KEPLER_OPTIONAL_ROLE_LIMIT,
  KEPLER_LIKE_STATES_BONUS,
  KEPLER_LIKE_STATES_OVERLAP,
  KEPLER_PLUMBING_PENALTY,
  KEPLER_PREDICTOR_ROLE,
  KEPLER_RESPONSE_ROLE,
  KEPLER_PREFERRED_MEASUREMENT_BONUS,
  KEPLER_PROMINENCE_WEIGHT,
  KEPLER_QUESTION_SLOT_SHARE,
  KEPLER_ROW_SEMANTICS_MISMATCH_PENALTY,
  KEPLER_SEMANTICS_MATCH_BONUS,
  KEPLER_SPECIFIC_INPUT_BONUS,
  KEPLER_TIME_COLUMN_BONUS,
  KEPLER_UNKNOWN_PENALTY,
  KEPLER_UNLIKE_STATES_PENALTY,
  evaluateKeplerOperationApplicability,
  getKeplerDatasetGeometry,
  getKeplerPrimaryQuestion,
  isKeplerDatasetCompatibleWithInput,
  listKeplerApplicableBindings,
  suggestKeplerAnalyses,
  type KeplerAnalysisSuggestion,
  type KeplerApplicabilityMissing,
  type KeplerApplicabilityOptions,
  type KeplerApplicabilityRequirement,
  type KeplerApplicabilityStatus,
  type KeplerApplicableBinding,
  type KeplerApplicableBindingGroup,
  type KeplerOperationApplicability,
  type KeplerSuggestionOptions,
  type KeplerSuggestionResult
} from './applicability';
export {
  evaluateKeplerWorkbenchInputs,
  profileKeplerWorkbenchData,
  type KeplerWorkbenchProfiles,
  evaluateKeplerWorkbenchApplicability,
  getKeplerWorkbenchCuratedPresets,
  getKeplerWorkbenchParameterDefaults,
  type KeplerWorkbenchApplicability,
  type KeplerWorkbenchCuratedPreset,
  type KeplerWorkbenchOptions,
  type KeplerWorkbenchParameterDefaults,
  type KeplerWorkbenchParameterValue,
  type KeplerWorkbenchSubject
} from './workbench-adapters';
export * from './descriptors-curated-emerging-hot-spots';
export * from './descriptors-curated-track-quality';
export * from './descriptors-curated-landscape-metrics';
export * from './descriptors-curated-fleet-dwell';

export * from './descriptors-curated-drainage-network';
export * from './descriptors-curated-surface-preparation';
export * from './descriptors-curated-zonal-exposure';
export * from './descriptors-curated-regional-raster-change';
export * from './descriptors-curated-network-disruption';
export * from './descriptors-curated-cell-flow-topology';
export * from './descriptors-curated-drive-time-bands';
