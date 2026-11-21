// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  addHotSpotAnalysisRecipe,
  GPU_HOT_SPOT_DEFAULT_MINIMUM_NEIGHBORS,
  GPU_HOT_SPOT_DEFAULT_MULTIPLE_TESTING
} from './hot-spot-analysis-recipe';
export type {
  GPUHotSpotAnalysisRecipeProps,
  GPUHotSpotAnalysisRecipeResult,
  GPUHotSpotAnalysisRecipeOutputs,
  GPUHotSpotAnalysisRecipeIntermediates,
  GPUHotSpotAnalysisRecipeSettings,
  GPUHotSpotColorOptions,
  GPUHotSpotLatticeSource,
  GPUHotSpotPermutationOptions,
  GPUHotSpotPointsSource,
  GPUHotSpotStudyArea,
  GPUHotSpotDistanceBand
} from './hot-spot-analysis-recipe';
export type {GPURecipeOverrides, GPURecipeResult} from './recipe-utils';

export {
  addRateClusterMapRecipe,
  GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH,
  GPU_RATE_CLUSTER_MAP_DEFAULT_MULTIPLE_TESTING
} from './rate-cluster-map-recipe';
export type {
  GPURateClusterMapRecipeProps,
  GPURateClusterMapRecipeResult,
  GPURateClusterMapRecipeOutputs,
  GPURateClusterMapRecipeIntermediates,
  GPURateClusterMapRecipeSettings,
  GPURateClusterMapRate
} from './rate-cluster-map-recipe';
export {
  addSpatialRegressionRecipe,
  getSpatialRegressionSpecification,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED
} from './spatial-regression-recipe';
export type {
  GPUSpatialRegressionRecipeProps,
  GPUSpatialRegressionRecipeResult,
  GPUSpatialRegressionRecipeOutputs,
  GPUSpatialRegressionRecipeSettings,
  GPUSpatialRegressionLocalFitOptions
} from './spatial-regression-recipe';
export {
  addSpaceTimeHotSpotsRecipe,
  getSpaceTimeDaysFromCivil,
  getSpaceTimeSliceIndex,
  getSpaceTimeStepIndex
} from './space-time-hot-spots-recipe';
export type {
  GPUSpaceTimeHotSpotsRecipeProps,
  GPUSpaceTimeHotSpotsRecipeResult,
  GPUSpaceTimeHotSpotsRecipeOutputs,
  GPUSpaceTimeHotSpotsRecipeIntermediates,
  GPUSpaceTimeHotSpotsRecipeSettings,
  GPUSpaceTimeSlices,
  GPUSpaceTimeCellIds,
  GPUSpaceTimeLattice,
  GPUSpaceTimeStep,
  GPUSpaceTimeOrigin
} from './space-time-hot-spots-recipe';
export {addPeriodComparisonRecipe} from './period-comparison-recipe';
export type {
  GPUPeriodComparisonRecipeProps,
  GPUPeriodComparisonRecipeResult,
  GPUPeriodComparisonRecipeOutputs,
  GPUPeriodComparisonRecipeIntermediates,
  GPUPeriodRows,
  GPUPeriodComparisonVariable
} from './period-comparison-recipe';
export {addPointsInPolygonsChoroplethRecipe} from './points-in-polygons-choropleth-recipe';
export type {
  GPUChoroplethColorOptions,
  GPUChoroplethPolygons,
  GPUChoroplethStatistic,
  GPUPointsInPolygonsChoroplethRecipeProps,
  GPUPointsInPolygonsChoroplethRecipeResult,
  GPUPointsInPolygonsChoroplethRecipeOutputs
} from './points-in-polygons-choropleth-recipe';
export {addClusterAndOutlineRecipe} from './cluster-and-outline-recipe';
export type {
  GPUClusterAndOutlineRecipeProps,
  GPUClusterAndOutlineRecipeResult,
  GPUClusterAndOutlineRecipeOutputs,
  GPUClusterAndOutlineRecipeIntermediates
} from './cluster-and-outline-recipe';
export {addChangeOfSupportRecipe} from './change-of-support-recipe';
export type {
  GPUChangeOfSupportRecipeProps,
  GPUChangeOfSupportRecipeResult,
  GPUChangeOfSupportRecipeOutputs,
  GPUChangeOfSupportRecipeIntermediates,
  GPUChangeOfSupportZones
} from './change-of-support-recipe';
export {addStraightLineCatchmentsRecipe} from './straight-line-catchments-recipe';
export type {
  GPUStraightLineCatchmentsRecipeProps,
  GPUStraightLineCatchmentsRecipeResult,
  GPUStraightLineCatchmentsRecipeOutputs
} from './straight-line-catchments-recipe';
export {addFleetDwellRecipe, addFleetDwellZoneEventsRecipe} from './fleet-dwell-recipe';
export type {
  GPUFleetDwellPolygonZones,
  GPUFleetDwellRecipeProps,
  GPUFleetDwellRecipeResult,
  GPUFleetDwellRecipeOutputs,
  GPUFleetDwellRecipeIntermediates,
  GPUFleetDwellTrajectories,
  GPUFleetDwellZoneEventsRecipeProps,
  GPUFleetDwellZoneEventsRecipeResult,
  GPUFleetDwellZoneEventsRecipeOutputs,
  GPUFleetDwellZoneEventsRecipeIntermediates,
  GPUFleetDwellZoneEventsTrajectories,
  GPUFleetDwellZoneTable,
  GPUFleetDwellZoneTableViews
} from './fleet-dwell-recipe';

export * from './zonal-distribution-recipe';

export * from './interpolation-diagnostics-recipe';

export * from './raster-summary-recipes';

export * from './surface-preparation-recipe';

export * from './drainage-network-recipe';

export * from './local-variability-recipe';
export * from './cell-flow-topology-recipe';

export * from './dasymetric-redistribution-recipe';

export * from '../spatial-autocorrelation/gpu-quantile-lisa';
export * from './rate-mapping-recipes';
export {
  addContiguityOrderCorrelogramRecipe,
  GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS
} from './contiguity-order-correlogram-recipe';
export {computePointsInPolygonsChoroplethOnCPU} from './cpu-points-in-polygons-choropleth-operation';
export type {CPUPointsInPolygonsChoroplethOptions} from './cpu-points-in-polygons-choropleth-operation';
export type {
  GPUContiguityOrderCorrelogramRecipeOutputs,
  GPUContiguityOrderCorrelogramRecipeProps,
  GPUContiguityOrderCorrelogramRecipeResult
} from './contiguity-order-correlogram-recipe';

export {createPointsInPolygonsChoroplethOperation} from './points-in-polygons-choropleth-operation';
export type {PointsInPolygonsChoroplethOperationProps} from './points-in-polygons-choropleth-operation';
export {createCPUPointsInPolygonsChoroplethOperation} from './cpu-points-in-polygons-choropleth-operation';
export {addStopValueCubeRecipe} from './stop-value-cube-recipe';
export type {
  GPUStopValueCubeRecipeProps,
  GPUStopValueCubeRecipeOutput,
  GPUStopValueCubeRecipeResult,
  GPUStopValueCubeRecipeStage
} from './stop-value-cube-recipe';
