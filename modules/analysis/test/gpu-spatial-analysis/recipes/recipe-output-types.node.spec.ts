// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expectTypeOf, it} from 'vitest';
import type {GPUEmergingHotSpotsProps} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import type {GPUPointInPolygonJoinProps} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import type {
  GPUChangeOfSupportRecipeResult,
  GPUClusterAndOutlineRecipeResult,
  GPUFleetDwellRecipeResult,
  GPUFleetDwellZoneEventsRecipeResult,
  GPUHotSpotAnalysisRecipeResult,
  GPUPeriodComparisonRecipeResult,
  GPUPointsInPolygonsChoroplethRecipeResult,
  GPURateClusterMapRecipeResult,
  GPUSpaceTimeHotSpotsRecipeResult,
  GPUSpatialRegressionRecipeResult,
  GPUStraightLineCatchmentsRecipeResult
} from '../../../src/gpu-spatial-analysis/recipes/index';

function createCubeConsumerProps(
  result: GPUSpaceTimeHotSpotsRecipeResult
): Pick<GPUEmergingHotSpotsProps, 'values'> {
  return {values: result.outputs.cube};
}

function createStopConsumerProps(
  result: GPUFleetDwellRecipeResult
): Pick<GPUPointInPolygonJoinProps, 'points'> {
  return {points: {kind: 'points', positions: result.outputs.stops.centroids}};
}

it('composes named outputs and intermediates without casts', () => {
  expectTypeOf<ReturnType<typeof createCubeConsumerProps>>().toEqualTypeOf<
    Pick<GPUEmergingHotSpotsProps, 'values'>
  >();
  expectTypeOf<ReturnType<typeof createStopConsumerProps>>().toEqualTypeOf<
    Pick<GPUPointInPolygonJoinProps, 'points'>
  >();
  expectTypeOf<GPUSpaceTimeHotSpotsRecipeResult['outputs']['cube']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUSpaceTimeHotSpotsRecipeResult['intermediates']['periods']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUFleetDwellRecipeResult['intermediates']['stopZones']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUFleetDwellZoneEventsRecipeResult['intermediates']['cellMask']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUHotSpotAnalysisRecipeResult['outputs']['zScores']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  expectTypeOf<GPURateClusterMapRecipeResult['outputs']['analyzedRates']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  expectTypeOf<GPUChangeOfSupportRecipeResult['outputs']['areas']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
  expectTypeOf<GPUPeriodComparisonRecipeResult['outputs']['colors']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUClusterAndOutlineRecipeResult['outputs']['labels']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUPointsInPolygonsChoroplethRecipeResult['outputs']['counts']>().toEqualTypeOf<
    GraphDataView<'uint32'>
  >();
  expectTypeOf<GPUSpatialRegressionRecipeResult['outputs']['ols']>().toEqualTypeOf<
    GPUSpatialRegressionRecipeResult['ols']
  >();
  expectTypeOf<GPUStraightLineCatchmentsRecipeResult['outputs']['distances']>().toEqualTypeOf<
    GraphDataView<'float32'>
  >();
});
