// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateOperationGraphNames} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateChangePointLayout} from './change-point-series-wgsl';
import type {PettittTestOperationProps} from './gpu-pettitt-test-operation';
import type {LocationChangePointsOperationProps} from './gpu-location-change-points-operation';

/** Largest `sliceCount`: every invocation compares `sliceCount^2` pairs. */
export const GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT = 512;
/**
 * Largest `sliceCount`. Every location keeps its standardized series and the dynamic program in
 * workgroup arrays and evaluates `sliceCount^2 / 2` segment costs, so the bound limits both.
 */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT = 512;
/**
 * Largest `sliceCount` of the `'slope'` cost: the time sums of a segment are exact unsigned
 * integers, and `n * sum(t^2)` with positions below 256 stays below `2^32`.
 */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT = 256;
/** Largest `maximumChangePoints` per location. */
export const GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS = 128;

/** @internal At most 512 smaller offsets keep both the mean sum and its square in float32. */
export const LOCATION_CHANGE_POINT_MEAN_SCALE_THRESHOLD_EXPONENT = 54;
/** @internal Scale wide segment offsets by 2^-16 before accumulating their moments. */
export const LOCATION_CHANGE_POINT_MEAN_SCALE_EXPONENT = 16;
/** @internal Keep smaller observations in their original units during standardization. */
export const LOCATION_CHANGE_POINT_STANDARDIZATION_THRESHOLD_EXPONENT = 54;
/** @internal Bound wide observations below 2^51 before centering and summing up to 512 squares. */
export const LOCATION_CHANGE_POINT_STANDARDIZATION_EXPONENT = 50;

/** GPU classes and operations share the same host-known cube bounds. @internal */
export function validatePettittShape(
  id: string,
  props: Pick<PettittTestOperationProps, 'cellCount' | 'sliceCount' | 'layout'>,
  maximumSliceCount = GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT
): void {
  validateChangePointLayout(id, props.layout);
  validateCapacity(id, 'cellCount', props.cellCount);
  validateCapacity(id, 'sliceCount', props.sliceCount, 1, maximumSliceCount);
  validateCapacity(id, 'cellCount * sliceCount', props.cellCount * props.sliceCount);
}

/** CPU partitioning keeps uint32 shapes while GPU routes impose their private-array bounds. @internal */
export function validateLocationChangeShape(
  id: string,
  props: Pick<
    LocationChangePointsOperationProps,
    'cellCount' | 'sliceCount' | 'layout' | 'cost' | 'maximumChangePoints'
  >,
  route: 'gpu' | 'cpu' = 'gpu'
): void {
  validateChangePointLayout(id, props.layout);
  validateRequirement(
    id,
    ['mean', 'variance', 'mean-variance', 'poisson', 'slope'].includes(props.cost),
    `cost must be mean, variance, mean-variance, poisson or slope; received ${props.cost}`
  );
  validateCapacity(id, 'cellCount', props.cellCount);
  validateCapacity(
    id,
    'sliceCount',
    props.sliceCount,
    1,
    route === 'gpu' ? GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT : 0xffffffff
  );
  if (route === 'gpu' && props.cost === 'slope')
    validateCapacity(
      id,
      "sliceCount of the 'slope' cost",
      props.sliceCount,
      1,
      GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT
    );
  validateCapacity(
    id,
    'maximumChangePoints',
    props.maximumChangePoints,
    1,
    route === 'gpu' ? GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS : 0xffffffff
  );
  validateCapacity(id, 'cellCount * sliceCount', props.cellCount * props.sliceCount);
  validateCapacity(
    id,
    'cellCount * maximumChangePoints',
    props.cellCount * props.maximumChangePoints
  );
}

/** Checks logical port names without instantiating a GPU operation or applying GPU limits. @internal */
export function validatePettittOperationPorts(id: string, props: PettittTestOperationProps): void {
  validateOperationGraphNames(id, {values: props.values}, props.output);
}

/** Checks logical port names without instantiating a GPU operation or applying GPU limits. @internal */
export function validateLocationChangeOperationPorts(
  id: string,
  props: LocationChangePointsOperationProps
): void {
  validateOperationGraphNames(
    id,
    {values: props.values, parameters: props.parameters},
    {
      changePointCount: props.output.changePointCount,
      changePoints: props.output.changePoints,
      ...(props.output.requiredChangePointCounts !== undefined
        ? {requiredChangePointCounts: props.output.requiredChangePointCounts}
        : {}),
      ...(props.output.optimalCost !== undefined ? {optimalCost: props.output.optimalCost} : {}),
      ...(props.output.status !== undefined ? {overflow: props.output.status.overflow} : {})
    }
  );
}
