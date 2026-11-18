// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPURasterFractionalZonalStatisticsInput} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import type {RasterFractionalZonalStatisticsOperationProps} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation';
import type {OperationGraphInput} from '../utils/mixed-execution-test-utils';
import {STATISTICS_PROPS} from './raster-fractional-zonal-statistics-operation-fixture';

export function createScalarAreaReviewFixture(
  scenario: 'offset' | 'oversized' | 'overflow' | 'large-affine'
) {
  const overflow = scenario === 'overflow';
  const cellCount = overflow ? 3 : 1;
  const props: RasterFractionalZonalStatisticsOperationProps = {
    ...STATISTICS_PROPS,
    zoneCount: overflow ? 2 : 1,
    grid: {
      width: cellCount,
      height: 1,
      affine: scenario === 'large-affine' ? [1e20, 0, 0, 0, 1e20, 0] : [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    },
    input: {id: 'values', format: 'float32', storage: {kind: 'buffer', values: 'values'}},
    parameters: undefined,
    cellAreas: 'cellAreas'
  };
  const input: CPURasterFractionalZonalStatisticsInput = {
    input: {
      id: 'values',
      format: 'float32',
      values: overflow ? Float32Array.of(1, 1, 2) : Float32Array.of(2)
    },
    cellAreas: overflow
      ? Float32Array.of(2e38, 2e38, 3)
      : Float32Array.of(99, 3, NaN).subarray(1, scenario === 'oversized' ? 3 : 2),
    coverage: {
      zoneIds: overflow ? Uint32Array.of(0, 0, 1) : Uint32Array.of(0),
      cellIds: overflow ? Uint32Array.of(0, 1, 2) : Uint32Array.of(0),
      fractions: new Float32Array(cellCount).fill(1),
      zoneValidity: new Uint32Array(props.zoneCount).fill(1),
      count: Uint32Array.of(cellCount),
      requiredCount: Uint32Array.of(cellCount),
      overflow: Uint32Array.of(0)
    }
  };
  const graphInputs: Record<string, OperationGraphInput> = {
    values: {format: 'float32', values: input.input.values},
    cellAreas: {format: 'float32', values: input.cellAreas!}
  };
  for (const [name, values] of Object.entries(input.coverage))
    graphInputs[name] = {format: name === 'fractions' ? 'float32' : 'uint32', values};
  return {props, input, graphInputs};
}
