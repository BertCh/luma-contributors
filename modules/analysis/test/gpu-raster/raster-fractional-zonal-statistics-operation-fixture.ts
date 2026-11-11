// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPURasterFractionalZonalStatisticsInput} from '../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation';
import type {RasterFractionalZonalStatisticsOperationProps} from '../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation';
import {getGPURasterFractionalCoverageParameterValues} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import type {OperationGraphInput} from '../utils/mixed-execution-test-utils';

export const STATISTICS_PROPS: RasterFractionalZonalStatisticsOperationProps = {
  id: 'fractional',
  grid: {width: 3, height: 2, affine: [1, 0, 0, 0, -1, 2], pixelInterpretation: 'area'},
  input: {
    id: 'band',
    format: 'float32',
    storage: {kind: 'buffer', values: 'values'},
    validity: 'inputValidity',
    scale: 2,
    offset: 1,
    noDataValue: 10
  },
  zoneCount: 4,
  quantity: 'density',
  parameters: 'parameters',
  coverage: {
    zoneIds: 'zoneIds',
    cellIds: 'cellIds',
    fractions: 'fractions',
    zoneValidity: 'zoneValidity',
    status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
  },
  output: {
    totals: 'totals',
    means: 'means',
    coveredArea: 'coveredArea',
    validArea: 'validArea',
    validity: 'validity'
  }
};

export function createStatisticsInput(): CPURasterFractionalZonalStatisticsInput {
  return {
    input: {
      id: 'band',
      format: 'float32',
      values: Float32Array.of(2, 4, 6, 8, 10, 12),
      validity: Uint32Array.of(1, 1, 0, 1, 1, 1),
      scale: 2,
      offset: 1,
      noDataValue: 10
    },
    // Repacked per-frame cell area 6; the compile-time grid's area is 1.
    parameters: getGPURasterFractionalCoverageParameterValues({
      ...STATISTICS_PROPS.grid,
      affine: [2, 0, 0, 0, -3, 6]
    }),
    coverage: {
      zoneIds: Uint32Array.of(0, 1, 0, 1, 0, 1, 2, 1, 99, 0),
      cellIds: Uint32Array.of(0, 3, 1, 4, 2, 5, 2, 99, 0, 5),
      fractions: Float32Array.of(0.5, 1, 0.25, 0.5, 0.75, 0.25, 1, 1, 1, 1),
      zoneValidity: Uint32Array.of(1, 1, 1, 1),
      // Tail tuple (zone 0, cell 5) is inactive; invalid IDs in active tuples are ignored.
      count: Uint32Array.of(9),
      requiredCount: Uint32Array.of(9),
      overflow: Uint32Array.of(0)
    }
  };
}

export function createStatisticsGraphInputs(
  input = createStatisticsInput()
): Record<string, OperationGraphInput> {
  const graphInputs: Record<string, OperationGraphInput> = {
    values: {format: input.input.format, values: input.input.values},
    inputValidity: {format: 'uint32', values: input.input.validity!},
    parameters: {format: 'float32', values: input.parameters!}
  };
  for (const [port, values] of Object.entries(input.coverage))
    graphInputs[port] = {format: port === 'fractions' ? 'float32' : 'uint32', values};
  return graphInputs;
}

export function createSingletonStatisticsProps(): RasterFractionalZonalStatisticsOperationProps {
  return {
    ...STATISTICS_PROPS,
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
    zoneCount: 1
  };
}

export function createSingletonStatisticsInput(
  surplusCoverageRows = false
): CPURasterFractionalZonalStatisticsInput {
  const props = createSingletonStatisticsProps();
  const input = createStatisticsInput();
  input.input.values = Float32Array.of(2); // Calibrated value 2 * 2 + 1 = 5.
  input.input.validity = Uint32Array.of(1);
  input.parameters = getGPURasterFractionalCoverageParameterValues(props.grid);
  input.coverage = {
    zoneIds: Uint32Array.of(0),
    cellIds: surplusCoverageRows ? Uint32Array.of(0, 0xffffffff) : Uint32Array.of(0),
    fractions: surplusCoverageRows ? Float32Array.of(1, 99) : Float32Array.of(1),
    zoneValidity: Uint32Array.of(1, 0),
    count: surplusCoverageRows ? Uint32Array.of(1, 0) : Uint32Array.of(1),
    requiredCount: surplusCoverageRows ? Uint32Array.of(1, 0) : Uint32Array.of(1),
    overflow: surplusCoverageRows ? Uint32Array.of(0, 1) : Uint32Array.of(0)
  };
  return input;
}
