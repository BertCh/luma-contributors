// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterTemporalResample,
  getGPURasterTemporalResampleParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPURasterCentredFractionalYears} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

function executeMillisecondGrid(
  times: BigInt64Array,
  gridCount: number,
  maximumGap: number,
  useGridEnd = false
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const years = getGPURasterCentredFractionalYears(times, 'millisecond');
  const values = createTransientView(graph, 'values', 'float32', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 3);
  const dates = createTransientView(graph, 'dates', 'uint32x2', 2);
  const resampled = createTransientView(graph, 'resampled', 'float32', gridCount);
  const status = createTransientView(graph, 'status', 'uint32', gridCount);
  const contributor = new GPURasterTemporalResample({
    cellCount: 1,
    sliceCount: 2,
    gridCount,
    ...(useGridEnd ? {gridEnd: years[1]} : {}),
    values,
    parameters,
    dateAxis: {
      acquisitionTimes: dates,
      temporalContext: {
        unit: 'millisecond',
        epoch: 0n,
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    },
    output: {values: resampled, status}
  });
  const memory = createCPUGraphMemory();
  memory.write(values, Float32Array.of(7, 11));
  memory.write(dates, new Uint32Array(times.buffer, times.byteOffset, times.length * 2));
  memory.write(
    parameters,
    getGPURasterTemporalResampleParameterValues({
      gridStart: years[0],
      gridStep: (years[1] - years[0]) / (gridCount - 1),
      maximumGap
    })
  );
  memory.write(status, new Uint32Array(gridCount).fill(0xdeadbeef));
  try {
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    return {
      years: Array.from(years),
      values: Array.from(memory.readFloat32(resampled)),
      status: Array.from(memory.readUint32(status))
    };
  } finally {
    device.destroy();
  }
}

it('retains millisecond endpoints supplied by the public centred-year helper', () => {
  const result = executeMillisecondGrid(BigInt64Array.of(0n, 82800000n), 2, 0.001);
  // The output grid consists of the two observations, irrespective of the intervening gap.
  expect(result.status).toEqual([0, 0]);
  expect(result.values).toEqual([7, 11]);
  // f32(41400000 / f32(31556952000)), with sign set by the centred timestamp.
  expect(result.years).toEqual([-0.0013119137147441506, 0.0013119137147441506]);
});

it('retains an explicitly supplied final date on a 24-point millisecond grid', () => {
  const result = executeMillisecondGrid(BigInt64Array.of(0n, 950400000n), 24, 1, true);
  // An inclusive line from two known observations is exactly 7+4*i/23, independent
  // of the centred-date representation or the packed grid step's rounding.
  expect(result.status).toEqual(new Array(24).fill(0));
  expect(result.values[0]).toBe(7);
  expect(result.values[23]).toBe(11);
  result.values.forEach((value, point) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - (7 + (4 * point) / 23))).toBeLessThanOrEqual(2e-6);
  });
});

it('keeps step-derived outside-range behavior when gridEnd is absent', () => {
  const result = executeMillisecondGrid(BigInt64Array.of(0n, 950400000n), 24, 1);
  // This regular step grid rounds its final coordinate past the last observation.
  // Only an explicit endpoint-inclusive option may replace that coordinate.
  expect(result.status).toEqual([...new Array(23).fill(0), 1]);
  expect(result.values[0]).toBe(7);
  expect(result.values[23]).toBeNaN();
});

it('executes exact observed endpoints before rejecting the interior of a long gap', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 3);
  const dates = createTransientView(graph, 'dates', 'uint32x2', 2);
  const resampled = createTransientView(graph, 'resampled', 'float32', 5);
  const status = createTransientView(graph, 'status', 'uint32', 5);
  const contributor = new GPURasterTemporalResample({
    cellCount: 1,
    sliceCount: 2,
    gridCount: 5,
    values,
    parameters,
    dateAxis: {
      acquisitionTimes: dates,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'int64-words',
        axis: {kind: 'irregular'}
      }
    },
    output: {values: resampled, status}
  });
  const memory = createCPUGraphMemory();
  memory.write(values, Float32Array.of(7, 11));
  // Dates 0 and two Gregorian mean years have exactly centred coordinates -1 and +1.
  memory.write(dates, Uint32Array.of(0, 0, 63113904, 0));
  memory.write(
    parameters,
    getGPURasterTemporalResampleParameterValues({gridStart: -2, gridStep: 1, maximumGap: 1})
  );
  memory.write(status, new Uint32Array(5).fill(0xdeadbeef));
  try {
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    // Observations at -1/+1 survive any gap limit. Only zero is inside their two-year gap;
    // -2/+2 require extrapolation, which is forbidden. These are independent endpoint semantics.
    expect(Array.from(memory.readUint32(status))).toEqual([1, 0, 2, 0, 1]);
    expect(Array.from(memory.readFloat32(resampled))).toEqual([NaN, 7, NaN, 11, NaN]);
  } finally {
    device.destroy();
  }
});
