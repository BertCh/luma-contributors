// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {getWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-wgsl';
import {getTemporalWindowMassWGSL} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-wgsl';
import {getGPUKernelDensityWindowOverlap} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {getGPUSpaceTimeKernelDensityParameterValues} from '../../../src/gpu-spatial-analysis/point-density/space-time-kernel-density-math';
import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {createDensityInvalidCountNodes} from '../../../src/gpu-spatial-analysis/point-density/density-scratch';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

function readRawMass(kernel: 'gaussian' | 'quartic', body: string): number[] {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const result = createTransientView(graph, 'result', 'float32', 3);
    const memory = createCPUGraphMemory();
    const node = createWGSLKernelNode(graph, {
      id: 'probe',
      operation: 'DensityReviewProbe',
      invocationCount: 1,
      bindings: [{name: 'result', view: result, type: 'f32', access: 'read_write'}],
      declarations:
        getWindowMassWGSL(kernel) + getTemporalWindowMassWGSL(kernel, kernel !== 'gaussian'),
      body
    });
    executeCommandNodesOnCPU([node], memory);
    return Array.from(memory.readFloat32(result));
  } finally {
    device.destroy();
  }
}
function relative(actual: number, expected: number, tolerance = 2e-5) {
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(tolerance);
}

it('B3 off-center Gaussian intervals preserve their original width', () => {
  const actual = readRawMass(
    'gaussian',
    `result[resultOffset] = getTemporalWindowMass(1.0, -1e-8, 1e-8, 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(1.0, 0.0), vec2f(-1e-8, -1.0), vec2f(1e-8, 1.0), 1.0);`
  );
  // quad(normal.pdf, -1e-8, 1e-8, args=(loc=1)); spatial multiplies by erf(1/sqrt(2)).
  relative(actual[0], 4.839414490382867e-9);
  relative(actual[1], 3.3038174206803345e-9);
});

it('B4 half and partially clipped quartic caps retain positive mass', () => {
  // Independent scipy dblquad/chord integrals, retained in the review reference script.
  relative(
    getGPUKernelDensityWindowOverlap('quartic', 1, [0, 0], [0.999999, 0, 1, 1]),
    8.231455459997955e-22,
    2e-7
  );
  relative(
    getGPUKernelDensityWindowOverlap('quartic', 1, [0, 0], [0.995, 0.02, 1, 1]),
    4.1932519134435055e-9,
    2e-7
  );
  const actual = readRawMass(
    'quartic',
    `result[resultOffset] = getWindowMass(vec2f(0.0), vec2f(.995, 0.0), vec2f(1.0), 1.0);
result[resultOffset + 1u] = getWindowMass(vec2f(0.0), vec2f(.995, .02), vec2f(1.0), 1.0);`
  );
  relative(actual[0], 7.240316852684013e-9, 1e-4);
  relative(actual[1], 4.193236664650067e-9, 1e-4);
});

it('B5 centered temporal quartic interval avoids subtracting two halves', () => {
  const actual = readRawMass(
    'quartic',
    'result[resultOffset] = getTemporalWindowMass(0.0, -2.9e-8, 2.9e-8, 1.0);'
  );
  relative(actual[0], 5.4375e-8);
});

it('B2 default time-window endpoint agrees with fused slice evaluation', () => {
  const packed = getGPUSpaceTimeKernelDensityParameterValues(
    {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0.1, sliceInterval: 0.05},
    makeExtentGrid(1, 1, [0, 0, 1, 1]),
    8
  );
  // Exact f32 operands summed before one rounding: float32(float32(.1) + 7*float32(.05)).
  expect(packed[13]).toBe(0.45000001788139343);
});

it('S1 slice representability preflight is bounded rather than visiting 500000 slices', () => {
  const round = vi.spyOn(Math, 'fround');
  try {
    getGPUSpaceTimeKernelDensityParameterValues(
      {bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
      makeExtentGrid(1, 1, [0, 0, 1, 1]),
      500000
    );
    expect(round.mock.calls.length).toBeLessThan(10000);
  } finally {
    round.mockRestore();
  }
});

it('B6 point planner checks the packed mask and only requested sum scratch', () => {
  const limits = {maxStorageBufferBindingSize: 1024, maxBufferSize: 4096};
  expect(
    GPUPointDensity.getExecutionPlan(
      {pointCount: 300, gridSize: [2, 1], masked: true, positionChunkLengths: [100, 100, 100]},
      limits
    ).feasible
  ).toBe(false);
  expect(
    GPUPointDensity.getExecutionPlan({pointCount: 3, gridSize: [10, 10], weighted: true}, limits)
      .feasible
  ).toBe(true);
  expect(() => GPUPointDensity.getExecutionPlan({pointCount: 3, gridSize: [2, 0]}, limits)).toThrow(
    /gridSize/
  );
});

it('B7 KDE planner checks weight topology and encoded frame validity', () => {
  const device = createNullWebGPUDevice();
  try {
    const shape = {pointCount: 4, gridSize: [2, 2] as const, indexGridSize: [1, 1] as const};
    expect(
      GPUKernelDensity.getExecutionPlan({...shape, weightChunkLengths: [2, 2]}, device.limits)
        .feasible
    ).toBe(false);
    expect(
      GPUKernelDensity.getExecutionPlan(
        {...shape, method: 'binned', positionChunkLengths: [2, 1]},
        device.limits
      ).feasible
    ).toBe(false);
    expect(() =>
      GPUPointDensity.getExecutionPlan(
        {pointCount: 4, gridSize: [2, 2], statistic: 'sum'},
        device.limits
      )
    ).toThrow(/requires weights/);
    expect(
      GPUKernelDensity.getExecutionPlan(
        {...shape, method: 'auto', bandwidth: 4.8, cellSize: [0.75, 0.75], binnedPadding: 32},
        device.limits
      ).route
    ).toBe('exact');
    expect(
      GPUKernelDensity.getExecutionPlan(
        {...shape, method: 'auto', bandwidth: 1e-29, cellSize: [1e-30, 1e-30], binnedPadding: 32},
        device.limits
      ).feasible
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('B8 fixed-bandwidth binned planning retains the existing 1200-chunk allowance', () => {
  const device = createNullWebGPUDevice();
  try {
    const plan = GPUKernelDensity.getExecutionPlan(
      {
        pointCount: 1200,
        positionChunkLengths: Array(1200).fill(1),
        gridSize: [1, 1],
        indexGridSize: [1, 1],
        method: 'binned'
      },
      device.limits
    );
    expect(plan.feasible).toBe(true);
    expect(
      plan.constraints.find(constraint => constraint.name === 'command-budget')?.received
    ).toBe(28112);
  } finally {
    device.destroy();
  }
});

it('S2 foreign vectors take the invalid-count chunk route', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const vector = new GraphVectorView({
      id: 'foreign',
      name: 'foreign',
      format: 'uint32',
      length: 2,
      valueLength: 2,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [
        createTransientView(graph, 'one', 'uint32', 1),
        createTransientView(graph, 'two', 'uint32', 1)
      ]
    });
    const nodes = createDensityInvalidCountNodes(
      graph,
      'count',
      {...vector} as GraphVectorView<'uint32'>,
      createTransientView(graph, 'out', 'uint32', 1)
    );
    expect(nodes.some(node => node.id === 'count-clear')).toBe(true);
  } finally {
    device.destroy();
  }
});

it('UNFINISHED dense smoothing options and bounds fail as named construction requirements', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      id: 'named-density',
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      gridSize: [2, 1] as const,
      bounds: [0, 0, 2, 1] as const,
      output: {values: createTransientView(graph, 'values', 'float32', 2)}
    };
    expect(
      () =>
        new GPUPointDensity({
          ...props,
          smoothing: {
            kernel: createTransientView(graph, 'kernel', 'float32', 1),
            kernelWidth: 1,
            kernelHeight: 1,
            strategy: 'typo' as never
          }
        })
    ).toThrow(/named-density.*strategy.*typo.*auto.*direct.*fft/);
    try {
      new GPUPointDensity({...props, bounds: [0, 0, -1, 1]});
      throw new Error('accepted');
    } catch (error) {
      expect((error as Error).name).toBe('AnalysisValidationError');
    }
    expect(() =>
      getGPUKernelDensityParameterValues(
        {bandwidth: 1, window: [0.7, 0, 0.700000001, 1]},
        makeExtentGrid(1, 1, [0, 0, 1, 1])
      )
    ).toThrow(/window x.*encoded/);
  } finally {
    device.destroy();
  }
});

it('B6 planned packed-mask refusal and count-only admission match declaration', () => {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBufferBindingSize: 1024, maxBufferSize: 4096});
  try {
    const graph = new GPUCommandGraph(device);
    const chunks = [0, 1, 2].map(index =>
      createTransientView(graph, `positions-${index}`, 'float32x2', 100)
    );
    const positions = new GraphVectorView({
      id: 'positions',
      name: 'positions',
      format: 'float32x2',
      length: 300,
      valueLength: 600,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: chunks
    });
    const masked = new GPUPointDensity({
      positions,
      mask: createTransientView(graph, 'mask', 'uint32', 300),
      gridSize: [2, 1],
      bounds: [0, 0, 2, 1],
      output: {values: createTransientView(graph, 'masked-values', 'float32', 2)}
    });
    expect(() => masked.getCommandNodes(graph)).toThrow(/mask/);
    const countOnly = new GPUPointDensity({
      positions: createTransientView(graph, 'packed-positions', 'float32x2', 3),
      weights: createTransientView(graph, 'weights', 'float32', 3),
      gridSize: [10, 10],
      bounds: [0, 0, 10, 10],
      output: {values: createTransientView(graph, 'count-values', 'float32', 100)}
    });
    expect(countOnly.getCommandNodes(graph)).toHaveLength(3);
  } finally {
    device.destroy();
  }
});

it('B8 fixed-bandwidth binned constructor admits the existing 1200 one-row chunks', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = new GraphVectorView({
      id: 'positions',
      name: 'positions',
      format: 'float32x2',
      length: 1200,
      valueLength: 2400,
      stride: 2,
      byteStride: 8,
      rowByteLength: 8,
      data: Array.from({length: 1200}, (_, index) =>
        createTransientView(graph, `point-${index}`, 'float32x2', 1)
      )
    });
    const density = new GPUKernelDensity({
      positions,
      method: 'binned',
      grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
      parameters: createTransientView(graph, 'params', 'float32', 9),
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 1, 1],
      output: {values: createTransientView(graph, 'values', 'float32', 1)}
    });
    expect(density.estimatedCommandNodeCount).toBe(28112);
  } finally {
    device.destroy();
  }
});

it('S1 bounded slice search agrees with independently enumerated dyadic controls', async () => {
  const {createDensitySliceTime, findDensityRepeatedSlice} = await import(
    '../../../src/gpu-spatial-analysis/point-density/density-slice-times'
  );
  for (const first of [0, 1, -1, 1024, -1024, 2 ** 24, -(2 ** 24)]) {
    for (const interval of [2 ** -25, 2 ** -23, 2 ** -13, 0.5, 1, 2]) {
      // These dyadic products/sums fit binary64 exactly at the 4096 tested indices.
      let expected: number | undefined;
      for (let index = 1; index < 4096; index++) {
        if (
          !(Math.fround(first + index * interval) > Math.fround(first + (index - 1) * interval))
        ) {
          expected = index;
          break;
        }
      }
      expect(
        findDensityRepeatedSlice(createDensitySliceTime(first, interval), interval, 4096),
        `${first},${interval}`
      ).toBe(expected);
    }
  }
});

it('B7 parallel auto gate and numeric planner use the same single-rounded products', async () => {
  const {DENSITY_AUTO_ROUTE_WGSL, getDensityFrameRoute} = await import(
    '../../../src/gpu-spatial-analysis/point-density/density-auto-route'
  );
  const {IS_FINITE_VALUE_WGSL} = await import('../../../src/analysis-core/kernels/wgsl-common');
  for (const [bandwidth, cell, valid, binned] of [
    [4.8, 0.75, 1, 0],
    [4.80001, 0.75, 1, 1],
    [1e-29, 1e-30, 0, 0]
  ]) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const parameters = createTransientView(graph, 'params', 'float32', 2);
      const result = createTransientView(graph, 'result', 'uint32', 2);
      memory.write(parameters, new Float32Array([bandwidth, cell]));
      const node = createWGSLKernelNode(graph, {
        id: 'route',
        operation: 'DensityReviewProbe',
        invocationCount: 1,
        bindings: [
          {name: 'params', view: parameters, type: 'f32', access: 'read'},
          {name: 'result', view: result, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL + DENSITY_AUTO_ROUTE_WGSL,
        body: 'let bandwidth = params[paramsOffset]; let cell = vec2f(params[paramsOffset + 1u]); result[resultOffset] = select(0u, 1u, isDensityFrameValid(bandwidth, cell)); result[resultOffset + 1u] = select(0u, 1u, useDensityBinnedRoute(bandwidth, cell, 33u));'
      });
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(result))).toEqual([valid, binned]);
      expect(getDensityFrameRoute(bandwidth, [cell, cell], 33)).toEqual({
        valid: Boolean(valid),
        useBinned: Boolean(binned)
      });
    } finally {
      device.destroy();
    }
  }
});
