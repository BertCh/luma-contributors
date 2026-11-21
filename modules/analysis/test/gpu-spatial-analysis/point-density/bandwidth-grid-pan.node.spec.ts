// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUReduction,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createKernelDensityBandwidthNodes} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-bandwidth';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import * as densityPasses from '../../../src/gpu-spatial-analysis/point-density/density-point-passes';
import * as densityScratch from '../../../src/gpu-spatial-analysis/point-density/density-scratch';
import {getGraphViewChunks} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it.each([
  false,
  true
])('B6 Scott bandwidth is unchanged by an output-grid pan (zero-weight outlier: %s)', withZeroWeight => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  // Trust upstream reductions: only emulate their sum/min contract, while executing the actual
  // contributor's float32 row, centering and bandwidth kernels on the WGSL interpreter.
  const reductionNode = (
    id: string,
    input: Parameters<typeof getGraphViewChunks>[0],
    output: GraphDataView,
    operation: 'sum' | 'min' | 'max'
  ) => ({
    id,
    type: 'compute' as const,
    resources: [],
    compile: () => ({
      encode: () => {
        const values = getGraphViewChunks(input).flatMap(chunk =>
          Array.from(memory.readFloat32(chunk))
        );
        const result =
          operation === 'min'
            ? Math.min(...values)
            : operation === 'max'
              ? Math.max(...values)
              : values.reduce((sum, value) => sum + value, 0);
        memory.write(output, new Float32Array([result]));
      }
    })
  });
  // The interpreter has no uniform binding support. Specialize only dispatch metadata;
  // all arithmetic in declarations/body remains the contributor's shader text.
  const passSpy = vi
    .spyOn(densityPasses, 'createDensityPointPass')
    .mockImplementation((targetGraph, props) =>
      createWGSLKernelNode(targetGraph, {
        ...props,
        declarations:
          Object.entries(props.runtimeConstants ?? {})
            .map(([name, value]) => `const ${name}: u32 = ${value}u;`)
            .join('\n') +
          '\n' +
          (props.declarations ?? '')
      })
    );
  const sumSpy = vi
    .spyOn(densityScratch, 'createDensityWeightSumNodes')
    .mockImplementation(
      (_graph, id, input, output) => [reductionNode(id, input, output, 'sum')] as never
    );
  const minimumSpy = vi
    .spyOn(GPUReduction.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUReduction) {
      if (this.operation !== 'min' && this.operation !== 'max')
        throw new Error('expected an origin or weight-scale reduction');
      return [reductionNode(this.id, this.input, this.output, this.operation)] as never;
    });
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', withZeroWeight ? 5 : 4);
    const weights = withZeroWeight
      ? createTransientView(graph, 'weights', 'float32', 5)
      : undefined;
    if (weights) memory.write(weights, new Float32Array([1, 1, 1, 1, 0]));
    const parameters = createTransientView(graph, 'parameters', 'float32', 9);
    const bandwidthState = createTransientView(graph, 'bandwidth', 'float32', 1);
    memory.write(
      positions,
      new Float32Array([
        500000,
        5000000,
        500000.03125,
        5000000,
        500000,
        5000000.5,
        500000.03125,
        5000000.5,
        ...(withZeroWeight ? [0, 0] : [])
      ])
    );
    const nodes = createKernelDensityBandwidthNodes(graph, {
      id: 'density',
      kernel: 'gaussian',
      props: {positions, weights, parameters, indexBounds: [0, 0, 1, 1]},
      edgeCorrection: 'none',
      bandwidthState
    });
    // Independent unbiased covariance: varianceX=(.03125)^2/3, varianceY=(.5)^2/3.
    // Scott h=4^(-1/6)*sqrt((varianceX+varianceY)/2)=.16232956552540634.
    for (const [startX, startY] of [
      [0, 0],
      [499999, 4999999],
      [-1000000, -1000000]
    ]) {
      memory.write(
        parameters,
        new Float32Array([startX, startY, startX + 2, startY + 2, 99, 0, 0, 1, 1])
      );
      executeCommandNodesOnCPU(nodes, memory);
      const actual = memory.readFloat32(bandwidthState)[0];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(0.16232956552540634, 7);
    }
  } finally {
    sumSpy.mockRestore();
    minimumSpy.mockRestore();
    passSpy.mockRestore();
    device.destroy();
  }
});

it.each([
  {
    population: 'included' as const,
    points: [0, 0, 0.25, 0.25, 0.5000000596046448, 0],
    weights: undefined,
    count: 2,
    sum: 2,
    invalid: 1,
    bandwidth: 0.15749013123685915
  },
  {
    population: 'included' as const,
    points: [0, 0, 0.25, 0, 0, 0.25, 0.75, 0, 0.5000000596046448, 0, 0.125, 0.125],
    weights: [1, 2, 4, 0, 3, 0],
    count: 4,
    sum: 7,
    invalid: 2,
    bandwidth: 0.13605918693404473
  },
  {
    population: 'all-finite' as const,
    points: [0, 0, 0.25, 0, 0, 0.25, 0.75, 0, 0.5000000596046448, 0, 0.125, 0.125],
    weights: [1, 2, 4, 0, 3, 0],
    count: 6,
    sum: 7,
    invalid: 2,
    bandwidth: 0.17287395607033731
  }
])('B1/S3 $population population, moments and density membership agree (count $count)', async testCase => {
  const {GPUKernelDensity, getGPUKernelDensityParameterValues} = await import(
    '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density'
  );
  const {makeExtentGrid} = await import('../spatial-interpolation/extent-grid');
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const reduction = (
    id: string,
    input: Parameters<typeof getGraphViewChunks>[0],
    output: GraphDataView,
    operation: string
  ) => ({
    id,
    type: 'compute' as const,
    resources: [],
    compile: () => ({
      encode: () => {
        const values = getGraphViewChunks(input).flatMap(chunk =>
          Array.from(
            chunk.format === 'uint32' ? memory.readUint32(chunk) : memory.readFloat32(chunk)
          )
        );
        const value =
          operation === 'min'
            ? Math.min(...values)
            : operation === 'max'
              ? Math.max(...values)
              : values.reduce((total, item) => total + item, 0);
        memory.write(
          output,
          output.format === 'uint32' ? new Uint32Array([value]) : new Float32Array([value])
        );
      }
    })
  });
  const passes = vi
    .spyOn(densityPasses, 'createDensityPointPass')
    .mockImplementation((target, props) =>
      createWGSLKernelNode(target, {
        ...props,
        declarations:
          Object.entries(props.runtimeConstants ?? {})
            .map(([name, value]) => `const ${name}: u32 = ${value}u;`)
            .join('\n') +
          '\n' +
          (props.declarations ?? '')
      })
    );
  const sums = vi
    .spyOn(densityScratch, 'createDensityWeightSumNodes')
    .mockImplementation(
      (_graph, id, input, output) => [reduction(id, input, output, 'sum')] as never
    );
  const reductions = vi
    .spyOn(GPUReduction.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUReduction) {
      return [reduction(this.id, this.input, this.output, this.operation)] as never;
    });
  try {
    const positions = createTransientView(
      graph,
      'positions',
      'float32x2',
      testCase.points.length / 2
    );
    const weights = testCase.weights
      ? createTransientView(graph, 'weights', 'float32', testCase.weights.length)
      : undefined;
    if (weights && testCase.weights) memory.write(weights, Float32Array.from(testCase.weights));
    const bandwidth = createTransientView(graph, 'bandwidth', 'float32', 1);
    const parameters = createTransientView(graph, 'parameters', 'float32', 9);
    const weightSum = createTransientView(graph, 'weight-sum', 'float32', 1);
    const count = createTransientView(graph, 'population', 'uint32', 1);
    const invalid = createTransientView(graph, 'invalid', 'uint32', 1);
    const grid = makeExtentGrid(1, 1, [-1, -1, 1, 1]);
    memory.write(positions, Float32Array.from(testCase.points));
    memory.write(
      parameters,
      getGPUKernelDensityParameterValues({bandwidth: 1, window: [-1, -1, 0.5, 0.5]}, grid)
    );
    const nodes = new GPUKernelDensity({
      id: 'membership',
      positions,
      weights,
      parameters,
      grid,
      indexGridSize: [1, 1],
      indexBounds: [-1, -1, 2, 2],
      bandwidthRule: 'scott',
      bandwidthPopulation: testCase.population,
      output: {
        values: createTransientView(graph, 'values', 'float32', 1),
        weightSum,
        bandwidth,
        bandwidthPopulationCount: count,
        status: {invalidCount: invalid}
      }
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(
      nodes.filter(
        node =>
          node.id.includes('-bandwidth-') ||
          node.id === 'membership-point-weights' ||
          node.id === 'membership-weight-sum' ||
          node.id === 'membership-invalid-count'
      ),
      memory
    );
    expect(Array.from(memory.readUint32(count))).toEqual([testCase.count]);
    expect(Array.from(memory.readFloat32(weightSum))).toEqual([testCase.sum]);
    expect(Array.from(memory.readUint32(invalid))).toEqual([testCase.invalid]);
    const actualBandwidth = memory.readFloat32(bandwidth);
    expect(actualBandwidth).toHaveLength(1);
    expect(Number.isFinite(actualBandwidth[0])).toBe(true);
    expect(Math.abs(actualBandwidth[0] / testCase.bandwidth - 1)).toBeLessThan(2e-6);
  } finally {
    passes.mockRestore();
    sums.mockRestore();
    reductions.mockRestore();
    device.destroy();
  }
});
