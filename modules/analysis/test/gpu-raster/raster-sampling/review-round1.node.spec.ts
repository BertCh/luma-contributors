// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {RASTER_CELL_INDEX_WGSL} from '../../../src/raster-analysis/raster-sampling/raster-sampling-utils';
import {GPURasterProfile} from '../../../src/raster-analysis/raster-sampling/gpu-raster-profile';
import {
  getGPURasterSamplingParameterValues,
  getGPURasterProfileParameterValues
} from '../../../src/raster-analysis/raster-sampling/raster-sampling-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('B2 compares original positions around zero against cell edges in their coordinate frame', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32', 3);
    const result = createTransientView(graph, 'result', 'float32', 3);
    const node = createWGSLKernelNode(graph, {
      id: 'boundary',
      operation: 'RasterCellBoundary',
      invocationCount: 3,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'result', view: result, type: 'f32', access: 'read_write'}
      ],
      declarations: RASTER_CELL_INDEX_WGSL,
      body: 'result[resultOffset + index] = getRasterCellIndex(positions[positionsOffset + index], -1.0, 1.0, 1.0);'
    });
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.of(-(2 ** -25), 0, 2 ** -25));
    executeCommandNodesOnCPU([node], memory);
    // [-1,0) owns the negative dyadic point; [0,1) owns zero and the positive point.
    expect(Array.from(memory.readFloat32(result))).toEqual([0, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('S3 rejects a profile sample capacity outside uint32, including status-only requests', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      values: createTransientView(graph, 'raster', 'float32', 1),
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      output: {
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    };
    expect(() => new GPURasterProfile({...props, sampleCapacity: 2 ** 32})).toThrow(
      /sampleCapacity.*4294967295/
    );
    expect(new GPURasterProfile({...props, sampleCapacity: 0xffffffff}).sampleCapacity).toBe(
      0xffffffff
    );
  } finally {
    device.destroy();
  }
});

it('S4 refuses coerced method keys at both packer boundaries', () => {
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  for (const method of [['nearest'], {toString: () => 'nearest'}]) {
    expect(() => getGPURasterSamplingParameterValues({method: method as never}, grid)).toThrow(
      /method.*nearest.*bilinear.*bicubic/
    );
    expect(() =>
      getGPURasterProfileParameterValues({method: method as never, spacing: 1}, grid)
    ).toThrow(/method.*nearest.*bilinear.*bicubic/);
  }
});

import {getCellSeeds} from '../distance-field/distance-field-oracle';
import {getGPUDistanceFieldParameterValues} from '../../../src/raster-analysis/distance-field';
import {sampleRasterOnCPU} from './raster-sampling-oracle';

it('N2 independent edge references select the correct seed and nearest sample cells', () => {
  for (const {step, origin, x, column} of [
    {step: 41, origin: 0, x: 41, column: 1},
    {step: 0.85, origin: 0, x: 17, column: 20},
    {step: 1, origin: -1, x: -(2 ** -25), column: 0},
    {step: 1, origin: -1, x: 0, column: 1},
    {step: 1, origin: -1, x: 2 ** -25, column: 1}
  ]) {
    const grid = makeGPURasterMetadata({
      width: 21,
      height: 1,
      cellSize: [step, 1],
      origin: [origin, 1]
    });
    const settings = getGPUDistanceFieldParameterValues({}, grid);
    const seeds = getCellSeeds({
      width: 21,
      height: 1,
      settings,
      positions: Float32Array.of(x, 0.5)
    });
    expect(Array.from(seeds).flatMap((value, cell) => (value === 0 ? [cell] : []))).toEqual([
      column
    ]);
    const raster = {
      width: 21,
      height: 1,
      values: Float32Array.from({length: 21}, (_, cell) => 10 * (cell + 1))
    };
    expect(
      sampleRasterOnCPU(
        raster,
        getGPURasterSamplingParameterValues({method: 'nearest'}, grid),
        x,
        0.5
      )
    ).toBe(10 * (column + 1));
  }
});
