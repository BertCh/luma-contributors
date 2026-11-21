// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {createLineDensitySeamNodes} from '../../../src/gpu-spatial-analysis/line-density/line-density-seam';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

it('B2 compares represented bandwidth to an exact three-cell threshold on either axis', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = createTransientView(graph, 'parameters', 'float32', 9);
  const resolution = createTransientView(graph, 'resolution', 'uint32', 1);
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    parameters,
    grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
    indexBounds: [0, 0, 1, 1],
    indexGridSize: [1, 1],
    method: 'binned',
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      status: {bandwidthBelowThreeCells: resolution}
    }
  });
  const diagnostic = density
    .getCommandNodes(graph)
    .find(node => node.id.endsWith('-bandwidth-resolution'))!;
  const source = getGPUCommandNodeKernelSources([diagnostic])[0].source;
  // Metal may lower division to reciprocal multiplication. These examples round in opposite directions.
  expect(Math.fround(183 * Math.fround(1 / 61))).toBeLessThan(3);
  expect(Math.fround(Math.fround(44.999996185302734) * Math.fround(1 / 15))).toBe(3);
  expect(source).toContain('isBandwidthBelowThreeCells');
  const memory = createCPUGraphMemory();
  for (const swapped of [false, true]) {
    for (const [cell, bandwidth, expected] of [
      [61, 183, 0],
      [15, 44.999996185302734, 1],
      [15, 45, 0]
    ]) {
      memory.write(
        parameters,
        new Float32Array([0, 0, swapped ? 1 : cell, swapped ? cell : 1, bandwidth, 0, 0, 1, 1])
      );
      executeCommandNodesOnCPU([diagnostic], memory);
      expect(Array.from(memory.readUint32(resolution))).toEqual([expected]);
    }
  }
  device.destroy();
});

it('B4 counts just below, equal and just above a half turn in both directions', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 12);
  const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 7);
  const output = createTransientView(graph, 'count', 'uint32', 1);
  const nodes = createLineDensitySeamNodes(graph, {
    id: 'seam',
    constants: {
      positions,
      lineOffsets,
      columns: 1,
      rows: 1,
      coordinateSystem: 'spherical',
      radius: 1,
      capacity: 1
    },
    output
  });
  const memory = createCPUGraphMemory();
  // Float32 endpoints. Exact real differences: 180-2^-20, 180, 180+2^-20, then reversed.
  const second = [-0.9999990463256836, -1, -1.0000009536743164];
  memory.write(
    positions,
    new Float32Array([
      ...second.flatMap(value => [179, 0, value, 0]),
      ...second.flatMap(value => [value, 0, 179, 0])
    ])
  );
  memory.write(lineOffsets, new Uint32Array([0, 2, 4, 6, 8, 10, 12]));
  executeCommandNodesOnCPU([nodes[0]], memory);
  const resource = nodes[0].resources![2];
  if (!('buffer' in resource)) throw new Error('expected flag output');
  expect(Array.from(memory.readUint32(resource.buffer as GraphDataView<'uint32'>))).toEqual([
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0
  ]);
  device.destroy();
});

it.each([
  'quartic',
  'epanechnikov'
] as const)('B7 auto %s writes a zero resolution diagnostic', kernel => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const resolution = createTransientView(graph, 'resolution', 'uint32', 1);
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(1, 1, [0, 0, 1, 1]),
    indexBounds: [0, 0, 1, 1],
    indexGridSize: [1, 1],
    method: 'auto',
    kernel,
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      status: {bandwidthBelowThreeCells: resolution}
    }
  });
  const diagnostic = density
    .getCommandNodes(graph)
    .filter(node => node.id.endsWith('-bandwidth-resolution'));
  expect(diagnostic).toHaveLength(1);
  const memory = createCPUGraphMemory();
  memory.write(resolution, new Uint32Array([9]));
  executeCommandNodesOnCPU(diagnostic, memory);
  expect(Array.from(memory.readUint32(resolution))).toEqual([0]);
  device.destroy();
});

it('S3 refuses long seam scalars and aliased writable outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    columns: 1,
    rows: 1
  };
  const lengths = createTransientView(graph, 'lengths', 'float32', 1);
  expect(
    () =>
      new GPULineDensity({
        ...props,
        output: {
          lengths,
          status: {
            overflow,
            antimeridianUncertainCount: createTransientView(graph, 'too-long', 'uint32', 2)
          }
        }
      })
  ).toThrow(/exactly one/);
  expect(
    () =>
      new GPULineDensity({
        ...props,
        output: {lengths, status: {overflow, antimeridianUncertainCount: overflow}}
      })
  ).toThrow(/overlap/);
  device.destroy();
});
