// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

for (const values of [
  [1, 3, 2 ** 27, -(2 ** 27)],
  [2 ** 27, 1, -(2 ** 27), 3],
  [3, -(2 ** 27), 1, 2 ** 27]
]) {
  it(`W5-15 retains the exact unit mean in order ${values}`, () => {
    const result = evaluateMoments(values, [0, 0, 0, 0]);
    // Exact integer sum 1+3+2^27-2^27=4, divided by four.
    expect(result.means[0]).toBe(1);
    expect(result.between).toBe(0);
  });
}

it('W5-16 subtracts compatible split means rather than separately rounded frames', () => {
  // (16777216+2)/2 = 8388609, and (16777216+2+8388609)/3 = 8388609.
  const result = evaluateMoments([16777216, 2, 8388609], [0, 0, 1]);
  expect(result.means).toEqual([8388609, 8388609]);
  expect(result.between).toBe(0);
});

it('W5-16 retains a small nonzero separation and exact constant tables', () => {
  const separated = evaluateMoments([2 ** 24, -(2 ** 24) + 1, -0.5, -0.5], [0, 0, 1, 1]);
  expect(separated.means).toEqual([0.5, -0.5]);
  expect(separated.between).toBe(1);
  const constant = evaluateMoments([1000000.125, 1000000.125, 1000000.125], [0, 0, 0]);
  expect(constant.means[0]).toBe(1000000.125);
  expect(constant.between).toBe(0);
  expect(constant.total).toBe(0);
  // Each region keeps its own origin; an unrelated large constant must not erase a tiny one.
  const distantConstants = evaluateMoments([2e38, 2e38, 1e-35, 1e-35], [0, 0, 1, 1]);
  expect(distantConstants.means).toEqual([Math.fround(2e38), Math.fround(1e-35)]);
});

for (const magnitude of [1, 2 ** 50]) {
  it(`B4 gives identical permuted regional distributions zero between SSD at ${magnitude}`, () => {
    const result = evaluateMoments([0, 0, magnitude, magnitude, 0, 0], [0, 0, 0, 1, 1, 1]);
    expect(result.between).toBe(0);
  });
}

for (const exponent of [-84, -80]) {
  it(`B5 retains a normal cancellation residual at scale 2^${exponent}`, () => {
    const result = evaluateMoments(
      [2 ** exponent, 2 ** (exponent - 26), -(2 ** exponent)],
      [0, 0, 0]
    );
    const expected = Math.fround(2 ** (exponent - 26) / 3);
    expect(Number.isFinite(result.means[0])).toBe(true);
    expect(result.means[0]).toBe(expected);
  });
}

function evaluateMoments(data: number[], labelData: number[]) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const values = createTransientView(graph, 'values', 'float32', data.length);
    const labels = createTransientView(graph, 'labels', 'uint32', data.length);
    const means = createTransientView(graph, 'region-means', 'float32', 2);
    const nodes = new GPURegionPartitionEvaluation({
      id: 'moments',
      values,
      labels,
      labelCapacity: 2,
      output: {summary: createTransientView(graph, 'summary', 'float32', 9), regionMeans: means}
    }).getCommandNodes(graph);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(
        [...kernel.source.matchAll(/@group\(0\)\s*@binding\(\d+\)/g)].length,
        kernel.id
      ).toBeLessThanOrEqual(8);
    }
    const binding = (suffix: string, name: string) =>
      getWGSLKernelNodeRecord(graph, `moments-${suffix}`)!.bindings.find(
        entry => entry.name === name
      )!.view;
    const memory = createCPUGraphMemory();
    memory.write(values, Float32Array.from(data));
    memory.write(labels, Uint32Array.from(labelData));
    const sorted = data
      .map((_, row) => row)
      .sort((first, second) => labelData[first] - labelData[second]);
    memory.write(binding('region-origins', 'sortedRows'), Uint32Array.from(sorted));
    memory.write(
      binding('region-origins', 'sortedKeys'),
      Uint32Array.from(sorted, row => labelData[row])
    );
    const firstSize = labelData.filter(label => label === 0).length;
    memory.write(
      binding('region-origins', 'labelOffsets'),
      Uint32Array.of(0, firstSize, data.length)
    );
    memory.write(
      binding('between-tiles', 'sizes'),
      Uint32Array.of(firstSize, data.length - firstSize)
    );
    memory.write(binding('origin-signs', 'labels'), Uint32Array.from(labelData));
    const selected = nodes.filter(
      node =>
        [
          'region-origins',
          'origin-signs',
          'select-origins',
          'region-centroids',
          'row-tiles',
          'means',
          'between-tiles',
          'deviation-tiles'
        ].some(suffix => node.id === `moments-${suffix}`) ||
        node.id.startsWith('moments-region-sums-')
    );
    executeCommandNodesOnCPU(selected, memory);
    return {
      means: Array.from(memory.readFloat32(means)),
      between: memory.readFloat32(binding('between-tiles', 'labelBetween'))[0],
      total: memory.readFloat32(binding('deviation-tiles', 'deviations'))[0]
    };
  } finally {
    device.destroy();
  }
}
