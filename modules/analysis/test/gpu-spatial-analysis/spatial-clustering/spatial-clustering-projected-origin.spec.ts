// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CORE-102: at a projected origin the shared point-neighbor grid writer filed points one cell
// away from where the 3x3 reader looks, so a neighbor one epsilon away was missed. Expectations
// are exact: the two points are exactly epsilon (or radius) apart, both of which are inclusive,
// so with minimumSamples 2 each point is core and both share one cluster. The float32 cell
// arithmetic is transcribed in test/analysis-core/point-neighbor-grid-unit-position.node.spec.ts.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUDBSCANParameterValues,
  GPUDBSCAN,
  GPU_DBSCAN_PARAMETER_LENGTH,
  type GPUDBSCANSettings
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createNeighborSearchHarness} from '../neighbor-search/neighbor-search-harness';

const GARBAGE = 0x7f7f7f7f;

async function runTwoPointClustering(
  device: Device,
  positions: Float32Array,
  parameters: GPUDBSCANSettings,
  gridSize: readonly [number, number]
): Promise<{labels: number[]; coreFlags: number[]; clusterCount: number}> {
  const rows = positions.length / 2;
  const positionsBuffer = createInputBuffer(device, positions);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'projected-origin-parameters',
    format: 'float32',
    length: GPU_DBSCAN_PARAMETER_LENGTH,
    values: getGPUDBSCANParameterValues(parameters)
  });
  const labels = createOutputBuffer(device, rows);
  const coreFlags = createOutputBuffer(device, rows);
  const clusterCount = createOutputBuffer(device, 1);
  for (const buffer of [labels, coreFlags, clusterCount]) {
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
  }
  const graph = new GPUCommandGraph(device, {id: 'projected-origin-clustering'});
  graph.add(
    new GPUDBSCAN({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      parameters: parameterBuffer.importToGraph(graph),
      gridSize,
      denseBoxShortcut: false,
      output: {
        labels: importGraphBuffer(graph, 'labels', labels, 'uint32', rows),
        coreFlags: importGraphBuffer(graph, 'core-flags', coreFlags, 'uint32', rows),
        clusterCount: importGraphBuffer(graph, 'cluster-count', clusterCount, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      labels: await readUint32(labels, rows),
      coreFlags: await readUint32(coreFlags, rows),
      clusterCount: (await readUint32(clusterCount, 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of [positionsBuffer, labels, coreFlags, clusterCount]) {
      buffer.destroy();
    }
    parameterBuffer.destroy();
  }
}

const CASES: {
  name: string;
  positions: number[];
  bounds: [number, number, number, number];
  epsilon: number;
  gridSize: [number, number];
}[] = [
  // The review probe: the previous writer filed the points in columns [1, 2], the reader in [0, 1].
  {
    name: 'origin 1e6, 8 columns',
    positions: [1000004, 0.5, 1000008, 0.5],
    bounds: [1000000, 0, 1000028, 1],
    epsilon: 4,
    gridSize: [8, 1]
  },
  {
    name: 'origin subtracted',
    positions: [4, 0.5, 8, 0.5],
    bounds: [0, 0, 28, 1],
    epsilon: 4,
    gridSize: [8, 1]
  },
  {
    name: 'origin 1e6, 16 columns',
    positions: [1000004, 0.5, 1000008, 0.5],
    bounds: [1000000, 0, 1000028, 1],
    epsilon: 4,
    gridSize: [16, 1]
  },
  // The previous writer filed the maximum-bound point in column 17 of 17 active columns (0-16),
  // which no walk reads, so it did not even find itself.
  {
    name: 'maximum bound at origin 8.5e6',
    positions: [8490392, 0.5, 8490393, 0.5],
    bounds: [8490376, 0, 8490393, 1],
    epsilon: Math.fround(1.0175639390945435),
    gridSize: [20, 1]
  }
];

it('GPUDBSCAN finds a neighbor one epsilon away at a projected origin (CORE-102)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of CASES) {
    const result = await runTwoPointClustering(
      device,
      Float32Array.from(testCase.positions),
      {bounds: testCase.bounds, epsilon: testCase.epsilon, minimumSamples: 2},
      testCase.gridSize
    );
    expect(result.coreFlags, testCase.name).toEqual([1, 1]);
    expect(result.labels, testCase.name).toEqual([0, 0]);
    expect(result.clusterCount, testCase.name).toBe(1);
  }
}, 60000);

it('GPUNeighborSearch radius finds a target one radius away at a projected origin (CORE-102)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of CASES) {
    const harness = createNeighborSearchHarness(device, {
      mode: 'radius',
      positions: Float32Array.from(testCase.positions),
      capacity: 4,
      gridSize: testCase.gridSize,
      parameters: {bounds: testCase.bounds, searchRadius: testCase.epsilon}
    });
    try {
      const result = await harness.run();
      // Rows never list themselves: each row lists the other.
      expect(result.offsets, testCase.name).toEqual([0, 1, 2]);
      expect(result.neighbors, testCase.name).toEqual([1, 0]);
      expect(result.overflow, testCase.name).toBe(0);
    } finally {
      harness.destroy();
    }
  }
}, 60000);
