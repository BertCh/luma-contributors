// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOPTICSStatusValues, readOPTICSStatus} from './optics-status-test-utils';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUOPTICS,
  GPU_OPTICS_GRID_MAXIMUM_ROWS,
  type GPUOPTICSProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {computeOPTICSOnCPU} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Isolated pairs on a two-dimensional lattice, with exactly one radius neighbor per row. */
function createPairPositions(rows: number): Float32Array {
  const positions = new Float32Array(rows * 2);
  for (let row = 0; row < rows; row++) {
    const pair = Math.floor(row / 2);
    positions[2 * row] = (pair % 1024) * 4 + (row % 2) * 0.25;
    positions[2 * row + 1] = Math.floor(pair / 1024) * 4;
  }
  return positions;
}

const PAIR_OPTIONS = {
  minimumSamples: 2,
  maximumDistance: 0.5,
  extractionDistance: 0.3
};

async function runPairs(
  positions: Float32Array,
  neighborIndex: 'grid' | 'brute-force',
  orderingMethod: GPUOPTICSProps['orderingMethod'] = 'sequential',
  repeat = false,
  frontierBatchSize?: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rows = positions.length / 2;
  const buffers: Buffer[] = [];
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = rows
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const positionBuffer = createInputBuffer(device, positions);
  buffers.push(positionBuffer);
  const outputs = {
    ordering: output('ordering', 'uint32'),
    coreDistances: output('core', 'float32'),
    reachability: output('reachability', 'float32'),
    predecessors: output('predecessors', 'uint32'),
    labels: output('labels', 'uint32')
  };
  const iterationCount = output('iteration-count', 'uint32', 1);
  const converged = output('converged', 'uint32', 1);
  const invalidCount = output('invalid-count', 'uint32', 1);
  const iterationLimitReached = output('iteration-limit-reached', 'uint32', 1);
  const candidateOverflow = neighborIndex === 'grid' ? output('overflow', 'uint32', 1) : undefined;
  const optics = new GPUOPTICS({
    ...PAIR_OPTIONS,
    neighborIndex,
    orderingMethod,
    frontierBatchSize,
    neighborCapacity: neighborIndex === 'grid' ? rows : undefined,
    positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rows),
    output: {
      ordering: outputs.ordering.view,
      coreDistances: outputs.coreDistances.view,
      reachability: outputs.reachability.view,
      predecessors: outputs.predecessors.view,
      labels: outputs.labels.view,
      ports: {
        status: {
          iterationCount: iterationCount.view,
          converged: converged.view,
          invalidCount: invalidCount.view,
          iterationLimitReached: iterationLimitReached.view,
          candidateOverflow: candidateOverflow?.view
        }
      }
    }
  });
  const nodes = optics.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(optics.estimatedCommandNodeCount);
  if (orderingMethod === 'sequential') expect(optics.estimatedCommandNodeCount).toBeLessThan(32768);
  graph.add(nodes);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    if (repeat) submitGraph(device, compiled, undefined);
    const result = {
      ordering: await readUint32(outputs.ordering.buffer, rows),
      coreDistances: await readFloat32(outputs.coreDistances.buffer, rows),
      reachability: await readFloat32(outputs.reachability.buffer, rows),
      predecessors: await readUint32(outputs.predecessors.buffer, rows),
      labels: await readUint32(outputs.labels.buffer, rows),
      status: await readOPTICSStatus({iterationCount, converged, invalidCount})
    };
    expect(await readUint32(iterationCount.buffer, 1)).toEqual([rows]);
    expect(await readUint32(converged.buffer, 1)).toEqual([1]);
    expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual([0]);
    if (candidateOverflow) expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
    return result;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('batched indexed OPTICS agrees exactly with the exhaustive, parallel and CPU routes', async () => {
  const positions = createPairPositions(258);
  const reference = computeOPTICSOnCPU(positions, PAIR_OPTIONS);
  const expected = {
    ordering: Array.from(reference.ordering),
    coreDistances: Array.from(reference.coreDistances),
    reachability: Array.from(reference.reachability),
    predecessors: Array.from(reference.predecessors),
    labels: Array.from(reference.labels),
    status: getOPTICSStatusValues(reference)
  };
  expect(await runPairs(positions, 'grid', 'sequential', true)).toEqual(expected);
  expect(await runPairs(positions, 'grid', 'sequential', false, 64)).toEqual(expected);
  expect(await runPairs(positions, 'brute-force')).toEqual(expected);
  expect(await runPairs(positions, 'grid', 'parallel')).toEqual(expected);
});

for (const rows of [65538, GPU_OPTICS_GRID_MAXIMUM_ROWS]) {
  it(`indexed OPTICS visits and labels all ${rows} valid rows with bounded candidate storage`, async () => {
    const result = await runPairs(createPairPositions(rows), 'grid');
    expect(result.status).toEqual([rows, 1, 0]);
    // Each pair is a complete two-row component: the lower row is the next root,
    // its partner is its only seed, and the DBSCAN extraction emits one cluster.
    expect(result.ordering.every((row, rank) => row === rank)).toBe(true);
    expect(result.coreDistances.every(distance => distance === 0.25)).toBe(true);
    expect(
      result.reachability.every((distance, row) => distance === (row % 2 ? 0.25 : Infinity))
    ).toBe(true);
    expect(
      result.predecessors.every(
        (predecessor, row) => predecessor === (row % 2 ? row - 1 : 0xffffffff)
      )
    ).toBe(true);
    expect(result.labels.every((label, row) => label === Math.floor(row / 2))).toBe(true);
  }, 600_000);
}
