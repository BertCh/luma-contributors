// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainBranchingFlowOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  BRANCHING_STATUS_GRID,
  BRANCHING_STATUS_SCENES,
  createBranchingStatusFractions,
  createBranchingStatusProps
} from './branching-flow-status-oracles-fixtures';

it('branching status overwrites poison and resets its unresolved reduction on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const fractions = createInputBuffer(device, createBranchingStatusFractions(true));
  const values = createInputBuffer(device, new Float32Array([1, 1, 1]));
  const validity = createInputBuffer(device, new Uint32Array([1, 1, 1]));
  const result = createOutputBuffer(device, 3);
  const status = createOutputBuffer(device, 3);
  status.write(new Uint32Array([17, 17, 17]));
  const unresolvedCount = createOutputBuffer(device, 1);
  unresolvedCount.write(new Uint32Array([17]));
  graph.add(
    new GPUTerrainBranchingFlow({
      grid: BRANCHING_STATUS_GRID,
      mode: 'transport',
      maximumIterations: 4,
      fractions: importGraphBuffer(graph, 'fractions', fractions, 'float32', 24),
      values: importGraphBuffer(graph, 'values', values, 'float32', 3),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 3),
      output: {
        values: importGraphBuffer(graph, 'result', result, 'float32', 3),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 3),
        unresolvedCount: importGraphBuffer(graph, 'unresolvedCount', unresolvedCount, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const scene of [...BRANCHING_STATUS_SCENES, BRANCHING_STATUS_SCENES[0]]) {
      fractions.write(createBranchingStatusFractions(scene.cycle));
      validity.write(new Uint32Array(scene.validity));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status, 3), scene.name).toEqual(scene.expectedStatus);
      expect(await readUint32(unresolvedCount, 1), scene.name).toEqual(scene.expectedCount);
    }
  } finally {
    compiled.destroy();
    for (const buffer of [fractions, values, validity, result, status, unresolvedCount])
      buffer.destroy();
  }
});

for (const scene of BRANCHING_STATUS_SCENES) {
  it(`GPU branching operation status counts hand-derived unresolved cells: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const result = await runOperationGraph(device, {
      inputs: {
        fractions: {format: 'float32', values: createBranchingStatusFractions(scene.cycle)},
        values: {format: 'float32', values: new Float32Array([1, 1, 1])},
        validity: {format: 'uint32', values: new Uint32Array(scene.validity)}
      },
      operations: [new GPUTerrainBranchingFlowOperation(createBranchingStatusProps())],
      outputs: ['status', 'unresolvedCount']
    });
    expect(Array.from(result['status'])).toEqual(scene.expectedStatus);
    expect(Array.from(result['unresolvedCount'])).toEqual(scene.expectedCount);
  });
}
