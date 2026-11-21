// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {createIslandMoran, importPublishedWeights} from './weights-loss-r19-fixtures';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('review B1 constructs the island-to-Moran GPU fixture before submission', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    const weights = rig.uploadWeights({
      offsets: [0, 1, 2, 3],
      neighbors: [1, 0, 1],
      weights: [1, 1, 1],
      distances: []
    });
    expect(() =>
      createIslandMoran(
        rig,
        weights,
        rig.output('float32', 3).view,
        rig.output('uint32', 3).view
      ).getCommandNodes(rig.graph)
    ).not.toThrow();
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('review S1 public panel regression forwards candidate status to the final code-3 suppression', () => {
  const device = createNullWebGPUDevice();
  const rig = new WeightsRig(device);
  try {
    const loss = rig.input(Uint32Array.of(1), 'uint32', 1);
    const weights = {
      ...rig.uploadWeights({
        offsets: [0, 1, 2, 3, 4],
        neighbors: [1, 2, 3, 0],
        weights: [1, 1, 1, 1],
        distances: []
      }),
      status: {candidateOverflow: loss}
    };
    const table = rig.output('float32', 12),
      summary = rig.output('float32', 16),
      code = rig.output('uint32', 1);
    const nodes = new GPUSpatialPanelRegression({
      weights,
      periodCount: 3,
      predictorCount: 1,
      predictors: rig.input(new Float32Array(12), 'float32', 12),
      response: rig.input(new Float32Array(12), 'float32', 12),
      output: {table: table.view, summary: summary.view, status: {code: code.view}}
    }).getCommandNodes(rig.graph);
    const finalGate = nodes.filter(
      node =>
        node.id.includes('-entity-weights-loss') ||
        node.id.includes('-pooled-lag-weights-loss') ||
        node.id.includes('-pooled-lag-suppress-incomplete')
    );
    expect(finalGate.length).toBeGreaterThan(0);
    const memory = createCPUGraphMemory();
    memory.write(loss, Uint32Array.of(1));
    memory.write(code.view, Uint32Array.of(0));
    executeCommandNodesOnCPU(finalGate, memory);
    expect(Array.from(memory.readUint32(code.view))).toEqual([3]);
    expect(Array.from(memory.readFloat32(table.view)).every(Number.isNaN)).toBe(true);
  } finally {
    rig.destroy();
    device.destroy();
  }
});

it('review B7 reimports the published CSR/status buffers rather than replacing candidate status', () => {
  const device = createNullWebGPUDevice();
  const producer = new WeightsRig(device),
    consumer = new WeightsRig(device);
  try {
    const published = {
      ...producer.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [1, 1],
        distances: []
      }),
      status: {candidateOverflow: producer.input(Uint32Array.of(1), 'uint32', 1)}
    };
    const imported = importPublishedWeights(consumer.graph, published);
    expect(imported.offsets.buffer.defaultBuffer).toBe(published.offsets.buffer.defaultBuffer);
    expect(imported.neighbors.buffer.defaultBuffer).toBe(published.neighbors.buffer.defaultBuffer);
    expect(imported.weights.buffer.defaultBuffer).toBe(published.weights.buffer.defaultBuffer);
    expect(imported.status?.candidateOverflow?.buffer.defaultBuffer).toBe(
      published.status.candidateOverflow.buffer.defaultBuffer
    );
    const absent = importPublishedWeights(new WeightsRig(device).graph, {
      ...published,
      status: undefined
    });
    expect(absent.status).toBeUndefined();
  } finally {
    producer.destroy();
    consumer.destroy();
    device.destroy();
  }
});
