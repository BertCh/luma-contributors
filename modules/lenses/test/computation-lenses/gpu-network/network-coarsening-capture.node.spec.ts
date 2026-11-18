// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';
import {networkCoarseningScenario} from '../../../src/computation-lenses/scenarios/gpu-network/network-coarsening.lens';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-network/network-coarsening.example';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';

it('allocates and returns the candidate-loss status used by the capture guard', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const example = await runExample(device, {
      graph,
      roads: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 6),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'weights', 'float32', 4),
        nodePositions: createTransientView(graph, 'positions', 'float32x2', 5)
      },
      zones: createTransientView(graph, 'labels', 'uint32', 5),
      groupCapacity: 4
    });
    expect(example.candidateOverflow).toBeDefined();
    expect(
      example.coarsening.props.output.edges.status.candidateOverflow === example.candidateOverflow
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('refuses a coarsening capture when five live labels exceed four nonempty groups', async () => {
  const scenario = networkCoarseningScenario;
  const values: Record<string, number[]> = {
    zones: [0, 1, 2, 3, 4],
    groupNodeCount: [1, 1, 1, 1],
    groupCentroid: [0, 0, 1, 0, 2, 0, 3, 0],
    nodePositions: [0, 0, 1, 0, 2, 0, 3, 0, 4, 0],
    sourceGroups: [0],
    targetGroups: [1],
    edgeCounts: [1],
    superedgeCount: [1],
    overflow: [0],
    candidateOverflow: [1]
  };
  const final = (name: string): GPUComputationCapture => ({
    id: name,
    name,
    elementType: 'u32',
    values: values[name],
    provenance: 'gpu-readback'
  });
  expect(() => scenario.capture(final, undefined as never, [])).toThrow(
    /excluded.*label|candidate/i
  );
  expect(scenario.retain).toContain('candidateOverflow');
  values['candidateOverflow'] = [0];
  expect(() => scenario.capture(final, undefined as never, [])).not.toThrow();
  values['overflow'] = [1];
  expect(() => scenario.capture(final, undefined as never, [])).toThrow(/superedge capacity/);
});
