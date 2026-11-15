// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {GPUCommandGraph, type GPUCommandGraphComputeNode} from '@luma.gl/graph/gpu-core';
import {describeExpansion} from '../../src/computation-lenses/scenarios/gpu-network/expansion.lens';
import {buildExpansion} from '../../src/computation-lenses/scenarios/gpu-dataframe/expansion.run';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';

for (const kind of [
  'network-structure',
  'network-strongly-connected-components',
  'alternative-routes',
  'network-disruption-recipe',
  'facility-scenario-recipe'
] as const) {
  test(`synthetic ${kind} constructs a valid recorded GPU graph`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const record = await describeExpansion(device, kind);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      expect(record.graphs[0].nodes.length).toBeGreaterThan(0);
      expect(record.graphs[0].nodes.some(node => node.kind === 'compute')).toBe(true);
    } finally {
      device.destroy();
    }
  });
}

class CountingGraph extends GPUCommandGraph<undefined> {
  computeNodeCount = 0;
  override addComputePass(node: Omit<GPUCommandGraphComputeNode<undefined>, 'type'>) {
    this.computeNodeCount++;
    super.addComputePass(node);
  }
}
for (const kind of [
  'cross-variogram',
  'space-time-variogram',
  'marked-point-statistics'
] as const) {
  test(`synthetic ${kind} constructs real contributor passes and capture views`, async () => {
    const device = createNullWebGPUDevice();
    const graph = new CountingGraph(device);
    try {
      const {recording, buffers} = await buildExpansion(device, kind, graph);
      try {
        expect(graph.computeNodeCount).toBeGreaterThan(1);
        expect(recording.specs['pairCounts'].view.length).toBe(
          // Space-time: (2 + 1) spatial x (2 + 1) temporal classes, with a zero-lag class on each axis.
          kind === 'space-time-variogram' ? 9 : 2
        );
        expect(recording.specs['status'].view.length).toBe(
          recording.specs['pairCounts'].view.length
        );
        expect(recording.specs['result'].view.length).toBe(
          kind === 'marked-point-statistics' ? 8 : recording.specs['pairCounts'].view.length
        );
      } finally {
        buffers.forEach(buffer => buffer.destroy());
      }
    } finally {
      device.destroy();
    }
  });
}

test('track quality recipe constructs a network-domain record with real passes', async () => {
  const {describeStructure} = await import(
    '../../src/computation-lenses/scenarios/gpu-network/track-quality-recipe.lens'
  );
  const device = createNullWebGPUDevice();
  try {
    const record = await describeStructure(device);
    expect(validateGPUComputationRecord(record).issues).toEqual([]);
    expect(record.domains).toEqual(['gpu-network']);
    expect(record.graphs[0].nodes.length).toBeGreaterThan(1);
  } finally {
    device.destroy();
  }
});
