// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/gpu-network-reachability';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it.each([
  false,
  true
])('F7 tie predecessor verifies a shared CSR slot and labels (service areas=%s)', serviceAreas => {
  const device = createNullWebGPUDevice();
  const context = createOracleContext(device);
  const {graph, input, output} = context;
  recordWGSLKernelNodes(graph);
  const offsets = input('offsets', 'uint32', [0, 2, 2, 0, 2]);
  const neighbors = input('neighbors', 'uint32', [1, 1]);
  const weights = input('weights', 'float32', [0, 1]);
  const costs = output('costs', 'float32', 4);
  const predecessors = output('parents', 'uint32', 4);
  const assignments = output('assignments', 'uint32', 4);
  const starts = input('starts', 'uint32', [0, 3]);
  const startCosts = input('start-costs', 'float32', [19, 2]);
  try {
    const nodes = serviceAreas
      ? new GPUNetworkServiceAreas({
          id: 'overlap',
          topology: {offsets, neighbors, weights},
          facilities: starts,
          facilityCosts: startCosts,
          maximumIterations: 4,
          maximumTieIterations: 4,
          output: {costs, predecessors, assignments}
        }).getCommandNodes(graph)
      : new GPUNetworkReachability({
          id: 'overlap',
          topology: {offsets, neighbors, weights},
          startNodes: starts,
          startCosts,
          maximumIterations: 4,
          maximumTieIterations: 4,
          output: {costs, predecessors}
        }).getCommandNodes(graph);
    const storage = new Map<GraphDataView, Uint32Array | Float32Array>([
      [offsets, Uint32Array.of(0, 2, 2, 0, 2)],
      [neighbors, Uint32Array.of(1, 1)],
      [weights, Float32Array.of(0, 1)],
      [costs, Float32Array.of(serviceAreas ? 2 : 19, 2, Infinity, 2)],
      [assignments, Uint32Array.of(0, 1, 0xffffffff, 1)],
      [predecessors, new Uint32Array(4).fill(0xffffffff)]
    ]);
    const execute = (id: string) => {
      const record = getWGSLKernelNodeRecord(graph, id)!;
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          let values = storage.get(binding.view);
          if (!values) {
            values = new Uint32Array(binding.view.length);
            storage.set(binding.view, values);
          }
          return [index, values];
        })
      );
      new WgslExec(
        new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
      ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
    };
    // Final costs certify only source 3: 2+0=2. Source 0 has cost 19 and cannot
    // parent node 1 at cost 2, even though its CSR row shares the same edge slot.
    // In the service-area control costs tie, but source 0 has a different facility label.
    execute('overlap-tie-initialize');
    const levels = getWGSLKernelNodeRecord(graph, 'overlap-tie-initialize')!.bindings.find(
      binding => binding.name === 'levels'
    )!;
    (storage.get(levels.view) as Uint32Array).set([0, 1, 0xffffffff, 0]);
    expect(nodes.some(node => node.id === 'overlap-tie-predecessors')).toBe(true);
    execute('overlap-tie-predecessors');
    expect(storage.get(predecessors)?.[1]).toBe(3);
  } finally {
    context.destroy();
    device.destroy();
  }
});
