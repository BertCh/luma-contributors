// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getViewBindingRange, type GraphBufferHandle} from '@luma.gl/graph/gpu-core';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord,
  type WGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {setupKShortestPaths} from './k-shortest-paths-test-utils';

it('GPU batch-gate WGSL preserves exact direct geometry and ANDs stale frontier triples across encodings', () => {
  const device = createNullWebGPUDevice();
  // Force 2D bounded dispatch, checking shapes rather than only total workgroup counts.
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupsPerDimension: 16}
  });
  const {graph, contributor} = setupKShortestPaths({
    device,
    nodeCount: 1000,
    queryCount: 1,
    routeCount: 2
  });
  recordWGSLKernelNodes(graph);
  const nodes = contributor.getCommandNodes(graph);
  const storage = new Map<GraphBufferHandle, ArrayBuffer>();
  const getStorage = (buffer: GraphBufferHandle) => {
    if (!storage.has(buffer)) storage.set(buffer, new ArrayBuffer(Math.max(4, buffer.byteLength)));
    return storage.get(buffer)!;
  };
  const gates = nodes.filter(node =>
    ['search-batch-gate', 'search-frontier-gate'].includes(node.workload?.variant ?? '')
  );
  expect(gates.length).toBeGreaterThan(2);
  let checkedTwoDimensions = false;
  const executeGate = (record: WGSLKernelNodeRecord) => {
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => {
        const {offset, size} = getViewBindingRange(binding.view);
        return [index, new Uint32Array(getStorage(binding.view.buffer), offset, size / 4)];
      })
    );
    const interpreter = new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    );
    interpreter.dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
  };
  for (const enabled of [1, 0, 1]) {
    for (const node of gates) {
      const record = getWGSLKernelNodeRecord(graph, node.id)!;
      const activeQueries = record.bindings.find(binding => binding.name === 'activeQueries')!.view;
      new Uint32Array(getStorage(activeQueries.buffer))[activeQueries.byteOffset / 4] = enabled;
      const frontier = record.bindings.find(binding => binding.name === 'frontierDispatch')?.view;
      if (frontier)
        new Uint32Array(getStorage(frontier.buffer), frontier.byteOffset, 3).set([7, 3, 2]);
      executeGate(record);
      if (frontier) {
        // Batch inactivity cannot consume or rewrite the producer's original dispatch.
        expect(
          Array.from(new Uint32Array(getStorage(frontier.buffer), frontier.byteOffset, 3))
        ).toEqual([7, 3, 2]);
        const output = record.bindings.find(binding => binding.name === 'dispatch')!.view;
        expect(
          Array.from(new Uint32Array(getStorage(output.buffer), output.byteOffset, 3))
        ).toEqual(enabled ? [7, 3, 2] : [0, 1, 1]);
      }
    }
    for (const node of nodes.filter(
      candidate => candidate.workload?.operation === 'GPUNetworkReachability'
    )) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      // Frontier kernels use their own live geometry and were checked via their AND gates above.
      if (!record || record.indirect) continue;
      expect(node.condition?.source).toBe('gpu');
      if (node.condition?.source !== 'gpu') throw new Error('missing GPU batch gate');
      const actual = Array.from(
        new Uint32Array(getStorage(node.condition.buffer), node.condition.byteOffset ?? 0, 3)
      );
      if (enabled) expect(actual, node.id).toEqual(record.workgroups);
      else expect(actual[0], node.id).toBe(0);
      if (record.workgroups[1] > 1) checkedTwoDimensions = true;
    }
  }
  expect(checkedTwoDimensions).toBe(true);
  device.destroy();
});
