// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getParallelMergeTreeNodes} from '../../src/analysis-core/kernels/parallel-merge-tree';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

it('resumes ranked-edge splicing in gated rounds with a constant per-endpoint work bound', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {nodes} = getParallelMergeTreeNodes(graph, {
    id: 'bounded-tree',
    operation: 'BoundedMergeTreeTest',
    nodeCount: 2050,
    edgeCount: 2048,
    edgeBindings: [],
    boundedSplicing: true,
    spliceMaximumWork: createTransientView(graph, 'maximum-work', 'uint32', 1),
    edgeBody: 'let firstEndpoint = 0u; let secondEndpoint = 1u; let node = 2u + index;'
  });
  const rounds = nodes.filter(node => /-splice-round-\d+$/.test(node.id));
  expect(rounds).toHaveLength(2);
  for (const node of rounds) {
    expect(node.condition).toBeDefined();
    expect(node.workload?.maximumInvocationCount).toBe(2048);
  }
  for (const [name, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source), name).toEqual([]);
    if (/-splice-round-/.test(name)) {
      expect(source).not.toMatch(/\bloop\s*\{/);
      expect(source).toContain('attempt < 2048u');
    }
  }
});
