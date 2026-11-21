// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('bounds preparation and cell input work independently of the DFS budget (review B3)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const)
      for (const positionCount of [3, 4096, 4097, 8193]) {
        const graph = new GPUCommandGraph(device);
        recordWGSLKernelNodes(graph);
        const covering = new GPUS2PolygonCovering({
          id: 'parallel-cover',
          positions: createTransientView(graph, 'positions', 'float32x3', positionCount),
          ringOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
          maximumLevel: 1,
          capacity: 8,
          maximumVisitedCells: 6,
          refinement,
          output: {
            cells: createTransientView(graph, 'cells', 'uint32x2', 8),
            counts: createTransientView(graph, 'counts', 'uint32x2', 1),
            status: createTransientView(graph, 'status', 'uint32', 1),
            visitedCount: createTransientView(graph, 'visits', 'uint32', 1)
          }
        });
        const nodes = covering.getCommandNodes(graph);
        expect(nodes.some(node => node.id === 'parallel-cover-prepare-vertices')).toBe(true);
        expect(nodes.some(node => node.id === 'parallel-cover-subtree-0')).toBe(true);
        expect(nodes.some(node => node.id === 'parallel-cover-publish-prefix-0')).toBe(true);
        expect(nodes.length).toBe(covering.estimatedCommandNodeCount);
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const kernel of sources)
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        for (const node of nodes) {
          const record = getWGSLKernelNodeRecord(graph, node.id);
          if (!record) continue;
          if (record.id.includes('-validate-pairs-')) {
            expect(record.invocationCount).toBeGreaterThan(1);
            expect(record.body).toContain('slot<4096u');
          }
          if (record.body.includes('coverRingContribution('))
            expect(record.source).toContain('vertex-begin<COVER_INPUT_TILE');
          expect(record.body.includes('for(var ring=')).toBe(false);
        }
      }
  } finally {
    device.destroy();
  }
});

// Scalar transcription of the GPU's tree/rank phases, with classification supplied separately.
// Expected prefixes below are hand-counted preorder visits of a complete four-way tree.
function computeTreePrefix(budget: number, capacity: number) {
  type Node = {
    key: bigint;
    code: number;
    parent: number;
    firstChild: number;
    size: number;
    emitted: number;
    rank: number;
    outputRank: number;
  };
  const levels: Node[][] = [];
  levels.push(
    Array.from({length: 6}, (_, face) => ({
      key: BigInt(2 * face + 1) << 60n,
      code: face === 0 ? 4 : face === 1 ? 1 : 0,
      parent: 0,
      firstChild: 0,
      size: 0,
      emitted: 0,
      rank: 0,
      outputRank: 0
    }))
  );
  for (let level = 0; level < 3; level++) {
    const next: Node[] = [];
    levels[level].forEach((node, parent) => {
      node.firstChild = next.length;
      if (node.code !== 4) return;
      const marker = 1n << BigInt(60 - 2 * level);
      const childMarker = marker >> 2n;
      for (let child = 0; child < 4; child++)
        next.push({
          key: node.key - marker + childMarker + BigInt(child) * 2n * childMarker,
          code: level === 2 ? 1 : 4,
          parent,
          firstChild: 0,
          size: 0,
          emitted: 0,
          rank: 0,
          outputRank: 0
        });
    });
    levels.push(next.slice(0, budget));
  }
  for (let level = 3; level >= 0; level--)
    for (const node of levels[level]) {
      node.size = 1;
      node.emitted = Number(node.code === 1);
      if (node.code === 4) {
        if (node.firstChild + 4 > budget) node.size = budget + 1;
        else
          for (const child of levels[level + 1].slice(node.firstChild, node.firstChild + 4)) {
            node.size = Math.min(budget + 1, node.size + child.size);
            node.emitted += child.emitted;
          }
      }
    }
  const output = new Array<bigint>(capacity).fill(0n);
  let required = 0;
  for (let level = 0; level < 4; level++)
    levels[level].forEach((node, row) => {
      const preceding =
        level === 0 ? levels[0].slice(0, row) : levels[level].slice(row - (row % 4), row);
      const parent = level === 0 ? undefined : levels[level - 1][node.parent];
      node.rank = Math.min(
        budget + 1,
        (parent ? parent.rank + 1 : 0) + preceding.reduce((sum, sibling) => sum + sibling.size, 0)
      );
      node.outputRank =
        (parent?.outputRank ?? 0) + preceding.reduce((sum, sibling) => sum + sibling.emitted, 0);
      if (node.rank < budget && node.code === 1) {
        required++;
        if (node.outputRank < capacity) output[node.outputRank] = node.key;
      }
    });
  const total = levels[0].reduce((sum, node) => sum + node.size, 0);
  return {
    output,
    counts: [Math.min(required, capacity), required],
    visited: Math.min(total, budget),
    status: Number(total > budget) * 4 + Number(required > capacity) * 2
  };
}

it('recovers exact DFS prefixes when deeper candidate levels are truncated (review B3)', () => {
  // Face 0 is a complete depth-3 four-way tree (85 visits, 64 leaves); face 1 emits its root;
  // the other four roots drop. Completion therefore takes 90 visits and emits 65 cells.
  // The first visits are root, child, grandchild, four leaves, next grandchild, next leaf.
  for (const [budget, required] of [
    [6, 3],
    [7, 4],
    [8, 4],
    [9, 5]
  ]) {
    const result = computeTreePrefix(budget, 8);
    expect(result.counts).toEqual([required, required]);
    expect(result.visited).toBe(budget);
    expect(result.status).toBe(4);
    expect(result.output).toEqual([
      ...Array.from({length: required}, (_, row) => BigInt(2 * row + 1) << 54n),
      ...new Array(8 - required).fill(0n)
    ]);
  }
  const complete = computeTreePrefix(128, 70);
  expect(complete.counts).toEqual([65, 65]);
  expect(complete.visited).toBe(90);
  expect(complete.status).toBe(0);
  expect(complete.output).toEqual([
    ...Array.from({length: 64}, (_, row) => BigInt(2 * row + 1) << 54n),
    0x3000000000000000n,
    ...new Array(5).fill(0n)
  ]);
  const overflow = computeTreePrefix(128, 3);
  expect(overflow.counts).toEqual([3, 65]);
  expect(overflow.status).toBe(2);
  expect(overflow.output).toEqual([1n << 54n, 3n << 54n, 5n << 54n]);
});
