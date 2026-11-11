// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createSegmentSumNode,
  getSortedSegmentSumNodes
} from '../../src/analysis-core/kernels/sorted-segment-sums';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

it('getSortedSegmentSumNodes orders sort, offset search, then gather and segment sum per reduction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const segmentKeys = createTransientView(graph, 'keys', 'uint32', 8);
  const contributionsX = createTransientView(graph, 'contributions-x', 'float32', 8);
  const contributionsY = createTransientView(graph, 'contributions-y', 'float32', 8);
  const sumsX = createTransientView(graph, 'sums-x', 'float32', 3);
  const sumsY = createTransientView(graph, 'sums-y', 'float32', 3);

  const nodes = getSortedSegmentSumNodes(graph, {
    id: 'demo',
    operation: 'GPUDemo',
    segmentCount: 3,
    segmentKeys,
    reductions: [
      {name: 'x', contributions: contributionsX, output: sumsX},
      {name: 'y', contributions: contributionsY, output: sumsY}
    ]
  });

  const ids = nodes.map(node => node.id);
  expect(ids.indexOf('demo-sort-prepare')).toBe(0);
  expect(ids.indexOf('demo-segment-offsets')).toBeGreaterThan(ids.indexOf('demo-sort-prepare'));
  const tail = ids.slice(ids.indexOf('demo-segment-offsets') + 1);
  expect(tail).toEqual(['demo-gather-x', 'demo-reduce-x', 'demo-gather-y', 'demo-reduce-y']);
  // Offsets are binary searches over the sorted keys: no per-row counting, no scan.
  expect(ids.some(id => id.includes('scan') || id.includes('total'))).toBe(false);
  expect(new Set(ids).size).toBe(ids.length);
});

it('createSegmentSumNode rejects offsets or an output shorter than the segment count', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segment-sum-limits'});
  const props = {
    id: 'sums',
    operation: 'GPUDemo',
    segmentCount: 4,
    input: createTransientView(graph, 'input', 'float32', 16),
    segmentOffsets: createTransientView(graph, 'offsets', 'uint32', 5),
    output: createTransientView(graph, 'output', 'float32', 4)
  };
  expect(() =>
    createSegmentSumNode(graph, {
      ...props,
      segmentOffsets: createTransientView(graph, 'short-offsets', 'uint32', 4)
    })
  ).toThrow(/^sums segmentCount \+ 1 must be at most 4/);
  expect(() =>
    createSegmentSumNode(graph, {
      ...props,
      output: createTransientView(graph, 'short-output', 'float32', 3)
    })
  ).toThrow(/^sums segmentCount must be at most 3/);
});
