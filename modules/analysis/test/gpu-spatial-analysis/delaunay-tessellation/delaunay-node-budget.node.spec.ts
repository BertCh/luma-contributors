// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {GPU_DELAUNAY_MAXIMUM_COMMAND_NODE_COUNT} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let viewCount = 0;

function createContributor(graph: GPUCommandGraph, pointCount: number, groupCount = 0) {
  const createView = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `delaunay-budget-view-${viewCount++}`, format, length);
  const createStatusWord = () => createView('uint32', 1);
  return new GPUDelaunayTessellation({
    positions: createView('float32x2', pointCount),
    groups: groupCount ? {ids: createView('uint32', pointCount), count: groupCount} : undefined,
    output: {
      triangles: createView('uint32x3', 4 * pointCount + 4),
      status: {
        count: createStatusWord(),
        requiredCount: createStatusWord(),
        overflow: createStatusWord(),
        invalidCount: createStatusWord(),
        duplicateCount: createStatusWord(),
        converged: createStatusWord(),
        iterationCount: createStatusWord(),
        iterationLimitReached: createStatusWord()
      },
      groups: groupCount
        ? {
            triangleCounts: createView('uint32', groupCount),
            triangleOffsets: createView('uint32', groupCount),
            status: {converged: createView('uint32', groupCount)}
          }
        : undefined
    }
  });
}

it('GPUDelaunayTessellation bounds its command node count by the fixed allowance', () => {
  // Up to 2,048 sites the steps are short; above them every input unrolls the full 4,096 steps.
  for (const [pointCount, groupCount] of [
    [1, 0],
    [100, 0],
    [100, 3],
    [2048, 0],
    [2049, 0],
    [100000, 0],
    [5000, 4]
  ]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, pointCount, groupCount);
    const nodeCount = contributor.getCommandNodes(graph).length;
    const label = `${pointCount} sites, ${groupCount} groups`;
    expect(contributor.estimatedCommandNodeCount, label).toBeGreaterThanOrEqual(nodeCount);
    expect(contributor.estimatedCommandNodeCount - nodeCount, label).toBeLessThanOrEqual(512);
    expect(contributor.estimatedCommandNodeCount, label).toBeLessThanOrEqual(
      GPU_DELAUNAY_MAXIMUM_COMMAND_NODE_COUNT
    );
  }
});

it('GPUDelaunayTessellation needs a larger budget than the shared one above 2,048 sites', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 4,096 steps of nine nodes exceed the shared 32,768, so large inputs depend on the override.
  expect(createContributor(graph, 2049).estimatedCommandNodeCount).toBeGreaterThan(32768);
  expect(GPU_DELAUNAY_MAXIMUM_COMMAND_NODE_COUNT).toBeLessThan(40000);
});
