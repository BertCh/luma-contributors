// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {
  getDelaunayParallelStepCount,
  getDelaunayStepCount,
  GPU_DELAUNAY_MAXIMUM_COMMAND_NODE_COUNT
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
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

function expectSchedule(
  nodes: readonly {id: string}[],
  contributor: GPUDelaunayTessellation,
  pointCount: number,
  groupCount: number,
  serial: boolean
): void {
  expect(nodes.some(node => node.id.endsWith('-continue-insertion'))).toBe(serial);
  expect(nodes.some(node => node.id.endsWith('-continue-flips'))).toBe(serial);
  expect(
    getDelaunayParallelStepCount(pointCount, groupCount) < getDelaunayStepCount(pointCount)
  ).toBe(serial);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const rounds = nodes.filter(node => /-step-\d+-gate$/.test(node.id)).length;
  expect(rounds).toBe(
    serial ? getDelaunayParallelStepCount(pointCount, groupCount) : getDelaunayStepCount(pointCount)
  );
}

it('bounds both continuation kernels by scratch triangles as well as sites', () => {
  // Scratch = 2*n + max(groups, 1) + 7. At n=2048: 4104 for 0/1 groups,
  // 4105 for 2 groups. Group-heavy n=100, groups=4000 needs 4207 slots.
  for (const [pointCount, groupCount, serial] of [
    [2048, 0, true],
    [2048, 1, true],
    [2048, 2, false],
    [100, 4000, false],
    [2040, 17, true],
    [2040, 18, false]
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = createContributor(graph, pointCount, groupCount);
    const nodes = contributor.getCommandNodes(graph);
    expectSchedule(nodes, contributor, pointCount, groupCount, serial);
  }
});

it('rejects a parallel replacement graph truncated to the initial rounds', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = createContributor(graph, 2040, 18);
  const nodes = contributor.getCommandNodes(graph);
  const truncated = nodes.filter(node => {
    const match = /-step-(\d+)-/.exec(node.id);
    return !match || Number(match[1]) < 99;
  });
  expect(() => expectSchedule(truncated, contributor, 2040, 18, false)).toThrow();
});

it('pins 99 versus 4096 rounds at the 2040-site grouped boundary', () => {
  for (const [groups, rounds] of [
    [17, 99],
    [18, 4096]
  ] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const nodes = createContributor(graph, 2040, groups).getCommandNodes(graph);
    expect(nodes.filter(node => /-step-\d+-gate$/.test(node.id))).toHaveLength(rounds);
  }
});
