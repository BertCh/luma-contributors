// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph, GPUFlagOffsets} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../../src/analysis-core/kernels/sorted-segment-sums';
import {
  createHierarchicalDensityClusterInfoNode as createSplitClusterInfoNode,
  createHierarchicalDensityContributionsNode as createSplitContributionsNode,
  createHierarchicalDensityPointerInitNode as createSplitPointerInitNode
} from '../../../src/gpu-spatial-analysis/spatial-clustering/hdbscan-kernels';
import {getHDBSCANDendrogramNodes} from '../../../src/gpu-spatial-analysis/spatial-clustering/hdbscan-tree';
import {
  createHierarchicalDensityClusterInfoNode,
  createHierarchicalDensityContributionsNode,
  createHierarchicalDensityLabelsNode,
  createHierarchicalDensityPointerInitNode,
  type HierarchicalDensityShape
} from '../../../src/gpu-spatial-analysis/spatial-clustering/hierarchical-density-clustering-kernels';
import {
  getHierarchicalDensityCondensingNodes,
  getHierarchicalDensityDendrogramNodes,
  getHierarchicalDensitySelectionNodes
} from '../../../src/gpu-spatial-analysis/spatial-clustering/hierarchical-density-tree';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {MERGE_TREE_SPLICE_MAXIMUM_ENDPOINT_STEPS} from '../../../src/analysis-core/kernels/parallel-merge-tree';
import {requireTestDevice} from '../../utils/require-test-device';

async function runTree(
  device: Device,
  rows: number,
  sortedEdges: readonly (readonly [number, number, number])[],
  minimumClusterSize: number,
  split: boolean
) {
  const graph = new GPUCommandGraph(device);
  const clusterCapacity = Math.floor((2 * rows) / minimumClusterSize) + 1;
  const shape: HierarchicalDensityShape = {
    rowCount: rows,
    minimumSamples: 3,
    minimumClusterSize,
    clusterCapacity,
    leafCapacity: 2 ** Math.ceil(Math.log2(rows)),
    internalNodeCount: 0,
    levelCount: 0
  };
  const edgeValues = new Uint32Array(rows * 3);
  for (let edge = 0; edge < sortedEdges.length; edge++) {
    const [first, second, weight] = sortedEdges[edge];
    edgeValues[edge * 3] = first;
    edgeValues[edge * 3 + 1] = second;
    edgeValues[edge * 3 + 2] = new Uint32Array(new Float32Array([weight]).buffer)[0];
  }
  const stateBuffer = createInputBuffer(device, new Uint32Array([rows, sortedEdges.length]));
  const edgesBuffer = createInputBuffer(device, edgeValues);
  const orderBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: rows}, (_, row) => row)
  );
  const dendrogramWords = split ? 3 : 5;
  const dendrogramBuffer = createOutputBuffer(device, rows * 2 * dendrogramWords);
  const stabilityBuffer = createOutputBuffer(device, clusterCapacity);
  const labelsBuffer = createOutputBuffer(device, rows);
  const countBuffer = createOutputBuffer(device, 1);
  const maximumWorkBuffer = createOutputBuffer(device, 1);
  const buffers = [
    stateBuffer,
    edgesBuffer,
    orderBuffer,
    dendrogramBuffer,
    stabilityBuffer,
    labelsBuffer,
    countBuffer,
    maximumWorkBuffer
  ];
  const createWords = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  const state = importGraphBuffer(graph, 'state', stateBuffer, 'uint32', 2);
  const edges = importGraphBuffer(graph, 'edges', edgesBuffer, 'uint32', rows * 3);
  const edgeOrder = importGraphBuffer(graph, 'edge-order', orderBuffer, 'uint32', rows);
  const dendrogram = importGraphBuffer(
    graph,
    'dendrogram',
    dendrogramBuffer,
    'uint32',
    rows * 2 * dendrogramWords
  );
  const stability = importGraphBuffer(
    graph,
    'stability',
    stabilityBuffer,
    'float32',
    clusterCapacity
  );
  const labels = importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', rows);
  const selectedCount = importGraphBuffer(graph, 'selected-count', countBuffer, 'uint32', 1);
  const spliceMaximumWork = importGraphBuffer(
    graph,
    'maximum-work',
    maximumWorkBuffer,
    'uint32',
    1
  );
  const treeProps = {id: 'tree', shape, state, edges, edgeOrder, dendrogram, spliceMaximumWork};
  const splitTree = split ? getHDBSCANDendrogramNodes(graph, treeProps) : undefined;
  const nodes = splitTree?.nodes ?? getHierarchicalDensityDendrogramNodes(graph, treeProps);
  const pointers = createWords('pointers', rows * 4);
  const startFlags = createWords('flags', rows * 2);
  const clusterOffsets = createWords('offsets', rows * 2);
  const clusterCount = createWords('cluster-count', 1);
  const clusterInfo = createWords('cluster-info', clusterCapacity * 3);
  const contributionKeys = createWords('contribution-keys', rows * 2);
  const contributions = createTransientView(graph, 'contributions', 'float32', rows * 2);
  const lowClusters = createWords('low-clusters', rows);
  const selection = createWords('selection', clusterCapacity * 4);
  const createPointerNode = split
    ? createSplitPointerInitNode
    : createHierarchicalDensityPointerInitNode;
  const createClusterInfoNode = split
    ? createSplitClusterInfoNode
    : createHierarchicalDensityClusterInfoNode;
  const createContributionsNode = split
    ? createSplitContributionsNode
    : createHierarchicalDensityContributionsNode;
  nodes.push(
    createPointerNode(graph, {
      id: 'pointer-init',
      shape,
      dendrogram,
      dendrogramChildren: splitTree?.children ?? dendrogram,
      pointers,
      startFlags
    }),
    ...getHierarchicalDensityCondensingNodes(graph, {id: 'condensing', rowCount: rows, pointers}),
    ...new GPUFlagOffsets({
      id: 'cluster-offsets',
      flags: startFlags,
      offsets: clusterOffsets,
      count: clusterCount
    }).getCommandNodes(graph),
    createClusterInfoNode(graph, {
      id: 'cluster-info',
      shape,
      dendrogram,
      startFlags,
      clusterOffsets,
      pointers,
      clusterInfo
    }),
    createContributionsNode(graph, {
      id: 'contributions',
      shape,
      dendrogram,
      startFlags,
      clusterOffsets,
      pointers,
      clusterInfo,
      contributionKeys,
      contributions,
      lowClusters
    }),
    ...getSortedSegmentSumNodes(graph, {
      id: 'stability',
      operation: 'HDBSCANTreeTest',
      segmentCount: clusterCapacity,
      segmentKeys: contributionKeys,
      reductions: [{name: 'stability', contributions, output: stability}]
    }),
    ...getHierarchicalDensitySelectionNodes(graph, {
      id: 'selection',
      shape,
      clusterCount,
      stability,
      clusterInfo,
      selection,
      selectedCount
    }),
    createHierarchicalDensityLabelsNode(graph, {
      id: 'labels',
      shape,
      lowClusters,
      selection,
      labels
    })
  );
  for (const node of nodes) graph.add(node);
  const compiled = await graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    const rawDendrogram = await readUint32(dendrogramBuffer);
    return {
      header: Array.from({length: rows * 2}, (_, node) =>
        rawDendrogram.slice(node * dendrogramWords, node * dendrogramWords + 3)
      ),
      stability: await readFloat32(stabilityBuffer),
      labels: await readUint32(labelsBuffer),
      count: (await readUint32(countBuffer))[0],
      maximumWork: (await readUint32(maximumWorkBuffer))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('split tree planes retain packed labels and stabilities, with closed-form two-square stabilities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const edges = [
      [0, 1, 1],
      [0, 2, 1],
      [1, 3, 1],
      [4, 5, 1],
      [4, 6, 1],
      [5, 7, 1],
      [1, 4, 9]
    ] as const;
    const packed = await runTree(device, 8, edges, 3, false);
    const split = await runTree(device, 8, edges, 3, true);
    expect(split.header).toEqual(packed.header);
    expect(split.labels).toEqual(packed.labels);
    expect(split.labels).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
    expect(split.count).toBe(2);
    const expected = [32 / 9, 32 / 9, 8 / 9, 0, 0, 0];
    for (let cluster = 0; cluster < expected.length; cluster++) {
      expect(split.stability[cluster]).toBeCloseTo(expected[cluster], 5);
      expect(split.stability[cluster]).toBe(packed.stability[cluster]);
    }
  } finally {
    device.destroy();
  }
}, 180000);

it('split Euler-tour sizes and root stability match a 257-leaf chain and its harmonic closed form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const rows = 257;
    const edges = Array.from({length: rows - 1}, (_, edge): [number, number, number] => [
      edge,
      edge + 1,
      edge + 1
    ]);
    const packed = await runTree(device, rows, edges, 2, false);
    const split = await runTree(device, rows, edges, 2, true);
    expect(split.header).toEqual(packed.header);
    for (let edge = 0; edge < rows - 1; edge++) expect(split.header[rows + edge][1]).toBe(edge + 2);
    let expectedRootStability = 1;
    for (let weight = 1; weight < rows; weight++) expectedRootStability += 1 / weight;
    expect(split.stability[0]).toBeCloseTo(expectedRootStability, 5);
    expect(split.stability).toEqual(packed.stability);
    expect(split.labels).toEqual(Array.from({length: rows}, () => 0xffffffff));
    expect(split.count).toBe(0);
  } finally {
    device.destroy();
  }
}, 180000);

it('large ranked stars have analytic parent/size/weight planes, stabilities and bounded per-dispatch work', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const rows = 65537;
    for (const increasing of [false, true]) {
      const edges = Array.from({length: rows - 1}, (_, edge): [number, number, number] => [
        0,
        edge + 1,
        increasing ? edge + 1 : 1
      ]);
      const actual = await runTree(device, rows, edges, 2, true);
      let mismatches = 0;
      const none = 0xffffffff;
      const weightBits = new Uint32Array(new Float32Array(edges.map(edge => edge[2])).buffer);
      for (let node = 0; node < rows * 2; node++) {
        let parent = none;
        let size = 0;
        let weight = 0;
        if (node < rows) {
          parent = rows + Math.max(node - 1, 0);
          size = 1;
        } else if (node < rows * 2 - 1) {
          const rank = node - rows;
          parent = rank < rows - 2 ? node + 1 : none;
          size = rank + 2;
          weight = weightBits[rank];
        }
        const header = actual.header[node];
        if (header[0] !== parent || header[1] !== size || header[2] !== weight) mismatches++;
      }
      expect(mismatches).toBe(0);
      let expectedStability = 2;
      if (increasing) for (let weight = 2; weight < rows; weight++) expectedStability += 1 / weight;
      else expectedStability = rows;
      expect(Math.abs(actual.stability[0] - expectedStability)).toBeLessThan(
        2e-5 * expectedStability
      );
      expect(actual.stability.slice(1).every(value => value === 0)).toBe(true);
      expect(actual.count).toBe(0);
      expect(actual.labels.every(label => label === none)).toBe(true);
      expect(actual.maximumWork).toBeGreaterThan(0);
      expect(actual.maximumWork).toBeLessThanOrEqual(2 * MERGE_TREE_SPLICE_MAXIMUM_ENDPOINT_STEPS);
    }
  } finally {
    device.destroy();
  }
}, 900000);
