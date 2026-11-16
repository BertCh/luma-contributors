// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getParallelMergeTreeNodes} from '../../analysis-core/kernels/parallel-merge-tree';
import type {HierarchicalDensityShape} from './hierarchical-density-clustering-kernels';

const OPERATION = 'GPUHierarchicalDensityClustering';
type WordView = GraphDataView<'uint32'>;

function readBinding(name: string, view: WordView): WGSLKernelBinding {
  return {name, view, type: 'u32', access: 'read'};
}
function writeBinding(name: string, view: WordView, atomic = false): WGSLKernelBinding {
  return {name, view, type: atomic ? 'atomic<u32>' : 'u32', access: 'read_write'};
}

/**
 * Exact ranked-edge reconstruction and Euler-tour sizes, using separate parent/size/weight,
 * children, tour-successor and tour-total planes. Each tour plane is four words per point,
 * rather than the eight-word combined record (and identical scratch) of the packed route.
 * The merge order and integer subtree sizes are unchanged. @internal
 */
export function getHDBSCANDendrogramNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    state: WordView;
    edges: WordView;
    edgeOrder: WordView;
    dendrogram: WordView;
    /** Test diagnostic of the bounded splice work per invocation. */
    spliceMaximumWork?: WordView;
  }
) {
  const rows = props.shape.rowCount;
  const nodeCount = rows * 2;
  const {nodes, createWordView, appendKernel, parents, children} = getParallelMergeTreeNodes(
    graph,
    {
      id: props.id,
      operation: OPERATION,
      nodeCount,
      edgeCount: rows,
      boundedSplicing: true,
      spliceMaximumWork: props.spliceMaximumWork,
      initializeBindings: [writeBinding('dendrogram', props.dendrogram)],
      initializeBody: `let base = dendrogramOffset + index * 3u;
  dendrogram[base] = NONE;
  dendrogram[base + 1u] = 0u;
  dendrogram[base + 2u] = 0u;`,
      edgeBindings: [
        readBinding('state', props.state),
        readBinding('edges', props.edges),
        readBinding('edgeOrder', props.edgeOrder),
        writeBinding('dendrogram', props.dendrogram)
      ],
      edgeBody: `if (index >= min(state[stateOffset + 1u], ROW_COUNT - 1u)) { return; }
  let edge = edgesOffset + edgeOrder[edgeOrderOffset + index] * 3u;
  let node = ROW_COUNT + index;
  dendrogram[dendrogramOffset + node * 3u + 2u] = edges[edge + 2u];
  let firstEndpoint = edges[edge];
  let secondEndpoint = edges[edge + 1u];`,
      declarations: `const ROW_COUNT: u32 = ${rows}u;`
    }
  );
  const tourLength = nodeCount * 2;
  const successors = createWordView('tour-successors', tourLength);
  const totals = createWordView('tour-totals', tourLength);
  const successorScratch = createWordView('tour-successor-scratch', tourLength);
  const totalScratch = createWordView('tour-total-scratch', tourLength);
  const pending = createWordView('tour-pending', 1);
  appendKernel(
    'tour-init',
    tourLength,
    [
      readBinding('parents', parents),
      readBinding('children', children),
      writeBinding('successors', successors),
      writeBinding('totals', totals)
    ],
    `let node = index / 2u;
  let first = children[childrenOffset + node * 2u];
  var successor = NONE;
  var contribution = 0u;
  if (index % 2u == 0u) {
    successor = node * 2u + 1u;
    if (first != NONE) { successor = first * 2u; }
    else if (node < ROW_COUNT) { contribution = 1u; }
  } else {
    let parent = parents[parentsOffset + node];
    if (parent != NONE) {
      successor = parent * 2u + 1u;
      if (children[childrenOffset + parent * 2u] == node) {
        successor = children[childrenOffset + parent * 2u + 1u] * 2u;
      }
    }
  }
  successors[successorsOffset + index] = successor;
  totals[totalsOffset + index] = contribution;`,
    `const ROW_COUNT: u32 = ${rows}u;`
  );
  const dispatch = createWGSLDispatchGate(graph, {
    id: `${props.id}-tour-dispatch`,
    operation: OPERATION,
    shapes: [[tourLength]]
  });
  appendKernel('tour-seed', 1, [writeBinding('pending', pending)], 'pending[pendingOffset] = 1u;');
  for (let round = 0; round < Math.ceil(Math.log2(tourLength)); round++) {
    appendKernel(
      `tour-gate-${round}`,
      1,
      [writeBinding('pending', pending), dispatch.binding],
      `if (pending[pendingOffset] != 0u) { ${dispatch.armWGSL} }
  else { ${dispatch.disarmWGSL} }
  pending[pendingOffset] = 0u;`
    );
    appendKernel(
      `tour-jump-${round}`,
      tourLength,
      [
        readBinding('successors', successors),
        readBinding('totals', totals),
        writeBinding('successorScratch', successorScratch),
        writeBinding('totalScratch', totalScratch),
        writeBinding('pending', pending, true)
      ],
      `var successor = successors[successorsOffset + index];
  var total = totals[totalsOffset + index];
  if (successor != NONE) {
    total += totals[totalsOffset + successor];
    successor = successors[successorsOffset + successor];
  }
  successorScratch[successorScratchOffset + index] = successor;
  totalScratch[totalScratchOffset + index] = total;
  if (successor != NONE) { atomicOr(&pending[pendingOffset], 1u); }`,
      '',
      dispatch.getGate(0)
    );
    appendKernel(
      `tour-copy-${round}`,
      tourLength,
      [
        readBinding('successorScratch', successorScratch),
        readBinding('totalScratch', totalScratch),
        writeBinding('successors', successors),
        writeBinding('totals', totals)
      ],
      `successors[successorsOffset + index] = successorScratch[successorScratchOffset + index];
  totals[totalsOffset + index] = totalScratch[totalScratchOffset + index];`,
      '',
      dispatch.getGate(0)
    );
  }
  appendKernel(
    'finish',
    nodeCount,
    [
      readBinding('parents', parents),
      readBinding('totals', totals),
      writeBinding('dendrogram', props.dendrogram)
    ],
    `let base = dendrogramOffset + index * 3u;
  dendrogram[base] = parents[parentsOffset + index];
  dendrogram[base + 1u] = totals[totalsOffset + index * 2u] - totals[totalsOffset + index * 2u + 1u];`
  );
  return {nodes, children};
}
