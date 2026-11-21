// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getParallelMergeTreeNodes} from '../../analysis-core/kernels/parallel-merge-tree';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';
import type {GPUTerrainDepressionHierarchyProps} from './gpu-terrain-depression-hierarchy';

/** Contracts same-activation edges, then skips single-child activations. @internal */
export function createTerrainDepressionActivationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    hierarchy: GPUTerrainDepressionHierarchyProps;
    elevation: GraphDataView<'float32'>;
    sortedCells: GraphDataView<'uint32'>;
    tree: GraphDataView<'uint32'>;
    levels: GraphDataView<'float32'>;
    /** Optional internal work-bound diagnostic used by GPU regression tests. */
    spliceMaximumWork?: GraphDataView<'uint32'>;
  }
) {
  const {id, hierarchy, elevation, sortedCells, tree, levels} = props;
  const count = elevation.length;
  const nodeCount = count * 9;
  const operation = 'GPUTerrainDepressionHierarchy';
  const common = `${getTerrainFlowTracingWGSL(hierarchy.grid.width, hierarchy.grid.height)}
${TERRAIN_FINITE_WGSL}
fn activationRank(node: u32) -> u32 {
 if (node < ${count}u) { return node; }
 return (node - ${count}u) / 8u;
}`;
  // Rank is also the leaf ID. Each rank has eight ordered edge slots; unused slots stay isolated.
  // All edges enter at their later endpoint, preserving the exact (height, cell ID) activation order.
  const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', count);
  const builder = getParallelMergeTreeNodes(graph, {
    id: `${id}-activation`,
    operation,
    nodeCount,
    boundedSplicing: true,
    spliceMaximumWork: props.spliceMaximumWork,
    edgeCount: count * 8,
    declarations: common,
    edgeBindings: [
      terrainBinding('order', sortedCells),
      terrainBinding('ranks', ranks),
      terrainBinding('elevation', elevation)
    ],
    edgeBody: `let rank = index / 8u; let cell = order[orderOffset + rank];
if (!isFiniteValue(elevation[elevationOffset + cell])) { return; }
let neighbor = getFlowReceiver(cell, 1u << (index % 8u));
if (neighbor == NO_CELL) { return; }
let neighborRank = ranks[ranksOffset + neighbor];
if (neighborRank >= rank) { return; }
let firstEndpoint = rank; let secondEndpoint = neighborRank; let node = ${count}u + index;`
  });
  const {nodes, createWordView, appendKernel, appendJumpNodes, parents} = builder;
  nodes.unshift(
    createWGSLKernelNode(graph, {
      id: `${id}-activation-ranks`,
      operation,
      invocationCount: count,
      bindings: [
        terrainBinding('order', sortedCells),
        terrainBinding('ranks', ranks, 'read_write')
      ],
      body: 'ranks[ranksOffset + order[orderOffset + index]] = index;'
    })
  );
  const tops = createWordView('tops', nodeCount);
  appendKernel(
    'tops-init',
    nodeCount,
    [terrainBinding('parents', parents), terrainBinding('tops', tops, 'read_write')],
    `let parent = parents[parentsOffset + index];
var next = index;
if (parent != NONE && activationRank(parent) == activationRank(index)) { next = parent; }
tops[topsOffset + index] = next;`,
    common
  );
  appendJumpNodes(
    'tops',
    nodeCount,
    1,
    Math.ceil(Math.log2(nodeCount)) + 1,
    tops,
    `let next = table[tableOffset + index]; let jumped = table[tableOffset + next];
scratch[scratchOffset + index] = jumped;
if (jumped != next) { atomicOr(&pending[pendingOffset], 1u); }`
  );
  const cellParents = createWordView('cell-parents', count);
  const childCounts = createWordView('child-counts', count);
  const firstChildren = createWordView('first-children', count);
  appendKernel(
    'cells-init',
    count,
    [
      terrainBinding('tops', tops),
      terrainBinding('parents', parents),
      terrainBinding('cellParents', cellParents, 'read_write'),
      terrainBinding('childCounts', childCounts, 'read_write'),
      terrainBinding('firstChildren', firstChildren, 'read_write')
    ],
    `let parent = parents[parentsOffset + tops[topsOffset + index]];
var parentRank = NONE;
if (parent != NONE) { parentRank = activationRank(parent); }
cellParents[cellParentsOffset + index] = parentRank;
childCounts[childCountsOffset + index] = 0u; firstChildren[firstChildrenOffset + index] = NONE;`,
    common
  );
  appendKernel(
    'cell-children',
    count,
    [
      terrainBinding('cellParents', cellParents),
      {name: 'childCounts', view: childCounts, type: 'atomic<u32>', access: 'read_write'},
      {name: 'firstChildren', view: firstChildren, type: 'atomic<u32>', access: 'read_write'}
    ],
    `let parent = cellParents[cellParentsOffset + index];
if (parent != NONE) {
 atomicAdd(&childCounts[childCountsOffset + parent], 1u);
 atomicMin(&firstChildren[firstChildrenOffset + parent], index);
}`
  );
  const flags = createWordView('flags', count);
  const offsets = createWordView('offsets', count);
  const links = createWordView('links', count * 2);
  appendKernel(
    'links-init',
    count,
    [
      terrainBinding('order', sortedCells),
      terrainBinding('elevation', elevation),
      terrainBinding('childCounts', childCounts),
      terrainBinding('firstChildren', firstChildren),
      terrainBinding('cellParents', cellParents),
      terrainBinding('flags', flags, 'read_write'),
      terrainBinding('links', links, 'read_write')
    ],
    `let valid = isFiniteValue(elevation[elevationOffset + order[orderOffset + index]]);
let event = valid && childCounts[childCountsOffset + index] != 1u;
flags[flagsOffset + index] = u32(event);
links[linksOffset + index * 2u] = select(firstChildren[firstChildrenOffset + index], index, event || !valid);
links[linksOffset + index * 2u + 1u] = cellParents[cellParentsOffset + index];`,
    common
  );
  // Down links find the last minimum/merge owning a cell; up links find the next such event.
  appendJumpNodes(
    'links',
    count,
    2,
    Math.ceil(Math.log2(count)) + 1,
    links,
    `let owner = table[tableOffset + index * 2u];
let jumpedOwner = table[tableOffset + owner * 2u];
var parent = table[tableOffset + index * 2u + 1u];
if (parent != NONE && flags[flagsOffset + parent] == 0u) { parent = table[tableOffset + parent * 2u + 1u]; }
scratch[scratchOffset + index * 2u] = jumpedOwner;
scratch[scratchOffset + index * 2u + 1u] = parent;
if (jumpedOwner != owner || (parent != NONE && flags[flagsOffset + parent] == 0u)) { atomicOr(&pending[pendingOffset], 1u); }`,
    '',
    [terrainBinding('flags', flags)]
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-activation-scan`,
      input: flags,
      output: offsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  const metadata = createWordView('metadata', count * 2);
  const ancestors = createWordView('ancestors', count);
  appendKernel(
    'metadata-init',
    count,
    [
      terrainBinding('metadata', metadata, 'read_write'),
      terrainBinding('ancestors', ancestors, 'read_write'),
      terrainBinding('links', links),
      terrainBinding('flags', flags)
    ],
    `metadata[metadataOffset + index * 2u] = NONE; metadata[metadataOffset + index * 2u + 1u] = NONE;
ancestors[ancestorsOffset + index] = select(NONE, links[linksOffset + index * 2u + 1u], flags[flagsOffset + index] != 0u);`
  );
  appendKernel(
    'metadata-cells',
    count,
    [
      terrainBinding('order', sortedCells),
      terrainBinding('elevation', elevation),
      terrainBinding('outlets', hierarchy.outlets),
      terrainBinding('links', links),
      {name: 'metadata', view: metadata, type: 'atomic<u32>', access: 'read_write'}
    ],
    `let cell = order[orderOffset + index]; if (!isFiniteValue(elevation[elevationOffset + cell])) { return; }
let owner = links[linksOffset + index * 2u];
atomicMin(&metadata[metadataOffset + owner * 2u], index);
if (outlets[outletsOffset + cell] != 0u) { atomicMin(&metadata[metadataOffset + owner * 2u + 1u], cell); }`,
    common
  );
  appendJumpNodes(
    'metadata',
    count,
    1,
    Math.ceil(Math.log2(count)) + 1,
    ancestors,
    `var parent = table[tableOffset + index];
if (parent != NONE) {
 atomicMin(&metadata[metadataOffset + parent * 2u], atomicLoad(&metadata[metadataOffset + index * 2u]));
 atomicMin(&metadata[metadataOffset + parent * 2u + 1u], atomicLoad(&metadata[metadataOffset + index * 2u + 1u]));
 parent = table[tableOffset + parent];
}
scratch[scratchOffset + index] = parent;
if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`,
    '',
    [{name: 'metadata', view: metadata, type: 'atomic<u32>', access: 'read_write'}]
  );
  appendKernel(
    'record',
    count,
    [
      terrainBinding('order', sortedCells),
      terrainBinding('elevation', elevation),
      terrainBinding('flags', flags),
      terrainBinding('offsets', offsets),
      terrainBinding('links', links),
      terrainBinding('metadata', metadata),
      terrainBinding('tree', tree, 'read_write'),
      terrainBinding('levels', levels, 'read_write')
    ],
    `if (flags[flagsOffset + index] == 0u) { return; }
let destination = offsets[offsetsOffset + index];
let parent = links[linksOffset + index * 2u + 1u];
var parentNode = NONE; var spill = infinity();
if (parent != NONE) { parentNode = offsets[offsetsOffset + parent]; spill = elevation[elevationOffset + order[orderOffset + parent]]; }
tree[treeOffset + destination * 4u] = parentNode;
tree[treeOffset + destination * 4u + 1u] = order[orderOffset + metadata[metadataOffset + index * 2u]];
tree[treeOffset + destination * 4u + 2u] = metadata[metadataOffset + index * 2u + 1u];
tree[treeOffset + destination * 4u + 3u] = 1u;
levels[levelsOffset + destination * 2u] = elevation[elevationOffset + order[orderOffset + index]];
levels[levelsOffset + destination * 2u + 1u] = spill;`,
    common
  );
  // Split ownership and status publishing from record to remain within eight storage bindings.
  appendKernel(
    'ownership',
    count,
    [
      terrainBinding('order', sortedCells),
      terrainBinding('elevation', elevation),
      terrainBinding('links', links),
      terrainBinding('offsets', offsets),
      terrainBinding('ownership', hierarchy.output.cellNodes, 'read_write')
    ],
    `let cell = order[orderOffset + index];
var owner = NONE;
if (isFiniteValue(elevation[elevationOffset + cell])) { owner = offsets[offsetsOffset + links[linksOffset + index * 2u]]; }
ownership[ownershipOffset + cell] = owner;`,
    common
  );
  const {status} = hierarchy.output;
  appendKernel(
    'status',
    1,
    [
      terrainBinding('offsets', offsets),
      terrainBinding('flags', flags),
      terrainBinding('storedCount', status.count, 'read_write'),
      terrainBinding('requiredCount', status.requiredCount, 'read_write'),
      terrainBinding('overflow', status.overflow, 'read_write')
    ],
    `let last = ${count - 1}u;
let required = offsets[offsetsOffset + last] + flags[flagsOffset + last];
storedCount[storedCountOffset] = min(required, ${hierarchy.nodeCapacity}u);
requiredCount[requiredCountOffset] = required;
overflow[overflowOffset] = u32(required > ${hierarchy.nodeCapacity}u);`
  );
  return nodes;
}
