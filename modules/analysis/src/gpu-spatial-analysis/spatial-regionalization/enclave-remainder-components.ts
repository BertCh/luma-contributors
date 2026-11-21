// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';

function getComponentRounds(rows: number) {
  const logarithm = Math.ceil(Math.log2(Math.max(2, rows)));
  return {rounds: 2 * logarithm + 8, jumps: logarithm + 1};
}

/** Hook/compress component rounds and four fixed passes, with one gate-arm pass. @internal */
export function getEnclaveRemainderComponentNodeCount(rows: number): number {
  const {rounds, jumps} = getComponentRounds(rows);
  return 4 + rounds * (jumps + 3);
}

/** Exact same-label components of unfinished enclave remainders. Complete regions are untouched.
 * The component containing the old seed keeps its ID; if the seed left, the first component does.
 * Other components use their minimum member row. No row can be another unfinished region's seed,
 * so these IDs do not collide. This is domain component finding, not a luma.gl API replacement.
 * @internal
 */
export function createEnclaveRemainderComponentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    startCount: number;
    weights: GPUSpatialWeights;
    rowState: GraphDataView<'uint32'>;
    regionPhase: GraphDataView<'uint32'>;
    claims: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, rowCount, startCount, weights, rowState, regionPhase, claims} = props;
  const count = rowCount * startCount;
  const parents = createTransientView(graph, `${id}-remainder-parents`, 'uint32', count);
  const hooks = createTransientView(graph, `${id}-remainder-hooks`, 'uint32', count);
  const changed = createTransientView(graph, `${id}-remainder-changed`, 'uint32', 1);
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-remainder-dispatch`,
    operation: 'GPUMaxPRegions',
    shapes: [[count]]
  });
  const declarations = `const ROWS: u32 = ${rowCount}u;
const NONE: u32 = 0xffffffffu;
const COMPLETE: u32 = 2u;
const FAILED: u32 = 3u;`;
  const locate = 'let start = index / ROWS; let row = index % ROWS; let base = start * ROWS;';
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-remainder-initialize`,
      operation: 'GPUMaxPRegions',
      invocationCount: count,
      declarations,
      bindings: [
        {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
        {name: 'parents', view: parents, type: 'u32', access: 'read_write'},
        {name: 'hooks', view: hooks, type: 'u32', access: 'read_write'},
        {name: 'claims', view: claims, type: 'u32', access: 'read_write'}
      ],
      body: `parents[parentsOffset + index] = index % ROWS;
  hooks[hooksOffset + index] = index % ROWS;
  // Freeze final enclave labels before any publication changes a row's label.
  rowState[rowStateOffset + 2u * index + 1u] = rowState[rowStateOffset + 2u * index];
  claims[claimsOffset + 4u * index + 1u] = NONE;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-remainder-arm`,
      operation: 'GPUMaxPRegions',
      invocationCount: 1,
      bindings: [{name: 'changed', view: changed, type: 'u32', access: 'read_write'}, gate.binding],
      body: `changed[changedOffset] = 0u; ${gate.armWGSL}`
    })
  ];
  const {rounds, jumps} = getComponentRounds(rowCount);
  for (let round = 0; round < rounds; round++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-remainder-${round}-hook`,
        operation: 'GPUMaxPRegions',
        invocationCount: count,
        ...gate.getGate(0),
        declarations,
        bindings: [
          {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
          {name: 'phase', view: regionPhase, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'parents', view: parents, type: 'u32', access: 'read'},
          {name: 'hooks', view: hooks, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `${locate}
  let label = rowState[rowStateOffset + 2u * index + 1u];
  if (label == NONE || phase[phaseOffset + base + label] == COMPLETE) { return; }
  let own = parents[parentsOffset + index];
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && rowState[rowStateOffset + 2u * (base + neighbor) + 1u] == label) {
      let other = parents[parentsOffset + base + neighbor];
      if (own != other) { atomicMin(&hooks[hooksOffset + base + max(own, other)], min(own, other)); }
    }
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-remainder-${round}-apply`,
        operation: 'GPUMaxPRegions',
        invocationCount: count,
        ...gate.getGate(0),
        declarations,
        bindings: [
          {name: 'parents', view: parents, type: 'u32', access: 'read_write'},
          {name: 'hooks', view: hooks, type: 'u32', access: 'read_write'},
          {name: 'changed', view: changed, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `let row = index % ROWS; let root = hooks[hooksOffset + index];
  if (root < row) { parents[parentsOffset + index] = root;
    hooks[hooksOffset + index] = row; atomicOr(&changed[changedOffset], 1u); }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-remainder-${round}-finish`,
        operation: 'GPUMaxPRegions',
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        body: `if (changed[changedOffset] == 0u) { ${gate.disarmWGSL} } changed[changedOffset] = 0u;`
      })
    );
    for (let jump = 0; jump < jumps; jump++)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-remainder-${round}-jump-${jump}`,
          operation: 'GPUMaxPRegions',
          invocationCount: count,
          ...gate.getGate(0),
          declarations,
          bindings: [{name: 'parents', view: parents, type: 'atomic<u32>', access: 'read_write'}],
          body: `let base = (index / ROWS) * ROWS;
  let parent = atomicLoad(&parents[parentsOffset + index]);
  atomicStore(&parents[parentsOffset + index], atomicLoad(&parents[parentsOffset + base + parent]));`
        })
      );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-remainder-first-component`,
      operation: 'GPUMaxPRegions',
      invocationCount: count,
      declarations,
      bindings: [
        {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
        {name: 'phase', view: regionPhase, type: 'u32', access: 'read'},
        {name: 'parents', view: parents, type: 'u32', access: 'read'},
        {name: 'claims', view: claims, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `${locate}
  let label = rowState[rowStateOffset + 2u * index + 1u];
  if (label != NONE && phase[phaseOffset + base + label] != COMPLETE) {
    atomicMin(&claims[claimsOffset + 4u * (base + label) + 1u], parents[parentsOffset + index]);
  }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-remainder-publish-components`,
      operation: 'GPUMaxPRegions',
      invocationCount: count,
      declarations,
      bindings: [
        {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
        {name: 'phase', view: regionPhase, type: 'atomic<u32>', access: 'read_write'},
        {name: 'parents', view: parents, type: 'u32', access: 'read'},
        {name: 'claims', view: claims, type: 'u32', access: 'read'}
      ],
      body: `${locate}
  let original = rowState[rowStateOffset + 2u * index + 1u];
  if (original == NONE || atomicLoad(&phase[phaseOffset + base + original]) == COMPLETE) { return; }
  let root = parents[parentsOffset + index];
  var preferred = claims[claimsOffset + 4u * (base + original) + 1u];
  if (rowState[rowStateOffset + 2u * (base + original) + 1u] == original) {
    preferred = parents[parentsOffset + base + original];
  }
  let label = select(root, original, root == preferred);
  rowState[rowStateOffset + 2u * index] = label;
  if (root == row) {
    atomicStore(&phase[phaseOffset + base + label], FAILED);
  }`
    })
  );
  return nodes;
}
