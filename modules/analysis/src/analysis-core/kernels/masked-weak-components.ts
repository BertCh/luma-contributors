// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from './wgsl-kernel-nodes';

export type MaskedWeakComponentProps = {
  id: string;
  operation: string;
  nodeCount: number;
  /** CSR offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** CSR neighbors; entries `>= nodeCount` (including `0xffffffff`) are ignored. Edges connect both ways. */
  neighbors: GraphDataView<'uint32'>;
  /** One label per node receiving the lowest node index of its weak component once converged. */
  labels: GraphDataView<'uint32'>;
  /** Optional one-row flag set to one when the last iteration changed no label. */
  converged?: GraphDataView<'uint32'>;
  iterations: number;
};

/**
 * Weak components of a masked or transient network CSR by bounded minimum-label propagation.
 * Analysis-domain helper: published `GPUGraphAlgorithmsConnectedComponents` only accepts a
 * `GPUGraphAlgorithmsTopology` rebuilt from source graph columns, which cannot wrap a masked CSR.
 * Convergence means the last iteration changed no label; the default is 32 iterations, matching
 * the published primitive.
 */
export function createMaskedWeakComponentNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: MaskedWeakComponentProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, nodeCount, offsets, neighbors, labels, converged, iterations} = props;
  const changed = createTransientView(graph, `${id}-changed`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-init`,
      operation,
      variant: 'component-init',
      bindings: [{name: 'labels', view: labels, type: 'u32', access: 'read_write'}],
      invocationCount: nodeCount,
      body: 'labels[labelsOffset + index] = index;'
    })
  ];
  for (let iteration = 0; iteration < iterations; iteration++) {
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-${iteration}`,
        operation,
        view: changed,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-step-${iteration}`,
        operation,
        variant: 'component-step',
        bindings: [
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'atomic<u32>', access: 'read_write'},
          {name: 'changed', view: changed, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: nodeCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u; const SLOT_CAPACITY: u32 = ${neighbors.length}u;`,
        // Each edge is treated as undirected: the lowest label is pulled from, and pushed to, both endpoints.
        body: `let first = min(offsets[offsetsOffset + index], SLOT_CAPACITY);
  let last = min(offsets[offsetsOffset + index + 1u], SLOT_CAPACITY);
  let current = atomicLoad(&labels[labelsOffset + index]);
  var best = min(current, atomicLoad(&labels[labelsOffset + current]));
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < NODE_COUNT) {
      best = min(best, atomicLoad(&labels[labelsOffset + neighbor]));
    }
  }
  var didChange = atomicMin(&labels[labelsOffset + index], best) > best;
  didChange = (atomicMin(&labels[labelsOffset + current], best) > best) || didChange;
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < NODE_COUNT) {
      didChange = (atomicMin(&labels[labelsOffset + neighbor], best) > best) || didChange;
    }
  }
  if (didChange) {
    atomicStore(&changed[changedOffset], 1u);
  }`
      })
    );
  }
  if (converged) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-converged`,
        operation,
        variant: 'component-converged',
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'converged', view: converged, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'converged[convergedOffset] = select(1u, 0u, changed[changedOffset] != 0u);'
      })
    );
  }
  return nodes;
}
