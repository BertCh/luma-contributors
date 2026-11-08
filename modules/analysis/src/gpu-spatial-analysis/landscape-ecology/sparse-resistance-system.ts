// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addGPUCommandNodes,
  createTransientView,
  GPUCOOToCSR,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode} from '../recipes/expansion-recipe-utils';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createMaskedWeakComponentNodes} from '../../analysis-core/kernels/masked-weak-components';

/** Lanes per workgroup of every wide sparse resistance kernel. @internal */
export const SPARSE_RESISTANCE_WORKGROUP_SIZE = 256;

/**
 * Word indices of the 16-word `state` buffer. Floats are stored as bit patterns.
 * @internal
 */
export const SPARSE_RESISTANCE_STATE_WGSL = /* wgsl */ `
const STATE_ACTIVE: u32 = 0u;
const STATE_ITERATIONS: u32 = 1u;
const STATE_REACHABLE: u32 = 2u;
const STATE_INVALID: u32 = 3u;
const STATE_RESIDUAL_DOT_PRECONDITIONED: u32 = 4u;
const STATE_ALPHA: u32 = 5u;
const STATE_BETA: u32 = 6u;
const STATE_RESIDUAL_DOT_RESIDUAL: u32 = 7u;
const STATE_BREAKDOWN: u32 = 8u;
const STATE_COMPONENTS: u32 = 9u;
const STATE_CONVERGED: u32 = 10u;
`;

/** `readStateFloat(word)` for a kernel that binds `state` as `u32`. @internal */
export const SPARSE_RESISTANCE_STATE_READ_WGSL = /* wgsl */ `
fn readStateFloat(word: u32) -> f32 { return bitcast<f32>(state[stateOffset + word]); }
`;

/** `writeStateFloat(word, value)` for a kernel that binds `state` as read-write `u32`. @internal */
export const SPARSE_RESISTANCE_STATE_WRITE_WGSL = /* wgsl */ `
fn writeStateFloat(word: u32, value: f32) { state[stateOffset + word] = bitcast<u32>(value); }
`;

/** Transient views of the assembled grounded-Laplacian system. @internal */
export type SparseResistanceSystem = {
  /** CSR row offsets, `vertexCount + 1` rows. Invalid half-edges lie beyond the last offset. */
  rowOffsets: GraphDataView<'uint32'>;
  /** CSR column indices, one per half-edge. */
  csrColumns: GraphDataView<'uint32'>;
  /** CSR conductances, one per half-edge; zero for self edges and barriers. */
  csrConductances: GraphDataView<'float32'>;
  /** Jacobi preconditioner `1 / degree`, 0 for isolated vertices. */
  inverseDiagonal: GraphDataView<'float32'>;
  /** Weak component label of each vertex over positive-conductance edges. */
  labels: GraphDataView<'uint32'>;
  /** One word, 1 when label propagation converged. */
  componentsConverged: GraphDataView<'uint32'>;
  /** 16 words, see {@link SPARSE_RESISTANCE_STATE_WGSL}. */
  state: GraphDataView<'uint32'>;
};

/**
 * Assembles the symmetric CSR conductance matrix of an undirected weighted edge list, the Jacobi
 * preconditioner and the component labels. Uses public `GPUSort` and `GPUCOOToCSR`.
 * @internal
 */
export function addSparseResistanceSystemNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    edges: GraphDataView<'uint32x2'>;
    conductances: GraphDataView<'float32'>;
    vertexCount: number;
    connectivityIterations: number;
  }
): SparseResistanceSystem {
  const {id, operation, edges, conductances, vertexCount, connectivityIterations} = props;
  const halfEdgeCount = edges.length * 2;
  const state = createTransientView(graph, `${id}-state`, 'uint32', 16);
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', halfEdgeCount);
  const halfEdges = createTransientView(graph, `${id}-half-edges`, 'uint32', halfEdgeCount);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', halfEdgeCount);
  const sortedHalfEdges = createTransientView(
    graph,
    `${id}-sorted-half-edges`,
    'uint32',
    halfEdgeCount
  );
  const entryColumns = createTransientView(graph, `${id}-entry-columns`, 'uint32', halfEdgeCount);
  const entryConductances = createTransientView(
    graph,
    `${id}-entry-conductances`,
    'float32',
    halfEdgeCount
  );
  const rowOffsets = createTransientView(graph, `${id}-row-offsets`, 'uint32', vertexCount + 1);
  const csrColumns = createTransientView(graph, `${id}-csr-columns`, 'uint32', halfEdgeCount);
  const csrConductances = createTransientView(
    graph,
    `${id}-csr-conductances`,
    'float32',
    halfEdgeCount
  );
  const componentNeighbors = createTransientView(
    graph,
    `${id}-component-neighbors`,
    'uint32',
    halfEdgeCount
  );
  const labels = createTransientView(graph, `${id}-labels`, 'uint32', vertexCount);
  const componentsConverged = createTransientView(graph, `${id}-components-converged`, 'uint32', 1);
  const inverseDiagonal = createTransientView(
    graph,
    `${id}-inverse-diagonal`,
    'float32',
    vertexCount
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-reset`,
      operation,
      invocationCount: 1,
      bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
      body: `for (var word = 0u; word < 16u; word++) { state[stateOffset + word] = 0u; }`
    })
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-expand`,
      operation,
      invocationCount: halfEdgeCount,
      declarations: `override VERTEX_COUNT: u32;
${IS_FINITE_VALUE_WGSL}${SPARSE_RESISTANCE_STATE_WGSL}`,
      constants: {VERTEX_COUNT: vertexCount},
      bindings: [
        {name: 'edges', view: edges, type: 'u32', access: 'read'},
        {name: 'conductances', view: conductances, type: 'f32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'halfEdges', view: halfEdges, type: 'u32', access: 'read_write'},
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
      ],
      body: `let edge = index >> 1u;
  let side = index & 1u;
  let first = edges[edgesOffset + edge * 2u];
  let second = edges[edgesOffset + edge * 2u + 1u];
  let conductance = conductances[conductancesOffset + edge];
  let valid = first < VERTEX_COUNT && second < VERTEX_COUNT && isFiniteValue(conductance) && conductance >= 0.0;
  if (!valid && side == 0u) { atomicAdd(&state[stateOffset + STATE_INVALID], 1u); }
  // Invalid half-edges sort after every vertex row, beyond rowOffsets[VERTEX_COUNT].
  keys[keysOffset + index] = select(VERTEX_COUNT, select(first, second, side == 1u), valid);
  halfEdges[halfEdgesOffset + index] = index;`
    })
  );

  // Keys lie in [0, vertexCount]; the sort is stable, so the CSR order is deterministic.
  addGPUCommandNodes(
    graph,
    new GPUSort({
      id: `${id}-sort`,
      keys,
      values: halfEdges,
      outputKeys: sortedKeys,
      outputValues: sortedHalfEdges,
      keyBits: getSortKeyBits(vertexCount)
    }).getCommandNodes(graph)
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-gather`,
      operation,
      invocationCount: halfEdgeCount,
      declarations: `override VERTEX_COUNT: u32;`,
      constants: {VERTEX_COUNT: vertexCount},
      bindings: [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'sortedHalfEdges', view: sortedHalfEdges, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'u32', access: 'read'},
        {name: 'conductances', view: conductances, type: 'f32', access: 'read'},
        {name: 'entryColumns', view: entryColumns, type: 'u32', access: 'read_write'},
        {name: 'entryConductances', view: entryConductances, type: 'f32', access: 'read_write'}
      ],
      body: `let halfEdge = sortedHalfEdges[sortedHalfEdgesOffset + index];
  let edge = halfEdge >> 1u;
  let side = halfEdge & 1u;
  let first = edges[edgesOffset + edge * 2u];
  let second = edges[edgesOffset + edge * 2u + 1u];
  let conductance = conductances[conductancesOffset + edge];
  // Self edges and barriers contribute nothing; column 0 keeps every read in range.
  let usable = sortedKeys[sortedKeysOffset + index] < VERTEX_COUNT && first != second && conductance > 0.0;
  entryColumns[entryColumnsOffset + index] = select(0u, select(second, first, side == 1u), usable);
  entryConductances[entryConductancesOffset + index] = select(0.0, conductance, usable);`
    })
  );

  addGPUCommandNodes(
    graph,
    new GPUCOOToCSR({
      id: `${id}-csr`,
      rowIndices: sortedKeys,
      columnIndices: entryColumns,
      values: entryConductances,
      rows: vertexCount,
      rowOffsets,
      outputColumnIndices: csrColumns,
      outputValues: csrConductances
    }).getCommandNodes(graph)
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-diagonal`,
      operation,
      invocationCount: vertexCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'rowOffsets', view: rowOffsets, type: 'u32', access: 'read'},
        {name: 'csrColumns', view: csrColumns, type: 'u32', access: 'read'},
        {name: 'csrConductances', view: csrConductances, type: 'f32', access: 'read'},
        {name: 'inverseDiagonal', view: inverseDiagonal, type: 'f32', access: 'read_write'},
        {name: 'componentNeighbors', view: componentNeighbors, type: 'u32', access: 'read_write'}
      ],
      body: `let first = rowOffsets[rowOffsetsOffset + index];
  let last = rowOffsets[rowOffsetsOffset + index + 1u];
  var degree = 0.0;
  for (var slot = first; slot < last; slot++) {
    let conductance = csrConductances[csrConductancesOffset + slot];
    degree += conductance;
    componentNeighbors[componentNeighborsOffset + slot] = select(0xffffffffu, csrColumns[csrColumnsOffset + slot], conductance > 0.0);
  }
  let inverse = 1.0 / degree;
  inverseDiagonal[inverseDiagonalOffset + index] = select(0.0, inverse, degree > 0.0 && isFiniteValue(inverse));`
    })
  );

  addGPUCommandNodes(
    graph,
    createMaskedWeakComponentNodes(graph, {
      id: `${id}-components`,
      operation,
      nodeCount: vertexCount,
      offsets: rowOffsets,
      neighbors: componentNeighbors,
      labels,
      converged: componentsConverged,
      iterations: connectivityIterations
    })
  );

  return {
    rowOffsets,
    csrColumns,
    csrConductances,
    inverseDiagonal,
    labels,
    componentsConverged,
    state
  };
}
