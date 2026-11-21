// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getDelaunayParallelStepCount,
  getDelaunayStepCount,
  MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT,
  MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT,
  MAXIMUM_STEP_COUNT
} from './delaunay-work-limits';

const OPERATION = 'GPUDelaunayTessellation';

/** @internal Bounded fallback kernels share the parallel route's topology and predicate text. */
export function getDelaunayContinuationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  context: {
    id: string;
    pointCount: number;
    groupCount: number;
    scratchTriangleCapacity: number;
    positionsBinding: WGSLKernelBinding;
    binding: (
      name: string,
      view: GraphDataView<'uint32'>,
      access?: 'read' | 'read_write',
      type?: 'u32' | 'atomic<u32>'
    ) => WGSLKernelBinding;
    triangleVertices: GraphDataView<'uint32'>;
    triangleAdjacency: GraphDataView<'uint32'>;
    replacement: GraphDataView<'uint32'>;
    children: GraphDataView<'uint32'>;
    votes: GraphDataView<'uint32'>;
    pointLocations: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    orientationDeclarations: string;
    flipDeclarations: string;
  },
  shaders: {
    TOPOLOGY_WRITE_WGSL: string;
    INSERTION_PRIORITY_WGSL: string;
    INSERTION_VOTE_BODY_WGSL: string;
    INSERTION_SPLIT_BODY_WGSL: string;
    REPAIR_ADJACENCY_BODY_WGSL: string;
    RELOCATE_BODY_WGSL: string;
    FLIP_EVALUATE_BODY_WGSL: string;
    FLIP_APPLY_BODY_WGSL: string;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    pointCount,
    positionsBinding,
    binding,
    triangleVertices,
    triangleAdjacency,
    replacement,
    children,
    votes,
    pointLocations,
    state
  } = context;
  if (
    pointCount > MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT ||
    context.scratchTriangleCapacity > MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT
  )
    return [];
  const {
    TOPOLOGY_WRITE_WGSL,
    INSERTION_PRIORITY_WGSL,
    INSERTION_VOTE_BODY_WGSL,
    INSERTION_SPLIT_BODY_WGSL,
    REPAIR_ADJACENCY_BODY_WGSL,
    RELOCATE_BODY_WGSL,
    FLIP_EVALUATE_BODY_WGSL,
    FLIP_APPLY_BODY_WGSL
  } = shaders;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const stepCount = getDelaunayParallelStepCount(pointCount, context.groupCount);
  if (stepCount < getDelaunayStepCount(pointCount)) {
    // Finish rare long insertion chains first. With no uninserted sites, the flip loop needs
    // no relocation; keeping these kernels separate also keeps one exact predicate call site.
    // Both kernels run on the GPU with one invocation, using the existing storage topology.
    const iterationLimit = `const ITERATION_LIMIT: u32 = ${Math.min(getDelaunayStepCount(pointCount), MAXIMUM_STEP_COUNT)}u;`;
    const readAtomicState = (body: string) =>
      body.replace(/state\[stateOffset(?: \+ \d+u)?\]/g, value => `atomicLoad(&${value})`);
    const serialVotes = (body: string) =>
      body.replace(/atomicMin\(&votes\[([^\]]+)\], ([^)]+)\)/g, 'votes[$1] = min(votes[$1], $2)');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-continue-insertion`,
        operation: OPERATION,
        variant: 'serial-insertion-continuation',
        bindings: [
          positionsBinding,
          binding('triangleVertices', triangleVertices, 'read_write'),
          binding('triangleAdjacency', triangleAdjacency, 'read_write'),
          binding('replacement', replacement, 'read_write'),
          binding('children', children, 'read_write'),
          binding('votes', votes, 'read_write'),
          binding('pointLocations', pointLocations, 'read_write'),
          binding('state', state, 'read_write', 'atomic<u32>')
        ],
        // MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT = 2048 sites; MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT = 4104 triangles.
        // contract-allow: single-invocation bounded by those route checks and MAXIMUM_STEP_COUNT = 4096 rounds.
        invocationCount: 1,
        workgroupSize: 32,
        declarations: `${context.orientationDeclarations}
${iterationLimit}
const PHASE_OFFSET: u32 = 0u;
${TOPOLOGY_WRITE_WGSL}
${INSERTION_PRIORITY_WGSL}
fn voteForInsertion(index: u32) { ${serialVotes(INSERTION_VOTE_BODY_WGSL)} }
fn splitInsertion(index: u32) { ${INSERTION_SPLIT_BODY_WGSL} }
fn repairInsertion(index: u32) { ${readAtomicState(REPAIR_ADJACENCY_BODY_WGSL)} }
fn relocateInsertion(index: u32) { ${RELOCATE_BODY_WGSL} }`,
        body: `loop {
    if (atomicLoad(&state[stateOffset + 1u]) != 0u || atomicLoad(&state[stateOffset + 2u]) == 0u ||
        atomicLoad(&state[stateOffset + 6u]) >= ITERATION_LIMIT) { break; }
    atomicAdd(&state[stateOffset + 3u], 2u);
    atomicAdd(&state[stateOffset + 6u], 1u);
    // Force legalization even if the last parallel round found no illegal edge.
    atomicStore(&state[stateOffset + 4u], 1u);
    atomicStore(&state[stateOffset + 5u], 0u);
    for (var site = 0u; site < POINT_COUNT; site++) { voteForInsertion(site); }
    for (var site = 0u; site < POINT_COUNT; site++) { splitInsertion(site); }
    for (var triangle = 0u; triangle < atomicLoad(&state[stateOffset]); triangle++) { repairInsertion(triangle); }
    for (var site = 0u; site < POINT_COUNT; site++) { relocateInsertion(site); }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-continue-flips`,
        operation: OPERATION,
        variant: 'serial-flip-continuation',
        bindings: [
          positionsBinding,
          binding('triangleVertices', triangleVertices, 'read_write'),
          binding('triangleAdjacency', triangleAdjacency, 'read_write'),
          binding('replacement', replacement, 'read_write'),
          binding('children', children, 'read_write'),
          binding('votes', votes, 'read_write'),
          binding('state', state, 'read_write')
        ],
        // MAXIMUM_DELAUNAY_SERIAL_SITE_COUNT = 2048 sites; MAXIMUM_DELAUNAY_SERIAL_SCRATCH_TRIANGLE_COUNT = 4104 triangles.
        // contract-allow: single-invocation bounded by those route checks and MAXIMUM_STEP_COUNT = 4096 rounds.
        invocationCount: 1,
        workgroupSize: 32,
        declarations: `${context.flipDeclarations}
${iterationLimit}
const PHASE_OFFSET: u32 = 1u;
${TOPOLOGY_WRITE_WGSL}
fn evaluateFlip(index: u32) {
${serialVotes(FLIP_EVALUATE_BODY_WGSL)
  .replace(/atomicLoad\(&([^)]*)\)/g, '$1')
  .replace(
    'atomicAdd(&state[stateOffset + 4u], 1u)',
    'state[stateOffset + 4u] = state[stateOffset + 4u] + 1u'
  )}
}
fn applyFlip(index: u32) { ${FLIP_APPLY_BODY_WGSL} }
fn repairFlip(index: u32) { ${REPAIR_ADJACENCY_BODY_WGSL} }`,
        body: `loop {
    if (state[stateOffset + 1u] != 0u || state[stateOffset + 2u] != 0u ||
        state[stateOffset + 4u] == 0u || state[stateOffset + 6u] >= ITERATION_LIMIT) { break; }
    state[stateOffset + 3u] = state[stateOffset + 3u] + 2u;
    state[stateOffset + 6u] = state[stateOffset + 6u] + 1u;
    state[stateOffset + 4u] = 0u;
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { evaluateFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { applyFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { repairFlip(triangle); }
  }
  state[stateOffset + 7u] = select(0u, 1u, state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);`
      })
    );
  }
  return nodes;
}
