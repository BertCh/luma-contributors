// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  SEGMENT_PARTIAL_SUMS_WGSL,
  getSegmentTreeSumWGSL
} from '../../analysis-core/kernels/sorted-segment-sums';
import {createWGSLDispatchGate} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  SPARSE_RESISTANCE_STATE_READ_WGSL,
  SPARSE_RESISTANCE_STATE_WGSL,
  SPARSE_RESISTANCE_STATE_WRITE_WGSL,
  SPARSE_RESISTANCE_WORKGROUP_SIZE,
  type SparseResistanceSystem
} from './sparse-resistance-system';

/**
 * Adds a Jacobi-preconditioned conjugate gradient solve of the grounded weighted graph Laplacian
 * with unit current injected at `source` and `sink` held at potential zero.
 *
 * The iteration is an unrolled bounded sequence: the three vertex-wide kernels of each iteration
 * are gated by a GPU-written indirect dispatch that the one-workgroup scalar nodes disarm on
 * convergence, breakdown or budget exhaustion. Dot products are deterministic two-level pairwise
 * trees, so results are reproducible for fixed inputs. Five nodes per iteration.
 *
 * The solve runs only when the system recorded `STATE_REACHABLE != 0`. Returns the voltages and
 * the partial-sum buffer (`3 * partialCount` words) for reuse by a later reduction.
 * @internal
 */
export function addSparseResistanceSolveNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    system: SparseResistanceSystem;
    vertexCount: number;
    source: number;
    sink: number;
    maximumIterations: number;
    tolerance: number;
  }
): {solution: GraphDataView<'float32'>; partials: GraphDataView<'float32'>; partialCount: number} {
  const {id, operation, system, vertexCount, source, sink, maximumIterations, tolerance} = props;
  const {rowOffsets, csrColumns, csrConductances, inverseDiagonal, labels, componentsConverged} =
    system;
  const {state} = system;
  const workgroupSize = SPARSE_RESISTANCE_WORKGROUP_SIZE;
  const partialCount = Math.ceil(vertexCount / workgroupSize);
  const solution = createTransientView(graph, `${id}-solution`, 'float32', vertexCount);
  const residual = createTransientView(graph, `${id}-residual`, 'float32', vertexCount);
  const search = createTransientView(graph, `${id}-search`, 'float32', vertexCount);
  const product = createTransientView(graph, `${id}-product`, 'float32', vertexCount);
  const partials = createTransientView(graph, `${id}-partials`, 'float32', 3 * partialCount);
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation,
    shapes: [[vertexCount, workgroupSize]]
  });
  const stateConstants = SPARSE_RESISTANCE_STATE_WGSL;
  const treeDeclarations = SEGMENT_PARTIAL_SUMS_WGSL;
  const sum = getSegmentTreeSumWGSL('sum');
  const partialCountConstants = {PARTIAL_COUNT: partialCount};

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-initialize`,
      operation,
      invocationCount: vertexCount,
      declarations: 'override SOURCE_VERTEX: u32;',
      constants: {SOURCE_VERTEX: source},
      bindings: [
        {name: 'inverseDiagonal', view: inverseDiagonal, type: 'f32', access: 'read'},
        {name: 'solution', view: solution, type: 'f32', access: 'read_write'},
        {name: 'residual', view: residual, type: 'f32', access: 'read_write'},
        {name: 'search', view: search, type: 'f32', access: 'read_write'}
      ],
      body: `let injected = select(0.0, 1.0, index == SOURCE_VERTEX);
  solution[solutionOffset + index] = 0.0;
  residual[residualOffset + index] = injected;
  search[searchOffset + index] = injected * inverseDiagonal[inverseDiagonalOffset + index];`
    })
  );

  // Ungated: this node writes the gate's dispatch buffer.
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-start`,
      operation,
      invocationCount: 1,
      declarations: `override SOURCE_VERTEX: u32;
override SINK_VERTEX: u32;
${stateConstants}${SPARSE_RESISTANCE_STATE_WRITE_WGSL}`,
      constants: {SOURCE_VERTEX: source, SINK_VERTEX: sink},
      bindings: [
        {name: 'labels', view: labels, type: 'u32', access: 'read'},
        {name: 'componentsConverged', view: componentsConverged, type: 'u32', access: 'read'},
        {name: 'inverseDiagonal', view: inverseDiagonal, type: 'f32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        gate.binding
      ],
      body: `let invalid = state[stateOffset + STATE_INVALID];
  let componentsKnown = componentsConverged[componentsConvergedOffset] != 0u;
  let sameComponent = labels[labelsOffset + SOURCE_VERTEX] == labels[labelsOffset + SINK_VERTEX];
  var reachable = select(select(2u, 0u, componentsKnown), 1u, sameComponent);
  if (invalid != 0u) { reachable = 0u; }
  // r0 . z0 for r0 = e_source.
  let preconditioned = inverseDiagonal[inverseDiagonalOffset + SOURCE_VERTEX];
  state[stateOffset + STATE_REACHABLE] = reachable;
  state[stateOffset + STATE_COMPONENTS] = u32(componentsKnown);
  state[stateOffset + STATE_ITERATIONS] = 0u;
  writeStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED, preconditioned);
  writeStateFloat(STATE_RESIDUAL_DOT_RESIDUAL, 1.0);
  if (reachable != 0u && preconditioned > 0.0) {
    state[stateOffset + STATE_ACTIVE] = 1u;
    ${gate.armWGSL}
  } else {
    state[stateOffset + STATE_ACTIVE] = 0u;
    ${gate.disarmWGSL}
  }`
    })
  );

  const stopResidualSquared = (tolerance / 2) ** 2;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const curvatureId = `${id}-curvature-${iteration}`;
    const updateId = `${id}-update-${iteration}`;
    const searchId = `${id}-search-${iteration}`;
    const getIterationGate = (nodeId: string) => {
      const {condition, extraResources} = gate.getGate(0);
      return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
    };

    graph.add(
      createRecipeKernelNode(graph, {
        id: curvatureId,
        operation,
        invocationCount: vertexCount,
        workgroupSize,
        guardIndex: false,
        ...getIterationGate(curvatureId),
        declarations: `override SINK_VERTEX: u32;
override PARTIAL_COUNT: u32;
${treeDeclarations}`,
        constants: {SINK_VERTEX: sink, ...partialCountConstants},
        bindings: [
          {name: 'rowOffsets', view: rowOffsets, type: 'u32', access: 'read'},
          {name: 'csrColumns', view: csrColumns, type: 'u32', access: 'read'},
          {name: 'csrConductances', view: csrConductances, type: 'f32', access: 'read'},
          {name: 'search', view: search, type: 'f32', access: 'read'},
          {name: 'product', view: product, type: 'f32', access: 'read_write'},
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        // No early return: the tree barriers must stay in uniform control flow.
        body: `var partialSum = 0.0;
  if (index < INVOCATION_COUNT) {
    let first = rowOffsets[rowOffsetsOffset + index];
    let last = rowOffsets[rowOffsetsOffset + index + 1u];
    let center = search[searchOffset + index];
    var flow = 0.0;
    for (var slot = first; slot < last; slot++) {
      let column = csrColumns[csrColumnsOffset + slot];
      flow += csrConductances[csrConductancesOffset + slot] * (center - search[searchOffset + column]);
    }
    // Grounded sink: its row and column leave the system; search[sink] stays 0.
    flow = select(flow, 0.0, index == SINK_VERTEX);
    product[productOffset + index] = flow;
    partialSum = center * flow;
  }
  ${sum}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = partialSums[0];
  }`
      })
    );

    // Ungated one-workgroup node: always reduces (uniform barriers); only lane 0 acts while active.
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-alpha-${iteration}`,
        operation,
        invocationCount: workgroupSize,
        workgroupSize,
        guardIndex: false,
        declarations: `override PARTIAL_COUNT: u32;
${treeDeclarations}${IS_FINITE_VALUE_WGSL}${stateConstants}${SPARSE_RESISTANCE_STATE_READ_WGSL}${SPARSE_RESISTANCE_STATE_WRITE_WGSL}`,
        constants: partialCountConstants,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        body: `var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum += partials[partialsOffset + partial];
  }
  ${sum}
  if (localInvocationIndex == 0u && state[stateOffset + STATE_ACTIVE] != 0u) {
    let curvature = partialSums[0];
    if (!(curvature > 0.0) || !isFiniteValue(curvature)) {
      state[stateOffset + STATE_ACTIVE] = 0u;
      state[stateOffset + STATE_BREAKDOWN] = 1u;
      ${gate.disarmWGSL}
    } else {
      writeStateFloat(STATE_ALPHA, readStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED) / curvature);
    }
  }`
      })
    );

    graph.add(
      createRecipeKernelNode(graph, {
        id: updateId,
        operation,
        invocationCount: vertexCount,
        workgroupSize,
        guardIndex: false,
        ...getIterationGate(updateId),
        declarations: `override PARTIAL_COUNT: u32;
${treeDeclarations}${stateConstants}`,
        constants: partialCountConstants,
        bindings: [
          {name: 'search', view: search, type: 'f32', access: 'read'},
          {name: 'product', view: product, type: 'f32', access: 'read'},
          {name: 'inverseDiagonal', view: inverseDiagonal, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'solution', view: solution, type: 'f32', access: 'read_write'},
          {name: 'residual', view: residual, type: 'f32', access: 'read_write'},
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        body: `let alpha = bitcast<f32>(state[stateOffset + STATE_ALPHA]);
  var preconditioned = 0.0;
  var squared = 0.0;
  var signedResidual = 0.0;
  if (index < INVOCATION_COUNT) {
    solution[solutionOffset + index] = solution[solutionOffset + index] + alpha * search[searchOffset + index];
    let nextResidual = residual[residualOffset + index] - alpha * product[productOffset + index];
    residual[residualOffset + index] = nextResidual;
    squared = nextResidual * nextResidual;
    signedResidual = nextResidual;
    preconditioned = squared * inverseDiagonal[inverseDiagonalOffset + index];
  }
  var partialSum = preconditioned;
  ${sum}
  let preconditionedTotal = partialSums[0];
  // Every lane has read the first total before the second tree rewrites partialSums.
  workgroupBarrier();
  partialSum = squared;
  ${sum}
  let squaredTotal = partialSums[0];
  workgroupBarrier();
  // The sink row is not iterated, so its true residual is minus the signed sum of the others.
  partialSum = signedResidual;
  ${sum}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = preconditionedTotal;
    partials[partialsOffset + PARTIAL_COUNT + workgroupIndex] = squaredTotal;
    partials[partialsOffset + 2u * PARTIAL_COUNT + workgroupIndex] = partialSums[0];
  }`
      })
    );

    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-beta-${iteration}`,
        operation,
        invocationCount: workgroupSize,
        workgroupSize,
        guardIndex: false,
        declarations: `override PARTIAL_COUNT: u32;
override MAXIMUM_ITERATIONS: u32;
override STOP_RESIDUAL_SQUARED: f32;
override TOLERANCE: f32;
${treeDeclarations}${IS_FINITE_VALUE_WGSL}${stateConstants}${SPARSE_RESISTANCE_STATE_READ_WGSL}${SPARSE_RESISTANCE_STATE_WRITE_WGSL}`,
        constants: {
          ...partialCountConstants,
          MAXIMUM_ITERATIONS: maximumIterations,
          STOP_RESIDUAL_SQUARED: stopResidualSquared,
          TOLERANCE: tolerance
        },
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        body: `var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum += partials[partialsOffset + partial];
  }
  ${sum}
  let preconditionedTotal = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum += partials[partialsOffset + PARTIAL_COUNT + partial];
  }
  ${sum}
  let squaredTotal = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum += partials[partialsOffset + 2u * PARTIAL_COUNT + partial];
  }
  ${sum}
  if (localInvocationIndex == 0u && state[stateOffset + STATE_ACTIVE] != 0u) {
    let residualSum = partialSums[0];
    let iterations = state[stateOffset + STATE_ITERATIONS] + 1u;
    let beta = preconditionedTotal / readStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED);
    state[stateOffset + STATE_ITERATIONS] = iterations;
    writeStateFloat(STATE_BETA, beta);
    writeStateFloat(STATE_RESIDUAL_DOT_PRECONDITIONED, preconditionedTotal);
    writeStateFloat(STATE_RESIDUAL_DOT_RESIDUAL, squaredTotal);
    let converged = squaredTotal <= STOP_RESIDUAL_SQUARED && abs(residualSum) <= TOLERANCE / 2.0;
    // An exactly zero residual divides 0 by 0 but has converged, so it is not a breakdown.
    let failed = !converged && !isFiniteValue(beta);
    if (converged || iterations >= MAXIMUM_ITERATIONS || failed) {
      state[stateOffset + STATE_ACTIVE] = 0u;
      state[stateOffset + STATE_BREAKDOWN] = u32(failed);
      ${gate.disarmWGSL}
    }
  }`
      })
    );

    graph.add(
      createRecipeKernelNode(graph, {
        id: searchId,
        operation,
        invocationCount: vertexCount,
        // Explicit: the guarded automatic size would differ from the gate slot.
        workgroupSize,
        ...getIterationGate(searchId),
        declarations: stateConstants,
        bindings: [
          {name: 'residual', view: residual, type: 'f32', access: 'read'},
          {name: 'inverseDiagonal', view: inverseDiagonal, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'search', view: search, type: 'f32', access: 'read_write'}
        ],
        body: `let beta = bitcast<f32>(state[stateOffset + STATE_BETA]);
  search[searchOffset + index] = residual[residualOffset + index] * inverseDiagonal[inverseDiagonalOffset + index] + beta * search[searchOffset + index];`
      })
    );
  }

  return {solution, partials, partialCount};
}
