// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createRecipeKernelNode,
  getRecipeView,
  validateRecipeBoundary
} from '../recipes/expansion-recipe-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  SEGMENT_PARTIAL_SUMS_WGSL,
  getSegmentTreeSumWGSL
} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  SPARSE_RESISTANCE_STATE_WGSL,
  SPARSE_RESISTANCE_WORKGROUP_SIZE,
  addSparseResistanceSystemNodes
} from './sparse-resistance-system';
import {addSparseResistanceSolveNodes} from './sparse-resistance-solver';

/** Largest vertex count: 2^20 keeps the vertex-wide reduction trees and label arrays within one binding. */
export const SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT = 1048576;
/** Largest edge count: 2^21 edges give 2^22 half-edge entries, within one 128 MiB binding of uint32 pairs. */
export const SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT = 2097152;
const MAXIMUM_VERTEX_COUNT = SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT;
const MAXIMUM_EDGE_COUNT = SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT;
/** Largest PCG iteration count: the iterations are unrolled, five command nodes each (20,480 at the limit). */
export const SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS = 4096;
/** Largest label-propagation round count: rounds are unrolled, two command nodes each. */
export const SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_CONNECTIVITY_ITERATIONS = 1024;
const MAXIMUM_ITERATIONS = SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS;
const MAXIMUM_CONNECTIVITY_ITERATIONS =
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_CONNECTIVITY_ITERATIONS;
const OPERATION = 'addSparseResistanceConnectivityRecipe';

/** Command nodes of one unrolled PCG iteration: curvature, alpha, update, beta, search. */
const COMMAND_NODES_PER_ITERATION = 5;
/** Command nodes of one unrolled label-propagation round: clear and step. */
const COMMAND_NODES_PER_CONNECTIVITY_ITERATION = 2;
/**
 * Allowance for the command nodes that do not grow with the iteration counts: reset, expand, the
 * published radix sort, gather, COO to CSR assembly, preconditioner, component initialisation,
 * solver setup and the four output nodes. The published `GPUSort` and `GPUScan` add nodes as the
 * edge and vertex counts grow (more histogram blocks and scan levels): the part measured at
 * 10.0.0-alpha.10 is 16 nodes at 4 edges, 33 at 255, 43 at 4,096 and 55 at the largest accepted
 * configuration (2,097,152 edges, 1,048,576 vertices). The allowance is above that largest
 * measurement, so the estimate is an upper bound and never below the real count. A test counts the
 * nodes a graph really receives at both extremes.
 */
export const SPARSE_RESISTANCE_FIXED_COMMAND_NODE_COUNT = 64;

/** Upper bound of the command nodes `addSparseResistanceConnectivityRecipe` adds to a graph. */
export function estimateSparseResistanceCommandNodeCount(
  maximumIterations: number,
  connectivityIterations: number
): number {
  return (
    SPARSE_RESISTANCE_FIXED_COMMAND_NODE_COUNT +
    COMMAND_NODES_PER_ITERATION * maximumIterations +
    COMMAND_NODES_PER_CONNECTIVITY_ITERATION * connectivityIterations
  );
}

export type GPUSparseResistanceConnectivityOutput = {
  /** `float32`, `vertexCount` rows: potential of every vertex that shares the sink's component. */
  potentials: GraphDataView<'float32'>;
  /** `float32`, one row per edge: signed current from the first to the second endpoint. */
  currents: GraphDataView<'float32'>;
  /** `float32`, `vertexCount` rows: Circuitscape-style node current. */
  vertexCurrents: GraphDataView<'float32'>;
  /** `float32x4`, one row: effective resistance, true residual, dissipated energy, recursive residual. */
  diagnostics: GraphDataView<'float32x4'>;
  /** `uint32x4`, one row: `[converged, iterations, reachable, invalidEdges]`. */
  status: GraphDataView<'uint32x4'>;
};

export type GPUSparseResistanceConnectivityProps = {
  /** Node and transient ID prefix. Default 'sparse-resistance-connectivity'. */
  id?: string;
  /**
   * Undirected edges listed exactly once. Parallel conductances add; zero conductance is a barrier;
   * self edges carry no current. Rows: 1 to 2,097,152.
   */
  edges: GraphDataView<'uint32x2'>;
  /**
   * One finite nonnegative conductance per edge. Invalid rows (endpoint >= vertexCount,
   * non-finite or negative conductance) are counted in status.w and block the solve.
   * Normalise so the largest conductance is about 1 (see the float32 precision note in the docs).
   */
  conductances: GraphDataView<'float32'>;
  /** 2 to 1,048,576. */
  vertexCount: number;
  source: number;
  sink: number;
  /** Unrolled PCG iterations, 1 to 4,096. Default 1,024. Compile-time. */
  maximumIterations?: number;
  /** Absolute maximum current-conservation residual for unit injection. Default 1e-4. */
  tolerance?: number;
  /** Unrolled connectivity label-propagation rounds, 1 to 1,024. Default 128. Compile-time. */
  connectivityIterations?: number;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUSparseResistanceConnectivityOutput}.
   */
  output?: Partial<GPUSparseResistanceConnectivityOutput>;
};

/**
 * Sparse grounded-Laplacian Jacobi-PCG resistance solve with unit current injection, component
 * masking, an independent current-conservation residual and per-edge and per-vertex currents.
 *
 * Outputs: `potentials` (vertexCount), `currents` (edgeCount, signed first to second endpoint),
 * `vertexCurrents` (vertexCount, Circuitscape-style node current) and `diagnostics`
 * `[effectiveResistance, maximumTrueResidual, dissipatedEnergy, recursiveResidualNorm]`.
 * `status` is `[converged, iterations, reachable, invalidEdges]`, where `reachable` is 1 when the
 * source and sink share a component, 0 when they provably do not and 2 when label propagation did
 * not converge within `connectivityIterations`.
 */
export function addSparseResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSparseResistanceConnectivityProps
) {
  const id = props.id ?? 'sparse-resistance-connectivity';
  const {edges, conductances, vertexCount, source, sink} = props;
  const edgeCount = edges.length;
  const maximumIterations = props.maximumIterations ?? 1024;
  const tolerance = props.tolerance ?? 1e-4;
  const connectivityIterations = props.connectivityIterations ?? 128;
  if (!Number.isSafeInteger(vertexCount) || vertexCount < 2 || vertexCount > MAXIMUM_VERTEX_COUNT) {
    throw new Error(`${id} vertexCount must be an integer in [2, ${MAXIMUM_VERTEX_COUNT}]`);
  }
  if (edgeCount < 1 || edgeCount > MAXIMUM_EDGE_COUNT) {
    throw new Error(`${id} edges must have between 1 and ${MAXIMUM_EDGE_COUNT} rows`);
  }
  validateCapacity(id, 'source', source, 0, vertexCount - 1);
  validateCapacity(id, 'sink', sink, 0, vertexCount - 1);
  validateRequirement(id, source !== sink, 'source and sink must be different vertices');
  validateViewLength(id, 'conductances', conductances, edgeCount);
  validateCapacity(id, 'maximumIterations', maximumIterations, 1, MAXIMUM_ITERATIONS);
  validateCapacity(
    id,
    'connectivityIterations',
    connectivityIterations,
    1,
    MAXIMUM_CONNECTIVITY_ITERATIONS
  );
  validateRequirement(
    id,
    Number.isFinite(tolerance) && tolerance > 0,
    'tolerance must be a positive finite number'
  );
  // The unrolled rounds set the command node count; the default budget of 32,768 nodes holds the
  // largest accepted configuration (4,096 iterations and 1,024 rounds: about 22,600 nodes).
  const estimatedCommandNodeCount = estimateSparseResistanceCommandNodeCount(
    maximumIterations,
    connectivityIterations
  );
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  const output = props.output;

  const potentials = getRecipeView(
    graph,
    `${id}-potentials`,
    'float32',
    vertexCount,
    output?.potentials
  );
  const currents = getRecipeView(graph, `${id}-currents`, 'float32', edgeCount, output?.currents);
  const vertexCurrents = getRecipeView(
    graph,
    `${id}-vertex-currents`,
    'float32',
    vertexCount,
    output?.vertexCurrents
  );
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1, output?.status);
  const diagnostics = getRecipeView(
    graph,
    `${id}-diagnostics`,
    'float32x4',
    1,
    output?.diagnostics
  );
  validateRecipeBoundary(
    graph,
    id,
    [edges, conductances],
    [potentials, currents, vertexCurrents, diagnostics, status]
  );

  const system = addSparseResistanceSystemNodes(graph, {
    id,
    operation: OPERATION,
    edges,
    conductances,
    vertexCount,
    connectivityIterations
  });
  const {solution, partials, partialCount} = addSparseResistanceSolveNodes(graph, {
    id,
    operation: OPERATION,
    system,
    vertexCount,
    source,
    sink,
    maximumIterations,
    tolerance
  });
  const {rowOffsets, csrColumns, csrConductances, labels, state} = system;
  const workgroupSize = SPARSE_RESISTANCE_WORKGROUP_SIZE;
  const stateConstants = SPARSE_RESISTANCE_STATE_WGSL;
  const sum = getSegmentTreeSumWGSL('sum');
  const maximum = getSegmentTreeSumWGSL('max');

  // Independent true residual over every vertex and the dissipated energy, per workgroup.
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-true-residual`,
      operation: OPERATION,
      invocationCount: vertexCount,
      workgroupSize,
      guardIndex: false,
      declarations: `override SOURCE_VERTEX: u32;
override SINK_VERTEX: u32;
override PARTIAL_COUNT: u32;
${SEGMENT_PARTIAL_SUMS_WGSL}${IS_FINITE_VALUE_WGSL}`,
      constants: {SOURCE_VERTEX: source, SINK_VERTEX: sink, PARTIAL_COUNT: partialCount},
      bindings: [
        {name: 'rowOffsets', view: rowOffsets, type: 'u32', access: 'read'},
        {name: 'csrColumns', view: csrColumns, type: 'u32', access: 'read'},
        {name: 'csrConductances', view: csrConductances, type: 'f32', access: 'read'},
        {name: 'solution', view: solution, type: 'f32', access: 'read'},
        {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
      ],
      // No early return: the tree barriers must stay in uniform control flow.
      body: `var residualMaximum = 0.0;
  var energy = 0.0;
  if (index < INVOCATION_COUNT) {
    let first = rowOffsets[rowOffsetsOffset + index];
    let last = rowOffsets[rowOffsetsOffset + index + 1u];
    let center = solution[solutionOffset + index];
    var flow = 0.0;
    for (var slot = first; slot < last; slot++) {
      let difference = center - solution[solutionOffset + csrColumns[csrColumnsOffset + slot]];
      let current = csrConductances[csrConductancesOffset + slot] * difference;
      flow += current;
      // Every edge is visited from both ends.
      energy += 0.5 * current * difference;
    }
    let expected = select(0.0, 1.0, index == SOURCE_VERTEX) - select(0.0, 1.0, index == SINK_VERTEX);
    let balance = abs(flow - expected);
    // Every vertex counts, including floating ones: their exact residual is 0 unless they hold the source.
    residualMaximum = select(3.0e38, balance, isFiniteValue(balance));
  }
  var partialSum = residualMaximum;
  ${maximum}
  let maximumTotal = partialSums[0];
  workgroupBarrier();
  partialSum = energy;
  ${sum}
  if (localInvocationIndex == 0u && workgroupIndex < PARTIAL_COUNT) {
    partials[partialsOffset + workgroupIndex] = maximumTotal;
    partials[partialsOffset + PARTIAL_COUNT + workgroupIndex] = partialSums[0];
  }`
    })
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-finish`,
      operation: OPERATION,
      invocationCount: workgroupSize,
      workgroupSize,
      guardIndex: false,
      declarations: `override PARTIAL_COUNT: u32;
override SOURCE_VERTEX: u32;
override TOLERANCE: f32;
override VERTEX_COUNT: u32;
${SEGMENT_PARTIAL_SUMS_WGSL}${IS_FINITE_VALUE_WGSL}${GET_NAN_WGSL}${stateConstants}`,
      constants: {
        PARTIAL_COUNT: partialCount,
        SOURCE_VERTEX: source,
        TOLERANCE: tolerance,
        VERTEX_COUNT: vertexCount
      },
      bindings: [
        {name: 'partials', view: partials, type: 'f32', access: 'read'},
        {name: 'solution', view: solution, type: 'f32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'},
        {name: 'diagnostics', view: diagnostics, type: 'f32', access: 'read_write'}
      ],
      body: `var partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum = max(partialSum, partials[partialsOffset + partial]);
  }
  ${maximum}
  let maximumResidual = partialSums[0];
  workgroupBarrier();
  partialSum = 0.0;
  for (var partial = localInvocationIndex; partial < PARTIAL_COUNT; partial += ${workgroupSize}u) {
    partialSum += partials[partialsOffset + PARTIAL_COUNT + partial];
  }
  ${sum}
  if (localInvocationIndex == 0u) {
    let energyTotal = partialSums[0];
    let reachable = state[stateOffset + STATE_REACHABLE];
    let invalid = state[stateOffset + STATE_INVALID];
    let solved = invalid == 0u && reachable != 0u;
    let resistance = solution[solutionOffset + SOURCE_VERTEX];
    // A maximum residual below 1 / n bounds the summed residual below 1, which proves the source
    // reaches the sink even when the component labels are undetermined (reachable == 2).
    let certified = reachable == 1u || maximumResidual * f32(VERTEX_COUNT) < 1.0;
    let converged = solved && certified && maximumResidual <= TOLERANCE && isFiniteValue(resistance) && isFiniteValue(energyTotal);
    state[stateOffset + STATE_CONVERGED] = u32(converged);
    status[statusOffset + 0u] = u32(converged);
    status[statusOffset + 1u] = state[stateOffset + STATE_ITERATIONS];
    status[statusOffset + 2u] = reachable;
    status[statusOffset + 3u] = invalid;
    let missing = getNaN();
    diagnostics[diagnosticsOffset + 0u] = select(missing, resistance, converged);
    diagnostics[diagnosticsOffset + 1u] = select(missing, maximumResidual, solved);
    diagnostics[diagnosticsOffset + 2u] = select(missing, energyTotal, solved);
    diagnostics[diagnosticsOffset + 3u] = sqrt(bitcast<f32>(state[stateOffset + STATE_RESIDUAL_DOT_RESIDUAL]));
  }`
    })
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-vertex-outputs`,
      operation: OPERATION,
      invocationCount: vertexCount,
      declarations: `override SINK_VERTEX: u32;
${GET_NAN_WGSL}${stateConstants}`,
      constants: {SINK_VERTEX: sink},
      bindings: [
        {name: 'rowOffsets', view: rowOffsets, type: 'u32', access: 'read'},
        {name: 'csrColumns', view: csrColumns, type: 'u32', access: 'read'},
        {name: 'csrConductances', view: csrConductances, type: 'f32', access: 'read'},
        {name: 'solution', view: solution, type: 'f32', access: 'read'},
        {name: 'labels', view: labels, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read_write'},
        {name: 'vertexCurrents', view: vertexCurrents, type: 'f32', access: 'read_write'}
      ],
      body: `let first = rowOffsets[rowOffsetsOffset + index];
  let last = rowOffsets[rowOffsetsOffset + index + 1u];
  let center = solution[solutionOffset + index];
  var inflow = 0.0;
  var outflow = 0.0;
  for (var slot = first; slot < last; slot++) {
    let current = csrConductances[csrConductancesOffset + slot] * (center - solution[solutionOffset + csrColumns[csrColumnsOffset + slot]]);
    outflow += max(current, 0.0);
    inflow += max(-current, 0.0);
  }
  let componentsKnown = state[stateOffset + STATE_COMPONENTS] != 0u;
  let connected = !componentsKnown || labels[labelsOffset + index] == labels[labelsOffset + SINK_VERTEX];
  let solved = state[stateOffset + STATE_REACHABLE] != 0u && state[stateOffset + STATE_INVALID] == 0u;
  let converged = state[stateOffset + STATE_CONVERGED] != 0u;
  potentials[potentialsOffset + index] = select(getNaN(), center, solved && connected);
  vertexCurrents[vertexCurrentsOffset + index] = select(getNaN(), max(inflow, outflow), converged && connected);`
    })
  );

  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-edge-currents`,
      operation: OPERATION,
      invocationCount: edgeCount,
      declarations: `${GET_NAN_WGSL}
fn readEdges(row: u32) -> vec2<u32> {
  return vec2<u32>(edges[edgesOffset + row * 2u + 0u], edges[edgesOffset + row * 2u + 1u]);
}
fn readStatus(row: u32) -> vec4<u32> {
  return vec4<u32>(
    status[statusOffset + row * 4u + 0u],
    status[statusOffset + row * 4u + 1u],
    status[statusOffset + row * 4u + 2u],
    status[statusOffset + row * 4u + 3u]
  );
}
`,
      bindings: [
        {name: 'edges', view: edges, type: 'u32', access: 'read'},
        {name: 'conductances', view: conductances, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'output', view: currents, type: 'f32', access: 'read_write'}
      ],
      body: `var current = getNaN();
  let endpoints = readEdges(index);
  if (readStatus(0u).x != 0u) {
    let conductance = conductances[conductancesOffset + index];
    current = 0.0;
    if (conductance > 0.0) {
      current = conductance * (potentials[potentialsOffset + endpoints.x] - potentials[potentialsOffset + endpoints.y]);
    }
  }
  output[outputOffset + index] = current;`
    })
  );

  return {
    outputs: {potentials, currents, vertexCurrents, diagnostics},
    status,
    /** Number of command nodes this recipe added to the graph. */
    estimatedCommandNodeCount,
    provenance: {
      edges,
      conductances,
      source,
      sink,
      injection: 1 as const,
      tolerance,
      maximumIterations,
      connectivityIterations,
      solver: 'grounded-jacobi-conjugate-gradient' as const
    }
  };
}
