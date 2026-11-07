// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {Buffer, type Binding} from '@luma.gl/core';
import type {GPUComputationDatumBinding} from '../computation-record/computation-record-types';
import {Kernel} from '../execution/compute/kernel';
import {
  createGPUGraphAlgorithmsIterationGate,
  type GPUGraphAlgorithmsIterationGate
} from './gpu-graph-iteration-gate';
import type {
  GPUCommandGraph,
  GraphBufferUse,
  GraphDataView,
  GPUCommandGraphGPUIndirectCondition,
  GraphVectorView
} from '../gpu-core/gpu-command-graph';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset
} from '../gpu-core/graph-data-view-utils';
import {
  getGPUGraphAlgorithmsPassAnnotation,
  withGPUGraphAlgorithmsIterationScope
} from './gpu-graph-computation-record';
import type {GPUGraphAlgorithmsSingleSourceShortestPath} from './gpu-graph-single-source-shortest-path';
import type {GPUGraphAlgorithmsAdjacency} from './gpu-graph-topology';

const SHORTEST_PATH_WORKGROUP_SIZE = 256;
const POSITIVE_INFINITY_BITS = 0x7f800000;
const UNREACHABLE_PREDECESSOR = 0xffffffff;

type ShortestPathDataView = GraphDataView<'uint32'> | GraphDataView<'float32'>;

type ImportedShortestPathAdjacency = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  edgeWeights?: GraphDataView<'float32'>;
  overflow: GraphDataView<'uint32'>;
};

type ImportedShortestPathState = {
  id: string;
  vertexCount: number;
  edgeCount: number;
  sourceVertex: number;
  maxIterations: number;
  invalidateOnNonConvergence: boolean;
  distances: GraphDataView<'float32'>;
  predecessors: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  invalidWeightCount: GraphDataView<'uint32'>;
  previousDistances?: GraphDataView<'float32'>;
  changed?: GraphDataView<'uint32'>;
  sourceVertices?: GraphVectorView<'uint32'>;
  targetVertices?: GraphVectorView<'uint32'>;
  sourceWeights?: GraphVectorView<'float32'>;
  primaryAdjacency: ImportedShortestPathAdjacency;
  secondaryAdjacency?: ImportedShortestPathAdjacency;
  maxComputeWorkgroupsPerDimension: number;
  /** Indirect vertex-dispatch triple that is zeroed once the search converges. */
  gate?: GPUGraphAlgorithmsIterationGate;
};

type ShortestPathBinding = {
  view: ShortestPathDataView;
  usage: GraphBufferUse['usage'];
  atomic?: boolean;
};

type ShortestPathPassProps = {
  operation: string;
  datum?: GPUComputationDatumBinding;
  invocationCount?: number;
  workgroupSize: number;
  id: string;
  source: string;
  bindings: Record<string, ShortestPathBinding>;
  dispatchLayout: GPUBoundedDispatchLayout;
  /** Skips the pass entirely (zero workgroups) once the iteration gate is disarmed. */
  condition?: GPUCommandGraphGPUIndirectCondition;
};

/** Adds GPU weighted Bellman-Ford passes using an explicit bounded dispatch limit. @internal */
export function addGPUGraphAlgorithmsSingleSourceShortestPathToGraphWithDispatchLimit<Parameters>(
  search: GPUGraphAlgorithmsSingleSourceShortestPath,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  const directed = search.topology.graph.directed;
  const useIncoming = directed && search.direction === 'incoming';
  const primaryAdjacency = importShortestPathAdjacency(
    commandGraph,
    `${search.id}-${useIncoming ? 'incoming' : 'outgoing'}`,
    useIncoming ? search.topology.reverse! : search.topology.forward
  );
  const state: ImportedShortestPathState = {
    id: search.id,
    vertexCount: search.topology.graph.vertexCount,
    edgeCount: search.topology.graph.edgeCount,
    sourceVertex: search.sourceVertex,
    maxIterations: search.maxIterations,
    invalidateOnNonConvergence: search.invalidateOnNonConvergence,
    distances: commandGraph.importGPUVector(`${search.id}-distances`, search.distances).data[0],
    predecessors: commandGraph.importGPUVector(`${search.id}-predecessors`, search.predecessors)
      .data[0],
    converged: search.converged
      ? commandGraph.importGPUVector(`${search.id}-converged`, search.converged).data[0]
      : createTransientView(commandGraph, `${search.id}-converged-scratch`, 'uint32', 1),
    invalidWeightCount: search.invalidWeightCount
      ? commandGraph.importGPUVector(`${search.id}-invalid-weight-count`, search.invalidWeightCount)
          .data[0]
      : createTransientView(commandGraph, `${search.id}-invalid-weight-scratch`, 'uint32', 1),
    ...(search.maxIterations > 0 && search.algorithm === 'bellman-ford'
      ? {
          previousDistances: createTransientView(
            commandGraph,
            `${search.id}-previous-distances`,
            'float32',
            search.topology.graph.vertexCount
          ),
          changed: createTransientView(commandGraph, `${search.id}-changed`, 'uint32', 1)
        }
      : {}),
    ...(search.topology.graph.edgeWeights
      ? {
          sourceVertices: commandGraph.importGPUVector(
            `${search.id}-source-vertices`,
            search.topology.graph.sourceVertices
          ),
          targetVertices: commandGraph.importGPUVector(
            `${search.id}-target-vertices`,
            search.topology.graph.targetVertices
          ),
          sourceWeights: commandGraph.importGPUVector(
            `${search.id}-source-weights`,
            search.topology.graph.edgeWeights
          )
        }
      : {}),
    primaryAdjacency,
    ...(directed && search.direction === 'both'
      ? {
          secondaryAdjacency: importShortestPathAdjacency(
            commandGraph,
            `${search.id}-incoming`,
            search.topology.reverse!
          )
        }
      : {}),
    maxComputeWorkgroupsPerDimension,
    ...(search.maxIterations > 0 && search.algorithm === 'bellman-ford'
      ? {
          gate: createGPUGraphAlgorithmsIterationGate(commandGraph, {
            id: `${search.id}-iteration-gate`,
            operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
            shapes: [[Math.max(search.topology.graph.vertexCount, 1), SHORTEST_PATH_WORKGROUP_SIZE]]
          })
        }
      : {})
  };

  addInitializationPass(commandGraph, state);
  for (const [chunkIndex, weights] of state.sourceWeights?.data.entries() ?? []) {
    if (weights.length > 0) {
      addWeightValidationPass(commandGraph, {
        state,
        weights,
        sources: state.sourceVertices!.data[chunkIndex],
        targets: state.targetVertices!.data[chunkIndex],
        chunkIndex
      });
    }
  }
  if (search.algorithm === 'frontier') {
    addFrontierSearch(commandGraph, state, useIncoming);
    if (state.invalidateOnNonConvergence) {
      addInvalidateNonConvergedPass(commandGraph, state);
    }
    return;
  }
  addSourcePass(commandGraph, state);

  for (let iteration = 0; iteration < state.maxIterations; iteration++) {
    withGPUGraphAlgorithmsIterationScope(
      commandGraph,
      {
        loopId: `${state.id}-iterations`,
        iteration,
        maximumIterations: state.maxIterations,
        gated: Boolean(state.gate)
      },
      () => {
        addSnapshotPass(commandGraph, state, iteration);
        addRelaxationPass(commandGraph, {
          state,
          adjacency: state.primaryAdjacency,
          iteration,
          direction: useIncoming ? 'incoming' : 'outgoing'
        });
        if (state.secondaryAdjacency) {
          addRelaxationPass(commandGraph, {
            state,
            adjacency: state.secondaryAdjacency,
            iteration,
            direction: 'incoming'
          });
        }
        addPredecessorResetPass(commandGraph, state, iteration);
        addPredecessorPass(commandGraph, {
          state,
          adjacency: state.primaryAdjacency,
          iteration,
          direction: useIncoming ? 'incoming' : 'outgoing'
        });
        if (state.secondaryAdjacency) {
          addPredecessorPass(commandGraph, {
            state,
            adjacency: state.secondaryAdjacency,
            iteration,
            direction: 'incoming'
          });
        }
        addConvergencePass(commandGraph, state, iteration);
      }
    );
  }
  if (state.invalidateOnNonConvergence) {
    addInvalidateNonConvergedPass(commandGraph, state);
  }
}

/** Replaces partial distances with NaN when the round budget ended before a fixed point. */
function addInvalidateNonConvergedPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState
): void {
  const bindings: Record<string, ShortestPathBinding> = {
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read'},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read'},
    ...getOverflowBindings(state)
  };
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getOverflowConstants(state)}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, Math.max(state.vertexCount, 1))}
  // Invalid weights and CSR overflow already fail closed; only an exhausted budget is invalidated.
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u ||
      invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET] != 0u || ${getOverflowGuard(state)}) { return; }
  atomicStore(&distances[DISTANCES_OFFSET + index], 0x7fc00000u);
  atomicStore(&predecessors[PREDECESSORS_OFFSET + index], ${UNREACHABLE_PREDECESSOR}u);
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-invalidate-non-converged`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, Math.max(state.vertexCount, 1))
  });
}

/** Imports the capacity-bounded CSR and optional edge weights for one traversal direction. */
function importShortestPathAdjacency<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  id: string,
  adjacency: GPUGraphAlgorithmsAdjacency
): ImportedShortestPathAdjacency {
  return {
    offsets: commandGraph.importGPUVector(`${id}-offsets`, adjacency.offsets).data[0],
    neighbors: commandGraph.importGPUVector(`${id}-neighbors`, adjacency.neighbors).data[0],
    ...(adjacency.edgeWeights
      ? {
          edgeWeights: commandGraph.importGPUVector(`${id}-edge-weights`, adjacency.edgeWeights)
            .data[0]
        }
      : {}),
    overflow: commandGraph.importGPUVector(`${id}-overflow`, adjacency.overflow).data[0]
  };
}

/** Resets every caller-visible output and internal status before each graph encoding. */
function addInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState
): void {
  const bindings: Record<string, ShortestPathBinding> = {
    distances: {view: state.distances, usage: 'storage-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-write', atomic: true},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-write', atomic: true},
    ...(state.gate ? {dispatch: {view: state.gate.view, usage: 'storage-write' as const}} : {})
  };
  const source = /* wgsl */ `
${state.gate ? `const dispatchOffset: u32 = ${getViewElementOffset(state.gate.view)}u;` : ''}
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, Math.max(state.vertexCount, 1))}
  if (index == 0u) {
    atomicStore(&converged[CONVERGED_OFFSET], 0u);
    atomicStore(&invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET], 0u);
    ${state.gate?.armWGSL ?? ''}
  }
  if (index >= VERTEX_COUNT) { return; }
  atomicStore(&distances[DISTANCES_OFFSET + index], ${POSITIVE_INFINITY_BITS}u);
  atomicStore(&predecessors[PREDECESSORS_OFFSET + index], ${UNREACHABLE_PREDECESSOR}u);
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-initialize`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, Math.max(state.vertexCount, 1))
  });
}

/** Counts every invalid original source edge exactly once, excluding invalid endpoint rows. */
function addWeightValidationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    state: ImportedShortestPathState;
    sources: GraphDataView<'uint32'>;
    targets: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    chunkIndex: number;
  }
): void {
  const {state} = props;
  const bindings: Record<string, ShortestPathBinding> = {
    sources: {view: props.sources, usage: 'storage-read'},
    targets: {view: props.targets, usage: 'storage-read'},
    weights: {view: props.weights, usage: 'storage-read'},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read-write', atomic: true}
  };
  const source = /* wgsl */ `
const EDGE_COUNT: u32 = ${props.weights.length}u;
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const SOURCES_OFFSET: u32 = ${getViewElementOffset(props.sources)}u;
const TARGETS_OFFSET: u32 = ${getViewElementOffset(props.targets)}u;
const WEIGHTS_OFFSET: u32 = ${getViewElementOffset(props.weights)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, props.weights.length)}
  if (index >= EDGE_COUNT) { return; }
  if (sources[SOURCES_OFFSET + index] >= VERTEX_COUNT ||
      targets[TARGETS_OFFSET + index] >= VERTEX_COUNT) { return; }
  let bits = bitcast<u32>(weights[WEIGHTS_OFFSET + index]);
  let magnitude = bits & 0x7fffffffu;
  let negative = (bits & 0x80000000u) != 0u && magnitude != 0u;
  if (magnitude >= 0x7f800000u || negative) {
    atomicAdd(&invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET], 1u);
  }
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'edge'},
    invocationCount: props.weights.length,
    id: `${state.id}-validate-weights-${props.chunkIndex}`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, props.weights.length)
  });
}

/** Publishes the source only after complete adjacency and weight status are known. */
function addSourcePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState
): void {
  const bindings: Record<string, ShortestPathBinding> = {
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read-write', atomic: true},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read'},
    ...getOverflowBindings(state)
  };
  const triviallyConverged = state.vertexCount <= 1 || state.edgeCount === 0;
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const SOURCE_VERTEX: u32 = ${state.sourceVertex}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getOverflowConstants(state)}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  if (invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET] != 0u || ${getOverflowGuard(state)}) {
    return;
  }
  if (VERTEX_COUNT > 0u) {
    atomicStore(&distances[DISTANCES_OFFSET + SOURCE_VERTEX], 0u);
  }
  atomicStore(&converged[CONVERGED_OFFSET], ${triviallyConverged ? '1u' : '0u'});
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: 1,
    invocationCount: 1,
    id: `${state.id}-source`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Saves the previous round so parallel relaxations cannot cross multiple hops in one dispatch. */
function addSnapshotPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState,
  iteration: number
): void {
  const previousDistances = state.previousDistances!;
  const changed = state.changed!;
  const bindings: Record<string, ShortestPathBinding> = {
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    previousDistances: {view: previousDistances, usage: 'storage-write'},
    changed: {view: changed, usage: 'storage-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read'}
  };
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREVIOUS_DISTANCES_OFFSET: u32 = ${getViewElementOffset(previousDistances)}u;
const CHANGED_OFFSET: u32 = ${getViewElementOffset(changed)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, Math.max(state.vertexCount, 1))}
  if (index == 0u) { atomicStore(&changed[CHANGED_OFFSET], 0u); }
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  previousDistances[PREVIOUS_DISTANCES_OFFSET + index] =
    bitcast<f32>(atomicLoad(&distances[DISTANCES_OFFSET + index]));
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${iteration}-snapshot`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, Math.max(state.vertexCount, 1)),
    condition: state.gate?.getCondition(`${state.id}-iteration-${iteration}-snapshot`, 0)
  });
}

/** Computes one exact-hop parallel Bellman-Ford relaxation through aligned weighted CSR. */
function addRelaxationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    state: ImportedShortestPathState;
    adjacency: ImportedShortestPathAdjacency;
    iteration: number;
    direction: 'outgoing' | 'incoming';
  }
): void {
  const {state, adjacency} = props;
  const previousDistances = state.previousDistances!;
  const changed = state.changed!;
  const bindings: Record<string, ShortestPathBinding> = {
    offsets: {view: adjacency.offsets, usage: 'storage-read'},
    neighbors: {view: adjacency.neighbors, usage: 'storage-read'},
    ...(adjacency.edgeWeights
      ? {edgeWeights: {view: adjacency.edgeWeights, usage: 'storage-read' as const}}
      : {}),
    previousDistances: {view: previousDistances, usage: 'storage-read'},
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    changed: {view: changed, usage: 'storage-read-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read'}
  };
  const weightOffset = adjacency.edgeWeights
    ? `const WEIGHTS_OFFSET: u32 = ${getViewElementOffset(adjacency.edgeWeights)}u;`
    : '';
  const weight = adjacency.edgeWeights ? 'edgeWeights[WEIGHTS_OFFSET + slot]' : '1.0';
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const CAPACITY: u32 = ${adjacency.neighbors.length}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(adjacency.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(adjacency.neighbors)}u;
const PREVIOUS_DISTANCES_OFFSET: u32 = ${getViewElementOffset(previousDistances)}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const CHANGED_OFFSET: u32 = ${getViewElementOffset(changed)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
${weightOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, state.vertexCount)}
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  let distance = previousDistances[PREVIOUS_DISTANCES_OFFSET + index];
  if (bitcast<u32>(distance) >= 0x7f800000u) { return; }
  let first = min(offsets[OFFSETS_OFFSET + index], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + index + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let candidate = distance + ${weight};
    let candidateBits = bitcast<u32>(candidate);
    if (candidateBits >= 0x7f800000u) { continue; }
    let previous = atomicMin(&distances[DISTANCES_OFFSET + neighbor], candidateBits);
    if (candidateBits < previous) { atomicStore(&changed[CHANGED_OFFSET], 1u); }
  }
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-relax-${props.direction}`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, state.vertexCount),
    condition: state.gate?.getCondition(
      `${state.id}-iteration-${props.iteration}-relax-${props.direction}`,
      0
    )
  });
}

/** Clears stale parents only where this synchronized round discovered a better distance. */
function addPredecessorResetPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState,
  iteration: number
): void {
  const previousDistances = state.previousDistances!;
  const bindings: Record<string, ShortestPathBinding> = {
    previousDistances: {view: previousDistances, usage: 'storage-read'},
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-read-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read'}
  };
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const SOURCE_VERTEX: u32 = ${state.sourceVertex}u;
const PREVIOUS_DISTANCES_OFFSET: u32 = ${getViewElementOffset(previousDistances)}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, state.vertexCount)}
  if (index >= VERTEX_COUNT || index == SOURCE_VERTEX || converged[CONVERGED_OFFSET] != 0u) {
    return;
  }
  let previous = bitcast<u32>(previousDistances[PREVIOUS_DISTANCES_OFFSET + index]);
  let current = atomicLoad(&distances[DISTANCES_OFFSET + index]);
  if (current < previous) {
    atomicStore(&predecessors[PREDECESSORS_OFFSET + index], ${UNREACHABLE_PREDECESSOR}u);
  }
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${iteration}-reset-predecessors`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, state.vertexCount),
    condition: state.gate?.getCondition(`${state.id}-iteration-${iteration}-reset-predecessors`, 0)
  });
}

/** Selects the lowest stable parent among equally short, same-hop newly discovered paths. */
function addPredecessorPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    state: ImportedShortestPathState;
    adjacency: ImportedShortestPathAdjacency;
    iteration: number;
    direction: 'outgoing' | 'incoming';
  }
): void {
  const {state, adjacency} = props;
  const previousDistances = state.previousDistances!;
  const bindings: Record<string, ShortestPathBinding> = {
    offsets: {view: adjacency.offsets, usage: 'storage-read'},
    neighbors: {view: adjacency.neighbors, usage: 'storage-read'},
    ...(adjacency.edgeWeights
      ? {edgeWeights: {view: adjacency.edgeWeights, usage: 'storage-read' as const}}
      : {}),
    previousDistances: {view: previousDistances, usage: 'storage-read'},
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-read-write', atomic: true},
    converged: {view: state.converged, usage: 'storage-read'}
  };
  const weightOffset = adjacency.edgeWeights
    ? `const WEIGHTS_OFFSET: u32 = ${getViewElementOffset(adjacency.edgeWeights)}u;`
    : '';
  const weight = adjacency.edgeWeights ? 'edgeWeights[WEIGHTS_OFFSET + slot]' : '1.0';
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const SOURCE_VERTEX: u32 = ${state.sourceVertex}u;
const CAPACITY: u32 = ${adjacency.neighbors.length}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(adjacency.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(adjacency.neighbors)}u;
const PREVIOUS_DISTANCES_OFFSET: u32 = ${getViewElementOffset(previousDistances)}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
${weightOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, state.vertexCount)}
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  let sourceDistance = previousDistances[PREVIOUS_DISTANCES_OFFSET + index];
  if (bitcast<u32>(sourceDistance) >= 0x7f800000u) { return; }
  let first = min(offsets[OFFSETS_OFFSET + index], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + index + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == SOURCE_VERTEX) { continue; }
    let previousDistance = bitcast<u32>(previousDistances[PREVIOUS_DISTANCES_OFFSET + neighbor]);
    let currentDistance = atomicLoad(&distances[DISTANCES_OFFSET + neighbor]);
    let candidate = bitcast<u32>(sourceDistance + ${weight});
    if (currentDistance < previousDistance && candidate == currentDistance) {
      atomicMin(&predecessors[PREDECESSORS_OFFSET + neighbor], index);
    }
  }
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-predecessors-${props.direction}`,
    source,
    bindings,
    dispatchLayout: getDispatchLayout(state, state.vertexCount),
    condition: state.gate?.getCondition(
      `${state.id}-iteration-${props.iteration}-predecessors-${props.direction}`,
      0
    )
  });
}

/** Publishes fixed-point convergence without synchronously reading status back to the CPU. */
function addConvergencePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState,
  iteration: number
): void {
  const changed = state.changed!;
  const bindings: Record<string, ShortestPathBinding> = {
    changed: {view: changed, usage: 'storage-read'},
    converged: {view: state.converged, usage: 'storage-read-write', atomic: true},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read'},
    ...getOverflowBindings(state),
    ...(state.gate ? {dispatch: {view: state.gate.view, usage: 'storage-write' as const}} : {})
  };
  const reachesSimplePathBound = iteration + 1 >= state.vertexCount - 1;
  const source = /* wgsl */ `
const CHANGED_OFFSET: u32 = ${getViewElementOffset(changed)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${state.gate ? `const dispatchOffset: u32 = ${getViewElementOffset(state.gate.view)}u;` : ''}
${getOverflowConstants(state)}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  if (invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET] != 0u || ${getOverflowGuard(state)}) {
    atomicStore(&converged[CONVERGED_OFFSET], 0u);
    return;
  }
  if (changed[CHANGED_OFFSET] == 0u${reachesSimplePathBound ? ' || true' : ''}) {
    atomicStore(&converged[CONVERGED_OFFSET], 1u);
    ${state.gate?.disarmWGSL ?? ''}
  }
}`;
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: 1,
    invocationCount: 1,
    id: `${state.id}-iteration-${iteration}-convergence`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Returns storage reads for every adjacency direction required by this traversal. */
function getOverflowBindings(
  state: ImportedShortestPathState
): Record<string, ShortestPathBinding> {
  return {
    overflow: {view: state.primaryAdjacency.overflow, usage: 'storage-read'},
    ...(state.secondaryAdjacency
      ? {secondaryOverflow: {view: state.secondaryAdjacency.overflow, usage: 'storage-read'}}
      : {})
  };
}

/** Emits aligned component offsets for primary and optional secondary adjacency status. */
function getOverflowConstants(state: ImportedShortestPathState): string {
  return [
    `const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.primaryAdjacency.overflow)}u;`,
    ...(state.secondaryAdjacency
      ? [
          `const SECONDARY_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.secondaryAdjacency.overflow)}u;`
        ]
      : [])
  ].join('\n');
}

/** Emits the fail-closed condition for every selected adjacency direction. */
function getOverflowGuard(state: ImportedShortestPathState): string {
  return `overflow[OVERFLOW_OFFSET] != 0u${
    state.secondaryAdjacency ? ' || secondaryOverflow[SECONDARY_OVERFLOW_OFFSET] != 0u' : ''
  }`;
}

/** Declares storage arrays, reinterpreting non-negative float32 distances as atomic uint32. */
function getBindingDeclarations(bindings: Record<string, ShortestPathBinding>): string {
  return Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      const element = binding.atomic
        ? 'atomic<u32>'
        : binding.view.format === 'float32'
          ? 'f32'
          : 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');
}

/** Compiles one bounded GPU pass without submitting, reading back, or owning caller buffers. */
function addShortestPathPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: ShortestPathPassProps
): void {
  const annotation = commandGraph.recording
    ? getGPUGraphAlgorithmsPassAnnotation(commandGraph, {
        operation: props.operation,
        source: props.source,
        bindings: Object.entries(props.bindings).map(([name, binding]) => ({name, ...binding})),
        workgroupSize: props.workgroupSize,
        dispatch: props.dispatchLayout,
        invocationCount: props.invocationCount,
        datum: props.datum
      })
    : undefined;
  commandGraph.addComputePass({
    ...(annotation && {annotation}),
    id: props.id,
    ...(props.condition ? {condition: props.condition} : {}),
    resources: Object.values(props.bindings).map(({view, usage}) => ({buffer: view, usage})),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source: props.source,
        shaderLayout: {
          bindings: Object.keys(props.bindings).map((name, location) => ({
            name,
            type: 'storage' as const,
            group: 0,
            location
          }))
        }
      });

      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {};
          for (const [name, binding] of Object.entries(props.bindings)) {
            bindings[name] = getViewBinding(binding.view, getBuffer);
          }
          const kernelBindings = bindings;
          kernel.dispatch(computePass, {
            bindings: kernelBindings,
            x: props.dispatchLayout.x,
            y: props.dispatchLayout.y,
            z: props.dispatchLayout.z
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** Plans one bounded vertex or source-edge dispatch using the owning adapter's limits. */
function getDispatchLayout(
  state: ImportedShortestPathState,
  elementCount: number
): GPUBoundedDispatchLayout {
  return getGPUGraphAlgorithmsSingleSourceShortestPathDispatchLayout(
    elementCount,
    state.maxComputeWorkgroupsPerDimension
  );
}

/** Emits a three-dimensional global invocation index for the planned bounded dispatch. */
function getInvocationIndexSource(state: ImportedShortestPathState, elementCount: number): string {
  return getBoundedInvocationIndexSource(
    getDispatchLayout(state, elementCount),
    SHORTEST_PATH_WORKGROUP_SIZE
  );
}

/** Plans one bounded three-dimensional weighted shortest-path dispatch. @internal */
export function getGPUGraphAlgorithmsSingleSourceShortestPathDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphAlgorithmsSingleSourceShortestPath',
    elementCount,
    SHORTEST_PATH_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}

const FRONTIER_INDEX_SOURCE = `let index = workgroupId.x * ${SHORTEST_PATH_WORKGROUP_SIZE}u + localInvocationIndex;`;

/**
 * Frontier relaxation: each round relaxes only the vertices whose distance fell in the previous
 * round. The frontier is compacted on the GPU (stamp-deduplicated appends), ping-ponged between
 * two lists, and a one-thread controller converts its length into the next indirect dispatch.
 * Rounds reproduce the exact-hop Bellman-Ford distances and lowest-parent predecessors.
 */
function addFrontierSearch<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedShortestPathState,
  useIncoming: boolean
): void {
  const frontierWorkgroups = Math.ceil(
    Math.max(state.vertexCount, 1) / SHORTEST_PATH_WORKGROUP_SIZE
  );
  if (frontierWorkgroups > state.maxComputeWorkgroupsPerDimension) {
    // Frontier dispatch is one-dimensional; use bellman-ford for graphs beyond the dimension limit.
    throw new Error(`${state.id} frontier search exceeds the compute dispatch dimension limit`);
  }
  const vertexCount = Math.max(state.vertexCount, 1);
  const frontier = {
    stamps: createTransientView(commandGraph, `${state.id}-frontier-stamps`, 'uint32', vertexCount),
    lists: [0, 1].map(index =>
      createTransientView(commandGraph, `${state.id}-frontier-${index}`, 'uint32', vertexCount)
    ),
    distances: [0, 1].map(index =>
      createTransientView(
        commandGraph,
        `${state.id}-frontier-distances-${index}`,
        'uint32',
        vertexCount
      )
    ),
    counts: createTransientView(commandGraph, `${state.id}-frontier-counts`, 'uint32', 2),
    dispatch: createTransientView(
      commandGraph,
      `${state.id}-frontier-dispatch`,
      'uint32',
      3,
      Buffer.STORAGE | Buffer.INDIRECT
    )
  };
  const condition: GPUCommandGraphGPUIndirectCondition = {
    id: `${state.id}-frontier-dispatch-condition`,
    source: 'gpu',
    mode: 'indirect',
    buffer: frontier.dispatch.buffer,
    byteOffset: frontier.dispatch.byteOffset
  };
  const dispatchLayout = {x: frontierWorkgroups, y: 1, z: 1};
  const workgroupSize = SHORTEST_PATH_WORKGROUP_SIZE;
  const operation = 'GPUGraphAlgorithmsSingleSourceShortestPath';

  // Clear the per-vertex round stamps and the frontier counters for this execution.
  addShortestPathPass(commandGraph, {
    operation,
    workgroupSize,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-frontier-initialize`,
    source: /* wgsl */ `
const VERTEX_COUNT: u32 = ${vertexCount}u;
const STAMPS_OFFSET: u32 = ${getViewElementOffset(frontier.stamps)}u;
const COUNTS_OFFSET: u32 = ${getViewElementOffset(frontier.counts)}u;
const DISPATCH_OFFSET: u32 = ${getViewElementOffset(frontier.dispatch)}u;
@group(0) @binding(0) var<storage, read_write> stamps: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch: array<u32>;
@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getInvocationIndexSource(state, vertexCount)}
  if (index == 0u) {
    counts[COUNTS_OFFSET] = 0u;
    counts[COUNTS_OFFSET + 1u] = 0u;
    dispatch[DISPATCH_OFFSET] = 0u;
    dispatch[DISPATCH_OFFSET + 1u] = 1u;
    dispatch[DISPATCH_OFFSET + 2u] = 1u;
  }
  if (index < VERTEX_COUNT) { stamps[STAMPS_OFFSET + index] = 0u; }
}`,
    bindings: {
      stamps: {view: frontier.stamps, usage: 'storage-write'},
      counts: {view: frontier.counts, usage: 'storage-write'},
      dispatch: {view: frontier.dispatch, usage: 'storage-write'}
    },
    dispatchLayout: getDispatchLayout(state, vertexCount)
  });
  addSourcePass(commandGraph, state);

  // Seed round zero with the source once weights and adjacency are known to be valid.
  const triviallyConverged = state.vertexCount <= 1 || state.edgeCount === 0;
  const seedBindings: Record<string, ShortestPathBinding> = {
    list: {view: frontier.lists[0], usage: 'storage-write'},
    listDistances: {view: frontier.distances[0], usage: 'storage-write'},
    counts: {view: frontier.counts, usage: 'storage-write'},
    dispatch: {view: frontier.dispatch, usage: 'storage-write'},
    invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read'},
    ...getOverflowBindings(state)
  };
  addShortestPathPass(commandGraph, {
    operation,
    workgroupSize: 1,
    invocationCount: 1,
    id: `${state.id}-frontier-seed`,
    source: /* wgsl */ `
const SOURCE_VERTEX: u32 = ${state.sourceVertex}u;
const LIST_OFFSET: u32 = ${getViewElementOffset(frontier.lists[0])}u;
const LIST_DISTANCES_OFFSET: u32 = ${getViewElementOffset(frontier.distances[0])}u;
const COUNTS_OFFSET: u32 = ${getViewElementOffset(frontier.counts)}u;
const DISPATCH_OFFSET: u32 = ${getViewElementOffset(frontier.dispatch)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getOverflowConstants(state)}
${getBindingDeclarations(seedBindings)}
@compute @workgroup_size(1)
fn main() {
  if (${triviallyConverged ? 'true' : 'false'} || invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET] != 0u ||
      ${getOverflowGuard(state)}) { return; }
  list[LIST_OFFSET] = SOURCE_VERTEX;
  listDistances[LIST_DISTANCES_OFFSET] = 0u;
  counts[COUNTS_OFFSET] = 1u;
  dispatch[DISPATCH_OFFSET] = 1u;
}`,
    bindings: seedBindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });

  const adjacencies: {
    adjacency: ImportedShortestPathAdjacency;
    direction: 'outgoing' | 'incoming';
  }[] = [
    {
      adjacency: state.primaryAdjacency,
      direction: useIncoming ? ('incoming' as const) : ('outgoing' as const)
    },
    ...(state.secondaryAdjacency
      ? [{adjacency: state.secondaryAdjacency, direction: 'incoming' as const}]
      : [])
  ];
  for (let iteration = 0; iteration < state.maxIterations; iteration++) {
    const current = iteration % 2;
    const next = 1 - current;
    const roundLabel = `${state.id}-iteration-${iteration}`;
    const countsSource = `const COUNTS_OFFSET: u32 = ${getViewElementOffset(frontier.counts)}u;`;
    withGPUGraphAlgorithmsIterationScope(
      commandGraph,
      {
        loopId: `${state.id}-iterations`,
        iteration,
        maximumIterations: state.maxIterations,
        gated: true
      },
      () => {
        // Record the round-start distance of every frontier vertex (the exact-hop snapshot).
        const snapshotBindings: Record<string, ShortestPathBinding> = {
          list: {view: frontier.lists[current], usage: 'storage-read'},
          listDistances: {view: frontier.distances[current], usage: 'storage-write'},
          counts: {view: frontier.counts, usage: 'storage-read'},
          distances: {view: state.distances, usage: 'storage-read-write', atomic: true}
        };
        addShortestPathPass(commandGraph, {
          operation,
          workgroupSize,
          datum: {keySpace: 'vertex'},
          id: `${roundLabel}-frontier-snapshot`,
          source: /* wgsl */ `
${countsSource}
const LIST_OFFSET: u32 = ${getViewElementOffset(frontier.lists[current])}u;
const LIST_DISTANCES_OFFSET: u32 = ${getViewElementOffset(frontier.distances[current])}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
${getBindingDeclarations(snapshotBindings)}
@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${FRONTIER_INDEX_SOURCE}
  if (index >= counts[COUNTS_OFFSET + ${current}u]) { return; }
  listDistances[LIST_DISTANCES_OFFSET + index] =
    atomicLoad(&distances[DISTANCES_OFFSET + list[LIST_OFFSET + index]]);
}`,
          bindings: snapshotBindings,
          dispatchLayout,
          condition: {...condition, id: `${roundLabel}-frontier-snapshot-gate`}
        });
        for (const {adjacency, direction} of adjacencies) {
          addFrontierRelaxationPass(commandGraph, {
            state,
            frontier,
            adjacency,
            direction,
            iteration,
            current,
            next,
            dispatchLayout,
            condition: {...condition, id: `${roundLabel}-frontier-relax-${direction}-gate`}
          });
        }
        for (const {adjacency, direction} of adjacencies) {
          addFrontierPredecessorPass(commandGraph, {
            state,
            frontier,
            adjacency,
            direction,
            iteration,
            current,
            dispatchLayout,
            condition: {...condition, id: `${roundLabel}-frontier-predecessors-${direction}-gate`}
          });
        }
        const controllerBindings: Record<string, ShortestPathBinding> = {
          counts: {view: frontier.counts, usage: 'storage-read-write'},
          dispatch: {view: frontier.dispatch, usage: 'storage-write'},
          converged: {view: state.converged, usage: 'storage-read-write', atomic: true},
          invalidWeightCount: {view: state.invalidWeightCount, usage: 'storage-read'},
          ...getOverflowBindings(state)
        };
        addShortestPathPass(commandGraph, {
          operation,
          workgroupSize: 1,
          invocationCount: 1,
          id: `${roundLabel}-frontier-controller`,
          source: /* wgsl */ `
${countsSource}
const DISPATCH_OFFSET: u32 = ${getViewElementOffset(frontier.dispatch)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;
const INVALID_WEIGHT_COUNT_OFFSET: u32 = ${getViewElementOffset(state.invalidWeightCount)}u;
${getOverflowConstants(state)}
${getBindingDeclarations(controllerBindings)}
@compute @workgroup_size(1)
fn main() {
  let nextCount = counts[COUNTS_OFFSET + ${next}u];
  counts[COUNTS_OFFSET + ${current}u] = 0u;
  dispatch[DISPATCH_OFFSET] = (nextCount + ${workgroupSize - 1}u) / ${workgroupSize}u;
  dispatch[DISPATCH_OFFSET + 1u] = 1u;
  dispatch[DISPATCH_OFFSET + 2u] = 1u;
  let valid = invalidWeightCount[INVALID_WEIGHT_COUNT_OFFSET] == 0u && !(${getOverflowGuard(state)});
  // A frontier left after the simple-path bound only holds final distances, as in Bellman-Ford.
  if (valid && (nextCount == 0u${iteration + 1 >= state.vertexCount - 1 ? ' || true' : ''})) {
    atomicStore(&converged[CONVERGED_OFFSET], 1u);
  }
}`,
          bindings: controllerBindings,
          dispatchLayout: {x: 1, y: 1, z: 1}
        });
      }
    );
  }
}

type FrontierResources = {
  stamps: GraphDataView<'uint32'>;
  lists: GraphDataView<'uint32'>[];
  distances: GraphDataView<'uint32'>[];
  counts: GraphDataView<'uint32'>;
  dispatch: GraphDataView<'uint32'>;
};

type FrontierPassProps = {
  state: ImportedShortestPathState;
  frontier: FrontierResources;
  adjacency: ImportedShortestPathAdjacency;
  direction: 'outgoing' | 'incoming';
  iteration: number;
  current: number;
  dispatchLayout: GPUBoundedDispatchLayout;
  condition: GPUCommandGraphGPUIndirectCondition;
};

/** Relaxes the edges of every frontier vertex and appends improved vertices to the next list. */
function addFrontierRelaxationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: FrontierPassProps & {next: number}
): void {
  const {state, frontier, adjacency, current, next} = props;
  const bindings: Record<string, ShortestPathBinding> = {
    offsets: {view: adjacency.offsets, usage: 'storage-read'},
    neighbors: {view: adjacency.neighbors, usage: 'storage-read'},
    ...(adjacency.edgeWeights
      ? {edgeWeights: {view: adjacency.edgeWeights, usage: 'storage-read' as const}}
      : {}),
    list: {view: frontier.lists[current], usage: 'storage-read'},
    listDistances: {view: frontier.distances[current], usage: 'storage-read'},
    nextList: {view: frontier.lists[next], usage: 'storage-read-write'},
    stamps: {view: frontier.stamps, usage: 'storage-read-write', atomic: true},
    counts: {view: frontier.counts, usage: 'storage-read-write', atomic: true},
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-write', atomic: true}
  };
  const weightOffset = adjacency.edgeWeights
    ? `const WEIGHTS_OFFSET: u32 = ${getViewElementOffset(adjacency.edgeWeights)}u;`
    : '';
  const weight = adjacency.edgeWeights ? 'edgeWeights[WEIGHTS_OFFSET + slot]' : '1.0';
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    id: `${state.id}-iteration-${props.iteration}-frontier-relax-${props.direction}`,
    source: /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const CAPACITY: u32 = ${adjacency.neighbors.length}u;
const ROUND: u32 = ${props.iteration + 1}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(adjacency.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(adjacency.neighbors)}u;
const LIST_OFFSET: u32 = ${getViewElementOffset(frontier.lists[current])}u;
const LIST_DISTANCES_OFFSET: u32 = ${getViewElementOffset(frontier.distances[current])}u;
const NEXT_LIST_OFFSET: u32 = ${getViewElementOffset(frontier.lists[next])}u;
const STAMPS_OFFSET: u32 = ${getViewElementOffset(frontier.stamps)}u;
const COUNTS_OFFSET: u32 = ${getViewElementOffset(frontier.counts)}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
${weightOffset}
${getBindingDeclarations(bindings)}
@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${FRONTIER_INDEX_SOURCE}
  if (index >= atomicLoad(&counts[COUNTS_OFFSET + ${current}u])) { return; }
  let vertex = list[LIST_OFFSET + index];
  let distance = bitcast<f32>(listDistances[LIST_DISTANCES_OFFSET + index]);
  let first = min(offsets[OFFSETS_OFFSET + vertex], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + vertex + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let candidateBits = bitcast<u32>(distance + ${weight});
    if (candidateBits >= 0x7f800000u) { continue; }
    let previous = atomicMin(&distances[DISTANCES_OFFSET + neighbor], candidateBits);
    if (candidateBits < previous) {
      // The better distance invalidates earlier parents; the predecessor pass reassigns them.
      atomicStore(&predecessors[PREDECESSORS_OFFSET + neighbor], ${UNREACHABLE_PREDECESSOR}u);
      if (atomicExchange(&stamps[STAMPS_OFFSET + neighbor], ROUND) != ROUND) {
        let slotIndex = atomicAdd(&counts[COUNTS_OFFSET + ${next}u], 1u);
        nextList[NEXT_LIST_OFFSET + slotIndex] = neighbor;
      }
    }
  }
}`,
    bindings,
    dispatchLayout: props.dispatchLayout,
    condition: props.condition
  });
}

/** Assigns the lowest stable parent among frontier vertices that produced each new distance. */
function addFrontierPredecessorPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: FrontierPassProps
): void {
  const {state, frontier, adjacency, current} = props;
  const bindings: Record<string, ShortestPathBinding> = {
    offsets: {view: adjacency.offsets, usage: 'storage-read'},
    neighbors: {view: adjacency.neighbors, usage: 'storage-read'},
    ...(adjacency.edgeWeights
      ? {edgeWeights: {view: adjacency.edgeWeights, usage: 'storage-read' as const}}
      : {}),
    list: {view: frontier.lists[current], usage: 'storage-read'},
    listDistances: {view: frontier.distances[current], usage: 'storage-read'},
    stamps: {view: frontier.stamps, usage: 'storage-read-write', atomic: true},
    counts: {view: frontier.counts, usage: 'storage-read-write', atomic: true},
    distances: {view: state.distances, usage: 'storage-read-write', atomic: true},
    predecessors: {view: state.predecessors, usage: 'storage-read-write', atomic: true}
  };
  const weightOffset = adjacency.edgeWeights
    ? `const WEIGHTS_OFFSET: u32 = ${getViewElementOffset(adjacency.edgeWeights)}u;`
    : '';
  const weight = adjacency.edgeWeights ? 'edgeWeights[WEIGHTS_OFFSET + slot]' : '1.0';
  addShortestPathPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsSingleSourceShortestPath',
    workgroupSize: SHORTEST_PATH_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    id: `${state.id}-iteration-${props.iteration}-frontier-predecessors-${props.direction}`,
    source: /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const SOURCE_VERTEX: u32 = ${state.sourceVertex}u;
const CAPACITY: u32 = ${adjacency.neighbors.length}u;
const ROUND: u32 = ${props.iteration + 1}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(adjacency.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(adjacency.neighbors)}u;
const LIST_OFFSET: u32 = ${getViewElementOffset(frontier.lists[current])}u;
const LIST_DISTANCES_OFFSET: u32 = ${getViewElementOffset(frontier.distances[current])}u;
const STAMPS_OFFSET: u32 = ${getViewElementOffset(frontier.stamps)}u;
const COUNTS_OFFSET: u32 = ${getViewElementOffset(frontier.counts)}u;
const DISTANCES_OFFSET: u32 = ${getViewElementOffset(state.distances)}u;
const PREDECESSORS_OFFSET: u32 = ${getViewElementOffset(state.predecessors)}u;
${weightOffset}
${getBindingDeclarations(bindings)}
@compute @workgroup_size(${SHORTEST_PATH_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${FRONTIER_INDEX_SOURCE}
  if (index >= atomicLoad(&counts[COUNTS_OFFSET + ${current}u])) { return; }
  let vertex = list[LIST_OFFSET + index];
  let distance = bitcast<f32>(listDistances[LIST_DISTANCES_OFFSET + index]);
  let first = min(offsets[OFFSETS_OFFSET + vertex], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + vertex + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == SOURCE_VERTEX) { continue; }
    if (atomicLoad(&stamps[STAMPS_OFFSET + neighbor]) != ROUND) { continue; }
    if (bitcast<u32>(distance + ${weight}) == atomicLoad(&distances[DISTANCES_OFFSET + neighbor])) {
      atomicMin(&predecessors[PREDECESSORS_OFFSET + neighbor], vertex);
    }
  }
}`,
    bindings,
    dispatchLayout: props.dispatchLayout,
    condition: props.condition
  });
}
