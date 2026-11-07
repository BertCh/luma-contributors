// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {type Binding} from '@luma.gl/core';
import type {GPUComputationDatumBinding} from '../computation-record/computation-record-types';
import {Kernel} from '../execution/compute/kernel';
import type {
  GPUCommandGraph,
  GPUCommandGraphGPUIndirectCondition,
  GraphBufferUse,
  GraphDataView
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
  createGPUGraphAlgorithmsIterationGate,
  type GPUGraphAlgorithmsIterationGate
} from './gpu-graph-iteration-gate';
import {
  getGPUGraphAlgorithmsPassAnnotation,
  withGPUGraphAlgorithmsIterationScope
} from './gpu-graph-computation-record';
import type {GPUGraphAlgorithmsLabelPropagation} from './gpu-graph-label-propagation';
import {
  importGPUGraphAlgorithmsColumn,
  importGPUGraphAlgorithmsOverflow,
  usesGPUGraphAlgorithmsViews,
  validateDistinctGPUGraphAlgorithmsHandles,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';

const LABEL_PROPAGATION_WORKGROUP_SIZE = 256;
const INVALID_COMMUNITY = 0xffffffff;

type ImportedLabelPropagation = {
  id: string;
  vertexCount: number;
  forwardOffsets: GraphDataView<'uint32'>;
  forwardNeighbors: GraphDataView<'uint32'>;
  forwardOverflow: GraphDataView<'uint32'>;
  reverseOffsets?: GraphDataView<'uint32'>;
  reverseNeighbors?: GraphDataView<'uint32'>;
  reverseOverflow?: GraphDataView<'uint32'>;
  output: GraphDataView<'uint32'>;
  scratch?: GraphDataView<'uint32'>;
  converged?: GraphDataView<'uint32'>;
  maxComputeWorkgroupsPerDimension: number;
  /** Zeroes later iterations once an iteration relabels no vertex. */
  gate?: GPUGraphAlgorithmsIterationGate;
};

type LabelPropagationBinding = {
  view: GraphDataView<'uint32'>;
  usage: GraphBufferUse['usage'];
  atomic?: boolean;
};

type LabelPropagationPassProps = {
  operation: string;
  datum?: GPUComputationDatumBinding;
  invocationCount?: number;
  workgroupSize: number;
  id: string;
  source: string;
  bindings: Record<string, LabelPropagationBinding>;
  dispatchLayout: GPUBoundedDispatchLayout;
  condition?: GPUCommandGraphGPUIndirectCondition;
};

/** Adds synchronous deterministic neighborhood-majority community propagation. @internal */
export function addGPUGraphAlgorithmsLabelPropagationToGraphWithDispatchLimit<Parameters>(
  propagation: GPUGraphAlgorithmsLabelPropagation<
    GPUGraphAlgorithmsTopologyLike,
    GPUGraphAlgorithmsColumn<'uint32'>
  >,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  const vertexCount = propagation.topology.graph.vertexCount;
  if (vertexCount === 0 && !propagation.converged) return;

  const reverse = propagation.topology.graph.directed ? propagation.topology.reverse : undefined;
  const state: ImportedLabelPropagation = {
    id: propagation.id,
    vertexCount,
    forwardOffsets: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${propagation.id}-forward-offsets`,
      propagation.topology.forward.offsets
    ),
    forwardNeighbors: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${propagation.id}-forward-neighbors`,
      propagation.topology.forward.neighbors
    ),
    forwardOverflow: importGPUGraphAlgorithmsOverflow(
      commandGraph,
      `${propagation.id}-forward-overflow`,
      propagation.topology.forward.overflow,
      'GPUGraphAlgorithmsLabelPropagation'
    ),
    ...(reverse
      ? {
          reverseOffsets: importGPUGraphAlgorithmsColumn(
            commandGraph,
            `${propagation.id}-reverse-offsets`,
            reverse.offsets
          ),
          reverseNeighbors: importGPUGraphAlgorithmsColumn(
            commandGraph,
            `${propagation.id}-reverse-neighbors`,
            reverse.neighbors
          ),
          reverseOverflow: importGPUGraphAlgorithmsOverflow(
            commandGraph,
            `${propagation.id}-reverse-overflow`,
            reverse.overflow,
            'GPUGraphAlgorithmsLabelPropagation'
          )
        }
      : {}),
    output: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${propagation.id}-output`,
      propagation.output
    ),
    ...(propagation.converged
      ? {
          converged: importGPUGraphAlgorithmsColumn(
            commandGraph,
            `${propagation.id}-converged`,
            propagation.converged
          )
        }
      : vertexCount > 0
        ? // Iteration gating needs a status word even when the caller does not request one.
          {
            converged: createTransientView(
              commandGraph,
              `${propagation.id}-converged-scratch`,
              'uint32',
              1
            )
          }
        : {}),
    ...(vertexCount > 0
      ? {
          scratch: createTransientView(
            commandGraph,
            `${propagation.id}-next-labels`,
            'uint32',
            vertexCount
          )
        }
      : {}),
    maxComputeWorkgroupsPerDimension
  };
  if (
    usesGPUGraphAlgorithmsViews(propagation.topology, [propagation.output, propagation.converged])
  ) {
    validateDistinctGPUGraphAlgorithmsHandles(
      propagation.id,
      [
        state.forwardOffsets,
        state.forwardNeighbors,
        state.forwardOverflow,
        ...(state.reverseOffsets
          ? [state.reverseOffsets, state.reverseNeighbors!, state.reverseOverflow!]
          : [])
      ],
      [
        {name: 'output', view: state.output},
        ...(state.converged ? [{name: 'converged', view: state.converged}] : [])
      ]
    );
  }

  if (vertexCount > 0 && propagation.iterations > 0) {
    state.gate = createGPUGraphAlgorithmsIterationGate(commandGraph, {
      id: `${state.id}-iteration-gate`,
      operation: 'GPUGraphAlgorithmsLabelPropagation',
      shapes: [
        [vertexCount, LABEL_PROPAGATION_WORKGROUP_SIZE],
        [1, LABEL_PROPAGATION_WORKGROUP_SIZE]
      ]
    });
  }
  addInitializationPass(commandGraph, state);
  if (vertexCount === 0) return;

  for (let iteration = 0; iteration < propagation.iterations; iteration++) {
    withGPUGraphAlgorithmsIterationScope(
      commandGraph,
      {
        loopId: `${state.id}-iterations`,
        iteration,
        maximumIterations: propagation.iterations,
        gated: Boolean(state.gate)
      },
      () => {
        addConvergenceResetPass(commandGraph, {state, iteration});
        addVotingPass(commandGraph, {state, iteration});
        addPublishPass(commandGraph, {state, iteration});
        if (state.gate) addGateUpdatePass(commandGraph, {state, iteration});
      }
    );
  }
}

/** Initializes one stable identity label per vertex or publishes explicit overflow sentinels. */
function addInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedLabelPropagation
): void {
  const bindings: Record<string, LabelPropagationBinding> = {
    output: {view: state.output, usage: 'storage-write'},
    forwardOverflow: {view: state.forwardOverflow, usage: 'storage-read'},
    ...(state.reverseOverflow
      ? {reverseOverflow: {view: state.reverseOverflow, usage: 'storage-read'}}
      : {}),
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-write', atomic: true}}
      : {}),
    ...(state.gate ? {dispatch: {view: state.gate.view, usage: 'storage-write' as const}} : {})
  };
  const reverseOffset = state.reverseOverflow
    ? `const REVERSE_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.reverseOverflow)}u;`
    : '';
  const reverseOverflow = state.reverseOverflow
    ? ' || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u'
    : '';
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const initializeConvergence = state.converged
    ? `if (index == 0u) {
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, VERTEX_COUNT == 0u && !hasOverflow));
    ${state.gate?.armWGSL ?? ''}
  }`
    : '';
  const dispatchLayout = getLabelPropagationDispatchLayout(state, Math.max(state.vertexCount, 1));
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.forwardOverflow)}u;
${reverseOffset}
${convergenceOffset}
${state.gate ? `const dispatchOffset: u32 = ${getViewElementOffset(state.gate.view)}u;` : ''}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${LABEL_PROPAGATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LABEL_PROPAGATION_WORKGROUP_SIZE)}
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${reverseOverflow};
  if (index < VERTEX_COUNT) {
    output[OUTPUT_OFFSET + index] = select(index, ${INVALID_COMMUNITY}u, hasOverflow);
  }
  ${initializeConvergence}
}`;
  addLabelPropagationPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsLabelPropagation',
    workgroupSize: LABEL_PROPAGATION_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-initialize`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Publishes a fresh optimistic convergence scalar after a globally synchronized boundary. */
function addConvergenceResetPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedLabelPropagation; iteration: number}
): void {
  const {state} = props;
  const converged = state.converged!;
  const bindings: Record<string, LabelPropagationBinding> = {
    forwardOverflow: {view: state.forwardOverflow, usage: 'storage-read'},
    ...(state.reverseOverflow
      ? {reverseOverflow: {view: state.reverseOverflow, usage: 'storage-read'}}
      : {}),
    converged: {view: converged, usage: 'storage-write', atomic: true}
  };
  const reverseOffset = state.reverseOverflow
    ? `const REVERSE_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.reverseOverflow)}u;`
    : '';
  const reverseOverflow = state.reverseOverflow
    ? ' || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u'
    : '';
  const dispatchLayout = getLabelPropagationDispatchLayout(state, 1);
  const source = /* wgsl */ `
const FORWARD_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.forwardOverflow)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(converged)}u;
${reverseOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${LABEL_PROPAGATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LABEL_PROPAGATION_WORKGROUP_SIZE)}
  if (index != 0u) { return; }
  let hasOverflow = forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${reverseOverflow};
  atomicStore(&converged[CONVERGED_OFFSET], select(1u, 0u, hasOverflow));
}`;
  addLabelPropagationPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsLabelPropagation',
    workgroupSize: LABEL_PROPAGATION_WORKGROUP_SIZE,
    invocationCount: 1,
    id: `${state.id}-iteration-${props.iteration}-reset`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-reset`, 1)
  });
}

/** Counts every weak-neighbor occurrence from one immutable label snapshot, using eight bindings. */
function addVotingPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedLabelPropagation; iteration: number}
): void {
  const {state} = props;
  const scratch = state.scratch!;
  const bindings: Record<string, LabelPropagationBinding> = {
    output: {view: state.output, usage: 'storage-read'},
    scratch: {view: scratch, usage: 'storage-write'},
    forwardOffsets: {view: state.forwardOffsets, usage: 'storage-read'},
    forwardNeighbors: {view: state.forwardNeighbors, usage: 'storage-read'},
    forwardOverflow: {view: state.forwardOverflow, usage: 'storage-read'},
    ...(state.reverseOffsets && state.reverseNeighbors && state.reverseOverflow
      ? {
          reverseOffsets: {view: state.reverseOffsets, usage: 'storage-read'},
          reverseNeighbors: {view: state.reverseNeighbors, usage: 'storage-read'},
          reverseOverflow: {view: state.reverseOverflow, usage: 'storage-read'}
        }
      : {})
  };
  const hasReverse = Boolean(
    state.reverseOffsets && state.reverseNeighbors && state.reverseOverflow
  );
  const reverseConstants = hasReverse
    ? `const REVERSE_CAPACITY: u32 = ${state.reverseNeighbors!.length}u;
const REVERSE_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.reverseOffsets!)}u;
const REVERSE_NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(state.reverseNeighbors!)}u;
const REVERSE_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.reverseOverflow!)}u;`
    : '';
  const reverseOverflow = hasReverse ? ' || reverseOverflow[REVERSE_OVERFLOW_OFFSET] != 0u' : '';
  const countReverseVotes = hasReverse
    ? `let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + vertex + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT && neighbor != vertex && output[OUTPUT_OFFSET + neighbor] == candidate) {
      votes++;
    }
  }`
    : '';
  const selectReverseCandidates = hasReverse
    ? `let reverseFirst = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index], REVERSE_CAPACITY);
  let reverseLast = min(reverseOffsets[REVERSE_OFFSETS_OFFSET + index + 1u], REVERSE_CAPACITY);
  for (var slot = reverseFirst; slot < reverseLast; slot++) {
    let neighbor = reverseNeighbors[REVERSE_NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    let candidate = output[OUTPUT_OFFSET + neighbor];
    let votes = countCandidateVotes(index, candidate);
    if (votes > selectedVotes || (votes == selectedVotes && candidate < selectedLabel)) {
      selectedLabel = candidate;
      selectedVotes = votes;
    }
  }`
    : '';
  const dispatchLayout = getLabelPropagationDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const FORWARD_CAPACITY: u32 = ${state.forwardNeighbors.length}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const SCRATCH_OFFSET: u32 = ${getViewElementOffset(scratch)}u;
const FORWARD_OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.forwardOffsets)}u;
const FORWARD_NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(state.forwardNeighbors)}u;
const FORWARD_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.forwardOverflow)}u;
${reverseConstants}
${getBindingDeclarations(bindings)}

fn countCandidateVotes(vertex: u32, candidate: u32) -> u32 {
  var votes = select(0u, 1u, output[OUTPUT_OFFSET + vertex] == candidate);
  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + vertex + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor < VERTEX_COUNT && neighbor != vertex && output[OUTPUT_OFFSET + neighbor] == candidate) {
      votes++;
    }
  }
  ${countReverseVotes}
  return votes;
}

@compute @workgroup_size(${LABEL_PROPAGATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LABEL_PROPAGATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  if (forwardOverflow[FORWARD_OVERFLOW_OFFSET] != 0u${reverseOverflow}) {
    scratch[SCRATCH_OFFSET + index] = ${INVALID_COMMUNITY}u;
    return;
  }

  var selectedLabel = output[OUTPUT_OFFSET + index];
  var selectedVotes = countCandidateVotes(index, selectedLabel);
  let first = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index], FORWARD_CAPACITY);
  let last = min(forwardOffsets[FORWARD_OFFSETS_OFFSET + index + 1u], FORWARD_CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = forwardNeighbors[FORWARD_NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    let candidate = output[OUTPUT_OFFSET + neighbor];
    let votes = countCandidateVotes(index, candidate);
    if (votes > selectedVotes || (votes == selectedVotes && candidate < selectedLabel)) {
      selectedLabel = candidate;
      selectedVotes = votes;
    }
  }
  ${selectReverseCandidates}
  scratch[SCRATCH_OFFSET + index] = selectedLabel;
}`;
  addLabelPropagationPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsLabelPropagation',
    workgroupSize: LABEL_PROPAGATION_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-vote`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-vote`, 0)
  });
}

/** Publishes one globally synchronized label snapshot and atomically marks actual changes. */
function addPublishPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedLabelPropagation; iteration: number}
): void {
  const {state} = props;
  const scratch = state.scratch!;
  const bindings: Record<string, LabelPropagationBinding> = {
    output: {view: state.output, usage: 'storage-read-write'},
    scratch: {view: scratch, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const markChanged = state.converged
    ? `if (nextLabel != previousLabel) { atomicStore(&converged[CONVERGED_OFFSET], 0u); }`
    : '';
  const dispatchLayout = getLabelPropagationDispatchLayout(state, state.vertexCount);
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const SCRATCH_OFFSET: u32 = ${getViewElementOffset(scratch)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${LABEL_PROPAGATION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, LABEL_PROPAGATION_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  let previousLabel = output[OUTPUT_OFFSET + index];
  let nextLabel = scratch[SCRATCH_OFFSET + index];
  output[OUTPUT_OFFSET + index] = nextLabel;
  ${markChanged}
}`;
  addLabelPropagationPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsLabelPropagation',
    workgroupSize: LABEL_PROPAGATION_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-publish`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-publish`, 0)
  });
}

/** Zeroes every later iteration once this iteration relabeled no vertex (a fixed point). */
function addGateUpdatePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedLabelPropagation; iteration: number}
): void {
  const {state} = props;
  const gate = state.gate!;
  const converged = state.converged!;
  const bindings: Record<string, LabelPropagationBinding> = {
    converged: {view: converged, usage: 'storage-read'},
    dispatch: {view: gate.view, usage: 'storage-write'}
  };
  const source = /* wgsl */ `
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(converged)}u;
const dispatchOffset: u32 = ${getViewElementOffset(gate.view)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(1)
fn main() {
  if (converged[CONVERGED_OFFSET] != 0u) {
    ${gate.disarmWGSL}
  }
}`;
  addLabelPropagationPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsLabelPropagation',
    workgroupSize: 1,
    invocationCount: 1,
    id: `${state.id}-iteration-${props.iteration}-gate`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Declares packed scalar and optional atomic convergence views in exact binding order. */
function getBindingDeclarations(bindings: Record<string, LabelPropagationBinding>): string {
  return Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      const element = binding.atomic ? 'atomic<u32>' : 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');
}

/** Compiles one graph-owned bounded community pass without submission or CPU synchronization. */
function addLabelPropagationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: LabelPropagationPassProps
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

function getLabelPropagationDispatchLayout(
  state: ImportedLabelPropagation,
  elementCount: number
): GPUBoundedDispatchLayout {
  return getGPUGraphAlgorithmsLabelPropagationDispatchLayout(
    elementCount,
    state.maxComputeWorkgroupsPerDimension
  );
}

/** Plans bounded true three-dimensional community-label or convergence dispatch. @internal */
export function getGPUGraphAlgorithmsLabelPropagationDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphAlgorithmsLabelPropagation',
    elementCount,
    LABEL_PROPAGATION_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
