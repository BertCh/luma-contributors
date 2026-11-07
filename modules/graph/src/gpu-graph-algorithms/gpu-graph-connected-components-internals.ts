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
import {createTransientView} from '../gpu-core/graph-data-view-utils';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {getViewBinding, getViewElementOffset} from '../gpu-core/graph-data-view-utils';
import {
  createGPUGraphAlgorithmsIterationGate,
  type GPUGraphAlgorithmsIterationGate
} from './gpu-graph-iteration-gate';
import {
  getGPUGraphAlgorithmsPassAnnotation,
  withGPUGraphAlgorithmsIterationScope
} from './gpu-graph-computation-record';
import type {GPUGraphAlgorithmsConnectedComponents} from './gpu-graph-connected-components';
import {
  importGPUGraphAlgorithmsColumn,
  importGPUGraphAlgorithmsOverflow,
  usesGPUGraphAlgorithmsViews,
  validateDistinctGPUGraphAlgorithmsHandles,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';

const CONNECTED_COMPONENTS_WORKGROUP_SIZE = 256;
const INVALID_COMPONENT = 0xffffffff;

type ImportedConnectedComponents = {
  id: string;
  vertexCount: number;
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  output: GraphDataView<'uint32'>;
  converged?: GraphDataView<'uint32'>;
  invalidateOnNonConvergence: boolean;
  maxComputeWorkgroupsPerDimension: number;
  /** Zeroes later iterations once an iteration leaves every label unchanged. */
  gate?: GPUGraphAlgorithmsIterationGate;
};

type ConnectedComponentsBinding = {
  view: GraphDataView<'uint32'>;
  usage: GraphBufferUse['usage'];
  atomic?: boolean;
};

type ConnectedComponentsPassProps = {
  operation: string;
  datum?: GPUComputationDatumBinding;
  invocationCount?: number;
  workgroupSize: number;
  id: string;
  source: string;
  bindings: Record<string, ConnectedComponentsBinding>;
  dispatchLayout: GPUBoundedDispatchLayout;
  condition?: GPUCommandGraphGPUIndirectCondition;
};

/** Adds bounded GPU weak-component hooking using an explicit dispatch limit. @internal */
export function addGPUGraphAlgorithmsConnectedComponentsToGraphWithDispatchLimit<Parameters>(
  components: GPUGraphAlgorithmsConnectedComponents<
    GPUGraphAlgorithmsTopologyLike,
    GPUGraphAlgorithmsColumn<'uint32'>
  >,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  if (
    components.topology.graph.vertexCount === 0 &&
    !components.converged &&
    !components.invalidateOnNonConvergence
  ) {
    return;
  }

  const state: ImportedConnectedComponents = {
    id: components.id,
    vertexCount: components.topology.graph.vertexCount,
    offsets: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${components.id}-offsets`,
      components.topology.forward.offsets
    ),
    neighbors: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${components.id}-neighbors`,
      components.topology.forward.neighbors
    ),
    overflow: importGPUGraphAlgorithmsOverflow(
      commandGraph,
      `${components.id}-overflow`,
      components.topology.forward.overflow,
      'GPUGraphAlgorithmsConnectedComponents'
    ),
    output: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${components.id}-output`,
      components.output
    ),
    ...(components.converged
      ? {
          converged: importGPUGraphAlgorithmsColumn(
            commandGraph,
            `${components.id}-converged`,
            components.converged
          )
        }
      : // Invalidation and iteration gating need a status word even when the caller does not request one.
        {
          converged: createTransientView(
            commandGraph,
            `${components.id}-converged-scratch`,
            'uint32',
            1
          )
        }),
    invalidateOnNonConvergence: components.invalidateOnNonConvergence,
    maxComputeWorkgroupsPerDimension
  };
  if (usesGPUGraphAlgorithmsViews(components.topology, [components.output, components.converged])) {
    validateDistinctGPUGraphAlgorithmsHandles(
      components.id,
      [state.offsets, state.neighbors, state.overflow],
      [
        {name: 'output', view: state.output},
        ...(state.converged ? [{name: 'converged', view: state.converged}] : [])
      ]
    );
  }

  if (state.vertexCount > 0 && components.iterations > 0) {
    state.gate = createGPUGraphAlgorithmsIterationGate(commandGraph, {
      id: `${state.id}-iteration-gate`,
      operation: 'GPUGraphAlgorithmsConnectedComponents',
      shapes: [
        [state.vertexCount, CONNECTED_COMPONENTS_WORKGROUP_SIZE],
        [1, CONNECTED_COMPONENTS_WORKGROUP_SIZE]
      ]
    });
  }
  addInitializationPass(commandGraph, state);
  if (state.vertexCount === 0) {
    return;
  }

  for (let iteration = 0; iteration < components.iterations; iteration++) {
    withGPUGraphAlgorithmsIterationScope(
      commandGraph,
      {
        loopId: `${state.id}-iterations`,
        iteration,
        maximumIterations: components.iterations,
        gated: Boolean(state.gate)
      },
      () => {
        addConvergenceResetPass(commandGraph, {state, iteration});
        addHookingPass(commandGraph, {state, iteration});
        addPointerJumpPass(commandGraph, {state, iteration});
        if (state.gate) {
          addGateUpdatePass(commandGraph, {state, iteration});
        }
      }
    );
  }
  if (state.invalidateOnNonConvergence) {
    addInvalidateNonConvergedPass(commandGraph, state);
  }
}

/** Publishes invalid labels when the last compiled iteration still changed a label. */
function addInvalidateNonConvergedPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedConnectedComponents
): void {
  const converged = state.converged!;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    output: {view: state.output, usage: 'storage-read-write', atomic: true},
    converged: {view: converged, usage: 'storage-read'}
  };
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    state.vertexCount,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(converged)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || converged[CONVERGED_OFFSET] != 0u) { return; }
  atomicStore(&output[OUTPUT_OFFSET + index], ${INVALID_COMPONENT}u);
}`;

  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-invalidate-non-converged`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Initializes identity labels, fails closed on overflow, and marks empty graphs converged. */
function addInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedConnectedComponents
): void {
  const bindings: Record<string, ConnectedComponentsBinding> = {
    output: {view: state.output, usage: 'storage-write', atomic: true},
    overflow: {view: state.overflow, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-write', atomic: true}}
      : {}),
    ...(state.gate ? {dispatch: {view: state.gate.view, usage: 'storage-write' as const}} : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const convergenceInitialization = state.converged
    ? `if (index == 0u) {
    let emptyAndValid = VERTEX_COUNT == 0u && !hasOverflow;
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, emptyAndValid));
    ${state.gate?.armWGSL ?? ''}
  }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    Math.max(state.vertexCount, 1),
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.overflow)}u;
${state.gate ? `const dispatchOffset: u32 = ${getViewElementOffset(state.gate.view)}u;` : ''}
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  let hasOverflow = overflow[OVERFLOW_OFFSET] != 0u;
  if (index < VERTEX_COUNT) {
    let component = select(index, ${INVALID_COMPONENT}u, hasOverflow);
    atomicStore(&output[OUTPUT_OFFSET + index], component);
  }
  ${convergenceInitialization}
}`;

  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-initialize`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Resets the caller-owned convergence scalar in a dedicated globally synchronized pass. */
function addConvergenceResetPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const converged = state.converged!;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    overflow: {view: state.overflow, usage: 'storage-read'},
    converged: {view: converged, usage: 'storage-write', atomic: true}
  };
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    1,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.overflow)}u;
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(converged)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index != 0u) { return; }
  atomicStore(&converged[CONVERGED_OFFSET], select(1u, 0u, overflow[OVERFLOW_OFFSET] != 0u));
}`;

  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    invocationCount: 1,
    id: `${state.id}-iteration-${props.iteration}-reset`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-reset`, 1)
  });
}

/** Hooks both endpoints of every forward edge into one order-independent weak component. */
function addHookingPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    offsets: {view: state.offsets, usage: 'storage-read'},
    neighbors: {view: state.neighbors, usage: 'storage-read'},
    output: {view: state.output, usage: 'storage-read-write', atomic: true},
    overflow: {view: state.overflow, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const markChanged = state.converged
    ? `if (previous > lower) { atomicStore(&converged[CONVERGED_OFFSET], 0u); }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    state.vertexCount,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const CAPACITY: u32 = ${state.neighbors.length}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(state.neighbors)}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.overflow)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || overflow[OVERFLOW_OFFSET] != 0u) { return; }
  let first = min(offsets[OFFSETS_OFFSET + index], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + index + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let sourceComponent = atomicLoad(&output[OUTPUT_OFFSET + index]);
    let neighborComponent = atomicLoad(&output[OUTPUT_OFFSET + neighbor]);
    let lower = min(sourceComponent, neighborComponent);
    let higher = max(sourceComponent, neighborComponent);
    if (higher == lower) { continue; }
    let previous = atomicMin(&output[OUTPUT_OFFSET + higher], lower);
    ${markChanged}
  }
}`;

  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-hook`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-hook`, 0)
  });
}

/** Shortcuts the monotone parent forest in a pass synchronized after all edge hooks. */
function addPointerJumpPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    output: {view: state.output, usage: 'storage-read-write', atomic: true},
    overflow: {view: state.overflow, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const markChanged = state.converged
    ? `if (previous > parentComponent) {
    atomicStore(&converged[CONVERGED_OFFSET], 0u);
  }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    state.vertexCount,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(state.overflow)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || overflow[OVERFLOW_OFFSET] != 0u) { return; }
  let component = atomicLoad(&output[OUTPUT_OFFSET + index]);
  let parentComponent = atomicLoad(&output[OUTPUT_OFFSET + component]);
  if (parentComponent >= component) { return; }
  let previous = atomicMin(&output[OUTPUT_OFFSET + index], parentComponent);
  ${markChanged}
}`;

  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    datum: {keySpace: 'vertex'},
    invocationCount: state.vertexCount,
    id: `${state.id}-iteration-${props.iteration}-jump`,
    source,
    bindings,
    dispatchLayout,
    condition: state.gate?.getCondition(`${state.id}-iteration-${props.iteration}-jump`, 0)
  });
}

/** Zeroes every later iteration once this iteration changed no label (a fixed point). */
function addGateUpdatePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const gate = state.gate!;
  const converged = state.converged!;
  const bindings: Record<string, ConnectedComponentsBinding> = {
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
  addConnectedComponentsPass(commandGraph, {
    operation: 'GPUGraphAlgorithmsConnectedComponents',
    workgroupSize: 1,
    invocationCount: 1,
    id: `${state.id}-iteration-${props.iteration}-gate`,
    source,
    bindings,
    dispatchLayout: {x: 1, y: 1, z: 1}
  });
}

/** Declares every storage binding in its exact generated shader-layout order. */
function getBindingDeclarations(bindings: Record<string, ConnectedComponentsBinding>): string {
  return Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      const element = binding.atomic ? 'atomic<u32>' : 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');
}

/** Compiles one bounded storage-buffer pass without hidden GPU allocation or submission. */
function addConnectedComponentsPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: ConnectedComponentsPassProps
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

/** Plans a bounded three-dimensional weak-component vertex or status dispatch. @internal */
export function getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphAlgorithmsConnectedComponents',
    elementCount,
    CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
