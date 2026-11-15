// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  type GPUNetworkAdjacency,
  validateGPUNetworkTopology,
  getGPUNetworkTopologyViews
} from '../../analysis-core/network-topology';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Default compiled state-machine steps; deep graphs may need a larger budget. */
export const GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS = 256;
/** Six nodes per step plus initialization and publication fit the shared 32768-node budget. */
export const GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS = 4096;
/** Reserve 0xffffffff for unresolved labels and keep vertex/CSR indexing within uint32. */
export const GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT = 0xfffffffe;

const OPERATION = 'GPUNetworkStronglyConnectedComponents';
const WORKGROUP_SIZE = 256;
const INVALID_COMPONENT = 0xffffffff;
const PHASE_TRIM = 0;
const PHASE_COLOR = 1;
const PHASE_BACK = 2;
const PHASE_FINALIZE = 3;
const PHASE_DONE = 4;
/** [phase, changed, remaining, executed steps]. */
const CONTROL_LENGTH = 4;
const FLAG_HAS_INPUT = 1;
const FLAG_HAS_OUTPUT = 2;
const FLAG_MARKED = 4;

export type GPUNetworkStronglyConnectedComponentsOutput = {
  /** Minimum vertex ID of each SCC; every label is 0xffffffff if the budget is exhausted. */
  strongComponents: GraphDataView<'uint32'>;
  /** All three status words are required, with one uint32 row each. */
  status: Required<GPUIterationStatusPort>;
};

export type GPUNetworkStronglyConnectedComponentsProps = {
  id?: string;
  /** Forward CSR only; weights are ignored. Offsets must be monotone and within slot capacity. */
  topology: GPUNetworkAdjacency;
  output: GPUNetworkStronglyConnectedComponentsOutput;
  /** Compiled state-machine steps (1–4096), default 256. */
  maximumIterations?: number;
};

/**
 * Exact directed SCCs using O(V) scratch: trim singleton vertices, propagate minimum colours,
 * mark backwards within colour classes, then finalize the marked components. Vertex kernels
 * visit only their outgoing CSR neighbourhood. No reverse CSR, readback or V-by-V table is used.
 *
 * All props are compile-time (buffer shapes and the step budget); there are no numeric per-frame
 * parameters. GPU dispatch gates select the active phase and disable vertex work after convergence.
 * Long chains and deep cycles can need one step per level; chains of SCCs require repeated phases.
 * A budget-exhausted run sets converged=0 and iterationLimitReached=1 and invalidates ALL labels,
 * including resolved singleton labels. iterationCount counts executed state-machine steps.
 *
 * The independent iterative Tarjan oracle is computeNetworkStructureOnCPU. It accepts the same
 * CSR and returns the same minimum-ID labels without the GPU step budget. Adapted from the user's
 * luma.gl fork e0ed33791 (trim/colour/backward-mark domain algorithm).
 */
export class GPUNetworkStronglyConnectedComponents implements GPUCommandNodeProducer {
  readonly id: string;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly maximumIterations: number;
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPUNetworkStronglyConnectedComponentsProps) {
    this.id = props.id ?? 'network-strongly-connected-components';
    const {id} = this;
    const shape = validateGPUNetworkTopology(id, props.topology);
    this.nodeCount = shape.nodeCount;
    this.edgeCount = shape.edgeCount;
    this.maximumIterations =
      props.maximumIterations ?? GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS;
    validateCapacity(
      id,
      'nodeCount',
      this.nodeCount,
      1,
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT
    );
    validateCapacity(
      id,
      'edgeCount',
      this.edgeCount,
      0,
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_COUNT
    );
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS
    );
    this.estimatedCommandNodeCount = 2 + 6 * this.maximumIterations;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const outputs = [props.output.strongComponents, ...Object.values(props.output.status)];
    for (const [name, view] of Object.entries({
      strongComponents: props.output.strongComponents,
      converged: props.output.status.converged,
      iterationCount: props.output.status.iterationCount,
      iterationLimitReached: props.output.status.iterationLimitReached
    })) {
      validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
      const length = name === 'strongComponents' ? this.nodeCount : 1;
      validateRequirement(
        id,
        view.length === length,
        `output.${name} must hold exactly ${length} uint32 rows`
      );
    }
    validateRequirement(
      id,
      new Set(outputs.map(view => view.buffer)).size === outputs.length,
      'output strongComponents and status must use distinct buffers'
    );
    validateGraphOutputsDisjointFromInputs(id, outputs, getGPUNetworkTopologyViews(props.topology));
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount: vertexCount, edgeCount} = this;
    const output = props.output.strongComponents;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      output,
      ...Object.values(props.output.status)
    ]);
    // Account for the prefix of a view within its aligned storage binding, as well as scratch.
    for (const [name, view] of Object.entries({
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      strongComponents: output,
      ...props.output.status
    })) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        `topology/output.${name}`,
        getViewBindingRange(view).size,
        'reduce nodeCount or edgeCount to fit the device storage binding limit'
      );
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'color/flags scratch',
      vertexCount * 4,
      'reduce nodeCount to fit the device storage binding limit'
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const color = createTransientView(graph, `${id}-color`, 'uint32', vertexCount);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', vertexCount);
    const control = createTransientView(graph, `${id}-control`, 'uint32', CONTROL_LENGTH);
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-gate`,
      operation: OPERATION,
      shapes: [
        [vertexCount, WORKGROUP_SIZE],
        [vertexCount, WORKGROUP_SIZE],
        [vertexCount, WORKGROUP_SIZE],
        [vertexCount, WORKGROUP_SIZE]
      ]
    });
    // Four fixed phase gates; emit their assignments without a serial data-sized loop.
    const selectPhaseWGSL =
      gate.armWGSL +
      '\n' +
      [PHASE_TRIM, PHASE_COLOR, PHASE_BACK, PHASE_FINALIZE]
        .map(phase => `if (next != ${phase}u) { dispatch[dispatchOffset + ${phase * 3}u] = 0u; }`)
        .join('\n');
    const declarations = `const VERTEX_COUNT: u32 = ${vertexCount}u;
const CAPACITY: u32 = ${edgeCount}u;
const INVALID: u32 = ${INVALID_COMPONENT}u;
const FLAG_HAS_INPUT: u32 = ${FLAG_HAS_INPUT}u;
const FLAG_HAS_OUTPUT: u32 = ${FLAG_HAS_OUTPUT}u;
const FLAG_MARKED: u32 = ${FLAG_MARKED}u;`;
    const adjacencyBindings = [
      {name: 'offsets', view: props.topology.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.topology.neighbors, type: 'u32', access: 'read'}
    ] as const;
    const labelBinding = {
      name: 'output',
      view: output,
      type: 'atomic<u32>',
      access: 'read_write'
    } as const;
    const colorBinding = {
      name: 'color',
      view: color,
      type: 'atomic<u32>',
      access: 'read_write'
    } as const;
    const flagsBinding = {
      name: 'flags',
      view: flags,
      type: 'atomic<u32>',
      access: 'read_write'
    } as const;
    const controlBinding = {
      name: 'control',
      view: control,
      type: 'atomic<u32>',
      access: 'read_write'
    } as const;
    const edgeRange = `let first = min(offsets[offsetsOffset + index], CAPACITY);
  let last = min(offsets[offsetsOffset + index + 1u], CAPACITY);`;
    const activeGuard = 'if (atomicLoad(&output[outputOffset + index]) != INVALID) { return; }';

    const addVertexNode = (
      nodeId: string,
      variant: string,
      slot: number | undefined,
      bindings: readonly WGSLKernelBinding[],
      body: string
    ) => {
      const gateProps = slot === undefined ? undefined : gate.getGate(slot);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: nodeId,
          operation: OPERATION,
          variant,
          invocationCount: vertexCount,
          workgroupSize: WORKGROUP_SIZE,
          bindings,
          declarations,
          body,
          ...(gateProps && {
            condition: {...gateProps.condition, id: `${nodeId}-gate`},
            extraResources: gateProps.extraResources
          })
        })
      );
    };

    addVertexNode(
      `${id}-initialize`,
      'initialize',
      undefined,
      [
        labelBinding,
        colorBinding,
        flagsBinding,
        controlBinding,
        {name: 'dispatch', view: gate.view, type: 'u32', access: 'read_write'}
      ],
      `atomicStore(&output[outputOffset + index], INVALID);
  atomicStore(&color[colorOffset + index], index);
  atomicStore(&flags[flagsOffset + index], 0u);
  if (index == 0u) {
    atomicStore(&control[controlOffset], ${PHASE_TRIM}u);
    atomicStore(&control[controlOffset + 1u], 0u);
    atomicStore(&control[controlOffset + 2u], 0u);
    atomicStore(&control[controlOffset + 3u], 0u);
    let next = ${PHASE_TRIM}u;
    ${selectPhaseWGSL}
  }`
    );

    for (let iteration = 0; iteration < this.maximumIterations; iteration++) {
      const prefix = `${id}-iteration-${iteration}`;
      // Trim: flag active vertices that still have an active in- or out-neighbour.
      addVertexNode(
        `${prefix}-trim-mark`,
        'trim-mark',
        PHASE_TRIM,
        [...adjacencyBindings, labelBinding, flagsBinding],
        `${activeGuard}
  ${edgeRange}
  var hasOutput = false;
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    if (atomicLoad(&output[outputOffset + neighbor]) != INVALID) { continue; }
    hasOutput = true;
    atomicOr(&flags[flagsOffset + neighbor], FLAG_HAS_INPUT);
  }
  if (hasOutput) { atomicOr(&flags[flagsOffset + index], FLAG_HAS_OUTPUT); }`
      );
      // Vertices without both an active in- and out-neighbour are singleton components.
      addVertexNode(
        `${prefix}-trim-remove`,
        'trim-remove',
        PHASE_TRIM,
        [labelBinding, flagsBinding, controlBinding],
        `${activeGuard}
  let previous = atomicAnd(&flags[flagsOffset + index], ~(FLAG_HAS_INPUT | FLAG_HAS_OUTPUT));
  if ((previous & (FLAG_HAS_INPUT | FLAG_HAS_OUTPUT)) != (FLAG_HAS_INPUT | FLAG_HAS_OUTPUT)) {
    atomicStore(&output[outputOffset + index], index);
    atomicStore(&control[controlOffset + 1u], 1u);
  } else {
    atomicAdd(&control[controlOffset + 2u], 1u);
  }`
      );
      // Colour: push the minimum colour along forward edges between active vertices.
      addVertexNode(
        `${prefix}-color`,
        'color',
        PHASE_COLOR,
        [...adjacencyBindings, labelBinding, colorBinding, controlBinding],
        `${activeGuard}
  ${edgeRange}
  let vertexColor = atomicLoad(&color[colorOffset + index]);
  for (var slot = first; slot < last; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    if (atomicLoad(&output[outputOffset + neighbor]) != INVALID) { continue; }
    if (atomicMin(&color[colorOffset + neighbor], vertexColor) > vertexColor) {
      atomicStore(&control[controlOffset + 1u], 1u);
    }
  }`
      );
      // Backward mark: vertices that reach their colour root inside the colour class.
      addVertexNode(
        `${prefix}-back`,
        'back',
        PHASE_BACK,
        [...adjacencyBindings, labelBinding, colorBinding, flagsBinding, controlBinding],
        `${activeGuard}
  if ((atomicLoad(&flags[flagsOffset + index]) & FLAG_MARKED) != 0u) { return; }
  ${edgeRange}
  let vertexColor = atomicLoad(&color[colorOffset + index]);
  var reachesRoot = vertexColor == index;
  for (var slot = first; slot < last && !reachesRoot; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= VERTEX_COUNT || neighbor == index) { continue; }
    reachesRoot = (atomicLoad(&flags[flagsOffset + neighbor]) & FLAG_MARKED) != 0u &&
      atomicLoad(&color[colorOffset + neighbor]) == vertexColor;
  }
  if (reachesRoot) {
    atomicOr(&flags[flagsOffset + index], FLAG_MARKED);
    atomicStore(&control[controlOffset + 1u], 1u);
  }`
      );
      // Finalize: publish marked components and reset the colours of the remaining vertices.
      addVertexNode(
        `${prefix}-finalize`,
        'finalize',
        PHASE_FINALIZE,
        [labelBinding, colorBinding, flagsBinding, controlBinding],
        `${activeGuard}
  if ((atomicLoad(&flags[flagsOffset + index]) & FLAG_MARKED) != 0u) {
    atomicStore(&output[outputOffset + index], atomicLoad(&color[colorOffset + index]));
    atomicStore(&flags[flagsOffset + index], 0u);
  } else {
    atomicStore(&color[colorOffset + index], index);
    atomicAdd(&control[controlOffset + 2u], 1u);
  }`
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${prefix}-controller`,
          operation: OPERATION,
          variant: 'controller',
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [controlBinding, gate.binding],
          body: `let phase = atomicLoad(&control[controlOffset]);
  if (phase == ${PHASE_DONE}u) { return; }
  atomicAdd(&control[controlOffset + 3u], 1u);
  let changed = atomicLoad(&control[controlOffset + 1u]);
  let remaining = atomicLoad(&control[controlOffset + 2u]);
  var next = phase;
  if (phase == ${PHASE_TRIM}u) {
    if (remaining == 0u) { next = ${PHASE_DONE}u; } else if (changed == 0u) { next = ${PHASE_COLOR}u; }
  } else if (phase == ${PHASE_COLOR}u) {
    if (changed == 0u) { next = ${PHASE_BACK}u; }
  } else if (phase == ${PHASE_BACK}u) {
    if (changed == 0u) { next = ${PHASE_FINALIZE}u; }
  } else if (phase == ${PHASE_FINALIZE}u) {
    next = select(${PHASE_TRIM}u, ${PHASE_DONE}u, remaining == 0u);
  }
  atomicStore(&control[controlOffset], next);
  atomicStore(&control[controlOffset + 1u], 0u);
  atomicStore(&control[controlOffset + 2u], 0u);
  ${selectPhaseWGSL}`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        invocationCount: vertexCount,
        workgroupSize: WORKGROUP_SIZE,
        bindings: [
          labelBinding,
          controlBinding,
          {
            name: 'converged',
            view: props.output.status.converged,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'iterationCount',
            view: props.output.status.iterationCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'iterationLimitReached',
            view: props.output.status.iterationLimitReached,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `let finished = atomicLoad(&control[controlOffset]) == ${PHASE_DONE}u;
  if (!finished) { atomicStore(&output[outputOffset + index], ${INVALID_COMPONENT}u); }
  if (index == 0u) {
    converged[convergedOffset] = select(0u, 1u, finished);
    iterationCount[iterationCountOffset] = atomicLoad(&control[controlOffset + 3u]);
    iterationLimitReached[iterationLimitReachedOffset] = select(1u, 0u, finished);
  }`
      })
    );
    return nodes;
  }
}
