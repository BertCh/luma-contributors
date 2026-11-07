// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {getBoundedDispatchLayout} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {withGPUGraphAlgorithmsIterationScope} from './gpu-graph-computation-record';
import type {GPUGraphAlgorithmsStronglyConnectedComponents} from './gpu-graph-strongly-connected-components';
import {
  importGPUGraphAlgorithmsColumn,
  importGPUGraphAlgorithmsOverflow,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';

const OPERATION = 'GPUGraphAlgorithmsStronglyConnectedComponents';
const WORKGROUP_SIZE = 256;
const INVALID_COMPONENT = 0xffffffff;

/** State machine phases; the numbers are shared with the controller shader. */
const PHASE_TRIM = 0;
const PHASE_COLOR = 1;
const PHASE_BACK = 2;
const PHASE_FINALIZE = 3;
const PHASE_DONE = 4;

/** Control words: `[phase, changed, remaining]`. */
const CONTROL_LENGTH = 4;

/** Per-vertex flag bits. */
const FLAG_HAS_INPUT = 1;
const FLAG_HAS_OUTPUT = 2;
const FLAG_MARKED = 4;

type ComponentsAdjacency = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/**
 * Adds the GPU strong-component state machine: trim, colour, backward-mark, finalize.
 * @internal
 */
export function addGPUGraphAlgorithmsStronglyConnectedComponentsToGraph<Parameters>(
  components: GPUGraphAlgorithmsStronglyConnectedComponents<
    GPUGraphAlgorithmsTopologyLike,
    GPUGraphAlgorithmsColumn<'uint32'>
  >,
  commandGraph: GPUCommandGraph<Parameters>
): void {
  const {id} = components;
  const vertexCount = components.topology.graph.vertexCount;
  const output = importGPUGraphAlgorithmsColumn(commandGraph, `${id}-output`, components.output);
  const converged = components.converged
    ? importGPUGraphAlgorithmsColumn(commandGraph, `${id}-converged`, components.converged)
    : createTransientView(commandGraph, `${id}-converged-scratch`, 'uint32', 1);

  if (vertexCount === 0) {
    if (components.converged) {
      commandGraph.add(
        createWGSLKernelNode(commandGraph, {
          id: `${id}-initialize-empty`,
          operation: OPERATION,
          variant: 'initialize-empty',
          invocationCount: 1,
          workgroupSize: 2,
          bindings: [{name: 'converged', view: converged, type: 'u32', access: 'read_write'}],
          body: 'converged[convergedOffset] = 1u;'
        })
      );
    }
    return;
  }

  const adjacency: ComponentsAdjacency = {
    offsets: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${id}-offsets`,
      components.topology.forward.offsets
    ),
    neighbors: importGPUGraphAlgorithmsColumn(
      commandGraph,
      `${id}-neighbors`,
      components.topology.forward.neighbors
    ),
    overflow: importGPUGraphAlgorithmsOverflow(
      commandGraph,
      `${id}-overflow`,
      components.topology.forward.overflow,
      OPERATION
    )
  };
  const color = createTransientView(commandGraph, `${id}-color`, 'uint32', vertexCount);
  const flags = createTransientView(commandGraph, `${id}-flags`, 'uint32', vertexCount);
  const control = createTransientView(commandGraph, `${id}-control`, 'uint32', CONTROL_LENGTH);
  const gate = createWGSLDispatchGate(commandGraph, {
    id: `${id}-gate`,
    operation: OPERATION,
    shapes: [
      [vertexCount, WORKGROUP_SIZE],
      [vertexCount, WORKGROUP_SIZE],
      [vertexCount, WORKGROUP_SIZE],
      [vertexCount, WORKGROUP_SIZE]
    ]
  });
  const layout = getBoundedDispatchLayout(
    OPERATION,
    vertexCount,
    WORKGROUP_SIZE,
    commandGraph.device.limits.maxComputeWorkgroupsPerDimension
  );

  const declarations = `const VERTEX_COUNT: u32 = ${vertexCount}u;
const CAPACITY: u32 = ${adjacency.neighbors.length}u;
const INVALID: u32 = ${INVALID_COMPONENT}u;
const FLAG_HAS_INPUT: u32 = ${FLAG_HAS_INPUT}u;
const FLAG_HAS_OUTPUT: u32 = ${FLAG_HAS_OUTPUT}u;
const FLAG_MARKED: u32 = ${FLAG_MARKED}u;`;
  const adjacencyBindings = [
    {name: 'offsets', view: adjacency.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: adjacency.neighbors, type: 'u32', access: 'read'}
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
    commandGraph.add(
      createWGSLKernelNode(commandGraph, {
        id: nodeId,
        operation: OPERATION,
        variant,
        record: {datum: {keySpace: 'vertex'}},
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
      {name: 'overflow', view: adjacency.overflow, type: 'u32', access: 'read'},
      labelBinding,
      colorBinding,
      flagsBinding,
      controlBinding,
      {name: 'dispatch', view: gate.view, type: 'u32', access: 'read_write'},
      {name: 'converged', view: converged, type: 'u32', access: 'read_write'}
    ],
    `atomicStore(&output[outputOffset + index], INVALID);
  atomicStore(&color[colorOffset + index], index);
  atomicStore(&flags[flagsOffset + index], 0u);
  if (index == 0u) {
    let valid = overflow[overflowOffset] == 0u;
    atomicStore(&control[controlOffset], select(${PHASE_DONE}u, ${PHASE_TRIM}u, valid));
    atomicStore(&control[controlOffset + 1u], 0u);
    atomicStore(&control[controlOffset + 2u], 0u);
    converged[convergedOffset] = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
      let base = dispatchOffset + slot * 3u;
      dispatch[base] = select(0u, ${layout.x}u, valid && slot == ${PHASE_TRIM}u);
      dispatch[base + 1u] = ${layout.y}u;
      dispatch[base + 2u] = ${layout.z}u;
    }
  }`
  );

  for (let iteration = 0; iteration < components.iterations; iteration++) {
    const prefix = `${id}-iteration-${iteration}`;
    withGPUGraphAlgorithmsIterationScope(
      commandGraph,
      {
        loopId: `${id}-iterations`,
        iteration,
        maximumIterations: components.iterations,
        gated: true
      },
      () => {
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
        commandGraph.add(
          createWGSLKernelNode(commandGraph, {
            id: `${prefix}-controller`,
            operation: OPERATION,
            variant: 'controller',
            invocationCount: 1,
            workgroupSize: 2,
            bindings: [
              controlBinding,
              {name: 'dispatch', view: gate.view, type: 'u32', access: 'read_write'},
              {name: 'converged', view: converged, type: 'u32', access: 'read_write'}
            ],
            body: `let phase = atomicLoad(&control[controlOffset]);
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
  for (var slot = 0u; slot < 4u; slot++) {
    let base = dispatchOffset + slot * 3u;
    dispatch[base] = select(0u, ${layout.x}u, next == slot);
    dispatch[base + 1u] = ${layout.y}u;
    dispatch[base + 2u] = ${layout.z}u;
  }
  if (next == ${PHASE_DONE}u && phase != ${PHASE_DONE}u) { converged[convergedOffset] = 1u; }`
          })
        );
      }
    );
  }

  if (components.invalidateOnNonConvergence) {
    commandGraph.add(
      createWGSLKernelNode(commandGraph, {
        id: `${id}-invalidate-non-converged`,
        operation: OPERATION,
        variant: 'invalidate',
        record: {datum: {keySpace: 'vertex'}},
        invocationCount: vertexCount,
        workgroupSize: WORKGROUP_SIZE,
        bindings: [labelBinding, {name: 'converged', view: converged, type: 'u32', access: 'read'}],
        body: `if (converged[convergedOffset] == 0u) {
    atomicStore(&output[outputOffset + index], ${INVALID_COMPONENT}u);
  }`
      })
    );
  }
}
