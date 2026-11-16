// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Published keyed-grid radix histogram/scatter kernels dispatch 256 lanes per workgroup. */
const DORLING_GRID_WORKGROUP_SIZE = 256;

/**
 * Gates one Dorling phase, including its published primitives. The gate is latched before any
 * phase work, so a metrics publication can stop future phases without skipping its own commit.
 *
 * The published keyed grid's scan and aggregation nodes expose exact padded workgroup totals;
 * its radix histogram/scatter (and one-workgroup small sort) dispatch over the input rows at 256
 * lanes. Reduction levels are annotated by the caller with their documented 256-way hierarchy;
 * finalization is one group. All use the public bounded layout. A node spec compares these
 * triples with the original encode callbacks, including a forced two-dimensional device limit.
 * This is specific to these Dorling primitives, not a replacement for a luma.gl gating API.
 */
export function gateDorlingPhase<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  nodes: readonly GPUCommandNode<Parameters>[],
  status: GraphDataView<'uint32'>,
  rowCount: number
): GPUCommandNode<Parameters>[] {
  const sortLayout = getBoundedDispatchLayout(
    'Dorling keyed sort',
    rowCount,
    DORLING_GRID_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const shapes = nodes.map(node => {
    validateRequirement(
      id,
      node.type === 'compute' && !node.condition,
      'phase nodes must be unconditional compute nodes'
    );
    const groupCount =
      node.workload?.maximumWorkgroupCount ?? sortLayout.x * sortLayout.y * sortLayout.z;
    // The public bounded-layout helper requires at least two lanes. Two dummy invocations per
    // known workgroup preserve its exact dimensions, without depending on shader lane counts.
    return [groupCount * 2, 2] as const;
  });
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation: 'GPUDorlingCartogram',
    shapes
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-phase-gate`,
      operation: 'GPUDorlingCartogram',
      variant: 'phase-gate',
      invocationCount: 1,
      bindings: [{name: 'status', view: status, type: 'u32', access: 'read'}, gate.binding],
      body: `if (status[statusOffset + 1u] == 0u) { ${gate.armWGSL} }
else { ${gate.disarmWGSL} }`
    }),
    ...nodes.map((node, slot) => {
      validateRequirement(id, node.type === 'compute', 'phase nodes must be compute nodes');
      const {condition, extraResources} = gate.getGate(slot);
      return {...node, condition, resources: [...(node.resources ?? []), ...extraResources]};
    })
  ];
}
