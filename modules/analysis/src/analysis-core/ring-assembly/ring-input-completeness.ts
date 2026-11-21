// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../kernels/wgsl-kernel-nodes';
import type {GPUSegmentRingAssemblyOutput} from './gpu-segment-ring-assembly';

/** Counted outline input; source capacity loss is independent of assembly capacity. */
export type GPUSegmentRingInputStatus = {
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  candidateOverflow?: GraphDataView<'uint32'>;
};

/** Publishes inherited loss, or suppresses geometry when no completeness word was requested. @internal */
export function getRingInputCompletenessNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GPUSegmentRingInputStatus | undefined;
    output: GPUSegmentRingAssemblyOutput;
    counters: GraphDataView<'uint32'>;
    segmentFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, input, output, counters, segmentFlags} = props;
  const bindings: WGSLKernelBinding[] = input
    ? [
        {name: 'upstreamOverflow', view: input.overflow, type: 'u32', access: 'read'},
        ...(input.candidateOverflow
          ? [
              {
                name: 'upstreamCandidate',
                view: input.candidateOverflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ]
    : [];
  const loss = input
    ? `upstreamOverflow[upstreamOverflowOffset] != 0u${input.candidateOverflow ? ' || upstreamCandidate[upstreamCandidateOffset] != 0u' : ''}`
    : 'false';
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (output.status.candidateOverflow || output.status.complete) {
    const candidate =
      output.status.candidateOverflow ??
      createTransientView(graph, `${id}-candidate-incomplete`, 'uint32', 1);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-input-completeness`,
        operation: 'GPUSegmentRingAssembly',
        variant: 'input-completeness',
        invocationCount: 1,
        bindings: [
          ...bindings,
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {
            name: 'candidate',
            view: candidate,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `candidate[candidateOffset] = select(0u, 1u, ${loss} || counters[countersOffset] > counters[countersOffset + 2u]);`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-conflict-completeness`,
        operation: 'GPUSegmentRingAssembly',
        variant: 'conflict-completeness',
        invocationCount: segmentFlags.length,
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'counters', view: counters, type: 'u32', access: 'read'},
          {name: 'candidate', view: candidate, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `if (index < counters[countersOffset + 5u] && (flags[flagsOffset + index] & 4u) != 0u) { atomicMax(&candidate[candidateOffset], 1u); }`
      })
    );
    if (output.status.complete) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-complete`,
          operation: 'GPUSegmentRingAssembly',
          variant: 'complete',
          invocationCount: 1,
          bindings: [
            {name: 'candidate', view: candidate, type: 'u32', access: 'read'},
            {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read'},
            {name: 'complete', view: output.status.complete, type: 'u32', access: 'read_write'}
          ],
          body: 'complete[completeOffset] = select(0u, 1u, candidate[candidateOffset] == 0u && overflow[overflowOffset] == 0u);'
        })
      );
    }
  } else if (input) {
    // Trace the retained prefix first, so local open-edge diagnostics and final overflow remain
    // independent. Suppress only its geometry publication when the outline was incomplete.
    const tables = [
      output.status.count,
      output.ringOffsets,
      output.polygons?.ringOffsets,
      output.polygons?.polygonOffsets,
      output.polygons?.featureOffsets
    ];
    for (const [index, table] of tables.entries()) {
      if (!table) continue;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-incomplete-${index}`,
          operation: 'GPUSegmentRingAssembly',
          variant: 'suppress-incomplete',
          invocationCount: index === 0 ? 1 : table.length,
          bindings: [...bindings, {name: 'table', view: table, type: 'u32', access: 'read_write'}],
          body: `if (${loss}) { table[tableOffset + index] = 0u; }`
        })
      );
    }
    for (const [index, mapping] of [
      output.segmentRings,
      output.segmentVertices,
      output.polygons?.sourceIds
    ].entries()) {
      if (!mapping) continue;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-suppress-incomplete-mapping-${index}`,
          operation: 'GPUSegmentRingAssembly',
          variant: 'suppress-incomplete-mapping',
          invocationCount: mapping.length,
          bindings: [
            ...bindings,
            {name: 'mapping', view: mapping, type: 'u32', access: 'read_write'}
          ],
          body: `if (${loss}) { mapping[mappingOffset + index] = 0xffffffffu; }`
        })
      );
    }
  }
  if (output.status.polygonCount && output.polygons) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-polygon-count`,
        operation: 'GPUSegmentRingAssembly',
        variant: 'polygon-count',
        invocationCount: 1,
        bindings: [
          {name: 'offsets', view: output.polygons.featureOffsets, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.polygonCount, type: 'u32', access: 'read_write'}
        ],
        body: `count[countOffset] = offsets[offsetsOffset + ${output.polygons.featureOffsets.length - 1}u];`
      })
    );
  }
  return nodes;
}
