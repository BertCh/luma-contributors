// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandGraphGPUIndirectCondition,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** @internal Shared by anti-only and inner-plus-unmatched publication. */
export type SpatialPredicateUnmatchedContext = {
  id: string;
  leftCount: number;
  leftFeatureCount?: GraphDataView<'uint32'>;
  candidateCapacity: number;
  countOnly?: boolean;
  finishOnly?: boolean;
  candidatePairs: GraphDataView<'uint32x2'>;
  flags: GraphDataView<'uint32'>;
  leftMatchCounts: GraphDataView<'uint32'>;
  unmatched: GPUCompactOutput;
  state: GraphDataView<'uint32'>;
  candidateTruncated?: GraphDataView<'uint32'>;
  candidateCondition?: GPUCommandGraphGPUIndirectCondition;
};

/** Counts complete classifications before pair truncation, then stably publishes known absences. @internal */
export function addSpatialPredicateUnmatchedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  nodes: GPUCommandNode<Parameters>[],
  context: SpatialPredicateUnmatchedContext
): void {
  const {
    id,
    leftCount,
    candidateCapacity,
    candidatePairs,
    flags,
    leftMatchCounts,
    unmatched,
    state
  } = context;
  const operation = 'GPUSpatialPredicateJoin';
  if (!context.finishOnly)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-count-left-matches`,
        operation,
        variant: 'count-left-matches',
        bindings: [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {
            name: 'leftMatchCounts',
            view: leftMatchCounts,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        condition: context.candidateCondition,
        body: `if (flags[flagsOffset + index] != 0u) {
    atomicAdd(&leftMatchCounts[leftMatchCountsOffset + candidatePairs[candidatePairsOffset + index * 2u]], 1u);
  }`
      })
    );
  if (context.countOnly) return;
  const unmatchedFlags = createTransientView(graph, `${id}-unmatched-flags`, 'uint32', leftCount);
  const unmatchedOffsets = createTransientView(
    graph,
    `${id}-unmatched-offsets`,
    'uint32',
    leftCount
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-unmatched-flags`,
      operation,
      variant: 'unmatched-flags',
      bindings: [
        {name: 'leftMatchCounts', view: leftMatchCounts, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read'},
        ...(context.leftFeatureCount
          ? [
              {
                name: 'liveCount',
                view: context.leftFeatureCount,
                type: 'u32',
                access: 'read'
              } as const
            ]
          : []),
        ...(context.candidateTruncated
          ? [
              {
                name: 'candidateTruncated',
                view: context.candidateTruncated,
                type: 'u32',
                access: 'read'
              } as const
            ]
          : []),
        {name: 'unmatchedFlags', view: unmatchedFlags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: leftCount,
      // Uncertainty has no per-left attribution: conservatively withhold all absence claims.
      body: `unmatchedFlags[unmatchedFlagsOffset + index] = select(0u, 1u, state[stateOffset + 1u] == 0u && ${context.leftFeatureCount ? `liveCount[liveCountOffset] <= ${leftCount}u && index < liveCount[liveCountOffset] && ` : ''}leftMatchCounts[leftMatchCountsOffset + index] == 0u${context.candidateTruncated ? ' && candidateTruncated[candidateTruncatedOffset + index] == 0u' : ''});`
    })
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan-unmatched`,
      input: unmatchedFlags,
      output: unmatchedOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-scatter-unmatched`,
      operation,
      variant: 'scatter-unmatched',
      bindings: [
        {name: 'unmatchedFlags', view: unmatchedFlags, type: 'u32', access: 'read'},
        {name: 'unmatchedOffsets', view: unmatchedOffsets, type: 'u32', access: 'read'},
        {name: 'ids', view: unmatched.ids, type: 'u32', access: 'read_write'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'}
      ],
      invocationCount: leftCount,
      declarations: `const UNMATCHED_CAPACITY: u32 = ${unmatched.ids.length}u;`,
      body: `if (unmatchedFlags[unmatchedFlagsOffset + index] != 0u) {
    let slot = unmatchedOffsets[unmatchedOffsetsOffset + index];
    if (slot < UNMATCHED_CAPACITY) { ids[idsOffset + slot] = index; }
  }
  if (index == ${leftCount - 1}u) {
    state[stateOffset + 3u] = unmatchedOffsets[unmatchedOffsetsOffset + index] + unmatchedFlags[unmatchedFlagsOffset + index];
  }`
    })
  );
}
