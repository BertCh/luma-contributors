// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/** Lexicographic objective, original candidate rank, and accumulated invalid flags. @internal */
export const FACILITY_SCORE_ORDER_WGSL = /* wgsl */ `
fn mergeFacilityScores(first: vec4<u32>, second: vec4<u32>) -> vec4<u32> {
  let preferSecond = second.x < first.x ||
    (second.x == first.x && (second.y < first.y ||
      (second.y == first.y && second.z < first.z)));
  let winner = select(first, second, preferSecond);
  return vec4<u32>(winner.xyz, first.w | second.w);
}`;

/** Reduces four-word candidate records without serial winner scans or float atomics. @internal */
export function createFacilityScoreReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    scores: GraphDataView<'uint32'>;
    candidateCount: number;
  }
): {nodes: GPUCommandNode<Parameters>[]; winner: GraphDataView<'uint32'>} {
  const nodes: GPUCommandNode<Parameters>[] = [];
  let source = props.scores;
  let candidateCount = props.candidateCount;
  let level = 0;
  while (candidateCount > 1) {
    const groupCount = Math.ceil(candidateCount / 64);
    const destination = createTransientView(
      graph,
      `${props.id}-${level}`,
      'uint32',
      groupCount * 4
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${props.id}-${level}`,
        operation: props.operation,
        variant: 'candidate-reduction',
        invocationCount: groupCount * 64,
        workgroupSize: 64,
        guardIndex: false,
        bindings: [
          {name: 'candidateScores', view: source, type: 'u32', access: 'read'},
          {name: 'reducedScores', view: destination, type: 'u32', access: 'read_write'}
        ],
        declarations: `${FACILITY_SCORE_ORDER_WGSL}
var<workgroup> scores: array<vec4<u32>, 64>;`,
        body: `var score = vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0u);
  if (index < ${candidateCount}u) {
    let offset = candidateScoresOffset + index * 4u;
    score = vec4<u32>(candidateScores[offset], candidateScores[offset + 1u], candidateScores[offset + 2u], candidateScores[offset + 3u]);
  }
  scores[localInvocationIndex] = score;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (localInvocationIndex < stride) {
      scores[localInvocationIndex] = mergeFacilityScores(scores[localInvocationIndex], scores[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && workgroupIndex < ${groupCount}u) {
    let offset = reducedScoresOffset + workgroupIndex * 4u;
    reducedScores[offset] = scores[0].x;
    reducedScores[offset + 1u] = scores[0].y;
    reducedScores[offset + 2u] = scores[0].z;
    reducedScores[offset + 3u] = scores[0].w;
  }`
      })
    );
    source = destination;
    candidateCount = groupCount;
    level++;
  }
  return {nodes, winner: source};
}
