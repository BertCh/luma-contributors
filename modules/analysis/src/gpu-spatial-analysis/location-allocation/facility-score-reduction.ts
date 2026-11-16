// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';

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
  validateGraphChunkBindingRanges(graph, props.id, {scores: props.scores});
  if (props.candidateCount > 1) {
    validateGPUStorageBindingByteLength(
      graph.device,
      props.id,
      'reduction parameters',
      16,
      'request a larger storage binding limit'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      props.id,
      'reduced scores',
      Math.ceil(props.candidateCount / 64) * 16,
      'reduce candidateCount'
    );
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  let source = props.scores;
  let candidateCount = props.candidateCount;
  let level = 0;
  while (candidateCount > 1) {
    const groupCount = Math.ceil(candidateCount / 64);
    const layout = getBoundedDispatchLayout(
      props.operation,
      groupCount * 64,
      64,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const destination = createTransientView(
      graph,
      `${props.id}-${level}`,
      'uint32',
      groupCount * 4
    );
    const parameters = createSpatialStatisticsParameterNodes(
      graph,
      `${props.id}-${level}-parameters`,
      Uint32Array.of(getViewElementOffset(source), candidateCount, layout.x, layout.y)
    );
    nodes.push(...parameters.nodes);
    // Dispatch dimensions belong to the node. Neither shader text nor specialization constants
    // depend on the reduction level, including the first level's possibly unaligned source.
    const input = source;
    nodes.push(
      createGPUComputeCommandNode<Parameters>({
        id: `${props.id}-${level}`,
        resources: [
          {buffer: input, usage: 'storage-read'},
          {buffer: parameters.parameters, usage: 'storage-read'},
          {buffer: destination, usage: 'storage-read-write'}
        ],
        workload: {
          operation: props.operation,
          variant: 'candidate-reduction',
          commandCount: 1,
          maximumWorkgroupCount: layout.x * layout.y * layout.z,
          maximumInvocationCount: groupCount * 64
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id: `${props.id}-reduction`,
            source: FACILITY_SCORE_REDUCTION_WGSL,
            shaderLayout: {
              bindings: [
                {name: 'candidateScores', type: 'read-only-storage', group: 0, location: 0},
                {name: 'parameters', type: 'read-only-storage', group: 0, location: 1},
                {name: 'reducedScores', type: 'storage', group: 0, location: 2}
              ]
            }
          });
          return {
            encode: ({computePass, getBuffer}) =>
              kernel.dispatch(computePass, {
                bindings: {
                  candidateScores: getViewBinding(input, getBuffer),
                  parameters: getViewBinding(parameters.parameters, getBuffer),
                  reducedScores: getViewBinding(destination, getBuffer)
                },
                x: layout.x,
                y: layout.y,
                z: layout.z
              }),
            destroy: () => kernel.destroy()
          };
        }
      })
    );
    source = destination;
    candidateCount = groupCount;
    level++;
  }
  return {nodes, winner: source};
}

const FACILITY_SCORE_REDUCTION_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> candidateScores: array<u32>;
@group(0) @binding(1) var<storage, read> parameters: array<u32>;
@group(0) @binding(2) var<storage, read_write> reducedScores: array<u32>;
${FACILITY_SCORE_ORDER_WGSL}
var<workgroup> scores: array<vec4<u32>, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>) {
  let workgroupIndex = (workgroupId.z * parameters[3] + workgroupId.y) * parameters[2] + workgroupId.x;
  let index = workgroupIndex * 64u + localInvocationIndex;
  var score = vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0u);
  if (index < parameters[1]) {
    let offset = parameters[0] + index * 4u;
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
  if (localInvocationIndex == 0u && workgroupIndex < arrayLength(&reducedScores) / 4u) {
    let offset = workgroupIndex * 4u;
    reducedScores[offset] = scores[0].x;
    reducedScores[offset + 1u] = scores[0].y;
    reducedScores[offset + 2u] = scores[0].z;
    reducedScores[offset + 3u] = scores[0].w;
  }
}`;
