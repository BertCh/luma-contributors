// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PipelineFactory, type Device} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../src/analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../src/analysis-core/kernels/construction-snapshot';

/** Tiny authoring fixture: output[row] = input[row] * 3 + globalRow + totalRows. */
export class ExampleChunkContributor implements GPUCommandNodeProducer {
  readonly plan: GraphChunkDispatchPlan;
  readonly estimatedCommandNodeCount: number;
  readonly options: Readonly<{id?: string; maximumWorkgroupsPerDimension?: number}>;

  constructor(
    input: GraphVectorView<'uint32'>,
    output: GraphVectorView<'uint32'>,
    options: {id?: string; maximumWorkgroupsPerDimension?: number} = {}
  ) {
    this.options = snapshotConstructionFields(options, ['id', 'maximumWorkgroupsPerDimension']);
    this.plan = planGraphChunkDispatch('example', {input, output});
    validateDisjointWritableViews('example', {output});
    validateGraphOutputsDisjointFromInputs('example', [output], [input]);
    this.estimatedCommandNodeCount =
      this.plan.nonemptyChunkCount + (this.plan.chunks.length ? 1 : 0);
    validateCommandNodeBudget('example', this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    return createExampleChunkNodes(graph, this.plan, this.options);
  }
}

// Fixed source and no pipeline overrides, including for unequal lengths and 3D dispatches.
export const EXAMPLE_CHUNK_SOURCE = /* wgsl */ `
// A 256-thread workgroup index must be guarded before multiplication to prevent uint32 wrap.
const MAXIMUM_LINEAR_WORKGROUP_COUNT: u32 = 16777216u;
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
@group(0) @binding(1) var<storage, read> inputValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= MAXIMUM_LINEAR_WORKGROUP_COUNT) { return; }
  let rowIndex = workgroupIndex * 256u + localInvocationIndex;
  if (rowIndex >= chunkParameters[1]) { return; }
  outputValues[rowIndex] = inputValues[rowIndex] * 3u + chunkParameters[0] + rowIndex + chunkParameters[2];
}`;

/** Test fixture for contributors whose chunk views start at aligned binding offsets. */
export function createExampleChunkNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  plan: GraphChunkDispatchPlan,
  options: {id?: string; maximumWorkgroupsPerDimension?: number} = {}
): readonly GPUCommandNode<Parameters>[] {
  const id = options.id ?? 'example';
  validateGraphChunkDispatchBindings(graph, id, plan);
  // Resolve ALL layouts before declaring the parameter resource as well.
  const layouts = plan.chunks.map(chunk =>
    getBoundedDispatchLayout(
      id,
      chunk.length,
      256,
      options.maximumWorkgroupsPerDimension ?? graph.device.limits.maxComputeWorkgroupsPerDimension
    )
  );
  const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
    id: `${id}-parameters`,
    plan
  });
  // One published factory per contributor keeps the graph's cold-cache creation count observable.
  const pipelineFactories = new WeakMap<Device, PipelineFactory>();
  // Order each upload after the producer of its input. This lets two dependent contributors'
  // parameter buffers have disjoint lifetimes, so the repeated-encoding test exercises reuse.
  const inputDependencies = [
    ...new Map(
      plan.chunks.map(chunk => {
        const input = chunk.views['input'];
        return [input.buffer, input] as const;
      })
    ).values()
  ];
  return [
    ...nodes.map(node => ({
      ...node,
      resources: [
        ...(node.resources ?? []),
        ...inputDependencies.map(buffer => ({buffer, usage: 'storage-read' as const}))
      ]
    })),
    ...plan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const parameters = slots[chunk.chunkIndex];
        const input = chunk.views['input'];
        const output = chunk.views['output'];
        const layout = layouts[chunk.chunkIndex];
        return createGPUComputeCommandNode<Parameters>({
          id: `${id}-chunk-${chunk.chunkIndex}`,
          resources: [
            {buffer: parameters, usage: 'storage-read'},
            {buffer: input, usage: 'storage-read'},
            {buffer: output, usage: 'storage-read-write'}
          ],
          compile: ({device}) => {
            let pipelineFactory = pipelineFactories.get(device);
            if (!pipelineFactory) {
              pipelineFactory = new PipelineFactory(device);
              pipelineFactories.set(device, pipelineFactory);
            }
            const kernel = new Kernel(device, {
              id: 'example-chunk',
              pipelineFactory,
              source: EXAMPLE_CHUNK_SOURCE,
              shaderLayout: {
                bindings: [
                  {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                  {name: 'inputValues', group: 0, location: 1, type: 'read-only-storage'},
                  {name: 'outputValues', group: 0, location: 2, type: 'storage'}
                ]
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: {
                    chunkParameters: getViewBinding(parameters, getBuffer),
                    inputValues: getViewBinding(input, getBuffer),
                    outputValues: getViewBinding(output, getBuffer)
                  },
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      })
  ];
}
