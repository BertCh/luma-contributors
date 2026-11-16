// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/** Matches the existing objective pair tree; each dispatch covers 256 pairs per workgroup. */
const AZP_OBJECTIVE_REDUCTION_WORKGROUP_SIZE = 256;

/** One level of AZP's fixed-order objective sum over its private, zero-offset scratch. @internal */
export function createAZPObjectiveReductionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    inputCount: number;
  }
): GPUCommandNode<Parameters> {
  const outputCount = Math.ceil(props.inputCount / 2);
  const input = graph.createDataView(props.input.buffer, {
    format: 'float32',
    length: props.inputCount
  });
  const output = graph.createDataView(props.output.buffer, {
    format: 'float32',
    length: outputCount
  });
  const layout = getBoundedDispatchLayout(
    'GPUAZPRegions',
    outputCount,
    AZP_OBJECTIVE_REDUCTION_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: [
      {buffer: input, usage: 'storage-read'},
      {buffer: output, usage: 'storage-read-write'}
    ],
    workload: {
      operation: 'GPUAZPRegions',
      variant: 'deterministic-objective-reduction',
      commandCount: 1,
      maximumInvocationCount: outputCount,
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: 'azp-objective-reduction',
        source: AZP_OBJECTIVE_REDUCTION_WGSL,
        shaderLayout: {
          bindings: [
            {name: 'valuesIn', type: 'read-only-storage', group: 0, location: 0},
            {name: 'valuesOut', type: 'storage', group: 0, location: 1}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: {
              valuesIn: getViewBinding(input, getBuffer),
              valuesOut: getViewBinding(output, getBuffer)
            },
            x: layout.x,
            y: layout.y,
            z: layout.z
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

const AZP_OBJECTIVE_REDUCTION_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> valuesIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> valuesOut: array<f32>;

@compute @workgroup_size(${AZP_OBJECTIVE_REDUCTION_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupIndex: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) lane: u32
) {
  let index = (workgroupIndex.x + workgroupCount.x * (workgroupIndex.y + workgroupCount.y * workgroupIndex.z)) * ${AZP_OBJECTIVE_REDUCTION_WORKGROUP_SIZE}u + lane;
  if (index >= arrayLength(&valuesOut)) { return; }
  let left = 2u * index;
  var total = valuesIn[left];
  if (left + 1u < arrayLength(&valuesIn)) { total += valuesIn[left + 1u]; }
  valuesOut[index] = total;
}`;
