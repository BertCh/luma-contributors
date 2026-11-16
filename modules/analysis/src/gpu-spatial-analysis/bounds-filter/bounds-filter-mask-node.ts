// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {BOUNDS_FILTER_ROWS_PER_WORD} from './bounds-filter-compaction';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Fixed workgroup shape shares a pipeline across uneven bounds chunks. */
const BOUNDS_FILTER_CHUNK_WORKGROUP_SIZE = 256;
/** Two aligned vec4s hold four binding offsets and the source row count. */
const BOUNDS_FILTER_CHUNK_PARAMETER_WORDS = 8;

/** Classifies a bounds chunk using runtime lengths and offsets, never chunk-specialized pipelines. @internal */
export function createBoundsFilterMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    bounds: GraphDataView<'float32x4'>;
    box: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
    callerMask?: GraphDataView<'uint32'>;
    comparison: string;
  }
): GPUCommandNode<Parameters> {
  const views = [
    props.bounds,
    props.box,
    props.mask,
    ...(props.callerMask ? [props.callerMask] : [])
  ];
  const values = new Uint32Array(BOUNDS_FILTER_CHUNK_PARAMETER_WORDS);
  for (const [index, view] of views.entries()) values[index] = getViewElementOffset(view);
  values[5] = props.bounds.length;
  const names = ['bounds', 'box', 'maskOut', 'callerMask'];
  const source = `${views.map((_, index) => `@group(0) @binding(${index}) var<storage, ${index < 2 ? 'read' : 'read_write'}> ${names[index]}: array<${index < 2 ? 'f32' : 'u32'}>;`).join('\n')}
@group(0) @binding(${views.length}) var<uniform> boundsSettings: array<vec4<u32>, 2>;
${IS_FINITE_BITS_WGSL}
@compute @workgroup_size(${BOUNDS_FILTER_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(local_invocation_index) localInvocationIndex: u32,
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) workgroupCounts: vec3<u32>) {
  let index = ((workgroupId.z * workgroupCounts.y + workgroupId.y) * workgroupCounts.x + workgroupId.x) * ${BOUNDS_FILTER_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let firstRow = index * ${BOUNDS_FILTER_ROWS_PER_WORD}u;
  if (firstRow >= boundsSettings[1][1]) { return; }
  var bits = 0u;
  for (var bit = 0u; bit < ${BOUNDS_FILTER_ROWS_PER_WORD}u; bit++) {
  let row = firstRow + bit;
  if (row >= boundsSettings[1][1]) { break; }
  let base = boundsSettings[0][0] + 4u * row;
  let fMin = vec2<f32>(bounds[base], bounds[base + 1u]);
  let fMax = vec2<f32>(bounds[base + 2u], bounds[base + 3u]);
  let boxBase = boundsSettings[0][1];
  let boxMin = vec2<f32>(box[boxBase], box[boxBase + 1u]);
  let boxMax = vec2<f32>(box[boxBase + 2u], box[boxBase + 3u]);
  let wrapped = boxMin.x > boxMax.x;
  let finite = isFiniteBits(fMin.x) && isFiniteBits(fMin.y) && isFiniteBits(fMax.x) && isFiniteBits(fMax.y);
  let accepted = select(0u, 1u, finite && ${props.comparison});
  bits |= accepted << bit;
  ${props.callerMask ? 'callerMask[boundsSettings[0][3] + row] = accepted;' : ''}
  }
  maskOut[boundsSettings[0][2] + index] = bits;
}`;
  const layout = getBoundedDispatchLayout(
    'GPUBoundsFilter',
    Math.ceil(props.bounds.length / BOUNDS_FILTER_ROWS_PER_WORD),
    BOUNDS_FILTER_CHUNK_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: views.map((view, index) => ({
      buffer: view,
      usage: index < 2 ? 'storage-read' : 'storage-read-write'
    })),
    workload: {
      operation: 'GPUBoundsFilter',
      maximumInvocationCount: Math.ceil(props.bounds.length / BOUNDS_FILTER_ROWS_PER_WORD)
    },
    compile: ({device}) => {
      const configuration = device.createBuffer({
        id: `${props.id}-settings`,
        data: values,
        usage: Buffer.UNIFORM | Buffer.COPY_DST
      });
      const kernel = new Kernel(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: [
            ...views.map((_, index) => ({
              name: names[index],
              type: index < 2 ? ('read-only-storage' as const) : ('storage' as const),
              group: 0,
              location: index
            })),
            {name: 'boundsSettings', type: 'uniform', group: 0, location: views.length}
          ]
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {boundsSettings: configuration};
          for (const [index, view] of views.entries())
            bindings[names[index]] = getViewBinding(view, getBuffer);
          kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
        },
        destroy: () => {
          kernel.destroy();
          configuration.destroy();
        }
      };
    }
  });
}
