// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A one-node GPU step that applies the column-major 4x4 transform a registration wrote to a point
// cloud and keeps the x and y of the result, so the aligned cloud is drawn from the registration's
// own output buffer.

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';

const WORKGROUP_SIZE = 64;
class TransformedPoints implements GPUCommandNodeProducer {
  constructor(
    private readonly id: string,
    private readonly points: GraphDataView<'float32x3'>,
    private readonly transform: GraphDataView<'float32'>,
    private readonly footprint: GraphDataView<'float32x2'>
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, points, transform, footprint} = this;
    const dispatch = getBoundedDispatchLayout(
      id,
      footprint.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const shaderSource = /* wgsl */ `
const pointsOffset: u32 = ${getViewElementOffset(points)}u;
const transformOffset: u32 = ${getViewElementOffset(transform)}u;
const footprintOffset: u32 = ${getViewElementOffset(footprint)}u;
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> transform: array<f32>;
@group(0) @binding(2) var<storage, read_write> footprint: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatch, WORKGROUP_SIZE)}
  if (index >= ${footprint.length}u) { return; }
  let point = vec4f(
    points[pointsOffset + index * 3u],
    points[pointsOffset + index * 3u + 1u],
    points[pointsOffset + index * 3u + 2u],
    1.0
  );
  for (var component = 0u; component < 2u; component++) {
    var sum = 0.0;
    for (var column = 0u; column < 4u; column++) {
      sum += transform[transformOffset + column * 4u + component] * point[column];
    }
    footprint[footprintOffset + index * 2u + component] = sum;
  }
}`;
    return [
      createGPUComputeCommandNode<Parameters>({
        id,
        resources: [
          {buffer: points, usage: 'storage-read'},
          {buffer: transform, usage: 'storage-read'},
          {buffer: footprint, usage: 'storage-read-write'}
        ],
        workload: {
          operation: 'TransformedPoints',
          variant: 'xy',
          commandCount: 1,
          maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z,
          maximumInvocationCount: dispatch.x * dispatch.y * dispatch.z * WORKGROUP_SIZE,
          readByteLength: footprint.length * 12,
          writeByteLength: footprint.length * 8
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id,
            source: shaderSource,
            shaderLayout: {
              bindings: ['points', 'transform', 'footprint'].map((name, location) => ({
                name,
                type: 'storage' as const,
                group: 0,
                location
              }))
            }
          });
          return {
            encode: ({computePass, getBuffer}) => {
              const bindings: Record<string, Binding> = {
                points: getViewBinding(points, getBuffer),
                transform: getViewBinding(transform, getBuffer),
                footprint: getViewBinding(footprint, getBuffer)
              };
              kernel.dispatch(computePass, {bindings, ...dispatch});
            },
            destroy: () => kernel.destroy()
          };
        }
      })
    ];
  }
}

/** Adds the x and y of `points` after `transform` (column-major 4x4, 16 floats), as float32x2. */
export function addTransformedFootprint(
  builder: RunBuilder,
  id: string,
  points: RunBuffer<'float32x3'>,
  transform: RunBuffer<'float32'>
): RunBuffer<'float32x2'> {
  const footprint = builder.allocate(id, 'float32x2', points.length);
  builder.graph.add(
    new TransformedPoints(`${id}-transform`, points.view, transform.view, footprint.view)
  );
  return footprint;
}
