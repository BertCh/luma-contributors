// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A one-node GPU step that turns the ring vertex offsets GPUPointBuffer wrote into closed outline
// segments (x0, y0, x1, y1): vertex v of point p is at center p plus offset p * ringSize + v, and
// each segment joins vertex v to vertex v + 1 of the same ring.

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
class RingSegments implements GPUCommandNodeProducer {
  constructor(
    private readonly id: string,
    private readonly centers: GraphDataView<'float32x2'>,
    private readonly offsets: GraphDataView<'float32x2'>,
    private readonly segments: GraphDataView<'float32x4'>,
    private readonly ringSize: number
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, centers, offsets, segments, ringSize} = this;
    const dispatch = getBoundedDispatchLayout(
      id,
      segments.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const shaderSource = /* wgsl */ `
const centersOffset: u32 = ${getViewElementOffset(centers)}u;
const offsetsOffset: u32 = ${getViewElementOffset(offsets)}u;
const segmentsOffset: u32 = ${getViewElementOffset(segments)}u;
@group(0) @binding(0) var<storage, read> centers: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
@group(0) @binding(2) var<storage, read_write> segments: array<f32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatch, WORKGROUP_SIZE)}
  if (index >= ${segments.length}u) { return; }
  let ring = index / ${ringSize}u;
  let vertex = index % ${ringSize}u;
  let next = (vertex + 1u) % ${ringSize}u;
  let center = vec2f(centers[centersOffset + ring * 2u], centers[centersOffset + ring * 2u + 1u]);
  let first = ring * ${ringSize}u + vertex;
  let second = ring * ${ringSize}u + next;
  let start = center + vec2f(offsets[offsetsOffset + first * 2u], offsets[offsetsOffset + first * 2u + 1u]);
  let end = center + vec2f(offsets[offsetsOffset + second * 2u], offsets[offsetsOffset + second * 2u + 1u]);
  segments[segmentsOffset + index * 4u] = start.x;
  segments[segmentsOffset + index * 4u + 1u] = start.y;
  segments[segmentsOffset + index * 4u + 2u] = end.x;
  segments[segmentsOffset + index * 4u + 3u] = end.y;
}`;
    return [
      createGPUComputeCommandNode<Parameters>({
        id,
        resources: [
          {buffer: centers, usage: 'storage-read'},
          {buffer: offsets, usage: 'storage-read'},
          {buffer: segments, usage: 'storage-read-write'}
        ],
        workload: {
          operation: 'RingSegments',
          variant: 'ring',
          commandCount: 1,
          maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z,
          maximumInvocationCount: dispatch.x * dispatch.y * dispatch.z * WORKGROUP_SIZE,
          readByteLength: segments.length * 16,
          writeByteLength: segments.length * 16
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id,
            source: shaderSource,
            shaderLayout: {
              bindings: ['centers', 'offsets', 'segments'].map((name, location) => ({
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
                centers: getViewBinding(centers, getBuffer),
                offsets: getViewBinding(offsets, getBuffer),
                segments: getViewBinding(segments, getBuffer)
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

/** Adds closed ring outlines (one segment per vertex) around `centers` from ring `offsets`. */
export function addRingSegments(
  builder: RunBuilder,
  id: string,
  centers: RunBuffer<'float32x2'>,
  offsets: RunBuffer<'float32x2'>,
  ringSize: number
): RunBuffer<'float32x4'> {
  const segments = builder.allocate(id, 'float32x4', offsets.length);
  builder.graph.add(
    new RingSegments(`${id}-ring`, centers.view, offsets.view, segments.view, ringSize)
  );
  return segments;
}
