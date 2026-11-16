// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  validatePackedView,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../analysis-core/contracts/earth';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {GUARDED_ATAN2_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateFiniteNumber,
  validateSinglePackedView,
  validateRequirement,
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs
} from '../analysis-core/kernels/gpu-contributor-utils';

import {
  createGraphChunkDispatchParameters,
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../analysis-core/kernels/graph-chunk-dispatch';

// Fixed workgroups and a guard before multiplication preserve uint32 indices at large dispatches.
const POINT_BUFFER_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const POINT_BUFFER_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/** Mean Earth radius in meters used by the geographic mode. */
export const GPU_POINT_BUFFER_EARTH_RADIUS_METERS = GPU_EARTH_MEAN_RADIUS_METERS;

/** Number of float32 values read from {@link GPUPointBufferProps.parameters}. */
export const GPU_POINT_BUFFER_PARAMETER_LENGTH = 2;

/**
 * Packs `[radius, 0]`: the radius is meters for the `'geographic'` coordinate system and
 * coordinate units for `'planar'`.
 */
export function getGPUPointBufferParameterValues(
  radius: number,
  target: Float32Array = new Float32Array(GPU_POINT_BUFFER_PARAMETER_LENGTH)
): Float32Array {
  // The radius must be a positive finite number: zero collapses the ring, negative mirrors it.
  validateFiniteNumber('GPUPointBuffer', 'radius', radius);
  validateRequirement('GPUPointBuffer', radius > 0, 'radius must be positive');
  target[0] = radius;
  target[1] = 0;
  return target;
}

export type GPUPointBufferProps<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Prefix for node IDs. Defaults to `'point-buffer'`. */
  id?: string;
  /** Point centers: longitude and latitude in degrees (`'geographic'`) or planar x and y. */
  points: Rows;
  /** Per-frame parameters with at least {@link GPU_POINT_BUFFER_PARAMETER_LENGTH} float32 values. */
  parameters: GraphDataView<'float32'>;
  /**
   * Ring vertex offsets from the point center, `verticesPerRing` rows per point in point order.
   * Geographic offsets are degrees of longitude and latitude, so a float64 consumer adds them to
   * the center without losing the float32 precision of a small radius.
   */
  offsets: Rows;
  /** Vertices per ring, at least 3. */
  verticesPerRing: number;
  /** `'geographic'` uses the spherical destination formula, `'planar'` a circle in coordinate units. */
  coordinateSystem: 'geographic' | 'planar';
};

/**
 * Generates inscribed circle vertices around points. Vertex `v` of a ring has bearing
 * `2 pi v / verticesPerRing` clockwise from north; the ring is not closed. Geographic offsets use
 * the spherical destination formula in a cancellation-free form (versine and half-angle
 * differences) so offsets keep full relative precision for small radii.
 *
 * Near the poles the formulation switches to the absolute destination (offsets then carry about
 * float32 precision of the absolute latitude), and the center latitude is split into sine and
 * cosine from the co-latitude so a center at exactly +/-90 degrees is well defined. Latitude offsets
 * are clamped so that `center + offset` stays within [-90, 90]; longitude offsets lie in
 * [-180, 180]. Antimeridian wrapping is left to the consumer, which has the float64 centers and
 * must wrap `center longitude + offset` into [-180, 180].
 *
 * Points may be chunked; each output chunk holds whole rings for the corresponding input chunk,
 * with exactly `verticesPerRing * pointsChunk.length` rows. Parameters stay packed.
 */
export class GPUPointBuffer<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  /** Upper bound including the chunk parameter upload. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  constructor(readonly props: GPUPointBufferProps<Rows>) {
    this.id = props.id ?? 'point-buffer';
    this.chunkPlan = planGraphChunkDispatch(this.id, {points: props.points});
    for (const [name, view] of [
      ['points', props.points],
      ['offsets', props.offsets]
    ] as const) {
      for (const chunk of getGraphViewChunks(view))
        validatePackedView(chunk, ['float32x2'], `${this.id} ${name}`);
    }
    // Rings need at least a triangle and one offset row per ring vertex.
    validateCapacity(this.id, 'verticesPerRing', props.verticesPerRing, 3);
    const offsetChunks = getGraphViewChunks(props.offsets);
    validateRequirement(
      this.id,
      offsetChunks.length === this.chunkPlan.chunks.length,
      'offsets must match points chunk count with whole rings per chunk'
    );
    // The public offset count and every local vertex index stay in the shared uint32 row domain.
    validateCapacity(
      this.id,
      'offsets.length',
      props.points.length * props.verticesPerRing,
      0,
      GRAPH_CHUNK_MAXIMUM_ROW_COUNT
    );
    validateRequirement(
      this.id,
      props.offsets.length === props.points.length * props.verticesPerRing,
      'offsets must have verticesPerRing rows per point'
    );
    for (const chunk of this.chunkPlan.chunks) {
      validateRequirement(
        this.id,
        offsetChunks[chunk.chunkIndex].length === chunk.length * props.verticesPerRing,
        `offsets chunk ${chunk.chunkIndex} must have verticesPerRing rows per points chunk row`
      );
    }
    validateDisjointWritableViews(this.id, {offsets: props.offsets});
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.offsets],
      [props.points, props.parameters]
    );
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    // Parameters hold the radius.
    validateSinglePackedView(props.parameters, ['float32'], `${this.id} parameters`);
    validateRequirement(
      this.id,
      props.parameters.length >= GPU_POINT_BUFFER_PARAMETER_LENGTH,
      `parameters must have at least ${GPU_POINT_BUFFER_PARAMETER_LENGTH} rows`
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {points, parameters, offsets, verticesPerRing, coordinateSystem} = this.props;
    validateGraphChunkDispatchBindings(graph, this.id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, this.id, {parameters, offsets});
    if (this.chunkPlan.nonemptyChunkCount === 0) return [];
    // The center is split into sine and cosine of latitude from the co-latitude near the poles:
    // cos(pi / 2) in float32 is -4e-8, not zero, which would flip the sign of the longitude term
    // and give arbitrary longitudes. Away from the poles and for small radii the cancellation-free
    // offset form keeps full relative precision; otherwise the absolute destination latitude is
    // computed from the spherical direct formula. Latitudes are clamped to [-90, 90].
    const geographicBody = `let clampedLatitude = clamp(centerLatitude, -90.0, 90.0);
  let phi = clampedLatitude * DEGREES_TO_RADIANS;
  let delta = radius / EARTH_RADIUS_METERS;
  let sinDelta = sin(delta);
  let cosDelta = cos(delta);
  let halfSine = sin(delta * 0.5);
  let versine = 2.0 * halfSine * halfSine;
  let nearPole = abs(clampedLatitude) > 45.0;
  let coLatitude = (90.0 - abs(clampedLatitude)) * DEGREES_TO_RADIANS;
  let sinPhi = select(sin(phi), sign(clampedLatitude) * cos(coLatitude), nearPole);
  let cosPhi = select(cos(phi), sin(coLatitude), nearPole);
  let cosTheta = cos(bearing);
  let sinTheta = sin(bearing);
  var latitudeOffset: f32;
  if (cosPhi < 0.05 || delta > 0.5 * cosPhi) {
    // Absolute destination: sin(latitude') and cos(latitude') times the longitude difference.
    let sineTarget = sinPhi * cosDelta + cosPhi * sinDelta * cosTheta;
    let northTerm = cosPhi * cosDelta - sinPhi * sinDelta * cosTheta;
    let targetLatitude = atan2(sineTarget, length(vec2f(northTerm, sinTheta * sinDelta)));
    latitudeOffset = targetLatitude - phi;
  } else {
    // sin(latitude') - sin(phi), without the cancellation of cos(delta) - 1.
    let sineDifference = -sinPhi * versine + cosPhi * sinDelta * cosTheta;
    latitudeOffset = sineDifference / cosPhi;
    for (var iteration = 0u; iteration < 2u; iteration++) {
      let cosMean = max(cos(phi + 0.5 * latitudeOffset), 1e-6);
      latitudeOffset = 2.0 * asin(clamp(sineDifference / (2.0 * cosMean), -1.0, 1.0));
    }
  }
  // The common cosPhi factor of the standard atan2 arguments is divided out, so the longitude
  // stays defined (and finite) at the poles, where cosPhi is exactly zero.
  let longitudeOffset = guardedAtan2(
    sinTheta * sinDelta,
    cosPhi * (1.0 - versine) - sinPhi * sinDelta * cosTheta
  );
  // Clamp only when the ring reaches past a pole, so small offsets keep their precision.
  let rawLatitudeOffset = latitudeOffset * RADIANS_TO_DEGREES;
  let latitudeOffsetDegrees = select(
    rawLatitudeOffset,
    clamp(clampedLatitude + rawLatitudeOffset, -90.0, 90.0) - clampedLatitude,
    abs(clampedLatitude + rawLatitudeOffset) > 90.0
  );
  offsetValues[offsetValuesOffset + 2u * index] = longitudeOffset * RADIANS_TO_DEGREES;
  offsetValues[offsetValuesOffset + 2u * index + 1u] = latitudeOffsetDegrees;`;
    const planarBody = `offsetValues[offsetValuesOffset + 2u * index] = radius * sin(bearing);
  offsetValues[offsetValuesOffset + 2u * index + 1u] = radius * cos(bearing);`;
    const kernelProps: WGSLKernelProps<Parameters> = {
      id: `${this.id}-offsets`,
      operation: 'GPUPointBuffer',
      variant: coordinateSystem,
      bindings: [
        {name: 'pointValues', view: getGraphViewChunks(points)[0], type: 'f32', access: 'read'},
        {name: 'settings', view: parameters, type: 'f32', access: 'read'},
        {
          name: 'offsetValues',
          view: getGraphViewChunks(offsets)[0],
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: offsets.length,
      declarations: `const VERTICES_PER_RING: u32 = ${verticesPerRing}u;
const EARTH_RADIUS_METERS: f32 = ${GPU_POINT_BUFFER_EARTH_RADIUS_METERS};
const TAU: f32 = 6.283185307179586;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${GUARDED_ATAN2_WGSL}`,
      body: `let pointIndex = index / VERTICES_PER_RING;
  let vertexIndex = index % VERTICES_PER_RING;
  let bearing = TAU * f32(vertexIndex) / f32(VERTICES_PER_RING);
  let radius = settings[settingsOffset];
  let centerLongitude = pointValues[pointValuesOffset + 2u * pointIndex];
  let centerLatitude = pointValues[pointValuesOffset + 2u * pointIndex + 1u];
  ${coordinateSystem === 'geographic' ? geographicBody : planarBody}`
    };
    if ('buffer' in points && 'buffer' in offsets)
      return [createWGSLKernelNode(graph, kernelProps)];
    const offsetChunks = getGraphViewChunks(offsets);
    const layouts = offsetChunks.map(chunk =>
      getBoundedDispatchLayout(
        this.id,
        chunk.length,
        POINT_BUFFER_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
@group(0) @binding(1) var<storage, read> pointValues: array<f32>;
@group(0) @binding(2) var<storage, read> settings: array<f32>;
@group(0) @binding(3) var<storage, read_write> offsetValues: array<f32>;
const settingsOffset: u32 = ${getViewElementOffset(parameters)}u;
${kernelProps.declarations}
@compute @workgroup_size(${POINT_BUFFER_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  _ = &pointValues;
  _ = &settings;
  _ = &offsetValues;
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${POINT_BUFFER_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${POINT_BUFFER_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let vertexCount = chunkParameters[1] * VERTICES_PER_RING;
  if (index >= vertexCount) { return; }
  let pointValuesOffset = arrayLength(&pointValues) - 2u * chunkParameters[1];
  let offsetValuesOffset = arrayLength(&offsetValues) - 2u * vertexCount;
  ${kernelProps.body}
}`;
    const {slots, nodes: uploadNodes} = createGraphChunkDispatchParameters(graph, {
      id: `${this.id}-chunks`,
      plan: this.chunkPlan
    });
    return [
      ...uploadNodes,
      ...this.chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk => {
          const chunkParameters = slots[chunk.chunkIndex];
          const pointValues = chunk.views['points'];
          const offsetValues = offsetChunks[chunk.chunkIndex];
          return createGPUComputeCommandNode<Parameters>({
            id: `${this.id}-offsets-${chunk.chunkIndex}`,
            workload: {
              operation: 'GPUPointBuffer',
              variant: coordinateSystem,
              commandCount: 1,
              maximumInvocationCount: offsetValues.length,
              maximumWorkgroupCount:
                layouts[chunk.chunkIndex].x *
                layouts[chunk.chunkIndex].y *
                layouts[chunk.chunkIndex].z
            },
            resources: [
              {buffer: chunkParameters, usage: 'storage-read'},
              {buffer: pointValues, usage: 'storage-read'},
              {buffer: parameters, usage: 'storage-read'},
              {buffer: offsetValues, usage: 'storage-read-write'}
            ],
            compile: ({device}) => {
              const kernel = new Kernel(device, {
                id: 'point-buffer-chunk',
                source,
                shaderLayout: {
                  bindings: [
                    {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                    {name: 'pointValues', group: 0, location: 1, type: 'read-only-storage'},
                    {name: 'settings', group: 0, location: 2, type: 'read-only-storage'},
                    {name: 'offsetValues', group: 0, location: 3, type: 'storage'}
                  ]
                }
              });
              return {
                encode: ({computePass, getBuffer}) =>
                  kernel.dispatch(computePass, {
                    bindings: {
                      chunkParameters: getViewBinding(chunkParameters, getBuffer),
                      pointValues: getViewBinding(pointValues, getBuffer),
                      settings: getViewBinding(parameters, getBuffer),
                      offsetValues: getViewBinding(offsetValues, getBuffer)
                    },
                    ...layouts[chunk.chunkIndex]
                  }),
                destroy: () => kernel.destroy()
              };
            }
          });
        })
    ];
  }
}
