// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNode, GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../analysis-core/contracts/earth';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {validatePackedView} from '../analysis-core/kernels/gpu-contributor-utils';

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
  assert(Number.isFinite(radius) && radius > 0);
  target[0] = radius;
  target[1] = 0;
  return target;
}

export type GPUPointBufferProps = {
  /** Prefix for node IDs. Defaults to `'point-buffer'`. */
  id?: string;
  /** Point centers: longitude and latitude in degrees (`'geographic'`) or planar x and y. */
  points: GraphDataView<'float32x2'>;
  /** Per-frame parameters with at least {@link GPU_POINT_BUFFER_PARAMETER_LENGTH} float32 values. */
  parameters: GraphDataView<'float32'>;
  /**
   * Ring vertex offsets from the point center, `verticesPerRing` rows per point in point order.
   * Geographic offsets are degrees of longitude and latitude, so a float64 consumer adds them to
   * the center without losing the float32 precision of a small radius.
   */
  offsets: GraphDataView<'float32x2'>;
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
 */
export class GPUPointBuffer implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointBufferProps) {
    this.id = props.id ?? 'point-buffer';
    validatePackedView(props.points, ['float32x2'], `${this.id} points`);
    validatePackedView(props.offsets, ['float32x2'], `${this.id} offsets`);
    // Rings need at least a triangle and one offset row per ring vertex.
    assert(Number.isInteger(props.verticesPerRing) && props.verticesPerRing >= 3);
    assert(props.offsets.length === props.points.length * props.verticesPerRing);
    // Parameters hold the radius.
    validatePackedView(props.parameters, ['float32'], `${this.id} parameters`);
    assert(props.parameters.length >= GPU_POINT_BUFFER_PARAMETER_LENGTH);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {points, parameters, offsets, verticesPerRing, coordinateSystem} = this.props;
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
  let longitudeOffset = atan2(
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
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${this.id}-offsets`,
        operation: 'GPUPointBuffer',
        variant: coordinateSystem,
        bindings: [
          {name: 'pointValues', view: points, type: 'f32', access: 'read'},
          {name: 'settings', view: parameters, type: 'f32', access: 'read'},
          {name: 'offsetValues', view: offsets, type: 'f32', access: 'read_write'}
        ],
        invocationCount: offsets.length,
        declarations: `const VERTICES_PER_RING: u32 = ${verticesPerRing}u;
const EARTH_RADIUS_METERS: f32 = ${GPU_POINT_BUFFER_EARTH_RADIUS_METERS};
const TAU: f32 = 6.283185307179586;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;`,
        body: `let pointIndex = index / VERTICES_PER_RING;
  let vertexIndex = index % VERTICES_PER_RING;
  let bearing = TAU * f32(vertexIndex) / f32(VERTICES_PER_RING);
  let radius = settings[settingsOffset];
  let centerLongitude = pointValues[pointValuesOffset + 2u * pointIndex];
  let centerLatitude = pointValues[pointValuesOffset + 2u * pointIndex + 1u];
  ${coordinateSystem === 'geographic' ? geographicBody : planarBody}`
      })
    ];
  }
}
