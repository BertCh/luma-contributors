// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {EllipsoidGeodesicExact} from '@math.gl/geospatial';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  wrapLongitude,
  computeLongitudeDelta,
  computeCosLatitude,
  computeCentralAngle,
  computeBearingComponents
} from '../../analysis-core/kernels/cpu-geodesic';
import {resolveGeodesicSpatialContext} from './gpu-geodesic-pairs';
import {GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN} from './geodesic-kernels';
import {GPU_RHUMB_MAX_LATITUDE} from './rhumb-wgsl';
import {
  GPUGeodesicPairsOperation,
  type GeodesicPairsOperationProps
} from './gpu-geodesic-pairs-operation';

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;

/** Packed longitude/latitude pairs in degrees. */
export type CPUGeodesicPairsInput = {origins: Float32Array; targets: Float32Array};
/** Fixed-size columns in GPUGeodesicPairs layout. Converged is meaningful for ellipsoidal only. */
export type CPUGeodesicPairsResult = {
  distances: Float32Array;
  initialBearings: Float32Array;
  finalBearings: Float32Array;
  midpoints: Float32Array;
  converged: Uint32Array;
  approximation: Uint32Array;
  iterationLimitReached: Uint32Array;
};

function computeBearing(longitudeDelta: number, latitude: number, targetLatitude: number): number {
  const [east, north] = computeBearingComponents(longitudeDelta, latitude, targetLatitude);
  return east === 0 && north === 0 ? 0 : Math.atan2(east, north) * RADIANS_TO_DEGREES;
}

function computeUnitVector(longitude: number, latitude: number): number[] {
  const cosine = computeCosLatitude(latitude);
  return [
    cosine * Math.cos(longitude * DEGREES_TO_RADIANS),
    cosine * Math.sin(longitude * DEGREES_TO_RADIANS),
    Math.sin(latitude * DEGREES_TO_RADIANS)
  ];
}

function computeMidpoint(
  longitude: number,
  latitude: number,
  targetLongitude: number,
  targetLatitude: number,
  angle: number
): number[] {
  if (Math.PI - angle < 1e-6) return [NaN, NaN];
  const first = computeUnitVector(longitude, latitude);
  const second = computeUnitVector(targetLongitude, targetLatitude);
  const weight = angle < 1e-6 ? 0.5 : Math.sin(angle / 2) / Math.sin(angle);
  const vector = first.map((component, index) => component * weight + second[index] * weight);
  if (angle < 1e-6) {
    const length = Math.hypot(...vector);
    for (let index = 0; index < 3; index++) vector[index] /= length;
  }
  const horizontal = Math.hypot(vector[0], vector[1]);
  // Signed zero components at a pole must not select a longitude through atan2.
  if (horizontal === 0) return [0, vector[2] < 0 ? -90 : 90];
  const resultLongitude = Math.atan2(vector[1], vector[0]) * RADIANS_TO_DEGREES;
  return [
    longitude + wrapLongitude(resultLongitude - longitude),
    Math.atan2(vector[2], horizontal) * RADIANS_TO_DEGREES
  ];
}

function computeIsometricLatitude(latitude: number): number {
  const magnitude = Math.abs(latitude);
  const result =
    magnitude < 45
      ? Math.atanh(Math.sin(magnitude * DEGREES_TO_RADIANS))
      : -Math.log(Math.tan(((90 - magnitude) * DEGREES_TO_RADIANS) / 2));
  return latitude < 0 ? -result : result;
}

function computeStretchedDelta(first: number, second: number): number {
  const halfDelta = ((second - first) * DEGREES_TO_RADIANS) / 2;
  const meanCosine =
    first * second >= 0
      ? Math.sin(
          (90 - Math.abs(first) - (Math.abs(second) - Math.abs(first)) / 2) * DEGREES_TO_RADIANS
        )
      : Math.cos(((first + second) * DEGREES_TO_RADIANS) / 2);
  const sine = Math.sin(halfDelta);
  const argument =
    (2 * meanCosine * sine) /
    (2 * sine * sine + computeCosLatitude(first) * computeCosLatitude(second));
  // The difference form avoids cancellation on short lines; the two log form avoids saturation
  // of atanh near the poles. Neither needs the float32 series or the GPU's argument clamp.
  return Math.abs(argument) < 0.5
    ? Math.atanh(argument)
    : computeIsometricLatitude(second) - computeIsometricLatitude(first);
}

function roundBearing(value: number): number {
  const rounded = Math.fround(value);
  return rounded <= -180 ? 180 : rounded === 0 ? 0 : rounded;
}

/**
 * Measures packed pairs in float64. Spherical central angle, bearings, seam wrapping and slerp
 * follow geodesic-wgsl.ts step by step; ellipsoidal inverse calls public math.gl Karney geodesy.
 *
 * What differs from the GPU contributor:
 * - Ellipsoidal distance/bearings use EllipsoidGeodesicExact, including near antipodes. Valid
 *   ellipsoidal rows always have converged=1; the Vincenty iteration cap and sphere fallback are
 *   absent. iterationLimitReached is always zero; maximumIterations on the operation is
 *   accepted but ignored by the CPU solver.
 * - Rhumb stretched latitudes use the full float64 closed form, without the short-line/series
 *   approximations or the 0.999999 atanh clamp. The latitude clamp is the GPU's float32 value.
 * - All arithmetic is float64 until output. Bearings that round to -180 are normalized to +180;
 *   zero bearings are positive zero. Longitude half-turn ties use round-to-even.
 * - Nonfinite coordinates or latitudes outside [-90, 90] emit NaN and converged=0; GPU results
 *   for these inputs are unspecified. Exactly antipodal spherical midpoints are nonunique;
 *   midpoints within 1e-6 radians of an antipode are NaN on both paths.
 * - approximation marks spherical bearing/midpoint conditioning near antipodes on both paths;
 *   the CPU exact ellipsoidal inverse never needs a sphere fallback.
 * - Midpoints remain spherical for the ellipsoidal model, as on the GPU. Rhumb midpoint latitude
 *   is the original (unclamped) mean, matching the current kernel.
 * - For (20,90) -> (20,90), the CPU midpoint is [0,90], choosing longitude zero via JavaScript
 *   atan2(0, 0). Pole midpoint longitudes follow JavaScript's signed-zero atan2 convention.
 *   Both paths guard zero-axis atan2 and select longitude zero for the pole midpoint.
 *
 * @throws If origins/targets are not aligned packed float32x2 rows or the context is unsupported.
 */
export function computeGeodesicPairsOnCPU(
  input: CPUGeodesicPairsInput,
  spatialContext: GPUSpatialContext
): CPUGeodesicPairsResult {
  if (input.origins.length % 2 !== 0) throw new Error('origins must contain float32x2 rows');
  if (input.targets.length !== input.origins.length)
    throw new Error('targets length must equal origins length');
  const metric = resolveGeodesicSpatialContext('computeGeodesicPairsOnCPU', spatialContext);
  const count = input.origins.length / 2;
  const result: CPUGeodesicPairsResult = {
    distances: new Float32Array(count).fill(NaN),
    initialBearings: new Float32Array(count).fill(NaN),
    finalBearings: new Float32Array(count).fill(NaN),
    midpoints: new Float32Array(count * 2).fill(NaN),
    converged: new Uint32Array(count),
    approximation: new Uint32Array(count),
    iterationLimitReached: new Uint32Array(count)
  };
  const geodesic =
    metric.model === 'wgs84'
      ? new EllipsoidGeodesicExact({
          semiMajorAxis: metric.ellipsoid.semiMajorAxis,
          semiMinorAxis: metric.ellipsoid.semiMajorAxis * (1 - metric.ellipsoid.flattening)
        })
      : undefined;
  for (let row = 0; row < count; row++) {
    const longitude = input.origins[row * 2];
    const latitude = input.origins[row * 2 + 1];
    const targetLongitude = input.targets[row * 2];
    const targetLatitude = input.targets[row * 2 + 1];
    if (
      ![longitude, latitude, targetLongitude, targetLatitude].every(Number.isFinite) ||
      Math.abs(latitude) > 90 ||
      Math.abs(targetLatitude) > 90
    )
      continue;
    const longitudeDelta = computeLongitudeDelta(longitude, targetLongitude) * DEGREES_TO_RADIANS;
    const angle = computeCentralAngle(longitudeDelta, latitude, targetLatitude);
    result.approximation[row] = Number(
      metric.model !== 'rhumb' && Math.PI - angle < GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN
    );
    let distance = angle * metric.radius;
    let initial = computeBearing(longitudeDelta, latitude, targetLatitude);
    let final = computeBearing(-longitudeDelta, targetLatitude, latitude) + 180;
    if (final > 180) final -= 360;
    let midpoint = computeMidpoint(longitude, latitude, targetLongitude, targetLatitude, angle);
    if (metric.model === 'rhumb') {
      const clamp = Math.fround(GPU_RHUMB_MAX_LATITUDE);
      const first = Math.max(-clamp, Math.min(clamp, latitude));
      const second = Math.max(-clamp, Math.min(clamp, targetLatitude));
      const stretched = computeStretchedDelta(first, second);
      const latitudeDelta = (second - first) * DEGREES_TO_RADIANS;
      const stretchRatio = stretched === 0 ? computeCosLatitude(first) : latitudeDelta / stretched;
      distance = Math.hypot(latitudeDelta, stretchRatio * longitudeDelta) * metric.radius;
      initial =
        stretched === 0 && longitudeDelta === 0
          ? 0
          : Math.atan2(longitudeDelta, stretched || 0) * RADIANS_TO_DEGREES;
      final = initial;
      const mean = (first + second) / 2;
      const firstHalf = computeStretchedDelta(first, mean);
      const secondHalf = computeStretchedDelta(-second, -mean);
      const fraction =
        firstHalf + secondHalf === 0
          ? 0.5
          : Math.max(0, Math.min(1, firstHalf / (firstHalf + secondHalf)));
      midpoint = [
        longitude + fraction * longitudeDelta * RADIANS_TO_DEGREES,
        latitude + (targetLatitude - latitude) / 2
      ];
    } else if (geodesic) {
      if (latitude === targetLatitude && (longitudeDelta === 0 || Math.abs(latitude) === 90)) {
        distance = 0;
        initial = 0;
        final = 0;
      } else {
        const inverse = geodesic.inverse(
          latitude,
          longitude,
          targetLatitude,
          longitude + longitudeDelta * RADIANS_TO_DEGREES
        );
        distance = inverse.s12;
        initial = inverse.azi1;
        final = inverse.azi2;
      }
    }
    result.distances[row] = distance;
    result.initialBearings[row] = roundBearing(initial);
    result.finalBearings[row] = roundBearing(final);
    result.midpoints.set(midpoint, row * 2);
    result.converged[row] = Number.isFinite(distance) ? 1 : 0;
  }
  return result;
}

/** CPU geodesic twin; exact ellipsoidal solver has no Vincenty iteration budget. */
export class CPUGeodesicPairsOperation extends GPUCPUOperation {
  constructor(props: GeodesicPairsOperationProps) {
    const id = props.id ?? 'geodesic-pairs';
    validateCapacity(id, 'pairCount', props.pairCount);
    validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 16);
    const {model} = resolveGeodesicSpatialContext(id, props.spatialContext);
    validateRequirement(
      id,
      Object.values(props.output).some(value => typeof value === 'string') ||
        props.output.status?.converged !== undefined ||
        props.output.status?.approximation !== undefined ||
        props.output.status?.iterationLimitReached !== undefined,
      'output requires at least one column'
    );
    validateRequirement(
      id,
      props.output.status?.converged === undefined || model === 'wgs84',
      "output.status.converged requires metric 'ellipsoidal'"
    );
    const operation = new GPUGeodesicPairsOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const origins = getCPUOperationFloat32Rows(id, inputs, 'origins', 'float32x2');
        const targets = getCPUOperationFloat32Rows(id, inputs, 'targets', 'float32x2');
        if (origins.length !== props.pairCount * 2)
          throw new Error(`${id} origins length must equal pairCount`);
        const result = computeGeodesicPairsOnCPU({origins, targets}, props.spatialContext);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUGeodesicPairsResult]
          ])
        );
      }
    });
  }
}
