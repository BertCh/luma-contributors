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
import {
  GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN,
  GPU_GEODESIC_MAXIMUM_ITERATIONS
} from './geodesic-kernels';
import {GPU_RHUMB_MAX_LATITUDE} from './rhumb-wgsl';
import {
  GPUGeodesicPairsOperation,
  captureGeodesicPairsOperationProps,
  type GeodesicPairsOperationProps
} from './gpu-geodesic-pairs-operation';

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;

/** Float32 auxiliary-longitude gate; an exact CPU inverse follows only after this gate succeeds. */
function getInverseIterationStatus(
  latitude: number,
  targetLatitude: number,
  longitudeDelta: number,
  angle: number,
  flattening: number,
  maximumIterations: number
): {converged: boolean; iterationLimitReached: boolean} {
  const round = Math.fround;
  const add = (left: number, right: number) => round(left + right);
  const subtract = (left: number, right: number) => round(left - right);
  const multiply = (left: number, right: number) => round(left * right);
  const divide = (left: number, right: number) => round(left / right);
  const sine = (value: number) => round(Math.sin(value));
  const cosine = (value: number) => round(Math.cos(value));
  const radians = round(DEGREES_TO_RADIANS);
  flattening = round(flattening);
  longitudeDelta = multiply(round(longitudeDelta / DEGREES_TO_RADIANS), radians);
  if (latitude === targetLatitude && (longitudeDelta === 0 || Math.abs(latitude) === 90))
    return {converged: true, iterationLimitReached: false};
  if (Math.PI - angle < 1e-6) return {converged: false, iterationLimitReached: false};
  const factor = subtract(1, flattening);
  const gap = (phi: number) => {
    const sinePhi = sine(phi),
      cosinePhi = cosine(phi);
    return round(
      Math.atan(
        divide(
          multiply(multiply(flattening, sinePhi), cosinePhi),
          add(multiply(cosinePhi, cosinePhi), multiply(multiply(factor, sinePhi), sinePhi))
        )
      )
    );
  };
  const phi = multiply(latitude, radians),
    targetPhi = multiply(targetLatitude, radians);
  const reduced = subtract(phi, gap(phi)),
    targetReduced = subtract(targetPhi, gap(targetPhi));
  const cosineReduced = (value: number, degrees: number) => {
    const colatitude = sine(multiply(subtract(90, Math.abs(degrees)), radians));
    if (Math.abs(degrees) <= round(89.999))
      return sine(add(multiply(subtract(90, Math.abs(degrees)), radians), gap(Math.abs(value))));
    const scaledSine = multiply(factor, sine(value));
    return divide(
      colatitude,
      round(Math.sqrt(add(multiply(colatitude, colatitude), multiply(scaledSine, scaledSine))))
    );
  };
  const sineReduced = sine(reduced),
    targetSineReduced = sine(targetReduced);
  const cosineFirst = cosineReduced(phi, latitude),
    cosineSecond = cosineReduced(targetPhi, targetLatitude);
  const deltaPhi = multiply(subtract(targetLatitude, latitude), radians);
  let deltaReduced = subtract(deltaPhi, subtract(gap(targetPhi), gap(phi)));
  if (Math.abs(deltaPhi) < round(0.01)) {
    const midpoint = multiply(0.5, add(phi, targetPhi));
    const sineMidpoint = sine(midpoint),
      cosineMidpoint = cosine(midpoint);
    deltaReduced = divide(
      multiply(deltaPhi, factor),
      add(
        multiply(cosineMidpoint, cosineMidpoint),
        multiply(multiply(multiply(factor, factor), sineMidpoint), sineMidpoint)
      )
    );
  }
  const sineDelta = sine(deltaReduced);
  let auxiliaryLongitude = longitudeDelta;
  for (let iteration = 0; iteration < maximumIterations; iteration++) {
    const sineLongitude = sine(auxiliaryLongitude);
    const sineHalfLongitude = sine(multiply(0.5, auxiliaryLongitude));
    const halfVersine = multiply(multiply(2, sineHalfLongitude), sineHalfLongitude);
    const north = add(sineDelta, multiply(multiply(sineReduced, cosineSecond), halfVersine));
    const east = multiply(cosineSecond, sineLongitude);
    const sineSigma = round(Math.sqrt(add(multiply(east, east), multiply(north, north))));
    if (sineSigma === 0) return {converged: false, iterationLimitReached: false};
    const cosineSigma = add(
      multiply(sineReduced, targetSineReduced),
      multiply(multiply(cosineFirst, cosineSecond), subtract(1, halfVersine))
    );
    const sigma = round(Math.atan2(sineSigma, cosineSigma));
    const sineAlpha = divide(
      multiply(multiply(cosineFirst, cosineSecond), sineLongitude),
      sineSigma
    );
    const squaredCosineAlpha = subtract(1, multiply(sineAlpha, sineAlpha));
    const cosineDoubleSigma =
      squaredCosineAlpha === 0
        ? 0
        : subtract(
            cosineSigma,
            divide(multiply(multiply(2, sineReduced), targetSineReduced), squaredCosineAlpha)
          );
    const correction = multiply(
      multiply(divide(flattening, 16), squaredCosineAlpha),
      add(4, multiply(flattening, subtract(4, multiply(3, squaredCosineAlpha))))
    );
    const previous = auxiliaryLongitude;
    auxiliaryLongitude = add(
      longitudeDelta,
      multiply(
        multiply(multiply(subtract(1, correction), flattening), sineAlpha),
        add(
          sigma,
          multiply(
            multiply(correction, sineSigma),
            add(
              cosineDoubleSigma,
              multiply(
                multiply(correction, cosineSigma),
                add(-1, multiply(multiply(2, cosineDoubleSigma), cosineDoubleSigma))
              )
            )
          )
        )
      )
    );
    if (
      Math.abs(subtract(auxiliaryLongitude, previous)) <=
      multiply(round(4.8e-7), Math.abs(auxiliaryLongitude))
    )
      return {converged: true, iterationLimitReached: false};
  }
  return {converged: false, iterationLimitReached: true};
}
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
  longitude = wrapLongitude(longitude);
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
 * - Without an explicit budget, ellipsoidal distance/bearings use EllipsoidGeodesicExact,
 *   including near antipodes, and valid ellipsoidal rows have converged=1. An explicit
 *   maximumIterations bounds auxiliary-longitude convergence with float32 updates and the
 *   GPU stopping criterion; exhaustion uses spherical distance/bearings
 *   and reports converged=0, approximation=1, iterationLimitReached=1.
 * - Rhumb stretched latitudes use the full float64 closed form, without the short-line/series
 *   approximations or the 0.999999 atanh clamp. The latitude clamp is the GPU's float32 value.
 * - Except for the explicit-budget convergence gate, arithmetic is float64 until output.
 *   Bearings that round to -180 are normalized to +180;
 *   zero bearings are positive zero. Longitude half-turn ties use round-to-even.
 * - Nonfinite coordinates or latitudes outside [-90, 90] emit NaN and converged=0; GPU results
 *   for these inputs are unspecified. Exactly antipodal spherical midpoints are nonunique;
 *   midpoints within 1e-6 radians of an antipode are NaN on both paths.
 * - approximation marks spherical bearing/midpoint conditioning near antipodes on both paths,
 *   and the spherical fallback when an explicit inverse budget does not converge.
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
  spatialContext: GPUSpatialContext,
  maximumIterations?: number
): CPUGeodesicPairsResult {
  if (maximumIterations !== undefined)
    validateCapacity(
      'computeGeodesicPairsOnCPU',
      'maximumIterations',
      maximumIterations,
      1,
      GPU_GEODESIC_MAXIMUM_ITERATIONS
    );
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
    // Measure the small arc to the antipode directly. Subtracting a rounded
    // central angle from pi loses the stored-coordinate cutoff on the GPU.
    const antipodalGap = computeCentralAngle(
      (180 - Math.abs(computeLongitudeDelta(longitude, targetLongitude))) * DEGREES_TO_RADIANS,
      latitude,
      -targetLatitude
    );
    result.approximation[row] = Number(
      metric.model !== 'rhumb' && antipodalGap < GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN
    );
    let distance = angle * metric.radius;
    let initial = computeBearing(longitudeDelta, latitude, targetLatitude);
    let final = computeBearing(-longitudeDelta, targetLatitude, latitude) + 180;
    if (final > 180) final -= 360;
    let midpoint = computeMidpoint(longitude, latitude, targetLongitude, targetLatitude, angle);
    let isConverged = true;
    if (geodesic && maximumIterations !== undefined) {
      const iterationStatus = getInverseIterationStatus(
        latitude,
        targetLatitude,
        longitudeDelta,
        angle,
        metric.ellipsoid.flattening,
        maximumIterations
      );
      isConverged =
        iterationStatus.converged && antipodalGap >= GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN;
      result.approximation[row] = Number(!isConverged);
      result.iterationLimitReached[row] = Number(iterationStatus.iterationLimitReached);
    }
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
    } else if (geodesic && isConverged) {
      if (latitude === targetLatitude && (longitudeDelta === 0 || Math.abs(latitude) === 90)) {
        distance = 0;
        initial = 0;
        final = 0;
      } else {
        const inverse = geodesic.inverse(
          latitude,
          wrapLongitude(longitude),
          targetLatitude,
          wrapLongitude(longitude) + longitudeDelta * RADIANS_TO_DEGREES
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
    result.converged[row] = isConverged && Number.isFinite(distance) ? 1 : 0;
  }
  return result;
}

/** CPU geodesic twin; explicit inverse budgets suppress the exact solve on exhaustion. */
export class CPUGeodesicPairsOperation extends GPUCPUOperation {
  constructor(props: GeodesicPairsOperationProps) {
    props = captureGeodesicPairsOperationProps(props);
    const id = props.id ?? 'geodesic-pairs';
    validateCapacity(id, 'pairCount', props.pairCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations ?? 16,
      1,
      GPU_GEODESIC_MAXIMUM_ITERATIONS
    );
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
        const result = computeGeodesicPairsOnCPU(
          {origins, targets},
          props.spatialContext,
          props.maximumIterations
        );
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
