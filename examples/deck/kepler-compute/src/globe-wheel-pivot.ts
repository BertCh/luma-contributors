// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact zoom-to-cursor for Deck's GlobeViewport.
 *
 * Deck's native globe wheel zoom rotates the camera by a damped fraction of the error and does not
 * account for the latitude dependent scale (`zoomAdjust`), so the geography under the cursor
 * drifts. This module intersects the cursor ray with the radius-256 common-space sphere to find the
 * pivot, then solves for the view center that puts the same geography back under the cursor at the
 * new zoom.
 *
 * Solver: Levenberg-Marquardt on the two unknowns (longitude, latitude) with residual
 * `unitVector(hit(pixel)) - unitVector(target)`. A 3D residual avoids the antimeridian wrap and the
 * pole singularity of a longitude difference. The Jacobian is a forward difference of the real
 * viewport projection (so swipe sub-rectangle projections stay exact); the zoom is re-derived per
 * trial latitude so that the apparent scale (`zoom - zoomAdjust(latitude)`) stays constant. It
 * normally converges in 2 to 4 iterations. Failure to converge or a ray that leaves the sphere
 * returns `null`, and callers fall back to Deck's native behavior.
 */

import {MAX_LATITUDE} from '@math.gl/web-mercator';

import {clamp, safeMod, toDegrees, toRadians} from '@math.gl/core';

/** Radius of Deck's globe in common space. */
export const GLOBE_COMMON_RADIUS = 256;

const MAX_ITERATIONS = 24;
const MAX_STEP_HALVINGS = 10;
const CONVERGENCE_TOLERANCE = 1e-12;
const JACOBIAN_STEP_DEGREES = 1e-5;
const DAMPING = 1e-12;

export type Vector3 = [number, number, number];

/** The part of Deck's GlobeViewport used here. */
export type GlobeViewportLike = {
  /** Column-major matrix mapping `[pixelX, pixelY (top-left), ndcDepth, 1]` to common space. */
  pixelUnprojectionMatrix: ArrayLike<number>;
};

export type GlobeViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  bearing?: number;
  pitch?: number;
};

export type GlobePivotInput<ViewState extends GlobeViewState> = {
  /** Builds a viewport (null when the view is not a globe, such as above Deck's Mercator switch). */
  makeViewport: (viewState: ViewState) => GlobeViewportLike | null;
  /** View state before the zoom step. */
  oldViewState: ViewState;
  /** View state produced by the native zoom step; only its zoom change is used. */
  newViewState: ViewState;
  /** Cursor in viewport pixels, origin top-left. */
  pixel: readonly [number, number];
};

/** Deck's latitude dependent zoom offset (`zoomAdjust(latitude, true)` in globe-viewport). */
export function getGlobeZoomAdjustment(latitude: number): number {
  const clampedLatitude = clamp(latitude, -MAX_LATITUDE, MAX_LATITUDE);
  return Math.log2(Math.PI * Math.cos(toRadians(clampedLatitude)));
}

function transformPoint(matrix: ArrayLike<number>, point: readonly number[]): Vector3 {
  const [x, y, z] = point;
  const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  return [
    (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w,
    (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w,
    (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) / w
  ];
}

/** Nearest intersection of a ray with an origin-centered sphere, or null on a miss or when behind. */
export function intersectRaySphere(
  origin: readonly number[],
  direction: readonly number[],
  radius: number
): Vector3 | null {
  const directionLength = Math.hypot(direction[0], direction[1], direction[2]);
  if (!(directionLength > 0)) return null;
  const unit = [
    direction[0] / directionLength,
    direction[1] / directionLength,
    direction[2] / directionLength
  ];
  const halfB = origin[0] * unit[0] + origin[1] * unit[1] + origin[2] * unit[2];
  const c = origin[0] ** 2 + origin[1] ** 2 + origin[2] ** 2 - radius * radius;
  const discriminant = halfB * halfB - c;
  if (discriminant < 0) return null;
  const distance = -halfB - Math.sqrt(discriminant);
  if (distance < 0) return null;
  return [
    origin[0] + unit[0] * distance,
    origin[1] + unit[1] * distance,
    origin[2] + unit[2] * distance
  ];
}

/** Common-space point where the cursor ray first meets the globe, or null when it misses. */
export function getGlobeRayHit(
  viewport: GlobeViewportLike,
  pixel: readonly [number, number]
): Vector3 | null {
  const start = transformPoint(viewport.pixelUnprojectionMatrix, [pixel[0], pixel[1], -1]);
  const end = transformPoint(viewport.pixelUnprojectionMatrix, [pixel[0], pixel[1], 1]);
  if (!start.every(Number.isFinite) || !end.every(Number.isFinite)) return null;
  return intersectRaySphere(
    start,
    [end[0] - start[0], end[1] - start[1], end[2] - start[2]],
    GLOBE_COMMON_RADIUS
  );
}

/** Same convention as GlobeViewport.projectPosition at zero elevation, on the unit sphere. */
export function getUnitVectorFromLngLat(longitude: number, latitude: number): Vector3 {
  const lambda = toRadians(longitude);
  const phi = toRadians(latitude);
  const cosPhi = Math.cos(phi);
  return [Math.sin(lambda) * cosPhi, -Math.cos(lambda) * cosPhi, Math.sin(phi)];
}

/** Inverse of {@link getUnitVectorFromLngLat}; accepts any positive length. */
export function getLngLatFromCommonPosition(position: readonly number[]): [number, number] {
  const length = Math.hypot(position[0], position[1], position[2]);
  return [
    toDegrees(Math.atan2(position[0], -position[1])),
    toDegrees(Math.asin(clamp(position[2] / length, -1, 1)))
  ];
}

/** Longitude and latitude under the cursor, or null when the ray misses the globe. */
export function getGlobeLngLatAtPixel(
  viewport: GlobeViewportLike,
  pixel: readonly [number, number]
): [number, number] | null {
  const hit = getGlobeRayHit(viewport, pixel);
  return hit && getLngLatFromCommonPosition(hit);
}

/** Maps a longitude to [-180, 180). */
export function normalizeLongitude(longitude: number): number {
  return safeMod(longitude + 180, 360) - 180;
}

/**
 * Finds the view center whose viewport puts `target` under `pixel`.
 * @param getHit Common-space ray hit under the pixel for a trial center, or null on a miss.
 * @returns Center `[longitude, latitude]`, or null when the solve does not converge.
 */
export function solveGlobeCenter(
  getHit: (longitude: number, latitude: number) => Vector3 | null,
  target: readonly [number, number],
  initialCenter: readonly [number, number]
): [number, number] | null {
  const targetVector = getUnitVectorFromLngLat(target[0], target[1]);
  const evaluate = (longitude: number, latitude: number): Vector3 | null => {
    const hit = getHit(longitude, latitude);
    if (!hit) return null;
    const length = Math.hypot(hit[0], hit[1], hit[2]);
    return [
      hit[0] / length - targetVector[0],
      hit[1] / length - targetVector[1],
      hit[2] / length - targetVector[2]
    ];
  };

  let longitude = initialCenter[0];
  let latitude = initialCenter[1];
  let residual = evaluate(longitude, latitude);
  if (!residual) return null;

  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const error = Math.hypot(residual[0], residual[1], residual[2]);
    if (error < CONVERGENCE_TOLERANCE) {
      return [normalizeLongitude(longitude), latitude];
    }

    // Step toward the pole's far side when latitude is at the limit so the difference stays inside [-90, 90].
    const latitudeStep =
      latitude + JACOBIAN_STEP_DEGREES > 90 ? -JACOBIAN_STEP_DEGREES : JACOBIAN_STEP_DEGREES;
    const longitudeResidual = evaluate(longitude + JACOBIAN_STEP_DEGREES, latitude);
    const latitudeResidual = evaluate(longitude, latitude + latitudeStep);
    if (!longitudeResidual || !latitudeResidual) return null;
    const columnLongitude = longitudeResidual.map(
      (value, index) => (value - residual![index]) / JACOBIAN_STEP_DEGREES
    );
    const columnLatitude = latitudeResidual.map(
      (value, index) => (value - residual![index]) / latitudeStep
    );

    // Normal equations (JtJ + damping) delta = -Jt residual.
    let a = 0;
    let b = 0;
    let d = 0;
    let gradientLongitude = 0;
    let gradientLatitude = 0;
    for (let index = 0; index < 3; index++) {
      a += columnLongitude[index] * columnLongitude[index];
      b += columnLongitude[index] * columnLatitude[index];
      d += columnLatitude[index] * columnLatitude[index];
      gradientLongitude -= columnLongitude[index] * residual[index];
      gradientLatitude -= columnLatitude[index] * residual[index];
    }
    const regularization = DAMPING + 1e-9 * (a + d);
    a += regularization;
    d += regularization;
    const determinant = a * d - b * b;
    if (!(Math.abs(determinant) > 0)) return null;
    const stepLongitude = (d * gradientLongitude - b * gradientLatitude) / determinant;
    const stepLatitude = (a * gradientLatitude - b * gradientLongitude) / determinant;

    // Backtrack until the error decreases and the ray still hits the globe.
    let scale = 1;
    let accepted = false;
    for (let halving = 0; halving <= MAX_STEP_HALVINGS; halving++, scale /= 2) {
      const nextLongitude = longitude + stepLongitude * scale;
      const nextLatitude = clamp(latitude + stepLatitude * scale, -90, 90);
      const nextResidual = evaluate(nextLongitude, nextLatitude);
      if (nextResidual && Math.hypot(nextResidual[0], nextResidual[1], nextResidual[2]) < error) {
        longitude = nextLongitude;
        latitude = nextLatitude;
        residual = nextResidual;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      // Stalled at floating point precision is acceptable; anything else is a failure.
      return error < 1e-9 ? [normalizeLongitude(longitude), latitude] : null;
    }
  }
  const finalError = Math.hypot(residual[0], residual[1], residual[2]);
  return finalError < 1e-9 ? [normalizeLongitude(longitude), latitude] : null;
}

/**
 * Returns the view state after a wheel zoom that keeps the geography under the cursor fixed, or
 * null when the cursor ray misses the globe, the zoom did not change, or the solve fails.
 * Bearing and pitch are preserved from the old view state.
 */
export function getExactGlobeWheelPivot<ViewState extends GlobeViewState>(
  input: GlobePivotInput<ViewState>
): ViewState | null {
  const {makeViewport, oldViewState, newViewState, pixel} = input;
  const zoomDelta = newViewState.zoom - oldViewState.zoom;
  if (!zoomDelta) return null;

  const oldViewport = makeViewport(oldViewState);
  const target = oldViewport && getGlobeLngLatAtPixel(oldViewport, pixel);
  if (!target) return null;

  const apparentZoom =
    oldViewState.zoom - getGlobeZoomAdjustment(oldViewState.latitude) + zoomDelta;
  const getViewState = (longitude: number, latitude: number): ViewState => ({
    ...oldViewState,
    longitude,
    latitude,
    zoom: apparentZoom + getGlobeZoomAdjustment(latitude)
  });
  const getHit = (longitude: number, latitude: number): Vector3 | null => {
    const viewport = makeViewport(getViewState(longitude, latitude));
    return viewport && getGlobeRayHit(viewport, pixel);
  };
  // Start from the native result, whose anchoring is close; retry from the old center if it misses.
  const center =
    solveGlobeCenter(getHit, target, [newViewState.longitude, newViewState.latitude]) ??
    solveGlobeCenter(getHit, target, [oldViewState.longitude, oldViewState.latitude]);
  return center && getViewState(center[0], center[1]);
}
