// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Orthographic globe and unit-sphere vector helpers for spherical previews (S2 cells, spherical
// polygons, geodesics). Angles are in degrees.

import {clamp} from './data.ts';
import type {Point} from './types.ts';

export type Vector3 = [number, number, number];

/** Where the globe sits in the scene and which longitude and latitude face the viewer. */
export type GlobeView = {
  centerX: number;
  centerY: number;
  radius: number;
  longitude: number;
  latitude: number;
};

const DEGREES = Math.PI / 180;

/**
 * Orthographic projection of a longitude and latitude. `depth` is the cosine of the angle from the
 * view center: positive on the visible hemisphere, so callers choose their own horizon cutoff.
 */
export function projectGlobe(
  view: GlobeView,
  longitude: number,
  latitude: number
): {point: Point; depth: number} {
  const lambda = (longitude - view.longitude) * DEGREES;
  const phi = latitude * DEGREES;
  const phiCenter = view.latitude * DEGREES;
  const depth =
    Math.sin(phiCenter) * Math.sin(phi) + Math.cos(phiCenter) * Math.cos(phi) * Math.cos(lambda);
  return {
    point: [
      view.centerX + view.radius * Math.cos(phi) * Math.sin(lambda),
      view.centerY -
        view.radius *
          (Math.cos(phiCenter) * Math.sin(phi) -
            Math.sin(phiCenter) * Math.cos(phi) * Math.cos(lambda))
    ],
    depth
  };
}

/** Unit vector of a longitude and latitude. */
export function toVector(longitude: number, latitude: number): Vector3 {
  const lambda = longitude * DEGREES;
  const phi = latitude * DEGREES;
  return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
}

/** Longitude and latitude of a unit vector. */
export function toLonLat(vector: Vector3): Point {
  return [Math.atan2(vector[1], vector[0]) / DEGREES, Math.asin(clamp(vector[2], -1, 1)) / DEGREES];
}

export function getVectorDot(first: Vector3, second: Vector3): number {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

export function getVectorCross(first: Vector3, second: Vector3): Vector3 {
  return [
    first[1] * second[2] - first[2] * second[1],
    first[2] * second[0] - first[0] * second[2],
    first[0] * second[1] - first[1] * second[0]
  ];
}

/** Spherical linear interpolation between two unit vectors. */
export function slerpVectors(first: Vector3, second: Vector3, amount: number): Vector3 {
  const angle = Math.acos(clamp(getVectorDot(first, second), -1, 1));
  if (angle < 1e-9) return first;
  const sine = Math.sin(angle);
  const firstWeight = Math.sin((1 - amount) * angle) / sine;
  const secondWeight = Math.sin(amount * angle) / sine;
  return [
    first[0] * firstWeight + second[0] * secondWeight,
    first[1] * firstWeight + second[1] * secondWeight,
    first[2] * firstWeight + second[2] * secondWeight
  ];
}
