// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, toRadians} from '@math.gl/core';
import {getLatitudeFromMercatorY} from '../../engine/web-mercator';

/**
 * Display-only Winkel Tripel, centered on Greenwich with standard parallel acos(2 / pi).
 * Dataset coordinates and geodesic distances remain unchanged. The host uses a Mercator camera:
 * encode the projected plane through inverse Mercator so its forward transform cancels out.
 * These display coordinates are also used by annotations and picking, never by analysis.
 */
const DEGREES_TO_RADIANS = toRadians(1);

/** Winkel Tripel plane in degree-scaled radians, with equal horizontal and vertical scale. */
export function projectNetworkPosition(longitude: number, latitude: number): [number, number] {
  const halfLongitude = longitude * DEGREES_TO_RADIANS * 0.5;
  const latitudeRadians = latitude * DEGREES_TO_RADIANS;
  const cosineLatitude = Math.cos(latitudeRadians);
  const angle = Math.acos(clamp(cosineLatitude * Math.cos(halfLongitude), -1, 1));
  const inverseSinc = angle < 1e-6 ? 1 : angle / Math.sin(angle);
  return [
    longitude / Math.PI +
      (cosineLatitude * Math.sin(halfLongitude) * inverseSinc) / DEGREES_TO_RADIANS,
    (latitude + (Math.sin(latitudeRadians) * inverseSinc) / DEGREES_TO_RADIANS) * 0.5
  ];
}

/** Encodes a planar point for the host camera, without introducing Mercator stretching. */
export function encodeNetworkPosition(position: readonly [number, number]): [number, number] {
  return [position[0], getLatitudeFromMercatorY(toRadians(position[1]))];
}

/** Geographic coordinate to the display coordinate shared by layers, labels and picking. */
export function projectNetworkDisplayPosition(
  longitude: number,
  latitude: number
): [number, number] {
  return encodeNetworkPosition(projectNetworkPosition(longitude, latitude));
}

/** Matches the CPU projection; arcs stay in geographic coordinates until the vertex shader. */
export const NETWORK_PROJECTION_WGSL = /* wgsl */ `
fn projectNetworkPosition(coordinate: vec2<f32>) -> vec2<f32> {
  let radiansPerDegree = 0.017453292519943295;
  let halfLongitude = coordinate.x * radiansPerDegree * 0.5;
  let latitude = coordinate.y * radiansPerDegree;
  let cosineLatitude = cos(latitude);
  let angle = acos(clamp(cosineLatitude * cos(halfLongitude), -1.0, 1.0));
  var inverseSinc = 1.0;
  if (angle > 0.000001) {
    inverseSinc = angle / sin(angle);
  }
  return vec2<f32>(
    coordinate.x / 3.141592653589793 + cosineLatitude * sin(halfLongitude) * inverseSinc / radiansPerDegree,
    (coordinate.y + sin(latitude) * inverseSinc / radiansPerDegree) * 0.5
  );
}

fn encodeNetworkPosition(position: vec2<f32>) -> vec2<f32> {
  let radiansPerDegree = 0.017453292519943295;
  return vec2<f32>(position.x, atan(sinh(position.y * radiansPerDegree)) / radiansPerDegree);
}
`;
