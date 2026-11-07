// luma.gl
// SPDX-License-Identifier: MIT AND BSD-2-Clause
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright (c) 2016, Mapbox

const TILE_SIZE = 256;

/**
 * Local ESM port of @mapbox/geo-viewport `bounds` for integer zoom (CommonJS-only upstream, which
 * cannot load in the browser test runner). Matches its rounded SphericalMercator pixel math.
 */
export function getViewportBounds(
  center: readonly [number, number],
  zoom: number,
  dimensions: readonly [number, number]
): [number, number, number, number] {
  const worldSize = TILE_SIZE * Math.pow(2, zoom);
  const halfWorld = worldSize / 2;
  const degreesPerPixel = worldSize / 360;
  const radiansPerPixel = worldSize / (2 * Math.PI);
  const sine = Math.min(Math.max(Math.sin((Math.PI / 180) * center[1]), -0.9999), 0.9999);
  const pixelX = Math.min(Math.round(halfWorld + center[0] * degreesPerPixel), worldSize);
  const pixelY = Math.min(
    Math.round(halfWorld + 0.5 * Math.log((1 + sine) / (1 - sine)) * -radiansPerPixel),
    worldSize
  );
  const getLongitude = (x: number): number => (x - halfWorld) / degreesPerPixel;
  const getLatitude = (y: number): number =>
    (180 / Math.PI) * (2 * Math.atan(Math.exp((y - halfWorld) / -radiansPerPixel)) - 0.5 * Math.PI);
  const west = getLongitude(pixelX - dimensions[0] / 2);
  const east = getLongitude(pixelX + dimensions[0] / 2);
  const north = getLatitude(pixelY - dimensions[1] / 2);
  const south = getLatitude(pixelY + dimensions[1] / 2);
  return [west, south, east, north];
}
