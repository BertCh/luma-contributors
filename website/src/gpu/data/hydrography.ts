// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import sample from './natural-earth-geometry.json';

type World = {width: number; height: number};
export const cityCoordinates = sample.cities;
const shoreProfiles = Object.values(sample.shoreRadii);
const convexProfiles = Object.values(sample.convexShores);
const riverReaches = Object.values(sample.rivers);
export const SHORE_PROFILE_COUNT = shoreProfiles.length;
export const RIVER_REACH_COUNT = riverReaches.length;

/** Six actual horizon vertices of a lake in CCW order after convexification. */
export function createConvexShoreRing(index: number, radius: number): number[][] {
  return convexProfiles[index % convexProfiles.length].map(([x, y]) => [x * radius, y * radius]);
}

/** An open ring from the sampled shoreline radii. A blend of zero gives a convex control ring. */
export function createShoreRing(
  profileIndex: number,
  centerX: number,
  centerY: number,
  radius: number,
  vertexCount: number,
  blend = 1
): number[][] {
  const profile = shoreProfiles[profileIndex % shoreProfiles.length];
  return Array.from({length: vertexCount}, (_, vertex) => {
    const fraction = vertex / vertexCount;
    const position = fraction * profile.length;
    const index = Math.floor(position);
    const weight = position - index;
    const distance = profile[index] * (1 - weight) + profile[(index + 1) % profile.length] * weight;
    const length = radius * (1 + blend * (distance - 1));
    const angle = fraction * 2 * Math.PI;
    return [centerX + Math.cos(angle) * length, centerY + Math.sin(angle) * length];
  });
}

/** The five Great Lakes in a single local equirectangular frame, preserving relative scale. */
export function createGreatLakeRings(
  world: World,
  lake?: keyof typeof sample.greatLakes
): number[][][] {
  const rings = lake ? [sample.greatLakes[lake]] : Object.values(sample.greatLakes);
  const longitudes = rings.flat().map(([longitude]) => longitude);
  const latitudes = rings.flat().map(([, latitude]) => latitude);
  const lowX = Math.min(...longitudes);
  const highX = Math.max(...longitudes);
  const lowY = Math.min(...latitudes);
  const highY = Math.max(...latitudes);
  const longitudeScale = Math.cos((((highY + lowY) / 2) * Math.PI) / 180);
  const scale =
    0.93 * Math.min(world.width / ((highX - lowX) * longitudeScale), world.height / (highY - lowY));
  return rings.map(ring =>
    ring.map(([longitude, latitude]) => [
      world.width / 2 + (longitude - (lowX + highX) / 2) * longitudeScale * scale,
      world.height / 2 + (latitude - (lowY + highY) / 2) * scale
    ])
  );
}

/** A river reach sampled along its length and rotated to fit one horizontal comparison panel. */
export function createRiverLine(
  index: number,
  vertexCount: number,
  world: World,
  panel = 0,
  panels = 1
): number[][] {
  return fitRiverReach(
    riverReaches[index % riverReaches.length],
    vertexCount,
    world,
    panel,
    panels
  );
}

/** Short high-sinuosity sections of Missouri and Ohio chosen for testing outside offset joins. */
export function createRiverBend(
  index: number,
  vertexCount: number,
  world: World,
  panel = 0,
  panels = 1
) {
  const reach =
    index % 2 === 0 ? sample.rivers.Missouri.slice(79, 90) : sample.rivers.Ohio.slice(25, 36);
  return fitRiverReach(reach, vertexCount, world, panel, panels);
}

function fitRiverReach(
  reach: number[][],
  vertexCount: number,
  world: World,
  panel: number,
  panels: number
): number[][] {
  const [firstLongitude, firstLatitude] = reach[0];
  const [lastLongitude, lastLatitude] = reach[reach.length - 1];
  const longitudeScale = Math.cos((((firstLatitude + lastLatitude) / 2) * Math.PI) / 180);
  const direction = Math.atan2(
    lastLatitude - firstLatitude,
    (lastLongitude - firstLongitude) * longitudeScale
  );
  const projected = reach.map(([longitude, latitude]) => {
    const x = (longitude - firstLongitude) * longitudeScale;
    const y = latitude - firstLatitude;
    return [
      x * Math.cos(direction) + y * Math.sin(direction),
      y * Math.cos(direction) - x * Math.sin(direction)
    ];
  });
  const distances = [0];
  for (let vertex = 1; vertex < reach.length; vertex++) {
    const [x, y] = projected[vertex];
    const [previousX, previousY] = projected[vertex - 1];
    distances.push(distances[vertex - 1] + Math.hypot(x - previousX, y - previousY));
  }
  const eastings = projected.map(([x]) => x);
  const northings = projected.map(([, y]) => y);
  const lowX = Math.min(...eastings);
  const highX = Math.max(...eastings);
  const lowY = Math.min(...northings);
  const highY = Math.max(...northings);
  const panelHeight = world.height / panels;
  const scale = Math.min(
    (world.width * 0.93) / (highX - lowX),
    (panelHeight * 0.85) / (highY - lowY)
  );
  let next = 1;
  return Array.from({length: vertexCount}, (_, vertex) => {
    const distance = (distances[distances.length - 1] * vertex) / (vertexCount - 1);
    while (next < distances.length - 1 && distances[next] < distance) next++;
    const weight = (distance - distances[next - 1]) / (distances[next] - distances[next - 1]);
    const x = projected[next - 1][0] * (1 - weight) + projected[next][0] * weight;
    const y = projected[next - 1][1] * (1 - weight) + projected[next][1] * weight;
    return [
      world.width / 2 + (x - (lowX + highX) / 2) * scale,
      (panel + 0.5) * panelHeight + (y - (lowY + highY) / 2) * scale
    ];
  });
}
