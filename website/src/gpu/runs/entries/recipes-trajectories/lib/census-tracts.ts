// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import tracts from '../../../../data/recipes/hyde-park-tracts.json';
import {fitLocalMap} from '../../../../data/fit-local-map';
import type {QuadZones} from './quads';

/** The twelve 2022 Hyde Park tracts, with one planar scale for both axes. */
export function createCensusTracts(world: {width: number; height: number}): QuadZones {
  const rings = tracts.map(tract =>
    tract.ring.map(([longitude, latitude]) => [
      (longitude + 87.59) * Math.cos((41.795 * Math.PI) / 180),
      latitude - 41.795
    ])
  );
  const project = fitLocalMap(rings.flat(), world);
  const positions: number[] = [],
    triangles: number[] = [],
    owners: number[] = [],
    outline: number[] = [],
    centers: number[] = [];
  const offsets = [0];
  for (const [unit, ring] of rings.entries()) {
    const start = positions.length / 2;
    const points = ring.map(([x, y]) => project(x, y));
    positions.push(...points.flat());
    for (const [index, point] of points.entries())
      outline.push(...point, ...points[(index + 1) % points.length]);
    centers.push(
      ...[0, 1].map(axis => points.reduce((sum, point) => sum + point[axis], 0) / points.length)
    );
    triangles.push(...tracts[unit].triangles.map(index => start + index));
    owners.push(...Array<number>(tracts[unit].triangles.length / 3).fill(unit));
    offsets.push(positions.length / 2);
  }
  return {
    count: tracts.length,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from({length: tracts.length + 1}, (_, index) => index),
    ringOffsets: Uint32Array.from(offsets),
    triangleIndices: Uint32Array.from(triangles),
    triangleOwners: Uint32Array.from(owners),
    triangleCount: owners.length,
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4,
    centers: Float32Array.from(centers)
  };
}

/** Cook County is constant in this fixture; the last six GEOID digits identify a tract. */
export function getCensusTractCode(unit: number): number {
  return Number(tracts[unit].geoid.slice(-6));
}

export function getCensusPopulation(unit: number): [number, number] {
  const {focal, population} = tracts[unit];
  return [focal, population - focal];
}
