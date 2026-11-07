// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {unwrapLongitudeNear} from './ocean-drifters-data';
import {clamp, toRadians} from '@math.gl/core';

export type PacificWindow = {
  id: 'A' | 'B';
  label: string;
  longitudeLatitude: [number, number][];
  unitPositions: Float32Array;
};

export type PacificBoundaryInputs = {
  eventLongitudeLatitude: Float32Array;
  eventUnitPositions: Float32Array;
  eventCount: number;
  earthquakeArchiveCount: number;
  archiveRowCount: number;
  windows: [PacificWindow, PacificWindow];
  coastlinePaths: [number, number][][];
};

type CoastlineCollection = {
  features: Array<{
    geometry:
      | {type: 'LineString'; coordinates: number[][]}
      | {type: 'MultiLineString'; coordinates: number[][][]};
  }>;
};

function toUnitPosition(longitude: number, latitude: number): [number, number, number] {
  const longitudeRadians = toRadians(longitude);
  const latitudeRadians = toRadians(latitude);
  const cosineLatitude = Math.cos(latitudeRadians);
  return [
    cosineLatitude * Math.cos(longitudeRadians),
    cosineLatitude * Math.sin(longitudeRadians),
    Math.sin(latitudeRadians)
  ];
}

function makeWindow(
  id: PacificWindow['id'],
  label: string,
  longitudeLatitude: [number, number][]
): PacificWindow {
  const unit = longitudeLatitude.map(([longitude, latitude]) =>
    toUnitPosition(longitude, latitude)
  );
  const center = unit.reduce(
    (sum, position) => [sum[0] + position[0], sum[1] + position[1], sum[2] + position[2]],
    [0, 0, 0]
  );
  const centerLength = Math.hypot(...center);
  if (centerLength < 1e-8) throw new Error(`Pacific window ${id} has no open-hemisphere center.`);
  const unitCenter = center.map(value => value / centerLength);
  let orientation = 0;
  for (let index = 0; index < unit.length; index++) {
    const first = unit[index];
    const second = unit[(index + 1) % unit.length];
    const cross = [
      first[1] * second[2] - first[2] * second[1],
      first[2] * second[0] - first[0] * second[2],
      first[0] * second[1] - first[1] * second[0]
    ];
    const centerSide =
      cross[0] * unitCenter[0] + cross[1] * unitCenter[1] + cross[2] * unitCenter[2];
    if (centerSide <= 1e-8)
      throw new Error(`Pacific window ${id} must be counterclockwise and convex.`);
    const edgeLength = Math.acos(
      clamp(first[0] * second[0] + first[1] * second[1] + first[2] * second[2], -1, 1)
    );
    if (edgeLength >= Math.PI - 1e-6)
      throw new Error(`Pacific window ${id} must use minor great-circle edges.`);
    const normalLength = Math.hypot(...cross);
    for (const position of unit) {
      const edgeSide =
        (cross[0] * position[0] + cross[1] * position[1] + cross[2] * position[2]) / normalLength;
      if (edgeSide < -1e-7) throw new Error(`Pacific window ${id} is not convex on the sphere.`);
    }
    orientation += centerSide;
  }
  if (
    unit.some(
      position =>
        position[0] * unitCenter[0] + position[1] * unitCenter[1] + position[2] * unitCenter[2] <=
        1e-5
    )
  ) {
    throw new Error(`Pacific window ${id} must fit strictly inside an open hemisphere.`);
  }
  if (orientation <= 0) throw new Error(`Pacific window ${id} orientation check failed.`);
  return {id, label, longitudeLatitude, unitPositions: Float32Array.from(unit.flat())};
}

function makeCoastlinePaths(collection: CoastlineCollection): [number, number][][] {
  const paths: [number, number][][] = [];
  for (const feature of collection.features) {
    const geometry = feature.geometry;
    const lines: number[][][] =
      geometry.type === 'LineString' ? [geometry.coordinates] : geometry.coordinates;
    for (const line of lines) {
      let path: [number, number][] = [];
      let previousLongitude: number | null = null;
      for (const [longitude, latitude] of line) {
        const adjustedLongitude: number =
          previousLongitude === null
            ? unwrapLongitudeNear(180, longitude)
            : unwrapLongitudeNear(previousLongitude, longitude);
        if (previousLongitude !== null && Math.abs(adjustedLongitude - previousLongitude) > 60) {
          if (path.length > 1) paths.push(path);
          path = [];
        }
        path.push([adjustedLongitude, latitude]);
        previousLongitude = adjustedLongitude;
      }
      if (path.length > 1) paths.push(path);
    }
  }
  return paths;
}

/** Converts only archived type-0 source rows into unit XYZ and constructs the analyst windows. */
export async function loadPacificBoundaryInputs(
  earthquakes: LoadedDataset,
  naturalEarth: LoadedDataset,
  signal: AbortSignal
): Promise<PacificBoundaryInputs> {
  const sourcePositions = earthquakes.column<Float32Array>('position');
  const eventTypes = earthquakes.column<Uint8Array>('type');
  const eventLongitudeLatitude: number[] = [];
  const eventUnitPositions: number[] = [];
  for (let row = 0; row < earthquakes.count; row++) {
    if (eventTypes[row] !== 0) continue;
    const longitude = sourcePositions[row * 2];
    const latitude = sourcePositions[row * 2 + 1];
    eventLongitudeLatitude.push(longitude, latitude);
    eventUnitPositions.push(...toUnitPosition(longitude, latitude));
  }

  const windows: [PacificWindow, PacificWindow] = [
    makeWindow('A', 'Window A · 145°E–150°W · 15°S–55°N', [
      [145, -15],
      [-150, -15],
      [-150, 55],
      [145, 55]
    ]),
    makeWindow('B', 'Window B · 170°E–115°W · 35°S–30°N', [
      [170, -35],
      [-115, -35],
      [-115, 30],
      [170, 30]
    ])
  ];
  const coastline = await fetch(naturalEarth.fileUrl('ne_110m_coastline.geojson'), {signal});
  if (!coastline.ok)
    throw new Error(`Natural Earth coastline request failed (${coastline.status}).`);
  const collection = (await coastline.json()) as CoastlineCollection;
  return {
    eventLongitudeLatitude: Float32Array.from(eventLongitudeLatitude),
    eventUnitPositions: Float32Array.from(eventUnitPositions),
    eventCount: eventLongitudeLatitude.length / 2,
    earthquakeArchiveCount: eventTypes.reduce(
      (count, eventType) => count + Number(eventType === 0),
      0
    ),
    archiveRowCount: earthquakes.count,
    windows,
    coastlinePaths: makeCoastlinePaths(collection)
  };
}
