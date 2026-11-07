// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';

export const EARTHQUAKE_PAIR_ORIGIN: [number, number] = [140.5, 36.5];
export const EARTHQUAKE_PAIR_BOUNDS = {
  minimumX: -135_000,
  minimumY: -166_000,
  maximumX: 135_000,
  maximumY: 166_000
} as const;
export const EARTHQUAKE_PAIR_DEPTH_THRESHOLD_KM = 70;
export const EARTHQUAKE_PAIR_MAXIMUM_RADIUS_METERS = 80_000;
export const EARTHQUAKE_PAIR_RADIUS_COUNT = 16;
export const EARTHQUAKE_PAIR_SPHERE_APPROXIMATION =
  'spherical azimuthal-equidistant (mean Earth radius)';

export type EarthquakePairEvent = {
  position: [number, number];
  depthKm: number;
  projectedPosition: [number, number];
  depthGroup: 0 | 1;
  exactTenKilometers: boolean;
  sourceRow: number;
  selectedIndex: number;
};

export type EarthquakePairData = {
  events: EarthquakePairEvent[];
  positions: Float32Array;
  depths: Float32Array;
  depthGroups: Uint32Array;
  allEventsMask: Uint32Array;
  excludingTenKilometersMask: Uint32Array;
  shallowCount: number;
  deepCount: number;
  exactTenKilometerCount: number;
  duplicateCoordinatePairs: number;
  rectanglePath: [number, number][];
  rectangleAreaSquareKilometers: number;
  projection: ReturnType<typeof createAnalysisFrameProjection>;
  depthExtent: readonly [number, number];
  sourceCount: number;
};

/** Select all type-0 archive rows within a fixed projected rectangle, retaining their file order. */
export function buildEarthquakePairData(dataset: LoadedDataset): EarthquakePairData {
  const projection = createAnalysisFrameProjection({
    kind: 'azimuthal-equidistant',
    origin: EARTHQUAKE_PAIR_ORIGIN
  });
  const sourcePositions = dataset.column<Float32Array>('position');
  const sourceDepths = dataset.column<Float32Array>('depth');
  const sourceTypes = dataset.column<Uint8Array>('type');
  const events: EarthquakePairEvent[] = [];
  let shallowCount = 0;
  let deepCount = 0;
  let exactTenKilometerCount = 0;
  const coordinateCounts = new Map<string, number>();

  for (let row = 0; row < dataset.count; row++) {
    if (sourceTypes[row] !== 0) continue;
    const longitude = sourcePositions[row * 2];
    const latitude = sourcePositions[row * 2 + 1];
    const [positionX, positionY] = projection.project(longitude, latitude);
    if (
      positionX < EARTHQUAKE_PAIR_BOUNDS.minimumX ||
      positionX > EARTHQUAKE_PAIR_BOUNDS.maximumX ||
      positionY < EARTHQUAKE_PAIR_BOUNDS.minimumY ||
      positionY > EARTHQUAKE_PAIR_BOUNDS.maximumY
    )
      continue;

    const depthKm = sourceDepths[row];
    const depthGroup = depthKm < EARTHQUAKE_PAIR_DEPTH_THRESHOLD_KM ? 0 : 1;
    const exactTenKilometers = depthKm === 10;
    shallowCount += Number(depthGroup === 0);
    deepCount += Number(depthGroup === 1);
    exactTenKilometerCount += Number(exactTenKilometers);
    const coordinateKey = `${longitude},${latitude}`;
    coordinateCounts.set(coordinateKey, (coordinateCounts.get(coordinateKey) ?? 0) + 1);
    events.push({
      position: [longitude, latitude],
      depthKm,
      projectedPosition: [positionX, positionY],
      depthGroup,
      exactTenKilometers,
      sourceRow: row,
      selectedIndex: events.length
    });
  }

  let duplicateCoordinatePairs = 0;
  for (const count of coordinateCounts.values()) {
    duplicateCoordinatePairs += (count * (count - 1)) / 2;
  }
  const positions = new Float32Array(events.length * 2);
  const depths = new Float32Array(events.length);
  const depthGroups = new Uint32Array(events.length);
  const allEventsMask = new Uint32Array(events.length);
  const excludingTenKilometersMask = new Uint32Array(events.length);
  allEventsMask.fill(1);
  let minimumDepth = Number.POSITIVE_INFINITY;
  let maximumDepth = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < events.length; index++) {
    const event = events[index];
    positions[index * 2] = event.projectedPosition[0];
    positions[index * 2 + 1] = event.projectedPosition[1];
    depths[index] = event.depthKm;
    depthGroups[index] = event.depthGroup;
    excludingTenKilometersMask[index] = Number(!event.exactTenKilometers);
    minimumDepth = Math.min(minimumDepth, event.depthKm);
    maximumDepth = Math.max(maximumDepth, event.depthKm);
  }
  const rectanglePath: [number, number][] = [];
  const edges: readonly (readonly [number, number, number, number])[] = [
    [
      EARTHQUAKE_PAIR_BOUNDS.minimumX,
      EARTHQUAKE_PAIR_BOUNDS.minimumY,
      EARTHQUAKE_PAIR_BOUNDS.maximumX,
      EARTHQUAKE_PAIR_BOUNDS.minimumY
    ],
    [
      EARTHQUAKE_PAIR_BOUNDS.maximumX,
      EARTHQUAKE_PAIR_BOUNDS.minimumY,
      EARTHQUAKE_PAIR_BOUNDS.maximumX,
      EARTHQUAKE_PAIR_BOUNDS.maximumY
    ],
    [
      EARTHQUAKE_PAIR_BOUNDS.maximumX,
      EARTHQUAKE_PAIR_BOUNDS.maximumY,
      EARTHQUAKE_PAIR_BOUNDS.minimumX,
      EARTHQUAKE_PAIR_BOUNDS.maximumY
    ],
    [
      EARTHQUAKE_PAIR_BOUNDS.minimumX,
      EARTHQUAKE_PAIR_BOUNDS.maximumY,
      EARTHQUAKE_PAIR_BOUNDS.minimumX,
      EARTHQUAKE_PAIR_BOUNDS.minimumY
    ]
  ];
  for (const [startX, startY, endX, endY] of edges) {
    for (let sample = 0; sample <= 24; sample++) {
      if (rectanglePath.length > 0 && sample === 0) continue;
      const fraction = sample / 24;
      rectanglePath.push(
        projection.unproject(
          startX + (endX - startX) * fraction,
          startY + (endY - startY) * fraction
        )
      );
    }
  }
  return {
    events,
    positions,
    depths,
    depthGroups,
    allEventsMask,
    excludingTenKilometersMask,
    shallowCount,
    deepCount,
    exactTenKilometerCount,
    duplicateCoordinatePairs,
    rectanglePath,
    rectangleAreaSquareKilometers:
      ((EARTHQUAKE_PAIR_BOUNDS.maximumX - EARTHQUAKE_PAIR_BOUNDS.minimumX) *
        (EARTHQUAKE_PAIR_BOUNDS.maximumY - EARTHQUAKE_PAIR_BOUNDS.minimumY)) /
      1_000_000,
    projection,
    depthExtent: [minimumDepth, maximumDepth],
    sourceCount: dataset.count
  };
}
