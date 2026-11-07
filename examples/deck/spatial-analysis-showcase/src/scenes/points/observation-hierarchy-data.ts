// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {AnalysisFrameProjection} from '../../engine/analysis-frame';

export const OBSERVATION_HIERARCHY_BOUNDS: readonly [number, number, number, number] = [
  -87.617, 41.849, -87.597, 41.87
];

export type ObservationHierarchyData = {
  positions: Float32Array;
  longitudeLatitude: Float32Array;
  sourceRows: Uint32Array;
  coordinateMultiplicity: Uint8Array;
  categoryIndex: number;
  archiveRowCount: number;
  uniqueCoordinateCount: number;
  recordsAtSharedCoordinates: number;
  maximumCoordinateMultiplicity: number;
};

/** Select every archived bird row in the declared Northerly Island frame, preserving source order. */
export function buildObservationHierarchyData(
  dataset: LoadedDataset,
  projection: AnalysisFrameProjection
): ObservationHierarchyData {
  const positions = dataset.column<Float32Array>('position');
  const categories = dataset.column<Uint8Array>('category');
  const categoryIndex = dataset.categories('category').indexOf('Birds');
  const [west, south, east, north] = OBSERVATION_HIERARCHY_BOUNDS;
  const projected: number[] = [];
  const longitudeLatitude: number[] = [];
  const sourceRows: number[] = [];
  const coordinateKeys: string[] = [];
  const coordinateCounts = new Map<string, number>();
  for (let row = 0; row < categories.length; row++) {
    const longitude = positions[row * 2];
    const latitude = positions[row * 2 + 1];
    if (
      categories[row] !== categoryIndex ||
      longitude < west ||
      longitude > east ||
      latitude < south ||
      latitude > north
    )
      continue;
    const [projectedX, projectedY] = projection.project(longitude, latitude);
    projected.push(projectedX, projectedY);
    longitudeLatitude.push(longitude, latitude);
    sourceRows.push(row);
    const coordinateKey = `${longitude},${latitude}`;
    coordinateKeys.push(coordinateKey);
    coordinateCounts.set(coordinateKey, (coordinateCounts.get(coordinateKey) ?? 0) + 1);
  }
  const multiplicities = Array.from(coordinateCounts.values());
  return {
    positions: Float32Array.from(projected),
    longitudeLatitude: Float32Array.from(longitudeLatitude),
    sourceRows: Uint32Array.from(sourceRows),
    coordinateMultiplicity: Uint8Array.from(
      coordinateKeys.map(coordinateKey => coordinateCounts.get(coordinateKey) ?? 1)
    ),
    categoryIndex,
    archiveRowCount: categories.length,
    uniqueCoordinateCount: coordinateCounts.size,
    recordsAtSharedCoordinates: multiplicities.reduce(
      (total, count) => total + (count > 1 ? count : 0),
      0
    ),
    maximumCoordinateMultiplicity: Math.max(0, ...multiplicities)
  };
}
