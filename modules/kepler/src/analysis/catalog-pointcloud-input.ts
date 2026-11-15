// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  getPointCloudLocalOrigin,
  makePointCloudLocalPositions,
  type PointCloudRasterGrid
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {CatalogParameterValue} from './catalog-binding-types';
import type {
  CatalogCoordinateSystem,
  CatalogFeatureCollection,
  CatalogOutputTable,
  CatalogPosition
} from './catalog-geojson';
import {makeLocalFrame, type LocalFrame} from './catalog-local-frame';

/** Bound exhaustive searches, serialized inputs and host readback independently of contributor limits. */
export const MAXIMUM_POINT_CLOUD_ROWS = 2048;
/** Bound morphology graph memory and per-cell work in this interactive catalog. */
export const MAXIMUM_POINT_CLOUD_CELLS = 4096;
export type PointCloudParameters = Readonly<Record<string, CatalogParameterValue>>;
export type PreparedPointCloud = {
  collection: CatalogFeatureCollection;
  positions: Float32Array;
  classifications: Uint32Array;
  voxelCounts?: Uint32Array;
  coordinates: number[][];
};
export type PreparedPointCloudInputs = {
  clouds: Record<string, PreparedPointCloud>;
  frame: LocalFrame;
  origin: [number, number, number];
};

/** Validate once, then pack all clouds relative to one shared binary64 origin. */
export function preparePointCloudInputs(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  parameters: PointCloudParameters
): PreparedPointCloudInputs {
  const identifier = 'catalog-pointcloud';
  const coordinates: Record<string, number[][]> = {};
  const voxelCounts: Record<string, Uint32Array> = {};
  const countProperty = parameters['countProperty'];
  if (typeof countProperty === 'string') {
    validateRequirement(
      'catalog-pointcloud-voxel-merge',
      parameters['disjointOwnedRows'] === true,
      'disjointOwnedRows must certify disjoint source rows and one identical voxel grid'
    );
    validateRequirement(
      'catalog-pointcloud-voxel-merge',
      parameters['coordinateSystem'] === 'planar',
      'coordinateSystem must be planar for arithmetic Cartesian voxel centroids'
    );
  }
  for (const [name, collection] of Object.entries(inputs)) {
    validateCapacity(
      identifier,
      `${name}.rows`,
      collection.features.length,
      1,
      MAXIMUM_POINT_CLOUD_ROWS
    );
    validateRequirement(
      identifier,
      collection.features.every(
        feature =>
          feature.geometry?.type === 'Point' &&
          Array.isArray(feature.geometry.coordinates) &&
          feature.geometry.coordinates.length === 3 &&
          feature.geometry.coordinates.every(
            value => typeof value === 'number' && Number.isFinite(value)
          )
      ),
      `${name}.positions must contain finite XYZ Point coordinates`
    );
    if (typeof countProperty === 'string') {
      const counts = collection.features.map(feature => feature.properties?.[countProperty]);
      validateRequirement(
        'catalog-pointcloud-voxel-merge',
        counts.every(
          value =>
            typeof value === 'number' &&
            Number.isInteger(value) &&
            value >= 0 &&
            value <= 0xffffffff
        ),
        `${name}.counts must be uint32 values`
      );
      voxelCounts[name] = Uint32Array.from(counts as number[]);
    }
    coordinates[name] = collection.features.map(
      feature => feature.geometry!.coordinates as number[]
    );
  }
  const allCoordinates = Object.values(coordinates).flat();
  if (parameters['coordinateSystem'] === 'EPSG:4326')
    validateRequirement(
      identifier,
      allCoordinates.every(position => Math.abs(position[0]) <= 180 && Math.abs(position[1]) <= 90),
      'positions longitude/latitude must be in [-180,180]/[-90,90]'
    );
  const frame = makeLocalFrame(
    parameters['coordinateSystem'] as CatalogCoordinateSystem,
    allCoordinates as unknown as CatalogPosition[]
  );
  const projected: Record<string, number[]> = {};
  for (const [name, positions] of Object.entries(coordinates))
    projected[name] = positions.flatMap(position => [
      ...frame.forward([position[0], position[1]]),
      position[2]
    ]);
  // Empty voxel coordinates are ignored by the merge contributor. Do not let their arbitrary
  // placeholders expand the frame and destroy precision for occupied centroids.
  const allProjected = Object.entries(projected).flatMap(([name, positions]) =>
    positions.filter(
      (_, index) => !voxelCounts[name] || voxelCounts[name][Math.floor(index / 3)] > 0
    )
  );
  validateRequirement(
    identifier,
    allProjected.every(Number.isFinite),
    'positions must project to a finite local frame'
  );
  const origin = getPointCloudLocalOrigin(allProjected);
  const clouds: Record<string, PreparedPointCloud> = {};
  for (const [name, collection] of Object.entries(inputs)) {
    const positions = makePointCloudLocalPositions(
      projected[name].map((value, index) =>
        voxelCounts[name]?.[Math.floor(index / 3)] === 0 ? origin[index % 3] : value
      ),
      origin
    );
    validateRequirement(
      identifier,
      positions.every(Number.isFinite),
      `${name}.positions offsets must be finite in float32`
    );
    const classificationProperty = parameters['classificationProperty'];
    const classes = collection.features.map(feature =>
      typeof classificationProperty === 'string'
        ? (feature.properties?.[classificationProperty] ?? 1)
        : 1
    );
    validateRequirement(
      identifier,
      classes.every(
        value => typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 255
      ),
      `${name}.classification must be an integer in [0,255]`
    );
    clouds[name] = {
      collection,
      positions,
      ...(voxelCounts[name] ? {voxelCounts: voxelCounts[name]} : {}),
      classifications: Uint32Array.from(classes as number[]),
      coordinates: coordinates[name]
    };
  }
  return {clouds, frame, origin};
}

/** Grid aligns to the Float32 cloud minimum and includes the maximum under half-open membership. */
export function makePointCloudCatalogGrid(
  cloud: PreparedPointCloud,
  cellSize: number
): PointCloudRasterGrid {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < cloud.positions.length; index += 3) {
    minimumX = Math.min(minimumX, cloud.positions[index]);
    minimumY = Math.min(minimumY, cloud.positions[index + 1]);
    maximumX = Math.max(maximumX, cloud.positions[index]);
    maximumY = Math.max(maximumY, cloud.positions[index + 1]);
  }
  const width = Math.max(
    2,
    Math.floor(Math.fround(Math.fround(maximumX - minimumX) / Math.fround(cellSize))) + 1
  );
  const height = Math.max(
    2,
    Math.floor(Math.fround(Math.fround(maximumY - minimumY) / Math.fround(cellSize))) + 1
  );
  validateMaximumCount(
    'catalog-pointcloud',
    'grid cell count (increase cellSize)',
    width * height,
    MAXIMUM_POINT_CLOUD_CELLS
  );
  return {origin: [minimumX, minimumY], cellSize, width, height, rowDirection: 'south'};
}

export function validatePointCloudRadii(radii: readonly number[], integer = false): void {
  validateCapacity('catalog-pointcloud', 'radii.length', radii.length, 1, 8);
  validateRequirement(
    'catalog-pointcloud',
    radii.every(
      (radius, index) =>
        Number.isFinite(radius) &&
        radius > 0 &&
        radius <= 32 &&
        (!integer || Number.isInteger(radius)) &&
        (index === 0 || radius > radii[index - 1])
    ),
    'radii must be strictly increasing positive values at most 32 (integer cells for ground)'
  );
}

/** Generated columns win collisions; original scalar fields remain available for styling. */
export function makePointCloudTable(
  cloud: PreparedPointCloud,
  operation: string,
  values: Record<string, readonly (number | string | boolean | null)[]> = {}
): CatalogOutputTable {
  const features = cloud.collection.features;
  const properties = [
    ...new Set(features.flatMap(feature => Object.keys(feature.properties ?? {})))
  ];
  const table: CatalogOutputTable = {
    id: 'points',
    label: `Point cloud ${operation}`,
    rowCount: features.length,
    columns: {},
    columnTypes: {}
  };
  for (const property of properties) {
    const column = features.map(feature => feature.properties?.[property] ?? null);
    const sample = column.find(value => value !== null);
    if (
      column.every(value => value === null || typeof value === typeof sample) &&
      ['number', 'string', 'boolean'].includes(typeof sample)
    ) {
      table.columns[property] = column.map(value =>
        typeof value === 'number' && !Number.isFinite(value) ? null : value
      ) as (number | string | boolean | null)[];
      table.columnTypes[property] =
        typeof sample === 'number' ? 'float64' : typeof sample === 'boolean' ? 'boolean' : 'string';
    }
  }
  const generated = {
    result_id: features.map((_, row) => `${operation}:${row}`),
    source_row: features.map((_, row) => row),
    source_id: features.map(feature => (feature.id == null ? null : String(feature.id))),
    longitude: cloud.coordinates.map(position => position[0]),
    latitude: cloud.coordinates.map(position => position[1]),
    altitude: cloud.coordinates.map(position => position[2]),
    ...values
  };
  for (const [name, column] of Object.entries(generated)) {
    table.columns[name] = column.map(value =>
      typeof value === 'number' && !Number.isFinite(value) ? null : value
    );
    table.columnTypes[name] =
      typeof column.find(value => value !== null) === 'string' || name === 'source_id'
        ? 'string'
        : 'float64';
  }
  return table;
}
