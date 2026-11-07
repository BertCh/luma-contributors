// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared pipeline of the zone runs: blob polygons are rasterized to a zone grid by the real
// GPUPolygonRasterization, then a downstream operation consumes the zones or the polygons.

import {
  getGPUPolygonRasterizationParameterValues,
  GPUPolygonRasterization,
  GPURasterFractionalCoverage
} from '@luma-contributors/analysis/raster-analysis';
import type {RunBuilder, RunBuffer, PolygonsRunLayer} from '../../../define-run';
import {createSmoothField, type Random, type World} from '../../../inputs';
import {makeWorldGrid} from './bands';
import {createBlobFeatures, type BlobFeatures} from './blobs';

export type ZoneInputs = {
  world: World;
  features: BlobFeatures;
  /** Smooth value field over the unit square, row 0 at the top. */
  field: (x: number, y: number) => number;
};

export function createZoneInputs(random: Random, world: World): ZoneInputs {
  return {
    world,
    features: createBlobFeatures(random, world),
    field: createSmoothField(random, 4, 3)
  };
}

/** Field values sampled on a grid of `columns` x `rows` cells (row 0 at the top). */
export function sampleField(
  inputs: Pick<ZoneInputs, 'field'>,
  columns: number,
  rows: number
): Float32Array {
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      values[row * columns + column] = inputs.field((column + 0.5) / columns, (row + 0.5) / rows);
    }
  }
  return values;
}

/** Uploads the polygon mesh for the 'polygons' layer. */
export function uploadZoneMesh(builder: RunBuilder, features: BlobFeatures) {
  return {
    vertices: builder.upload('mesh-vertices', 'float32x2', features.meshVertices),
    indices: builder.upload('mesh-indices', 'uint32', features.meshIndices),
    owners: builder.upload('mesh-owners', 'uint32', features.meshOwners),
    outline: builder.upload('outline', 'float32x4', features.outline)
  };
}

/** A filled-polygons layer colored by one value per zone. */
export function drawZones(
  features: BlobFeatures,
  mesh: ReturnType<typeof uploadZoneMesh>,
  values: RunBuffer,
  look: Partial<PolygonsRunLayer> = {}
): PolygonsRunLayer {
  return {
    layer: 'polygons',
    positions: mesh.vertices,
    indices: mesh.indices,
    owners: mesh.owners,
    values,
    triangleCount: features.meshTriangleCount,
    ...look
  };
}

/** Adds GPUPolygonRasterization to the graph and returns the grid and the zone raster. */
export function addZoneRaster(builder: RunBuilder, inputs: ZoneInputs, gridWidth: number) {
  const {world, features} = inputs;
  const worldGrid = makeWorldGrid(world, gridWidth);
  const positions = builder.upload('positions', 'float32x2', features.positions);
  const featureOffsets = builder.upload('feature-offsets', 'uint32', features.featureOffsets);
  const polygonOffsets = builder.upload('polygon-offsets', 'uint32', features.polygonOffsets);
  const ringOffsets = builder.upload('ring-offsets', 'uint32', features.ringOffsets);
  const extent = builder.upload(
    'extent',
    'float32',
    getGPUPolygonRasterizationParameterValues(worldGrid.grid)
  );
  const zones = builder.allocate('zones', 'uint32', worldGrid.cellCount);
  const boundary = builder.allocate('boundary', 'uint32', worldGrid.cellCount);
  const overflow = builder.allocate('overflow', 'uint32', 1);
  builder.graph.add(
    new GPUPolygonRasterization({
      grid: worldGrid.grid,
      parameters: extent.view,
      positions: positions.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view,
      ringOffsets: ringOffsets.view,
      crossingCapacity: 4 * worldGrid.cellCount,
      output: {zones: zones.view, boundary: boundary.view, status: {overflow: overflow.view}}
    })
  );
  return {
    ...worldGrid,
    extent,
    zones,
    boundary,
    positions,
    featureOffsets,
    polygonOffsets,
    ringOffsets
  };
}

/** Adds GPURasterFractionalCoverage (polygon area per cell, exactly) and returns its tuples. */
export function addFractionalCoverage(builder: RunBuilder, inputs: ZoneInputs, gridWidth: number) {
  const {features, world} = inputs;
  const worldGrid = makeWorldGrid(world, gridWidth);
  const positions = builder.upload('positions', 'float32x2', features.positions);
  const featureOffsets = builder.upload('feature-offsets', 'uint32', features.featureOffsets);
  const polygonOffsets = builder.upload('polygon-offsets', 'uint32', features.polygonOffsets);
  const ringOffsets = builder.upload('ring-offsets', 'uint32', features.ringOffsets);
  // Upper bound: the cells under each polygon's bounding box, one extra on every side.
  const [cellWidth, cellHeight] = worldGrid.cellSize;
  let tupleCapacity = 0;
  for (let feature = 0; feature < features.featureCount; feature++) {
    tupleCapacity +=
      (Math.floor(features.extents[feature * 2] / cellWidth) + 2) *
      (Math.floor(features.extents[feature * 2 + 1] / cellHeight) + 2);
  }
  tupleCapacity = Math.min(tupleCapacity, worldGrid.cellCount * features.featureCount);
  const coverage = {
    zoneIds: builder.allocate('coverage-zones', 'uint32', tupleCapacity),
    cellIds: builder.allocate('coverage-cells', 'uint32', tupleCapacity),
    fractions: builder.allocate('coverage-fractions', 'float32', tupleCapacity),
    count: builder.allocate('coverage-count', 'uint32', 1),
    requiredCount: builder.allocate('coverage-required', 'uint32', 1),
    overflow: builder.allocate('coverage-overflow', 'uint32', 1),
    zoneValidity: builder.allocate('coverage-validity', 'uint32', features.featureCount)
  };
  const output = {
    zoneIds: coverage.zoneIds.view,
    cellIds: coverage.cellIds.view,
    fractions: coverage.fractions.view,
    status: {
      count: coverage.count.view,
      requiredCount: coverage.requiredCount.view,
      overflow: coverage.overflow.view
    },
    zoneValidity: coverage.zoneValidity.view
  };
  builder.graph.add(
    new GPURasterFractionalCoverage({
      grid: worldGrid.grid,
      zones: {
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view
      },
      output
    })
  );
  return {...worldGrid, coverage, output};
}
