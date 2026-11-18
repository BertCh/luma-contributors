// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared pipeline of the zone runs: Chicago tract polygons are rasterized to a zone grid by the real
// GPUPolygonRasterization, then a downstream operation consumes the zones or the polygons.

import {
  getGPURasterFractionalCoverageParameterValues,
  getGPUPolygonRasterizationParameterValues,
  GPUPolygonRasterization,
  GPURasterFractionalCoverage
} from '@luma-contributors/analysis/raster-analysis';
import type {RunBuilder, RunBuffer, PolygonsRunLayer} from '../../../define-run';
import {createSmoothField, createTracts, type Random, type World} from '../../../inputs';
import {findPolygonAt} from '../../../pick';
import {makeWorldGrid} from './bands';
import {createTractFeatures, type TractFeatures} from './tract-zones';

export type ZoneInputs = {
  world: World;
  features: TractFeatures;
  /** Smooth value field over the unit square, row 0 at the top. */
  field: (x: number, y: number) => number;
  /** The tract the pointer last probed; `afterRun` rewrites highlight buffers from it. */
  focus: {tract: number};
};

export function createZoneInputs(random: Random, world: World): ZoneInputs {
  const features = createTractFeatures(createTracts(random, world, 7));
  const detail = createSmoothField(random, 4, 4);
  return {
    world,
    features,
    focus: {tract: -1},
    // A continuous surface, so every tract covers a spread of values: inverse-distance
    // interpolation of the normalized ACS poverty measure at the tract centers, plus fine detail.
    field: (x, y) => {
      let weightSum = 0;
      let valueSum = 0;
      for (let row = 0; row < features.featureCount; row++) {
        const dx = x * world.width - features.centers[row * 2];
        const dy = (1 - y) * world.height - features.centers[row * 2 + 1];
        const weight = 1 / (dx * dx + dy * dy + 0.0004) ** 1.5;
        weightSum += weight;
        valueSum += weight * features.values[row];
      }
      const spread = Math.min(1, Math.max(0, 0.5 + 1.8 * (detail(x, y) - 0.5)));
      return 0.5 * (valueSum / weightSum) + 0.5 * spread;
    }
  };
}

/** The tract under a drawing-world position, or -1. */
export function getTractAt(features: TractFeatures, x: number, y: number): number {
  return findPolygonAt(
    features.meshVertices,
    features.meshIndices,
    features.meshOwners,
    x,
    y,
    features.meshTriangleCount
  );
}

/** The closed ring of a tract as (x, y) pairs, for a polyline glyph. */
export function getTractRing(features: TractFeatures, tract: number): Float32Array {
  return features.positions.slice(
    features.ringOffsets[tract] * 2,
    features.ringOffsets[tract + 1] * 2
  );
}

/** Segments of a regular grid over the drawing world, as (x0, y0, x1, y1) rows. */
export function createGridLines(world: World, columns: number, rows: number): Float32Array {
  const segments: number[] = [];
  for (let column = 0; column <= columns; column++) {
    const x = (column / columns) * world.width;
    segments.push(x, 0, x, world.height);
  }
  for (let row = 0; row <= rows; row++) {
    const y = (row / rows) * world.height;
    segments.push(0, y, world.width, y);
  }
  return Float32Array.from(segments);
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
export function uploadZoneMesh(builder: RunBuilder, features: TractFeatures) {
  return {
    vertices: builder.upload('mesh-vertices', 'float32x2', features.meshVertices),
    indices: builder.upload('mesh-indices', 'uint32', features.meshIndices),
    owners: builder.upload('mesh-owners', 'uint32', features.meshOwners),
    outline: builder.upload('outline', 'float32x4', features.outline)
  };
}

/** A filled-polygons layer colored by one value per zone. */
export function drawZones(
  features: TractFeatures,
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
export function addZoneRaster(
  builder: RunBuilder,
  inputs: Pick<ZoneInputs, 'features' | 'world'>,
  gridWidth: number
) {
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
    candidateOverflow: builder.allocate('coverage-candidate-overflow', 'uint32', 1),
    zoneValidity: builder.allocate('coverage-validity', 'uint32', features.featureCount)
  };
  const output = {
    zoneIds: coverage.zoneIds.view,
    cellIds: coverage.cellIds.view,
    fractions: coverage.fractions.view,
    status: {
      count: coverage.count.view,
      requiredCount: coverage.requiredCount.view,
      overflow: coverage.overflow.view,
      candidateOverflow: coverage.candidateOverflow.view
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
      parameters: builder.upload(
        'coverage-parameters',
        'float32',
        getGPURasterFractionalCoverageParameterValues(worldGrid.grid)
      ).view,
      output
    })
  );
  return {...worldGrid, coverage, output};
}

type CoverageReadback = {
  zoneIds: Uint32Array | Float32Array | Int32Array | null;
  cellIds: Uint32Array | Float32Array | Int32Array | null;
  fractions: Uint32Array | Float32Array | Int32Array | null;
  count: Uint32Array | Float32Array | Int32Array | null;
  requiredCount: Uint32Array | Float32Array | Int32Array | null;
  overflow: Uint32Array | Float32Array | Int32Array | null;
  candidateOverflow?: Uint32Array | Float32Array | Int32Array | null;
  zoneValidity: Uint32Array | Float32Array | Int32Array | null;
};

/**
 * Writes the covered fraction of every cell the zone touches into `target` (NaN elsewhere) from the
 * (zone, cell, fraction) tuples the operation produced, and summarizes them.
 */
export function readZoneCoverage(
  target: Float32Array,
  coverage: CoverageReadback,
  zone: number
): {status: 'pending' | 'incomplete' | 'complete'; cells: number; partial: number; area: number} {
  target.fill(Number.NaN);
  const unavailable = {cells: NaN, partial: NaN, area: NaN};
  const {
    zoneIds,
    cellIds,
    fractions,
    count,
    requiredCount,
    overflow,
    candidateOverflow,
    zoneValidity
  } = coverage;
  if (
    !zoneIds ||
    !cellIds ||
    !fractions ||
    !count?.length ||
    !requiredCount?.length ||
    !overflow?.length ||
    !zoneValidity ||
    candidateOverflow === null ||
    (candidateOverflow && !candidateOverflow.length) ||
    zone < 0
  )
    return {status: 'pending', ...unavailable};
  if (
    zone >= zoneValidity.length ||
    zoneValidity[zone] === 0 ||
    overflow[0] !== 0 ||
    (candidateOverflow?.[0] ?? 0) !== 0 ||
    requiredCount[0] !== count[0] ||
    count[0] > Math.min(zoneIds.length, cellIds.length, fractions.length)
  )
    return {status: 'incomplete', ...unavailable};
  const summary = {status: 'complete' as const, cells: 0, partial: 0, area: 0};
  for (let tuple = 0; tuple < count[0]; tuple++) {
    if (zoneIds[tuple] !== zone) continue;
    target[cellIds[tuple]] = fractions[tuple];
    summary.cells++;
    summary.area += fractions[tuple];
    if (fractions[tuple] < 0.999) summary.partial++;
  }
  return summary;
}
