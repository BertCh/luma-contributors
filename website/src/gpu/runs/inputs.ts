// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Seeded input generators for runs. Everything is deterministic for a given random source and
// small enough (at most ~20k elements) to run in a few milliseconds. Positions are in the run's
// world: x in [0, world.width], y in [0, world.height] (see RunInputContext). Pass the context's
// `random` and `world`.

import {drawGaussianCluster} from '../demos/kit';
import {createStreetNetwork, type StreetNetwork} from '../demos/street-network';
import type {GPUNetworkTopology} from '@luma-contributors/analysis/gpu-network';
import type {GPUSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuilder} from './define-run';

export type World = {width: number; height: number};
export type Random = () => number;

// ---------------------------------------------------------------------------------------------
// Smooth fields

/** A smooth random scalar field in [0, 1]: value noise summed over octaves, sampled anywhere. */
export type SmoothField = (x: number, y: number) => number;

export function createSmoothField(random: Random, octaves = 4, baseCells = 3): SmoothField {
  const layers = Array.from({length: octaves}, (_, octave) => {
    const cells = baseCells * 2 ** octave;
    const lattice = Float32Array.from({length: (cells + 2) * (cells + 2)}, () => random());
    return {cells, lattice, weight: 0.5 ** octave};
  });
  let total = 0;
  for (const layer of layers) total += layer.weight;
  return (x, y) => {
    let sum = 0;
    for (const {cells, lattice, weight} of layers) {
      const gridX = Math.min(Math.max(x, 0), 0.9999) * cells;
      const gridY = Math.min(Math.max(y, 0), 0.9999) * cells;
      const cellX = Math.floor(gridX);
      const cellY = Math.floor(gridY);
      const fractionX = smoothStep(gridX - cellX);
      const fractionY = smoothStep(gridY - cellY);
      const stride = cells + 2;
      const at = (column: number, row: number) => lattice[row * stride + column];
      const top = at(cellX, cellY) * (1 - fractionX) + at(cellX + 1, cellY) * fractionX;
      const bottom = at(cellX, cellY + 1) * (1 - fractionX) + at(cellX + 1, cellY + 1) * fractionX;
      sum += (top * (1 - fractionY) + bottom * fractionY) * weight;
    }
    return sum / total;
  };
}

function smoothStep(value: number): number {
  return value * value * (3 - 2 * value);
}

// ---------------------------------------------------------------------------------------------
// Rasters

export type RasterInput = {
  columns: number;
  rows: number;
  /** Cell size in world units. */
  cellSize: number;
  /** Row-major, row 0 at the top (y = world.height), values in [0, 1] unless scaled. */
  values: Float32Array;
};

/** A fractal DEM-like raster covering the world: `rows` cells high, columns follow the aspect. */
export function createFractalRaster(
  random: Random,
  world: World,
  rows = 72,
  octaves = 5
): RasterInput {
  const columns = Math.max(16, Math.round((rows * world.width) / world.height));
  const field = createSmoothField(random, octaves);
  const values = new Float32Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      values[row * columns + column] = field((column + 0.5) / columns, (row + 0.5) / rows);
    }
  }
  return {columns, rows, cellSize: world.height / rows, values};
}

export type ImageryBands = RasterInput & {
  green: Float32Array;
  red: Float32Array;
  nearInfrared: Float32Array;
  shortwaveInfrared: Float32Array;
  /** Land cover per cell: 0 water, 1 bare soil, 2 vegetation. */
  landCover: Uint32Array;
};

/** Multi-band surface reflectance in [0, 1] from a land-cover map: water, soil and vegetation. */
export function createImageryBands(random: Random, world: World, rows = 72): ImageryBands {
  const base = createFractalRaster(random, world, rows, 4);
  const moisture = createSmoothField(random, 3, 4);
  const cellCount = base.values.length;
  const green = new Float32Array(cellCount);
  const red = new Float32Array(cellCount);
  const nearInfrared = new Float32Array(cellCount);
  const shortwaveInfrared = new Float32Array(cellCount);
  const landCover = new Uint32Array(cellCount);
  for (let index = 0; index < cellCount; index++) {
    const row = Math.floor(index / base.columns);
    const column = index % base.columns;
    const elevation = base.values[index];
    const wet = moisture((column + 0.5) / base.columns, (row + 0.5) / base.rows);
    const cover = elevation < 0.38 ? 0 : wet > 0.46 ? 2 : 1;
    landCover[index] = cover;
    const noise = () => (random() - 0.5) * 0.04;
    // Reflectance signatures: water absorbs NIR, vegetation reflects it and absorbs red.
    const signature =
      cover === 0
        ? [0.08, 0.06, 0.03, 0.02]
        : cover === 1
          ? [0.2, 0.28, 0.34, 0.42]
          : [0.1, 0.06, 0.5, 0.22];
    // Vegetation density and soil brightness vary smoothly, so indices have gradients, not just classes.
    const density = cover === 2 ? 0.65 + 0.7 * wet : 1;
    const brightness = cover === 1 ? 0.8 + 0.5 * elevation : 1;
    green[index] = clamp01(signature[0] * brightness + noise());
    red[index] = clamp01(
      signature[1] * brightness * (cover === 2 ? 1.5 - 0.6 * density : 1) + noise()
    );
    nearInfrared[index] = clamp01(signature[2] * brightness * density + noise());
    shortwaveInfrared[index] = clamp01(signature[3] * brightness + noise());
  }
  return {...base, green, red, nearInfrared, shortwaveInfrared, landCover};
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

// ---------------------------------------------------------------------------------------------
// Points

export type PointsInput = {
  count: number;
  /** (x, y) pairs. */
  positions: Float32Array;
  /** A smooth spatial field plus noise, in [0, 1]. */
  values: Float32Array;
};

/** Points scattered uniformly over the world, valued by a smooth field plus noise. */
export function createUniformPoints(
  random: Random,
  world: World,
  count = 600,
  noise = 0.08
): PointsInput {
  const field = createSmoothField(random, 3);
  const positions = new Float32Array(count * 2);
  const values = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const x = random() * world.width;
    const y = random() * world.height;
    positions[index * 2] = x;
    positions[index * 2 + 1] = y;
    values[index] = clamp01(field(x / world.width, y / world.height) + (random() - 0.5) * noise);
  }
  return {count, positions, values};
}

/** Points in Gaussian clusters (`clusterCount` blobs) plus a share of uniform background noise. */
export function createClusteredPoints(
  random: Random,
  world: World,
  count = 800,
  clusterCount = 4,
  noiseShare = 0.1
): PointsInput & {labels: Uint32Array} {
  const centers = Array.from({length: clusterCount}, () => [
    (0.12 + 0.76 * random()) * world.width,
    0.15 + 0.7 * random(),
    0.03 + 0.04 * random()
  ]);
  const positions = new Float32Array(count * 2);
  const values = new Float32Array(count);
  const labels = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    if (random() < noiseShare) {
      positions[index * 2] = random() * world.width;
      positions[index * 2 + 1] = random() * world.height;
      labels[index] = clusterCount;
      values[index] = random();
    } else {
      const cluster = Math.floor(random() * clusterCount);
      const [centerX, centerY, spread] = centers[cluster];
      const [x, y] = drawGaussianCluster(random, centerX, centerY, spread);
      positions[index * 2] = x;
      positions[index * 2 + 1] = y;
      labels[index] = cluster;
      values[index] = (cluster + 0.5) / clusterCount;
    }
  }
  return {count, positions, values, labels};
}

// ---------------------------------------------------------------------------------------------
// Spatial weights

export type CsrWeights = {
  rowCount: number;
  /** `rowCount + 1` row offsets. */
  offsets: Uint32Array;
  neighbors: Uint32Array;
  /** One weight per slot (binary 1 unless stated). */
  weights: Float32Array;
  distances: Float32Array;
};

/** k nearest neighbors of every point (brute force, fine for a few thousand points). */
export function createKnnWeights(positions: Float32Array, neighborCount = 6): CsrWeights {
  const rowCount = positions.length / 2;
  return buildRows(rowCount, row => {
    const candidates: [number, number][] = [];
    for (let other = 0; other < rowCount; other++) {
      if (other === row) continue;
      candidates.push([
        Math.hypot(
          positions[row * 2] - positions[other * 2],
          positions[row * 2 + 1] - positions[other * 2 + 1]
        ),
        other
      ]);
    }
    candidates.sort((first, second) => first[0] - second[0]);
    return candidates.slice(0, neighborCount).map(([distance, other]) => [other, distance]);
  });
}

/** Every pair of points closer than `radius` (world units). */
export function createDistanceBandWeights(positions: Float32Array, radius: number): CsrWeights {
  const rowCount = positions.length / 2;
  return buildRows(rowCount, row => {
    const found: [number, number][] = [];
    for (let other = 0; other < rowCount; other++) {
      if (other === row) continue;
      const distance = Math.hypot(
        positions[row * 2] - positions[other * 2],
        positions[row * 2 + 1] - positions[other * 2 + 1]
      );
      if (distance <= radius) found.push([other, distance]);
    }
    return found;
  });
}

/** Builds CSR rows (neighbors ascending within each row) from a per-row neighbor finder. */
export function buildRows(
  rowCount: number,
  findNeighbors: (row: number) => readonly (readonly [number, number])[]
): CsrWeights {
  const offsets = new Uint32Array(rowCount + 1);
  const neighborList: number[] = [];
  const distanceList: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const rowNeighbors = [...findNeighbors(row)].sort((first, second) => first[0] - second[0]);
    for (const [neighbor, distance] of rowNeighbors) {
      neighborList.push(neighbor);
      distanceList.push(distance);
    }
    offsets[row + 1] = neighborList.length;
  }
  return {
    rowCount,
    offsets,
    neighbors: Uint32Array.from(neighborList),
    weights: new Float32Array(neighborList.length).fill(1),
    distances: Float32Array.from(distanceList)
  };
}

/** Uploads CSR weights as the `GPUSpatialWeights` the statistics operations take. */
export function uploadSpatialWeights(
  builder: RunBuilder,
  csr: CsrWeights,
  id = 'weights'
): GPUSpatialWeights {
  return {
    offsets: builder.upload(`${id}-offsets`, 'uint32', csr.offsets).view,
    neighbors: builder.upload(`${id}-neighbors`, 'uint32', csr.neighbors).view,
    weights: builder.upload(`${id}-weights`, 'float32', csr.weights).view,
    distances: builder.upload(`${id}-distances`, 'float32', csr.distances).view,
    information: {
      rowCount: csr.rowCount,
      transform: 'raw',
      symmetric: 'unknown',
      provenance: {source: 'upload'}
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Tracts: contiguous polygons

export type TractsInput = {
  count: number;
  /** Site (tract center) positions, (x, y) pairs. */
  centers: Float32Array;
  /** Mesh vertices (x, y): the tract corners followed by the sites. */
  vertices: Float32Array;
  /** Triangle fan vertex ids, three per triangle. */
  triangleIndices: Uint32Array;
  /** Tract row per triangle. */
  triangleOwners: Uint32Array;
  triangleCount: number;
  /** Tract boundary as (x0, y0, x1, y1) segments. */
  outline: Float32Array;
  outlineCount: number;
  /** Smooth spatial field plus noise per tract, in [0, 1]. */
  values: Float32Array;
  /** Edge-sharing (rook) contiguity, binary weights. */
  contiguity: CsrWeights;
};

/**
 * Irregular hexagon-like tracts: Voronoi cells of a jittered triangular lattice, so they tile the
 * world without gaps and every interior tract has about six rook neighbors.
 */
export function createTracts(random: Random, world: World, rows = 14, noise = 0.1): TractsInput {
  const columns = Math.max(6, Math.round(((rows * world.width) / world.height) * 0.866));
  const stepX = world.width / (columns - 1.5);
  const stepY = world.height / (rows - 1);
  // Lattice with a margin ring so border tracts are complete cells; tracts are the inner points.
  const margin = 1;
  const totalColumns = columns + 2 * margin;
  const totalRows = rows + 2 * margin;
  const point = (row: number, column: number): [number, number] => [
    (column - margin + (row % 2) * 0.5) * stepX + (random() - 0.5) * 0.5 * stepX,
    (row - margin) * stepY + (random() - 0.5) * 0.5 * stepY
  ];
  const lattice = new Float32Array(totalRows * totalColumns * 2);
  for (let row = 0; row < totalRows; row++) {
    for (let column = 0; column < totalColumns; column++) {
      const [x, y] = point(row, column);
      lattice[(row * totalColumns + column) * 2] = x;
      lattice[(row * totalColumns + column) * 2 + 1] = y;
    }
  }
  const siteAt = (row: number, column: number) => (row * totalColumns + column) * 2;
  // Counterclockwise neighbor offsets (right, upper-right, upper-left, left, lower-left, lower-right).
  const neighborOffsets = (row: number): [number, number][] =>
    row % 2 === 0
      ? [
          [0, 1],
          [1, 0],
          [1, -1],
          [0, -1],
          [-1, -1],
          [-1, 0]
        ]
      : [
          [0, 1],
          [1, 1],
          [1, 0],
          [0, -1],
          [-1, 0],
          [-1, 1]
        ];

  const centersList: number[] = [];
  const cornerList: number[] = [];
  const ownerIndexBySite = new Map<number, number>();
  const polygonCorners: number[][] = [];
  for (let row = margin; row < totalRows - margin; row++) {
    for (let column = margin; column < totalColumns - margin; column++) {
      const siteX = lattice[siteAt(row, column)];
      const siteY = lattice[siteAt(row, column) + 1];
      //       // A margin beyond the world, so tracts that cross the border still cover the canvas.
      if (siteX < -0.05 * world.height || siteX > world.width + 0.05 * world.height) continue;
      if (siteY < -0.05 * world.height || siteY > 1.05 * world.height) continue;
      const around = neighborOffsets(row).map(([rowStep, columnStep]) => {
        const neighborRow = row + rowStep;
        const neighborColumn = column + columnStep;
        return [
          lattice[siteAt(neighborRow, neighborColumn)],
          lattice[siteAt(neighborRow, neighborColumn) + 1]
        ];
      });
      const corners: number[] = [];
      for (let corner = 0; corner < 6; corner++) {
        const [firstX, firstY] = around[corner];
        const [secondX, secondY] = around[(corner + 1) % 6];
        const [x, y] = getCircumcenter(siteX, siteY, firstX, firstY, secondX, secondY);
        corners.push(cornerList.length / 2);
        cornerList.push(x, y);
      }
      ownerIndexBySite.set(row * totalColumns + column, centersList.length / 2);
      centersList.push(siteX, siteY);
      polygonCorners.push(corners);
    }
  }
  const count = centersList.length / 2;
  const cornerCount = cornerList.length / 2;
  const vertices = Float32Array.from([...cornerList, ...centersList]);
  const triangleIndices: number[] = [];
  const triangleOwners: number[] = [];
  const outline: number[] = [];
  polygonCorners.forEach((corners, tract) => {
    for (let corner = 0; corner < 6; corner++) {
      const next = (corner + 1) % 6;
      triangleIndices.push(cornerCount + tract, corners[corner], corners[next]);
      triangleOwners.push(tract);
      outline.push(
        cornerList[corners[corner] * 2],
        cornerList[corners[corner] * 2 + 1],
        cornerList[corners[next] * 2],
        cornerList[corners[next] * 2 + 1]
      );
    }
  });

  const field = createSmoothField(random, 3, 2);
  const values = new Float32Array(count);
  for (let tract = 0; tract < count; tract++) {
    values[tract] = clamp01(
      field(centersList[tract * 2] / world.width, centersList[tract * 2 + 1] / world.height) +
        (random() - 0.5) * noise
    );
  }
  const centers = Float32Array.from(centersList);
  const siteKeys = [...ownerIndexBySite.keys()];
  const contiguity = buildRows(count, tract => {
    const key = siteKeys[tract];
    const row = Math.floor(key / totalColumns);
    const column = key % totalColumns;
    const found: [number, number][] = [];
    for (const [rowStep, columnStep] of neighborOffsets(row)) {
      const neighbor = ownerIndexBySite.get((row + rowStep) * totalColumns + column + columnStep);
      if (neighbor === undefined) continue;
      found.push([
        neighbor,
        Math.hypot(
          centers[tract * 2] - centers[neighbor * 2],
          centers[tract * 2 + 1] - centers[neighbor * 2 + 1]
        )
      ]);
    }
    return found;
  });
  return {
    count,
    centers,
    vertices,
    triangleIndices: Uint32Array.from(triangleIndices),
    triangleOwners: Uint32Array.from(triangleOwners),
    triangleCount: triangleOwners.length,
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4,
    values,
    contiguity
  };
}

function getCircumcenter(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number
): [number, number] {
  const determinant = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(determinant) < 1e-12) return [(ax + bx + cx) / 3, (ay + by + cy) / 3];
  const aSquared = ax * ax + ay * ay;
  const bSquared = bx * bx + by * by;
  const cSquared = cx * cx + cy * cy;
  return [
    (aSquared * (by - cy) + bSquared * (cy - ay) + cSquared * (ay - by)) / determinant,
    (aSquared * (cx - bx) + bSquared * (ax - cx) + cSquared * (bx - ax)) / determinant
  ];
}

// ---------------------------------------------------------------------------------------------
// Networks

export type NetworkInput = StreetNetwork & {
  /** Congestion-free length per directed CSR slot. */
  lengths: Float32Array;
};

/** A jittered street grid (CSR topology, edge pairs for drawing). `rows` intersections high. */
export function createNetwork(random: Random, world: World, rows = 18): NetworkInput {
  const network = createStreetNetwork(random, world.width, rows);
  const lengths = new Float32Array(network.slotCount);
  network.fillWeights(lengths, 0);
  return {...network, lengths};
}

/** Uploads a network's CSR as the `GPUNetworkTopology` the network operations take. */
export function uploadNetworkTopology(
  builder: RunBuilder,
  network: NetworkInput,
  costs: Float32Array = network.lengths
) {
  const weights = builder.upload('network-weights', 'float32', costs);
  const topology: GPUNetworkTopology = {
    offsets: builder.upload('network-offsets', 'uint32', network.offsets).view,
    neighbors: builder.upload('network-neighbors', 'uint32', network.neighbors).view,
    weights: weights.view
  };
  return {topology, weights};
}

// ---------------------------------------------------------------------------------------------
// Lines, trajectories, time series, point clouds, cells

export type SegmentsInput = {
  count: number;
  /** (x0, y0, x1, y1) per segment. */
  segments: Float32Array;
};

/** Random-walk trajectories: `trackCount` tracks of `stepCount` steps, as consecutive segments. */
export function createTrajectories(
  random: Random,
  world: World,
  trackCount = 24,
  stepCount = 40
): SegmentsInput & {points: Float32Array; trackOffsets: Uint32Array; timestamps: Float32Array} {
  const points: number[] = [];
  const timestamps: number[] = [];
  const segments: number[] = [];
  const trackOffsets = new Uint32Array(trackCount + 1);
  for (let track = 0; track < trackCount; track++) {
    let x = random() * world.width;
    let y = random() * world.height;
    let heading = random() * Math.PI * 2;
    const speed = (0.012 + 0.012 * random()) * world.height;
    for (let step = 0; step <= stepCount; step++) {
      points.push(x, y);
      timestamps.push(step);
      heading += (random() - 0.5) * 0.7;
      const nextX = Math.min(Math.max(x + Math.cos(heading) * speed, 0), world.width);
      const nextY = Math.min(Math.max(y + Math.sin(heading) * speed, 0), world.height);
      if (step < stepCount) segments.push(x, y, nextX, nextY);
      x = nextX;
      y = nextY;
    }
    trackOffsets[track + 1] = points.length / 2;
  }
  return {
    count: segments.length / 4,
    segments: Float32Array.from(segments),
    points: Float32Array.from(points),
    trackOffsets,
    timestamps: Float32Array.from(timestamps)
  };
}

/** Wandering polylines as vertex lists plus offsets (for line and path operations). */
export function createPolylines(
  random: Random,
  world: World,
  lineCount = 12,
  vertexCount = 16
): {vertices: Float32Array; offsets: Uint32Array; segments: Float32Array; segmentCount: number} {
  const tracks = createTrajectories(random, world, lineCount, vertexCount - 1);
  return {
    vertices: tracks.points,
    offsets: tracks.trackOffsets,
    segments: tracks.segments,
    segmentCount: tracks.count
  };
}

export type TimeSeriesInput = {
  panelCount: number;
  stepCount: number;
  /** Row-major panels x steps: a seasonal wave, a trend and noise, with an optional break. */
  values: Float32Array;
};

/** Panels of seasonal time series with per-panel trend and an abrupt change in some panels. */
export function createTimeSeries(random: Random, panelCount = 32, stepCount = 96): TimeSeriesInput {
  const values = new Float32Array(panelCount * stepCount);
  for (let panel = 0; panel < panelCount; panel++) {
    const trend = (random() - 0.5) * 0.4;
    const phase = random() * Math.PI;
    const breakStep = random() < 0.4 ? Math.floor(stepCount * (0.4 + 0.3 * random())) : -1;
    for (let step = 0; step < stepCount; step++) {
      const season = Math.sin((step / 12) * Math.PI * 2 + phase) * 0.25;
      const jump = breakStep >= 0 && step >= breakStep ? -0.3 : 0;
      values[panel * stepCount + step] =
        0.55 + season + trend * (step / stepCount) + jump + (random() - 0.5) * 0.06;
    }
  }
  return {panelCount, stepCount, values};
}

/** A terrain-like point cloud: ground, plus tree-like clumps rising above it. */
export function createPointCloud(
  random: Random,
  world: World,
  count = 6000
): {count: number; positions: Float32Array; heights: Float32Array; classes: Uint32Array} {
  const ground = createSmoothField(random, 4);
  const trees = Array.from({length: 24}, () => [
    random() * world.width,
    random(),
    0.02 + 0.02 * random()
  ]);
  const positions = new Float32Array(count * 2);
  const heights = new Float32Array(count);
  const classes = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const x = random() * world.width;
    const y = random() * world.height;
    let height = ground(x / world.width, y / world.height) * 20;
    let pointClass = 2;
    for (const [treeX, treeY, radius] of trees) {
      if (Math.hypot(x - treeX, y - treeY) < radius) {
        height += 4 + random() * 6;
        pointClass = 5;
        break;
      }
    }
    positions[index * 2] = x;
    positions[index * 2 + 1] = y;
    heights[index] = height;
    classes[index] = pointClass;
  }
  return {count, positions, heights, classes};
}

/** Hexagonal cells covering the world: centers and the six corners of each, as outline segments. */
export function createHexCells(
  world: World,
  rows = 12
): {
  count: number;
  centers: Float32Array;
  outline: Float32Array;
  outlineCount: number;
  radius: number;
} {
  const radius = world.height / (rows * 1.5 + 0.5);
  const stepX = Math.sqrt(3) * radius;
  const columns = Math.ceil(world.width / stepX) + 1;
  const centers: number[] = [];
  const outline: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = (column + (row % 2) * 0.5) * stepX;
      const y = radius + row * radius * 1.5;
      centers.push(x, y);
      for (let corner = 0; corner < 6; corner++) {
        const first = Math.PI / 6 + (corner * Math.PI) / 3;
        const second = Math.PI / 6 + ((corner + 1) * Math.PI) / 3;
        outline.push(
          x + radius * Math.cos(first),
          y + radius * Math.sin(first),
          x + radius * Math.cos(second),
          y + radius * Math.sin(second)
        );
      }
    }
  }
  return {
    count: centers.length / 2,
    centers: Float32Array.from(centers),
    outline: Float32Array.from(outline),
    outlineCount: outline.length / 4,
    radius
  };
}
