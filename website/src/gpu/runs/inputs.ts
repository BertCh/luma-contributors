// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared observation samples, modeled inputs and GPU upload adapters. Seeded sampling keeps
// a run reproducible. Drawing coordinates use the run's world, x right and y up.

import {sampleElevation, normalizeElevation} from '../data/elevation';
import {sampleReflectance} from '../data/reflectance';
import {getEarthquakes, getRainStations} from '../data/point-observations';
import {sampleVesselTracks} from '../data/vessel-tracks';
import {createStreetNetwork, type StreetNetwork} from '../demos/street-network';
import type {GPUNetworkTopology} from '@luma-contributors/analysis/gpu-network';
import type {GPUSpatialWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuilder} from './define-run';
import chicagoTracts from '../data/chicago-tracts.json';

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

/** Grand Canyon topography in display coordinates, normalized for scalar raster operations. */
export function createElevationRaster(world: World, rows = 72): RasterInput {
  const columns = Math.max(16, Math.round((rows * world.width) / world.height));
  const {elevation} = sampleElevation(columns, rows);
  return {columns, rows, cellSize: world.height / rows, values: normalizeElevation(elevation)};
}

export type ImageryBands = RasterInput & {
  blue: Float32Array;
  green: Float32Array;
  red: Float32Array;
  nearInfrared: Float32Array;
  shortwaveInfrared: Float32Array;
  shortwaveInfrared2: Float32Array;
  /** Spectral proxy for land cover per cell: 0 water, 1 bare soil, 2 vegetation. */
  landCover: Uint32Array;
};

/** Sentinel-2 reflectance and a simple index-based cover classification for categorical runs. */
export function createImageryBands(world: World, rows = 72): ImageryBands {
  const columns = Math.max(16, Math.round((rows * world.width) / world.height));
  const bands = sampleReflectance(columns, rows);
  const {red, green, nearInfrared} = bands;
  const landCover = Uint32Array.from(nearInfrared, (nir, index) => {
    if (green[index] > nir) return 0;
    const ndvi = (nir - red[index]) / Math.max(0.001, nir + red[index]);
    return ndvi > 0.35 ? 2 : 1;
  });
  return {...bands, columns, rows, cellSize: world.height / rows, values: nearInfrared, landCover};
}

// ---------------------------------------------------------------------------------------------
// Points

export type PointsInput = {
  count: number;
  /** (x, y) pairs. */
  positions: Float32Array;
  /** Normalized measurements, with optional simulated noise, in [0, 1]. */
  values: Float32Array;
};

/** A sample of GHCN rain gauges, with a controllable measurement-noise scenario. */
export function createUniformPoints(
  random: Random,
  world: World,
  count = 600,
  noise = 0.08
): PointsInput {
  const stations = getRainStations(random, world, count);
  const values = Float32Array.from(stations.rain, rain =>
    Math.min(1, Math.max(0, rain / 343.4 + (random() - 0.5) * noise))
  );
  return {count, positions: stations.positions, values};
}

/** Earthquake epicenters with an adjustable share sampled from sparsely occupied cells. Longitude quantiles give
 * grouped geometry examples reproducible region labels, independent of any inferred clusters. */
export function createClusteredPoints(
  random: Random,
  world: World,
  count = 800,
  clusterCount = 4,
  noiseShare = 0.1
): PointsInput & {labels: Uint32Array} {
  const {positions, magnitude, scattered} = getEarthquakes(random, world, count, noiseShare);
  const rows = Array.from({length: count}, (_, row) => row);
  rows.sort((first, second) => positions[first * 2] - positions[second * 2]);
  const labels = new Uint32Array(count);
  rows.forEach((row, rank) => {
    labels[row] = scattered[row] ? clusterCount : Math.floor((rank * clusterCount) / count);
  });
  const values = Float32Array.from(magnitude, value => Math.min(1, (value - 4) / 4));
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
  geoids: string[];
  /** Map other observations in the Chicago meter frame to this tract window. */
  projectMeters: (east: number, north: number) => [number, number];
  centers: Float32Array;
  /** Packed open shell vertices. One ring and one polygon per feature. */
  vertices: Float32Array;
  ringOffsets: Uint32Array;
  triangleIndices: Uint32Array;
  triangleOwners: Uint32Array;
  triangleCount: number;
  outline: Float32Array;
  outlineCount: number;
  population: Float32Array;
  age65: Float32Array;
  povertyShare: Float32Array;
  unemploymentShare: Float32Array;
  /** Share below 150% of the poverty threshold, normalized to the full fixture range. */
  values: Float32Array;
  contiguity: CsrWeights;
};

const minimumPoverty = Math.min(...chicagoTracts.map(tract => tract.povertyPercent));
const maximumPoverty = Math.max(...chicagoTracts.map(tract => tract.povertyPercent));

/** A connected, north-up window of Chicago Census tracts, fitted without stretching. */
export function createTracts(_random: Random, world: World, rows = 14, _noise = 0): TractsInput {
  // Older runs express their sample size as a linear resolution. Keep its quadratic scaling.
  const tracts = chicagoTracts.slice(0, Math.max(1, Math.round(rows * rows)));
  const count = tracts.length;
  let west = Infinity,
    east = -Infinity,
    south = Infinity,
    north = -Infinity;
  for (const {positions} of tracts) {
    for (let index = 0; index < positions.length; index += 2) {
      west = Math.min(west, positions[index]);
      east = Math.max(east, positions[index]);
      south = Math.min(south, positions[index + 1]);
      north = Math.max(north, positions[index + 1]);
    }
  }
  const scale = 0.92 * Math.min(world.width / (east - west), world.height / (north - south));
  const projectX = (x: number) => world.width / 2 + (x - (west + east) / 2) * scale;
  const projectY = (y: number) => world.height / 2 + (y - (south + north) / 2) * scale;
  const centers = new Float32Array(count * 2);
  const ringOffsets = new Uint32Array(count + 1);
  const vertices: number[] = [];
  const triangleIndices: number[] = [];
  const triangleOwners: number[] = [];
  const outlines: number[] = [];
  tracts.forEach((tract, row) => {
    centers.set([projectX(tract.center[0]), projectY(tract.center[1])], row * 2);
    const start = vertices.length / 2;
    for (let index = 0; index < tract.positions.length; index += 2) {
      const next = (index + 2) % tract.positions.length;
      const x = projectX(tract.positions[index]);
      const y = projectY(tract.positions[index + 1]);
      vertices.push(x, y);
      outlines.push(x, y, projectX(tract.positions[next]), projectY(tract.positions[next + 1]));
    }
    ringOffsets[row + 1] = vertices.length / 2;
    for (const corner of tract.triangles) triangleIndices.push(start + corner);
    for (let index = 0; index < tract.triangles.length; index += 3) triangleOwners.push(row);
  });
  const contiguity = buildRows(count, row =>
    tracts[row].neighbors
      .filter(other => other < count)
      .map(other => [
        other,
        Math.hypot(
          centers[row * 2] - centers[other * 2],
          centers[row * 2 + 1] - centers[other * 2 + 1]
        )
      ])
  );
  return {
    count,
    geoids: tracts.map(tract => tract.geoid),
    projectMeters: (east, north) => [projectX(east), projectY(north)],
    centers,
    vertices: Float32Array.from(vertices),
    ringOffsets,
    triangleIndices: Uint32Array.from(triangleIndices),
    triangleOwners: Uint32Array.from(triangleOwners),
    triangleCount: triangleOwners.length,
    outline: Float32Array.from(outlines),
    outlineCount: outlines.length / 4,
    population: Float32Array.from(tracts, tract => tract.population),
    age65: Float32Array.from(tracts, tract => tract.age65),
    povertyShare: Float32Array.from(tracts, tract => tract.povertyPercent / 100),
    unemploymentShare: Float32Array.from(tracts, tract => tract.unemploymentPercent / 100),
    values: Float32Array.from(
      tracts,
      tract => (tract.povertyPercent - minimumPoverty) / (maximumPoverty - minimumPoverty)
    ),
    contiguity
  };
}

// ---------------------------------------------------------------------------------------------
// Networks

export type NetworkInput = StreetNetwork & {
  /** Congestion-free length per directed CSR slot. */
  lengths: Float32Array;
};

/** A connected TIGER/Line street extract, bounded to the original scene budget. */
export function createNetwork(
  _random: Random,
  world: World,
  rows = 18,
  maximumNodes = rows * Math.max(16, Math.round(rows * world.width))
): NetworkInput {
  const network = createStreetNetwork(world.width, maximumNodes);
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

/** San Francisco Bay AIS tracks at a fixed number of fixes per track. */
export function createTrajectories(
  _random: Random,
  world: World,
  trackCount = 24,
  stepCount = 40
): SegmentsInput & {points: Float32Array; trackOffsets: Uint32Array; timestamps: Float32Array} {
  return sampleVesselTracks(world, trackCount, stepCount + 1, stepCount + 1, stepCount);
}

/** Observed vessel paths as vertex lists plus offsets (for line and path operations). */
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
  /** Row-major counts of observed flashes, by longitude band and time bin. */
  values: Float32Array;
};

export {createTimeSeries} from '../data/lightning';
export {createPointCloud} from '../data/king-county';

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
