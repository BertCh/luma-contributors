// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Small seeded generators shared by the table, point-pattern and point-cloud runs of this group.

import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {Random, TractsInput} from '../../../inputs';
import {createPointIndex, findPolygonAt} from '../../../pick';
import {createPointCloud} from '../../../../data/king-county';
import {sampleFlashes} from '../../../../data/lightning';

// Keep the measured energy distribution at spread = 1. Higher powers stretch its tail.
export function getEnergyValues(energy: Float32Array, spread: number): Float32Array {
  return Float32Array.from(energy, value => 75 * (value / 75) ** spread);
}

/** Box-Muller normal deviate. */
export function drawNormal(random: Random): number {
  return Math.sqrt(-2 * Math.log(random() + 1e-9)) * Math.cos(random() * Math.PI * 2);
}

/** Row indices 0..count-1 as floats, for the x axis of line and scatter charts. */
export function createRowAxis(count: number): Float32Array {
  return Float32Array.from({length: count}, (_, index) => index);
}

/** Counts of `values` in `binCount` equal-width bins over [minimum, maximum]. */
export function createHistogram(
  values: ArrayLike<number>,
  binCount: number,
  minimum: number,
  maximum: number
): Float32Array {
  const counts = new Float32Array(binCount);
  for (let index = 0; index < values.length; index++) {
    const bin = Math.floor(((values[index] - minimum) / (maximum - minimum)) * binCount);
    if (bin >= 0 && bin < binCount) counts[bin]++;
  }
  return counts;
}

/** A seeded CSR reference and the measured flash pattern in the same unit window. */
export function createPatternSource(random: Random, count: number) {
  return {
    count,
    observed: sampleFlashes({width: 1, height: 1}, count).positions,
    uniform: Float32Array.from({length: count * 2}, () => random()),
    membership: Float32Array.from({length: count}, () => random())
  };
}

/** At the observed end of the control every point is a measured flash. */
export function getPatternPositions(
  source: ReturnType<typeof createPatternSource>,
  width: number,
  height: number,
  observedShare: number
): Float32Array {
  const positions = new Float32Array(source.count * 2);
  for (let index = 0; index < source.count; index++) {
    const selected = source.membership[index] < observedShare ? source.observed : source.uniform;
    positions[index * 2] = selected[index * 2] * width;
    positions[index * 2 + 1] = selected[index * 2 + 1] * height;
  }
  return positions;
}

/** Uploads the drawing buffers of tracts (polygon mesh, owners, outline). */
export function uploadTractGeometry(builder: RunBuilder, tracts: TractsInput) {
  return {
    vertices: builder.upload('vertices', 'float32x2', tracts.vertices),
    indices: builder.upload('indices', 'uint32', tracts.triangleIndices),
    owners: builder.upload('owners', 'uint32', tracts.triangleOwners),
    outline: builder.upload('outline', 'float32x4', tracts.outline)
  };
}

/** A stable pseudo-random number in [0, 1) for an integer, for masks that do not reshuffle. */
export function hashToUnit(index: number): number {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

/** Measured satellite lightning flashes in the left panel [0, windowWidth] x [0, 1]. */
export type FieldSample = {
  count: number;
  windowWidth: number;
  positions: Float32Array;
  /** Log-transformed optical flash energy in fJ. */
  field: Float32Array;
  /** Log-transformed flash footprint area in square kilometres. */
  secondField: Float32Array;
  normals: Float32Array;
};

export function createFieldSample(random: Random, windowWidth: number, count: number): FieldSample {
  const flashes = sampleFlashes({width: windowWidth, height: 1}, count);
  return {
    count,
    windowWidth,
    positions: flashes.positions,
    field: Float32Array.from(flashes.energy, Math.log1p),
    secondField: Float32Array.from(flashes.area, Math.log1p),
    normals: Float32Array.from({length: count}, () => drawNormal(random))
  };
}

/** The field plus independent noise of standard deviation `noise` (a nugget effect). */
export function getFieldValues(sample: FieldSample, noise: number): Float32Array {
  return Float32Array.from(sample.field, (value, index) => value + sample.normals[index] * noise);
}

/** Population variance of values. */
export function getVariance(values: ArrayLike<number>): number {
  let sum = 0;
  let sumOfSquares = 0;
  for (let index = 0; index < values.length; index++) {
    sum += values[index];
    sumOfSquares += values[index] ** 2;
  }
  const mean = sum / values.length;
  return sumOfSquares / values.length - mean * mean;
}

/** USGS King County lidar returns, with optional simulated coordinate jitter. */
export type TerrainCloud = {
  count: number;
  width: number;
  /** (x, y, z) per point, in drawing-world units. */
  positions: Float32Array;
  /** (x, y) per point for drawing. */
  footprint: Float32Array;
  /** Original LAS class from the survey, including class 2 ground. */
  classes: Uint32Array;
  /** Fixed normal deviates for jitter, (x, y, z) per point. */
  normals: Float32Array;
  /** World units per metre: z in `positions` is metres times this. */
  scale: number;
};

export function createTerrainCloud(random: Random, width: number, count: number): TerrainCloud {
  const cloud = createPointCloud(random, {width, height: 1}, count);
  const positions = new Float32Array(count * 3);
  for (let row = 0; row < count; row++) {
    positions[row * 3] = cloud.positions[row * 2];
    positions[row * 3 + 1] = cloud.positions[row * 2 + 1];
    positions[row * 3 + 2] = cloud.heights[row] * cloud.scale;
  }
  return {
    count,
    width,
    positions,
    footprint: cloud.positions,
    classes: cloud.classes,
    scale: cloud.scale,
    normals: Float32Array.from({length: count * 3}, () => drawNormal(random))
  };
}

/** The cloud with Gaussian jitter of `noise` world units added to every coordinate. */
export function getJitteredPositions(cloud: TerrainCloud, noise: number): Float32Array {
  return Float32Array.from(cloud.positions, (value, index) => value + cloud.normals[index] * noise);
}

/** A raster grid over the cloud's footprint [0, width] x [0, 1] with square cells. */
export function createCloudGrid(width: number, cellSize: number) {
  return {
    origin: [0, 0] as [number, number],
    rowDirection: 'south' as const,
    cellSize,
    width: Math.ceil(width / cellSize),
    height: Math.ceil(1 / cellSize)
  };
}

export type BoxBounds = {minimumX: number; minimumY: number; maximumX: number; maximumY: number};

/**
 * The rectangle a spatial mask keeps: `sharePercent` of the `width` x `height` window by area, with
 * the window's aspect ratio, centered at `center` and kept inside the window.
 */
export function getBoxMaskBounds(
  width: number,
  height: number,
  sharePercent: number,
  center: readonly [number, number] = [width / 2, height / 2]
): BoxBounds {
  const scale = Math.sqrt(Math.min(Math.max(sharePercent, 0), 100) / 100);
  const halfWidth = (width * scale) / 2;
  const halfHeight = (height * scale) / 2;
  const centerX = Math.min(Math.max(center[0], halfWidth), width - halfWidth);
  const centerY = Math.min(Math.max(center[1], halfHeight), height - halfHeight);
  return {
    minimumX: centerX - halfWidth,
    minimumY: centerY - halfHeight,
    maximumX: centerX + halfWidth,
    maximumY: centerY + halfHeight
  };
}

/** A mask of 1 for rows whose (x, y) lies inside `bounds` and 0 elsewhere. */
export function createBoxMask(positions: ArrayLike<number>, bounds: BoxBounds): Uint32Array {
  const count = positions.length >> 1;
  return Uint32Array.from({length: count}, (_, row) =>
    positions[row * 2] >= bounds.minimumX &&
    positions[row * 2] <= bounds.maximumX &&
    positions[row * 2 + 1] >= bounds.minimumY &&
    positions[row * 2 + 1] <= bounds.maximumY
      ? 1
      : 0
  );
}

/** Which tract is under the pointer, which rows belong to it, and its outline. */
export function createTractProbe(tracts: TractsInput, keys: Uint32Array) {
  const members: number[][] = Array.from({length: tracts.count}, () => []);
  keys.forEach((key, row) => members[key]?.push(row));
  let richest = 0;
  members.forEach((rows, tract) => {
    if (rows.length > members[richest].length) richest = tract;
  });
  return {
    members,
    /** The tract with the most rows: the poster rests the pointer on it. */
    richest,
    /** The tract containing (x, y), or -1. */
    findTract: (x: number, y: number) =>
      findPolygonAt(
        tracts.vertices,
        tracts.triangleIndices,
        tracts.triangleOwners,
        x,
        y,
        tracts.triangleCount
      ),
    /** The tract's outline as packed (x, y) vertices. */
    getRing: (tract: number) =>
      tracts.vertices.subarray(tracts.ringOffsets[tract] * 2, tracts.ringOffsets[tract + 1] * 2)
  };
}

/** Writes the (x, y) of `rows` to the front of a float32x2 buffer; returns how many were written. */
export function writeRowPositions(
  buffer: RunBuffer,
  positions: ArrayLike<number>,
  rows: readonly number[]
): number {
  // Unused rows sit off-screen, so a layer that must draw at least one dot shows nothing.
  const packed = new Float32Array(buffer.length * 2).fill(-10);
  const count = Math.min(rows.length, buffer.length);
  for (let index = 0; index < count; index++) {
    packed[index * 2] = positions[rows[index] * 2];
    packed[index * 2 + 1] = positions[rows[index] * 2 + 1];
  }
  buffer.write(packed);
  return count;
}

/**
 * The lidar runs draw the footprint in the upper part of the picture and a side profile (x against
 * z) in a strip below it, so operations that work in z show what they do.
 */
export type CloudLayout = {
  width: number;
  /** Uniform scale of the footprint map. */
  scale: number;
  offsetX: number;
  offsetY: number;
  profileBottom: number;
  profileHeight: number;
  heightMaximum: number;
};

export function createCloudLayout(width: number, heightMaximum: number): CloudLayout {
  const profileShare = 0.22;
  const scale = 1 - profileShare - 0.02;
  return {
    width,
    scale,
    offsetX: (width - width * scale) / 2,
    offsetY: profileShare + 0.02,
    profileBottom: 0.03,
    profileHeight: profileShare - 0.05,
    heightMaximum
  };
}

export function toMapPoint(layout: CloudLayout, x: number, y: number): [number, number] {
  return [layout.offsetX + x * layout.scale, layout.offsetY + y * layout.scale];
}

export function fromMapPoint(layout: CloudLayout, x: number, y: number): [number, number] {
  return [(x - layout.offsetX) / layout.scale, (y - layout.offsetY) / layout.scale];
}

export function getProfilePoint(layout: CloudLayout, x: number, z: number): [number, number] {
  return [
    layout.offsetX + x * layout.scale,
    layout.profileBottom + Math.min(z / layout.heightMaximum, 1) * layout.profileHeight
  ];
}

/** (x, y) per point of `positions` (x, y, z triples), placed in the map region. */
export function createMapPositions(
  layout: CloudLayout,
  positions: ArrayLike<number>,
  count: number
) {
  const mapped = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    mapped.set(toMapPoint(layout, positions[row * 3], positions[row * 3 + 1]), row * 2);
  }
  return mapped;
}

/** (x, z) per point of `positions` (x, y, z triples), placed in the side-profile strip. */
export function createProfilePositions(
  layout: CloudLayout,
  positions: ArrayLike<number>,
  count: number
) {
  const profile = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    profile.set(getProfilePoint(layout, positions[row * 3], positions[row * 3 + 2]), row * 2);
  }
  return profile;
}

/** The highest z of the cloud, rounded up, for the side profile's vertical extent. */
export function getHeightMaximum(positions: ArrayLike<number>, count: number): number {
  let maximum = 0;
  for (let row = 0; row < count; row++) maximum = Math.max(maximum, positions[row * 3 + 2]);
  return maximum * 1.05;
}

/** Nearest-point and 3D neighbor lookups on the footprint of a terrain cloud. */
export function createCloudProbe(cloud: TerrainCloud) {
  const index = createPointIndex(cloud.footprint, cloud.count);
  // A non-ground return (class 2 is ground) near the 80th percentile of height: the poster rests on
  // it, so the probe starts on a tree or a roof.
  const rows = Array.from({length: cloud.count}, (_, row) => row);
  const above = rows.filter(row => cloud.classes[row] !== 2);
  const byHeight = (above.length > 20 ? above : rows).sort(
    (first, second) => cloud.positions[first * 3 + 2] - cloud.positions[second * 3 + 2]
  );
  return {
    index,
    restRow: byHeight[Math.floor(byHeight.length * 0.8)],
    /** Rows within `radius` of `row` in 3D, nearest first, without `row` itself. */
    findNeighbors(row: number, radius: number): number[] {
      const x = cloud.positions[row * 3];
      const y = cloud.positions[row * 3 + 1];
      const z = cloud.positions[row * 3 + 2];
      const found: {row: number; distance: number}[] = [];
      for (const other of index.within(x, y, radius)) {
        if (other === row) continue;
        const distance = Math.hypot(
          cloud.positions[other * 3] - x,
          cloud.positions[other * 3 + 1] - y,
          cloud.positions[other * 3 + 2] - z
        );
        if (distance <= radius) found.push({row: other, distance});
      }
      return found.sort((first, second) => first.distance - second.distance).map(item => item.row);
    }
  };
}
