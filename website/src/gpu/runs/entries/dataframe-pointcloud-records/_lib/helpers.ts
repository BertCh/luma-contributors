// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Small seeded generators shared by the table, point-pattern and point-cloud runs of this group.

import type {RunBuilder} from '../../../define-run';
import {createSmoothField, type Random, type TractsInput} from '../../../inputs';

/** Fixed random draws behind a heavy-tailed column, so a spread slider reshapes the same data. */
export type SkewedSource = {normals: Float32Array; centers: Float32Array};

export function createSkewedSource(random: Random, count: number): SkewedSource {
  const normals = new Float32Array(count);
  const centers = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    normals[index] = drawNormal(random);
    centers[index] = random() < 0.35 ? 2.6 : 1.5;
  }
  return {normals, centers};
}

/** Heavy-tailed positive values (a mixture of two lognormal-like groups), like incomes or fares. */
export function getSkewedValues(source: SkewedSource, spread = 1): Float32Array {
  return Float32Array.from(source.normals, (normal, index) =>
    Math.exp(source.centers[index] + normal * 0.45 * spread)
  );
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

/** Fixed random draws behind a point pattern, so a clustering slider reshapes the same points. */
export type PatternSource = {
  count: number;
  /** Uniform (x, y) in the unit square. */
  uniform: Float32Array;
  /** Per point, compared with `clustering` to decide whether it joins a cluster. */
  membership: Float32Array;
  /** Parent cluster per point. */
  parentIndex: Uint32Array;
  /** Parent (x, y) in the unit square. */
  parents: Float32Array;
  /** Offset deviates around the parent, (x, y) per point. */
  normals: Float32Array;
};

export function createPatternSource(random: Random, count: number, parentCount = 7): PatternSource {
  return {
    count,
    uniform: Float32Array.from({length: count * 2}, () => random()),
    membership: Float32Array.from({length: count}, () => random()),
    parentIndex: Uint32Array.from({length: count}, () => Math.floor(random() * parentCount)),
    parents: Float32Array.from({length: parentCount * 2}, () => 0.1 + 0.8 * random()),
    normals: Float32Array.from({length: count * 2}, () => drawNormal(random))
  };
}

/** Points in [0, width] x [0, height]; the share `clustering` of them sit in Gaussian clusters. */
export function getPatternPositions(
  source: PatternSource,
  width: number,
  height: number,
  clustering: number
): Float32Array {
  const positions = new Float32Array(source.count * 2);
  for (let index = 0; index < source.count; index++) {
    let x = source.uniform[index * 2];
    let y = source.uniform[index * 2 + 1];
    if (source.membership[index] < clustering) {
      const parent = source.parentIndex[index];
      x = Math.min(1, Math.max(0, source.parents[parent * 2] + source.normals[index * 2] * 0.035));
      y = Math.min(
        1,
        Math.max(0, source.parents[parent * 2 + 1] + source.normals[index * 2 + 1] * 0.035)
      );
    }
    positions[index * 2] = x * width;
    positions[index * 2 + 1] = y * height;
  }
  return positions;
}

/** A table of point records: positions plus a heavy-tailed numeric column that varies in space. */
export function createRecordColumn(values: Float32Array, maximum = 50): Float32Array {
  return Float32Array.from(values, value => 2 + (maximum - 2) * Math.pow(value, 2.2));
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

/** Index of the nearest site for every point (a key per record, like a spatial join to tracts). */
export function assignNearestSites(positions: Float32Array, sites: Float32Array): Uint32Array {
  const keys = new Uint32Array(positions.length / 2);
  for (let point = 0; point < keys.length; point++) {
    let best = 0;
    let bestDistance = Infinity;
    for (let site = 0; site < sites.length / 2; site++) {
      const distance =
        (positions[point * 2] - sites[site * 2]) ** 2 +
        (positions[point * 2 + 1] - sites[site * 2 + 1]) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = site;
      }
    }
    keys[point] = best;
  }
  return keys;
}

/** A stable pseudo-random number in [0, 1) for an integer, for masks that do not reshuffle. */
export function hashToUnit(index: number): number {
  const value = Math.sin(index * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

/** Observations of a smooth spatial field in the left panel [0, windowWidth] x [0, 1]. */
export type FieldSample = {
  count: number;
  windowWidth: number;
  positions: Float32Array;
  /** Smooth field value per point, 0 to 10. */
  field: Float32Array;
  /** A second smooth field, independent of the first. */
  secondField: Float32Array;
  normals: Float32Array;
};

export function createFieldSample(random: Random, windowWidth: number, count: number): FieldSample {
  const first = createSmoothField(random, 3, 2);
  const second = createSmoothField(random, 3, 2);
  const positions = new Float32Array(count * 2);
  const field = new Float32Array(count);
  const secondField = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const x = random();
    const y = random();
    positions[index * 2] = x * windowWidth;
    positions[index * 2 + 1] = y;
    field[index] = first(x, y) * 10;
    secondField[index] = second(x, y) * 10;
  }
  return {
    count,
    windowWidth,
    positions,
    field,
    secondField,
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

/** An airborne-lidar-like cloud: rolling ground plus tree clumps, with fixed random draws. */
export type TerrainCloud = {
  count: number;
  width: number;
  /** (x, y, z) per point, in drawing-world units. */
  positions: Float32Array;
  /** (x, y) per point for drawing. */
  footprint: Float32Array;
  /** LAS-style class: 1 unclassified (trees), 2 ground. */
  classes: Uint32Array;
  /** Fixed normal deviates for jitter, (x, y, z) per point. */
  normals: Float32Array;
};

export function createTerrainCloud(random: Random, width: number, count: number): TerrainCloud {
  const ground = createSmoothField(random, 3, 2);
  const trees = Array.from({length: 26}, () => [
    random() * width,
    random(),
    0.05 + 0.05 * random(),
    0.12 + 0.14 * random()
  ]);
  const positions = new Float32Array(count * 3);
  const footprint = new Float32Array(count * 2);
  const classes = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const x = random() * width;
    const y = random();
    let height = ground(x / width, y) * 0.2;
    let pointClass = 2;
    for (const [treeX, treeY, radius, treeHeight] of trees) {
      // Under a crown, about half the returns reach the ground below it.
      if (Math.hypot(x - treeX, y - treeY) < radius && random() < 0.55) {
        height += treeHeight * (0.5 + random() * 0.5);
        pointClass = 1;
        break;
      }
    }
    positions.set([x, y, height], index * 3);
    footprint.set([x, y], index * 2);
    classes[index] = pointClass;
  }
  return {
    count,
    width,
    positions,
    footprint,
    classes,
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
    cellSize,
    width: Math.ceil(width / cellSize),
    height: Math.ceil(1 / cellSize)
  };
}
