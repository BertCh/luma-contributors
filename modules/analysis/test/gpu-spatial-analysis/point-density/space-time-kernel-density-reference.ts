// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUKernelDensityKernel} from '../../../src/gpu-spatial-analysis/point-density/index';
import {getOracleKernelMassFan} from '../../analysis-core/study-window-oracle';

/** Independent normalized radial densities, with support radius h (Gaussian: 5h). */
function getSpatialKernel(
  kernel: GPUKernelDensityKernel,
  bandwidth: number,
  distance: number
): number {
  const squared = (distance / bandwidth) ** 2;
  if (kernel === 'gaussian') {
    return squared < 25 ? Math.exp(-0.5 * squared) / (2 * Math.PI * bandwidth ** 2) : 0;
  }
  if (squared >= 1) return 0;
  return kernel === 'quartic'
    ? (3 * (1 - squared) ** 2) / (Math.PI * bandwidth ** 2)
    : (2 * (1 - squared)) / (Math.PI * bandwidth ** 2);
}

function getSpatialMass(
  kernel: GPUKernelDensityKernel,
  bandwidth: number,
  center: readonly [number, number],
  extent: readonly [number, number, number, number]
): number {
  const [minimumX, minimumY, maximumX, maximumY] = extent;
  // Independent angular integration of exact radial mass over a rectangle's oriented edges.
  return getOracleKernelMassFan(
    {
      positions: new Float64Array([
        minimumX,
        minimumY,
        maximumX,
        minimumY,
        maximumX,
        maximumY,
        minimumX,
        maximumY
      ]),
      polygonOffsets: new Uint32Array([0, 1]),
      ringOffsets: new Uint32Array([0, 4])
    },
    center,
    bandwidth,
    kernel
  );
}

/**
 * Binary64 CPU reference of the space-time kernel density of sparr 2.3-16 `spattemp.density`
 * (fixed bandwidths, 'uniform' edge correction). sparr is R-only and R is not installed here, so
 * the sparr fixture is pending and this brute-force triple loop, written from the definition in the
 * `GPUSpaceTimeKernelDensity` documentation, is the oracle.
 *
 * Spatial and temporal kernels are independent formulas. Spatial window mass uses angular
 * integration of radial mass; temporal window mass uses a Simpson integral. No runtime math
 * from the contributor under test is used.
 */
export type SpaceTimeDensityScene = {
  /** Interleaved x, y; already rounded to float32 by the caller. */
  positions: number[];
  times: number[];
  weights?: number[];
};

export type SpaceTimeDensityReferenceOptions = {
  scene: SpaceTimeDensityScene;
  columns: number;
  rows: number;
  /** `[minX, minY, maxX, maxY]` of both the grid and the index domain; rows advance north. */
  extent: readonly [number, number, number, number];
  bandwidth: number;
  temporalBandwidth: number;
  firstSliceTime: number;
  sliceInterval: number;
  sliceCount: number;
  spatialKernel: GPUKernelDensityKernel;
  temporalKernel: 'gaussian' | 'quartic' | 'epanechnikov';
  spatialEdge: boolean;
  temporalEdge: boolean;
  timeWindow?: readonly [number, number];
  normalization: 'intensity' | 'probability';
};

function getTemporalKernel(
  kernel: SpaceTimeDensityReferenceOptions['temporalKernel'],
  bandwidth: number,
  offset: number
): number {
  const s = offset / bandwidth;
  if (kernel === 'gaussian') {
    return Math.abs(s) < 5 ? Math.exp(-0.5 * s * s) / Math.sqrt(2 * Math.PI) / bandwidth : 0;
  }
  if (Math.abs(s) >= 1) {
    return 0;
  }
  return kernel === 'epanechnikov'
    ? (0.75 * (1 - s * s)) / bandwidth
    : (0.9375 * (1 - s * s) ** 2) / bandwidth;
}

/** Simpson integral of the (untruncated for gaussian) temporal kernel over a window. */
function getTemporalMass(
  kernel: SpaceTimeDensityReferenceOptions['temporalKernel'],
  bandwidth: number,
  time: number,
  window: readonly [number, number]
): number {
  const span = kernel === 'gaussian' ? 12 * bandwidth : bandwidth;
  const low = Math.max(window[0], time - span);
  const high = Math.min(window[1], time + span);
  if (!(high > low)) {
    return 0;
  }
  const steps = 4000;
  const step = (high - low) / steps;
  let sum = 0;
  for (let index = 0; index <= steps; index++) {
    const factor = index === 0 || index === steps ? 1 : index % 2 === 1 ? 4 : 2;
    const value =
      kernel === 'gaussian'
        ? Math.exp(-0.5 * ((low + index * step - time) / bandwidth) ** 2) /
          Math.sqrt(2 * Math.PI) /
          bandwidth
        : getTemporalKernel(kernel, bandwidth, low + index * step - time);
    sum += factor * value;
  }
  return (sum * step) / 3;
}

/**
 * Returns the density in cell-major order, `(row * columns + column) * sliceCount + slice`, NaN
 * where the voxel is undefined. Also returns the included weight and the excluded row count.
 */
export function getSpaceTimeDensityReference(options: SpaceTimeDensityReferenceOptions): {
  values: number[];
  weightSum: number;
  invalidCount: number;
} {
  const {scene, columns, rows, extent, bandwidth, temporalBandwidth, sliceCount} = options;
  const rowCount = scene.times.length;
  const cellWidth = (extent[2] - extent[0]) / columns;
  const cellHeight = (extent[3] - extent[1]) / rows;
  const timeWindow: readonly [number, number] = options.timeWindow ?? [
    options.firstSliceTime,
    options.firstSliceTime + (sliceCount - 1) * options.sliceInterval
  ];
  const included: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const x = scene.positions[2 * row];
    const y = scene.positions[2 * row + 1];
    const time = scene.times[row];
    const weight = scene.weights ? scene.weights[row] : 1;
    let keep =
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      x >= extent[0] &&
      x <= extent[2] &&
      y >= extent[1] &&
      y <= extent[3] &&
      Number.isFinite(time) &&
      Number.isFinite(weight) &&
      weight >= 0;
    if (keep && options.temporalEdge) {
      keep = time >= timeWindow[0] && time <= timeWindow[1];
    }
    if (keep) {
      included.push(row);
    }
  }
  const weightSum = included.reduce(
    (sum, row) => sum + (scene.weights ? scene.weights[row] : 1),
    0
  );
  const values: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const centerX = extent[0] + (column + 0.5) * cellWidth;
      const centerY = extent[1] + (row + 0.5) * cellHeight;
      for (let slice = 0; slice < sliceCount; slice++) {
        const time = options.firstSliceTime + slice * options.sliceInterval;
        let total = 0;
        for (const point of included) {
          const distance = Math.hypot(
            scene.positions[2 * point] - centerX,
            scene.positions[2 * point + 1] - centerY
          );
          total +=
            (scene.weights ? scene.weights[point] : 1) *
            getSpatialKernel(options.spatialKernel, bandwidth, distance) *
            getTemporalKernel(options.temporalKernel, temporalBandwidth, time - scene.times[point]);
        }
        let valid = true;
        if (options.spatialEdge) {
          // Included points are inside the extent, which is also the spatial window here.
          total /= getSpatialMass(options.spatialKernel, bandwidth, [centerX, centerY], extent);
        }
        if (options.temporalEdge) {
          valid = time >= timeWindow[0] && time <= timeWindow[1];
          total /= getTemporalMass(options.temporalKernel, temporalBandwidth, time, timeWindow);
        }
        if (options.normalization === 'probability') {
          valid = valid && weightSum > 0;
          total /= weightSum;
        }
        values.push(valid ? total : Number.NaN);
      }
    }
  }
  return {values, weightSum, invalidCount: rowCount - included.length};
}

/** A reproducible scene: clusters, ties, coincident points and every kind of excluded row. */
export function createSpaceTimeDensityScene(
  seed: number,
  withWeights: boolean
): SpaceTimeDensityScene {
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  const scene: SpaceTimeDensityScene = {
    positions: [],
    times: [],
    weights: withWeights ? [] : undefined
  };
  const add = (x: number, y: number, time: number, weight = 0.5 + Math.floor(random() * 6) / 4) => {
    scene.positions.push(Math.fround(x), Math.fround(y));
    scene.times.push(Math.fround(time));
    scene.weights?.push(Math.fround(weight));
  };
  // Two moving clusters plus background.
  for (let index = 0; index < 90; index++) {
    const progress = random();
    add(3 + 1.2 * (random() - 0.5) + 2 * progress, 2 + 1.2 * (random() - 0.5), progress * 8);
  }
  for (let index = 0; index < 70; index++) {
    add(9 + 1.5 * (random() - 0.5), 6 + 1.5 * (random() - 0.5), 6 + random() * 8);
  }
  for (let index = 0; index < 60; index++) {
    add(random() * 12, random() * 8, random() * 14);
  }
  // Ties in time, a coincident pair, and points on the domain edge.
  add(6, 4, 5);
  add(6, 4, 5);
  add(6, 4.001, 5);
  add(0, 0, 0);
  add(12, 8, 14);
  // Excluded rows: non-finite time, non-finite position, negative weight, outside the domain.
  add(5, 3, Number.NaN);
  add(Number.NaN, 3, 4);
  add(5, 3, 4, -1);
  add(30, 3, 4);
  add(5, -9, 4);
  return scene;
}
