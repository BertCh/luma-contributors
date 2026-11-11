// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Stand-in embedding raster for the raster-embedding runs. No learned embedding extract is checked
// in, so every pixel's vector is its six Sentinel-2 bands (blue, green, red, near infrared, two
// shortwave infrared), standardized per band over the scene and scaled to unit length, the way
// AlphaEarth vectors are unit length. Cosine similarity of two such vectors compares spectral shape.

import {sampleReflectance} from '../../../../data/reflectance';
import {getRasterCell, getRasterCellCenter} from '../../../pick';

export const DIMENSIONS = 6;
export const BAND_NAMES = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2'] as const;
export const CLASS_NAMES = ['water', 'forest', 'open'] as const;

export type EmbeddingRaster = {
  columns: number;
  rows: number;
  pixelCount: number;
  /** Unit vectors, `pixel * DIMENSIONS + band`, row 0 at the north (top) edge. */
  vectors: Float32Array;
  /** Near-infrared reflectance, for a context raster. */
  nearInfrared: Float32Array;
  /** Normalized difference vegetation index. */
  vegetationIndex: Float32Array;
};

type World = {width: number; height: number};

export function createEmbeddingRaster(
  aspect: number,
  rows = 36,
  columnMultiple = 1
): EmbeddingRaster {
  const columns = Math.max(24, Math.round((rows * aspect) / columnMultiple) * columnMultiple);
  const pixelCount = columns * rows;
  const reflectance = sampleReflectance(columns, rows);
  const bands = [
    reflectance.blue,
    reflectance.green,
    reflectance.red,
    reflectance.nearInfrared,
    reflectance.shortwaveInfrared,
    reflectance.shortwaveInfrared2
  ];
  const vectors = new Float32Array(pixelCount * DIMENSIONS);
  bands.forEach((band, bandIndex) => {
    let sum = 0;
    for (const value of band) sum += value;
    const mean = sum / pixelCount;
    let squares = 0;
    for (const value of band) squares += (value - mean) ** 2;
    const deviation = Math.sqrt(squares / pixelCount) || 1;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      vectors[pixel * DIMENSIONS + bandIndex] = (band[pixel] - mean) / deviation;
    }
  });
  for (let pixel = 0; pixel < pixelCount; pixel++) normalizeVector(vectors, pixel);
  const vegetationIndex = reflectance.nearInfrared.map(
    (nearInfrared, pixel) =>
      (nearInfrared - reflectance.red[pixel]) /
      Math.max(1e-6, nearInfrared + reflectance.red[pixel])
  );
  return {
    columns,
    rows,
    pixelCount,
    vectors,
    nearInfrared: reflectance.nearInfrared,
    vegetationIndex
  };
}

export function normalizeVector(vectors: Float32Array, pixel: number): void {
  let length = 0;
  for (let band = 0; band < DIMENSIONS; band++) length += vectors[pixel * DIMENSIONS + band] ** 2;
  length = Math.sqrt(length) || 1;
  for (let band = 0; band < DIMENSIONS; band++) vectors[pixel * DIMENSIONS + band] /= length;
}

export function getVector(vectors: ArrayLike<number>, pixel: number): number[] {
  return Array.from({length: DIMENSIONS}, (_, band) => vectors[pixel * DIMENSIONS + band]);
}

export function getCosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let dot = 0;
  let leftLength = 0;
  let rightLength = 0;
  for (let band = 0; band < left.length; band++) {
    dot += left[band] * right[band];
    leftLength += left[band] ** 2;
    rightLength += right[band] ** 2;
  }
  return dot / Math.sqrt(leftLength * rightLength || 1);
}

/** The pixel under a world position; the raster fills the drawing world. */
export function getPixelAt(
  raster: {columns: number; rows: number},
  world: World,
  x: number,
  y: number
) {
  return getRasterCell(raster.columns, raster.rows, [0, 0, world.width, world.height], x, y);
}

export function getPixelCenter(
  raster: {columns: number; rows: number},
  world: World,
  pixel: number
): [number, number] {
  return getRasterCellCenter(
    raster.columns,
    raster.rows,
    [0, 0, world.width, world.height],
    pixel % raster.columns,
    Math.floor(pixel / raster.columns)
  );
}

/** The world-space box of one pixel, grown by `radius` pixels on every side. */
export function getPixelBox(
  raster: {columns: number; rows: number},
  world: World,
  pixel: number,
  radius = 0
) {
  const [x, y] = getPixelCenter(raster, world, pixel);
  const halfWidth = ((radius + 0.5) * world.width) / raster.columns;
  const halfHeight = ((radius + 0.5) * world.height) / raster.rows;
  return {
    shape: 'box' as const,
    minimum: [x - halfWidth, y - halfHeight] as [number, number],
    maximum: [x + halfWidth, y + halfHeight] as [number, number]
  };
}

/**
 * A small labelled set for the supervised runs, chosen by spectral rules (not field data): the
 * darkest near-infrared pixels are `water`, the greenest are `forest`, the rest `open`.
 */
export function createTrainingSet(raster: EmbeddingRaster, random: () => number, perClass = 8) {
  const order = (values: Float32Array) =>
    Array.from(values.keys()).sort((left, right) => values[left] - values[right]);
  const byNearInfrared = order(raster.nearInfrared);
  const byVegetation = order(raster.vegetationIndex);
  const tenth = Math.floor(raster.pixelCount / 10);
  const pools = [
    byNearInfrared.slice(0, tenth),
    byVegetation.slice(-3 * tenth),
    byVegetation.slice(2 * tenth, 6 * tenth).filter(pixel => raster.nearInfrared[pixel] > 0.12)
  ];
  const pixels: number[] = [];
  const labels: number[] = [];
  const used = new Set<number>();
  pools.forEach((pool, label) => {
    let picked = 0;
    for (let attempt = 0; picked < perClass && attempt < 200; attempt++) {
      const pixel = pool[Math.floor(random() * pool.length)];
      if (pixel === undefined || used.has(pixel)) continue;
      used.add(pixel);
      pixels.push(pixel);
      labels.push(label);
      picked++;
    }
  });
  const vectors = new Float32Array(pixels.length * DIMENSIONS);
  pixels.forEach((pixel, sample) =>
    vectors.set(
      raster.vectors.subarray(pixel * DIMENSIONS, (pixel + 1) * DIMENSIONS),
      sample * DIMENSIONS
    )
  );
  return {pixels, labels: Uint32Array.from(labels), vectors, count: pixels.length};
}

/** AlphaEarth int8 encoding, the inverse of `sign(q) * (q / 127.5)^2`. */
export function quantizeComponent(value: number): number {
  const magnitude = Math.round(Math.sqrt(Math.min(1, Math.abs(value))) * 127.5);
  return Math.max(-127, Math.min(127, Math.sign(value) * magnitude));
}

/** Packs signed bytes four to a little-endian uint32 word. */
export function packSignedBytes(bytes: ArrayLike<number>): Uint32Array {
  const int8 = Int8Array.from(bytes);
  const padded = new Int8Array(Math.ceil(int8.length / 4) * 4);
  padded.set(int8);
  return new Uint32Array(padded.buffer);
}
