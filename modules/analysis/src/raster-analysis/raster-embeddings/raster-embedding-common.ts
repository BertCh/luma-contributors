// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {RasterEmbeddingView} from './raster-embedding-chunks';
import type {GraphEmbeddingMatrix} from '@luma.gl/graph/gpu-vector-search';
import {
  getGraphViewChunks,
  validateCapacity,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Bounds per-pixel band loops to 1,024 streamed bands; each storage-backed chunk must also fit its binding. */
export const MAXIMUM_RASTER_EMBEDDING_DIMENSIONS = 1024;

/** Reference kernels retain their private-memory layouts and the CPU dense eigensolver its original work bound. */
export const MAXIMUM_REFERENCE_RASTER_EMBEDDING_DIMENSIONS = 128;

/** Every flattened component address within one chunk must fit a uint32 WGSL index. */
export const MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT = 0xffffffff;

/** NoData of AlphaEarth's int8 embeddings; a pixel with this value in any band is invalid. */
export const RASTER_EMBEDDING_NODATA = -128;

/** Label of an invalid pixel in classification and clustering outputs. */
export const RASTER_EMBEDDING_NO_LABEL = 0xffffffff;

/** Embedding rasters are addressed `pixel * dimensions + band`; this is the shared element-count check. */
export function validateRasterEmbeddingShape(
  id: string,
  pixelCount: number,
  dimensions: number,
  maximumDimensions: number = MAXIMUM_REFERENCE_RASTER_EMBEDDING_DIMENSIONS,
  validatePackedElementCount: boolean = true
): void {
  validateCapacity(id, 'pixelCount', pixelCount);
  validateCapacity(id, 'dimensions', dimensions, 1, maximumDimensions);
  // Element indices are u32 in the kernels.
  validateRequirement(
    id,
    !validatePackedElementCount ||
      pixelCount * dimensions <= MAXIMUM_RASTER_EMBEDDING_ELEMENT_COUNT,
    'pixelCount times dimensions must stay below 2^32'
  );
}

/** Validates a float32 `pixelCount * dimensions` view and an optional `pixelCount` mask view. */
export function validateRasterEmbeddingVectors(
  id: string,
  name: string,
  vectors: RasterEmbeddingView<'float32'>,
  pixelCount: number,
  dimensions: number,
  mask?: RasterEmbeddingView<'uint32'>
): void {
  for (const chunk of getGraphViewChunks(vectors)) {
    validateSinglePackedView(chunk, ['float32'], `${id} ${name}`);
    validateRequirement(
      id,
      chunk.length % dimensions === 0,
      `${name} chunks must hold complete pixel vectors`
    );
  }
  validateRequirement(
    id,
    vectors.length === pixelCount * dimensions,
    `${name} must hold pixelCount times dimensions entries`
  );
  if (mask) {
    for (const chunk of getGraphViewChunks(mask))
      validateSinglePackedView(chunk, ['uint32'], `${id} ${name} mask`);
    validateRequirement(
      id,
      mask.length === pixelCount,
      `${name} mask must hold pixelCount entries`
    );
  }
}

/** Validates the packed int8 view: four components per `uint32`, little-endian, pixel-major. */
export function validateRasterEmbeddingQuantized(
  id: string,
  name: string,
  quantized: RasterEmbeddingView<'uint32'>,
  pixelCount: number,
  dimensions: number
): void {
  for (const chunk of getGraphViewChunks(quantized))
    validateSinglePackedView(chunk, ['uint32'], `${id} ${name}`);
  validateRequirement(
    id,
    quantized.length === Math.ceil((pixelCount * dimensions) / 4),
    `${name} must hold ceil(pixelCount times dimensions / 4) words`
  );
}

/**
 * WGSL `fn decodeEmbeddingComponent(element: u32) -> i32` that reads signed byte `element` (pixel
 * times dimensions plus band) of the packed int8 binding `bindingName`, plus
 * `fn getEmbeddingValue(byte: i32) -> f32`, the AlphaEarth decoding `sign(q) * (q / 127.5)^2`.
 *
 * @internal
 */
export function getRasterEmbeddingDecodeWGSL(bindingName: string): string {
  return /* wgsl */ `
fn readEmbeddingByte(element: u32) -> i32 {
  let word = ${bindingName}[${bindingName}Offset + (element >> 2u)];
  // extractBits on an i32 sign-extends the selected byte.
  return extractBits(bitcast<i32>(word), (element & 3u) * 8u, 8u);
}
fn getEmbeddingValue(byte: i32) -> f32 {
  let scaled = f32(byte) / 127.5;
  return select(scaled * scaled, -(scaled * scaled), byte < 0);
}`;
}

/**
 * Describes one float32 `pixelCount * dimensions` view (and optional mask) as the single-chunk
 * `GraphEmbeddingMatrix` that the published `@luma.gl/graph/gpu-vector-search` classes consume.
 *
 * @internal
 */
export function createRasterEmbeddingMatrix(
  vectors: GraphDataView<'float32'>,
  pixelCount: number,
  dimensions: number,
  mask?: GraphDataView<'uint32'>
): GraphEmbeddingMatrix {
  return {
    dimensions,
    rowCount: pixelCount,
    chunks: [
      {
        values: vectors,
        rowCount: pixelCount,
        rowStride: dimensions,
        byteOffset: vectors.byteOffset,
        sourceRowOffset: 0,
        validity: mask
      }
    ]
  };
}
