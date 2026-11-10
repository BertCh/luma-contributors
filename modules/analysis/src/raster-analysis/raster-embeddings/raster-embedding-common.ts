// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GraphEmbeddingMatrix} from '@luma.gl/graph/gpu-vector-search';
import {
  validateCapacity,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Longest embedding vector. Kernels keep one vector (and the nearest-neighbour or probe state next
 * to it) in per-thread private memory, so the length is a per-thread bound: 128 floats is 512
 * bytes, well inside the 8 KiB function-memory budget of the kernel audit. AlphaEarth has 64
 * bands and TESSERA v2 up to 128.
 */
export const MAXIMUM_RASTER_EMBEDDING_DIMENSIONS = 128;

/** NoData of AlphaEarth's int8 embeddings; a pixel with this value in any band is invalid. */
export const RASTER_EMBEDDING_NODATA = -128;

/** Label of an invalid pixel in classification and clustering outputs. */
export const RASTER_EMBEDDING_NO_LABEL = 0xffffffff;

/** Embedding rasters are addressed `pixel * dimensions + band`; this is the shared element-count check. */
export function validateRasterEmbeddingShape(
  id: string,
  pixelCount: number,
  dimensions: number
): void {
  validateCapacity(id, 'pixelCount', pixelCount);
  validateCapacity(id, 'dimensions', dimensions, 1, MAXIMUM_RASTER_EMBEDDING_DIMENSIONS);
  // Element indices are u32 in the kernels.
  validateRequirement(
    id,
    pixelCount * dimensions <= 0xffffffff,
    'pixelCount times dimensions must stay below 2^32'
  );
}

/** Validates a float32 `pixelCount * dimensions` view and an optional `pixelCount` validity view. */
export function validateRasterEmbeddingVectors(
  id: string,
  name: string,
  vectors: GraphDataView<'float32'>,
  pixelCount: number,
  dimensions: number,
  validity?: GraphDataView<'uint32'>
): void {
  validatePackedView(vectors, ['float32'], `${id} ${name}`);
  validateRequirement(
    id,
    vectors.length === pixelCount * dimensions,
    `${name} must hold pixelCount times dimensions entries`
  );
  if (validity) {
    validatePackedView(validity, ['uint32'], `${id} ${name} validity`);
    validateRequirement(
      id,
      validity.length === pixelCount,
      `${name} validity must hold pixelCount entries`
    );
  }
}

/** Validates the packed int8 view: four components per `uint32`, little-endian, pixel-major. */
export function validateRasterEmbeddingQuantized(
  id: string,
  name: string,
  quantized: GraphDataView<'uint32'>,
  pixelCount: number,
  dimensions: number
): void {
  validatePackedView(quantized, ['uint32'], `${id} ${name}`);
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
 * Describes one float32 `pixelCount * dimensions` view (and optional validity) as the single-chunk
 * `GraphEmbeddingMatrix` that the published `@luma.gl/graph/gpu-vector-search` classes consume.
 *
 * @internal
 */
export function createRasterEmbeddingMatrix(
  vectors: GraphDataView<'float32'>,
  pixelCount: number,
  dimensions: number,
  validity?: GraphDataView<'uint32'>
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
        validity
      }
    ]
  };
}
