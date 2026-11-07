// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUChunkedSpatialWeightsAlgebra,
  GPUChunkedSpatialWeightsTranspose,
  GPUChunkedSpatialWeightsTransform,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsTransformOperation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import type {CatalogGpuSession} from './catalog-gpu-session';
import type {
  WeightsCombineRule,
  WeightsCsr,
  WeightsSetOperation,
  WeightsTransformOperation
} from './catalog-platform4-weights';

/** One row range of a CSR: offsets restart at 0 and neighbors keep global row ids. */
export type WeightsCsrChunk = {rowStart: number; rowCount: number; csr: WeightsCsr};

/** Splits a CSR into consecutive row chunks of at most `chunkRowCount` rows (at least one chunk). */
export function splitWeightsCsr(csr: WeightsCsr, chunkRowCount: number): WeightsCsrChunk[] {
  const rowCount = csr.offsets.length - 1;
  const size = Math.max(1, Math.min(rowCount, chunkRowCount || rowCount));
  const chunks: WeightsCsrChunk[] = [];
  for (let rowStart = 0; rowStart < rowCount; rowStart += size) {
    const count = Math.min(size, rowCount - rowStart);
    const begin = csr.offsets[rowStart];
    const end = csr.offsets[rowStart + count];
    chunks.push({
      rowStart,
      rowCount: count,
      csr: {
        offsets: Uint32Array.from(
          csr.offsets.subarray(rowStart, rowStart + count + 1),
          value => value - begin
        ),
        neighbors: csr.neighbors.slice(begin, end),
        weights: csr.weights.slice(begin, end)
      }
    });
  }
  return chunks;
}

/**
 * Rows per chunk: the requested size, or (0) as many rows as keep every chunk's slot arrays inside
 * one storage binding of the device.
 */
export function getWeightsChunkRowCount(
  device: Device,
  csr: WeightsCsr,
  requested: number
): number {
  const rowCount = csr.offsets.length - 1;
  if (requested > 0) return Math.min(requested, rowCount);
  const slotLimit = Math.floor(device.limits.maxStorageBufferBindingSize / 4);
  if (csr.neighbors.length <= slotLimit) return rowCount;
  const slotsPerRow = Math.max(1, Math.ceil(csr.neighbors.length / rowCount));
  return Math.max(1, Math.floor(slotLimit / slotsPerRow));
}

function uploadChunks(
  session: CatalogGpuSession,
  name: string,
  csr: WeightsCsr,
  chunkRowCount: number,
  transform: 'raw' | 'row'
): {chunked: GPUChunkedSpatialWeights; chunks: WeightsCsrChunk[]} {
  const rowCount = csr.offsets.length - 1;
  const chunks = splitWeightsCsr(csr, chunkRowCount);
  return {
    chunks,
    chunked: {
      rowCount,
      information: {rowCount, transform, symmetric: 'unknown'},
      chunks: chunks.map((chunk, index) => ({
        rowStart: chunk.rowStart,
        rowCount: chunk.rowCount,
        weights: {
          offsets: session.input(`${name}-${index}-offsets`, chunk.csr.offsets, 'uint32'),
          neighbors: session.input(`${name}-${index}-neighbors`, chunk.csr.neighbors, 'uint32'),
          weights: session.input(`${name}-${index}-weights`, chunk.csr.weights, 'float32'),
          information: {rowCount: chunk.rowCount, transform, symmetric: 'unknown'}
        }
      }))
    }
  };
}

export type ChunkedWeightsResult = {csr: WeightsCsr; chunkCount: number};

/** Joins per-chunk CSR readbacks into one CSR for the table output (the catalog publishes plain rows). */
function joinChunkResults(
  parts: {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array; rowCount: number}[]
): WeightsCsr {
  const rowCount = parts.reduce((sum, part) => sum + part.rowCount, 0);
  const offsets = new Uint32Array(rowCount + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  let row = 0;
  for (const part of parts) {
    const used = part.offsets[part.rowCount];
    for (let index = 0; index < part.rowCount; index++)
      offsets[row + index + 1] = neighbors.length + part.offsets[index + 1];
    for (let slot = 0; slot < used; slot++) {
      neighbors.push(part.neighbors[slot]);
      weights.push(part.weights[slot]);
    }
    row += part.rowCount;
  }
  return {offsets, neighbors: Uint32Array.from(neighbors), weights: Float32Array.from(weights)};
}

export async function runChunkedWeightsAlgebraGpu(
  device: Device,
  left: WeightsCsr,
  right: WeightsCsr,
  operation: WeightsSetOperation,
  rule: WeightsCombineRule,
  requestedChunkRows: number
): Promise<ChunkedWeightsResult> {
  const chunkRowCount = getWeightsChunkRowCount(
    device,
    left.neighbors.length > right.neighbors.length ? left : right,
    requestedChunkRows
  );
  return withCatalogGpuSession(device, 'catalog-chunked-weights-algebra', async session => {
    const leftUpload = uploadChunks(session, 'left', left, chunkRowCount, 'raw');
    const rightUpload = uploadChunks(session, 'right', right, chunkRowCount, 'raw');
    const outputs = leftUpload.chunks.map((chunk, index) => {
      // A union cannot have more slots than both operand chunks together.
      const capacity = Math.max(
        1,
        chunk.csr.neighbors.length + rightUpload.chunks[index].csr.neighbors.length
      );
      return {
        rowCount: chunk.rowCount,
        offsets: session.output(`out-${index}-offsets`, 'uint32', chunk.rowCount + 1),
        neighbors: session.output(`out-${index}-neighbors`, 'uint32', capacity),
        weights: session.output(`out-${index}-weights`, 'float32', capacity),
        overflow: session.output(`out-${index}-overflow`, 'uint32', 1)
      };
    });
    session.graph.add(
      new GPUChunkedSpatialWeightsAlgebra({
        id: 'catalog-chunked-weights-algebra',
        operation,
        left: leftUpload.chunked,
        right: rightUpload.chunked,
        weightRule: rule,
        output: {
          chunks: outputs.map((output, index) => ({
            rowStart: leftUpload.chunks[index].rowStart,
            rowCount: output.rowCount,
            weights: {
              offsets: output.offsets.view,
              neighbors: output.neighbors.view,
              weights: output.weights.view,
              status: {overflow: output.overflow.view}
            }
          }))
        }
      })
    );
    session.run();
    const parts = [];
    for (const output of outputs) {
      // The capacity is the sum of both operand chunks, so the overflow flag cannot be set.
      if ((await session.readScalar(output.overflow)) !== 0)
        throw new CatalogGpuUnsupportedError('Chunk overflow');
      const offsets = await session.readUint32(output.offsets);
      parts.push({
        rowCount: output.rowCount,
        offsets,
        neighbors: await session.readUint32(output.neighbors),
        weights: await session.readFloat32(output.weights)
      });
    }
    return {csr: joinChunkResults(parts), chunkCount: outputs.length};
  });
}

export async function runChunkedWeightsTransposeGpu(
  device: Device,
  csr: WeightsCsr,
  requestedChunkRows: number
): Promise<ChunkedWeightsResult> {
  const rowCount = csr.offsets.length - 1;
  const chunkRowCount = getWeightsChunkRowCount(device, csr, requestedChunkRows);
  return withCatalogGpuSession(device, 'catalog-chunked-weights-transpose', async session => {
    const upload = uploadChunks(session, 'input', csr, chunkRowCount, 'raw');
    // The transpose is one whole CSR; its capacity is the summed chunk capacity.
    const capacity = Math.max(1, csr.neighbors.length);
    const offsets = session.output('out-offsets', 'uint32', rowCount + 1);
    const neighbors = session.output('out-neighbors', 'uint32', capacity);
    const weights = session.output('out-weights', 'float32', capacity);
    session.graph.add(
      new GPUChunkedSpatialWeightsTranspose({
        id: 'catalog-chunked-weights-transpose',
        weights: upload.chunked,
        output: {offsets: offsets.view, neighbors: neighbors.view, weights: weights.view}
      })
    );
    session.run();
    const resultOffsets = await session.readUint32(offsets);
    const used = resultOffsets[rowCount];
    return {
      csr: {
        offsets: resultOffsets,
        neighbors: (await session.readUint32(neighbors)).slice(0, used),
        weights: (await session.readFloat32(weights)).slice(0, used)
      },
      chunkCount: upload.chunks.length
    };
  });
}

export async function runChunkedWeightsTransformGpu(
  device: Device,
  csr: WeightsCsr,
  operation: WeightsTransformOperation,
  requestedChunkRows: number
): Promise<{weights: Float32Array; chunkCount: number}> {
  const chunkRowCount = getWeightsChunkRowCount(device, csr, requestedChunkRows);
  return withCatalogGpuSession(device, 'catalog-chunked-weights-transform', async session => {
    const upload = uploadChunks(session, 'input', csr, chunkRowCount, 'raw');
    const outputs = upload.chunks.map((chunk, index) =>
      session.output(`out-${index}-weights`, 'float32', Math.max(1, chunk.csr.neighbors.length))
    );
    session.graph.add(
      new GPUChunkedSpatialWeightsTransform({
        id: 'catalog-chunked-weights-transform',
        operation: operation as GPUSpatialWeightsTransformOperation,
        weights: upload.chunked,
        output: outputs.map(output => output.view)
      })
    );
    session.run();
    const joined = new Float32Array(csr.neighbors.length);
    let slot = 0;
    for (let index = 0; index < outputs.length; index++) {
      const count = upload.chunks[index].csr.neighbors.length;
      joined.set((await session.readFloat32(outputs[index])).subarray(0, count), slot);
      slot += count;
    }
    return {weights: joined, chunkCount: outputs.length};
  });
}
