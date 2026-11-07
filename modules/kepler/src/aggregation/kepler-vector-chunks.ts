// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';

/** Returns the ordered chunk views of a vector, or the single view itself. */
export function getKeplerVectorChunks<Format extends GPUVectorFormat>(
  vector: GraphDataView<Format> | GraphVectorView<Format>
): readonly GraphDataView<Format>[] {
  return vector instanceof GraphVectorView ? vector.data : [vector];
}

/**
 * Pairs the non-empty chunks of source columns that come from the same table.
 * Columns of one table share record-batch boundaries, so chunks are zipped one to one;
 * mismatched boundaries are rejected instead of being re-sliced or repacked.
 */
export function zipKeplerVectorChunks<const Formats extends readonly GPUVectorFormat[]>(
  vectors: {
    readonly [Index in keyof Formats]:
      | GraphDataView<Formats[Index]>
      | GraphVectorView<Formats[Index]>;
  }
): readonly {readonly [Index in keyof Formats]: GraphDataView<Formats[Index]>}[] {
  const columns = vectors.map(vector =>
    getKeplerVectorChunks(vector).filter(chunk => chunk.length)
  );
  const chunkCount = columns[0]?.length ?? 0;
  for (const column of columns) {
    // Group By source columns must share chunk boundaries
    if (column.length !== chunkCount) throw new Error('chunk boundary mismatch');
  }
  const result: GraphDataView[][] = [];
  for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
    const chunks = columns.map(column => column[chunkIndex]);
    // Group By source columns must share chunk boundaries
    if (chunks.some(chunk => chunk.length !== chunks[0].length))
      throw new Error('chunk boundary mismatch');
    result.push(chunks);
  }
  return result as {readonly [Index in keyof Formats]: GraphDataView<Formats[Index]>}[];
}
