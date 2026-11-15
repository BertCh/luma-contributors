// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphViewChunks} from '@luma-contributors/analysis/analysis-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {
  type GraphVectorView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';

/**
 * Intersects the logical chunk boundaries of several equal-length graph vectors using borrowed
 * views through the public graph.createDataView, never packed buffers. Each span keeps the format
 * at its operand position. Matches the unexported alignGraphVectorViews of published
 * @luma.gl/graph.
 */
export function alignKeplerGraphVectorViews<
  Parameters,
  const Formats extends readonly GPUVectorFormat[]
>(
  graph: GPUCommandGraph<Parameters>,
  vectors: {
    readonly [Index in keyof Formats]:
      | GraphDataView<Formats[Index]>
      | GraphVectorView<Formats[Index]>;
  }
): readonly {readonly [Index in keyof Formats]: GraphDataView<Formats[Index]>}[] {
  const operands = vectors as readonly (
    | GraphDataView<GPUVectorFormat>
    | GraphVectorView<GPUVectorFormat>
  )[];
  const length = operands[0]?.length ?? 0;
  if (operands.some(vector => vector.length !== length)) {
    throw new Error('Aligned vectors must have equal logical lengths');
  }
  const chunks = operands.map(getGraphViewChunks);
  const indices = operands.map(() => 0);
  const offsets = operands.map(() => 0);
  const result: GraphDataView<GPUVectorFormat>[][] = [];
  let position = 0;
  while (position < length) {
    for (let index = 0; index < chunks.length; index++) {
      while (chunks[index][indices[index]]?.length === offsets[index]) {
        indices[index]++;
        offsets[index] = 0;
      }
    }
    const current = chunks.map((data, index) => data[indices[index]]);
    const span = Math.min(...current.map((data, index) => data.length - offsets[index]));
    const aligned: GraphDataView<GPUVectorFormat>[] = [];
    for (let index = 0; index < current.length; index++) {
      const data = current[index];
      // Preserve identity for in-place operands, but never merge distinct source views.
      const matchingIndex = current.findIndex(
        (candidate, candidateIndex) =>
          candidateIndex < index && candidate === data && offsets[candidateIndex] === offsets[index]
      );
      aligned.push(
        matchingIndex >= 0
          ? aligned[matchingIndex]
          : offsets[index] === 0 && span === data.length
            ? data
            : graph.createDataView(data.buffer, {
                format: data.format,
                length: span,
                byteOffset: data.byteOffset + offsets[index] * data.byteStride,
                byteStride: data.byteStride,
                rowByteLength: data.rowByteLength
              })
      );
    }
    result.push(aligned);
    for (let index = 0; index < offsets.length; index++) offsets[index] += span;
    position += span;
  }
  return result as unknown as readonly {
    readonly [Index in keyof Formats]: GraphDataView<Formats[Index]>;
  }[];
}
