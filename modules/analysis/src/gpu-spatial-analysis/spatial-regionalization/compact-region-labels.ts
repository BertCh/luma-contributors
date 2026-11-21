// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Dense row labels and the original join key at each dense label; missing rows stay 0xffffffff. */
export type CompactRegionLabelsResult = {
  labels: Uint32Array;
  sourceLabels: Uint32Array;
  count: number;
};

/**
 * Compacts a uint32 label column in ascending source-label order without changing the input.
 * Use `Math.max(1, count)` for the evaluator's `labelCapacity` and output allocations, including
 * all-missing columns whose true region count is zero. `sourceLabels[labels[row]]` recovers the
 * source join key. Accepts packed buffer reads, respecting their byte offset and byte length.
 */
export function compactRegionLabels(
  input: Uint32Array | Uint8Array | ArrayBuffer
): CompactRegionLabelsResult {
  const byteLength = input.byteLength;
  validateRequirement(
    'compactRegionLabels',
    byteLength % 4 === 0,
    `input must hold whole uint32 elements; received ${byteLength} bytes`
  );
  const source =
    input instanceof Uint32Array
      ? input
      : new Uint32Array(
          input instanceof ArrayBuffer
            ? input.slice(0)
            : new Uint8Array(input.buffer, input.byteOffset, input.byteLength).slice().buffer
        );
  const sourceLabels = Uint32Array.from(
    [...new Set(source)]
      .filter(label => label !== 0xffffffff)
      .sort((first, second) => first - second)
  );
  const ranks = new Map(Array.from(sourceLabels, (label, rank) => [label, rank]));
  return {
    labels: Uint32Array.from(source, label => ranks.get(label) ?? 0xffffffff),
    sourceLabels,
    count: sourceLabels.length
  };
}
