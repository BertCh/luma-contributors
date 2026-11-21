// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * @internal Validate a word-pair layout and read only the selected pair. Ordinary numeric
 * arrays/array-likes and Int32/Uint32 views are words; only Uint8 views and ArrayBuffer are
 * bytes. Other typed views are rejected rather than guessed from their numeric length.
 */
export function getWideNumericWordPair(
  input: ArrayLike<number> | ArrayBuffer,
  row: number,
  port: string,
  exactPair: boolean
): readonly [number, number] {
  const tag = Object.prototype.toString.call(input);
  const isView = ArrayBuffer.isView(input);
  const byteView = isView && tag === '[object Uint8Array]' ? (input as Uint8Array) : undefined;
  const isBuffer = tag === '[object ArrayBuffer]';
  validateRequirement(
    port,
    !isView || Boolean(byteView) || tag === '[object Uint32Array]' || tag === '[object Int32Array]',
    `unsupported typed view ${tag}; use Uint32Array or Int32Array words, Uint8Array bytes, or ArrayBuffer`
  );
  const byteLength =
    byteView?.byteLength ?? (isBuffer ? (input as ArrayBuffer).byteLength : undefined);
  validateRequirement(
    port,
    byteLength === undefined || byteLength % 4 === 0,
    `bytes must contain whole 4-byte elements; received ${byteLength} bytes`
  );
  const words = input as ArrayLike<number>;
  const wordCount = byteLength === undefined ? words?.length : byteLength / 4;
  const requiredWordCount = (row + 1) * 2;
  validateRequirement(
    port,
    Number.isSafeInteger(wordCount) && wordCount >= 0,
    `expected a word array or byte readback; received length ${wordCount}`
  );
  validateRequirement(
    port,
    !exactPair || wordCount === 2,
    `expected length 2, received length ${wordCount}`
  );
  validateRequirement(
    port,
    wordCount >= requiredWordCount,
    `expected at least ${requiredWordCount} words, received length ${wordCount}`
  );
  validateRequirement(
    port,
    wordCount % 2 === 0,
    `expected complete 2-word pairs, received length ${wordCount}`
  );
  let pair: readonly [number, number];
  if (byteLength !== undefined) {
    const view = new DataView(
      byteView?.buffer ?? (input as ArrayBuffer),
      byteView?.byteOffset ?? 0,
      byteLength
    );
    pair = [view.getUint32(row * 8, true), view.getUint32(row * 8 + 4, true)];
  } else {
    pair = [words[row * 2], words[row * 2 + 1]];
  }
  for (let lane = 0; lane < 2; lane++) {
    validateRequirement(
      port,
      Number.isInteger(pair[lane]) && pair[lane] >= -0x80000000 && pair[lane] <= 0xffffffff,
      `lane ${lane} must contain a 32-bit word; received ${String(pair[lane])}`
    );
  }
  return pair;
}
