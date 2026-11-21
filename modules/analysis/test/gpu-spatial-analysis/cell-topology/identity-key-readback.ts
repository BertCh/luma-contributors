// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Little-endian identity readback, shared by host layout and device assertions. */
export function joinIdentityWords(words: ArrayLike<number>, expectedKeys: number): bigint[] {
  if (words.length !== 2 * expectedKeys)
    throw new Error(
      `identity readback needs exactly ${2 * expectedKeys} words; got ${words.length}`
    );
  return Array.from(
    {length: words.length / 2},
    (_, index) => BigInt(words[2 * index]) | (BigInt(words[2 * index + 1]) << 32n)
  );
}
