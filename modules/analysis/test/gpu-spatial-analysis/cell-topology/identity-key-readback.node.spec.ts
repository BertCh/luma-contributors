// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import {joinIdentityWords} from './identity-key-readback';

it('accepts typed-array identity words and requires the complete fixed-stride readback (B3)', () => {
  expectTypeOf<Parameters<typeof joinIdentityWords>[0]>().toEqualTypeOf<ArrayLike<number>>();
  // Literal low/high words identify two distinct keys, including a high-word-only key.
  expect(joinIdentityWords(Uint32Array.of(7, 0, 0, 1), 2)).toEqual([7n, 0x100000000n]);
  expect(() => joinIdentityWords(Uint32Array.of(7, 0), 2)).toThrow(/exactly 4 words/);
  expect(() => joinIdentityWords(Uint32Array.of(7, 0, 0, 1, 9, 0), 2)).toThrow(/exactly 4 words/);
});
