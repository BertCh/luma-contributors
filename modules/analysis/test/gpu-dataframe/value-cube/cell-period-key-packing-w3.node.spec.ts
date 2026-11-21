// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUCellKeyFreeBits,
  getGPUCellPeriodKeyPeriodBits,
  packGPUCellPeriodKey,
  unpackGPUCellPeriodKeyChecked
} from '../../../src/dataframe-analysis/value-cube/cell-period-key-packing';

const LEGACY_LAYOUT = 'legacy-one-bit';

it('requires an explicit layout on every host packing entry point (EGR-04 and EGR-05)', () => {
  expect(() => getGPUCellPeriodKeyPeriodBits(1, undefined as never)).toThrow(/layout.*received/);
  expect(() => packGPUCellPeriodKey(3n, 0, 1, 'ones', undefined as never)).toThrow(
    /layout.*received/
  );
  expect(() => unpackGPUCellPeriodKeyChecked(2n, 1, 'ones', undefined as never)).toThrow(
    /layout.*received/
  );
});

it('preserves cell bits with identity and reads persisted keys with the legacy layout (EGR-04)', () => {
  const layout = 'zero-period-bit-identity';
  expect(getGPUCellPeriodKeyPeriodBits(1, layout)).toBe(0);
  expect(packGPUCellPeriodKey(1n, 0, 1, 'zeros', layout)).toBe(1n);
  expect(unpackGPUCellPeriodKeyChecked(1n, 1, 'zeros', layout)).toEqual({cellKey: 1n, period: 0});
  expect(getGPUCellPeriodKeyPeriodBits(1, LEGACY_LAYOUT)).toBe(1);
  expect(packGPUCellPeriodKey(3n, 0, 1, 'ones', LEGACY_LAYOUT)).toBe(2n);
  expect(unpackGPUCellPeriodKeyChecked(2n, 1, 'ones', LEGACY_LAYOUT)).toEqual({
    cellKey: 3n,
    period: 0
  });
  expect(() => getGPUCellPeriodKeyPeriodBits(2, layout)).toThrow(/periodCount.*1/);
  expect(() => getGPUCellPeriodKeyPeriodBits(1, 'typo' as never)).toThrow(
    /legacy-one-bit.*zero-period-bit-identity/
  );
});

it('distinguishes a missing key from malformed period payloads (EGR-05)', () => {
  expect(
    unpackGPUCellPeriodKeyChecked(0xffffffffffffffffn, 3, 'zeros', LEGACY_LAYOUT)
  ).toBeUndefined();
  expect(() => unpackGPUCellPeriodKeyChecked(3n, 3, 'zeros', LEGACY_LAYOUT)).toThrow(
    /period.*3.*periodCount.*3/
  );
  expect(() => unpackGPUCellPeriodKeyChecked(-1n, 3, 'zeros', LEGACY_LAYOUT)).toThrow(/key/);
  expect(() => packGPUCellPeriodKey(0n, 0, 1, 'typo' as never, LEGACY_LAYOUT)).toThrow(
    /ones.*zeros/
  );
});

it('rejects non-sentinel packed keys that reconstruct a missing cell (B6)', () => {
  const malformed = 0xfffffffffffffffen;
  expect(
    unpackGPUCellPeriodKeyChecked(0xffffffffffffffffn, 1, 'ones', LEGACY_LAYOUT)
  ).toBeUndefined();
  expect(() => unpackGPUCellPeriodKeyChecked(malformed, 1, 'ones', LEGACY_LAYOUT)).toThrow(
    /reconstruct.*missing cell/
  );
});

it('rejects unknown free-bit cell families (S4)', () => {
  expect(() => getGPUCellKeyFreeBits('typo' as never, 8)).toThrow(/h3.*quadbin.*s2/);
});

it('reserves the missing sentinel on the encoded side (earlier-fix note 2)', () => {
  // All low three bits become period 7, so fff8 | 7 would equal the reserved all-ones key.
  expect(packGPUCellPeriodKey(0xfffffffffffffff8n, 7, 8, 'zeros', LEGACY_LAYOUT)).toBeUndefined();
  expect(packGPUCellPeriodKey(0xfffffffffffffff8n, 6, 8, 'zeros', LEGACY_LAYOUT)).toBe(
    0xfffffffffffffffen
  );
});
