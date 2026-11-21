// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import {BIN_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/bin-index-wgsl';
import {BIN_INDEX_WGSL as COMPATIBILITY_BIN_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/shared/bin-index-wgsl';

it('regression owns unchanged equal-width bin WGSL while the old path remains usable', () => {
  // SHA-256 from git show 85a29fcc:.../shared/bin-index-wgsl.ts, between the backticks.
  expect(createHash('sha256').update(BIN_INDEX_WGSL).digest('hex')).toBe(
    '59aa6817d8b4e0efe3f53a30e9c05211343312f25c60086cadd0c1b397ab5f5d'
  );
  expect(COMPATIBILITY_BIN_INDEX_WGSL).toBe(BIN_INDEX_WGSL);
  expect(BIN_INDEX_WGSL).toContain('if (!(upper > lower)) { return 0u; }');
  expect(BIN_INDEX_WGSL).toContain('return min(u32(fraction * f32(binCount)), binCount - 1u);');
});
