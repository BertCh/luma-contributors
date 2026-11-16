// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_CELL_INDEX_FAMILIES,
  isGPUCellIndexFamily,
  validateCellIndexResolution,
  type GPUCellIndexFamily
} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-index-families';

it('accepts only declared cell families, excluding inherited Object properties', () => {
  for (const family of GPU_CELL_INDEX_FAMILIES) expect(isGPUCellIndexFamily(family)).toBe(true);
  for (const value of [
    'toString',
    'constructor',
    '__proto__',
    'valueOf',
    'hasOwnProperty',
    '',
    null,
    1,
    {}
  ]) {
    expect(isGPUCellIndexFamily(value)).toBe(false);
    expect(() =>
      validateCellIndexResolution('checked-family', value as GPUCellIndexFamily, 0)
    ).toThrow(/^checked-family family must be one of/);
  }
});
