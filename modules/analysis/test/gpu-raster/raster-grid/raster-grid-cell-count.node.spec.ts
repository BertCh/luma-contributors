// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  validateRasterGrid,
  RASTER_GRID_MAXIMUM_CELL_COUNT
} from '../../../src/raster-analysis/raster-band-utils';

const grid = {
  width: 1,
  height: 1,
  affine: [1, 0, 0, 0, -1, 0] as const,
  pixelInterpretation: 'area' as const
};

it('distinguishes invalid dimensions from an excessive uint32 grid cell count without allocation', () => {
  for (const size of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    for (const dimension of ['width', 'height']) {
      expect(() => validateRasterGrid('grid-probe', {...grid, [dimension]: size})).toThrow(
        /grid-probe grid dimensions must be positive integers/
      );
    }
  }
  expect(() => validateRasterGrid('grid-probe', {...grid, width: 65536, height: 65536})).toThrow(
    'grid-probe grid cell count (4294967296) must be at most 4294967295'
  );
  // (2^16 - 1)(2^16 + 1) = 2^32 - 1, exactly the largest uint32 cell count.
  expect(validateRasterGrid('grid-probe', {...grid, width: 65535, height: 65537})).toBe(4294967295);
});

it('names the uint32 grid cell-count maximum used by admission', () => {
  expect(RASTER_GRID_MAXIMUM_CELL_COUNT).toBe(4294967295);
});
