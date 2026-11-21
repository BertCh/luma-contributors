// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as generators from '../../../src/gpu-spatial-analysis/grid-generators/index';

it('exports the exact float32 half-cell axis limit through the grid barrel', () => {
  // Half-integer cell-center indices remain representable through 2^23 - 0.5.
  expect(generators.MAXIMUM_GPU_GRID_CENTER_AXIS_CELL_COUNT).toBe(8388608);
});

it('public sizing helpers reject unknown grid spellings', () => {
  expect(() => generators.getGPUGridCellCount('hexagonal' as never, 2, 3)).toThrow(
    /gridType must be.*square.*hex.*triangle.*point/
  );
  expect(() => generators.getGPUGridCornerCount('hexagonal' as never, 2, 3)).toThrow(
    /gridType must be.*square.*hex/
  );
});
