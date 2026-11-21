// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePointToCellOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {CLIPPED_LONGITUDES, CLIPPED_LONGITUDE_FIXTURES} from './clipped-longitude-fixtures';

it('clips both longitude signs beyond 2^24 with family-specific endpoint bins', () => {
  for (const fixture of CLIPPED_LONGITUDE_FIXTURES) {
    const result = computePointToCellOnCPU(CLIPPED_LONGITUDES, fixture);
    const keys = Array.from(
      {length: 4},
      (_, row) => BigInt(result.cells[2 * row]) | (BigInt(result.cells[2 * row + 1]) << 32n)
    );
    expect(keys).toEqual([fixture.negative, fixture.positive, fixture.negative, fixture.positive]);
    expect(Array.from(result.validity)).toEqual([1, 1, 1, 1]);
  }
});
