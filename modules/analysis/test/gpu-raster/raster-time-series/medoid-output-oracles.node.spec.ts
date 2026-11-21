// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {computeRasterMedoidCompositeOnCPU} from '../../../src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation';
import {expectLiteralRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';

it('medoid CPU ports match the hand-derived Euclidean minimizer and all-masked sentinel', () => {
  for (const missing of [false, true]) {
    // Sums of distances: 8.414214, 10.236068, 12.162278, 6.812559; unique minimizer is slice 3.
    const result = computeRasterMedoidCompositeOnCPU(
      new Float32Array([0, 0, 3, 0, 0, 4, 1, 1]),
      {cellCount: 1, sliceCount: 4, bandCount: 2},
      new Uint32Array(4).fill(missing ? 0 : 1)
    );
    expectLiteralRows(result.composite, missing ? [NaN, NaN] : [1, 1], 0);
    expectLiteralRows(result.sliceIndex, missing ? [0xffffffff] : [3], 0);
    expectLiteralRows(result.status, missing ? [1] : [0], 0);
  }
});
