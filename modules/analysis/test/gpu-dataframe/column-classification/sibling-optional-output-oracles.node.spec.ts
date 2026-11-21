// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeClassBreaksOnCPU} from '../../../src/dataframe-analysis/column-classification/cpu-class-breaks-operation';
import {computeColumnQuantilesOnCPU} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {getGPUColumnQuantilesParameterValues} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';

it('class breaks and quantiles CPU twins match hand-derived output literals', () => {
  // Sorted finite unmasked rows are [0, 2, 4, 6]. See the adjacent GPU spec's derivation.
  const values = Float32Array.of(0, 2, 4, 6, NaN, 100);
  const mask = Uint32Array.of(1, 1, 1, 1, 1, 0);
  const breaks = computeClassBreaksOnCPU(
    values,
    getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 3}, 4),
    {maximumClassCount: 4, methods: ['equal-interval']},
    mask
  );
  expect(Array.from(breaks.breaks.subarray(0, 4))).toEqual([0, 2, 4, 6]);
  expect(Number.isNaN(breaks.breaks[4])).toBe(true);
  expect(Array.from(breaks.classCount)).toEqual([3]);
  expect(Array.from(breaks.classCounts)).toEqual([1, 1, 2, 0]);
  const quantiles = computeColumnQuantilesOnCPU(
    {
      values,
      mask,
      parameters: getGPUColumnQuantilesParameterValues({
        quantiles: [0, 0.5, 1],
        filterRange: [0.25, 0.75]
      })
    },
    3
  );
  expect(Array.from(quantiles.quantiles)).toEqual([0, 3, 6]);
  expect(Array.from(quantiles.validCount)).toEqual([4]);
  expect(Array.from(quantiles.filterBounds)).toEqual([2, 4]);
  expect(Array.from(quantiles.outputMask)).toEqual([0, 1, 1, 0, 0, 0]);
});
