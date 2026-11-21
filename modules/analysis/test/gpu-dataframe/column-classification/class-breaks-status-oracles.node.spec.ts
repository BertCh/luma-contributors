// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeClassBreaksOnCPU} from '../../../src/dataframe-analysis/column-classification/cpu-class-breaks-operation';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';

it('CPU class-break counts follow the same independent interval, empty and constant-column oracles', () => {
  const parameters = getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 4);
  for (const scenario of [
    // [0,2) contains both zeros; [2,4] contains 2 and 4. NaN is excluded.
    {values: [0, 0, 2, 4, NaN], mask: [1, 1, 1, 1, 1], count: [2], counts: [2, 2, 0, 0]},
    {values: [0, 0, 2, 4, NaN], mask: [0, 0, 0, 0, 0], count: [0], counts: [0, 0, 0, 0]},
    // One finite distinct value collapses to one class with support five.
    {values: [7, 7, 7, 7, 7], mask: [1, 1, 1, 1, 1], count: [1], counts: [5, 0, 0, 0]}
  ]) {
    const result = computeClassBreaksOnCPU(
      Float32Array.from(scenario.values),
      parameters,
      {maximumClassCount: 4, methods: ['equal-interval']},
      Uint32Array.from(scenario.mask)
    );
    expect(Array.from(result.classCount)).toEqual(scenario.count);
    expect(Array.from(result.classCounts)).toEqual(scenario.counts);
  }
});
