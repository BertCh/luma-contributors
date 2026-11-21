// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSinkhornDivergenceOnCPU} from '../../../src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';

it('Sinkhorn CPU counts invalid masses in both distributions independently', () => {
  for (const dirty of [true, false]) {
    // Same hand-counted oracle as the operation spec: three source plus two target invalid cells.
    const result = computeSinkhornDivergenceOnCPU(
      Float32Array.from(dirty ? [1, NaN, -2, Infinity] : [1, 0, 0, 0]),
      Float32Array.from(dirty ? [1, -Infinity, 0, NaN] : [1, 0, 0, 0]),
      getGPUSinkhornDivergenceParameterValues({cellWidth: 1, cellHeight: 1, epsilon: 1}),
      {
        grid: {width: 4, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
        maximumIterations: 2
      }
    );
    expect(Array.from(result.invalidCount)).toEqual([dirty ? 5 : 0]);
    expect(result.divergence[0]).toBe(0);
  }
});
