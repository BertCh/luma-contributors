// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it, expect} from 'vitest';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';
import {
  getRegressionTileLayout,
  getRowTileSumNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {getKahanSumWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-wgsl';
import {createRegressionTestContext} from './math-review-regression-test-utils';

/**
 * The merges of the regression family sum at most a few thousand tile partials. These scenes are
 * built so that a plain left-to-right float32 sum is visibly wrong and the exact answer is known
 * by hand:
 * - `[2^24, 1, 1, ...]` (999 ones): float32 `2^24 + 1` rounds back to `2^24`, so a plain sum
 *   stays at 16,777,216; the exact total is 16,778,215 (float32 spacing 2 there);
 * - 4,000 equal partials of 0.1 after a first partial of 2^20: the plain sum drifts by the
 *   rounding of 0.1 at that magnitude (spacing 0.125), far more than one unit.
 * A float Kahan or Neumaier correction is folded to zero by the shader compiler of the test GPU
 * (Metal, fast math), so the merges use a pairwise tree, which needs no error-free transformation.
 */
const ONE_COUNT = 999;
const EXACT_TOTAL = 2 ** 24 + ONE_COUNT;

it('the row-tile merge of the regression family sums [2^24, 1, 1, ...] to within one spacing', async () => {
  const context = await createRegressionTestContext();
  const {graph, input, output} = context;
  try {
    const tileCount = ONE_COUNT + 1;
    const rowCount = tileCount * 64;
    const layout = getRegressionTileLayout(rowCount);
    expect(layout).toEqual({tileRowCount: 64, tileCount});
    const total = output('float32', 1);
    // One term per tile: 2^24 in the first row of tile 0, 1 in the first row of every other tile.
    const flags = new Float32Array(rowCount);
    for (let tile = 0; tile < tileCount; tile++) flags[tile * 64] = tile === 0 ? 2 ** 24 : 1;
    for (const node of getRowTileSumNodes(graph, {
      id: 'tile-merge-sum',
      operation: 'RegressionTileMergeSpec',
      variant: 'sum',
      rowCount,
      layout,
      sumCount: 1,
      bindings: [{name: 'terms', view: input('float32', flags), type: 'f32', access: 'read'}],
      declarations: `fn accumulateRow(row: u32, sums: ptr<function, array<f32, TERM_COUNT>>) { (*sums)[0] += terms[termsOffset + row]; }`,
      partial: createTransientView(graph, 'tile-merge-partials', 'float32', tileCount),
      output: total.view
    }))
      graph.add(node);
    context.run();
    const value = (await readFloat32(total.buffer, 1))[0];
    // A plain serial sum returns exactly 2^24 (an error of 999).
    expect(Math.abs(value - EXACT_TOTAL), `row-tile merge returned ${value}`).toBeLessThanOrEqual(
      2
    );
  } finally {
    context.destroy();
  }
});

for (const [name, values, exact, tolerance] of [
  ['[2^24, 1, 1, ...]', [2 ** 24, ...new Array<number>(ONE_COUNT).fill(1)], EXACT_TOTAL, 2],
  // Math.fround(0.1) summed exactly in float64, plus 2^20.
  [
    '[2^20, 0.1, 0.1, ...]',
    [2 ** 20, ...new Array<number>(4000).fill(0.1)],
    2 ** 20 + 4000 * Math.fround(0.1),
    // Twelve pairwise levels, each rounding to the spacing of 0.125 at 2^20; the plain sum is 100 off.
    1
  ]
] as const)
  it(`the partial merge of the maximum-likelihood family sums ${name} to within a few spacings`, async () => {
    const context = await createRegressionTestContext();
    const {graph, input, output} = context;
    try {
      const total = output('float32', 1);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'partial-merge-sum',
          operation: 'RegressionPartialMergeSpec',
          variant: 'sum',
          invocationCount: 1,
          bindings: [
            {name: 'partial', view: input('float32', values), type: 'f32', access: 'read'},
            {name: 'total', view: total.view, type: 'f32', access: 'read_write'}
          ],
          body: `${getKahanSumWGSL('partial', 'index', '1u', `${values.length}u`)}
  total[totalOffset] = sum;`
        })
      );
      context.run();
      const value = (await readFloat32(total.buffer, 1))[0];
      expect(
        Math.abs(value - exact),
        `partial merge returned ${value}, exact ${exact}`
      ).toBeLessThanOrEqual(tolerance);
    } finally {
      context.destroy();
    }
  });
