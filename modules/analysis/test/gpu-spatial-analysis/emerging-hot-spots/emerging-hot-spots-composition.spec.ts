// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  createCompositionOutputs,
  createSparseGatherComposition
} from './emerging-hot-spots-composition-fixture';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const unknown of [false, true])
  it(`CMP-03/CMP-12 composes a sparse gather in fixed cell-major order, unknown=${unknown}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const {producers, gathered, sliceOutput, output, explicitOutput} =
        createSparseGatherComposition(rig, unknown);
      rig.run(...producers);
      // Explicit sparse slice [0,3]: n=2, mean=1.5, variance=2.25; unknown leaves singleton [3].
      expectNumericArrayClose(
        await rig.readFloat(sliceOutput.globalStatistics),
        unknown ? [1, 3, 0, 0] : [2, 1.5, 2.25, 1.5],
        {absoluteTolerance: 1e-6}
      );
      // Hand moments of [0,1,3,2]: n=4, mean=1.5, variance=5/4. Excluding unknown:
      // [1,3,2]: n=3, mean=2, variance=2/3. NaN does not represent an observed zero.
      const moments = unknown
        ? [3, 2, 2 / 3, 0.816496580927726]
        : [4, 1.5, 1.25, 1.118033988749895];
      expectNumericArrayClose(await rig.readFloat(output.globalStatistics), moments, {
        absoluteTolerance: 1e-6
      });
      expectNumericArrayClose(await rig.readFloat(explicitOutput.globalStatistics), moments, {
        absoluteTolerance: 1e-6
      });
      expectNumericArrayClose(await rig.readUint(output.includedCount), [unknown ? 3 : 4]);
      const scores = unknown
        ? [NaN, -1.224744871391589, 1.224744871391589, 0]
        : [-1.341640786499874, -0.447213595499958, 1.341640786499874, 0.447213595499958];
      expectNumericArrayClose(await rig.readFloat(output.giZScores), scores, {
        absoluteTolerance: 1e-6
      });
      expectNumericArrayClose(await rig.readUint(output.category), [0, 0]);
      expectNumericArrayClose(await rig.readFloat(output.trendZ), unknown ? [NaN, 0] : [0, 0]);
      if (unknown)
        expectNumericArrayClose(await rig.readFloat(gathered as typeof output.giZScores), [
          NaN,
          1,
          3,
          2
        ]);
      else
        expectNumericArrayClose(
          await rig.readUint(gathered as typeof output.category),
          [0, 1, 3, 2]
        );
    } finally {
      rig.destroy();
    }
  });

it('CMP-12 maps events to a fixed dictionary in weights order before making the cell-major cube', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    // Caller identities [20,10] deliberately differ from numeric key sort order.
    const rows = rig.output('uint32', 3);
    const join = new GPUKeyLookup({
      leftKeys: rig.input(Uint32Array.of(10, 20, 20), 'uint32'),
      rightKeys: rig.input(Uint32Array.of(20, 10), 'uint32'),
      output: {rightRows: rows}
    });
    const counts = rig.output('uint32', 4);
    const cube = new GPUValueCube({
      cellCount: 2,
      periodCount: 2,
      cells: rows,
      periods: rig.input(Uint32Array.of(0, 1, 1), 'uint32'),
      values: rig.input(Float32Array.of(1, 1, 1), 'float32'),
      output: {counts}
    });
    const output = createCompositionOutputs(rig);
    rig.run(
      join,
      cube,
      new GPUEmergingHotSpots({
        values: counts,
        weights: rig.weights({offsets: [0, 0, 0], neighbors: [], weights: []}),
        sliceCount: 2,
        multipleTesting: 'none',
        parameters: rig.input(Float32Array.of(0, 0, 1.96, 0.05, 0.9, 2, 0.05, 0), 'float32'),
        output
      })
    );
    expectNumericArrayClose(await rig.readUint(rows), [1, 0, 0]);
    expectNumericArrayClose(await rig.readUint(counts), [0, 2, 1, 0]);
    // Population [0,2,1,0]: mean=3/4, variance=(0+4+1+0)/4 - (3/4)^2=11/16.
    expectNumericArrayClose(
      await rig.readFloat(output.globalStatistics),
      [4, 0.75, 0.6875, 0.82915619758885],
      {absoluteTolerance: 1e-6}
    );
    expectNumericArrayClose(
      await rig.readFloat(output.giZScores),
      [-0.904534033733291, 1.507556722888818, 0.301511344577764, -0.904534033733291],
      {absoluteTolerance: 1e-6}
    );
    expectNumericArrayClose(await rig.readUint(output.category), [0, 0]);
    expectNumericArrayClose(await rig.readUint(output.includedCount), [4]);
  } finally {
    rig.destroy();
  }
});
