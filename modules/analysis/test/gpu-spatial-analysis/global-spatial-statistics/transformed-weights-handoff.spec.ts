// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUGlobalSpatialStatistics} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics';
import {GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/global-spatial-statistics-layout';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32} from '../../utils/gpu-contributor-test-utils';

it('UST-02 in-place row transformation publishes the metadata used by masked Moran', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    // Undirected four-row chain. After masking row 3, included deviations are (-4,-1,5)/3.
    // Original handle: S0=5/2, sum(w_ij*c_i*c_j)=1/9, sum(c_i^2)=14/3 => I=1/35.
    // Transformed handle re-standardizes surviving rows: S0=3, numerator=-1/6 => I=-1/28.
    const original = {
      ...rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      }),
      information: {rowCount: 4, transform: 'binary' as const, symmetric: true as const}
    };
    const transform = new GPUSpatialWeightsTransform({operation: 'row', weights: original});
    expect(transform.outputWeights.weights).toBe(original.weights);
    expect(original.information.transform).toBe('binary');
    expect(transform.outputWeights.information?.transform).toBe('row');
    const values = rig.input(Float32Array.of(1, 2, 4, 8), 'float32', 4);
    const mask = rig.input(Uint32Array.of(1, 1, 1, 0), 'uint32', 4);
    const summaries = [
      rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
      rig.output('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
    ];
    rig.run(
      transform,
      ...[original, transform.outputWeights].map(
        (weights, index) =>
          new GPUGlobalSpatialStatistics({
            id: `handoff-${index}`,
            mode: 'reference',
            weights,
            values,
            mask,
            statistics: ['moran'],
            output: {summary: summaries[index].view}
          })
      )
    );
    for (const [index, expected] of [0.02857142857142857, -0.03571428571428571].entries()) {
      const actual = (
        await readFloat32(summaries[index].buffer, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
      )[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran];
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeCloseTo(expected, 6);
    }
  } finally {
    rig.destroy();
  }
});

for (const kernel of ['quartic', 'bisquare'] as const) {
  it(`NAM-16 ${kernel} writes the spatial-weights normalization`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const weights = rig.uploadWeights({
        offsets: [0, 1, 1],
        neighbors: [1],
        weights: [0],
        distances: [0.5]
      });
      const output = rig.output('float32', 1);
      rig.run(
        new GPUSpatialWeightsTransform({
          operation: 'kernel',
          weights,
          output: output.view,
          kernel,
          bandwidth: 1
        })
      );
      const [actual] = await readFloat32(output.buffer, 1);
      // (15/16) * (1 - 1/4)^2 = 135/256, exactly representable in float32.
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBe(0.52734375);
    } finally {
      rig.destroy();
    }
  });
}
