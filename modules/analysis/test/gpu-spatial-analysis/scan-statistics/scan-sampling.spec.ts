// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {test} from 'vitest';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runWGSLModuleKernel} from '../../utils/wgsl-module-harness';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  SCAN_SAMPLING_BOUNDS,
  SAMPLING_DECLARATIONS,
  SAMPLING_BODY,
  createConditionalSamplingScene,
  expectConditionalLaw,
  expectPoissonLaw
} from './scan-sampling-fixtures';
import {expect} from 'vitest';

test('non-positive and non-finite baseline cells draw no replicate cases', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const draws = await runWGSLModuleKernel(device, {
    id: 'scan-excluded-baselines',
    inputs: {
      means: Float32Array.of(
        Number.NEGATIVE_INFINITY,
        Number.NaN,
        Number.POSITIVE_INFINITY,
        -1,
        -0,
        0,
        -3e38,
        -1e-30
      )
    },
    outputLength: 8,
    invocationCount: 8,
    declarations: SAMPLING_DECLARATIONS,
    body: `var stream=createPhiloxStream(vec2u(17u,0u),0u,index,0x45424353u); output[outputOffset+index]=drawPhiloxPoisson(&stream,means[meansOffset+index]);`
  });
  expect(Array.from(draws)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
});

test('Poisson draws satisfy independent SciPy bounds across all sampler routes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = SCAN_SAMPLING_BOUNDS.sampleCount * SCAN_SAMPLING_BOUNDS.cases.length;
  const draws = await runWGSLModuleKernel(device, {
    id: 'scan-sampling-law',
    inputs: {means: Float32Array.from(SCAN_SAMPLING_BOUNDS.cases, row => row.mean)},
    outputLength: count,
    invocationCount: count,
    declarations: SAMPLING_DECLARATIONS,
    body: SAMPLING_BODY
  });
  expectPoissonLaw(draws);
});

for (const variant of [
  'poisson',
  'bernoulli',
  'bernoulli-complement',
  'space-time-permutation'
] as const) {
  const model = variant === 'bernoulli-complement' ? 'bernoulli' : variant;
  for (const parallelDraws of model === 'poisson' ? [false] : [false, true]) {
    test(`conditional ${variant} ${parallelDraws ? 'parallel' : 'serial'} draws follow the exact enumerated null distribution`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      try {
        const scene = createConditionalSamplingScene(variant);
        const statistics = rig.output('float32', scene.permutationCount + 1);
        rig.run(
          new GPUSpatialScanStatistic({
            model,
            parallelDraws,
            positions: rig.input(scene.positions, 'float32x2', 8),
            cases: rig.input(scene.cases, 'uint32'),
            ...(scene.baseline ? {baseline: rig.input(scene.baseline, 'float32')} : {}),
            ...(scene.controls ? {controls: rig.input(scene.controls, 'uint32')} : {}),
            timeBuckets: scene.timeBuckets,
            maximumWindowZones: 1,
            maximumPermutations: scene.permutationCount,
            maximumClusters: 1,
            parameters: rig.input(getGPUSpatialScanStatisticParameterValues(scene), 'uint32'),
            output: {
              statistics,
              clusterIndices: rig.output('uint32', 4),
              clusterStatistics: rig.output('float32', 8),
              summary: rig.output('uint32', 4)
            }
          })
        );
        expectConditionalLaw(variant, (await rig.readFloat(statistics)).slice(1));
      } finally {
        rig.destroy();
      }
    });
  }
}
