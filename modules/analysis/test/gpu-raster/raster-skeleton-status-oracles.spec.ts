// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPURasterSkeleton} from '../../src/raster-analysis/gpu-raster-skeleton';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from './census-raster-oracle-utils';

for (const solid of [true, false]) {
  it(`skeleton stopping words have an independent pending-deletion witness (solid=${solid})`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const result = await harness.run(
      new GPURasterSkeleton({
        grid: {
          width: 9,
          height: 9,
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        input: harness.input(
          'uint32',
          Array.from({length: 81}, (_, cell) => Number(solid || cell === 40))
        ),
        maximumIterations: 1,
        output: {
          skeleton: harness.output('skeleton', 'uint32', 81),
          status: {
            converged: harness.output('converged', 'uint32', 1),
            iterationLimitReached: harness.output('limit', 'uint32', 1),
            iterationCount: harness.output('iterations', 'uint32', 1)
          }
        }
      })
    );
    // The adjacent set-based oracle gives 18,15,13,11 deletions across the four executed
    // Zhang-Suen phases. Cell (2,2) survives them and is deletable in the next phase:
    // work really remains, independently of the implementation's convergence word.
    // A singleton has zero neighbors, so neither phase can remove it; one pass verifies it.
    // The harness poisons every output word, including the expected zero stopping word.
    expectCensusValues(result['converged'], [solid ? 0 : 1]);
    expectCensusValues(result['limit'], [solid ? 1 : 0]);
    expectCensusValues(result['iterations'], [solid ? 2 : 1]);
    if (solid) {
      expectCensusValues([result['skeleton'][20]], [1]);
    } else {
      expectCensusValues(
        result['skeleton'],
        [
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
      );
    }
  });
}
