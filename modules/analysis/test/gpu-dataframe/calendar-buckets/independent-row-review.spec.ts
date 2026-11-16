// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createReviewScenario, REVIEW_CASES} from './independent-row-review.test-utils';

for (const kind of REVIEW_CASES) {
  it(`${kind} composes two contributor IDs in one graph and preserves independent results on replay`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scenario = createReviewScenario(device, kind, [0, 513, 0, 1, 257, 0]);
    scenario.graph.add(scenario.createContributor('first-contributor'));
    scenario.graph.add(scenario.createContributor('second-contributor'));
    const compiled = scenario.graph.compile();
    try {
      for (let encoding = 0; encoding < 3; encoding++) {
        scenario.updateInputs(encoding);
        submitGraph(device, compiled, undefined);
        await scenario.check(encoding);
      }
    } finally {
      compiled.destroy();
      scenario.destroy();
    }
  });

  for (const separateOutputBuffers of [false, true]) {
    it(`${kind} validates guarded ${separateOutputBuffers ? 'separate' : 'shared'} output buffers, empty chunks and mask replay at workgroup boundaries`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const lengths of [
        [255, 256, 257],
        [513, 1, 257],
        [0, 513, 0, 1, 257, 0]
      ]) {
        const scenario = createReviewScenario(device, kind, lengths, separateOutputBuffers);
        scenario.graph.add(scenario.contributor);
        const compiled = scenario.graph.compile();
        try {
          for (let encoding = 0; encoding < 3; encoding++) {
            scenario.updateInputs(encoding);
            submitGraph(device, compiled, undefined);
            await scenario.check(encoding);
          }
        } finally {
          compiled.destroy();
          scenario.destroy();
        }
      }
    });
  }
  it(`${kind} preserves exact packed parity with three uneven batches and independent expected values`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const results: Record<string, number[]>[] = [];
    for (const lengths of [[771], [513, 1, 257]]) {
      const scenario = createReviewScenario(device, kind, lengths);
      scenario.graph.add(scenario.contributor);
      const compiled = scenario.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        results.push(await scenario.check(0));
      } finally {
        compiled.destroy();
        scenario.destroy();
      }
    }
    expect(results[1]).toEqual(results[0]);
  });
}
