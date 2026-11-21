// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge/gpu-line-merge';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it('line merge clears poisoned overflow for a hand-derived single chain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const count = fixture.output('uint32', 1);
  const overflow = fixture.output('uint32', 1);
  // Two end-to-start segments form one three-vertex chain. Construction requires
  // room for every input vertex and line: overflow is always zero on accepted inputs.
  fixture.graph.add(
    new GPULineMerge({
      positions: fixture.input('float32x2', [0, 0, 1, 0, 1, 0, 2, 0]),
      lineOffsets: fixture.input('uint32', [0, 2, 4]),
      output: {
        positions: fixture.output('float32x2', 4).view,
        chainOffsets: fixture.output('uint32', 3).view,
        status: {count: count.view, overflow: overflow.view}
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      count.poison();
      overflow.poison();
      submitGraph(device, compiled, undefined);
      expect(await count.read()).toEqual([1]);
      expect(await overflow.read()).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
