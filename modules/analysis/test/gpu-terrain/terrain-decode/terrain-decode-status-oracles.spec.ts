// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPUTerrainSpikeRepair} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';

it('spike repair has a mandatory changed first round on one edge and clears on an edgeless replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  const elevation = fixture.input('float32', [0, 0, NaN]);
  const validity = fixture.output('uint32', 3);
  const repaired = fixture.output('float32', 3);
  const labels = fixture.output('uint32', 3);
  const converged = fixture.output('uint32', 1);
  const limit = fixture.output('uint32', 1);
  for (const output of [validity, converged, limit])
    output.buffer.write(new Uint32Array(output.view.length).fill(17));
  const contributor = new GPUTerrainSpikeRepair({
    grid: makeGPURasterMetadata({width: 3, height: 1}),
    maximumComponentIterations: 1,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: elevation.view}
    },
    output: {
      values: repaired.view,
      outputValidity: validity.view,
      labels: labels.view,
      status: {converged: converged.view, iterationLimitReached: limit.view}
    }
  });
  fixture.graph.add(contributor);
  const compiled = fixture.graph.compile();
  try {
    // Initial component labels are cell IDs 0 and 1. Their sole edge must lower label 1
    // to 0 in round one, regardless of scheduling; no unchanged round was executed.
    // A 1000m difference exceeds jump=200, removes that edge and makes initial labels
    // already stable. The NaN third cell is always invalid and never unresolved.
    for (const connected of [true, false, true]) {
      elevation.buffer.write(new Float32Array([0, connected ? 0 : 1000, NaN]));
      fixture.submit(compiled);
      expect(await converged.read()).toEqual(connected ? [0] : [1]);
      expect(await limit.read()).toEqual(connected ? [1] : [0]);
      expect(await validity.read()).toEqual(connected ? [0, 0, 0] : [1, 1, 0]);
      expect(await repaired.read()).toEqual(connected ? [Infinity, Infinity, NaN] : [0, 1000, NaN]);
      expect(await labels.read()).toEqual(connected ? [0xffffffff, 0xffffffff, 2] : [0, 1, 2]);
    }
  } finally {
    compiled.destroy();
    contributor.destroy();
    fixture.destroy();
  }
});
