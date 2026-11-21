// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUTrajectoryHampel,
  getGPUTrajectoryHampelParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-hampel';
import {
  GPUTrajectoryThinning,
  getGPUTrajectoryThinningParameterValues
} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it('TRJ-108 Hampel uses per-track medians and MADs, preserving endpoints and NaN', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'hampel-track-seam');
  try {
    const filtered = fixture.output('filtered', 'float32', 12);
    const flags = fixture.output('flags', 'uint32', 12);
    const count = fixture.output('count', 'uint32', 1);
    fixture.graph.add(
      new GPUTrajectoryHampel({
        values: fixture.input(
          'values',
          Float32Array.of(0, 0, 100, 0, 0, 100, 10, 10, 100, 10, 10, NaN),
          'float32',
          12
        ),
        trackOffsets: fixture.input('offsets', Uint32Array.of(0, 6, 12), 'uint32', 3),
        windowSize: 3,
        parameters: fixture.input(
          'parameters',
          getGPUTrajectoryHampelParameterValues({numberOfSigmas: 3}),
          'float32',
          4
        ),
        output: {filtered: filtered.view, flags: flags.view, outlierCount: count.view}
      })
    );
    fixture.run();
    // Interior peaks: medians 0 and 10, MAD=0, hence replacement. Other interiors
    // equal their median. Endpoint row 5 stays 100; a seam-crossing [0,100,10]
    // window would have median=10, MAD=10 and replace it (90 > 3*1.4826*10).
    expect(await fixture.readFloat32(filtered, 12)).toEqual([
      0,
      0,
      0,
      0,
      0,
      100,
      10,
      10,
      10,
      10,
      10,
      NaN
    ]);
    expect(await fixture.readUint32(flags, 12)).toEqual([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
    expect(await fixture.readUint32(count, 1)).toEqual([2]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  undefined,
  6000
])('TRJ-108 thinning search cap %s reports required loss evidence', async maximumSearchRows => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `thinning-default-${maximumSearchRows}`);
  try {
    const positions = new Float32Array(12000);
    positions[11998] = 10;
    const rows = fixture.output('rows', 'uint32', 4);
    const count = fixture.output('count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const candidateOverflow = fixture.output('candidateOverflow', 'uint32', 1);
    fixture.graph.add(
      new GPUTrajectoryThinning({
        positions: fixture.input('positions', positions, 'float32x2', 6000),
        timestamps: fixture.input(
          'times',
          Float32Array.from({length: 6000}, (_, row) => row),
          'float32',
          6000
        ),
        trackOffsets: fixture.input('offsets', Uint32Array.of(0, 6000), 'uint32', 2),
        parameters: fixture.input(
          'parameters',
          getGPUTrajectoryThinningParameterValues({tolerance: 5}),
          'float32',
          4
        ),
        mode: 'minimum-distance',
        maximumSearchRows,
        output: {
          sourceRows: rows.view,
          trackOffsets: fixture.output('output-offsets', 'uint32', 2).view,
          status: {
            count: count.view,
            overflow: overflow.view,
            candidateOverflow: candidateOverflow.view
          }
        }
      })
    );
    fixture.run();
    // Bounded GPU contract: default 4096 checks rows 1..4096, then advances to
    // 4097 when no point reaches distance 5. Exact full-search contract: the only
    // first qualifying fix is row 5999, ten units away. Both preserve endpoints.
    const expected = maximumSearchRows === undefined ? [0, 4097, 5999] : [0, 5999];
    expect(await fixture.readUint32(count, 1)).toEqual([expected.length]);
    expect((await fixture.readUint32(rows, 4)).slice(0, expected.length)).toEqual(expected);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(candidateOverflow, 1)).toEqual([
      maximumSearchRows === undefined ? 1 : 0
    ]);
  } finally {
    fixture.destroy();
  }
});
