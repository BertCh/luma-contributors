// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createSharedTruncationFixture,
  createSharedUncertaintyFixture
} from './line-clip-output-oracle-fixture';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it('GPU shared paths rejects co-bound adjacent scalars and resets separately written adjacent words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  try {
    const fixture = createSharedTruncationFixture(harness.graph, (format, length, values) =>
      harness.allocate('shared', format, length, values)
    );
    const count = harness.graph.createDataView(fixture.packed.buffer, {
      format: 'uint32',
      length: 1
    });
    const overflow = harness.graph.createDataView(fixture.packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUSharedPaths({
          ...fixture.contributor.props,
          output: {...fixture.contributor.props.output, status: {count, overflow}}
        })
    ).toThrow(/count.*overflow|overflow.*count/);
    const nodes = fixture.contributor.getCommandNodes(harness.graph);
    for (let replay = 0; replay < 2; replay++) {
      if (replay)
        harness.write(
          fixture.contributor.props.right.positions,
          Float32Array.of(0, 0, 1, 0, 0, 4, 1, 4)
        );
      await harness.run(nodes);
      expect(await harness.read(fixture.packed)).toEqual([replay === 0 ? 1 : 0, 0]);
      expect(await harness.read(fixture.count)).toEqual([1]);
    }
  } finally {
    harness.destroy();
  }
});
it('GPU shared paths retains nonzero uncertainty in aggregate overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  try {
    const fixture = createSharedUncertaintyFixture(
      (format, length, values) => harness.allocate('uncertain', format, length, values),
      true
    );
    await harness.run(fixture.contributor.getCommandNodes(harness.graph));
    expect(await harness.read(fixture.count)).toEqual([0]);
    expect(await harness.read(fixture.overflow)).toEqual([1]);
    expect(await harness.read(fixture.uncertainCount!)).toEqual([1]);
  } finally {
    harness.destroy();
  }
});
