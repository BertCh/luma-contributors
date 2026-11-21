// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from './line-output-oracle-harness';
import {createSharedUncertaintyFixture} from './line-clip-output-oracle-fixture';
import {createSharedTruncationFixture} from './line-clip-output-oracle-fixture';

it('shared paths requires six vertices for two three-vertex runs even when only one fits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of ['required', 'uncertain', 'absent'] as const) {
    const harness = createLineOracleHarness(device);
    const {graph, view, read} = harness;
    const geometry = () => ({
      kind: 'lines' as const,
      positions: view('float32x2', 6, [0, 0, 2, 0, 4, 0, 0, 3, 2, 3, 4, 3]),
      lineOffsets: view('uint32', 3, [0, 3, 6])
    });
    const positions = view('float32x2', 3);
    const offsets = view('uint32', 2);
    const leftSourceIds = view('uint32', 1);
    const rightSourceIds = view('uint32', 1);
    const forward = view('uint32', 1);
    const count = view('uint32', 1);
    const requiredVertexCount = optional === 'required' ? view('uint32', 1) : undefined;
    const uncertainCount = optional === 'uncertain' ? view('uint32', 1) : undefined;
    graph.add(
      new GPUSharedPaths({
        left: geometry(),
        right: geometry(),
        intersectionCapacity: 16,
        output: {
          runs: {positions, offsets, leftSourceIds, rightSourceIds, forward},
          status: {count, requiredVertexCount, uncertainCount}
        }
      })
    );
    const destroy = await harness.run();
    try {
      // Two disjoint parallel lines each share both edges. Run 0 fits; run 1 does not.
      expect(await read(count)).toEqual([1]);
      expect(await read(offsets)).toEqual([0, 3]);
      expect(await read(positions)).toEqual([0, 0, 2, 0, 4, 0]);
      expect(await read(leftSourceIds)).toEqual([0]);
      expect(await read(rightSourceIds)).toEqual([0]);
      expect(await read(forward)).toEqual([1]);
      if (requiredVertexCount) expect(await read(requiredVertexCount)).toEqual([6]);
      if (uncertainCount) expect(await read(uncertainCount)).toEqual([0]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});

it('shared paths keeps overflow and uncertainty in separate packed words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createSharedTruncationFixture(harness.graph, harness.view);
  harness.graph.add(fixture.contributor);
  const destroy = await harness.run();
  try {
    // Two disjoint unit runs need two slots. One fits, with no numerical uncertainty.
    expect(await harness.read(fixture.count)).toEqual([1]);
    expect(await harness.read(fixture.requiredCount)).toEqual([2]);
    expect(await harness.read(fixture.packed)).toEqual([1, 0]);
  } finally {
    destroy();
    harness.destroy();
  }
});

it('shared paths counts a segment pair outside the exact predicate exponent range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const includeUncertain of [true, false]) {
    const harness = createLineOracleHarness(device);
    const fixture = createSharedUncertaintyFixture(harness.view, includeUncertain);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      expect(await harness.read(fixture.count)).toEqual([0]);
      expect(await harness.read(fixture.overflow)).toEqual([1]);
      if (fixture.uncertainCount) expect(await harness.read(fixture.uncertainCount)).toEqual([1]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});
