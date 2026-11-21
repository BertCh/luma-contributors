// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';
import {
  createMergeOracleFixture,
  createMergeReuseFixture,
  MERGE_OPTIONALS
} from './line-merge-output-oracle-fixture';

it('merge publishes fixed chain counts, leaders and member orders with isolated optional outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const optional of MERGE_OPTIONALS) {
    const harness = createLineOracleHarness(device);
    const fixture = createMergeOracleFixture(harness.view, optional);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      expect(await harness.read(fixture.count)).toEqual([2]);
      expect(await harness.read(fixture.chainOffsets)).toEqual([0, 3, 5, 5, 5, 5]);
      expect(await harness.read(fixture.positions, 5)).toEqual([0, 0, 1, 0, 2, 0, 7, 7, 8, 8]);
      if (fixture.scalar)
        expect(await harness.read(fixture.scalar)).toEqual([optional === 'requiredCount' ? 2 : 5]);
      if (fixture.sourceIds)
        expect(await harness.read(fixture.sourceIds)).toEqual([0, 3, 0xffffffff, 0xffffffff]);
      if (fixture.lineOrders)
        expect(await harness.read(fixture.lineOrders)).toEqual([0, 1, 0xffffffff, 0]);
      if (fixture.lineReversed)
        expect(await harness.read(fixture.lineReversed)).toEqual([0, 1, 0, 0]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});

it('merge preserves captured reversal output when caller mutates the descriptor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createMergeReuseFixture(harness.view);
  fixture.props.output.lineReversed = harness.graph.createDataView(fixture.inputOffsets.buffer, {
    format: 'uint32',
    length: 2
  });
  harness.graph.add(fixture.contributor);
  const destroy = await harness.run();
  try {
    expect(await harness.read(fixture.inputOffsets)).toEqual([0, 2, 4]);
    expect(await harness.read(fixture.lineReversed)).toEqual([0, 1]);
  } finally {
    destroy();
    harness.destroy();
  }
});

it('merge clears a reused reversal flag and writes every spare offset row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createMergeReuseFixture(harness.view);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    // Joined forward/reversed edges form (0,0)-(1,0)-(2,0): three vertices.
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.lineReversed)).toEqual([0, 1]);
    expect(await harness.read(fixture.chainOffsets)).toEqual([0, 3, 3, 3]);
    harness.write(fixture.inputOffsets, Uint32Array.of(0, 2, 3));
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.lineReversed)).toEqual([0, 0]);
    expect(await harness.read(fixture.chainOffsets)).toEqual([0, 2, 2, 2]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
