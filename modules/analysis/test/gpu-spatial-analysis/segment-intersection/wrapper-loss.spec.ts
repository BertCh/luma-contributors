// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createCapacityBoundaryFixture,
  createLateCrossingFixture,
  createLateSharedFixture,
  createStatusOnlyFixture,
  createWrapperLossFixture,
  LOSS_WRAPPERS
} from './wrapper-loss-fixture';

it.each([4, 16])('GPU late crossing remains observable with %s leaves', async leafCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createLateCrossingFixture(leafCapacity, harness.view);
  harness.graph.add([fixture.predicates, fixture.split]);
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.intersectionCount)).toEqual([leafCapacity === 4 ? 0 : 1]);
    expect(await harness.read(fixture.predicateOverflow)).toEqual([leafCapacity === 4 ? 1 : 0]);
    expect(await harness.read(fixture.simple)).toEqual([0]);
    expect(await harness.read(fixture.pieceCount)).toEqual([leafCapacity === 4 ? 1 : 3]);
    expect(await harness.read(fixture.pieceOverflow)).toEqual([0]);
    expect(await harness.read(fixture.candidateOverflow)).toEqual([leafCapacity === 4 ? 1 : 0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
it.each([4, 8])('GPU late shared segment remains observable with %s leaves', async leafCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createLateSharedFixture(leafCapacity, harness.view);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.count)).toEqual([leafCapacity === 4 ? 0 : 1]);
    expect(await harness.read(fixture.overflow)).toEqual([leafCapacity === 4 ? 1 : 0]);
    expect(await harness.read(fixture.candidateOverflow)).toEqual([leafCapacity === 4 ? 1 : 0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each(LOSS_WRAPPERS)('%s forwards nested BVH loss with spare final capacity', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createWrapperLossFixture(kind, harness.view);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      harness.write(fixture.overflow, Uint32Array.of(99));
      harness.write(fixture.candidateOverflow, Uint32Array.of(99));
      submitGraph(device, compiled, undefined);
      expect(await harness.read(fixture.overflow)).toEqual([kind === 'split' ? 0 : 1]);
      if (kind === 'predicates') expect(await harness.read(fixture.isSimple)).toEqual([0]);
      if (kind !== 'predicates') {
        expect(await harness.read(fixture.candidateOverflow)).toEqual([1]);
        expect((await harness.read(fixture.requiredCount))[0]).toBeLessThan(64);
      }
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each([
  'count',
  'overflow',
  'both'
] as const)('status-only %s is written and reset on reuse', async request => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createStatusOnlyFixture(request, harness.view);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    for (const clean of [false, true]) {
      if (clean) harness.write(fixture.positions, fixture.cleanPositions);
      harness.write(fixture.intersectionCount, Uint32Array.of(99));
      harness.write(fixture.overflow, Uint32Array.of(99));
      submitGraph(device, compiled, undefined);
      if (request !== 'overflow')
        expect(await harness.read(fixture.intersectionCount)).toEqual([clean ? 0 : 1]);
      if (request !== 'count')
        expect(await harness.read(fixture.overflow)).toEqual([
          request === 'overflow' && !clean ? 1 : 0
        ]);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each([
  'split',
  'clip',
  'shared'
] as const)('%s retains one complete line at the vertex capacity boundary', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createCapacityBoundaryFixture(kind, harness.view);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.count)).toEqual([1]);
    expect(await harness.read(fixture.overflow)).toEqual([1]);
    expect(await harness.read(fixture.candidateOverflow)).toEqual([0]);
    expect(await harness.read(fixture.offsets)).toEqual([0, 2, 2]);
    expect(await harness.read(fixture.positions)).toEqual([0, 0, 1, 0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU complete predicate tree preserves a positive simplicity result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createWrapperLossFixture('predicates', harness.view, 8);
  harness.graph.add(fixture.contributor);
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await harness.read(fixture.isSimple)).toEqual([1]);
    expect(await harness.read(fixture.overflow)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
