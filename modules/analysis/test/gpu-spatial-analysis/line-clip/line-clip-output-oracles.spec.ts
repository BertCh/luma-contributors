// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from './line-output-oracle-harness';
import {
  createClipCapacityFixture,
  createClipPaddingFixture,
  createClipUncertaintyFixture
} from './line-clip-output-oracle-fixture';

it('clip publishes hand-derived edge rows and six required vertices independently of optional ports', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [4, 6]) {
    for (const optional of [
      'edges',
      'required',
      'vertices',
      'counts',
      'parts',
      'absent'
    ] as const) {
      const harness = createLineOracleHarness(device);
      const fixture = createClipCapacityFixture(harness.view, capacity, optional);
      harness.graph.add(fixture.contributor);
      const destroy = await harness.run();
      try {
        // Each line needs three vertices. With four slots, only the first complete
        // piece fits: written=3, required=6. Integer intersections are exact in f32.
        const truncated = capacity === 4;
        expect(await harness.read(fixture.count)).toEqual([truncated ? 1 : 2]);
        expect(await harness.read(fixture.overflow)).toEqual([truncated ? 1 : 0]);
        // Caller identities are separate from original part rows. Both IDs differ
        // from row indices and poison 93, so neither an identity swap nor an
        // unwritten second identity can pass.
        expect(await harness.read(fixture.sourceIds)).toEqual(
          truncated ? [71, 0xffffffff] : [71, 159]
        );
        if (fixture.sourcePartRows) {
          expect(await harness.read(fixture.sourcePartRows)).toEqual(
            truncated ? [0, 0xffffffff] : [0, 1]
          );
        }
        expect(await harness.read(fixture.lineOffsets)).toEqual(truncated ? [0, 3, 3] : [0, 3, 6]);
        expect(await harness.read(fixture.positions, truncated ? 3 : 6)).toEqual(
          truncated ? [0, 1, 2, 1, 4, 1] : [0, 1, 2, 1, 4, 1, 1, 2, 2, 2, 3, 2]
        );
        if (fixture.vertexCount)
          expect(await harness.read(fixture.vertexCount)).toEqual([truncated ? 3 : 6]);
        if (fixture.requiredVertexCount)
          expect(await harness.read(fixture.requiredVertexCount)).toEqual([6]);
        if (fixture.vertexSourceEdges) {
          // Clip defines incoming source edges at terminal vertices and NONE in
          // unused rows, a stronger promise than the general split-pieces layout.
          expect(await harness.read(fixture.vertexSourceEdges)).toEqual(
            truncated ? [0, 1, 1, 0xffffffff] : [0, 1, 1, 3, 4, 4]
          );
        }
      } finally {
        destroy();
        harness.destroy();
      }
    }
  }
});

it('clip ignores unused midpoints inside an extreme polygon bounding box', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const intersectionCapacity of [1, 2]) {
    const harness = createLineOracleHarness(device);
    const fixture = createClipPaddingFixture(harness.view, intersectionCapacity);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      // Bounding boxes are disjoint. Neither live segment predicates nor the
      // capacity padding can contribute an uncertainty or incomplete result.
      expect(await harness.read(fixture.count)).toEqual([0]);
      expect(await harness.read(fixture.overflow)).toEqual([0]);
      expect(await harness.read(fixture.uncertainCount)).toEqual([0]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});

it('clip preserves incomplete status for an uncertifiable empty result with uncertainty present or absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const includeUncertain of [true, false]) {
    const harness = createLineOracleHarness(device);
    const fixture = createClipUncertaintyFixture(harness.view, includeUncertain);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      expect(await harness.read(fixture.count)).toEqual([0]);
      expect(await harness.read(fixture.overflow)).toEqual([1]);
      if (fixture.uncertainCount) expect(await harness.read(fixture.uncertainCount)).toEqual([2]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});
