// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineSplit} from '../../../src/gpu-spatial-analysis/line-split/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';
import {createSplitOracleFixture} from './line-split-output-oracle-fixture';

it('split pins exact X edge rows and bow-tie self-contact bits with isolated optional outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const selfContact of [false, true]) {
    for (const optional of ['edges', 'uncertain', 'absent'] as const) {
      const harness = createLineOracleHarness(device);
      const fixture = createSplitOracleFixture(harness.view, selfContact, optional);
      harness.graph.add(fixture.contributor);
      const destroy = await harness.run();
      try {
        expect(await harness.read(fixture.count)).toEqual([selfContact ? 3 : 4]);
        expect(await harness.read(fixture.overflow)).toEqual([0]);
        expect(await harness.read(fixture.lineOffsets)).toEqual(
          selfContact ? [0, 2, 6, 8, 8] : [0, 2, 4, 6, 8]
        );
        expect(await harness.read(fixture.sourceIds)).toEqual(
          selfContact ? [0, 0, 0, 0xffffffff] : [0, 0, 1, 1]
        );
        expect(await harness.read(fixture.positions)).toEqual([
          0, 0, 1, 1, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1, 2, 0
        ]);
        if (fixture.uncertainCount) expect(await harness.read(fixture.uncertainCount)).toEqual([0]);
        if (fixture.vertexSourceEdges) {
          const edges = await harness.read(fixture.vertexSourceEdges);
          // Terminal vertices are unspecified. Nonterminal rows identify outgoing edges.
          expect((selfContact ? [0, 2, 3, 4, 6] : [0, 2, 4, 6]).map(row => edges[row])).toEqual(
            selfContact ? [2147483648, 2147483648, 1, 2147483650, 2147483650] : [0, 0, 2, 2]
          );
        }
      } finally {
        destroy();
        harness.destroy();
      }
    }
  }
});

it('split counts one uncertifiable pair and propagates incomplete output when the scalar is absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const includeUncertain of [true, false]) {
    const harness = createLineOracleHarness(device);
    const {graph, view, read} = harness;
    const count = view('uint32', 1);
    const overflow = view('uint32', 1);
    const uncertainCount = includeUncertain ? view('uint32', 1) : undefined;
    // Exactly one candidate pair, with orientation terms spanning about 2^226,
    // outside the bounded exact-predicate accumulator. Neither line is split.
    graph.add(
      new GPULineSplit({
        lines: {
          kind: 'lines',
          positions: view(
            'float32x2',
            4,
            [1e-30, 1e-30, 2e-30, 2e-30, 1.5e-30, 1.5e-30, 1e38, 1e38]
          ),
          lineOffsets: view('uint32', 3, [0, 2, 4])
        },
        intersectionCapacity: 4,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: view('float32x2', 4),
              lineOffsets: view('uint32', 3)
            },
            sourceIds: view('uint32', 2),
            status: {count, overflow}
          },
          status: {uncertainCount}
        }
      })
    );
    const destroy = await harness.run();
    try {
      expect(await read(count)).toEqual([2]);
      expect(await read(overflow)).toEqual([1]);
      if (uncertainCount) expect(await read(uncertainCount)).toEqual([1]);
    } finally {
      destroy();
      harness.destroy();
    }
  }
});
