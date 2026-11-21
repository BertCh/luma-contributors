// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const family of ['quadbin', 'h3', 's2'] as const) {
  for (const empty of [false, true]) {
    it(`preserves diagnostic padding after ${empty ? 'empty' : 'clean'} cover (${family})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const invalidCount = input('invalid', 'uint32', Uint32Array.of(99, 0x12345678));
      const uncertainCount = input('uncertain', 'uint32', Uint32Array.of(99, 0x12345678));
      graph.add(
        new GPUCellCover({
          family,
          resolution: 2,
          candidateCapacity: 128,
          candidateEnumeration: 'bounds',
          edgeSlabs: false,
          positions: input(
            'positions',
            'float32x2',
            Float32Array.of(-80, 10, -10, 10, -10, 50, -80, 50)
          ),
          featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
          polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
          ringOffsets: input('rings', 'uint32', Uint32Array.of(0, empty ? 0 : 4)),
          output: {
            cells: output('cells', 'uint32x2', 32),
            featureIds: output('featureIds', 'uint32', 32),
            status: {
              count: output('count', 'uint32', 1),
              overflow: output('overflow', 'uint32', 1),
              candidateOverflow: output('candidate', 'uint32', 1),
              invalidCount,
              uncertainCount
            }
          }
        })
      );
      const compiled = graph.compile();
      try {
        // A planar rectangle within the ordinary latitude range, or an empty ring,
        // has zero invalid/polar diagnostics. The second word is caller padding.
        expect(
          await fixture.submitTwice(compiled, () =>
            fixture.readRows([invalidCount, uncertainCount])
          )
        ).toEqual([0, 0x12345678, 0, 0x12345678]);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
