// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDotDensity,
  GPURandomPointsInPolygon,
  GPURandomPointsOnLine,
  getGPUDotDensityParameterValues
} from '../../../src/gpu-spatial-analysis/dot-density/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// One nondegenerate triangle requests two slots. A unit mask accepts every
// draw; a zero mask rejects every attempt. Rejected dots retain their slots,
// so count/required remain 2 and only failedCount changes from 0 to 2.
for (const route of ['values', 'counts'] as const) {
  it(`polygon sampling statuses clear poisoned words and count certain failures (${route})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const mask = input('mask', 'float32', Float32Array.of(0));
    const failedCount = output('failed', 'uint32', 1),
      slotCounts = output('slot-counts', 'uint32', 1);
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      candidateOverflow: output('candidate', 'uint32', 1)
    };
    const common = {
      positions: input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0, 0, 1)),
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 3)),
      parameters: input(
        'parameters',
        'uint32',
        getGPUDotDensityParameterValues({
          seed: 7,
          maskPlacement: {origin: [-1, -1], cellSize: [2, 2]}
        })
      ),
      mask: {weights: mask, width: 1, height: 1},
      maximumAttempts: 1,
      output: {
        positions: output('points', 'float32x2', 4),
        dots: {ids: output('ids', 'uint32', 4), status},
        slotCounts,
        status: {failedCount}
      }
    };
    graph.add(
      route === 'values'
        ? new GPUDotDensity({...common, values: input('values', 'float32', Float32Array.of(2))})
        : new GPURandomPointsInPolygon({
            ...common,
            counts: input('counts', 'uint32', Uint32Array.of(2))
          })
    );
    const compiled = graph.compile();
    try {
      for (const weight of [0, 1, 0, 1]) {
        fixture.writeInput(mask, Float32Array.of(weight));
        const actual = await fixture.submitTwice(compiled, () =>
          fixture.readRows([
            status.count,
            status.requiredCount,
            status.overflow,
            status.candidateOverflow,
            failedCount,
            slotCounts
          ])
        );
        expect(actual).toEqual(weight === 0 ? [2, 2, 0, 0, 2, 2] : [2, 2, 0, 0, 0, 2]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

// One nonzero two-vertex line has no geometric rejection. The documented
// per-line bound is min(capacity + 1, floor(0xffffffff / 1), 2^31): 2 or 5 here.
// At capacity 1, requests 2 and 3 both retain 2, but only 3 loses candidates.
// Final overflow and candidate loss are separate full words, independent of draws.
for (const capacity of [1, 4]) {
  it(`line sampling status follows requested cardinality on poisoned replay (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const counts = input('counts', 'uint32', Uint32Array.of(3));
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      candidateOverflow: output('candidate', 'uint32', 1)
    };
    graph.add(
      new GPURandomPointsOnLine({
        positions: input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0)),
        lineOffsets: input('lines', 'uint32', Uint32Array.of(0, 2)),
        counts,
        parameters: input('parameters', 'uint32', getGPUDotDensityParameterValues({seed: 7})),
        output: {
          positions: output('points', 'float32x2', capacity),
          points: {ids: output('ids', 'uint32', capacity), status}
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const requested of [3, 0, 2, 1, 3]) {
        fixture.writeInput(counts, Uint32Array.of(requested));
        expect(
          await fixture.submitTwice(compiled, () =>
            fixture.readRows([
              status.count,
              status.requiredCount,
              status.overflow,
              status.candidateOverflow
            ])
          )
        ).toEqual(
          requested === 0
            ? [0, 0, 0, 0]
            : requested === 1
              ? [1, 1, 0, 0]
              : capacity === 1
                ? requested === 2
                  ? [1, 2, 1, 0]
                  : [1, 2, 1, 1]
                : requested === 2
                  ? [2, 2, 0, 0]
                  : [3, 3, 0, 0]
        );
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
