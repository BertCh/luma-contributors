// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('direct counted outline suppresses rings when source complete clears and recovers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const complete = input('source-complete', 'uint32', Uint32Array.of(1));
  const source = {
    cells: input('cells', 'uint32x2', Uint32Array.of(0xffffffff, 0x4823ffff)),
    status: {
      count: input('source-count', 'uint32', Uint32Array.of(1)),
      overflow: input('source-overflow', 'uint32', Uint32Array.of(0)),
      complete
    }
  };
  const status = {
    count: output('count', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    candidateOverflow: output('candidate', 'uint32', 1)
  };
  const ringCount = output('ring-count', 'uint32', 1);
  graph.add(
    new GPUCellSetOutline({
      family: 'quadbin',
      source,
      output: {
        rows: output('rows', 'uint32', 4),
        cells: output('out-cells', 'uint32x2', 4),
        edgeIndices: output('edges', 'uint32', 4),
        endpoints: output('endpoints', 'float32x4', 4),
        status
      },
      rings: {
        output: {
          positions: output('positions', 'float32x2', 5),
          ringOffsets: output('offsets', 'uint32', 2),
          status: {count: ringCount, overflow: output('ring-overflow', 'uint32', 1)}
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const certificate of [1, 0, 1]) {
      fixture.writeInput(complete, Uint32Array.of(certificate));
      const actual = await fixture.submitTwice(compiled, () =>
        fixture.readRows([status.count, status.overflow, status.candidateOverflow, ringCount])
      );
      // One tile has four edges and one ring. Missing source coverage invalidates the retained ring.
      expect(actual).toEqual([4, 0, 1 - certificate, certificate]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

for (const reportLoss of [false, true]) {
  for (const sourceCapacity of [1, 2]) {
    for (const capacity of [3, 8]) {
      it(`counted aggregation outlines suppress nested rings on either loss and recover (${sourceCapacity}, ${capacity}, report ${reportLoss})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = createOutputOracleFixture(device);
        const {graph, input, output} = fixture;
        const mask = input('mask', 'uint32', Uint32Array.of(1, 0));
        const source = {
          cells: output('table-cells', 'uint32x2', sourceCapacity),
          counts: output('table-counts', 'uint32', sourceCapacity),
          status: {
            count: output('table-count', 'uint32', 1),
            overflow: output('table-overflow', 'uint32', 1),
            candidateOverflow: output('table-candidate', 'uint32', 1)
          }
        };
        graph.add(
          new GPUCellAggregation({
            family: 'quadbin',
            resolution: 2,
            cells: input(
              'cells',
              'uint32x2',
              Uint32Array.of(0xffffffff, 0x4820ffff, 0xffffffff, 0x4821ffff)
            ),
            mask,
            output: source
          })
        );
        const status = {
          count: output('count', 'uint32', 1),
          requiredCount: output('required', 'uint32', 1),
          overflow: output('overflow', 'uint32', 1),
          candidateOverflow: reportLoss ? output('candidate', 'uint32', 1) : undefined
        };
        const ringStatus = {
          count: output('ring-count', 'uint32', 1),
          overflow: output('ring-overflow', 'uint32', 1),
          openSegmentCount: output('open', 'uint32', 1),
          touchingSegmentCount: output('touching', 'uint32', 1),
          candidateOverflow: output('ring-candidate', 'uint32', 1),
          complete: output('ring-complete', 'uint32', 1)
        };
        graph.add(
          new GPUCellSetOutline({
            family: 'quadbin',
            source,
            output: {
              rows: output('rows', 'uint32', capacity),
              cells: output('out-cells', 'uint32x2', capacity),
              edgeIndices: output('edges', 'uint32', capacity),
              endpoints: output('endpoints', 'float32x4', capacity),
              status
            },
            rings: {
              output: {
                ringOffsets: output('offsets', 'uint32', 3),
                positions: output('positions', 'float32x2', 9),
                status: ringStatus
              }
            }
          })
        );
        const compiled = graph.compile();
        try {
          for (const selected of [1, 2, 1]) {
            fixture.writeInput(mask, Uint32Array.of(1, selected === 2 ? 1 : 0));
            const actual = await fixture.submitTwice(compiled, async () => ({
              outline: await fixture.readRows([
                status.count,
                status.requiredCount,
                status.overflow,
                ...(status.candidateOverflow ? [status.candidateOverflow] : [])
              ]),
              rings: await fixture.readRows([
                ringStatus.count,
                ringStatus.overflow,
                ringStatus.candidateOverflow,
                ringStatus.complete
              ])
            }));
            // Two adjacent unit squares share one edge: 4 + 4 - 2 = 6 boundary
            // segments. Keeping only one source square gives four plausible edges.
            const sourceLoss = selected > sourceCapacity;
            const required = selected === 2 && sourceCapacity === 2 ? 6 : 4;
            const localLoss = required > capacity;
            expect(actual.outline).toEqual([
              !reportLoss && sourceLoss ? 0 : Math.min(required, capacity),
              required,
              localLoss ? 1 : 0,
              ...(reportLoss ? [sourceLoss ? 1 : 0] : [])
            ]);
            expect(actual.rings).toEqual(sourceLoss || localLoss ? [0, 0, 1, 0] : [1, 0, 0, 1]);
          }
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      });
    }
  }
}
