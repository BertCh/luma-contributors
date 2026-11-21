// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellCompaction} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-compaction';
import {createOutputOracleFixture} from './output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

// Two distinct resolution-2 tiles, (0,0) and (2,0), cannot merge (they are
// not siblings). Uncompacting them to their existing resolution also yields
// exactly two cells. Row 2 is the invalid zero key: including it drops one
// input row, but never adds an output. This is a hand cardinality oracle.
for (const type of ['compact', 'uncompact'] as const) {
  for (const capacity of [1, 4]) {
    it(`compaction status overwrites every poisoned scalar (${type}, capacity ${capacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const count = input('input-count', 'uint32', Uint32Array.of(2));
      const status = {
        count: output('count', 'uint32', 1),
        requiredCount: output('required', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        droppedCount: output('dropped', 'uint32', 1)
      };
      graph.add(
        new GPUCellCompaction({
          family: 'quadbin',
          operation: {type, resolution: 2},
          count,
          cells: input(
            'input-cells',
            'uint32x2',
            Uint32Array.of(0xffffffff, 0x4820ffff, 0xffffffff, 0x4824ffff, 0, 0)
          ),
          output: {cells: output('cells', 'uint32x2', capacity), status}
        })
      );
      const compiled = graph.compile();
      try {
        for (const selected of [2, 3, 0, 2]) {
          fixture.writeInput(count, Uint32Array.of(selected));
          const actual = await fixture.submitTwice(compiled, () =>
            fixture.readRows([
              status.count,
              status.requiredCount,
              status.overflow,
              status.droppedCount
            ])
          );
          expect(actual).toEqual(
            selected === 0
              ? [0, 0, 0, 0]
              : capacity === 1
                ? [1, 2, 1, selected === 3 ? 1 : 0]
                : [2, 2, 0, selected === 3 ? 1 : 0]
          );
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const type of ['compact', 'uncompact'] as const) {
  for (const capacity of [2, 8]) {
    it(`counted source keeps missing children and destination overflow distinct (${type}, capacity ${capacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      // The four zoom-2 Morton prefixes 0,1,2,3 are siblings: three cannot
      // merge; adding the fourth yields exactly the zoom-1 tile with prefix 0.
      // Two zoom-1 parents expand to zoom-2 prefixes 0..7 in ascending order.
      const source = {
        cells: input(
          'source',
          'uint32x2',
          type === 'compact'
            ? Uint32Array.of(
                0xffffffff,
                0x4820ffff,
                0xffffffff,
                0x4821ffff,
                0xffffffff,
                0x4822ffff,
                0xffffffff,
                0x4823ffff
              )
            : Uint32Array.of(0xffffffff, 0x4813ffff, 0xffffffff, 0x4817ffff)
        ),
        status: {
          count: input('source-count', 'uint32', Uint32Array.of(type === 'compact' ? 3 : 1)),
          overflow: input('source-overflow', 'uint32', Uint32Array.of(0)),
          candidateOverflow: input('source-candidate', 'uint32', Uint32Array.of(0))
        }
      };
      const status = {
        count: output('count', 'uint32', 1),
        requiredCount: output('required', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        candidateOverflow: output('candidate', 'uint32', 1),
        droppedCount: output('dropped', 'uint32', 1)
      };
      const cells = output('cells', 'uint32x2', capacity);
      graph.add(
        new GPUCellCompaction({
          family: 'quadbin',
          source,
          operation: {type, resolution: 2, ...(type === 'compact' ? {minimumResolution: 1} : {})},
          output: {cells, status}
        })
      );
      const compiled = graph.compile();
      try {
        for (const [overflow, candidate, supplyMissing] of [
          [1, 0, false],
          [0, 1, false],
          [1, 1, false],
          [0, 0, false],
          [0, 0, true],
          [0, 1, false],
          [0, 0, true]
        ] as const) {
          fixture.writeInput(
            source.status.count,
            Uint32Array.of(type === 'compact' ? (supplyMissing ? 4 : 3) : supplyMissing ? 2 : 1)
          );
          fixture.writeInput(source.status.overflow, Uint32Array.of(overflow));
          fixture.writeInput(source.status.candidateOverflow, Uint32Array.of(candidate));
          const childKeys = [
            [0xffffffff, 0x4820ffff],
            [0xffffffff, 0x4821ffff],
            [0xffffffff, 0x4822ffff],
            [0xffffffff, 0x4823ffff],
            [0xffffffff, 0x4824ffff],
            [0xffffffff, 0x4825ffff],
            [0xffffffff, 0x4826ffff],
            [0xffffffff, 0x4827ffff]
          ];
          const expected =
            type === 'compact'
              ? supplyMissing
                ? [[0xffffffff, 0x4813ffff]]
                : childKeys.slice(0, 3)
              : childKeys.slice(0, supplyMissing ? 8 : 4);
          const count = Math.min(expected.length, capacity);
          const actual = await fixture.submitTwice(compiled, async () => ({
            status: await fixture.readRows([
              status.count,
              status.requiredCount,
              status.overflow,
              status.candidateOverflow,
              status.droppedCount
            ]),
            cells: await fixture.read(cells)
          }));
          expect(actual.status).toEqual([
            count,
            expected.length,
            expected.length > capacity ? 1 : 0,
            overflow || candidate ? 1 : 0,
            0
          ]);
          expect(actual.cells).toEqual([
            ...expected.slice(0, count).flat(),
            ...Array((capacity - count) * 2).fill(0)
          ]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
