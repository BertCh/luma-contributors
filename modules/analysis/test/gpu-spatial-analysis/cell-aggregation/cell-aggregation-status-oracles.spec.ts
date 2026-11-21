// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCellAggregation,
  GPUCellPyramid,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

// Independent integer/tile oracle: Quadbin header 0x4800000000000000, resolution
// 2 in bits 52..56, Morton prefixes 0000,0001,0100, and 48 low one-bits encode
// tiles (0,0), (1,0), (2,0). The first two share their resolution-1 parent;
// the third has a different parent. At resolution 0 all share the world cell.
// Six observations A,A,B,C,C,C therefore have fine counts [2,1,3], parent
// counts [3,3], world count [6]. Capacity retains the smallest-key prefix.
// Status order throughout is count, requiredCount, overflow, candidateOverflow.
const KEY_WORDS = [
  0xffffffff, 0x4820ffff, 0xffffffff, 0x4820ffff, 0xffffffff, 0x4821ffff, 0xffffffff, 0x4824ffff,
  0xffffffff, 0x4824ffff, 0xffffffff, 0x4824ffff
];

for (const scenario of [
  {
    name: 'raw truncated',
    pyramid: false,
    capacities: [2],
    counts: [[2, 1]],
    status: [[2, 3, 1, 0]]
  },
  {
    name: 'raw spare capacity',
    pyramid: false,
    capacities: [4],
    counts: [[2, 1, 3, 0]],
    status: [[3, 3, 0, 0]]
  },
  {
    name: 'pyramid fine loss',
    pyramid: true,
    capacities: [2, 1, 1],
    counts: [[2, 1], [3], [3]],
    status: [
      [2, 3, 1, 0],
      [1, 1, 0, 1],
      [1, 1, 0, 1]
    ]
  },
  {
    name: 'pyramid middle loss',
    pyramid: true,
    capacities: [4, 1, 1],
    counts: [[2, 1, 3, 0], [3], [3]],
    status: [
      [3, 3, 0, 0],
      [1, 2, 1, 0],
      [1, 1, 0, 1]
    ]
  },
  {
    name: 'pyramid complete',
    pyramid: true,
    capacities: [4, 3, 2],
    counts: [
      [2, 1, 3, 0],
      [3, 3, 0],
      [6, 0]
    ],
    status: [
      [3, 3, 0, 0],
      [2, 2, 0, 0],
      [1, 1, 0, 0]
    ]
  }
]) {
  it(`cell status and every count follow the tile cardinalities (${scenario.name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const mask = input('mask', 'uint32', Uint32Array.of(1, 1, 1, 1, 1, 1));
    const tables: GPUCellTable[] = scenario.capacities.map((capacity, level) => ({
      cells: output(`cells-${level}`, 'uint32x2', capacity),
      counts: output(`counts-${level}`, 'uint32', capacity),
      status: {
        count: output(`count-${level}`, 'uint32', 1),
        requiredCount: output(`required-${level}`, 'uint32', 1),
        overflow: output(`overflow-${level}`, 'uint32', 1),
        candidateOverflow: output(`candidate-${level}`, 'uint32', 1)
      }
    }));
    const levelCounts = scenario.pyramid ? output('level-counts', 'uint32', 3) : undefined;
    const common = {
      family: 'quadbin' as const,
      cells: input('keys', 'uint32x2', Uint32Array.from(KEY_WORDS)),
      mask
    };
    graph.add(
      scenario.pyramid
        ? new GPUCellPyramid({
            ...common,
            resolutions: [2, 1, 0],
            output: {levels: tables, levelCounts}
          })
        : new GPUCellAggregation({...common, resolution: 2, output: tables[0]})
    );
    const compiled = graph.compile();
    try {
      for (const frame of ['all', 'one', 'empty', 'all'] as const) {
        fixture.writeInput(
          mask,
          Uint32Array.from(
            frame === 'all'
              ? [1, 1, 1, 1, 1, 1]
              : frame === 'one'
                ? [0, 0, 0, 1, 1, 1]
                : [0, 0, 0, 0, 0, 0]
          )
        );
        const result = await fixture.submitTwice(compiled, async () => ({
          counts: await Promise.all(tables.map(table => fixture.read(table.counts))),
          status: await Promise.all(
            tables.map(table =>
              fixture.readRows([
                table.status.count,
                table.status.requiredCount!,
                table.status.overflow,
                table.status.candidateOverflow!
              ])
            )
          ),
          levelCounts: levelCounts ? await fixture.read(levelCounts) : undefined
        }));
        const expectedStatus =
          frame === 'all'
            ? scenario.status
            : scenario.capacities.map(() => (frame === 'one' ? [1, 1, 0, 0] : [0, 0, 0, 0]));
        expect(result.status).toEqual(expectedStatus);
        expect(result.counts).toEqual(
          frame === 'all'
            ? scenario.counts
            : scenario.capacities.map(capacity =>
                frame === 'one' ? [3, ...Array(capacity - 1).fill(0)] : Array(capacity).fill(0)
              )
        );
        if (scenario.pyramid) {
          // Compare directly to the independent counts, never to GPU table.status.count.
          expect(result.levelCounts).toEqual(expectedStatus.map(status => status[0]));
        }
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
