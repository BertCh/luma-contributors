// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCellRollup,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

// Quadbin resolution-2 tiles (0,0), (1,0), (2,0): Morton prefixes 0000,0001,0100.
// The first two share a resolution-1 parent, the third has another parent.
// Source counts [2,1,3] fold to [3,3]. Destination capacity alone determines
// overflow; either nonzero source loss flag yields candidateOverflow=1.
for (const capacity of [1, 3]) {
  it(`rollup status separates destination and inherited loss on poisoned replay (capacity ${capacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const count = input('source-count', 'uint32', Uint32Array.of(3));
    const overflow = input('source-overflow', 'uint32', Uint32Array.of(0));
    const candidateOverflow = input('source-candidate', 'uint32', Uint32Array.of(0));
    const source: GPUCellTable = {
      cells: input(
        'source-cells',
        'uint32x2',
        Uint32Array.of(0xffffffff, 0x4820ffff, 0xffffffff, 0x4821ffff, 0xffffffff, 0x4824ffff)
      ),
      counts: input('source-counts', 'uint32', Uint32Array.of(2, 1, 3)),
      status: {count, overflow, candidateOverflow}
    };
    const destination: GPUCellTable = {
      cells: output('cells', 'uint32x2', capacity),
      counts: output('counts', 'uint32', capacity),
      status: {
        count: output('count', 'uint32', 1),
        requiredCount: output('required', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1),
        candidateOverflow: output('candidate', 'uint32', 1)
      }
    };
    graph.add(
      new GPUCellRollup({
        family: 'quadbin',
        sourceResolution: 2,
        resolution: 1,
        source,
        output: destination
      })
    );
    const compiled = graph.compile();
    try {
      for (const frame of [
        {
          source: [3, 0, 0],
          status: capacity === 1 ? [1, 2, 1, 0] : [2, 2, 0, 0],
          counts: capacity === 1 ? [3] : [3, 3, 0]
        },
        {source: [1, 2, 0], status: [1, 1, 0, 1], counts: [2, ...Array(capacity - 1).fill(0)]},
        {source: [1, 0, 4], status: [1, 1, 0, 1], counts: [2, ...Array(capacity - 1).fill(0)]},
        {source: [0, 0, 0], status: [0, 0, 0, 0], counts: Array(capacity).fill(0)},
        {
          source: [3, 0, 0],
          status: capacity === 1 ? [1, 2, 1, 0] : [2, 2, 0, 0],
          counts: capacity === 1 ? [3] : [3, 3, 0]
        }
      ]) {
        fixture.writeInput(count, Uint32Array.of(frame.source[0]));
        fixture.writeInput(overflow, Uint32Array.of(frame.source[1]));
        fixture.writeInput(candidateOverflow, Uint32Array.of(frame.source[2]));
        const result = await fixture.submitTwice(compiled, async () => ({
          status: await fixture.readRows([
            destination.status.count,
            destination.status.requiredCount!,
            destination.status.overflow,
            destination.status.candidateOverflow!
          ]),
          counts: await fixture.read(destination.counts)
        }));
        expect(result.status).toEqual(frame.status);
        expect(result.counts).toEqual(frame.counts);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
