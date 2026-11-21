// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

it('B1 B2 S1 preserves padded ring counts and shared scalar layouts through loss and clean replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const count = input('input-count', 'uint32', Uint32Array.of(1));
  const flags = output('shared-flags', 'uint32', 128, {preserveOnReplay: true});
  const overflow = graph.createDataView(flags.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: flags.byteOffset
  });
  const candidateOverflow = graph.createDataView(flags.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: flags.byteOffset + 4
  });
  const ringOverflow = graph.createDataView(flags.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: flags.byteOffset + 256
  });
  const ringCount = input('ring-count', 'uint32', Uint32Array.of(99, 91, 9898));
  const ringOffsets = output('ring-offsets', 'uint32', 3);
  const status = {
    count: output('count', 'uint32', 1),
    requiredCount: output('required', 'uint32', 1),
    overflow,
    candidateOverflow
  };
  graph.add(
    new GPUCellSetOutline({
      family: 'quadbin',
      count,
      // Two separated zoom-2 squares (Morton prefixes 3 and 7) have eight
      // exterior edges. Capacity four retains a closed square, which must not
      // appear as the complete two-component outline. A one-cell replay fits.
      cells: input(
        'cells',
        'uint32x2',
        Uint32Array.of(0xffffffff, 0x4823ffff, 0xffffffff, 0x4827ffff)
      ),
      output: {
        rows: output('rows', 'uint32', 4),
        cells: output('output-cells', 'uint32x2', 4),
        edgeIndices: output('edges', 'uint32', 4),
        endpoints: output('endpoints', 'float32x4', 4),
        status
      },
      rings: {
        output: {
          positions: output('positions', 'float32x2', 10),
          ringOffsets,
          status: {
            count: ringCount,
            overflow: ringOverflow,
            openSegmentCount: output('open', 'uint32', 1),
            touchingSegmentCount: output('touching', 'uint32', 1)
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const selected of [1, 2, 1]) {
      fixture.writeInput(count, Uint32Array.of(selected));
      const actual = await fixture.submitTwice(compiled, async () => ({
        status: await fixture.readRows([status.count, status.requiredCount]),
        flags: await fixture.read(flags),
        ringCount: await fixture.read(ringCount),
        offsets: await fixture.read(ringOffsets)
      }));
      expect(actual.status).toEqual([4, selected === 2 ? 8 : 4]);
      expect(actual.ringCount).toEqual([selected === 2 ? 0 : 1, 91, 9898]);
      expect(actual.offsets).toEqual(selected === 2 ? [0, 0, 0] : [0, 5, 5]);
      const expectedFlags = Array(128).fill(0xa5a5a5a5);
      expectedFlags[0] = selected === 2 ? 1 : 0;
      expectedFlags[1] = 0;
      expectedFlags[64] = 0;
      expect(actual.flags).toEqual(expectedFlags);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
