// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUHealpixCellAggregation,
  GPUHealpixPointToCell,
  createHealpixDGGSJSON
} from '../../../src/gpu-spatial-analysis/healpix-cells/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const scheme of ['nested', 'uniq'] as const) {
  for (const masked of [false, true]) {
    it(`HEALPix ${scheme} validity handoff ${masked ? 'masked' : 'unmasked'} preserves real pixel zero (DGGS-108)`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      // Point-to-cell's invalid-row sentinel followed by genuine order-0 nested pixel zero.
      // UNIQ adds 4^(order+1), so that real cell is key 4. No arithmetic oracle is shared.
      const realKey = scheme === 'uniq' ? 4 : 0;
      const cellWords = rig.output('uint32', 4);
      const cells = rig.graph.createDataView(cellWords.buffer, {
        format: 'uint32x2',
        length: 2,
        byteOffset: cellWords.byteOffset
      });
      const rowWords = rig.output('uint32', 4);
      const rowCells = rig.graph.createDataView(rowWords.buffer, {
        format: 'uint32x2',
        length: 2,
        byteOffset: rowWords.byteOffset
      });
      const validity = rig.output('uint32', 2);
      const counts = rig.output('uint32', 2);
      const status = {
        count: rig.output('uint32', 1),
        overflow: rig.output('uint32', 1),
        invalidCount: rig.output('uint32', 1)
      };
      try {
        rig.run(
          new GPUHealpixPointToCell({
            order: 0,
            scheme,
            positions: rig.input(Float32Array.of(NaN, 0, 45, 45), 'float32x2', 2),
            output: {cells: rowCells, validity}
          }),
          new GPUHealpixCellAggregation({
            order: 0,
            scheme,
            cellCapacity: 2,
            cells: rowCells,
            mask: masked ? validity : undefined,
            output: {cells, counts, status}
          })
        );
        expect(await rig.readUint(rowWords)).toEqual([0, 0, realKey, 0]);
        expect(await rig.readUint(validity)).toEqual([0, 1]);
        const expectedCount = masked || scheme === 'uniq' ? 1 : 2;
        expect(await rig.readUint(status.count)).toEqual([1]);
        expect(await rig.readUint(status.overflow)).toEqual([0]);
        expect(await rig.readUint(status.invalidCount)).toEqual([2 - expectedCount]);
        const words = await rig.readUint(cellWords);
        const writtenCounts = await rig.readUint(counts);
        expect(words.slice(0, 2)).toEqual([realKey, 0]);
        expect(writtenCounts[0]).toBe(expectedCount);
        const document = createHealpixDGGSJSON({
          order: 0,
          scheme,
          cells: [BigInt(words[0]) | (BigInt(words[1]) << 32n)],
          properties: {count: {values: writtenCounts.slice(0, 1)}}
        });
        expect(document.values['count'][0].data).toEqual([expectedCount]);
      } finally {
        rig.destroy();
      }
    });
  }
}
