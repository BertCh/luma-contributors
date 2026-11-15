// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createPointNeighborGridIndexNodes} from '../../../src/analysis-core/kernels/point-neighbor-grid-index';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

for (const keys of [
  // Empty leading/interior cells, ties, the last cell, and excluded rows in between valid rows.
  [7, 2, 8, 2, 5, 8, 7, 2, 8],
  // No indexed rows: every offset must be zero and excluded rows keep input order.
  [8, 8, 8, 8, 8, 8, 8, 8, 8]
]) {
  it(`the keyed neighbor index preserves stable rows and excludes sentinel keys, ${keys.join(',')}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const cellCount = 8;
    const rowCount = keys.length;
    // Finite, in-range padding must never be counted. Both input views start after four words.
    const paddedKeys = rig.input(Uint32Array.from([0, 0, 0, 0, ...keys, 1, 1]), 'uint32');
    const cellKeys = rig.graph.createDataView(paddedKeys.buffer, {
      format: 'uint32',
      length: rowCount,
      byteOffset: 16
    });
    const paddedRows = rig.input(
      Uint32Array.from([99, 99, 99, 99, ...keys.map((_, row) => row), 99]),
      'uint32'
    );
    const sourceRows = rig.graph.createDataView(paddedRows.buffer, {
      format: 'uint32',
      length: rowCount,
      byteOffset: 16
    });
    const result = rig.output('uint32', cellCount + 1 + rowCount + 2);
    try {
      rig.run({
        getCommandNodes: graph => {
          const index = createPointNeighborGridIndexNodes(graph, {
            id: 'cells',
            cellKeys,
            sourceRows,
            cellCount,
            rowCount
          });
          return [
            ...index.nodes,
            createWGSLKernelNode(graph, {
              id: 'read-cells',
              operation: 'NeighborCellIndexTest',
              variant: 'readback',
              bindings: [
                {name: 'offsets', view: index.cellOffsets, type: 'u32', access: 'read'},
                {name: 'rows', view: index.sortedRows, type: 'u32', access: 'read'},
                {name: 'count', view: index.gridCount, type: 'u32', access: 'read'},
                {name: 'overflow', view: index.gridOverflow, type: 'u32', access: 'read'},
                {name: 'result', view: result, type: 'u32', access: 'read_write'}
              ],
              invocationCount: result.length,
              body: `var value = 0u;
  if (index < ${cellCount + 1}u) {
    value = offsets[offsetsOffset + index];
  } else if (index < ${cellCount + 1 + rowCount}u) {
    value = rows[rowsOffset + index - ${cellCount + 1}u];
  } else if (index == ${cellCount + 1 + rowCount}u) {
    value = count[countOffset];
  } else {
    value = overflow[overflowOffset];
  }
  result[resultOffset + index] = value;`
            })
          ];
        }
      });
      // Independent bucket concatenation gives both exclusive offsets and stable row order.
      const buckets = Array.from({length: cellCount + 1}, (_, cell) =>
        keys.flatMap((key, row) => (key === cell ? [row] : []))
      );
      const offsets = [0];
      for (let cell = 0; cell < cellCount; cell++) {
        offsets.push(offsets[cell] + buckets[cell].length);
      }
      expect(await rig.readUint(result)).toEqual([
        ...offsets,
        ...buckets.flat(),
        offsets[cellCount],
        0
      ]);
    } finally {
      rig.destroy();
    }
  });
}
