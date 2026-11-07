// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import {GPUKeplerTableInspector} from '../../src/exploration/gpu-table-inspector';
import {exportSelectedSource} from '../../../../examples/deck/kepler-compute/src/source-export';

for (const external of [false, true]) {
  test(`source-only exploration preserves original batches and bounded identities (external=${external})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) throw new Error('WebGPU is required');
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 1024 * 1024});
    const first = new arrow.Table({
      geometry: arrow.vectorFromArray(['point-a', null], new arrow.Utf8())
    });
    const empty = new arrow.Table({geometry: arrow.vectorFromArray([], new arrow.Utf8())});
    const last = new arrow.Table({
      geometry: arrow.vectorFromArray(['point-b', 'point-c'], new arrow.Utf8())
    });
    const source = new arrow.Table(first.schema, [
      ...first.batches,
      ...empty.batches,
      ...last.batches
    ]);
    const dataset = session.importArrowDataset({
      id: 'source-only',
      table: source,
      normalization: {columns: []}
    });
    const sourceLengths = source.batches.map(batch => batch.numRows);
    expect(sourceLengths).toEqual([2, 0, 2]);
    const mask = new GPUVector({
      type: 'data',
      format: 'uint32',
      data: [new Uint32Array([1, 0]), new Uint32Array(), new Uint32Array([0, 1])].map(
        values =>
          new GPUData({
            format: 'uint32',
            length: values.length,
            ownsBuffer: true,
            buffer: device.createBuffer({
              byteLength: Math.max(4, values.byteLength),
              data: values.length ? values : undefined,
              usage: Buffer.STORAGE | Buffer.COPY_DST
            })
          })
      ),
      ownsData: true
    });
    const explorer = session.openExplorer(dataset.id, {
      dimensions: external ? [{id: 'geometry', kind: 'external', mask}] : [],
      workingMemoryBudgetBytes: 65536,
      maximumReadbackBytes: 128,
      maximumPageSize: 2
    });
    const inspector = new GPUKeplerTableInspector({session, explorer, sortMemoryBudgetBytes: 0});
    const submit = () => {
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      device.submit(encoder.finish());
    };
    try {
      expect(dataset.value.columns).toEqual({});
      expect(dataset.value.analytics.table.batches.map(batch => batch.numRows)).toEqual(
        sourceLengths
      );
      expect(dataset.value.uploadedBufferBytes).toBe(0);
      expect(explorer.selectionMask.data.map(chunk => chunk.length)).toEqual(sourceLengths);
      submit();
      expect((await explorer.readSummary())?.selectedCount).toBe(4);
      const page = await explorer.readPage(1, 2);
      expect(page?.sourceRowIndices).toEqual(new Uint32Array([1, 2]));
      expect(page?.columns).toEqual({});
      expect(page?.validity).toEqual({});
      expect((await explorer.readRows([3, 1]))?.sourceRowIndices).toEqual(new Uint32Array([3, 1]));
      if (external) {
        explorer.setExternalActive('geometry', true);
        submit();
        expect((await explorer.readSummary())?.selectedCount).toBe(2);
      }
      const tablePage = await inspector.readPage({
        offset: 0,
        limit: 2,
        population: 'selected',
        columns: []
      });
      expect(tablePage?.sourceRowIndices).toEqual(new Uint32Array(external ? [0, 3] : [0, 1]));
      expect(tablePage?.columns).toEqual({});
      const exported = await exportSelectedSource(session, explorer, {
        mode: 'all-selected',
        maximumRows: 4,
        maximumReadbackBytes: 64,
        maximumCpuBytes: 64,
        maximumOutputBytes: 65536,
        format: 'arrow'
      });
      try {
        const restored = arrow.tableFromIPC(exported.bytes);
        expect([...restored.getChild('geometry')!]).toEqual(
          external ? ['point-a', 'point-c'] : ['point-a', null, 'point-b', 'point-c']
        );
        expect(restored.schema.fields.map(field => field.name)).toEqual([
          'geometry',
          exported.sourceRowColumn
        ]);
        expect([...restored.getChild(exported.sourceRowColumn)!]).toEqual(
          external ? [0, 3] : [0, 1, 2, 3]
        );
      } finally {
        exported.destroy();
      }
      explorer.clearAll();
      submit();
      expect((await explorer.readSelectedCount())?.selectedCount).toBe(4);
      await expect(explorer.readPage(0, 3)).rejects.toThrow(/bounded/);
    } finally {
      await inspector.destroy();
      const fence = device.createFence();
      try {
        await session.destroy(fence.signaled);
        expect(session.registry.memoryUsage.totalBytes).toBe(0);
        expect(mask.data[0].buffer.destroyed).toBe(false);
      } finally {
        fence.destroy();
        mask.destroy();
      }
    }
  });
}
