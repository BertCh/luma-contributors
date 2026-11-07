// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentScopeBinding} from '../../../examples/deck/kepler-compute/src/resident-scope-binding';
import {
  installKeplerComputeTableBridge,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

const VALUES = [0, 1, 2, 3, 4, 5];
const CATEGORIES = [0, 1, 0, 1, 0, 1];

type Harness = {
  update: (filters: readonly KeplerComputeFilter[]) => Promise<number[]>;
  destroy: () => Promise<void>;
};

async function makeHarness(device: Device, datasetId: string): Promise<Harness> {
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: datasetId,
    table: new arrow.Table({
      value: arrow.makeVector(new Float32Array(VALUES)),
      category: arrow.makeVector(new Uint32Array(CATEGORIES))
    })
  });
  const positions = new GPUVector({
    type: 'buffer',
    name: 'positions',
    format: 'float32x2',
    length: VALUES.length,
    buffer: device.createBuffer({
      data: new Float32Array(VALUES.flatMap(value => [value, 0])),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    }),
    ownsBuffer: true
  });
  const failures: string[] = [];
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {
      value: {kind: 'range', dimension: 'value', domain: [0, 5]},
      category: {kind: 'category', dimension: 'category', domain: [0, 1], categories: [0, 1]}
    },
    plots: {},
    layers: [{id: 'layer', geometryRevision: 1, geometry: {kind: 'point', positions}}],
    onUpdate() {},
    onUnsupported: reason => {
      failures.push(reason);
    },
    onPlotUnsupported() {}
  });
  const primary = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 1024 * 1024,
    maximumConcurrentReads: 1,
    maximumReadbackBytes: 4096,
    dimensions: [binding.externalDimension]
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'value'}, {name: 'category'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers?: readonly unknown[]): void {}
    },
    binding.bind(primary)
  );
  const nativeLayers = [{id: 'layer', type: 'point', config: {dataId: dataset.id}}];
  return {
    async update(filters) {
      table.filterTable(filters, nativeLayers);
      expect(failures).toEqual([]);
      const encoder = device.createCommandEncoder();
      primary.encode(encoder);
      device.submit(encoder.finish());
      expect(await binding.readSummary()).toBe(true);
      const page = await primary.readPage(0, 6, ['value']);
      return Array.from(page?.sourceRowIndices ?? []);
    },
    async destroy() {
      const fence = device.createFence();
      try {
        await Promise.all([binding.destroy(fence.signaled), primary.close(fence.signaled)]);
        await session.destroy(fence.signaled);
      } finally {
        fence.destroy();
        positions.destroy();
      }
    }
  };
}

function makeRange(id: string, field: string, value: [number, number]): KeplerComputeFilter {
  return {id, dataId: 'incremental', name: [field], type: 'range', value, fixedDomain: true};
}

function makeCategory(id: string, value: number[]): KeplerComputeFilter {
  return {id, dataId: 'incremental', name: ['category'], type: 'multiSelect', value};
}

it('resident scope binding add, remove and reassign filter equals a full rebuild', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const incremental = await makeHarness(device, 'incremental');
  const lower = makeRange('lower', 'value', [1.1, 5]);
  const upper = makeRange('upper', 'value', [0, 4]);
  const category = makeCategory('category', [1]);
  const steps: {filters: KeplerComputeFilter[]; expected: number[]}[] = [
    {filters: [lower], expected: [2, 3, 4, 5]},
    // add a filter
    {filters: [lower, category], expected: [3, 5]},
    {filters: [lower, category, upper], expected: [3]},
    // remove a filter
    {filters: [category, upper], expected: [1, 3]},
    // reassign the same id to a different field and kind
    {filters: [makeRange('category', 'value', [2, 5]), upper], expected: [2, 3, 4]},
    // reassign the same id back to the category field
    {filters: [category, upper], expected: [1, 3]},
    // change only a control value
    {filters: [makeCategory('category', [0]), upper], expected: [0, 2, 4]},
    // remove everything, then add from empty
    {filters: [], expected: [0, 1, 2, 3, 4, 5]},
    {filters: [upper, category], expected: [1, 3]}
  ];
  try {
    for (const [index, step] of steps.entries()) {
      const incrementalRows = await incremental.update(step.filters);
      const fresh = await makeHarness(device, 'incremental');
      try {
        const rebuiltRows = await fresh.update(step.filters);
        expect(rebuiltRows, `rebuild step ${index}`).toEqual(step.expected);
      } finally {
        await fresh.destroy();
      }
      expect(incrementalRows, `incremental step ${index}`).toEqual(step.expected);
    }
  } finally {
    await incremental.destroy();
  }
});
