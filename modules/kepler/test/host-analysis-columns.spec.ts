// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';
import {KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

test('published analysis computes global statistics across source batches and changes columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    income: arrow.vectorFromArray([10, null], new arrow.Float32()),
    population: arrow.makeVector(new Float32Array([1, 2]))
  });
  const second = new arrow.Table({
    income: arrow.vectorFromArray([30, 50], new arrow.Float32()),
    population: arrow.makeVector(new Float32Array([3, 4]))
  });
  const dataset = session.importArrowDataset({id: 'published', table: first.concat(second)});
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'population', kind: 'range', column: 'population'}]
  });
  const sourceChunks = [...dataset.value.columns.income.data];
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024,
    getDerivedColumns: () => analysis.getTableColumns()
  });
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
  };
  try {
    analysis.setMode('z-score');
    await render();
    expect(analysis.colors.data.map(chunk => chunk.length)).toEqual([2, 2]);
    expect(
      analysis.getTableColumns()?.columns[analysis.resultColumn].data.map(chunk => chunk.length)
    ).toEqual([2, 2]);
    expect(analysis.summary?.datasetId).toBe('published');
    expect(analysis.summary?.column).toBe('income');
    expect(analysis.summary?.count).toBe(3);
    expect((await analysis.readRow(0))?.value).toBeCloseTo(-1, 5);
    expect((await analysis.readRow(1))?.value).toBeNaN();
    expect((await analysis.readRow(3))?.value).toBeCloseTo(1, 5);
    const page = await inspector.readPage({
      offset: 0,
      limit: 2,
      columns: [analysis.resultColumn],
      population: 'all',
      sort: {column: analysis.resultColumn, direction: 'descending'}
    });
    expect(Array.from(page?.sourceRowIndices ?? [])).toEqual([3, 2]);
    expect(page?.columns[analysis.resultColumn]?.[0]).toBeCloseTo(1, 5);
    expect(page?.columns[analysis.resultColumn]?.[1]).toBeCloseTo(0, 5);
    analysis.setPopulation('selected');
    explorer.setRange('population', 2, 4);
    await render();
    expect(analysis.summary?.count).toBe(2);
    expect((await analysis.readRow(2))?.value).toBeCloseTo(-Math.SQRT1_2, 5);
    analysis.setColumn('population');
    analysis.setMode('quantile');
    analysis.setClassCount(2);
    analysis.setPopulation('all');
    await render();
    expect(analysis.summary?.breaks).toEqual([1, 2.5, 4]);
    expect(analysis.summary?.classCounts).toEqual([2, 2]);
    expect((await analysis.readRow(3))?.value).toBe(1);
    expect(dataset.value.columns.income.data).toEqual(sourceChunks);
    expect(analysis.transferredReadbackBytes).toBeLessThan(1024);
  } finally {
    const fence = device.createFence();
    try {
      await inspector.destroy();
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

test('published standardization methods retain exact recipes and derive resident color domains', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'transforms',
    table: arrow
      .tableFromArrays({income: new Float32Array([2, 4])})
      .concat(arrow.tableFromArrays({income: new Float32Array([6, 8])}))
  });
  const explorer = session.openExplorer('transforms', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const cases = [
    {method: 'deviation-from-mean', expected: [-3, -1, 1, 3]},
    {method: 'mad', expected: [-1.5, -0.5, 0.5, 1.5]},
    {method: 'range-adjust', expected: [1 / 3, 2 / 3, 1, 4 / 3]},
    {method: 'range-standardize', expected: [0, 1 / 3, 2 / 3, 1]},
    {method: 'z-score', expected: [-3, -1, 1, 3].map(value => value / Math.sqrt(5))}
  ] as const;
  try {
    analysis.setVariance('population');
    for (const {method, expected} of cases) {
      analysis.setMode(method);
      const saved = analysis.serializeRecipe();
      analysis.setMode('off');
      analysis.setVariance('sample');
      analysis.restoreRecipe(saved);
      expect(analysis.settings.mode).toBe(method);
      expect(analysis.settings.variance).toBe('population');
      const encoder = device.createCommandEncoder();
      explorer.encode(encoder);
      analysis.encode(encoder);
      device.submit(encoder.finish());
      await analysis.readSummary();
      expect(analysis.summary?.operation).toBe(method);
      expect(analysis.resultColumn).toBe(`analysis_${method.replaceAll('-', '_')}`);
      for (const [index, value] of expected.entries()) {
        const result = await analysis.readRow(index);
        expect(result?.value).toBeCloseTo(value, 5);
        expect(result?.kind).toBe(method);
      }
      const buffer = analysis.colors.data[0].buffer;
      if (!(buffer instanceof Buffer)) throw new Error('Expected analysis-owned buffer');
      const bytes = await buffer.readAsync(0, 16);
      if (method === 'z-score') {
        // Switching back must retain the fixed z-score domain, not the prior output range.
        expect(Array.from(bytes.slice(0, 3))).not.toEqual([49, 54, 149]);
      } else {
        expect(Array.from(bytes.slice(0, 4))).toEqual([49, 54, 149, 255]);
        expect(Array.from(bytes.slice(12, 16))).toEqual([165, 0, 38, 255]);
      }
    }
  } finally {
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

test('published exact natural breaks retains its mode and classifies a bounded two-cluster source', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  session.importArrowDataset({
    id: 'clusters',
    table: arrow
      .tableFromArrays({value: new Float32Array([1, 2])})
      .concat(arrow.tableFromArrays({value: new Float32Array([10, 11])}))
  });
  const explorer = session.openExplorer('clusters', {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: []
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.supportsExactNaturalBreaks).toBe(true);
    analysis.setMode('natural-breaks');
    analysis.setClassCount(2);
    analysis.setNaturalBreaksMode('exact');
    const saved = analysis.serializeRecipe();
    analysis.setNaturalBreaksMode('approximate');
    analysis.setMode('quantile');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.mode).toBe('natural-breaks');
    expect(analysis.settings.naturalBreaksMode).toBe('exact');
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    expect(analysis.summary?.breaks).toEqual([1, 10, 11]);
    expect(analysis.summary?.classCounts).toEqual([2, 2]);
    expect((await analysis.readRow(0))?.value).toBe(0);
    expect((await analysis.readRow(3))?.value).toBe(1);
  } finally {
    const fence = device.createFence();
    try {
      await analysis.destroy(fence.signaled);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
