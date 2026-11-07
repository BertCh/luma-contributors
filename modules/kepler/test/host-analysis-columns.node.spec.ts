// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {type KeplerSessionExplorer, KeplerComputeSession} from '../src/session/compute-session';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';

test('analysis selects published Float32 fields and preserves multi-batch source ownership', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = arrow.tableFromArrays({
    income: new Float32Array([10, 20]),
    population: new Float32Array([100, 400]),
    category: ['a', 'b']
  });
  const second = arrow.tableFromArrays({
    income: new Float32Array([30, 40]),
    population: new Float32Array([800, 1600]),
    category: ['c', 'd']
  });
  const dataset = session.importArrowDataset({id: 'published', table: first.concat(second)});
  const sourceChunks = [...dataset.value.columns.income.data];
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.colors.data.map(chunk => chunk.length)).toEqual([2, 2]);
    expect(analysis.colors.data.map(chunk => chunk.byteOffset)).toEqual([0, 8]);
    expect(analysis.datasetId).toBe('published');
    expect(analysis.availableColumns).toEqual(['income', 'population']);
    expect(analysis.settings.column).toBe('income');
    analysis.setColumn('population');
    analysis.setMode('quantile');
    const saved = analysis.serializeRecipe();
    expect(JSON.parse(saved).operation.column).toBe('population');
    analysis.setColumn('income');
    analysis.restoreRecipe(saved);
    expect(analysis.settings.column).toBe('population');
    expect(() => analysis.setColumn('category')).toThrow('Float32');
    const invalid = JSON.parse(saved);
    invalid.operation.column = 'missing';
    expect(() => analysis.restoreRecipe(JSON.stringify(invalid))).toThrow('Float32');
    expect(analysis.settings.column).toBe('population');
    expect(dataset.value.columns.income.data).toEqual(sourceChunks);
    expect(dataset.value.columns.income.data.map(chunk => chunk.length)).toEqual([2, 2]);
    await analysis.destroy(Promise.resolve());
    expect(sourceChunks.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});

test('exact natural breaks validates total source capacity before mutating analysis', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'capacity',
    table: arrow.tableFromArrays({value: new Float32Array(1025)})
  });
  const explorer = {
    dataset,
    isCurrent: true,
    currentRevision: 0
  } as unknown as KeplerSessionExplorer;
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  try {
    expect(analysis.supportsExactNaturalBreaks).toBe(false);
    analysis.setMode('natural-breaks');
    const before = analysis.settings;
    const saved = JSON.parse(analysis.serializeRecipe());
    saved.operation.naturalBreaksMode = 'exact';
    expect(() => analysis.setNaturalBreaksMode('exact')).toThrow('1024 source rows');
    expect(analysis.settings).toEqual(before);
    expect(() => analysis.restoreRecipe(JSON.stringify(saved))).toThrow('1024 source rows');
    expect(analysis.settings).toEqual(before);
    analysis.setMode('off');
    analysis.setNaturalBreaksMode('exact');
    expect(() => analysis.setMode('natural-breaks')).toThrow('1024 source rows');
    expect(analysis.settings.mode).toBe('off');
  } finally {
    await analysis.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    device.destroy();
  }
});
