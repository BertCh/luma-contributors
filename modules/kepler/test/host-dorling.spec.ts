// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerDorlingSettings} from '../src';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

test('Dorling publishes source-aligned exhausted layouts, recomputes selected rows and gates projected map overlays through replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const first = new arrow.Table({
    x: arrow.vectorFromArray([0, 1], new arrow.Float32()),
    y: arrow.makeVector(new Float32Array(2)),
    value: arrow.makeVector(new Float32Array([4, 9])),
    row: arrow.makeVector(new Float32Array([0, 1]))
  });
  const second = new arrow.Table({
    x: arrow.vectorFromArray([8, 10, null], new arrow.Float32()),
    y: arrow.makeVector(new Float32Array(3)),
    value: arrow.makeVector(new Float32Array([0, -1, 16])),
    row: arrow.makeVector(new Float32Array([2, 3, 4]))
  });
  const dataset = session.importArrowDataset({id: 'dorling-source', table: first.concat(second)});
  const sourceChunks = ['x', 'y', 'value'].flatMap(name => dataset.value.columns[name].data);
  const sourceBuffers = [
    ...new Set(
      sourceChunks.map(chunk =>
        chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer
      )
    )
  ];
  const sourceWrites = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  const uploadedBufferBytes = dataset.value.uploadedBufferBytes;
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate: () => {}});
  const settings: KeplerDorlingSettings = {
    coordinateSpace: 'planar',
    xColumn: 'x',
    yColumn: 'y',
    valueColumn: 'value',
    crs: 'LOCAL:fixture',
    metric: 'native',
    units: 'meters',
    bounds: [-1, -1, 12, 1],
    areaScale: Math.PI,
    maximumIterations: 0,
    overlapTolerance: 0.01,
    attraction: 0
  };
  const normalizedSettings = {
    ...settings,
    areaScale: Math.fround(Math.PI),
    overlapTolerance: Math.fround(0.01)
  };
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    expect(analysis.error).toBe('');
  };
  const readColumn = async (name: string, format: 'float32' | 'uint32') => {
    const buffer = analysis.getTableColumns()!.columns[name].data[0].buffer;
    if (!(buffer instanceof Buffer)) throw new Error('Expected owned cartogram output');
    const bytes = await buffer.readAsync(0, 20);
    return format === 'float32' ? new Float32Array(bytes.buffer) : new Uint32Array(bytes.buffer);
  };
  try {
    analysis.configureCartogram(settings);
    const savedRecipe = analysis.serializeRecipe();
    expect(JSON.parse(savedRecipe).operation).toEqual({
      kind: 'dorling-cartogram',
      ...normalizedSettings
    });
    await render();
    expect(analysis.cartogramSummary).toMatchObject({
      operation: 'dorling-cartogram',
      sourceRevision: dataset.revision,
      count: 3,
      nullCount: 2,
      cartogram: {
        iterations: 0,
        converged: false,
        budgetExhausted: true,
        excludedCount: 2,
        maximumDisplacement: 0,
        crs: 'LOCAL:fixture',
        units: 'meters',
        display: 'table-only'
      }
    });
    expect(analysis.cartogramSummary?.cartogram.maximumPenetration).toBeGreaterThan(0);
    expect(analysis.getResultLayers()).toEqual([]);
    expect(analysis.cartogramUnavailableReason).toMatch(/Declare the CRS unit size/);
    const snapshot = analysis.getTableColumns()!;
    expect(snapshot.isCurrent()).toBe(true);
    expect(Object.keys(snapshot.columns)).toHaveLength(4);
    for (const vector of Object.values(snapshot.columns))
      expect(vector.data.map(chunk => chunk.length)).toEqual([2, 3]);
    const radii = await readColumn('analysis_cartogram_radius_meters', 'float32');
    const centerX = await readColumn('analysis_cartogram_center_x', 'float32');
    const centerY = await readColumn('analysis_cartogram_center_y', 'float32');
    expect(Array.from(await readColumn('analysis_cartogram_source_id', 'uint32'))).toEqual([
      0, 1, 2, 3, 4
    ]);
    for (const [sourceRowIndex, value] of [4, 9, 0].entries()) {
      expect(radii[sourceRowIndex]).toBeCloseTo(
        Math.sqrt((normalizedSettings.areaScale * value) / Math.PI),
        5
      );
      expect(centerX[sourceRowIndex]).toBe([0, 1, 8][sourceRowIndex]);
      expect(centerY[sourceRowIndex]).toBe(0);
      expect(await analysis.readRow(sourceRowIndex)).toMatchObject({
        kind: 'dorling-cartogram',
        sourceRowIndex,
        sourceRevision: dataset.revision,
        sourceId: sourceRowIndex,
        centerX: centerX[sourceRowIndex],
        centerY: 0,
        radius: radii[sourceRowIndex],
        value: radii[sourceRowIndex]
      });
    }
    // Zero is valid and stationary; negative values and null coordinates keep their source slots.
    expect(radii[2]).toBe(0);
    for (const sourceRowIndex of [3, 4]) {
      expect(radii[sourceRowIndex]).toBeNaN();
      expect(centerX[sourceRowIndex]).toBeNaN();
      expect(centerY[sourceRowIndex]).toBeNaN();
      expect(await analysis.readRow(sourceRowIndex)).toMatchObject({
        sourceRowIndex,
        sourceId: sourceRowIndex,
        centerX: null,
        centerY: null,
        radius: null,
        value: NaN
      });
    }

    analysis.setPopulation('selected');
    explorer.setRange('row', 0, 0);
    expect(snapshot.isCurrent()).toBe(false);
    await render();
    expect(analysis.cartogramSummary).toMatchObject({
      count: 1,
      nullCount: 4,
      cartogram: {
        iterations: 0,
        converged: true,
        budgetExhausted: false,
        excludedCount: 4,
        maximumPenetration: 0
      }
    });
    expect((await analysis.readRow(0))?.value).toBeCloseTo(radii[0], 5);
    expect((await analysis.readRow(1))?.value).toBeNaN();
    expect(analysis.getTableColumns()!.columns['analysis_cartogram_radius_meters']).not.toBe(
      snapshot.columns['analysis_cartogram_radius_meters']
    );

    analysis.setMode('off');
    analysis.restoreRecipe(savedRecipe);
    await render();
    expect(analysis.settings.population).toBe('all');
    expect(analysis.settings.cartogram).toEqual(normalizedSettings);
    expect(Array.from(await readColumn('analysis_cartogram_radius_meters', 'float32'))).toEqual(
      Array.from(radii)
    );
    expect(Array.from(await readColumn('analysis_cartogram_center_x', 'float32'))).toEqual(
      Array.from(centerX)
    );
    expect(analysis.cartogramSummary?.cartogram.budgetExhausted).toBe(true);

    analysis.configureCartogram({...settings, crs: 'EPSG:3857'});
    await render();
    expect(analysis.cartogramSummary?.cartogram.display).toBe('web-mercator-overlay');
    expect(analysis.cartogramUnavailableReason).toBeUndefined();
    expect(analysis.getResultLayers()).toHaveLength(1);
    const mercatorRecipe = analysis.serializeRecipe();
    // Finite radii can extend beyond the world even when every source center is in bounds.
    analysis.configureCartogram({...settings, crs: 'EPSG:3857', areaScale: 1e15});
    await render();
    expect(analysis.cartogramSummary?.cartogram.display).toBe('table-only');
    expect(analysis.cartogramSummary?.cartogram.budgetExhausted).toBe(true);
    expect(analysis.cartogramUnavailableReason).toMatch(/world/i);
    expect(analysis.getResultLayers()).toHaveLength(0);
    expect(analysis.getTableColumns()?.isCurrent()).toBe(true);
    expect(Number.isFinite((await analysis.readRow(1))?.value)).toBe(true);
    const resultBounds = analysis.cartogramSummary!.cartogram.resultBounds!;
    expect(resultBounds).toHaveLength(4);
    expect(resultBounds.every(Number.isFinite)).toBe(true);
    expect(resultBounds.some(bound => Math.abs(bound) > Math.PI * 6378137)).toBe(true);
    analysis.restoreRecipe(mercatorRecipe);
    await render();
    expect(analysis.cartogramSummary?.cartogram.display).toBe('web-mercator-overlay');
    expect(analysis.cartogramUnavailableReason).toBeUndefined();
    expect(analysis.getResultLayers()).toHaveLength(1);
    expect(
      analysis
        .getTableColumns()!
        .columns['analysis_cartogram_radius_meters'].data.map(chunk => chunk.length)
    ).toEqual([2, 3]);
    for (const [index, chunk] of ['x', 'y', 'value']
      .flatMap(name => dataset.value.columns[name].data)
      .entries())
      expect(chunk).toBe(sourceChunks[index]);
    expect(dataset.value.uploadedBufferBytes).toBe(uploadedBufferBytes);
    for (const write of sourceWrites) expect(write).not.toHaveBeenCalled();
  } finally {
    for (const write of sourceWrites) write.mockRestore();
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
