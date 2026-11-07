// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerDBSCANSettings} from '../src';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';
import {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';

const settings: KeplerDBSCANSettings = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  epsilon: 500,
  minimumSamples: 3
};

// Independent spherical distance oracle uses the same Float32 source coordinates, in Float64.
function sphericalMeters(first: readonly number[], second: readonly number[]): number {
  const radians = Math.PI / 180;
  const latitudeDifference = (second[1] - first[1]) * radians;
  const longitudeDifference = (second[0] - first[0]) * radians;
  const haversine =
    Math.sin(latitudeDifference / 2) ** 2 +
    Math.cos(first[1] * radians) *
      Math.cos(second[1] * radians) *
      Math.sin(longitudeDifference / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(1, haversine)));
}

test('geographic DBSCAN preserves source rows, crosses the antimeridian and recomputes selected clusters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const longitudes = [179.999, -179.999, 179.9985, 0, 0.001, 0.002, 30, 181, null];
  const latitudes = [0, 0, 0, 10, 10, 10, 20, 0, 0];
  const batch = (start: number, end: number) =>
    new arrow.Table({
      longitude: arrow.vectorFromArray(longitudes.slice(start, end), new arrow.Float32()),
      latitude: arrow.makeVector(new Float32Array(latitudes.slice(start, end))),
      row: arrow.makeVector(Float32Array.from({length: end - start}, (_, index) => start + index))
    });
  const dataset = session.importArrowDataset({
    id: 'dbscan-source',
    table: batch(0, 4).concat(batch(4, 9))
  });
  const sourceChunks = [...dataset.value.columns.longitude.data];
  // Arrow uploads may use DynamicBuffer; retain the actual backing buffers for ownership checks.
  const sourceBuffers = sourceChunks.map(chunk =>
    chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer
  );
  const explorer = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const analysis = new AnalysisRuntime({session, explorer, requestUpdate() {}});
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer,
    sortMemoryBudgetBytes: 8 * 1024 * 1024,
    getDerivedColumns: () => analysis.getTableColumns()
  });
  const derivedBuffers = new Set<Buffer>();
  const render = async () => {
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    analysis.encode(encoder);
    device.submit(encoder.finish());
    await analysis.readSummary();
    expect(analysis.error).toBe('');
  };
  try {
    analysis.configureDBSCAN(settings);
    const savedRecipe = analysis.serializeRecipe();
    await render();
    expect(analysis.summary).toMatchObject({
      operation: 'dbscan-points',
      count: 7,
      dbscan: {clusterCount: 2, noiseCount: 1, validCount: 7, excludedCount: 2, complete: true}
    });
    const expectedLabels = [0, 0, 0, 1, 1, 1, -1, -1, -1];
    for (const [row, expected] of expectedLabels.entries())
      expect((await analysis.readRow(row))?.value).toBe(expected);
    const snapshot = analysis.getTableColumns()!;
    expect(Object.keys(snapshot.columns)).toHaveLength(5);
    expect(Object.keys(snapshot.columns).some(name => name.includes('ordering'))).toBe(false);
    for (const vector of Object.values(snapshot.columns)) {
      expect(vector.data.map(chunk => chunk.length)).toEqual([4, 5]);
      for (const chunk of vector.data) {
        if (!(chunk.buffer instanceof Buffer)) throw new Error('Expected resident derived buffer');
        derivedBuffers.add(chunk.buffer);
      }
    }
    const labelColumn = 'analysis_dbscan_label';
    const coreColumn = 'analysis_dbscan_core_distance_meters';
    const reachabilityColumn = 'analysis_dbscan_reachability_meters';
    const predecessorColumn = 'analysis_dbscan_predecessor_row';
    const page = await inspector.readPage({
      offset: 0,
      limit: 9,
      population: 'all',
      columns: [
        analysis.resultColumn,
        labelColumn,
        coreColumn,
        reachabilityColumn,
        predecessorColumn
      ]
    });
    expect(Array.from(page!.sourceRowIndices)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(page!.columns[analysis.resultColumn])).toEqual(expectedLabels);
    expect(Array.from(page!.validity[analysis.resultColumn])).toEqual([1, 1, 1, 1, 1, 1, 1, 0, 0]);
    expect(Array.from(page!.columns[labelColumn])).toEqual([
      0, 0, 0, 1, 1, 1, 0xffffffff, 0xffffffff, 0xffffffff
    ]);
    const coordinates = longitudes
      .slice(0, 7)
      .map((longitude, index) => [Math.fround(longitude!), Math.fround(latitudes[index])]);
    for (let row = 0; row < 6; row++) {
      const coreDistance = coordinates
        .map(point => sphericalMeters(coordinates[row], point))
        .sort((first, second) => first - second)[2];
      expect(Math.abs(page!.columns[coreColumn][row] - coreDistance)).toBeLessThan(1);
    }
    expect(page!.columns[coreColumn][6]).toBe(Infinity);
    expect(page!.columns[reachabilityColumn][0]).toBe(Infinity);
    expect(page!.columns[predecessorColumn][0]).toBe(0xffffffff);
    const colors = analysis.colors.data[0].buffer;
    if (!(colors instanceof Buffer)) throw new Error('Expected resident color buffer');
    expect(Array.from(await colors.readAsync(0, 9 * 4))).toEqual([
      31, 119, 180, 255, 31, 119, 180, 255, 31, 119, 180, 255, 255, 127, 14, 255, 255, 127, 14, 255,
      255, 127, 14, 255, 150, 150, 150, 255, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    analysis.setPopulation('selected');
    explorer.setRange('row', 1, 8);
    expect(snapshot.isCurrent()).toBe(false);
    await render();
    // Removing row0 leaves only two antimeridian samples, below self-inclusive minimumSamples.
    // The other cluster is rebuilt and now receives compact label0, not its former label1.
    expect(analysis.summary?.dbscan).toEqual({
      clusterCount: 1,
      noiseCount: 3,
      validCount: 6,
      excludedCount: 3,
      complete: true
    });
    for (const [row, expected] of [-1, -1, -1, 0, 0, 0, -1, -1, -1].entries())
      expect((await analysis.readRow(row))?.value).toBe(expected);
    expect(Array.from(await colors.readAsync(0, 4 * 4))).toEqual([
      0, 0, 0, 0, 150, 150, 150, 255, 150, 150, 150, 255, 31, 119, 180, 255
    ]);
    expect(analysis.pointWeightsCacheStats.buildCount).toBe(0);
    analysis.setMode('off');
    analysis.restoreRecipe(savedRecipe);
    await render();
    expect(analysis.summary?.dbscan?.clusterCount).toBe(2);
    expect((await analysis.readRow(3))?.value).toBe(1);
    expect(dataset.value.columns.longitude.data).toEqual(sourceChunks);
    expect(sourceChunks.map(chunk => chunk.length)).toEqual([4, 5]);
  } finally {
    const fence = device.createFence();
    try {
      await inspector.destroy();
      await analysis.destroy(fence.signaled);
      expect(sourceBuffers.every(buffer => !buffer.destroyed)).toBe(true);
      expect([...derivedBuffers].every(buffer => buffer.destroyed)).toBe(true);
    } finally {
      await session.destroy(fence.signaled);
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
