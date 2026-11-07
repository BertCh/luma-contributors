// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
import {
  SANDBOX_SAMPLES,
  createSandboxSample
} from '../../../examples/deck/kepler-compute/src/sandbox-samples';
import {prepareSandboxTrips} from '../../../examples/deck/kepler-compute/src/trip-source-options';
import {
  preparePublishedGeometry,
  preparePublishedH3
} from '../../../examples/deck/kepler-compute/src/source-publication';

for (const description of SANDBOX_SAMPLES) {
  test(`sample ${description.id} survives Arrow transport and normal source preparation`, () => {
    const sample = createSandboxSample(description.id);
    const original = sample.table;
    const restored = arrow.tableFromIPC(arrow.tableToIPC(original, 'stream'));
    const repeated = createSandboxSample(description.id).table;
    expect(original.batches).toHaveLength(2);
    expect(restored.batches.map(batch => batch.numRows)).toEqual(
      original.batches.map(batch => batch.numRows)
    );
    expect([...restored.getChild('source_row_id')!]).toEqual(
      Array.from({length: original.numRows}, (_, row) => row)
    );
    for (const field of original.schema.fields) {
      expect([...restored.getChild(field.name)!]).toEqual([...original.getChild(field.name)!]);
      expect([...repeated.getChild(field.name)!]).toEqual([...original.getChild(field.name)!]);
    }
    const metadata = JSON.parse(restored.schema.metadata.get('luma:sandbox-sample')!);
    expect(metadata.id).toBe(description.id);
    expect(metadata.synthetic).toBe(true);
    const prepared = prepareArrowTableForGPUAnalytics(restored, {
      columns: restored.schema.fields
        .map(field => field.name)
        .filter(name => name !== sample.geometryColumn && name !== sample.h3Column)
    });
    expect(prepared.table.numRows).toBe(original.numRows);
    expect(prepared.table.batches.map(batch => batch.numRows)).toEqual(
      original.batches.map(batch => batch.numRows)
    );
    if (sample.h3Column) {
      const cells = preparePublishedH3(restored, sample.h3Column);
      expect(metadata.h3Column).toBe(sample.h3Column);
      expect(cells.validRowCount).toBe(4);
      expect(cells.nullRowCount).toBe(2);
      expect(cells.invalidRows).toEqual([]);
      expect(restored.getChild(sample.h3Column)!.type.toString()).toBe('Uint64');
      expect(restored.getChild(sample.h3Column)!.get(1)).toBe(BigInt('0x8928308280fffff'));
      expect(cells.batches.map(batch => batch.rowCount)).toEqual([3, 3]);
    } else if (sample.tripOptions?.grouping && sample.geometryColumn) {
      const trips = prepareSandboxTrips(restored, sample.geometryColumn, sample.tripOptions);
      expect(trips.tripCount).toBe(8);
      expect(trips.vertexCount).toBe(48);
      expect(trips.invalidRows).toEqual([]);
      // Every trip starts in batch 0 and spans both source batches without repacking them.
      expect(trips.batches.map(batch => Array.from(batch.sourceRows))).toEqual([
        [0, 1, 2, 3, 4, 5, 6, 7],
        []
      ]);
      expect(trips.batches.map(batch => batch.rowCount)).toEqual([24, 24]);
      expect(Array.from(trips.batches[0].timestamps.subarray(0, 6))).toEqual([
        0, 120000, 240000, 360000, 480000, 600000
      ]);
    } else if (sample.tripOptions && sample.geometryColumn) {
      const trips = prepareSandboxTrips(restored, sample.geometryColumn, sample.tripOptions);
      expect(metadata.tripOptions).toEqual(sample.tripOptions);
      expect(trips.tripCount).toBe(8);
      expect(trips.vertexCount).toBe(48);
      expect(trips.actualTimeErrorMilliseconds).toBe(0);
      expect(trips.invalidRows).toEqual([]);
      expect(trips.batches.map(batch => Array.from(batch.sourceRows))).toEqual([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
      ]);
      expect(trips.batches.map(batch => batch.rowCount)).toEqual([4, 4]);
      expect(trips.batches[0].sourceTimestamps[0]).toBe(1767225600);
    } else if (sample.geometryColumn)
      expect(() => preparePublishedGeometry(restored, sample.geometryColumn!)).not.toThrow();
    if (sample.coordinates) {
      for (const name of Object.values(sample.coordinates)) {
        expect(prepared.table.getChild(name)?.type.toString()).toBe('Float32');
        expect(prepared.table.getChild(name)?.nullCount).toBe(0);
      }
    }
  });
}

test('time samples retain exact UTC ticks and monotone four-coordinate trip paths', () => {
  const points = createSandboxSample('points-time').table;
  const timestamp = points.getChild('timestamp')!;
  expect(timestamp.type.toString()).toBe('Timestamp<MILLISECOND, UTC>');
  expect(timestamp.data.flatMap(data => Array.from(data.values, String))).toEqual(
    Array.from({length: 256}, (_, row) => String(1767225600000 + row * 900000))
  );
  expect(points.getChild('reading')!.nullCount).toBeGreaterThan(0);
  const trips = createSandboxSample('trips').table;
  for (let row = 0; row < trips.numRows; row++) {
    const coordinates = JSON.parse(trips.getChild('geometry')!.get(row)).coordinates;
    expect(coordinates).toHaveLength(6);
    expect(coordinates.every(position => position.length === 4)).toBe(true);
    expect(coordinates.map(position => position[3])).toEqual(
      Array.from({length: 6}, (_, vertex) => 1767225600 + row * 120 + vertex * 120)
    );
    expect(Number(trips.getChild('departure')!.get(row))).toBe(coordinates[0][3] * 1000);
  }
});

test('spatial samples declare topology, row-major grid shape and directed asymmetric costs', () => {
  const districts = createSandboxSample('districts').table;
  expect(JSON.parse(districts.getChild('geometry')!.get(5)).coordinates).toHaveLength(2);
  expect(JSON.parse(districts.getChild('geometry')!.get(10)).type).toBe('MultiPolygon');
  const grid = createSandboxSample('numeric-grid').table;
  expect(JSON.parse(grid.schema.metadata.get('luma:sandbox-sample')!).grid).toEqual({
    width: 16,
    height: 16,
    order: 'row-major'
  });
  expect(grid.numRows).toBe(256);
  expect(grid.getChild('value')!.nullCount).toBe(3);
  for (let row = 0; row < grid.numRows; row++)
    expect(
      Number(grid.getChild('grid_row')!.get(row)) * 16 +
        Number(grid.getChild('grid_column')!.get(row))
    ).toBe(row);
  const network = createSandboxSample('directed-network').table;
  expect(network.numRows).toBe(48);
  const edges = new Map<string, number>();
  for (let row = 0; row < network.numRows; row++) {
    const source = Number(network.getChild('source_node')!.get(row));
    const target = Number(network.getChild('target_node')!.get(row));
    const cost = Number(network.getChild('cost_seconds')!.get(row));
    expect(cost).toBeGreaterThan(0);
    edges.set(`${source}:${target}`, cost);
  }
  expect(edges.size).toBe(48);
  for (const [edge, cost] of edges) {
    const [source, target] = edge.split(':');
    expect(edges.has(`${target}:${source}`)).toBe(true);
    expect(edges.get(`${target}:${source}`)).not.toBe(cost);
  }
});
