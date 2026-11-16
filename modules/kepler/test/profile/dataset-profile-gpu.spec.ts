// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {createKeplerDatasetProfileCache} from '../../src/profile/dataset-profile-cache';
import {createKeplerDatasetProfileStages} from '../../src/profile/dataset-profile-stages';
import {createKeplerDatasetProfileGpuRunner} from '../../src/profile/dataset-profile-gpu';
import type {
  KeplerDatasetProfile,
  KeplerDatasetProfileProgress
} from '../../src/profile/dataset-profile-types';

function metadata(rowCount: number, columnCount: number): KeplerDatasetProfile {
  return {
    datasetId: 'resident',
    label: 'Resident GPU',
    revision: 1,
    kind: 'points',
    catalogGeometry: 'point',
    rowCount,
    vertexCount: rowCount,
    geometryColumns: [],
    coordinates: {model: 'planar', source: 'declared', reason: 'Fixture local linear units'},
    columns: Array.from({length: columnCount}, (_, column) => ({
      name: `value${column}`,
      type: 'number',
      rowCount,
      nullCount: 0,
      sampleSize: 0
    }))
  };
}
function upload(device: Device, values: Float32Array | Uint32Array) {
  return device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST});
}
function check(condition: boolean, message: string, report: unknown): void {
  if (!condition) throw new Error(`${message}\n${JSON.stringify(report)}`);
}

it('profiles one million resident points and eight columns in stages, recording timing without thresholds', async context => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 1_000_000;
  const side = 1000;
  // Nonzero offsets guard the resident slice contract, including the two-component position view.
  const points = new Float32Array(rowCount * 2 + 4);
  for (let row = 0; row < rowCount; row++) {
    points[row * 2 + 4] = row % side;
    points[row * 2 + 5] = Math.floor(row / side);
  }
  const positions = upload(device, points);
  const columns = Array.from({length: 8}, (_, column) => {
    const values = new Float32Array(rowCount + 4);
    for (let row = 0; row < rowCount; row++) values[row + 4] = (row % 16) + column;
    return {
      name: `value${column}`,
      buffer: upload(device, values),
      length: rowCount,
      byteOffset: 16
    };
  });
  const allocated = vi.spyOn(device, 'createBuffer');
  try {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 2});
    cache.setDeclarations('resident', {measurement: {value0: 'amount'}, rowSemantics: 'objects'});
    const progress: KeplerDatasetProfileProgress[] = [];
    const stages = createKeplerDatasetProfileStages({
      cache,
      onProgress: event => progress.push(event)
    });
    const job = stages.start({
      metadata: metadata(rowCount, 8),
      getRevision: () => 1,
      runner: createKeplerDatasetProfileGpuRunner({
        device,
        rowCount,
        columns,
        positions: {buffer: positions, length: rowCount, byteOffset: 16},
        positionsCoverage: 'all-locations'
      })
    });
    check(
      progress.length === 1 &&
        progress[0].stage === 'metadata' &&
        progress[0].profile?.columns[0].sampleSize === 0,
      'Metadata was not synchronous',
      progress
    );
    const result = await job.finished;
    const report = {progress, result};
    check(Boolean(result), 'Profile did not complete', report);
    result!.columns.forEach((column, index) => {
      check(
        column.minimum === index &&
          column.maximum === index + 15 &&
          column.rowCount === rowCount &&
          column.nullCount === 0 &&
          column.sampleSize === undefined,
        'Column full-population statistics mismatch',
        report
      );
      check(
        Math.abs(column.mean! - (7.5 + index)) < 1e-4 &&
          Math.abs(column.standardDeviation! - Math.sqrt(21.25)) < 1e-4,
        'Column moments mismatch',
        report
      );
      check(
        column.distinctCountIsEstimate === true &&
          column.distinctCountLowerBound! <= 16 &&
          column.distinctCountUpperBound! >= 16,
        'Distinct guarantees mismatch',
        report
      );
    });
    check(
      result!.columns[0].measurement === 'amount' && result!.rowSemantics?.value === 'objects',
      'Declarations were lost',
      report
    );
    const spacing = result!.spacing;
    check(
      spacing?.pointCount === rowCount &&
        spacing.sampled === false &&
        spacing.sampleSize === rowCount &&
        spacing.duplicateLocationCount === 0,
      'Spacing population mismatch',
      report
    );
    check(
      Math.abs(spacing!.center[0] - 499.5) < 1e-3 &&
        Math.abs(spacing!.center[1] - 499.5) < 1e-3 &&
        Math.abs(spacing!.standardDistance - Math.sqrt((side * side - 1) / 6)) < 1e-2,
      'Spacing center/spread mismatch',
      report
    );
    check(
      spacing!.nearestNeighbor?.minimum === 1 &&
        spacing!.nearestNeighbor?.median === 1 &&
        spacing!.nearestNeighbor?.maximum === 1 &&
        spacing!.kDistance?.distances.length === 512,
      'Indexed neighbour statistics mismatch',
      report
    );
    check(progress.at(-1)?.stage === 'complete', 'Missing completion progress', report);
    await context.annotate(
      JSON.stringify({
        rows: rowCount,
        columns: 8,
        timings: progress.at(-1)?.timings,
        elapsedMilliseconds: progress.at(-1)?.elapsedMilliseconds
      }),
      'AX-20 staged GPU timings'
    );
    const owned = allocated.mock.results
      .filter(result => result.type === 'return')
      .map(result => result.value as Buffer);
    check(
      owned.length > 0 && owned.every(buffer => buffer.destroyed),
      'Stage buffers survived job completion',
      owned.map(buffer => ({id: buffer.id, destroyed: buffer.destroyed}))
    );
    check(
      !positions.destroyed && columns.every(column => !column.buffer.destroyed),
      'Borrowed resident buffers were destroyed',
      report
    );
  } finally {
    allocated.mockRestore();
    positions.destroy();
    columns.forEach(column => column.buffer.destroy());
  }
}, 180_000);

it('drains submitted GPU columns on cancellation and retains borrowed buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffer = upload(device, new Float32Array([1, 2, 3, 4]));
  const allocated = vi.spyOn(device, 'createBuffer');
  try {
    const runner = createKeplerDatasetProfileGpuRunner({
      device,
      rowCount: 4,
      columns: [{name: 'value0', buffer, length: 4}]
    });
    const controller = new AbortController();
    const spacing = vi.fn();
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    const stages = createKeplerDatasetProfileStages({cache});
    const job = stages.start({
      metadata: metadata(4, 1),
      getRevision: () => 1,
      signal: controller.signal,
      runner: {
        columns: context => {
          const pending = runner.columns!(context);
          controller.abort();
          return pending;
        },
        spacing
      }
    });
    expect(await job.finished).toBeUndefined();
    expect(spacing).not.toHaveBeenCalled();
    check(
      allocated.mock.results
        .filter(result => result.type === 'return')
        .every(result => (result.value as Buffer).destroyed),
      'Cancelled GPU work leaked buffers',
      {}
    );
    expect(buffer.destroyed).toBe(false);
  } finally {
    allocated.mockRestore();
    buffer.destroy();
  }
});

it('profiles both locations of a single MultiPoint row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffer = upload(device, new Float32Array([0, 0, 10, 0]));
  try {
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    const source = {...metadata(1, 0), vertexCount: 2, extent: [0, 0, 10, 0] as const};
    const result = await createKeplerDatasetProfileStages({cache}).start({
      metadata: source,
      getRevision: () => 1,
      runner: createKeplerDatasetProfileGpuRunner({
        device,
        rowCount: 1,
        columns: [],
        positions: {buffer, length: 2},
        positionsCoverage: 'all-locations'
      })
    }).finished;
    check(
      result?.rowCount === 1 &&
        result.spacing?.pointCount === 2 &&
        result.spacing.nearestNeighbor?.maximum === 10,
      'MultiPoint location coverage mismatch',
      result
    );
    expect(result?.extent).toEqual([0, 0, 10, 0]);
    expect(buffer.destroyed).toBe(false);
  } finally {
    buffer.destroy();
  }
});

it.each([
  8, 1
])('decodes queen counts including islands and refuses overflow at capacity %s', async neighborCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = upload(
    device,
    new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1, 4, 0, 5, 0, 5, 1, 4, 1])
  );
  const ringOffsets = upload(device, new Uint32Array([0, 4, 8, 12]));
  const polygonOffsets = upload(device, new Uint32Array([0, 1, 2, 3]));
  const featureOffsets = upload(device, new Uint32Array([0, 1, 2, 3]));
  const allocated = vi.spyOn(device, 'createBuffer');
  try {
    const source = {
      ...metadata(3, 0),
      kind: 'polygons' as const,
      catalogGeometry: 'polygon' as const,
      vertexCount: 12
    };
    const cache = createKeplerDatasetProfileCache({maximumEntries: 1});
    const job = createKeplerDatasetProfileStages({cache}).start({
      metadata: source,
      getRevision: () => 1,
      runner: createKeplerDatasetProfileGpuRunner({
        device,
        rowCount: 3,
        columns: [],
        polygons: {
          positions: {buffer: positions, length: 12},
          ringOffsets: {buffer: ringOffsets, length: 4},
          polygonOffsets: {buffer: polygonOffsets, length: 4},
          featureOffsets: {buffer: featureOffsets, length: 4},
          neighborCapacity,
          pairCapacity: 32
        }
      })
    });
    if (neighborCapacity === 1) {
      await expect(job.finished).rejects.toThrow(/overflow/);
      expect(cache.get('resident', 1)?.contiguity).toBeUndefined();
    } else {
      const result = await job.finished;
      check(
        JSON.stringify(result?.contiguity) ===
          JSON.stringify({minimum: 0, median: 1, maximum: 1, islandCount: 1}),
        'Queen contiguity mismatch',
        result
      );
    }
    check(
      allocated.mock.results
        .filter(result => result.type === 'return')
        .every(result => (result.value as Buffer).destroyed),
      'Contiguity buffers leaked',
      {}
    );
    check(
      [positions, ringOffsets, polygonOffsets, featureOffsets].every(buffer => !buffer.destroyed),
      'Borrowed polygons destroyed',
      {}
    );
  } finally {
    allocated.mockRestore();
    [positions, ringOffsets, polygonOffsets, featureOffsets].forEach(buffer => buffer.destroy());
  }
});

it('decodes normalized numeric and category nulls and leaves unsupported metadata intact', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const numeric = upload(device, new Float32Array([1, 2, NaN, 4]));
  const category = upload(device, new Uint32Array([0, 1, 0, 0xffffffff]));
  try {
    const source = metadata(4, 1);
    source.columns = [
      ...source.columns,
      {name: 'category', type: 'category', rowCount: 4, nullCount: 0},
      {name: 'text', type: 'text', rowCount: 4, nullCount: 0, sampleSize: 0}
    ];
    const runner = createKeplerDatasetProfileGpuRunner({
      device,
      rowCount: 4,
      columns: [
        {name: 'value0', buffer: numeric, length: 4},
        {name: 'category', buffer: category, length: 4, categoryCount: 2}
      ]
    });
    const columns = await runner.columns!({profile: source, signal: new AbortController().signal});
    check(
      columns[0].nullCount === 1 &&
        Math.abs(columns[0].mean! - 7 / 3) < 1e-5 &&
        columns[1].nullCount === 1 &&
        columns[1].minimum === undefined &&
        columns[2] === source.columns[2],
      'Normalized column decoding mismatch',
      columns
    );
  } finally {
    numeric.destroy();
    category.destroy();
  }
});
