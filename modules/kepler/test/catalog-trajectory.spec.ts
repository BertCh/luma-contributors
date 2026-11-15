// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

const ORIGIN = Date.UTC(2026, 0, 1);
function observations(
  coordinates: number[][],
  times = coordinates.map((_, index) => ORIGIN + index * 1000),
  scalarValues = coordinates.map((_, index) => index),
  trackId = 'a'
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {track_id: trackId, timestamp: times[index], value: scalarValues[index]}
    }))
  };
}
async function runGpu(
  id: string,
  tracks: CatalogFeatureCollection,
  parameters: Record<string, unknown> = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding(id)!;
  const result = await runCatalogOperationAsync(
    binding,
    {coordinateSystem: 'planar', ...parameters},
    {tracks},
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.dispatchedContributors).toEqual(binding.gpuContributorNames);
  return result;
}

test('GPU Hampel uses the real contributor, preserves track boundaries and publishes filtered scalars', async () => {
  const tracks = observations(
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0]
    ],
    undefined,
    [0, 0, 100, 0, 0]
  );
  tracks.features.push(
    ...observations(
      [
        [10, 0],
        [11, 0]
      ],
      undefined,
      [50, 60],
      'b'
    ).features
  );
  const result = await runGpu('trajectory-hampel', tracks, {target: 'property', windowSize: 3});
  expect(result.tables[0].columns['filtered_value']).toEqual([0, 0, 0, 0, 0, 50, 60]);
  expect(result.tables[0].columns['outlier']).toEqual([
    false,
    false,
    true,
    false,
    false,
    false,
    false
  ]);
  expect(result.tables[1].rowCount).toBe(2);
  expect(result.tables[1].columns['source_indices']).toEqual(['[0,1,2,3,4]', '[5,6]']);
});

test('GPU Hampel coordinate target replaces a displaced coordinate in native timed paths', async () => {
  const tracks = observations([
    [0, 0],
    [0, 0],
    [100, 0],
    [0, 0],
    [0, 0]
  ]);
  const result = await runGpu('trajectory-hampel', tracks, {target: 'x', windowSize: 3});
  expect(JSON.parse(result.tables[1].columns['geometry'][0] as string).coordinates[2]).toEqual([
    0,
    0,
    0,
    ORIGIN + 2000
  ]);
});

test('GPU outlier flags mark the displaced fix and retain valid successors in cleaned tracks', async () => {
  const result = await runGpu(
    'trajectory-outlier-flags',
    observations([
      [0, 0],
      [1, 0],
      [100, 0],
      [3, 0],
      [4, 0]
    ]),
    {maximumSpeed: 10}
  );
  expect(result.tables[0].columns['outlier_flags']).toEqual([0, 0, 1, 0, 0]);
  expect(result.tables[1].columns['source_indices']).toEqual(['[0,1,3,4]']);
  expect(
    JSON.parse(result.tables[1].columns['geometry'][0] as string).coordinates.map(
      (coordinate: number[]) => coordinate[3]
    )
  ).toEqual([ORIGIN, ORIGIN + 1000, ORIGIN + 3000, ORIGIN + 4000]);
});

test('GPU relative times subtract large absolute epochs exactly per track and retain source identity', async () => {
  const tracks = observations(
    [
      [2, 0],
      [0, 0],
      [1, 0]
    ],
    [ORIGIN + 2, ORIGIN, ORIGIN + 1]
  );
  tracks.features.push(
    ...observations(
      [
        [10, 0],
        [11, 0]
      ],
      [ORIGIN + 100000, ORIGIN + 100001],
      undefined,
      'b'
    ).features
  );
  const result = await runGpu('trajectory-track-relative-times', tracks);
  expect(result.tables[0].columns['relative_milliseconds']).toEqual([0, 1, 2, 0, 1]);
  expect(result.tables[0].columns['source_index']).toEqual([1, 2, 0, 3, 4]);
  expect(result.tables[0].columns['timestamp']).toEqual([
    ORIGIN,
    ORIGIN + 1,
    ORIGIN + 2,
    ORIGIN + 100000,
    ORIGIN + 100001
  ]);
});

test('GPU relative times refuse errors beyond the declared millisecond budget', async () => {
  await expect(
    runGpu(
      'trajectory-track-relative-times',
      observations(
        [
          [0, 0],
          [1, 0]
        ],
        [ORIGIN, ORIGIN + 16777217]
      )
    )
  ).rejects.toThrow(/maximumTimeErrorMilliseconds/);
  const result = await runGpu(
    'trajectory-track-relative-times',
    observations(
      [
        [0, 0],
        [1, 0]
      ],
      [ORIGIN, ORIGIN + 16777217]
    ),
    {maximumTimeErrorMilliseconds: 1}
  );
  expect(result.tables[0].columns['relative_milliseconds']).toEqual([0, 16777216]);
});

test('GPU isopleths include the crossing cell and stable ties, then publish polygon cells', async () => {
  const result = await runGpu(
    'trajectory-isopleths',
    observations([
      [0, 0],
      [0, 0]
    ]),
    {
      gridWidth: 2,
      gridHeight: 2,
      padding: 1,
      motionVariance: 0,
      locationError: 1,
      subStepCount: 1,
      level: 0.4
    }
  );
  const table = result.tables[0];
  expect(table.rowCount).toBe(2);
  expect(table.columns['cell_id']).toEqual([0, 1]);
  expect(table.columns['volume']).toEqual([0.25, 0.5]);
  expect(table.columns['inside']).toEqual([true, true]);
  for (const geometry of table.columns['geometry'])
    expect(JSON.parse(geometry as string).type).toBe('Polygon');
});

test('GPU isopleths publish no home range for a zero-mass grid', async () => {
  const result = await runGpu(
    'trajectory-isopleths',
    observations([
      [0, 0],
      [0, 0]
    ]),
    {gridWidth: 2, gridHeight: 2, motionVariance: 0, locationError: 0}
  );
  expect(result.tables[0].rowCount).toBe(0);
});
