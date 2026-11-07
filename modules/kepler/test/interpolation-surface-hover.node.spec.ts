// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  InterpolationSurfaceHoverStore,
  getInterpolationSurfaceTooltipId,
  makeInterpolationSurfaceProvider
} from '../../../examples/deck/kepler-compute/src/interpolation-surface-hover';
import {getInterpolationCellIndex} from '../../../examples/deck/kepler-compute/src/interpolation-tooltip';
import {readTooltipValues} from '../../../examples/deck/kepler-compute/src/resident-tooltip-fields';

const grid = {bounds: [0, 0, 10, 10] as const, width: 10, height: 10};

test('surface hover maps the pointer location to the north-up cell and reads kriging fields', async () => {
  const cellIndex = getInterpolationCellIndex(grid, 2.5, 9.5);
  expect(cellIndex).toBe(2);
  const reads: number[][] = [];
  const provider = makeInterpolationSurfaceProvider({
    settings: {mode: 'kriging-points', column: 'value'},
    readInterpolationPage: async (offset, limit) => {
      reads.push([offset, limit ?? 16]);
      return [
        {
          cellIndex: offset,
          longitude: 2.5,
          latitude: 9.5,
          value: 4.25,
          variance: 0.5,
          lower: 3,
          upper: 5.5,
          confidenceLevel: 0.95
        }
      ];
    }
  });
  const values = await provider(getInterpolationSurfaceTooltipId('d'), 'l', cellIndex!);
  expect(reads).toEqual([[2, 1]]);
  expect(values?.['Interpolated value']).toBe(4.25);
  expect(values?.['Kriging variance']).toBe(0.5);
  expect(values?.['Kriging interval lower (0.95)']).toBe(3);
  expect(values?.['Cell center longitude']).toBe(2.5);
});

test('surface hover reports IDW neighbors and treats a stale surface as no values', async () => {
  const idw = makeInterpolationSurfaceProvider({
    settings: {mode: 'idw-points', column: 'value'},
    readInterpolationPage: async offset => [
      {cellIndex: offset, longitude: 1, latitude: 2, value: null, neighbors: 7}
    ]
  });
  const values = await idw('d', 'l', 0);
  expect(values?.['Interpolation neighbors']).toBe(7);
  expect(values?.['Interpolated value']).toBeNull();
  const stale = makeInterpolationSurfaceProvider({
    settings: {mode: 'idw-points', column: 'value'},
    readInterpolationPage: () => Promise.reject(new Error('stale'))
  });
  expect(await stale('d', 'l', 0)).toBeNull();
  const state = await readTooltipValues({
    getRevision: () => 'r',
    readOriginal: async () => ({}),
    readComputed: () => stale('d', 'l', 0)
  });
  expect(state.status).toBe('empty');
});

test('hover store follows the pointer in a cell, replaces cells, and clears when disabled', () => {
  const store = new InterpolationSurfaceHoverStore();
  let notifications = 0;
  store.subscribe(() => notifications++);
  expect(store.setPick({surfaceId: 'd', cellIndex: 3, x: 1, y: 2})).toBe(true);
  const revision = store.getState()!.revision;
  store.setPick({surfaceId: 'd', cellIndex: 3, x: 5, y: 6});
  expect(store.getState()).toMatchObject({x: 5, y: 6, revision});
  store.setPick({surfaceId: 'd', cellIndex: 4, x: 5, y: 6});
  expect(store.getState()!.revision).toBeGreaterThan(revision);
  store.setInspectionEnabled(false);
  expect(store.getState()).toBeNull();
  expect(store.setPick(null)).toBe(false);
  expect(notifications).toBeGreaterThan(2);
});
