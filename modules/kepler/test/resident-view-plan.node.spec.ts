// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  createResidentViewLayerFilter,
  getResidentSubViewProjection,
  planResidentViews
} from '../../../examples/deck/kepler-compute/src/resident-view-plan';

const layers = [
  {id: 'a', isVisible: true},
  {id: 'b', isVisible: true},
  {id: 'c', isVisible: false}
];

test('single view lists enabled layers and refuses unsupported split requests', () => {
  expect(planResidentViews({layers})).toEqual({
    supported: true,
    views: [{index: 0, visibleLayerIds: ['a', 'b']}]
  });
  expect(planResidentViews({layers, splitMaps: [{}, {}], mapSplitMode: 'DUAL_MAP'})).toMatchObject({
    supported: false
  });
});

test('two views share one plan and differ only by native per-map layer visibility', () => {
  expect(
    planResidentViews({
      layers,
      supportedViewCount: 2,
      mapSplitMode: 'DUAL_MAP',
      splitMaps: [{layers: {a: true, c: true}}, {layers: {a: true, b: true}}]
    })
  ).toEqual({
    supported: true,
    views: [
      {index: 0, visibleLayerIds: ['a']},
      {index: 1, visibleLayerIds: ['a', 'b']}
    ]
  });
  expect(
    planResidentViews({
      layers,
      supportedViewCount: 2,
      splitMaps: [{}, {}, {}],
      mapSplitMode: 'DUAL_MAP'
    })
  ).toMatchObject({supported: false});
  expect(
    planResidentViews({
      layers,
      supportedViewCount: 2,
      splitMaps: [{}, {}],
      mapSplitMode: 'SINGLE_MAP'
    })
  ).toMatchObject({supported: false});
});

test('layer filter confines resident layers to their views and leaves overlays visible', () => {
  const filter = createResidentViewLayerFilter(
    [
      {index: 0, visibleLayerIds: ['a']},
      {index: 1, visibleLayerIds: ['a', 'b']}
    ],
    id => id.replace(/-fill$/, '')
  );
  expect(filter({layer: {id: 'b-fill'}, viewport: {id: 'view-0'}})).toBe(false);
  expect(filter({layer: {id: 'b-fill'}, viewport: {id: 'view-1'}})).toBe(true);
  expect(filter({layer: {id: 'a'}, viewport: {id: 'view-0'}})).toBe(true);
  expect(filter({layer: {id: 'editor'}, viewport: {id: 'view-1'}})).toBe(true);
});

test('swipe plan admits SWIPE_COMPARE with per-side layer visibility', () => {
  expect(
    planResidentViews({
      layers,
      supportedViewCount: 2,
      mapSplitMode: 'SWIPE_COMPARE',
      splitMaps: [{layers: {a: true}}, {layers: {b: true}}]
    })
  ).toEqual({
    supported: true,
    views: [
      {index: 0, visibleLayerIds: ['a']},
      {index: 1, visibleLayerIds: ['b']}
    ]
  });
});

test('sub-view projection maps the full-canvas clip space onto the view rectangle', () => {
  const full = [2, 0, 0, 0, 0, 3, 0, 0, 0.25, 0.5, -1.5, -1, 0, 0, -2, 0];
  const canvasWidth = 800;
  const clipX = (matrix: number[], point: number[]) =>
    matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
  const clipW = (matrix: number[], point: number[]) =>
    matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
  const point = [0.3, -0.2, -5];
  const fullNdc = clipX(full, point) / clipW(full, point);
  const screenX = ((fullNdc + 1) / 2) * canvasWidth;
  for (const [x, width] of [
    [0, 200],
    [200, 600],
    [640, 160]
  ]) {
    const sub = getResidentSubViewProjection(full, canvasWidth, x, width);
    const subNdc = clipX(sub, point) / clipW(sub, point);
    expect(x + ((subNdc + 1) / 2) * width).toBeCloseTo(screenX, 6);
    expect(sub[5]).toBe(full[5]);
  }
});
