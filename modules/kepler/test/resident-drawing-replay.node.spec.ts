// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Feature, Geometry} from 'geojson';
import {
  fitReplayBounds,
  planDrawingReplay,
  stepDrawingReplay
} from '../../../examples/deck/kepler-compute/src/resident-drawing-replay';

const VIEWPORT = {width: 1024, height: 768};
const SQUARE: Feature<Geometry> = {
  type: 'Feature',
  id: 's1',
  properties: {},
  geometry: {
    type: 'Polygon',
    coordinates: [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
      ]
    ]
  }
};
const POINT: Feature<Geometry> = {
  type: 'Feature',
  id: 'p1',
  properties: {},
  geometry: {type: 'Point', coordinates: [5, 5]}
};

test('replay plans shapes in order then annotations in creation order', () => {
  const steps = planDrawingReplay(
    [SQUARE, POINT],
    [
      {id: 'b', kind: 'POINT', anchorPoint: [2, 3]},
      {id: 'a', kind: 'CIRCLE', anchorPoint: [1, 1], radiusInMeters: 50_000},
      {id: 'hidden', kind: 'TEXT', anchorPoint: [0, 0], isVisible: false}
    ],
    VIEWPORT
  );
  expect(steps.map(step => step.id)).toEqual(['s1', 'p1', 'a', 'b']);
  expect(steps[0]).toMatchObject({longitude: 5, latitude: 5, kind: 'feature'});
  expect(steps[1].zoom).toBe(15);
  expect(steps[2].zoom).toBeGreaterThan(5);
});

test('fit zoom shrinks as the bounds grow', () => {
  const small = fitReplayBounds([0, 0, 1, 1], VIEWPORT).zoom;
  const large = fitReplayBounds([0, 0, 40, 40], VIEWPORT).zoom;
  expect(small).toBeGreaterThan(large);
  expect(large).toBeGreaterThanOrEqual(0);
});

test('replay transitions wrap, pause and stop at the last step', () => {
  let state = {index: 0, playing: false};
  state = stepDrawingReplay(state, 3, 'previous');
  expect(state.index).toBe(2);
  state = stepDrawingReplay(state, 3, 'next');
  expect(state.index).toBe(0);
  state = stepDrawingReplay(state, 3, 'play');
  state = stepDrawingReplay(state, 3, 'tick');
  state = stepDrawingReplay(state, 3, 'tick');
  expect(state).toEqual({index: 2, playing: true});
  state = stepDrawingReplay(state, 3, 'tick');
  expect(state).toEqual({index: 2, playing: false});
  expect(stepDrawingReplay(state, 0, 'play')).toEqual({index: 0, playing: false});
});
