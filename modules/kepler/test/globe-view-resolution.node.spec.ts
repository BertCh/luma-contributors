// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeView as GlobeView, _GlobeViewport as GlobeViewport} from '@deck.gl/core';
import {expect, test} from 'vitest';
import {
  convertGlobeDegreesToIntervals,
  withoutGlobeViewClear
} from '../../../examples/deck/kepler-compute/src/main-globe-layers';

test('Kepler globe view resolution in degrees becomes deck 10 intervals', () => {
  expect(convertGlobeDegreesToIntervals(5)).toBe(72);
  const keplerView = new GlobeView({id: 'globe', resolution: 5, clear: true, clearColor: false});
  const rebuilt = withoutGlobeViewClear(keplerView, 'spec') as GlobeView;
  expect(rebuilt.props.resolution).toBe(72);
  expect(rebuilt.props.clearColor).toBeUndefined();
  // A second pass leaves the rebuilt view alone.
  expect(withoutGlobeViewClear(rebuilt, 'spec')).toBe(rebuilt);
  const viewport = rebuilt.makeViewport({
    width: 100,
    height: 100,
    viewState: {longitude: 0, latitude: 0, zoom: 1}
  });
  expect(viewport!.resolution).toBe(72);
});

test('a globe view without resolution is left alone', () => {
  const view = new GlobeView({id: 'globe'});
  expect(withoutGlobeViewClear(view, 'spec-plain')).toBe(view);
});

test.each([
  [0, 0],
  [-30, 20],
  [170, -45]
])('deck 10 globe keeps east right and north up at %s, %s', (longitude, latitude) => {
  const viewport = new GlobeViewport({width: 800, height: 600, longitude, latitude, zoom: 2});
  const center = viewport.project([longitude, latitude]);
  expect(center[0]).toBeCloseTo(400, 7);
  expect(center[1]).toBeCloseTo(300, 7);
  expect(viewport.project([longitude + 1, latitude])[0]).toBeGreaterThan(center[0]);
  expect(viewport.project([longitude - 1, latitude])[0]).toBeLessThan(center[0]);
  expect(viewport.project([longitude, latitude + 1])[1]).toBeLessThan(center[1]);
  expect(viewport.unproject(viewport.project([longitude + 1, latitude + 1]))).toEqual(
    expect.arrayContaining([expect.closeTo(longitude + 1, 6), expect.closeTo(latitude + 1, 6)])
  );
});
