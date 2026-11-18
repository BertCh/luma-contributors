// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {mountResultView} from '../../../src/computation-views/views/result-view';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';

test('mounted geometry keeps drawable samples and replaces them when updated', () => {
  const element = document.createElement('div');
  const record = makeY2Record([
    makeY2Capture(
      'polygons',
      Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1]).flat(),
      {
        components: 2,
        role: 'output',
        geometry: {kind: 'polygons', coordinateModel: 'projected', offsetsCapture: 'offsets-id'}
      }
    ),
    makeY2Capture(
      'offsets',
      Array.from({length: 202}, (_, feature) => feature * 3)
    )
  ]);
  const mounted = mountResultView(element, record);
  try {
    expect(element.querySelectorAll('path[data-role="output"]')).toHaveLength(133);
    expect(element.textContent).toContain('133 of 201 polygons shown');
    mounted.update(
      makeY2Record([
        makeY2Capture('lines', [0, 0, 1, 1], {
          components: 2,
          role: 'output',
          geometry: {kind: 'lines', coordinateModel: 'projected', offsetsCapture: 'offsets-id'}
        }),
        makeY2Capture('offsets', [0, 2])
      ])
    );
    const paths = element.querySelectorAll('path[data-role="output"]');
    expect(paths).toHaveLength(1);
    expect(paths[0].getAttribute('d')).not.toContain('Z');
  } finally {
    mounted.destroy();
  }
  expect(element.childElementCount).toBe(0);
});

test('R1/R2 mounted polygon samples retain visible bounds with spare coordinate capacity', () => {
  const element = document.createElement('div');
  document.body.append(element);
  const coordinates = Array.from({length: 201}, (_, feature) => [
    feature * 3,
    0,
    feature * 3,
    1,
    feature * 3 + 2,
    0,
    feature * 3 + 1.75,
    0,
    feature * 3 + 1.5,
    0,
    feature * 3 + 1.25,
    0,
    feature * 3 + 1,
    0
  ]).flat();
  const record = makeY2Record([
    makeY2Capture('polygons', [...coordinates, 3.402823e38, 3.402823e38], {
      role: 'output',
      components: 2,
      geometry: {kind: 'polygons', coordinateModel: 'projected', offsetsCapture: 'offsets-id'}
    }),
    makeY2Capture(
      'offsets',
      Array.from({length: 202}, (_, feature) => feature * 7)
    )
  ]);
  const mounted = mountResultView(element, record);
  try {
    const paths = element.querySelectorAll<SVGPathElement>('path[data-role="output"]');
    expect(paths).toHaveLength(133);
    for (const path of paths) {
      const bounds = path.getBBox();
      expect(Number.isFinite(bounds.width) && Number.isFinite(bounds.height)).toBe(true);
      expect(bounds.width).toBeGreaterThan(0);
      expect(bounds.height).toBeGreaterThan(0);
    }
  } finally {
    mounted.destroy();
    element.remove();
  }
});
