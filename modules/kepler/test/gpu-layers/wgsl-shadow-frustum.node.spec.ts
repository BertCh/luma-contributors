// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeViewport as GlobeViewport, OrbitViewport} from '@deck.gl/core';
import {Matrix4, Vector3} from '@math.gl/core';
import {expect, test} from 'vitest';
import {
  getWGSLShadowFramePoints,
  makeWGSLShadowProjection,
  mergeWGSLShadowLayerBounds,
  WGSL_SHADOW_GLOBE_RADIUS
} from '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow-frustum';

function projectToNdc(matrix: number[], point: number[]): number[] {
  const clip = new Matrix4(matrix).transform([...point, 1]);
  return [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
}

const lightView = () => new Matrix4().lookAt({eye: new Vector3([-1, 0, -1]).negate()});

test('orbit viewports fit the light frustum to explicit scene bounds', () => {
  const viewport = new OrbitViewport({
    width: 400,
    height: 300,
    orbitAxis: 'Z',
    rotationOrbit: 30,
    rotationX: 40,
    zoom: 0
  });
  const points = getWGSLShadowFramePoints(viewport as never, {
    globe: false,
    sceneBounds: [
      [-100, -100, 0],
      [100, 100, 50]
    ]
  });
  expect(points?.source).toBe('scene-bounds');
  const projection = makeWGSLShadowProjection(lightView(), points!);
  for (const corner of [
    [-100, -100, 0],
    [100, 100, 50],
    [100, -100, 25]
  ]) {
    const ndc = projectToNdc(projection.viewProjectionMatrix, viewport.projectPosition(corner));
    expect(Math.abs(ndc[0])).toBeLessThanOrEqual(1.0001);
    expect(Math.abs(ndc[1])).toBeLessThanOrEqual(1.0001);
    expect(ndc[2]).toBeGreaterThanOrEqual(-1.0001);
    expect(ndc[2]).toBeLessThanOrEqual(1.0001);
  }
});

test('orbit viewports without bounds clamp the far plane instead of using it', () => {
  const viewport = new OrbitViewport({width: 400, height: 300, zoom: 0});
  const points = getWGSLShadowFramePoints(viewport as never, {globe: false});
  expect(points?.source).toBe('view-frustum');
  const farthest = Math.max(
    ...points!.depthPoints.map(point =>
      Math.hypot(...point.map((value, axis) => value - viewport.cameraPosition[axis]))
    )
  );
  const target = Math.hypot(
    ...viewport.cameraPosition.map((value, axis) => value - viewport.center[axis])
  );
  expect(farthest).toBeLessThan(target * 2.2);
});

test('globe viewports fit the light frustum to the visible sphere', () => {
  const viewport = new GlobeViewport({
    width: 400,
    height: 300,
    longitude: 10,
    latitude: 20,
    zoom: 1
  });
  const points = getWGSLShadowFramePoints(viewport as never, {globe: true});
  expect(points?.source).toBe('globe');
  expect(points!.extentPoints.length).toBeGreaterThan(10);
  for (const point of points!.extentPoints) {
    expect(Math.hypot(...point)).toBeCloseTo(WGSL_SHADOW_GLOBE_RADIUS, 3);
  }
  const projection = makeWGSLShadowProjection(lightView(), points!);
  for (const point of points!.extentPoints) {
    const ndc = projectToNdc(projection.viewProjectionMatrix, point);
    expect(Math.abs(ndc[0])).toBeLessThanOrEqual(1.0001);
    expect(Math.abs(ndc[1])).toBeLessThanOrEqual(1.0001);
    expect(ndc[2]).toBeGreaterThanOrEqual(-1.0001);
    expect(ndc[2]).toBeLessThanOrEqual(1.0001);
  }
});

test('layer bounds merge ignores layers without bounds', () => {
  expect(mergeWGSLShadowLayerBounds([{getBounds: () => null}, {}])).toBeNull();
  expect(
    mergeWGSLShadowLayerBounds([
      {
        getBounds: () => [
          [0, 0, 0],
          [1, 2, 3]
        ]
      },
      {
        getBounds: () => [
          [-1, 1],
          [0, 5]
        ]
      },
      {}
    ])
  ).toEqual([
    [-1, 0, 0],
    [1, 5, 3]
  ]);
});
