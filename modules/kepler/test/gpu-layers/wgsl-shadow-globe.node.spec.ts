// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeViewport as GlobeViewport} from '@deck.gl/core';
import {Matrix4, Vector3} from '@math.gl/core';
import {expect, test} from 'vitest';
import {
  getWGSLShadowFramePoints,
  makeWGSLShadowProjection,
  WGSL_SHADOW_GLOBE_METERS_PER_UNIT
} from '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow-frustum';

function projectToNdc(matrix: number[], point: number[]): number[] {
  const clip = new Matrix4(matrix).transform([...point, 1]);
  return [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
}

function getTexelsPerMeter(extentSize: [number, number], width: number, height: number): number {
  return (
    Math.min(width / extentSize[0], height / extentSize[1]) / WGSL_SHADOW_GLOBE_METERS_PER_UNIT
  );
}

test('aspect-fitted globe projections keep the visible cap inside the map', () => {
  const viewport = new GlobeViewport({
    width: 800,
    height: 400,
    longitude: 30,
    latitude: 40,
    zoom: 3,
    pitch: 30,
    bearing: 50
  });
  const points = getWGSLShadowFramePoints(viewport as never, {globe: true})!;
  const lightView = new Matrix4().lookAt({eye: new Vector3([-1, 1.6, -0.3]).negate()});
  const projection = makeWGSLShadowProjection(lightView, points, {aspectRatio: 2});
  for (const point of points.extentPoints) {
    const ndc = projectToNdc(projection.viewProjectionMatrix, point);
    expect(Math.abs(ndc[0])).toBeLessThanOrEqual(1.0001);
    expect(Math.abs(ndc[1])).toBeLessThanOrEqual(1.0001);
    expect(ndc[2]).toBeGreaterThanOrEqual(-1.0001);
    expect(ndc[2]).toBeLessThanOrEqual(1.0001);
  }
});

test('rotating the light axes and a 2048 budget raise globe texels per meter', () => {
  const width = 512;
  const height = 256;
  const viewport = new GlobeViewport({width, height, longitude: 0, latitude: 0, zoom: 2.2});
  const points = getWGSLShadowFramePoints(viewport as never, {globe: true})!;
  const lightView = new Matrix4().lookAt({eye: new Vector3([-1, 1.6, 0]).negate()});
  const legacy = makeWGSLShadowProjection(lightView, points);
  const fitted = makeWGSLShadowProjection(lightView, points, {aspectRatio: width / height});
  const before = getTexelsPerMeter(legacy.extentSize, width, height);
  const afterFittedOnly = getTexelsPerMeter(fitted.extentSize, width, height);
  const afterBudget = getTexelsPerMeter(fitted.extentSize, 2048, 1024);
  // The rotation search never needs a larger map than the unrotated fit.
  expect(afterFittedOnly).toBeGreaterThanOrEqual(before * 0.999);
  // 4x the texels per axis at 512 x 256 gives 4x the density before any rotation gain.
  expect(afterBudget).toBeGreaterThanOrEqual(before * 3.99);
  expect(before).toBeGreaterThan(0);
});

test('a 300 m caster shadows on a zoomed globe while the shell-wide depth range stays valid', async () => {
  const {getWGSLShadowOcclusion} = await import(
    '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow'
  );
  const viewport = new GlobeViewport({
    width: 800,
    height: 600,
    longitude: -122.4,
    latitude: 37.8,
    zoom: 15
  });
  const points = getWGSLShadowFramePoints(viewport as never, {globe: true})!;
  const direction = new Vector3([-1, 1.6, -0.3]).normalize();
  const lightView = new Matrix4().lookAt({eye: new Vector3(direction).negate()});
  const projection = makeWGSLShadowProjection(lightView, points, {
    aspectRatio: 800 / 600,
    mapSize: [2048, 1536],
    depthBias: 2 / WGSL_SHADOW_GLOBE_METERS_PER_UNIT
  });
  // The frustum depth covers the whole shell: the legacy 0.1% bias would be thousands of meters.
  const [depthRange, bias] = projection.depthParams;
  expect(depthRange * 0.001 * WGSL_SHADOW_GLOBE_METERS_PER_UNIT).toBeGreaterThan(5000);
  expect(bias * WGSL_SHADOW_GLOBE_METERS_PER_UNIT).toBeLessThan(50);

  // Occluder at 300 m altitude, receiver 300 m of path further along the light.
  const occluder = viewport.projectPosition([-122.4, 37.8, 300]);
  const pathUnits = 300 / WGSL_SHADOW_GLOBE_METERS_PER_UNIT;
  const receiver = occluder.map((value, axis) => value + direction[axis] * pathUnits);
  const depthDifference =
    projectToNdc(projection.viewProjectionMatrix, receiver)[2] -
    projectToNdc(projection.viewProjectionMatrix, occluder)[2];
  expect(depthDifference).toBeGreaterThan(0);
  expect(getWGSLShadowOcclusion(depthDifference, projection.depthParams)).toBeGreaterThan(0.9);
  // The legacy normalized bias renders no shadow for the same pair.
  expect(getWGSLShadowOcclusion(depthDifference)).toBe(0);
  // Self-shadowing within the bias stays lit.
  expect(getWGSLShadowOcclusion(depthDifference / 300, projection.depthParams)).toBe(0);
});
