// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WebMercatorViewport} from '@deck.gl/core';
import {applyScenegraphWGSLShadow} from '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow-scenegraph';
import {
  applyWGSLShadow,
  getWGSLShadowUniforms,
  wgslShadow
} from '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow';
import {
  makeWGSLShadowMatrices,
  normalizeWGSLShadowColor
} from '../../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow-effect';
import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';

test('normalizeWGSLShadowColor accepts 0-255 and 0-1 colors', () => {
  expect(normalizeWGSLShadowColor([255, 0, 0, 0.5])).toEqual([1, 0, 0, 0.5]);
  expect(normalizeWGSLShadowColor([0.5, 0.25, 0, 0.8])).toEqual([0.5, 0.25, 0, 0.8]);
  expect(normalizeWGSLShadowColor(undefined)[3]).toBeCloseTo(200 / 255);
});

test('makeWGSLShadowMatrices builds at most two light view matrices', () => {
  const matrices = makeWGSLShadowMatrices([
    {direction: [-1, 3, -1]},
    {direction: [1, -8, -2.5]},
    {direction: [0, 0, -1]}
  ]);
  expect(matrices).toHaveLength(2);
  expect(matrices[0]).toHaveLength(16);
});

test('shadow module uniforms are disabled without a viewport and enabled with maps', () => {
  const fallback = {id: 'fallback'} as never;
  const disabled = getWGSLShadowUniforms({shadowEnabled: false, dummyShadowMap: fallback});
  expect(disabled).toMatchObject({drawShadowMap: 0, useShadowMap: 0, shadowMap0: fallback});
  const viewport = new WebMercatorViewport({
    width: 200,
    height: 100,
    longitude: 0,
    latitude: 0,
    zoom: 12,
    pitch: 40
  });
  const mapOne = {id: 'one'} as never;
  const enabled = getWGSLShadowUniforms({
    project: {viewport},
    shadowMatrices: makeWGSLShadowMatrices([{direction: [-1, 3, -1]}]) as never,
    shadowMaps: [mapOne],
    dummyShadowMap: fallback,
    shadowColor: [0, 0, 0, 0.5]
  });
  expect(enabled).toMatchObject({
    drawShadowMap: 0,
    useShadowMap: 1,
    lightCount: 1,
    shadowMap0: mapOne,
    shadowMap1: fallback,
    color: [0, 0, 0, 0.5]
  });
  expect((enabled['viewProjectionMatrix0'] as number[]).length).toBe(16);
  expect((enabled['projectCenter0'] as number[]).length).toBe(4);
  const drawing = getWGSLShadowUniforms({
    project: {viewport},
    shadowMatrices: makeWGSLShadowMatrices([{direction: [-1, 3, -1]}]) as never,
    drawToShadowMap: true,
    shadowLightId: 0,
    dummyShadowMap: fallback
  });
  expect(drawing).toMatchObject({drawShadowMap: 1, shadowMap0: fallback});
});

test('shadow module declares the uniform block, two maps and the layer entry points', () => {
  const source = (wgslShadow as {source: string}).source;
  for (const name of [
    'var<uniform> shadow:',
    'shadowMap0: texture_2d<f32>',
    'shadowMap1: texture_2d<f32>',
    'fn shadow_setVertexPosition',
    'fn shadow_getVertexPosition',
    'fn shadow_filterPremultiplied'
  ])
    expect(source).toContain(name);
});

test('applyWGSLShadow rewrites the clip position, varyings and final color', () => {
  const source = [
    'struct Out {',
    '  @location(0) color: vec4<f32>,',
    '};',
    'fn v() { output.position = projected.clipPosition; }',
    'fn f() { return vec4<f32>(color.rgb * alpha, alpha); }'
  ].join('\n');
  const patched = applyWGSLShadow(source, {
    varyingAnchor: '@location(0) color: vec4<f32>,',
    firstLocation: 1,
    positionAssignment: 'output.position = projected.clipPosition;',
    commonPosition: 'projected.commonPosition',
    clipPosition: 'projected.clipPosition',
    finalReturn: 'return vec4<f32>(color.rgb * alpha, alpha);',
    finalColorExpression: 'vec4<f32>(color.rgb, alpha)'
  });
  expect(patched).toContain('@location(1) shadowPosition0: vec3<f32>');
  expect(patched).toContain('@location(2) shadowPosition1: vec3<f32>');
  expect(patched).toContain(
    'output.position = shadow_setVertexPosition(projected.commonPosition, projected.clipPosition);'
  );
  expect(patched).toContain(
    'output.shadowPosition1 = shadow_getVertexPosition(1u, projected.commonPosition);'
  );
  expect(patched).toContain(
    'return shadow_filterPremultiplied(vec4<f32>(color.rgb, alpha), input.position.z'
  );
});

test('applyScenegraphWGSLShadow patches Deck 9.4 scenegraph WGSL at every anchor', () => {
  const deckSource = readFileSync(
    'node_modules/@deck.gl/mesh-layers/dist/scenegraph-layer/scenegraph-layer.wgsl.js',
    'utf8'
  );
  const patched = applyScenegraphWGSLShadow(deckSource);
  expect(patched).toContain('shadowPosition0: vec3<f32>');
  expect(patched).toContain('outputs.position = shadow_setVertexPosition(geometry.position');
  expect(patched).toContain('return shadow_filterPremultiplied(fragColor, inputs.position.z');
});
