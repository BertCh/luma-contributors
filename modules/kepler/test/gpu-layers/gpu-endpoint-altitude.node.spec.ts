// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUArcLayer,
  GPUArrowHeadLayer,
  GPULineLayer
} from '../../../../examples/deck/kepler-compute/gpu-layers/index';
import {applyEndpointAltitudeToShader} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-endpoint-altitude';
import {GPU_ARROW_HEAD_SHADER} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-arrow-head-layer';
import type {LayerContext} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {expect, test, vi} from 'vitest';

const SHADER = `
@vertex fn vertexMain(
  @location(0) sourcePositions: vec2<f32>,
  @location(1) targetPositions: vec2<f32>,
  @builtin(vertex_index) vertexIndex: u32
) -> VertexOutput {
  let a = vec3<f32>(sourcePositions, 0.0) + vec3<f32>(targetPositions, 0.0);
}`;

test('float32x2 endpoints keep vec2 inputs and zero altitude', () => {
  const shader = applyEndpointAltitudeToShader(SHADER, 'float32x2', 'float32x2');
  expect(shader).toContain('@location(0) sourcePositions: vec2<f32>');
  expect(shader).toContain('let sourceAltitude = 0.0;');
  expect(shader).toContain('let sourceLongitudeLatitude = sourcePositions;');
  expect(shader).toContain('vec3<f32>(sourceLongitudeLatitude, 0.0)');
});

test('float32x3 endpoints get vec3 inputs, split altitude and the scale expression', () => {
  const shader = applyEndpointAltitudeToShader(SHADER, 'float32x3', 'float32x2', 'style.scale');
  expect(shader).toContain('@location(0) sourcePositions: vec3<f32>');
  expect(shader).toContain('@location(1) targetPositions: vec2<f32>');
  expect(shader).toContain('let sourceAltitude = sourcePositions.z * style.scale;');
  expect(shader).toContain('let targetAltitude = 0.0;');
});

test('shaders without endpoint inputs are rejected', () => {
  expect(() => applyEndpointAltitudeToShader('fn x() {}', 'float32x2', 'float32x2')).toThrow(
    'Endpoint altitude requires endpoint inputs'
  );
});

test('arrow head shader transforms for mixed endpoint formats', () => {
  const shader = applyEndpointAltitudeToShader(
    GPU_ARROW_HEAD_SHADER,
    'float32x3',
    'float32x3',
    'arrowStyle.elevationScale'
  );
  expect(shader).toContain('vec3<f32>(targetLongitudeLatitude, targetAltitude)');
  expect(shader).toContain('previousGlobe.z += mix(sourceAltitude, targetAltitude');
});

for (const [name, Layer] of [
  ['line', GPULineLayer],
  ['arc', GPUArcLayer],
  ['arrow head', GPUArrowHeadLayer]
] as const) {
  test(`${name} layer rejects non-position endpoint formats and accepts float32x3`, () => {
    const bad = new Layer({
      getSourcePosition: makeVector('float32x2', 1),
      getTargetPosition: makeVector('float32', 1)
    } as never);
    expect(() => bad.initializeState({device: {type: 'webgpu'}} as LayerContext)).toThrow(
      /must be one of float32x2, float32x3/
    );
    const device = {
      type: 'webgpu',
      limits: {minUniformBufferOffsetAlignment: 256},
      createBuffer: () => {
        throw new Error('format accepted');
      }
    };
    const good = new Layer({
      getSourcePosition: makeVector('float32x3', 1),
      getTargetPosition: makeVector('float32x2', 1)
    } as never);
    expect(() => good.initializeState({device} as unknown as LayerContext)).toThrow(
      'format accepted'
    );
  });
}

function makeVector<Format extends 'float32' | 'float32x2' | 'float32x3'>(
  format: Format,
  length: number
): GPUVector<Format> {
  const byteLength = length * (format === 'float32' ? 4 : format === 'float32x2' ? 8 : 12);
  return new GPUVector({
    type: 'data',
    format,
    data: [
      new GPUData({
        format,
        length,
        buffer: {byteLength, destroy: vi.fn()} as unknown as Buffer
      })
    ]
  });
}
