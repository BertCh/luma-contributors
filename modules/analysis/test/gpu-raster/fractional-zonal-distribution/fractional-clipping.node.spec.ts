// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getRasterFractionalClippingWGSL} from '../../../src/raster-analysis/fractional-coverage/raster-fractional-coverage-index';

function transformRectangle(
  column: number,
  affine: number[],
  coordinates?: number[],
  leftOnly = false,
  pixelInterpretation: 'area' | 'point' = 'area'
) {
  const positions = Float32Array.from(coordinates ?? [column + 0.125, 0, column + 0.375, 1]);
  const output = new Float32Array(1);
  // wgsl_reflect does not implement vector-bool select; expand only that builtin componentwise.
  const clipping = getRasterFractionalClippingWGSL(pixelInterpretation).replace(
    'select(pixel,nearest,abs(pixel-nearest)<=tolerance)',
    'vec2<f32>(select(pixel.x,nearest.x,abs(pixel.x-nearest.x)<=tolerance.x),select(pixel.y,nearest.y,abs(pixel.y-nearest.y)<=tolerance.y))'
  );
  const source = `
@group(0) @binding(0) var<storage, read> settings: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const settingsOffset:u32=0u; const positionsOffset:u32=0u;
${clipping}
@compute @workgroup_size(1)
fn main() {
  let left = point(0u); let right = point(1u);
  output[0] = ${leftOnly ? 'left.x' : '(right.x-left.x)*(right.y-left.y)'};
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: Float32Array.from(affine), 1: positions, 2: output}
  });
  return output[0];
}

it('preserves exactly representable quarter-cell support at pixel 2^20 on the identity grid', () => {
  const identity = [0, 0, 1, 0, 0, 1, 1, 0];
  expect(transformRectangle(0, identity)).toBe(0.25);
  expect(transformRectangle(2 ** 20, identity)).toBe(0.25);
});

it('R5 snaps decimal point-grid edges to half-integer boundaries', () => {
  expect(
    transformRectangle(0, [0, 0, 0.3, 0, 0, 1, 0.3, 0], [0.45, -0.5, 0.75, 0.5], false, 'point')
  ).toBe(1);
});

it('R8 exposes translated decimal slivers beyond the absolute snap cap', () => {
  const left = transformRectangle(0, [0, 0, 0.3, 0, 0, 1, 0.3, 0], [39.3, 0, 39.6, 1], true);
  expect(left).toBe(130.99998474121094);
  expect(131 - left).toBeGreaterThan(1e-5);
});

it('bounds nonidentity snapping in subpixel units and retains decimal boundary snapping', () => {
  expect(transformRectangle(2 ** 20, [0, 0, 2, 0, 0, 1, 2, 0])).toBe(0.125);
  expect(transformRectangle(0, [0, 0, 0.3, 0, 0, 0.7, 0.21, 0], [0.9, 0.7, 1.2, 1.4])).toBe(1);
});
