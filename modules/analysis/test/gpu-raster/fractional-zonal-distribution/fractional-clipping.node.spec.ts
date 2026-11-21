// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser, ScalarData} from 'wgsl_reflect';
import {getRasterFractionalClippingWGSL} from '../../../src/raster-analysis/fractional-coverage/raster-fractional-coverage-index';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';

for (const affine of [
  [1, 1, 0, 1, 1 + 3 * 2 ** -25, 0],
  [1, 1, 0, 1 + 3 * 2 ** -25, 1, 0],
  [1 + 2 ** -23, 1, 0, 1 + 2 ** -22, 1 + 2 ** -23, 0],
  [2 ** -67, 2 ** -66, 0, 3 * 2 ** -67, 2 ** -65, 0]
] as const) {
  it(`W5-12 inverts the packed basis despite determinant cancellation: ${affine.join(',')}`, () => {
    const settings = getGPURasterFractionalCoverageParameterValues({
      width: 1,
      height: 1,
      affine,
      pixelInterpretation: 'area'
    });
    const positions = Float32Array.of(settings[2], settings[4], settings[3], settings[5]);
    const output = new Float32Array(5);
    const clipping = getRasterFractionalClippingWGSL('area').replace(
      'select(pixel,nearest,abs(pixel-nearest)<=tolerance)',
      'vec2<f32>(select(pixel.x,nearest.x,abs(pixel.x-nearest.x)<=tolerance.x),select(pixel.y,nearest.y,abs(pixel.y-nearest.y)<=tolerance.y))'
    );
    const source = `
@group(0) @binding(0) var<storage, read> settings: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const settingsOffset:u32=0u; const positionsOffset:u32=0u;
${clipping}
@compute @workgroup_size(1) fn main() {
 let first=point(0u); output[0]=first.x; output[1]=first.y;
 let second=point(1u); output[2]=second.x; output[3]=second.y;
 output[4]=abs(output[0]*output[3]-output[1]*output[2])*0.5;
}`;
    const executor = new WgslExec(new WgslParser().parse(source));
    // Model Metal's permitted f32 reciprocal-multiply lowering, including overflow.
    const evaluateBinary = executor._evalBinaryOp.bind(executor);
    executor._evalBinaryOp = (node, context) => {
      const result = evaluateBinary(node, context);
      if (node.operator === '/' && result instanceof ScalarData && result.typeInfo.name === 'f32') {
        const numerator = executor.evalExpression(node.left, context);
        const denominator = executor.evalExpression(node.right, context);
        if (numerator instanceof ScalarData && denominator instanceof ScalarData) {
          result.value = Math.fround(
            Math.fround(numerator.value) * Math.fround(1 / denominator.value)
          );
        }
      }
      return result;
    };
    executor.dispatchWorkgroups('main', 1, {
      0: {0: settings, 1: positions, 2: output}
    });
    // An inverse maps its own columns to the unit basis; their triangle has area 1/2.
    // Second determinant is (1+u)^2-(1+2u)=u^2=2^-46, lost by plain f32 products.
    expect(Array.from(output)).toEqual([1, 0, 0, 1, 0.5]);
  });
}

it('W5-12 accepts a legacy area determinant different from the packed-axis determinant', () => {
  expect(
    transformRectangle(
      0,
      [0, 0, 1, 1, 1, 1 + 2 ** -23, 3 * 2 ** -25, 0],
      [1, 1, 1, 1 + 2 ** -23],
      true
    )
  ).toBe(1);
});

it('review S3 scaling preserves nonfinite per-frame axis rejection', () => {
  // An infinite first axis cannot define a finite inverse. Normalizing it must not turn
  // exponent bits into a finite coefficient when the other product has exponent 100.
  expect(transformRectangle(0, [0, 0, Infinity, 1, 2 ** 100, 1, 1, 0], [1, 1, 2, 2])).toBeNaN();
});

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
