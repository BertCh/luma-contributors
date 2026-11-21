// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CORE-103: the scaled edge cross product of a thin integer triangle cancels to zero when its two
// products are rounded separately, and the circle was reported invalid although the exact
// orientation is -1. The shared WGSL runs in the wgsl_reflect interpreter (float32 arithmetic).
// Expected centres are exact rationals (Python fractions):
//   bx, by, cx, cy = s, s - 1, s - 1, s - 2; d = 2 (bx cy - by cx) = -2
//   X = (cy (bx^2 + by^2) - by (cx^2 + cy^2)) / d, Y = (bx (cx^2 + cy^2) - cx (bx^2 + by^2)) / d
// s = 8192: (-67084289.5, 67100671.5), radius 94883096.563; s = 2^23: (-70368719011841.5,
// 70368735789055.5), radius 99516408656650.2.

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {TRIANGLE_CIRCUMCIRCLE_WGSL} from '../../../src/gpu-spatial-analysis/shared/triangle-circumcircle-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

/** Runs `getTriangleCircumcircle` once and returns `[orientation, radius, centerX, centerY, predicate]`. */
function runCircumcircle(positions: number[]): number[] {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const input = createTransientView(graph, 'positions', 'float32', 6);
    const output = createTransientView(graph, 'circle', 'float32', 5);
    const memory = createCPUGraphMemory();
    memory.write(input, Float32Array.from(positions));
    const node = createWGSLKernelNode(graph, {
      id: 'circle',
      operation: 'CircumcircleCancellation',
      invocationCount: 1,
      bindings: [
        {name: 'positions', view: input, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${EXACT_ORIENTATION_WGSL}\n${TRIANGLE_CIRCUMCIRCLE_WGSL}`,
      body: `let circle = getTriangleCircumcircle(vec2f(positions[0], positions[1]), vec2f(positions[2], positions[3]), vec2f(positions[4], positions[5]));
output[0] = f32(circle.orientation); output[1] = circle.radius; output[2] = circle.center.x; output[3] = circle.center.y; output[4] = f32(orientSign(vec2f(positions[0], positions[1]), vec2f(positions[2], positions[3]), vec2f(positions[4], positions[5])));`
    });
    executeCommandNodesOnCPU([node], memory);
    return Array.from(memory.readFloat32(output));
  } finally {
    device.destroy();
  }
}

function expectCircle(result: number[], center: [number, number], radius: number): void {
  expect(result).toHaveLength(5);
  expect(result.every(Number.isFinite)).toBe(true);
  expect(result[0]).toBe(-1);
  // Retain the established float32 tolerance for the thin integer fixtures; the dyadic
  // fixtures below use a tighter tolerance. Before correction the circle was absent.
  expect(Math.hypot(result[2] - center[0], result[3] - center[1])).toBeLessThan(radius * 2 ** -9);
  expect(Math.abs(result[1] - radius)).toBeLessThan(radius * 2 ** -9);
}

it('CORE-103 a thin integer triangle keeps its circle when the rounded cross products cancel', () => {
  expectCircle(
    runCircumcircle([0, 0, 8192, 8191, 8191, 8190]),
    [-67084289.5, 67100671.5],
    94883096.5634592
  );
  // (0,0), (2^23, 2^23 - 1), (2^23 - 1, 2^23 - 2) is checked by the transcription below and the
  // headless spec: the interpreter evaluates wide u32 products inexactly, so neither the integer
  // orientation fallback nor the exact product difference runs faithfully there at this size.
});

it('CORE-103 a well-conditioned triangle at the same scale is unchanged', () => {
  // Right triangle: centre is the hypotenuse midpoint (4096, 4095.5), radius hypot(4096, 4095.5).
  const result = runCircumcircle([0, 0, 8192, 0, 0, 8191]);
  expect(result).toHaveLength(5);
  expect(result.every(Number.isFinite)).toBe(true);
  expect(result[0]).toBe(1);
  expect(result[2]).toBe(4096);
  expect(result[3]).toBe(4095.5);
  expect(result[1]).toBe(Math.fround(Math.hypot(4096, 4095.5)));
});

it('CORE-103 the exact cross product is used only where the rounded products cancel', () => {
  const source = TRIANGLE_CIRCUMCIRCLE_WGSL.replace(/\s+/g, ' ');
  expect(source).toContain(
    'if (abs(cross) <= 0.0009765625 * (abs(leftProduct) + abs(rightProduct))) { cross = getTriangleProductDifference(firstEdge.x, secondEdge.y, firstEdge.y, secondEdge.x); }'
  );
});

// Transcription of getTriangleProductPart / getTriangleProductDifference with WGSL u32 semantics
// (multiply24 from EXACT_ORIENTATION_WGSL included), checked against BigInt arithmetic.
const toFloat32 = Math.fround;
function multiply24(first: number, second: number): [number, number] {
  const firstHigh = first >>> 12;
  const firstLow = first & 0xfff;
  const secondHigh = second >>> 12;
  const secondLow = second & 0xfff;
  const highProduct = Math.imul(firstHigh, secondHigh) >>> 0;
  const middle = (Math.imul(firstHigh, secondLow) + Math.imul(firstLow, secondHigh)) >>> 0;
  const lowProduct = Math.imul(firstLow, secondLow) >>> 0;
  const partial = (lowProduct + (((middle & 0xfffff) << 12) >>> 0)) >>> 0;
  const carryOne = partial < lowProduct ? 1 : 0;
  const low = (partial + ((highProduct << 24) >>> 0)) >>> 0;
  const carryTwo = low < partial ? 1 : 0;
  return [low, ((highProduct >>> 8) + (middle >>> 20) + carryOne + carryTwo) >>> 0];
}
const bitsScratch = new Float32Array(1);
const bitsView = new Uint32Array(bitsScratch.buffer);
function getBits(value: number): number {
  bitsScratch[0] = value;
  return bitsView[0];
}
type Part = {low: number; high: number; exponent: number; sign: number};
function getProductPart(first: number, second: number): Part {
  const firstBits = getBits(first);
  const secondBits = getBits(second);
  const firstMagnitude = firstBits & 0x7fffffff;
  const secondMagnitude = secondBits & 0x7fffffff;
  if (firstMagnitude === 0 || secondMagnitude === 0) return {low: 0, high: 0, exponent: 0, sign: 2};
  const firstExponentBits = firstMagnitude >>> 23;
  const secondExponentBits = secondMagnitude >>> 23;
  const firstSignificand =
    firstExponentBits === 0 ? firstMagnitude & 0x7fffff : 0x800000 | (firstMagnitude & 0x7fffff);
  const secondSignificand =
    secondExponentBits === 0 ? secondMagnitude & 0x7fffff : 0x800000 | (secondMagnitude & 0x7fffff);
  const firstExponent = firstExponentBits === 0 ? -149 : firstExponentBits - 150;
  const secondExponent = secondExponentBits === 0 ? -149 : secondExponentBits - 150;
  const [low, high] = multiply24(firstSignificand, secondSignificand);
  return {
    low,
    high,
    exponent: firstExponent + secondExponent,
    sign: (firstBits ^ secondBits) >>> 31
  };
}
function getProductDifference(first: number, second: number, third: number, fourth: number) {
  const rounded = toFloat32(toFloat32(first * second) - toFloat32(third * fourth));
  const left = getProductPart(first, second);
  const right = getProductPart(third, fourth);
  if (left.sign === 2 || right.sign === 2 || left.sign !== right.sign) return rounded;
  const shift = left.exponent - right.exponent;
  if (shift > 16 || shift < -16) return rounded;
  const exponent = Math.min(left.exponent, right.exponent);
  if (exponent < -250 || exponent > 100) return rounded;
  let {low: leftLow, high: leftHigh} = left;
  let {low: rightLow, high: rightHigh} = right;
  if (shift > 0) {
    leftHigh = ((leftHigh << shift) | (leftLow >>> (32 - shift))) >>> 0;
    leftLow = (leftLow << shift) >>> 0;
  } else if (shift < 0) {
    rightHigh = ((rightHigh << -shift) | (rightLow >>> (32 + shift))) >>> 0;
    rightLow = (rightLow << -shift) >>> 0;
  }
  const leftLarger = leftHigh > rightHigh || (leftHigh === rightHigh && leftLow >= rightLow);
  const [largeLow, largeHigh] = leftLarger ? [leftLow, leftHigh] : [rightLow, rightHigh];
  const [smallLow, smallHigh] = leftLarger ? [rightLow, rightHigh] : [leftLow, leftHigh];
  const differenceLow = (largeLow - smallLow) >>> 0;
  const differenceHigh = (largeHigh - smallHigh - (largeLow < smallLow ? 1 : 0)) >>> 0;
  const scaled = toFloat32(
    toFloat32(toFloat32(differenceHigh) * 4294967296) + toFloat32(differenceLow)
  );
  const magnitude = toFloat32(scaled * 2 ** exponent);
  const negativeResult = left.sign === 1 ? leftLarger : !leftLarger;
  return negativeResult ? -magnitude : magnitude;
}
/** Exact value of a float32 as a BigInt numerator over 2^200. */
function toScaledBigInt(value: number): bigint {
  const bits = getBits(value);
  const exponentBits = (bits >>> 23) & 0xff;
  const significand = exponentBits === 0 ? bits & 0x7fffff : 0x800000 | (bits & 0x7fffff);
  const exponent = (exponentBits === 0 ? -149 : exponentBits - 150) + 100;
  const magnitude = BigInt(significand) << BigInt(exponent);
  return bits >>> 31 ? -magnitude : magnitude;
}
function getExactDifference(first: number, second: number, third: number, fourth: number): number {
  const numerator =
    toScaledBigInt(first) * toScaledBigInt(second) - toScaledBigInt(third) * toScaledBigInt(fourth);
  return Number(numerator) / 2 ** 200;
}

it('CORE-103 transcribed exact product difference recovers cancelled cross products', () => {
  // Scaled edges of the two review triangles (power-of-two scaling is exact).
  expect(getProductDifference(0.5, 0.999755859375, 0.49993896484375, 0.9998779296875)).toBe(
    -(2 ** -27)
  );
  const half = 0.5;
  const nearHalf = toFloat32(0.4999999403953552);
  const nearOne = toFloat32(0.9999998807907104);
  const nearOneLower = toFloat32(0.9999997615814209);
  const roundedCross = toFloat32(toFloat32(half * nearOneLower) - toFloat32(nearHalf * nearOne));
  expect(roundedCross === 0).toBe(true);
  expect(getProductDifference(half, nearOneLower, nearHalf, nearOne)).toBe(-(2 ** -47));
  // Random cancelling pairs: within a few float32 units of the exact difference.
  let seed = 0x9e3779b9;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  for (let trial = 0; trial < 20000; trial++) {
    const first = toFloat32((random() < 0.5 ? -1 : 1) * (0.5 + random() / 2));
    const second = toFloat32((random() < 0.5 ? -1 : 1) * 2 ** (-40 * random()));
    const third = toFloat32(first * (1 + (random() - 0.5) * 2 ** -20));
    const fourth = toFloat32((first * second) / third);
    const exact = getExactDifference(first, second, third, fourth);
    const result = getProductDifference(first, second, third, fourth);
    expect(Number.isFinite(result)).toBe(true);
    if (exact === 0) {
      expect(result === 0).toBe(true);
    } else {
      expect(Math.sign(result)).toBe(Math.sign(exact));
      expect(Math.abs(result - exact)).toBeLessThanOrEqual(Math.abs(exact) * 2 ** -21);
    }
  }
});

// Unlike the integer fixtures, these edges lose their distinct directions during subtraction,
// before multiplication. Exact algebra with u = 2^-26 gives determinant u^2 - 2u < 0,
// center X = Y = (2 - u^2)/(4 - 2u) and radius sqrt(2)*(1-X).
// The middle cyclic permutation reaches orientExact in wgsl_reflect, whose wide integer
// accumulator is inexact (as with the 2^23 integer triangle above). All three are device cases.
it.each([
  0, 2
])('CORE-103 dyadic edges retain their circle through cyclic permutation %s', rotation => {
  const unit = 2 ** -26;
  const positions = [1, 1, unit, 0, 0, unit];
  const center = 0.5000000037252903;
  const radius = 0.7071067759181915;
  const offset = rotation * 2;
  const result = runCircumcircle([...positions.slice(offset), ...positions.slice(0, offset)]);
  expect(result).toHaveLength(5);
  expect(result.every(Number.isFinite)).toBe(true);
  expect(result[4]).toBe(-1);
  expect(result[0]).toBe(-1);
  expect(Math.abs(result[2] - center)).toBeLessThan(2 ** -22);
  expect(Math.abs(result[3] - center)).toBeLessThan(2 ** -22);
  expect(Math.abs(result[1] - radius)).toBeLessThan(2 ** -22);
});

// Independent exact-coordinate circumcenter (solve 2*(B-A).center=|B|^2-|A|^2
// and 2*(C-A).center=|C|^2-|A|^2 with Python fractions); the rounded cross is nonzero.
// Rotation 1 reaches the same unsupported interpreter accumulator as the symmetric fixture;
// it remains covered by the device test with the unchanged composed predicate.
it.each([
  0, 2
])('B2 asymmetric dyadic circle retains its center in cyclic rotation %s', rotation => {
  const positions = [1, 1, 0, 2 ** -26, 2 ** -24, 0];
  const offset = rotation * 2;
  const result = runCircumcircle([...positions.slice(offset), ...positions.slice(0, offset)]);
  expect(result).toHaveLength(5);
  expect(result.every(Number.isFinite)).toBe(true);
  expect(result[4]).toBe(1);
  expect(result[0]).toBe(1);
  expect(Math.abs(result[2] - 0.2000000247359275)).toBeLessThan(2 ** -22);
  expect(Math.abs(result[3] - 0.7999999871850011)).toBeLessThan(2 ** -22);
  expect(Math.abs(result[1] - 0.8246211042342516)).toBeLessThan(2 ** -22);
});
