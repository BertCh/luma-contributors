// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {ScalarData, WgslExec, WgslParser} from 'wgsl_reflect';
import * as accumulator from '../../src/analysis-core/kernels/fixed-point-accumulator';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {decodeMomentProbe} from './fixed-point-moments.test-utils';

it('VAL-06 preflights the caller prop before writes', () => {
  for (const value of [-1, 0.5, NaN, Infinity, 4294967296]) {
    expect(() =>
      accumulator.validateFixedPointMaximumTermCount('raster', 'maximumStreamSliceCount', value)
    ).toThrow(/raster.*maximumStreamSliceCount/);
  }
  for (const value of [0, 1, 4294967295]) {
    expect(() =>
      accumulator.validateFixedPointMaximumTermCount('raster', 'maximumStreamSliceCount', value)
    ).not.toThrow();
  }
});

function getMomentIncrement(
  priorSum: bigint,
  priorCount: number,
  value: number,
  sumUnitExponent: number,
  momentUnitExponent: number
): number {
  expect(accumulator.FIXED_POINT_MOMENTS_WGSL).toBeTypeOf('string');
  const words = Array.from({length: 3}, (_, index) =>
    Number((BigInt.asUintN(96, priorSum) >> BigInt(32 * index)) & 0xffffffffn)
  );
  const output = new Uint32Array(3);
  // wgsl_reflect 1.2.3 does not implement ldexp. These fixtures use normal, exactly
  // representable powers of two; multiplication by exp2 is exact for this test-only shim.
  const source = `${accumulator.FIXED_POINT_WGSL.replaceAll('ldexp(', 'testLdexp(')}
${accumulator.FIXED_POINT_MOMENTS_WGSL}
fn testLdexp(fraction: f32, exponent: i32) -> f32 { return fraction * exp2(f32(exponent)); }
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() {
  let increment = getFixedPointMomentIncrement(
    vec3<u32>(${words.map(word => `${word}u`).join(', ')}), ${priorCount}u,
    ${value.toExponential()}, ${sumUnitExponent}, ${momentUnitExponent});
  output[0] = increment.x; output[1] = increment.y; output[2] = increment.z;
}`;
  const execution = new WgslExec(new WgslParser().parse(source));
  // Its generic vec3<u32> constructor also stores words in Float32Array. Select the
  // equivalent alias in its type metadata so vectors retain uint32 words on this instance.
  const getTypeInfo = execution.getTypeInfo.bind(execution);
  execution.getTypeInfo = type => {
    const information = getTypeInfo(type);
    const format = (information as {format?: {name: string}} | null)?.format;
    if (information?.name === 'vec3' && format?.name === 'u32') information.name = 'vec3u';
    return information;
  };
  // The same interpreter uses signed JS >> for WGSL u32 >>. Correct that primitive
  // on this instance only; all production word arithmetic and quantization still execute.
  const evaluateBinary = execution._evalBinaryOp.bind(execution);
  execution._evalBinaryOp = (node, context) => {
    if (node.operator === '>>') {
      const left = execution.evalExpression(node.left, context);
      const right = execution.evalExpression(node.right, context);
      if (
        left instanceof ScalarData &&
        right instanceof ScalarData &&
        left.typeInfo.name === 'u32'
      ) {
        return new ScalarData(left.value >>> right.value, left.typeInfo);
      }
    }
    return evaluateBinary(node, context);
  };
  execution.dispatchWorkgroups('main', 1, {0: {0: output}});
  const raw = BigInt(output[0]) | (BigInt(output[1]) << 32n) | (BigInt(output[2]) << 64n);
  return Number(BigInt.asIntN(96, raw)) * 2 ** momentUnitExponent;
}

it('SCL-01 reproduces the old float32 cancellation against a hand-worked half-unit moment', () => {
  const legacy = (first: number, second: number) => {
    const updatedMean = Math.fround(Math.fround(first + second) / 2);
    return Math.fround(Math.fround(second - first) * Math.fround(second - updatedMean));
  };
  // Mean = 10000000.5; two squared deviations of .25 give .5, in either order.
  expect(legacy(10000000, 10000001)).toBe(1);
  expect(legacy(10000001, 10000000)).toBe(-0);
});

it.each([
  [10000000, 10000001],
  [10000001, 10000000],
  [-10000000, -10000001],
  [-10000001, -10000000],
  [8388640, 8388641],
  [8388641, 8388640]
])('SCL-01 preserves the centered residual for %s then %s', (first, second) => {
  // Sum units 2^-46, moment units 2^-21: the ordinary default raster encoding at this bound.
  const actual = getMomentIncrement(BigInt(first) << 46n, 1, second, -46, -21);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBe(0.5);
});

it('B2 selects the wide squared bound before quantizing the centered moment', () => {
  // x1=2^69, x2=2^70, bound=4*(2^70)^2=2^142. For n<=2, h=2, so unit=142+1+2-95=50.
  // Independently M2=(x2-x1)^2/2=(2^69)^2/2=2^137.
  const unitExponent = accumulator.getFixedPointMagnitudeUnitExponent(4 * (2 ** 70) ** 2, 2);
  expect(unitExponent).toBe(50);
  const sumUnitExponent = accumulator.getFixedPointMagnitudeUnitExponent(2 ** 70, 2);
  const actual = getMomentIncrement(
    1n << BigInt(69 - sumUnitExponent),
    1,
    2 ** 70,
    sumUnitExponent,
    unitExponent
  );
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBe(2 ** 137);
});

it.each([
  'primitive',
  'streaming',
  'calibration'
] as const)('S1 rejects extra %s readback words before selecting a triple', route => {
  const length = route === 'streaming' ? 3 : 9;
  expect(() => decodeMomentProbe(new Array<number>(length + 1).fill(0), route, 0)).toThrow();
  expect(decodeMomentProbe(new Array<number>(length).fill(0), route, 0)).toBe(0);
});

it.each([
  // For two terms h=2, so the expected unit exponent is E-92. Zero retains E=-126.
  [0, -218],
  [1, -92],
  [2, -91],
  [2 - Number.EPSILON, -92],
  [2 ** 142, 50],
  // Preserve the quantizer's legacy minimum-normal-f32 scale for subnormal inputs:
  // quantizeFixedPoint bounds left shifts to 71 and does not normalize subnormal significands.
  [2 ** -149, -218],
  [Number.MIN_VALUE, -218],
  [2 ** -1022, -218],
  [4 * Math.fround(3.4028234663852886e38) ** 2, 165]
])('B2 selects a binary64 bound %s with unit exponent %s', (bound, expected) => {
  expect(accumulator.getFixedPointMagnitudeUnitExponent(bound, 2)).toBe(expected);
});

it('B2 rejects an invalid magnitude bound through the shared validation helper', () => {
  for (const bound of [-1, NaN, Infinity, -Infinity]) {
    expect(() => accumulator.getFixedPointMagnitudeUnitExponent(bound, 2)).toThrow(
      /fixed-point.*maximumMagnitude.*finite.*nonnegative/
    );
  }
});

it('S1 reads calibration and primitive moments at offset 6 after checking all nine words', () => {
  const words = [100, 0, 0, 200, 0, 0, 300, 0, 0];
  expect(decodeMomentProbe(words, 'calibration', 0)).toBe(300);
  expect(decodeMomentProbe(words, 'primitive', 0)).toBe(300);
  expect(decodeMomentProbe(words.slice(0, 3), 'streaming', 0)).toBe(100);
});

it('SCL-01 returns exactly zero for the first observation and an unchanged mean', () => {
  expect(getMomentIncrement(0n, 0, 10000000, -46, -21)).toBe(0);
  expect(getMomentIncrement(20000000n << 46n, 2, 10000000, -46, -21)).toBe(0);
});

it('SCL-01 multiplies the residual by the exact integer count above the float32 integer range', () => {
  const count = 16777217;
  // Choose a prior sum exactly one observation-unit below n * value: residual = 1,
  // so M2 increases by 1/(16777217*16777218), independently of the large baseline.
  const actual = getMomentIncrement(
    (BigInt(count) * 10000000n - 1n) << 46n,
    count,
    10000000,
    -46,
    -100
  );
  // /usr/bin/python3: float(Fraction(1, 16777217 * 16777218)).
  const expected = 3.552713043525879e-15;
  expect(Number.isFinite(actual)).toBe(true);
  expect(Math.abs(actual / expected - 1)).toBeLessThan(3e-7);
});

it('SCL-01 keeps nearest-unit ties-to-even moment quantization', () => {
  // n=1, S=0, x=1 -> .5 rounds to 0 whole units (even). n=3 -> .75 rounds
  // to 1 in half-units: 1.5 units rounds to 2 (even), covering the other tie direction.
  expect(getMomentIncrement(0n, 1, 1, -46, 0)).toBe(0);
  expect(getMomentIncrement(0n, 3, 1, -46, -1)).toBe(1);
});

it('SCL-01 retains an unsigned 96-bit centered residual near the signed-state bound', () => {
  // n=1, old=-3*2^93, new=+3*2^93 in integer units: residual=3*2^94 needs bit 95.
  // M2 = residual^2/2 = 9*2^187, with moment unit 2^126, hence exactly 9*2^61 units.
  const actual = getMomentIncrement(-(3n << 93n), 1, 3 * 2 ** 93, 0, 126);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBe(9 * 2 ** 187);
});

it('SCL-01 scales before multiplication when the squared deviation exceeds float32', () => {
  // For observations +2^70 and -2^70, the mean is zero and M2 = 2 * (2^70)^2 = 2^141.
  const actual = getMomentIncrement(1n << 62n, 1, -(2 ** 70), 8, 80);
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBe(2 ** 141);
});

it('SCL-01 shared moment primitive parses and uses no bindings or large private arrays', () => {
  expect(accumulator.FIXED_POINT_MOMENTS_WGSL).toBeTypeOf('string');
  const source = `${accumulator.FIXED_POINT_WGSL}\n${accumulator.FIXED_POINT_MOMENTS_WGSL}`;
  expect(() => new WgslParser().parse(source)).not.toThrow();
  const usage = getWGSLModuleUsage(source);
  expect(usage.storageBuffers).toBe(0);
  expect(usage.functionBytes).toBeLessThan(8192);
});
