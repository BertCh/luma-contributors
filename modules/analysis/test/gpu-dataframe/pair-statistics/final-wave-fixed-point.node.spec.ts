// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {PAIR_STATISTICS_UNIT_SCALING_WGSL} from '../../../src/dataframe-analysis/pair-statistics/pair-statistics-fixed-point-unit';

function getScales(unit: number, power: number): Float32Array {
  const result = new Float32Array(1);
  // wgsl_reflect does not implement ldexp. Substitute this single intrinsic's exact
  // fixture value; execute the production branch/reciprocal selection unchanged.
  const scaling = PAIR_STATISTICS_UNIT_SCALING_WGSL.replace(
    'ldexp(unit, unitPower)',
    String(Math.fround(unit * 2 ** power))
  );
  const source = `${scaling}
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn main() {
  result[0] = getPairStatisticsFixedPointScale(${unit}, ${power});
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
  return result;
}

function decodeAmount(amount: number, scale: number, lowerReciprocal: boolean): number {
  const reciprocal = Math.fround(1 / scale);
  // Model Metal's allowed division lowering and flushing of a subnormal reciprocal.
  const loweredReciprocal = reciprocal < 2 ** -126 ? 0 : reciprocal;
  const declarations = lowerReciprocal
    ? PAIR_STATISTICS_UNIT_SCALING_WGSL.replace('amount / scale', `amount * ${loweredReciprocal}`)
    : PAIR_STATISTICS_UNIT_SCALING_WGSL;
  const result = new Float32Array(1);
  const source = `${declarations}
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn main() {
  result[0] = decodePairStatisticsAmount(${amount.toExponential(17)}, 1.0, 0, ${scale.toExponential(17)});
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
  return result[0];
}

for (const {name, amount, scale, expectedBits} of [
  {name: 'rounding', amount: 61060528, scale: 7.372319129075409e28, expectedBits: 0x1c7a5200},
  {name: 'flushing', amount: 2199023255552, scale: 1.5123660864721763e38, expectedBits: 0x14900000}
])
  it(`B1 decode keeps quotient rounding under reciprocal ${name}`, () => {
    // NumPy float32 probe: the first encoded term is 61060528 at scale 0x6f6e3665.
    // Second fixture: six zeros and +/-1.5*2^-45, variance=(2a^2)/8,
    // zero IQR gives u=variance/4096. All 28 pairs sum to 8a^2;
    // encoded sum=8*4096*4*2^24=2^41. Decoded total is 9*2^-89.
    for (const lowering of [false, true]) {
      const decoded = decodeAmount(amount, scale, lowering);
      expect(Number.isFinite(decoded)).toBe(true);
      expect(new Uint32Array(Float32Array.of(decoded).buffer)[0]).toBe(expectedBits);
    }
  });

it('W5-14 uses the representable physical reciprocal for ordinary and geometric-mean units', () => {
  // Independent NumPy float32 derivation in scratch/fixed-point.py:
  // a=179*2^-43, gamma=(2a)^2/2=64082*2^-86 -> bits 0x1c7a5200.
  // The IQR-based unit stored at exponent -64 is bits 0x3b898ed0.
  const ordinaryUnit = 0.004197932779788971;
  const geometricUnit = Math.fround(Math.fround(Math.sqrt(ordinaryUnit)) ** 2);
  for (const unit of [ordinaryUnit, geometricUnit]) {
    const scale = getScales(unit, -64)[0];
    expect(Number.isFinite(scale)).toBe(true);
    expect(scale).toBeGreaterThan(0);
    // Replay individual f32 encode, integer quantization and decode operations.
    const encoded = Math.fround(8.282404377128612e-22 * scale);
    const quantized = Math.fround(Math.trunc(Math.fround(encoded + 0.5)));
    const decoded = Float32Array.of(Math.fround(quantized / scale));
    expect(new Uint32Array(decoded.buffer)[0]).toBe(0x1c7a5200);
  }
});

it('W5-14 retains exponent-separated fallback for unrepresentable tiny-unit reciprocals', () => {
  // A stored 2^-66 at exponent -64 represents physical 2^-130; 2^24/u overflows.
  expect(getScales(2 ** -66, -64)[0]).toBe(0);
  // Exactly 2^-104 also overflows; its immediate successor has a finite reciprocal.
  expect(getScales(2 ** -104, 0)[0]).toBe(0);
  expect(Number.isFinite(getScales(2 ** -104 * (1 + 2 ** -23), 0)[0])).toBe(true);
  expect(getScales(2 ** -104 * (1 + 2 ** -23), 0)[0]).toBeGreaterThan(0);
});

it('B1 integer quotient preserves signs, subnormal outputs and ordinary quotient rounding', () => {
  // Closed forms: 1/3, -1/3, 1/2^127, and (2^24-1)/2^24.
  for (const [amount, scale, expectedBits] of [
    [1, 3, 0x3eaaaaab],
    [-1, 3, 0xbeaaaaab],
    [1, 2 ** 127, 0x00400000],
    [16777215, 16777216, 0x3f7fffff],
    [-61060528, 7.372319129075409e28, 0x9c7a5200],
    [0, 3, 0]
  ]) {
    const result = decodeAmount(amount, scale, true);
    expect(Number.isFinite(result)).toBe(true);
    expect(new Uint32Array(Float32Array.of(result).buffer)[0]).toBe(expectedBits);
  }
});
