// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectMetricReadback} from './r19-readback-assertions';
import {getGPUSpatialWeightsKernelValue} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getNormalIntervalMass,
  getComplementaryErrorFunction
} from '../../../src/gpu-spatial-analysis/shared/complementary-error-function';
import {getComplementaryErrorFunction as getDensityErfc} from '../../../src/gpu-spatial-analysis/point-density/complementary-error-function';
import {BIN_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/shared/bin-index-wgsl';
import {BIN_INDEX_WGSL as REGRESSION_BIN_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/bin-index-wgsl';
import {resolveKernelShapeName} from '../../../src/gpu-spatial-analysis/shared/kernel-shape-name';
import {
  GPU_CENTERED_INCOMPLETE_GAMMA_WGSL,
  GPU_UPPER_INCOMPLETE_GAMMA_WGSL
} from '../../../src/analysis-core/kernels/incomplete-gamma-wgsl';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  EXACT_ORIENTATION_WGSL,
  EXACT_ORIENTATION_MAGNITUDE_WGSL
} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';

import {getWGSLModuleUsage} from '../../../src/testing/wgsl-memory-usage';

// Independent literals: scipy.stats.norm.sf(lo)-sf(hi); symmetric central
// interval is 2e-8/sqrt(2*pi) to double precision (next term is < 2e-25).
it.each([
  [-1e-8, 1e-8, 7.978845608028654e-9],
  [8, 9, 6.219831985865787e-16],
  [-9, -8, 6.219831985865787e-16],
  [3, 4, 0.0013182267897969735]
])('NUM-10 positive normal interval [%s,%s]', (low, high, expected) => {
  const result = getNormalIntervalMass(low, high);
  expect(Number.isFinite(result)).toBe(true);
  expect(Math.abs(result / expected - 1)).toBeLessThan(3e-7);
});
it('NUM-10 zero width and entire distribution', () => {
  expect(getNormalIntervalMass(0, 0)).toBe(0);
  expect(getNormalIntervalMass(-Infinity, Infinity)).toBe(1);
});
it('W3-01 exposes a magnitude companion distinct from the sign', () => {
  expect(EXACT_ORIENTATION_MAGNITUDE_WGSL).toContain('fn orientExactDeterminant(');
  const source = `${EXACT_ORIENTATION_WGSL}\n${EXACT_ORIENTATION_MAGNITUDE_WGSL}`;
  expect(() => new WgslParser().parse(source)).not.toThrow();
  const usage = getWGSLModuleUsage(source);
  expect(usage.functionBytes).toBeLessThan(8192);
  expect(usage.storageBuffers).toBe(0);
});
// Independent scipy.stats.chi2.sf(statistic,df), no WH reference in expectations.
it.each([
  [1, 9, 0.00269979606326019],
  [4, 20, 0.0004993992273873336],
  [65535, 67000, 2.9268476745227096e-5],
  [65535, 65535, 0.4992653724170944],
  [4096, 4800, 7.928264752204537e-14],
  [10000, 11000, 3.61832955809647e-12]
])('W3-02 centered upper tail df=%s statistic=%s', (degrees, value, expected) => {
  const output = new Float32Array(1);
  const source = `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${GPU_CENTERED_INCOMPLETE_GAMMA_WGSL}
 @group(0) @binding(0) var<storage,read_write> output: array<f32>;
 @compute @workgroup_size(1) fn main(){output[0]=getCenteredChiSquareSurvivalValue(${value}.0,${degrees}.0);}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: output}
  });
  expect(Number.isFinite(output[0])).toBe(true);
  expect(Math.abs(output[0] / expected - 1)).toBeLessThan(0.0002);
});
it('NAM-16 canonicalizes only the shared aliases and preserves accepted family shapes', () => {
  for (const name of ['quartic', 'bisquare'])
    expect(resolveKernelShapeName('fixture', name, ['gaussian', 'quartic'])).toBe('quartic');
  expect(resolveKernelShapeName('fixture', 'gaussian', ['gaussian', 'quartic'])).toBe('gaussian');
  expect(() => resolveKernelShapeName('fixture', 'bogus', ['gaussian', 'quartic'])).toThrow(
    /fixture.*kernel.*bogus.*gaussian.*quartic.*bisquare/
  );
  // Weights keep their own normalization: 15/16 * (1 - .5^2)^2.
  expect(getGPUSpatialWeightsKernelValue('bisquare', 0.5)).toBe(0.52734375);
});
it('STD-09 compatibility paths retain identical family values', () => {
  expect(BIN_INDEX_WGSL).toBe(REGRESSION_BIN_INDEX_WGSL);
  for (const value of [-Infinity, -8, -1, -0, 0, 1e-8, 1, 8, Infinity])
    expect(getComplementaryErrorFunction(value)).toBe(getDensityErfc(value));
});

it('B4 rejects incomplete and overlong metric vectors', () => {
  expect(() => expectMetricReadback([-4, 4, 0])).toThrow();
  expect(() => expectMetricReadback([-4, 4, 0, NaN, NaN, NaN, NaN])).toThrow();
  expect(() => expectMetricReadback([-4, 4, 0, NaN, NaN, NaN])).not.toThrow();
});
it('S2 central boundary is continuous and has exact half-distribution mass', () => {
  expect(getNormalIntervalMass(0, Infinity)).toBe(0.5);
  expect(getNormalIntervalMass(-Infinity, 0)).toBe(0.5);
  const smaller = getNormalIntervalMass(0, 1);
  const larger = getNormalIntervalMass(-1e-15, 1);
  expect(Number.isFinite(smaller) && Number.isFinite(larger)).toBe(true);
  expect(larger).toBeGreaterThanOrEqual(smaller);
});

function evaluateGammaProbe(
  body: string,
  declarations = GPU_CENTERED_INCOMPLETE_GAMMA_WGSL
): number[] {
  const output = new Float32Array(3);
  const source = `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${declarations}
 @group(0) @binding(0) var<storage,read_write> output: array<f32>;
 @compute @workgroup_size(1) fn main(){${body}}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: output}
  });
  return Array.from(output);
}
it('B2 exhausted centered gamma returns unsupported instead of plausible probability', () => {
  // scipy.stats.chi2.sf(2000000,2000000) = 0.4998670192391274.
  // The bounded series cannot converge here; it must not report its old ~0.52015 value.
  const result = evaluateGammaProbe(`let tails = getCenteredGammaTails(1000000.0,1000000.0);
 output[0]=getCenteredChiSquareSurvivalValue(2000000.0,2000000.0);
 output[1]=tails.x; output[2]=tails.y;`);
  expect(result.every(Number.isNaN)).toBe(true);
});
it('S1 lower-series fixture rejects restoration of the old 500-term budget', () => {
  const declarations = GPU_CENTERED_INCOMPLETE_GAMMA_WGSL.replace(
    'iteration < 2048',
    'iteration < 500'
  );
  const result = evaluateGammaProbe(
    'output[0]=getCenteredChiSquareSurvivalValue(65535.0,65535.0);',
    declarations
  );
  // A short budget must either refuse or fail the independent SciPy tolerance.
  expect(Number.isNaN(result[0]) || Math.abs(result[0] / 0.4992653724170944 - 1) > 0.0002).toBe(
    true
  );
});
it('S4 paired gamma tails handle runtime positive infinity and invalid shape', () => {
  const result = evaluateGammaProbe(`var infinityBits=0x7f800000u;
 let value=bitcast<f32>(infinityBits);
 let tails=getCenteredGammaTails(5.0,value);
 output[0]=tails.x; output[1]=tails.y;
 let invalid=getCenteredGammaTails(0.0,0.0); output[2]=invalid.y;`);
  expect(result.slice(0, 2)).toEqual([1, 0]);
  expect(Number.isNaN(result[2])).toBe(true);
});

it.each([
  [false, true],
  [true, true],
  [false, false],
  [true, false]
])('S3 maximum-float boundary negative=%s above=%s', (negative, above) => {
  // Independent exact integer determinant: 2^128 - 2^64; maximum finite is 2^128 - 2^104.
  // Exercise the production integer-to-float conversion directly, avoiding WgslExec's
  // signed-minimum bug in the preceding coordinate accumulation.
  // Factoring out 2 keeps the leading word below 2^31, avoiding the interpreter's
  // separate signed-right-shift defect, while minimumExponent=1 restores the exact value.
  const magnitude = ((1n << 128n) - (1n << BigInt(above ? 64 : 104))) >> 1n;
  const words = Array.from({length: 8}, (_, index) =>
    Number((magnitude >> BigInt(index * 32)) & 0xffffffffn)
  );
  const lowestBit = (magnitude & -magnitude).toString(2).length - 1;
  const start = EXACT_ORIENTATION_MAGNITUDE_WGSL.indexOf('  let highestBit =');
  const conversion = EXACT_ORIENTATION_MAGNITUDE_WGSL.slice(
    start,
    EXACT_ORIENTATION_MAGNITUDE_WGSL.lastIndexOf('}')
  );
  const source = `fn convert() -> f32 {
 var invalidBits=0x7fc00000u; let invalid=bitcast<f32>(invalidBits);
 var difference=array<u32,8>(${words.map(word => `${word}u`).join(',')});
 let highest=3u; let minimumExponent=1; let negativeResult=${negative};
 let highestWord=difference[highest]; let lowerWord=difference[highest-1u];
 let lowestBit=${lowestBit}u;
 ${conversion}}
 @group(0) @binding(0) var<storage,read_write> output: array<f32>;
 @compute @workgroup_size(1) fn main(){output[0]=convert();}`;
  const output = new Float32Array(1);
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: output}
  });
  if (above) expect(Number.isNaN(output[0])).toBe(true);
  else expect(output[0]).toBe(negative ? -3.4028234663852886e38 : 3.4028234663852886e38);
});
