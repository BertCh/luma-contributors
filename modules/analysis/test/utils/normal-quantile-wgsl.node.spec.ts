// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GET_NORMAL_QUANTILE_WGSL} from '../../src/analysis-core/kernels/normal-quantile-wgsl';

// Every coefficient of the kernel against the published table of Wichura (1988), Algorithm AS 241
// (PPND16), as carried by CPython's `statistics._normal_dist_inv_cdf`. Three were mistyped until
// the math review of 2026-11-10 (finding core-numerics-04); two of those move the float32 result
// by less than one unit in the last place, so only a comparison of digits can pin them.

/** AS 241 PPND16 coefficients in the order the kernel evaluates them (highest power first). */
const PUBLISHED_COEFFICIENTS = [
  // central numerator
  '2.5090809287301226727e+3',
  '3.3430575583588128105e+4',
  '6.7265770927008700853e+4',
  '4.5921953931549871457e+4',
  '1.3731693765509461125e+4',
  '1.9715909503065514427e+3',
  '1.3314166789178437745e+2',
  '3.3871328727963666080e+0',
  // central denominator
  '5.2264952788528545610e+3',
  '2.8729085735721942674e+4',
  '3.9307895800092710610e+4',
  '2.1213794301586595867e+4',
  '5.3941960214247511077e+3',
  '6.8718700749205790830e+2',
  '4.2313330701600911252e+1',
  // intermediate numerator
  '7.74545014278341407640e-4',
  '2.27238449892691845833e-2',
  '2.41780725177450611770e-1',
  '1.27045825245236838258e+0',
  '3.64784832476320460504e+0',
  '5.76949722146069140550e+0',
  '4.63033784615654529590e+0',
  '1.42343711074968357734e+0',
  // intermediate denominator
  '1.05075007164441684324e-9',
  '5.47593808499534494600e-4',
  '1.51986665636164571966e-2',
  '1.48103976427480074590e-1',
  '6.89767334985100004550e-1',
  '1.67638483018380384940e+0',
  '2.05319162663775882187e+0',
  // far-tail numerator
  '2.01033439929228813265e-7',
  '2.71155556874348757815e-5',
  '1.24266094738807843860e-3',
  '2.65321895265761230930e-2',
  '2.96560571828504891230e-1',
  '1.78482653991729133580e+0',
  '5.46378491116411436990e+0',
  '6.65790464350110377720e+0',
  // far-tail denominator
  '2.04426310338993978564e-15',
  '1.42151175831644588870e-7',
  '1.84631831751005468180e-5',
  '7.86869131145613259100e-4',
  '1.48753612908506148525e-2',
  '1.36929880922735805310e-1',
  '5.99832206555887937690e-1'
];

/** Constants of the algorithm that are not table coefficients (split points, shifts, the unit). */
const STRUCTURAL_CONSTANTS = new Set(['0.180625', '1.0', '0.0', '1.6', '5.0', '0.5', '0.425']);

/** A decimal literal as its significant digits and the power of ten of the first digit. */
function getCanonicalDecimal(literal: string): string {
  const [mantissa, exponentText = '0'] = literal.toLowerCase().split('e');
  const [whole, fraction = ''] = mantissa.split('.');
  const digits = (whole + fraction).replace(/^0+/, '');
  const leadingZeros = (whole + fraction).length - (whole + fraction).replace(/^0+/, '').length;
  const exponent = Number(exponentText) + whole.length - 1 - leadingZeros;
  return `${digits.replace(/0+$/, '')}e${exponent}`;
}

it('canonical decimals compare digits, not rounded doubles', () => {
  expect(getCanonicalDecimal('2509.0809287301226727')).toBe(
    getCanonicalDecimal('2.5090809287301226727e+3')
  );
  expect(getCanonicalDecimal('0.0227238449892691845833')).toBe(
    getCanonicalDecimal('2.27238449892691845833e-2')
  );
  expect(getCanonicalDecimal('7.7454501427834140764e-4')).toBe(
    getCanonicalDecimal('7.74545014278341407640e-4')
  );
  // These two differ in the 17th significant digit only.
  expect(getCanonicalDecimal('4.6303378461565456959')).not.toBe(
    getCanonicalDecimal('4.6303378461565452959')
  );
});

it('the normal quantile kernel carries the published AS 241 coefficients digit for digit', () => {
  const body = GET_NORMAL_QUANTILE_WGSL.slice(
    GET_NORMAL_QUANTILE_WGSL.indexOf('let radius = 0.180625')
  );
  const literals = (body.match(/(?<![\w.])\d+\.\d+(?:e[+-]?\d+)?/g) ?? []).filter(
    literal => !STRUCTURAL_CONSTANTS.has(literal)
  );
  expect(literals.length).toBe(PUBLISHED_COEFFICIENTS.length);
  literals.forEach((literal, index) => {
    expect(getCanonicalDecimal(literal), `coefficient ${index} (${literal})`).toBe(
      getCanonicalDecimal(PUBLISHED_COEFFICIENTS[index])
    );
  });
});
