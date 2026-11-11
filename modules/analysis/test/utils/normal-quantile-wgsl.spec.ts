// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GET_NORMAL_QUANTILE_WGSL} from '../../src/analysis-core/kernels/normal-quantile-wgsl';
import {requireTestDevice} from './require-test-device';
import {getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// getNormalQuantileValue against CPython 3.15 `statistics.NormalDist().inv_cdf`, evaluated in
// float64 at the float32 probability the kernel receives. The rows cover the three branches of
// AS 241: the central rational, the intermediate tail and the far tail (tail probability below
// exp(-25) = 1.389e-11), where a mistyped coefficient gave 2e-7 to 7e-7 relative error
// (math review finding core-numerics-04).

/** `[float32 probability, inv_cdf(probability)]`. */
const REFERENCE: [number, number][] = [
  // central branch, |p - 0.5| <= 0.425
  [0.07500000298023224, -1.4395314498849903],
  [0.10000000149011612, -1.2815515570538296],
  [0.25, -0.6744897501960817],
  [0.4000000059604645, -0.2533470877078727],
  [0.5, 0.0],
  [0.6000000238418579, 0.253347164847509],
  [0.75, 0.6744897501960817],
  [0.8999999761581421, 1.281551429692279],
  [0.925000011920929, 1.4395315551523238],
  // intermediate branch, tail probability in (exp(-25), 0.075)
  [0.07000000029802322, -1.4757910259595544],
  [0.02500000037252903, -1.9599639781660514],
  [0.009999999776482582, -2.3263478824273123],
  [0.0010000000474974513, -3.090232292061434],
  [9.999999747378752e-5, -3.7190164918374546],
  [9.999999974752427e-7, -4.7534243093331225],
  [9.99999993922529e-9, -5.612001245226192],
  [1.000000013351432e-10, -6.361340900353646],
  [1.9999999920083944e-11, -6.604077590997671],
  [0.9300000071525574, 1.475791081449959],
  [0.9750000238418579, 1.9599643924763865],
  [0.9990000128746033, 3.0902361298491052],
  [0.9999899864196777, 4.26458777186022],
  // far tail, tail probability below exp(-25)
  [1.3000000381735433e-11, -6.667611452199755],
  [9.999999960041972e-13, -7.034483825858317],
  [9.9999998245167e-15, -7.650628095191647],
  [1.0000000168623835e-16, -8.222082214108651],
  [9.999999682655225e-21, -9.262340093185976],
  [1.0000000195414814e-25, -10.42045219894461],
  [1.0000000031710769e-30, -11.464024688169062],
  [1.0000000180025095e-35, -12.421412037997747],
  [9.99999991097579e-38, -12.784556667294455],
  [1.500000042698307e-38, -12.931225118736823]
];

/**
 * Relative tolerance: three float32 units in the last place. The float32 evaluation with the
 * published coefficients stays within 2e-7 of the reference on these rows; the mistyped far-tail
 * coefficient was off by 2.6e-7 at 1e-12 and by 5e-7 to 7e-7 below 1e-14.
 */
const RELATIVE_TOLERANCE = 3.6e-7;

it('the normal quantile matches inv_cdf across the central, intermediate and far-tail branches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const probabilities = Float32Array.from(REFERENCE, ([probability]) => probability);
  // Every row is an exact float32, so the kernel sees the probability the reference used.
  REFERENCE.forEach(([probability], index) => expect(probabilities[index]).toBe(probability));
  const boundaries = Float32Array.of(0, 1, -0.5, 1.5);
  const words = await runWGSLModuleKernel(device, {
    id: 'normal-quantile',
    inputs: {probabilities, boundaries},
    outputLength: probabilities.length + boundaries.length,
    invocationCount: probabilities.length + boundaries.length,
    declarations: GET_NORMAL_QUANTILE_WGSL,
    body: `if (index < ${probabilities.length}u) {
    output[outputOffset + index] = bitcast<u32>(getNormalQuantileValue(probabilities[probabilitiesOffset + index]));
  } else {
    output[outputOffset + index] = bitcast<u32>(getNormalQuantileValue(boundaries[boundariesOffset + index - ${probabilities.length}u]));
  }`
  });
  const values = getFloatWords(words);
  const mismatches: string[] = [];
  REFERENCE.forEach(([probability, expected], index) => {
    const error = Math.abs(values[index] - expected);
    if (!(error <= RELATIVE_TOLERANCE * Math.abs(expected))) {
      mismatches.push(
        `p = ${probability}: ${values[index]} against ${expected} (relative ${error / Math.abs(expected)})`
      );
    }
  });
  expect(mismatches).toEqual([]);
  // Outside the open interval the kernel returns a large finite value of the right sign.
  const outside = Array.from(values.slice(probabilities.length));
  expect(outside).toEqual([-1e15, 1e15, -1e15, 1e15].map(Math.fround));
});
