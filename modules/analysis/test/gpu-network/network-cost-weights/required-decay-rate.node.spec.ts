// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUNetworkAccessibilityParameterValues} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {getGPUNetworkCostWeightsParameterValues} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';

// Compiled by the lead: rate omission is also a type error, while broad forwarded modes
// with an explicit rate remain accepted (for settings builders and oracle fixtures).
function checkSettingsTypes(decay: 'none' | 'exponential' | 'power') {
  getGPUNetworkAccessibilityParameterValues({threshold: 10, decay, beta: 1});
  getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay, beta: 1});
  // @ts-expect-error Exponential needs a rate.
  getGPUNetworkAccessibilityParameterValues({threshold: 10, decay: 'exponential'});
  // @ts-expect-error Power needs a rate.
  getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay: 'power'});
}
void checkSettingsTypes;

for (const decay of ['exponential', 'power'] as const) {
  it(`DEF-08 requires an explicit finite nonnegative beta for ${decay}`, () => {
    const packers = [
      (beta?: number) =>
        getGPUNetworkAccessibilityParameterValues({threshold: 10, decay, beta: beta!}),
      (beta?: number) =>
        getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay, beta: beta!})
    ];
    for (const pack of packers) {
      for (const beta of [undefined, -1, NaN, Infinity, 1e39])
        expect(() => pack(beta)).toThrow(/beta.*finite.*received/);
      for (const beta of [0, 1])
        expect(Array.from(pack(beta))).toEqual([10, decay === 'exponential' ? 1 : 2, beta, 1]);
      const maximumFiniteFloat32 = 3.4028234663852886e38;
      expect(pack(maximumFiniteFloat32)[2]).toBe(maximumFiniteFloat32);
    }
  });
}

it('DEF-08 preserves binary defaults and the Gaussian cutoff guard', () => {
  expect(Array.from(getGPUNetworkAccessibilityParameterValues({threshold: 10}))).toEqual([
    10, 0, 0, 1
  ]);
  expect(Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10}))).toEqual([
    10, 0, 0, 1
  ]);
  expect(
    Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10, decay: 'gaussian'}))
  ).toEqual([10, 3, 0, 1]);
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({costLimit: Infinity, decay: 'gaussian'})
  ).toThrow(/finite positive costLimit/);
});
