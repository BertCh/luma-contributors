// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPURasterFalseDiscoveryRateMaskParameterValues,
  GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-raster-false-discovery-rate-mask';

it('PCK-15 rejects short parameter targets before writing and preserves exact/padded layouts', () => {
  const backing = Float32Array.of(77, 77, 77);
  expect(() =>
    getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05}, backing.subarray(1, 1))
  ).toThrow(/getGPURasterFalseDiscoveryRateMaskParameterValues.*required.*1.*supplied.*0/);
  expect(backing).toEqual(Float32Array.of(77, 77, 77));
  const exact = new Float32Array(GPU_RASTER_FALSE_DISCOVERY_RATE_MASK_PARAMETER_LENGTH).fill(77);
  expect(getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.05}, exact)).toBe(exact);
  expect(exact).toEqual(Float32Array.of(0.05));
  expect(getGPURasterFalseDiscoveryRateMaskParameterValues({alpha: 0.25}, backing)).toBe(backing);
  expect(backing).toEqual(Float32Array.of(0.25, 77, 77));
});
