// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPURasterBFASTMonitorParameterValues,
  GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-bfast-monitor';
import {
  getGPURasterHANTSParameterValues,
  GPU_RASTER_HANTS_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {
  getGPURasterHarmonicRegressionParameterValues,
  GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression';
import {
  getGPURasterLaggedCorrelationParameterValues,
  GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation';
import {
  getGPURasterWhittakerSmootherParameterValues,
  GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-whittaker-smoother';

const cases = [
  {
    name: 'BFASTMonitor',
    length: GPU_RASTER_BFAST_MONITOR_PARAMETER_LENGTH,
    write: (target: Float32Array) =>
      getGPURasterBFASTMonitorParameterValues({criticalValue: 3}, target),
    expected: [1, 0.25, 3]
  },
  {
    name: 'HANTS',
    length: GPU_RASTER_HANTS_PARAMETER_LENGTH,
    write: (target: Float32Array) =>
      getGPURasterHANTSParameterValues({period: 1, fitErrorTolerance: 2}, target),
    expected: [1, 2, 1]
  },
  {
    name: 'HarmonicRegression',
    length: GPU_RASTER_HARMONIC_REGRESSION_PARAMETER_LENGTH,
    write: (target: Float32Array) =>
      getGPURasterHarmonicRegressionParameterValues({period: 1}, target),
    expected: [1]
  },
  {
    name: 'LaggedCorrelation',
    length: GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH,
    write: (target: Float32Array) => getGPURasterLaggedCorrelationParameterValues({}, target),
    expected: [3]
  },
  {
    name: 'WhittakerSmoother',
    length: GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH,
    write: (target: Float32Array) =>
      getGPURasterWhittakerSmootherParameterValues({smoothingParameter: 10}, target),
    expected: [10, Math.fround(0.01)]
  }
];
for (const {name, length, write, expected} of cases) {
  it(`${name} parameter writer rejects short targets without mutation and preserves padding`, () => {
    // Literal layouts and defaults come from each documented fixed parameter record.
    for (const size of new Set([0, length - 1])) {
      const backing = new Float32Array(length + 4).fill(77);
      const before = backing.slice();
      const target = backing.subarray(1, size + 1);
      expect(() => write(target)).toThrow(/target must hold/);
      expect(backing).toEqual(before);
    }
    for (const size of [length, length + 3]) {
      const backing = new Float32Array(size + 2).fill(77);
      const target = backing.subarray(1, size + 1);
      expect(write(target)).toBe(target);
      expect(Array.from(target.subarray(0, length))).toEqual(expected);
      expect(Array.from(target.subarray(length))).toEqual(new Array(size - length).fill(77));
      expect(backing[0]).toBe(77);
      expect(backing[backing.length - 1]).toBe(77);
    }
  });
}
