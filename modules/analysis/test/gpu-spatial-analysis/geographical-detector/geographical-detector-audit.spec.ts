// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeographicalDetector} from '../../../src/gpu-spatial-analysis/geographical-detector/gpu-geographical-detector';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getGeographicalDetectorPValue} from '../../../src/gpu-spatial-analysis/geographical-detector/geographical-detector-p-value';

it.each([
  1,
  2 ** 50
])('B4 identical permuted strata have zero q and F at scale %s', async magnitude => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const summary = rig.output('float32', 8);
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(Float32Array.of(0, 0, magnitude, magnitude, 0, 0), 'float32'),
        labels: rig.input(Uint32Array.of(0, 0, 0, 1, 1, 1), 'uint32'),
        labelCapacity: 2,
        output: {summary}
      })
    );
    const actual = await rig.readFloat(summary);
    // Both strata have mean M/3, so SSB = 3*(M/3-M/3)^2 twice = 0.
    // SST = SSW = 4*M^2/3 > 0 and df=(1,4), hence q = F = 0 exactly.
    expect(Number.isFinite(actual[0])).toBe(true);
    expect(Number.isFinite(actual[1])).toBe(true);
    expect(actual[0]).toBe(0);
    expect(actual[1]).toBe(0);
    expect(Array.from(actual.slice(3, 7))).toEqual([2, 6, 1, 4]);
  } finally {
    rig.destroy();
  }
});

it('W5-26 exact perfect split has q one and positive infinite F; constants stay undefined', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const constant of [false, true]) {
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('float32', 8);
      const values = rig.input(Float32Array.of(constant ? 1 : 0, 1, 1, 1, 1), 'float32');
      const contributor = new GPUGeographicalDetector({
        values,
        labels: rig.input(Uint32Array.of(0, 1, 1, 1, 1), 'uint32'),
        labelCapacity: 2,
        output: {summary}
      });
      rig.run(contributor);
      const actual = await rig.readFloat(summary);
      expect(actual).toHaveLength(8);
      // Mean=4/5, SSW=0, SSB=SST=4/5, df=(1,3); q=1 and F=+Infinity.
      if (constant) {
        expect(actual[0]).toBeNaN();
        expect(actual[1]).toBeNaN();
        expect(actual[2]).toBeNaN();
        expect(actual[7]).toBe(0);
      } else {
        expect(actual[0]).toBe(1);
        expect(actual[1]).toBe(Infinity);
        // (1 - 2²/5)/(SST/5) = .2/.16 = 1.25; SST = 4/5.
        expect(Number.isFinite(actual[2])).toBe(true);
        expect(Math.abs(actual[2] - 1.25)).toBeLessThan(2e-6);
        expect(Number.isFinite(actual[7])).toBe(true);
        expect(Math.abs(actual[7] - 0.8)).toBeLessThan(2e-6);
      }
      expect(Array.from(actual.slice(3, 7))).toEqual([2, 5, 1, 3]);
    } finally {
      rig.destroy();
    }
  }
});

it('B3 detector retains q five-sevenths and finite F across power-of-two rescaling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const power of [0, -76, -145]) {
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('float32', 8);
      rig.run(
        new GPUGeographicalDetector({
          values: rig.input(
            Float32Array.from([0, 2, 4, 8], value => value * 2 ** power),
            'float32'
          ),
          labels: rig.input(Uint32Array.of(0, 0, 1, 1), 'uint32'),
          labelCapacity: 2,
          output: {summary}
        })
      );
      const actual = await rig.readFloat(summary);
      expect(actual).toHaveLength(8);
      // SSW=10h², SSB=25h², SST=35h²; df=(1,2), lambda=10/7.
      for (const [index, expected] of [
        [0, 5 / 7],
        [1, 5],
        [2, 10 / 7]
      ] as const) {
        expect(Number.isFinite(actual[index])).toBe(true);
        expect(Math.abs(actual[index] - expected)).toBeLessThan(2e-5);
      }
      expect(actual.slice(3, 7)).toEqual([2, 4, 1, 2]);
      // Physical SST can round to a subnormal or zero; the dimensionless ratios stay defined.
      expect(actual[7]).toBe(Math.fround(35 * 2 ** (2 * power)));
    } finally {
      rig.destroy();
    }
  }
});

it('review 2 R3 detector keeps an exponent-safe F of 2^127 and its positive tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const summary = rig.output('float32', 8);
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(
          Float32Array.of(0, 2 ** -50, 8192, -8192, 8192, -8192, 8192, -8192),
          'float32'
        ),
        labels: rig.input(Uint32Array.of(0, 0, 1, 2, 3, 4, 5, 6), 'uint32'),
        labelCapacity: 7,
        output: {summary}
      })
    );
    const actual = await rig.readFloat(summary),
      expected = [1, 2 ** 127, 8, 7, 8, 6, 1, 6 * 2 ** 26];
    // SSW=2^-101, SSB=6*2^26 and df=(6,1). Final F is representable even
    // though the unscaled SSD ratio is not; allow 4e-6 for float32 exponent-safe arithmetic.
    expected.forEach((value, index) => {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
        [1, 2, 7].includes(index) ? Math.abs(value) * 4e-6 : 0
      );
    });
    const probability = getGeographicalDetectorPValue(actual);
    expect(probability).toBeGreaterThan(0);
    expect(Math.abs(probability / 9.0780105121473e-20 - 1)).toBeLessThan(2e-5);
  } finally {
    rig.destroy();
  }
});

it('review 2 R2 detector retains a small directly accumulated between-stratum variation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const summary = rig.output('float32', 8),
      delta = 2 ** -13;
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(
          Float32Array.of(-1, 1, -1, 1, -1 + delta, 1 + delta, -1 + delta, 1 + delta),
          'float32'
        ),
        labels: rig.input(Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1), 'uint32'),
        labelCapacity: 2,
        output: {summary}
      })
    );
    const actual = await rig.readFloat(summary),
      expected = [2 ** -28, 6 * 2 ** -28, 2 ** -27, 2, 8, 1, 6, 8];
    // SSB=2^-25, SSW=8. Relative tolerance preserves the tiny nonzero q/F;
    // an absolute epsilon near 1e-6 would incorrectly accept the old zero result.
    expected.forEach((value, index) => {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
        index < 3 ? Math.abs(value) * 1e-5 : 0
      );
    });
    expect(Math.abs(getGeographicalDetectorPValue(actual) - 0.999885559082955)).toBeLessThan(2e-9);
  } finally {
    rig.destroy();
  }
});

it.each([
  2,
  4096 * 256 + 1
])('R5 nearly perfect strata retain finite F despite q rounding to one (label capacity %s)', async labelCapacity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const summary = rig.output('float32', 8);
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(Float32Array.of(0, 0, 0, 1, 10000, 10000, 10000, 10001), 'float32'),
        labels: rig.input(
          Uint32Array.of(
            0,
            0,
            0,
            0,
            labelCapacity - 1,
            labelCapacity - 1,
            labelCapacity - 1,
            labelCapacity - 1
          ),
          'uint32'
        ),
        labelCapacity,
        output: {summary}
      })
    );
    const actual = await rig.readFloat(summary);
    // geographical-detector-audit.oracle.py: SSW=1.5, SSB=200000000.
    // Sixteen float32 ulps in F allow upstream reductions without admitting infinity.
    // Direct SSD reductions and division allow two float32 epsilons at unity for q;
    // the old 1-SSW/SST formula happened to round to exactly 1 on this fixture.
    const expected = [1, 800000000, 2, 2, 8, 1, 6, 200000000];
    expected.forEach((value, index) => {
      expect(Number.isFinite(actual[index]), `word ${index}`).toBe(true);
      expect(
        Math.abs(actual[index] - value),
        `word ${index}: ${actual[index]}`
      ).toBeLessThanOrEqual(
        index === 0 ? 2 ** -22 : [1, 7].includes(index) ? 1024 : index === 2 ? 1e-6 : 0
      );
    });
    const probability = getGeographicalDetectorPValue(actual);
    expect(probability).toBeGreaterThan(0);
    expect(Math.abs(probability / 1.5205077624435442e-24 - 1)).toBeLessThan(1e-5);
  } finally {
    rig.destroy();
  }
});

it('R5 nonperfect F overflow is unavailable rather than an exactly perfect zero tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const summary = rig.output('float32', 8);
    // SSW=.75e-30>0, SSB~=2e8, so F~=1.6e39 is finite in real arithmetic
    // but outside float32. Infinity is reserved for an exactly zero SSW.
    rig.run(
      new GPUGeographicalDetector({
        values: rig.input(Float32Array.of(0, 0, 0, 1e-15, 10000, 10000, 10000, 10000), 'float32'),
        labels: rig.input(Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1), 'uint32'),
        labelCapacity: 2,
        output: {summary}
      })
    );
    const actual = await rig.readFloat(summary);
    expect(actual[0]).toBe(1);
    expect(actual[1]).toBeNaN();
    expect(actual.slice(3, 7)).toEqual([2, 8, 1, 6]);
    expect(Math.abs(actual[2] - 2)).toBeLessThan(1e-6);
    expect(actual[7]).toBe(200000000);
    expect(getGeographicalDetectorPValue(actual)).toBeNaN();
  } finally {
    rig.destroy();
  }
});

it('audit-validation factor detector matches all NumPy summary words with a singleton and empty stratum', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [shift, scale, noncentrality, total] of [
    [0, 1, 7.7618241754832376, 258.22222222222223],
    [4, 2, 8.580640170465687, 1032.888888888889]
  ]) {
    const rig = new AnalysisRig(device);
    try {
      const summary = rig.output('float32', 8);
      const props = {
        values: rig.input(
          Float32Array.from(
            [1, 2, 2, 4, 8, 9, 12, 13, 17, NaN, 99, 100],
            value => shift + scale * value
          ),
          'float32'
        ),
        labels: rig.input(
          Uint32Array.from([0, 0, 0, 1, 1, 1, 2, 2, 3, 2, 0xffffffff, 8]),
          'uint32'
        ),
        labelCapacity: 5,
        output: {summary}
      };
      const contributor = new GPUGeographicalDetector(props);
      props.labelCapacity = 1;
      props.values = rig.input(Float32Array.of(0), 'float32');
      props.labels = rig.input(Uint32Array.of(0), 'uint32');
      props.output.summary = rig.output('float32', 1);
      rig.run(contributor);
      const actual = await rig.readFloat(summary);
      const expected = [0.9412650602409639, 26.709401709401735, noncentrality, 4, 9, 3, 5, total];
      // geographical-detector-audit.oracle.py: float32 q->F magnifies error;
      // 5e-5 relative accommodates the final division and noncentrality cancellation.
      expected.forEach((value, index) => {
        expect(Number.isFinite(actual[index])).toBe(true);
        if ([3, 4, 5, 6].includes(index)) expect(actual[index]).toBe(value);
        else expect(Math.abs(actual[index] - value)).toBeLessThan(5e-5 * Math.abs(value));
      });
    } finally {
      rig.destroy();
    }
  }
});
