// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Densities and the histogram of a cell's calibration sample, for the insets of the standardized
// index runs. The fits come from the GPU; only the curve evaluation is CPU.

export type IndexDistribution = 'gamma' | 'fisk';

const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
];

/** Natural logarithm of the gamma function, Lanczos approximation. */
function getLogGamma(value: number): number {
  if (value < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * value)) - getLogGamma(1 - value);
  const shifted = value - 1;
  let sum = 0.99999999999980993;
  LANCZOS.forEach((coefficient, index) => {
    sum += coefficient / (shifted + index + 1);
  });
  const tail = shifted + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(tail) - tail + Math.log(sum);
}

/** Density of the fitted distribution at `value`; `shape` and `scale` are the GPU fit. */
export function getDensity(
  distribution: IndexDistribution,
  shape: number,
  scale: number,
  value: number
): number {
  if (!(value > 0) || !(shape > 0) || !(scale > 0)) return 0;
  if (distribution === 'gamma') {
    return Math.exp(
      (shape - 1) * Math.log(value) - value / scale - shape * Math.log(scale) - getLogGamma(shape)
    );
  }
  const ratio = (value / scale) ** shape;
  return ((shape / scale) * (value / scale) ** (shape - 1)) / (1 + ratio) ** 2;
}

export type CellSample = {
  /** Histogram density per bin, normalized so the largest of density and curve is 1. */
  bars: Float32Array;
  /** The fitted density at the bin centers, on the same scale. */
  curve: Float32Array;
  /** Upper end of the value axis, in the unit of the sample. */
  maximum: number;
};

/** Histogram of `sample` against the fitted density, over `binCount` bins from zero. */
export function getCellSample(
  sample: readonly number[],
  distribution: IndexDistribution,
  shape: number,
  scale: number,
  maximum: number,
  binCount: number
): CellSample {
  const width = maximum / binCount;
  const bars = new Float32Array(binCount);
  for (const value of sample) {
    if (Number.isFinite(value) && value > 0) {
      bars[Math.min(binCount - 1, Math.floor(value / width))] += 1 / (sample.length * width);
    }
  }
  const curve = Float32Array.from({length: binCount}, (_, bin) =>
    getDensity(distribution, shape, scale, (bin + 0.5) * width)
  );
  const largest = Math.max(...bars, ...curve, 1e-9);
  return {
    bars: bars.map(value => value / largest),
    curve: curve.map(value => value / largest),
    maximum
  };
}
