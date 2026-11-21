// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  sampleFieldOnCPU,
  type FieldRaster
} from '../../src/raster-analysis/vector-field-common/vector-field-sampling';
import {getPhilox4x32, getPhiloxUnitFloat} from '../../src/analysis-core/kernels/philox-random';
import {LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE} from '../../src/raster-analysis/flow-texture/line-integral-convolution-parameters';

/** Result of {@link convolveLineIntegralOnCPU}. */
export type LineIntegralConvolutionCPUResult = {
  /** Row-major LIC values in `[0, 1]`, NaN where the pixel centre has no field data. */
  values: Float32Array;
  /** Field speed at each pixel centre, NaN where it has no field data. */
  speeds: Float32Array;
  /** Row-major white noise in `[0, 1)`. */
  noise: Float32Array;
};

/** White noise of output pixel `(column, row)`: Philox `((column, row, 0, 0), (seed, 1)).x`. */
export function getLineIntegralConvolutionNoise(column: number, row: number, seed: number): number {
  return getPhiloxUnitFloat(
    getPhilox4x32([column, row, 0, 0], [seed, LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE])[0]
  );
}

/** Kernel weight of signed step `step`: a Hann window times the optional animated ripple. */
export function getLineIntegralConvolutionWeight(
  step: number,
  stepCount: number,
  phase: number,
  period: number
): number {
  const window = 0.5 * (1 + Math.cos((Math.PI * step) / (stepCount + 1)));
  const ripple = period > 0 ? 0.5 * (1 + Math.cos(2 * Math.PI * (step / period - phase))) : 1;
  return window * ripple;
}

/**
 * CPU oracle of `GPULineIntegralConvolution`.
 *
 * Mirrors the WGSL control flow exactly; values differ from the GPU by transcendental and
 * normalisation rounding, which can occasionally move a streamline sample into a neighbouring
 * pixel near a pixel edge. RK2 state stays in output-cell coordinates; velocity samples map
 * those cells into the field grid, and noise lookup floors the output-cell landing.
 *
 * @param field Packed vector field.
 * @param width Output width in pixels.
 * @param height Output height in pixels.
 * @param stepCount Steps per direction `L`.
 * @param parameters Float parameters from `getGPULineIntegralConvolutionParameterValues`.
 * @param seed Noise seed.
 */
export function convolveLineIntegralOnCPU(
  field: FieldRaster,
  width: number,
  height: number,
  stepCount: number,
  parameters: ArrayLike<number>,
  seed: number
): LineIntegralConvolutionCPUResult {
  const fround = Math.fround;
  const originX = fround(parameters[4]);
  const originY = fround(parameters[5]);
  // Sample displacements from the output origin, never a rounded world centre.
  const fieldExtent = [
    fround(fround(parameters[0]) - originX),
    fround(fround(parameters[1]) - originY),
    fround(parameters[2]),
    fround(parameters[3])
  ];
  const cellWidth = fround(parameters[6]);
  const cellHeight = fround(parameters[7]);
  const stepLength = fround(parameters[8]);
  // Convert the world direction to output cells, then normalize to stepLength pixels.
  const getStepVector = (direction: [number, number]): [number, number] => {
    const pixelX = fround(direction[0] / cellWidth);
    const pixelY = fround(direction[1] / cellHeight);
    const scale = fround(stepLength / getSpeed([pixelX, pixelY]));
    return [fround(scale * pixelX), fround(scale * pixelY)];
  };
  const minimumSpeed = fround(parameters[9]);
  const phase = fround(parameters[10]);
  const period = fround(parameters[11]);
  const noise = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      noise[row * width + column] = getLineIntegralConvolutionNoise(column, row, seed);
    }
  }
  const values = new Float32Array(width * height);
  const speeds = new Float32Array(width * height);
  const getSpeed = (velocity: [number, number]): number =>
    fround(
      Math.sqrt(fround(fround(velocity[0] * velocity[0]) + fround(velocity[1] * velocity[1])))
    );
  const samplePixel = (x: number, y: number): [number, number] | undefined =>
    sampleFieldOnCPU(field, fround(x * cellWidth), fround(y * cellHeight), fieldExtent);
  const getDirection = (x: number, y: number, sign: number): [number, number] | undefined => {
    const sample = samplePixel(x, y);
    if (!sample) {
      return undefined;
    }
    // WGSL length and its threshold comparison operate on binary32 values. A
    // binary64 norm can lie just below an equality boundary that the shader admits.
    const speed = getSpeed(sample);
    if (!(speed > 0) || speed < minimumSpeed) {
      return undefined;
    }
    return [fround((sign * sample[0]) / speed), fround((sign * sample[1]) / speed)];
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const pixel = row * width + column;
      const centreX = fround(column + 0.5);
      const centreY = fround(row + 0.5);
      const centre = samplePixel(centreX, centreY);
      if (!centre) {
        values[pixel] = NaN;
        speeds[pixel] = NaN;
        continue;
      }
      speeds[pixel] = getSpeed(centre);
      let weightSum = getLineIntegralConvolutionWeight(0, stepCount, phase, period);
      let sum = weightSum * noise[pixel];
      for (const sign of [1, -1]) {
        let x = centreX;
        let y = centreY;
        for (let step = 1; step <= stepCount; step++) {
          const first = getDirection(x, y, sign);
          if (!first) {
            break;
          }
          const firstStep = getStepVector(first);
          const middle = getDirection(
            fround(x + 0.5 * firstStep[0]),
            fround(y + 0.5 * firstStep[1]),
            sign
          );
          if (!middle) {
            break;
          }
          const middleStep = getStepVector(middle);
          x = fround(x + middleStep[0]);
          y = fround(y + middleStep[1]);
          if (!getDirection(x, y, sign)) {
            break;
          }
          const sampleColumn = Math.floor(x);
          const sampleRow = Math.floor(y);
          if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= width || sampleRow >= height) {
            break;
          }
          const weight = getLineIntegralConvolutionWeight(sign * step, stepCount, phase, period);
          sum += weight * noise[sampleRow * width + sampleColumn];
          weightSum += weight;
        }
      }
      values[pixel] = weightSum > 0 ? sum / weightSum : noise[pixel];
    }
  }
  return {values, speeds, noise};
}
