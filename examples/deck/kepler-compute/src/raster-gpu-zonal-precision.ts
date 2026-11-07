// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterZoneColumns} from './raster-gpu-zonal-zones';

/** Unit roundoff of float32 arithmetic. */
const FLOAT32_UNIT_ROUNDOFF = 2 ** -24;

/** Honest statement of what the GPU zonal reduction guarantees. */
export const RASTER_GPU_ZONAL_PRECISION_NOTE =
  'GPU zonal sums are float32 reductions per tile. Uploaded bands are centered on a per-tile ' +
  'float64 mean before the reduction and re-shifted in float64; tiles are merged with ' +
  'Neumaier-compensated float64 addition. Bands read from a resident r32float texture are not ' +
  'centered, so their per-tile sums keep the float32 rounding error reported in sumErrorBound.';

/**
 * Per-tile centering offset: the float64 mean of up to `sampleLimit` evenly strided valid samples.
 * Subtracting it before float32 conversion keeps magnitudes small, which removes most of the
 * rounding error for bands with a large offset (elevation in millimetres, timestamps, kelvin).
 */
export function getRasterZonalShift(
  values: ArrayLike<number>,
  validity: ArrayLike<number>,
  sampleLimit = 4096
): number {
  const stride = Math.max(1, Math.floor(values.length / sampleLimit));
  let total = 0;
  let count = 0;
  for (let index = 0; index < values.length; index += stride) {
    const value = values[index];
    if (validity[index] && Number.isFinite(value)) {
      total += value;
      count++;
    }
  }
  return count ? total / count : 0;
}

/** float64 subtraction first, float32 rounding of the small difference second. */
export function shiftValuesToFloat32(values: ArrayLike<number>, shift: number): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) output[index] = values[index] - shift;
  return output;
}

/** Undoes `shiftValuesToFloat32` on one tile's GPU columns (float64 arithmetic). */
export function unshiftRasterZoneColumns(
  columns: RasterZoneColumns,
  shift: number
): RasterZoneColumns {
  if (shift === 0) return columns;
  const length = columns.valueCounts.length;
  const sums = new Float64Array(length);
  const minimums = new Float64Array(length);
  const maximums = new Float64Array(length);
  for (let zone = 0; zone < length; zone++) {
    const valueCount = columns.valueCounts[zone];
    sums[zone] = columns.sums[zone] + shift * valueCount;
    minimums[zone] = valueCount ? columns.minimums[zone] + shift : columns.minimums[zone];
    maximums[zone] = valueCount ? columns.maximums[zone] + shift : columns.maximums[zone];
  }
  return {...columns, sums, minimums, maximums};
}

/** Neumaier compensated accumulator (float64) for merging per-tile sums. */
export class CompensatedSum {
  sum = 0;
  private compensation = 0;

  add(value: number): void {
    const total = this.sum + value;
    this.compensation +=
      Math.abs(this.sum) >= Math.abs(value) ? this.sum - total + value : value - total + this.sum;
    this.sum = total;
  }

  get value(): number {
    return this.sum + this.compensation;
  }
}

/**
 * A-priori worst-case absolute error of a float32 serial sum of `valueCount` values bounded by
 * `maximumMagnitude`: (n - 1) * u * sum|x| <= n * n * u * max|x|. Pairwise or tree reductions are
 * far tighter, but the order of GPU accumulation is not specified, so the serial bound is reported.
 */
export function estimateRasterZoneSumErrorBound(
  valueCount: number,
  maximumMagnitude: number
): number {
  return valueCount * valueCount * FLOAT32_UNIT_ROUNDOFF * maximumMagnitude;
}
