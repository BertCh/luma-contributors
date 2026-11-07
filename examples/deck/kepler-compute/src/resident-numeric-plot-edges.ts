// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {encodeResidentNumericThreshold} from './resident-numeric-domain';

/** Pinned native `BINS` (@kepler.gl/constants) used by `getRangeFilterBins`; a test asserts equality. */
export const NATIVE_NUMERIC_PLOT_BINS = 30;

/** Pinned native `MAX_NUMERIC_BINS`; larger intervals fall back to automatic ticks, plus one terminal bin. */
const MAXIMUM_NATIVE_NUMERIC_INTERVAL_BINS = 2048;
export const MAXIMUM_RESIDENT_NUMERIC_PLOT_BINS = MAXIMUM_NATIVE_NUMERIC_INTERVAL_BINS + 1;

export type ResidentNumericPlotEdges = {
  /** Native histogram bin edges: `[domain[0], ...interior thresholds, domain[1]]`, one more than bins. */
  edges: readonly number[];
  /** Interior thresholds after the pinned d3 trimming; bin `k` holds `tz[k - 1] <= x < tz[k]`. */
  thresholds: readonly number[];
  domain: readonly [number, number];
};

/**
 * Edges of the pinned native range-filter plot: `histogramFromDomain(domain, rows, BINS)` (d3
 * `ticks` thresholds) or, for a positive numeric interval, `histogramFromThreshold(
 * getNumericBinThresholds(domain, interval))`. The d3-array ticks and the native threshold builder
 * are ported verbatim from the pinned packages (this module must run without the Kepler runtime);
 * a node spec compares them with the pinned native functions.
 * Both keep empty bins, and a tick equal to the upper bound leaves a final zero-width bin that
 * holds only that value. Rows join a bin by d3 `bisectRight` over the interior thresholds when
 * `domain[0] <= x <= domain[1]`.
 */
export function makeResidentNumericPlotEdges(
  domain: readonly number[],
  numericInterval?: number
): ResidentNumericPlotEdges {
  if (
    domain.length < 2 ||
    !Number.isFinite(domain[0]) ||
    !Number.isFinite(domain[1]) ||
    domain[0] > domain[1]
  )
    throw new Error('Numeric plot requires a finite ordered native domain');
  const bounds: [number, number] = [domain[0], domain[1]];
  const thresholds =
    Number.isFinite(numericInterval) && numericInterval! > 0
      ? getNativeNumericThresholds(bounds, numericInterval!)
      : getD3Ticks(bounds[0], bounds[1], NATIVE_NUMERIC_PLOT_BINS);
  const interior = trimD3Thresholds(thresholds, bounds);
  if (interior.length + 1 > MAXIMUM_RESIDENT_NUMERIC_PLOT_BINS)
    throw new Error('Numeric plot exceeds the bounded native bin count');
  const edges = [bounds[0], ...interior, bounds[1]];
  return {edges, thresholds: interior, domain: bounds};
}

/** `x >= threshold` for ordered source words; `always`/`never` cover thresholds outside the format. */
export type ResidentNumericEdgeWords = {
  /** Interleaved `(word, state)` pairs: lower edge, interior thresholds, then the inclusive upper limit. */
  words: Uint32Array;
  thresholdCount: number;
};

const FORMAT_LIMITS = {
  uint32: [0, 4294967295],
  sint32: [-2147483648, 2147483647],
  float32: [-3.4028234663852886e38, 3.4028234663852886e38]
} as const;

/**
 * Compare in the source format's sortable bits (see `decodeResidentNumericOrder`): a double
 * threshold `t` satisfies `x >= t` for a stored value `x` exactly when `order(x) >= word`.
 * State 0 uses the word, 1 is always true, 2 is never true.
 */
export function makeResidentNumericEdgeWords(
  plot: ResidentNumericPlotEdges,
  format: keyof typeof FORMAT_LIMITS
): ResidentNumericEdgeWords {
  const [minimum, maximum] = FORMAT_LIMITS[format];
  const words: number[] = [];
  const lower = (threshold: number) => {
    if (format === 'float32') {
      words.push(encodeResidentNumericThreshold(threshold, format), 0);
    } else if (threshold <= minimum) {
      words.push(0, 1);
    } else if (threshold > maximum) {
      words.push(0, 2);
    } else {
      words.push(encodeResidentNumericThreshold(threshold, format), 0);
    }
  };
  lower(plot.domain[0]);
  for (const threshold of plot.thresholds) lower(threshold);
  // `x <= limit` holds for the largest stored value not above the double limit.
  const limit = plot.domain[1];
  if (format === 'float32') {
    const rounded = Math.fround(limit);
    const floor = rounded > limit ? previousFloat32(rounded) : rounded;
    words.push(encodeResidentNumericThreshold(floor, format), 0);
  } else if (limit >= maximum) {
    words.push(0, 1);
  } else if (limit < minimum) {
    words.push(0, 2);
  } else {
    words.push(encodeResidentNumericThreshold(Math.floor(limit), format), 0);
  }
  return {words: Uint32Array.from(words), thresholdCount: plot.thresholds.length};
}

function previousFloat32(value: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  if (value === 0) return -Math.fround(1.401298464324817e-45);
  view.setUint32(0, value > 0 ? bits - 1 : bits + 1);
  return view.getFloat32(0);
}

/** CPU reference of the shader: native bin index for a stored value, or -1 outside the domain. */
export function getResidentNumericPlotBin(
  value: number | null,
  plot: ResidentNumericPlotEdges
): number {
  // d3 compares a null value as zero, and NaN fails the domain comparison.
  const x = value === null ? 0 : value;
  if (!(plot.domain[0] <= x && x <= plot.domain[1])) return -1;
  let bin = 0;
  while (bin < plot.thresholds.length && plot.thresholds[bin] <= x) bin++;
  return bin;
}

const E10 = Math.sqrt(50);
const E5 = Math.sqrt(10);
const E2 = Math.sqrt(2);

function getD3TickIncrement(start: number, stop: number, count: number): number {
  const step = (stop - start) / Math.max(0, count);
  const power = Math.floor(Math.log(step) / Math.LN10);
  const error = step / Math.pow(10, power);
  const factor = error >= E10 ? 10 : error >= E5 ? 5 : error >= E2 ? 2 : 1;
  return power >= 0 ? factor * Math.pow(10, power) : -Math.pow(10, -power) / factor;
}

/** d3-array 2.12.1 `ticks`, the thresholds of the pinned native `histogramFromDomain`. */
function getD3Ticks(startValue: number, stopValue: number, count: number): number[] {
  let start = +startValue;
  let stop = +stopValue;
  if (start === stop && count > 0) return [start];
  const reverse = stop < start;
  if (reverse) [start, stop] = [stop, start];
  let step = getD3TickIncrement(start, stop, count);
  if (step === 0 || !Number.isFinite(step)) return [];
  let ticks: number[];
  if (step > 0) {
    let first = Math.round(start / step);
    let last = Math.round(stop / step);
    if (first * step < start) ++first;
    if (last * step > stop) --last;
    ticks = Array.from({length: last - first + 1}, (_, index) => (first + index) * step);
  } else {
    step = -step;
    let first = Math.round(start * step);
    let last = Math.round(stop * step);
    if (first / step < start) ++first;
    if (last / step > stop) --last;
    ticks = Array.from({length: last - first + 1}, (_, index) => (first + index) / step);
  }
  return reverse ? ticks.reverse() : ticks;
}

/** Pinned native `getNumericBinThresholds(domain, interval)` for a positive interval. */
function getNativeNumericThresholds(domain: readonly [number, number], interval: number): number[] {
  const [start, end] = domain;
  if (!(end > start)) return [start, end];
  const rawCount = (end - start) / interval;
  const roundedCount = Math.round(rawCount);
  const count =
    Math.abs(rawCount - roundedCount) <= Number.EPSILON * Math.max(1, Math.abs(rawCount)) * 16
      ? roundedCount
      : Math.ceil(rawCount);
  if (count > MAXIMUM_NATIVE_NUMERIC_INTERVAL_BINS) {
    return [
      start,
      ...getD3Ticks(start, end, NATIVE_NUMERIC_PLOT_BINS).filter(
        value => value > start && value < end
      ),
      end
    ];
  }
  const thresholds = [start];
  for (let index = 1; index < count; index++) {
    const boundary = start + index * interval;
    if (boundary > thresholds[thresholds.length - 1] && boundary < end) thresholds.push(boundary);
  }
  if (end > thresholds[thresholds.length - 1]) thresholds.push(end);
  return thresholds;
}

/** d3-array `bin` removal of thresholds outside the domain: lower ones inclusive, upper ones exclusive of x1. */
function trimD3Thresholds(
  thresholds: readonly number[],
  domain: readonly [number, number]
): number[] {
  const trimmed = [...thresholds];
  while (trimmed.length && trimmed[0] <= domain[0]) trimmed.shift();
  while (trimmed.length && trimmed[trimmed.length - 1] > domain[1]) trimmed.pop();
  return trimmed;
}
