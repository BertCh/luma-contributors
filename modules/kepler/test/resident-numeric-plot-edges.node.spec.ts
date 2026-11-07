// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createRequire} from 'node:module';
import {dirname} from 'node:path';
import {
  NATIVE_NUMERIC_PLOT_BINS,
  getResidentNumericPlotBin,
  makeResidentNumericEdgeWords,
  makeResidentNumericPlotEdges
} from '../../../examples/deck/kepler-compute/src/resident-numeric-plot-edges';

// The pinned native package is the oracle (CommonJS build; it cannot be imported in Node as ESM).
const require = createRequire(import.meta.url);
const nativeDirectory = dirname(require.resolve('@kepler.gl/utils'));
const {histogramFromDomain, histogramFromThreshold, getNumericBinThresholds} = require(
  `${nativeDirectory}/plot.js`
);
const {BINS} = require(`${dirname(require.resolve('@kepler.gl/constants'))}/time.js`);

function makeRandom(initialSeed: number): () => number {
  let seed = initialSeed;
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}

type Format = 'float32' | 'uint32' | 'sint32';

/** The sortable-bit encoding used by the plot shader (see resident-numeric-domain). */
function encodeOrder(value: number, format: Format): number {
  if (format === 'uint32') return value >>> 0;
  if (format === 'sint32') return (value ^ 0x80000000) >>> 0;
  const words = new Uint32Array(new Float32Array([value === 0 ? 0 : value]).buffer);
  return words[0] & 0x80000000 ? ~words[0] >>> 0 : (words[0] ^ 0x80000000) >>> 0;
}

/** CPU twin of the classification shader: ordered-word comparisons only. */
function classifyWords(value: number | null, format: Format, words: Uint32Array, count: number) {
  if (value !== null && Number.isNaN(value)) return -1;
  const order = encodeOrder(value ?? 0, format);
  const atLeast = (index: number) =>
    words[2 * index + 1] === 1 || (words[2 * index + 1] === 0 && order >= words[2 * index]);
  const upper = count + 1;
  const atMost =
    words[2 * upper + 1] === 1 || (words[2 * upper + 1] === 0 && order <= words[2 * upper]);
  if (!atLeast(0) || !atMost) return -1;
  let bin = 0;
  while (bin < count && atLeast(bin + 1)) bin++;
  return bin;
}

describe('resident numeric plot edges', () => {
  test('uses the pinned native bin count', () => {
    expect(NATIVE_NUMERIC_PLOT_BINS).toBe(BINS);
  });

  test('matches native histogramFromDomain edges, empty bins and terminal zero-width bin', () => {
    for (const domain of [
      [-10, 100],
      [0, 10],
      [0, 1],
      [5, 5],
      [0.1, 0.7],
      [-3.5, 1e9],
      [1, 1.0000001],
      [0, 1000],
      [-1e-7, 3e-7]
    ] as [number, number][]) {
      const native = histogramFromDomain(domain, [], BINS);
      const plot = makeResidentNumericPlotEdges(domain);
      expect(plot.edges).toEqual([native[0].x0, ...native.map(bin => bin.x1)]);
      expect(plot.thresholds).toEqual(plot.edges.slice(1, -1));
    }
    // A tick on the upper bound leaves the zero-width bin [100, 100].
    expect(makeResidentNumericPlotEdges([-10, 100]).edges.slice(-2)).toEqual([100, 100]);
    expect(makeResidentNumericPlotEdges([5, 5]).edges).toEqual([5, 5]);
  });

  test('numeric intervals match native thresholds, including the automatic fallback', () => {
    for (const [domain, interval] of [
      [[0, 100], 25],
      [[0, 100], 30],
      [[-10, 10], 0.5],
      [[0, 100000], 1],
      [[0, 10], 0]
    ] as [[number, number], number][]) {
      const thresholds = getNumericBinThresholds(domain, interval, BINS);
      const native =
        interval > 0
          ? histogramFromThreshold(thresholds, [], undefined, false)
          : histogramFromDomain(domain, [], BINS);
      expect(makeResidentNumericPlotEdges(domain, interval).edges).toEqual([
        native[0].x0,
        ...native.map(bin => bin.x1)
      ]);
    }
    expect(makeResidentNumericPlotEdges([0, 100], 25).edges).toEqual([0, 25, 50, 75, 100, 100]);
  });

  test('random domains and intervals match the pinned native edges', () => {
    const random = makeRandom(99);
    for (let index = 0; index < 500; index++) {
      const start = (random() - 0.5) * 10 ** Math.floor(random() * 12 - 4);
      const end = start + random() * 10 ** Math.floor(random() * 12 - 4);
      const interval = random() < 0.5 ? undefined : (end - start) / (1 + random() * 3000);
      const plot = makeResidentNumericPlotEdges([start, end], interval);
      const native =
        interval !== undefined && interval > 0
          ? histogramFromThreshold(
              getNumericBinThresholds([start, end], interval, BINS),
              [],
              undefined,
              false
            )
          : histogramFromDomain([start, end], [], BINS);
      expect(plot.edges, `${start} ${end} ${interval}`).toEqual([
        native[0].x0,
        ...native.map((bin: {x1: number}) => bin.x1)
      ]);
    }
  });

  test('rejects unordered or nonfinite domains', () => {
    expect(() => makeResidentNumericPlotEdges([1, 0])).toThrow('finite ordered');
    expect(() => makeResidentNumericPlotEdges([0, Infinity])).toThrow('finite ordered');
  });

  test('ordered-word classification equals native d3 binning for every source format', () => {
    const random = makeRandom(12345);
    for (const format of ['float32', 'sint32', 'uint32'] as Format[]) {
      for (const [domain, interval] of [
        [[-10, 100], undefined],
        [[0, 255], 10],
        [[3, 3], undefined],
        [[-1e9, 5e9], undefined],
        [[0.25, 7.75], 0.5]
      ] as [[number, number], number | undefined][]) {
        const plot = makeResidentNumericPlotEdges(domain, interval);
        const edgeWords = makeResidentNumericEdgeWords(plot, format);
        const values: (number | null)[] = [null, 0, domain[0], domain[1], ...plot.edges];
        if (format === 'float32') values.push(NaN, -0, 1e-39, -1e-39);
        for (let index = 0; index < 400; index++) {
          const span = domain[1] - domain[0] || 1;
          values.push(domain[0] - span * 0.1 + random() * span * 1.2);
        }
        const stored = values.map(value => {
          if (value === null || Number.isNaN(value)) return value;
          if (format === 'float32') return Math.fround(value);
          return Math.max(
            format === 'uint32' ? 0 : -2147483648,
            Math.min(format === 'uint32' ? 4294967295 : 2147483647, Math.round(value))
          );
        });
        const rows = stored.map((_, index) => index);
        const native = interval
          ? histogramFromThreshold(
              getNumericBinThresholds(plot.domain, interval, NATIVE_NUMERIC_PLOT_BINS),
              rows,
              (index: number) => stored[index],
              false
            )
          : histogramFromDomain(
              plot.domain,
              rows,
              NATIVE_NUMERIC_PLOT_BINS,
              (index: number) => stored[index]
            );
        const expected = new Map<number, number>();
        native.forEach((bin, binIndex) => bin.indexes.forEach(row => expected.set(row, binIndex)));
        stored.forEach((value, row) => {
          const cpu = getResidentNumericPlotBin(value, plot);
          expect(cpu, `${format} cpu ${value} in ${domain}`).toBe(expected.get(row) ?? -1);
          expect(
            classifyWords(value, format, edgeWords.words, plot.thresholds.length),
            `${format} words ${value} in ${domain}`
          ).toBe(expected.get(row) ?? -1);
        });
      }
    }
  });
});
