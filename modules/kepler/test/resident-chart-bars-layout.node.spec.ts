// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getBarIndexAtPixel,
  getBarRangeAtPixel,
  getDominantBarIndex,
  makeBarLayout,
  shouldUseCanvasBars
} from '../../../examples/deck/kepler-compute/src/resident-chart-bars-layout';

describe('resident-chart-bars-layout', () => {
  it('switches to canvas above the threshold', () => {
    expect(shouldUseCanvasBars(512)).toBe(false);
    expect(shouldUseCanvasBars(513)).toBe(true);
  });

  it('maps wide bars one-to-one and hit-tests pixels', () => {
    const layout = makeBarLayout([1, 2, 3, 4], [1, 1, 1, 1], 400);
    expect(layout.columns).toHaveLength(4);
    expect(layout.barWidth).toBe(100);
    expect(getBarIndexAtPixel(layout, 0)).toBe(0);
    expect(getBarIndexAtPixel(layout, 199)).toBe(1);
    expect(getBarIndexAtPixel(layout, 399.9)).toBe(3);
    expect(getBarIndexAtPixel(layout, 400)).toBe(-1);
    expect(getBarIndexAtPixel(layout, -1)).toBe(-1);
  });

  it('decimates sub-pixel bars with max value and summed counts', () => {
    const values = Array.from({length: 1000}, (_, index) => (index === 503 ? 99 : 1));
    const counts = Array(1000).fill(2);
    const layout = makeBarLayout(values, counts, 100);
    expect(layout.columns).toHaveLength(100);
    expect(layout.binsPerColumn).toBe(10);
    expect(layout.columns[50]).toMatchObject({
      value: 99,
      count: 20,
      firstIndex: 500,
      lastIndex: 509
    });
    expect(layout.columns[0]).toMatchObject({value: 1, firstIndex: 0, lastIndex: 9});
    expect(layout.columns.reduce((total, column) => total + column.count, 0)).toBe(2000);
    expect(getBarRangeAtPixel(layout, 50.5)).toEqual([500, 509]);
    expect(getDominantBarIndex(values, [500, 509])).toBe(503);
  });

  it('covers every bin exactly once when columns do not divide evenly', () => {
    const layout = makeBarLayout(Array(16384).fill(1), Array(16384).fill(1), 997);
    expect(layout.columns.reduce((total, column) => total + column.count, 0)).toBe(16384);
    expect(layout.columns[0].firstIndex).toBe(0);
    expect(layout.columns.at(-1)!.lastIndex).toBe(16383);
  });

  it('preserves negative spikes by absolute maximum and handles empty input', () => {
    const layout = makeBarLayout([1, -5, 2, 3], [1, 1, 1, 1], 2.5);
    expect(layout.columns[0].value).toBe(-5);
    expect(makeBarLayout([], [], 100).columns).toEqual([]);
  });

  it('lays out 16384 bars quickly', () => {
    const values = Float64Array.from({length: 16384}, (_, index) => Math.sin(index) * 100);
    const counts = new Float64Array(16384).fill(3);
    makeBarLayout(values, counts, 800);
    const startTime = performance.now();
    for (let repeat = 0; repeat < 20; repeat++) makeBarLayout(values, counts, 800);
    const perLayout = (performance.now() - startTime) / 20;
    console.log(`bars-layout 16384 bins -> 800px: ${perLayout.toFixed(3)} ms per layout`);
    expect(perLayout).toBeLessThan(50);
  });
});
