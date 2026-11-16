// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  AnalysisGeographicExtentAccumulator,
  normalizeAnalysisLongitude
} from '../../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-geographic-extent';

describe('exact geographic covering extent', () => {
  it('normalizes aliases without changing zero or tiny positive longitudes', () => {
    expect([-720, -360, -0, 0, 360, 720].map(normalizeAnalysisLongitude)).toEqual([
      0, 0, 0, 0, 0, 0
    ]);
    expect(normalizeAnalysisLongitude(-181)).toBe(179);
    expect(normalizeAnalysisLongitude(1e-14)).toBe(1e-14);
    expect(normalizeAnalysisLongitude(-1e-14)).toBe(0);
  });

  it.each([
    {longitudes: [178, -178], expected: [178, -178]},
    {longitudes: [150, 250], expected: [150, 250]},
    {longitudes: [350, 10], expected: [-10, 10]},
    {longitudes: [-180, 180, 540], expected: [180, 180]},
    {longitudes: [-170, -60, 50, 170], expected: [170, 50]},
    {longitudes: [0, 90, 180, 270], expected: [0, 270]}
  ])('covers point longitudes $longitudes with deterministic ties', ({longitudes, expected}) => {
    for (const ordered of [longitudes, [...longitudes].reverse()]) {
      const accumulator = new AnalysisGeographicExtentAccumulator();
      for (const longitude of ordered) accumulator.add(longitude, 90);
      expect(accumulator.getCovering()).toEqual([expected[0], 90, expected[1], 90]);
      expect(accumulator.getPlain()).toEqual([
        Math.min(...longitudes),
        90,
        Math.max(...longitudes),
        90
      ]);
    }
  });

  it('covers literal broad edges and separately cut dateline parts', () => {
    const broad = new AnalysisGeographicExtentAccumulator();
    broad.addPath(
      [
        [-170, -10],
        [170, 10]
      ],
      false
    );
    expect(broad.getCovering()).toEqual([-170, -10, 170, 10]);
    const cut = new AnalysisGeographicExtentAccumulator();
    cut.addPath(
      [
        [170, -10],
        [180, 0]
      ],
      false
    );
    cut.addPath(
      [
        [-180, 0],
        [-170, 10]
      ],
      false
    );
    expect(cut.getCovering()).toEqual([170, -10, -170, 10]);
  });

  it('finds gaps inside a former bucket, including between merged edge intervals', () => {
    const accumulator = new AnalysisGeographicExtentAccumulator();
    // Two literal edges cover the circle except (0.02, 0.08), inside one 0.1-degree bucket.
    accumulator.addPath(
      [
        [0.08, -5],
        [180, 5]
      ],
      false
    );
    accumulator.addPath(
      [
        [-180, 0],
        [0.02, 1]
      ],
      false
    );
    const extent = accumulator.getCovering();
    expect(extent?.[0]).toBeCloseTo(0.08, 12);
    expect(extent?.[2]).toBeCloseTo(0.02, 12);
    expect(extent?.[1]).toBe(-5);
    expect(extent?.[3]).toBe(5);
    expect(accumulator.getCovering()).toEqual(extent);
  });

  it('includes closing edges and preserves full-circle source bounds', () => {
    const open = new AnalysisGeographicExtentAccumulator();
    open.addPath(
      [
        [10, 0],
        [180, 1],
        [350, 2]
      ],
      false
    );
    expect(open.getCovering()).toEqual([10, 0, 350, 2]);
    const closed = new AnalysisGeographicExtentAccumulator();
    closed.addPath(
      [
        [10, 0],
        [180, 1],
        [350, 2]
      ],
      true
    );
    expect(closed.getCovering()).toEqual([10, 0, 350, 2]);
    closed.addPath(
      [
        [-360, -90],
        [720, 90]
      ],
      false
    );
    expect(closed.getCovering()).toEqual([-360, -90, 720, 90]);
  });

  it('ignores invalid positions without bridging their edges and supports later additions', () => {
    const accumulator = new AnalysisGeographicExtentAccumulator();
    expect(accumulator.getCovering()).toBeUndefined();
    accumulator.addPath(
      [
        [178, 0],
        [NaN, 1],
        [-178, 2]
      ],
      false
    );
    expect(accumulator.getCovering()).toEqual([178, 0, -178, 2]);
    accumulator.add(0, 1);
    expect(accumulator.getCovering()).toEqual([178, 0, 0, 2]);
  });
});
