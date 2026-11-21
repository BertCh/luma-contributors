// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {expectCocircularVoronoiReadback} from './voronoi-cocircular-probe';

const FIXTURE = {name: 'diagonal 0,2', triangles: [0, 1, 2, 0, 2, 3], diagonal: '0,2'};

function createReadback() {
  // Four analytic side rays of the unit square plus its zero-length diagonal dual.
  return {
    written: 5,
    required: 5,
    invalidCount: 0,
    siteIds: [0, 1, 1, 2, 2, 3, 0, 3, 0, 2],
    segments: [
      0.5, 0.5, 0.5, -1, 0.5, 0.5, 2, 0.5, 0.5, 0.5, 0.5, 2, 0.5, 0.5, -1, 0.5, 0.5, 0.5, 0.5, 0.5
    ]
  };
}

for (const failure of ['invalid-count', 'count', 'nonfinite', 'positive-bisector']) {
  it(`records raw cocircular evidence before rejecting ${failure}`, () => {
    const result = createReadback();
    if (failure === 'invalid-count') result.invalidCount = 1;
    if (failure === 'count') result.required = 6;
    if (failure === 'nonfinite') result.segments[0] = NaN;
    if (failure === 'positive-bisector') result.segments[3] = -0.75;
    const diagnostic = vi.spyOn(console, 'info').mockImplementation(() => {});
    try {
      expect(() => expectCocircularVoronoiReadback(FIXTURE, result)).toThrow();
      expect(diagnostic).toHaveBeenCalledOnce();
      const [label, payload] = diagnostic.mock.calls[0];
      expect(label).toBe('GEO-123 cocircular active-prefix probe');
      const observed = JSON.parse(payload as string);
      expect(observed['triangles']).toEqual(FIXTURE.triangles);
      expect(observed['written']).toBe(result.written);
      expect(observed['required']).toBe(result.required);
      expect(observed['invalidCount']).toBe(result.invalidCount);
      expect(observed['siteIds']).toEqual(result.siteIds);
      expect(observed['segments']).toEqual(
        result.segments.map(value => (Number.isFinite(value) ? value : String(value)))
      );
      expect(observed['zeroRows']).toEqual([
        {row: 4, ids: '0,2', coordinates: [0.5, 0.5, 0.5, 0.5]}
      ]);
    } finally {
      diagnostic.mockRestore();
    }
  });
}

it('accepts analytic cocircular readback with or without a published zero diagonal', () => {
  for (const includeZero of [false, true]) {
    const result = createReadback();
    if (!includeZero) {
      result.written = 4;
      result.required = 4;
      result.siteIds = result.siteIds.slice(0, 8);
      result.segments = result.segments.slice(0, 16);
    }
    const diagnostic = vi.spyOn(console, 'info').mockImplementation(() => {});
    try {
      expect(() => expectCocircularVoronoiReadback(FIXTURE, result)).not.toThrow();
      expect(diagnostic).toHaveBeenCalledOnce();
      const observed = JSON.parse(diagnostic.mock.calls[0][1] as string);
      expect(observed['zeroRows']).toEqual(
        includeZero ? [{row: 4, ids: '0,2', coordinates: [0.5, 0.5, 0.5, 0.5]}] : []
      );
    } finally {
      diagnostic.mockRestore();
    }
  }
});
