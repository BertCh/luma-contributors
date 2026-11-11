// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GLOBE_SPLIT_AGGREGATES,
  GLOBE_SPLIT_REGIONS
} from '../../../examples/deck/kepler-compute/scripts/globe-split-matrix.mjs';

describe('globe split aggregate matrix', () => {
  it('covers grid, hexagon and heatmap with drawable settings', () => {
    expect(Object.keys(GLOBE_SPLIT_AGGREGATES)).toEqual(['grid', 'hexagon', 'heatmap']);
    for (const [name, scenario] of Object.entries<any>(GLOBE_SPLIT_AGGREGATES)) {
      expect(scenario.layerType).toBe(name);
      expect(scenario.minPixels).toBeGreaterThan(0);
      expect(scenario.visConfig.opacity).toBe(1);
      expect(scenario.visConfig.colorRange.colors).toHaveLength(2);
    }
  });

  it('keeps the left and right regions disjoint with a gap for the divider', () => {
    const {left, right, full} = GLOBE_SPLIT_REGIONS;
    expect(left[1]).toBeLessThan(right[0]);
    expect(full[0]).toBe(left[0]);
    expect(full[1]).toBeGreaterThanOrEqual(right[1]);
  });
});
