// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {GPU_FLOW_CIRCLES_WGSL} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-flow-circles-layer';
import {
  getFlowAnimationTime,
  getFlowLinesShaderSource,
  packFlowPositions,
  packFlowStaggering
} from '../../../../examples/deck/kepler-compute/gpu-layers/layers/gpu-flow-lines-layer';

describe('GPU flow layers', () => {
  it('splits float64 positions into float32 high and low parts', () => {
    const {high, low} = packFlowPositions(new Float64Array([0.1, 0.2, 7, 8, 9, 10]), 3, 2);
    expect(high.length).toBe(6);
    expect(high[0]).toBe(Math.fround(0.1));
    expect(high[0] + low[0]).toBeCloseTo(0.1, 12);
    expect(Array.from(high.slice(3))).toEqual([8, 9, 10]);
  });

  it('creates deterministic staggering in [0, 1)', () => {
    const staggering = packFlowStaggering(8);
    expect(Array.from(staggering)).toEqual(Array.from(packFlowStaggering(8)));
    expect(Array.from(staggering).every(value => value >= 0 && value < 1)).toBe(true);
  });

  it('wraps animation time in flowmap.gl units', () => {
    expect(getFlowAnimationTime(0)).toBe(0);
    expect(getFlowAnimationTime(45)).toBe(900);
    expect(getFlowAnimationTime(90)).toBe(0);
  });

  it('assembles WGSL entry points for every mode', () => {
    for (const mode of ['straight', 'curved', 'animated-straight'] as const) {
      const source = getFlowLinesShaderSource(mode);
      expect(source).toContain('@vertex fn vertexMain');
      expect(source).toContain('@fragment fn fragmentMain');
    }
    expect(getFlowLinesShaderSource('straight')).toContain('fwidth');
    expect(getFlowLinesShaderSource('curved')).toContain('curveOffsets');
    expect(getFlowLinesShaderSource('animated-straight')).toContain('staggerings');
    expect(GPU_FLOW_CIRCLES_WGSL).toContain('outOfScale');
  });
});
