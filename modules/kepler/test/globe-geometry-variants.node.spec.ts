// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {
  GlobeGeometryVariants,
  getGlobeZoomBand
} from '../../../examples/deck/kepler-compute/src/globe-geometry-variants';

describe('getGlobeZoomBand', () => {
  it('selects bands with hysteresis', () => {
    expect(getGlobeZoomBand(1)).toBe(0);
    expect(getGlobeZoomBand(3.1)).toBe(1);
    expect(getGlobeZoomBand(3.1, 0)).toBe(0);
    expect(getGlobeZoomBand(3.3, 0)).toBe(1);
    expect(getGlobeZoomBand(2.9, 1)).toBe(1);
    expect(getGlobeZoomBand(2.7, 1)).toBe(0);
  });
});

function createHarness(options: {steps?: number; vectorBytes?: number} = {}) {
  const tasks: (() => void)[] = [];
  const destroyed: number[] = [];
  const released: number[] = [];
  const prepared: number[] = [];
  let sequence = 0;
  const variants = new GlobeGeometryVariants({
    createPreparation: function* (resolutionDegrees) {
      prepared.push(resolutionDegrees);
      for (let step = 0; step < (options.steps ?? 3); step++) yield;
      return {resolutionDegrees} as never;
    },
    makeVectors: source => {
      const id = ++sequence;
      return {
        resolution: (source as never as {resolutionDegrees: number}).resolutionDegrees,
        gpuByteLength: options.vectorBytes ?? 10,
        mappingByteLength: 0,
        destroy: () => destroyed.push(id)
      } as never;
    },
    reserve: () => {
      const id = sequence + 1;
      return {release: () => released.push(id)} as never;
    },
    getPlanarByteLength: () => 100,
    requestUpdate: vi.fn(),
    schedule: callback => tasks.push(callback),
    cancel: () => undefined,
    now: () => 0,
    sliceMilliseconds: 0
  });
  const runAll = () => {
    for (let guard = 0; tasks.length && guard < 100; guard++) tasks.shift()!();
  };
  return {variants, runAll, destroyed, released, prepared, tasks};
}

describe('GlobeGeometryVariants', () => {
  it('returns undefined while preparing, then the variant, one band at a time', () => {
    const {variants, runAll, prepared} = createHarness();
    expect(variants.getVariant(1)).toBeUndefined();
    expect(variants.status.state).toBe('preparing');
    runAll();
    expect(variants.getVariant(1)).toBeDefined();
    expect(variants.status.residentResolutionDegrees).toBe(5);
    expect(variants.getVariant(4)).toBeDefined();
    runAll();
    expect(prepared).toEqual([5, 2]);
    expect(variants.status.residentResolutionDegrees).toBe(2);
  });

  it('releases the resident variant after a grace period', () => {
    const {variants, runAll, destroyed} = createHarness();
    variants.getVariant(1);
    runAll();
    variants.release();
    expect(variants.status.state).toBe('idle');
    runAll();
    expect(destroyed).toHaveLength(1);
  });

  it('records an over-budget band as failed without retrying it', () => {
    const {variants, runAll, prepared} = createHarness({vectorBytes: 1 << 30});
    variants.getVariant(1);
    runAll();
    expect(variants.status.state).toBe('failed');
    variants.getVariant(1);
    runAll();
    expect(prepared).toEqual([5]);
  });
});
