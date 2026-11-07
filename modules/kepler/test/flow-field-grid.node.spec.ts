// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  advanceFlowFieldParticle,
  buildFlowFieldGrid,
  convertSpeedDirectionToVector,
  getFlowFieldBinCounts,
  sampleFlowFieldGrid
} from '../../../examples/deck/kepler-compute/src/flow-field-grid';
import {
  getFlowFieldPalette,
  getFlowFieldSpawnRegion,
  normalizeFlowFieldColor,
  planFlowField
} from '../../../examples/deck/kepler-compute/src/flow-field-plan';

function makeUniformSamples(east: number, north: number) {
  const longitudes: number[] = [];
  const latitudes: number[] = [];
  const first: number[] = [];
  const second: number[] = [];
  for (let row = 0; row < 8; row++)
    for (let column = 0; column < 8; column++) {
      longitudes.push(-10 + column);
      latitudes.push(40 + row);
      first.push(east);
      second.push(north);
    }
  return {longitudes, latitudes, first, second};
}

describe('flow field grid', () => {
  test('direction is the bearing the flow comes from', () => {
    const [fromNorthEast, fromNorthNorth] = convertSpeedDirectionToVector(10, 0);
    expect(fromNorthEast).toBeCloseTo(0);
    expect(fromNorthNorth).toBeCloseTo(-10);
    const [fromWestEast, fromWestNorth] = convertSpeedDirectionToVector(10, 270);
    expect(fromWestEast).toBeCloseTo(10);
    expect(fromWestNorth).toBeCloseTo(0);
  });

  test('bin counts keep the aspect ratio and clamp axes', () => {
    expect(getFlowFieldBinCounts(0, 0, 20, 10, 64)).toEqual({columns: 64, rows: 32});
    expect(getFlowFieldBinCounts(0, 0, 10, 20, 64)).toEqual({columns: 32, rows: 64});
    expect(getFlowFieldBinCounts(0, 0, 1000, 1, 64).rows).toBe(4);
  });

  test('u/v samples bin to texels with validity and speed range', () => {
    const grid = buildFlowFieldGrid(makeUniformSamples(3, 4), {
      mode: 'UV',
      gridResolution: 8,
      smoothing: 0
    });
    expect(grid).not.toBeNull();
    expect(grid!.bounds).toEqual([-10, 40, -3, 47]);
    expect(grid!.texels.length).toBe(grid!.columns * grid!.rows * 4);
    expect(grid!.texels[0]).toBeCloseTo(3);
    expect(grid!.texels[1]).toBeCloseTo(4);
    expect(grid!.texels[2]).toBe(1);
    expect(grid!.maximumSpeed).toBeCloseTo(5);
  });

  test('speed and direction mode matches the u/v field it describes', () => {
    const samples = makeUniformSamples(10, 270);
    const grid = buildFlowFieldGrid(samples, {mode: 'SPEED_DIR', gridResolution: 8, smoothing: 0});
    expect(grid!.texels[0]).toBeCloseTo(10);
    expect(grid!.texels[1]).toBeCloseTo(0);
  });

  test('elevation mode flows downhill', () => {
    const samples = makeUniformSamples(0, 0);
    samples.first = samples.longitudes.map(longitude => (longitude + 10) * 100);
    const grid = buildFlowFieldGrid(samples, {mode: 'ELEVATION', gridResolution: 8, smoothing: 0});
    // Altitude rises to the east, so flow goes west.
    const sample = sampleFlowFieldGrid(grid!, -6.5, 43.5);
    expect(sample!.east).toBeLessThan(0);
    expect(Math.abs(sample!.north)).toBeLessThan(1e-6);
  });

  test('rejects invalid, polar and degenerate samples', () => {
    expect(
      buildFlowFieldGrid(
        {longitudes: [0, 1, 2], latitudes: [0, 1, 2], first: [1, 1, 1], second: [1, 1, 1]},
        {mode: 'UV', gridResolution: 8, smoothing: 0}
      )
    ).toBeNull();
    const samples = makeUniformSamples(1, 1);
    samples.first[0] = Number.NaN;
    samples.latitudes[1] = 89;
    const grid = buildFlowFieldGrid(samples, {mode: 'UV', gridResolution: 8, smoothing: 0});
    expect(grid!.skippedCount).toBe(2);
  });

  test('CPU reference advects a particle along the field and leaves the domain', () => {
    const grid = buildFlowFieldGrid(makeUniformSamples(10, 0), {
      mode: 'UV',
      gridResolution: 8,
      smoothing: 0
    })!;
    const next = advanceFlowFieldParticle(grid, -8, 43, 1000)!;
    expect(next[0]).toBeGreaterThan(-8);
    expect(next[1]).toBeCloseTo(43, 6);
    expect(advanceFlowFieldParticle(grid, -3.2, 43, 1e9)).toBeNull();
  });
});

describe('flow field plan', () => {
  const field = {bounds: [-10, 40, -3, 47] as const, maximumSpeed: 5};
  test('visual policy rescales so the fastest cell crosses a share of the field per lifetime', () => {
    const plan = planFlowField({cycleSeconds: 20, lineLifetime: 0.5}, field);
    expect(plan.lifetimeSeconds).toBe(10);
    expect(plan.velocityScale).toBeGreaterThan(100);
    expect(planFlowField({}, field, 'physical').velocityScale).toBe(1);
  });

  test('clamps particle counts and reports unsupported native options', () => {
    expect(planFlowField({linesPerScreen: 1e9}, field).particleCount).toBe(262144);
    expect(planFlowField({linesPerScreen: 1}, field).particleCount).toBe(1024);
    // Color by speed, the debug grid and terrain Z are all rendered now: no plan-level limits.
    const plan = planFlowField(
      {colorBySpeed: true, debugGrid: true, elevationMultiplier: 3},
      field
    );
    expect(plan.limits).toHaveLength(0);
  });

  test('palettes are linear 0-1 RGB, at most eight bands', () => {
    expect(getFlowFieldPalette(undefined)).toEqual([]);
    expect(getFlowFieldPalette({colors: ['#ff0000', [0, 255, 0]]})).toEqual([
      [1, 0, 0],
      [0, 1, 0]
    ]);
    const many = getFlowFieldPalette({
      colors: Array.from({length: 20}, (_, index) => [index * 10, 0, 0])
    });
    expect(many).toHaveLength(8);
    // First and last bands come from the ends of the palette.
    expect(many[0][0]).toBeLessThan(0.1);
    expect(many[7][0]).toBeGreaterThan(0.6);
  });

  test('the spawn region concentrates particles on the viewport', () => {
    const bounds: [number, number, number, number] = [0, 0, 100, 50];
    expect(getFlowFieldSpawnRegion(undefined, bounds)).toBeUndefined();
    // A viewport covering the field keeps the whole-field budget.
    expect(getFlowFieldSpawnRegion([-10, -10, 120, 60], bounds)).toBeUndefined();
    const region = getFlowFieldSpawnRegion([40, 20, 60, 30], bounds)!;
    // Viewport plus a tenth of its size on each side, normalized to the field.
    expect(region[0]).toBeCloseTo(0.38);
    expect(region[2]).toBeCloseTo(0.62);
    expect(region[1]).toBeCloseTo(0.38);
    expect(region[3]).toBeCloseTo(0.62);
    // World copies: a viewport shifted by 360 degrees maps back onto the field.
    expect(getFlowFieldSpawnRegion([400, 20, 420, 30], bounds)![0]).toBeCloseTo(0.38);
    // Missing the field entirely.
    expect(getFlowFieldSpawnRegion([200, 20, 220, 30], bounds)).toBeUndefined();
  });

  test('colors are normalized to linear 0-1 RGBA', () => {
    expect(normalizeFlowFieldColor([255, 0, 51], 0.5)).toEqual([1, 0, 0.2, 0.5]);
  });
});
