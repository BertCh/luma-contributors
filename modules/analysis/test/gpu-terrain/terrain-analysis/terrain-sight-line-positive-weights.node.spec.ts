// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {TERRAIN_SIGHT_LINE_SAMPLE_POSITION_WGSL} from '../../../src/gpu-terrain/terrain-analysis/terrain-sight-line';
import {computeTerrainSightLine, computeTerrainViewshed} from './terrain-analysis-oracle';

for (const route of ['viewshed', 'line-of-sight'] as const) {
  it(`B1 preserves the exact diagonal ridge hit: ${route}`, () => {
    const values = new Float32Array(30 * 30);
    values[15 * 30 + 15] = 50;
    values[15 * 30 + 14] = values[15 * 30 + 16] = NaN;
    // Exact rational step 15 is (15,15), slope 50/(15*sqrt(2)); target slope
    // 5/(29*sqrt(2)) is smaller, so the height-50 ridge blocks the target.
    const code =
      route === 'viewshed'
        ? computeTerrainViewshed(values, undefined, 30, 30, [0, 0, 0, 5, 0, 1, 1, 0, 0, 0])[899]
        : computeTerrainSightLine(
            values,
            undefined,
            30,
            30,
            [0, 0],
            [29, 29],
            [0, 5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
          ).code;
    expect(code).toBe(0);
  });

  it(`B1 retains genuinely fractional missing-corner participation: ${route}`, () => {
    const values = new Float32Array(900);
    values[15 * 30 + 15] = 50;
    values[14 * 30 + 15] = NaN;
    // At column 15 the ray to (29,28) has row 420/29, strictly between 14 and 15.
    // Both the missing corner and ridge participate, so that interior sample is skipped.
    const code =
      route === 'viewshed'
        ? computeTerrainViewshed(values, undefined, 30, 30, [0, 0, 0, 5, 0, 1, 1, 0, 0, 0])[869]
        : computeTerrainSightLine(
            values,
            undefined,
            30,
            30,
            [0, 0],
            [29, 28],
            [0, 5, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
          ).code;
    expect(code).toBe(1);
  });

  it(`B2 keeps a flat grazing ray visible: ${route}`, () => {
    const values = new Float32Array(8).fill(3);
    // Every interpolation of the constant surface is exactly 3; eye and target
    // ground are both 3, so every apparent slope and target slope is exactly zero.
    const code =
      route === 'viewshed'
        ? computeTerrainViewshed(values, undefined, 2, 4, [0, 0.125, 0, 0, 0, 1, 1, 0, 0, 0])[7]
        : computeTerrainSightLine(
            values,
            undefined,
            2,
            4,
            [0, 0.125],
            [1, 3],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
          ).code;
    expect(code).toBe(1);
  });
}

it('URT-02 ignores zero-weight missing corners at endpoints and at an intervening ridge', () => {
  // Unit cells, zero eye/target heights and curvature. The integer target has height zero;
  // the ridge at distance 1 has slope 10, strictly above the target slope 0 at distance 2.
  const sightSettings = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0];
  const viewshedSettings = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0];
  expect(
    computeTerrainSightLine([0, 0, NaN], undefined, 3, 1, [0, 0], [1, 0], sightSettings).code
  ).toBe(1);
  expect(computeTerrainViewshed([0, 0, NaN], undefined, 3, 1, viewshedSettings)[1]).toBe(1);
  const ridge = [0, 10, 0, 0, 0, NaN];
  expect(computeTerrainSightLine(ridge, undefined, 3, 2, [0, 0], [2, 0], sightSettings).code).toBe(
    0
  );
  expect(computeTerrainViewshed(ridge, undefined, 3, 2, viewshedSettings)[2]).toBe(0);
});

it('URT-02 retains missing participating endpoint and ray-sample policies', () => {
  const settings = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0];
  // A fractional endpoint gives the missing third cell positive weight: noData.
  expect(
    computeTerrainSightLine([0, 0, NaN], undefined, 3, 1, [0, 0], [1.5, 0], settings).code
  ).toBe(3);
  // A missing participating ridge sample is skipped under the existing ray policy: visible.
  expect(computeTerrainSightLine([0, NaN, 0], undefined, 3, 1, [0, 0], [2, 0], settings).code).toBe(
    1
  );
  // Explicit validity has the same positive-weight policy as a nonfinite elevation.
  expect(
    computeTerrainSightLine([0, 10, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], 3, 2, [0, 0], [2, 0], settings)
      .code
  ).toBe(0);
});

it('B1 executes the shared WGSL integer-hit arithmetic without wide product overflow', () => {
  const cases = [
    [0, 29, 15, 29],
    [29, -29, 15, 29],
    [0, 14, 15, 30],
    [0, 21, 14, 28],
    [0, 16777216, 8191, 16382],
    [0, 29, 15, 30]
  ];
  const input = Float32Array.from(cases.flat());
  const output = new Float32Array(cases.length);
  const shader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
${TERRAIN_SIGHT_LINE_SAMPLE_POSITION_WGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x * 4u;
  output[invocation.x] = rayCoordinate(input[row], input[row+1u], u32(input[row+2u]), u32(input[row+3u]));
}`;
  new WgslExec(new WgslParser().parse(shader)).dispatchWorkgroups('main', [cases.length, 1, 1], {
    0: {0: input, 1: output}
  });
  // Exact rational results. Case five's unreduced product exceeds 2^32, but its quotient is 2^23.
  expect(Array.from(output)).toEqual([15, 14, 7, 10.5, 8388608, 14.5]);
});
