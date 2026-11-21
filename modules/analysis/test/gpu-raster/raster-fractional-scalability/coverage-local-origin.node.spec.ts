// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {
  getRasterRowAreaScale,
  getRasterRowEdgeWGSL
} from '../../../src/raster-analysis/raster-grid/raster-cell-area-scale';

const grid = {
  width: 1,
  height: 1,
  affine: [1, 0, 0, 0, 1, 4600000] as const,
  pixelInterpretation: 'area' as const
};

it('UNI-03 rebases binary64 vertices before float32 upload and retains absolute row latitude', () => {
  // A full-width strip has area 1 * (.75 - .25) = .5, independently of the clipper.
  expect(Array.from(Float32Array.of(4600000.25, 4600000.75))).toEqual([4600000, 4600001]);
  const localVertices = Float32Array.from([4600000.25, 4600000.75], value => value - 4600000);
  expect(localVertices[1] - localVertices[0]).toBe(0.5);
  const parameters = getGPURasterFractionalCoverageParameterValues(grid, {
    localOrigin: [0, 4600000]
  });
  expect(parameters[1]).toBe(0);
  expect(parameters[7]).toBe(0);
  expect(parameters[8]).toBe(2);
  expect(parameters[9]).toBe(4600000);
  for (const mode of ['web-mercator', 'geographic'] as const) {
    const absoluteGrid =
      mode === 'geographic' ? {...grid, affine: [1, 0, 0, 0, 1, 45] as const} : grid;
    const packed = getGPURasterFractionalCoverageParameterValues(absoluteGrid, {
      localOrigin: [0, absoluteGrid.affine[5]]
    });
    // Independent WGS84 midpoint formula (see the queued georeference spec for derivation).
    const expected = mode === 'geographic' ? 8686608245.953474 : 0.6175050465639351;
    const actual = getRasterRowAreaScale(absoluteGrid, mode, 0, packed);
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual / expected).toBeCloseTo(1, 10);
    const source = `@group(0) @binding(0) var<storage, read> settings: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
const settingsOffset: u32 = 0u;
@compute @workgroup_size(1) fn main() {
let row = 0u;
${getRasterRowEdgeWGSL(absoluteGrid, 'settings')}
result[0] = edge; }`;
    const result = new Float32Array(1);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {
      0: {0: packed, 1: result}
    });
    expect(Number.isFinite(result[0])).toBe(true);
    expect(result[0]).toBe(absoluteGrid.affine[5] + 0.5);
  }
});

it('UNI-12 rejects axes that collapse in float32 without changing any reused target word', () => {
  const target = Float32Array.from({length: 12}, (_, index) => index + 10);
  const before = target.slice();
  expect(() =>
    getGPURasterFractionalCoverageParameterValues(
      {
        ...grid,
        affine: [1, 1, 0, 1, 1 + 2 ** -25, 0]
      },
      target
    )
  ).toThrow(/finite and invertible in float32/);
  expect(target).toEqual(before);
});
