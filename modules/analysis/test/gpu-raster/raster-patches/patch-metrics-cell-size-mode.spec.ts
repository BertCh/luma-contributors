// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {getRasterGroundScale} from '../../../src/analysis-core/contracts/raster-grid';
import {getGPURasterFractionalCoverageParameterValues} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {
  GPURasterPatchMetrics,
  getGPURasterPatchMetricsParameterValues,
  type GPURasterPatchMetricsProps
} from '../../../src/raster-analysis/raster-patches/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../../gpu-spatial-analysis/spatial-weights/spatial-weights-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Eight one-degree columns by six rows, row centres from 60.5 down to 55.5 degrees north.
const WIDTH = 8;
const HEIGHT = 6;
const GRID: GPURasterMetadata = {
  width: WIDTH,
  height: HEIGHT,
  affine: [1, 0, 10, 0, -1, 61],
  pixelInterpretation: 'area'
};
// Patch 1: columns 0 to 2 of every row. Patch 2: columns 4 to 5 of rows 0 and 1. Patch 3: a single
// pixel at the bottom right.
const LABELS = Uint32Array.from({length: WIDTH * HEIGHT}, (_, pixel) => {
  const column = pixel % WIDTH;
  const row = Math.floor(pixel / WIDTH);
  if (column <= 2) return 1;
  if (column >= 4 && column <= 5 && row <= 1) return 2;
  return column === 7 && row === 5 ? 3 : 0;
});

/** Float64 reference: sum over pixels and exposed faces with the WGS84 scale of each pixel's row. */
function getReference(mode: 'uniform' | 'geographic') {
  const area = [0, 0, 0, 0];
  const perimeter = [0, 0, 0, 0];
  for (let row = 0; row < HEIGHT; row++) {
    const latitude = 61 - (row + 0.5);
    const [scaleX, scaleY] = mode === 'uniform' ? [1, 1] : getRasterGroundScale(mode, latitude);
    for (let column = 0; column < WIDTH; column++) {
      const label = LABELS[row * WIDTH + column];
      if (label === 0) continue;
      area[label] += scaleX * scaleY;
      for (const [dx, dy] of [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0]
      ]) {
        const neighborColumn = column + dx;
        const neighborRow = row + dy;
        const exposed =
          neighborColumn < 0 ||
          neighborRow < 0 ||
          neighborColumn >= WIDTH ||
          neighborRow >= HEIGHT ||
          LABELS[neighborRow * WIDTH + neighborColumn] !== label;
        if (exposed) {
          // Faces between vertically adjacent pixels run along a column step (east-west).
          perimeter[label] += dy !== 0 ? scaleX : scaleY;
        }
      }
    }
  }
  return {area, perimeter};
}

async function run(
  cellSizeMode: GPURasterPatchMetricsProps['cellSizeMode'],
  options: {grid?: GPURasterMetadata; parameterGrid?: GPURasterMetadata} = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const capacity = 4;
  const outputs = {
    pixelCounts: rig.output('uint32', capacity),
    areas: rig.output('float32', capacity),
    perimeters: rig.output('float32', capacity)
  };
  rig.run(
    new GPURasterPatchMetrics({
      grid: options.grid ?? GRID,
      labels: rig.input(LABELS, 'uint32', LABELS.length),
      cellSizeMode,
      parameters: options.parameterGrid
        ? rig.input(
            getGPURasterFractionalCoverageParameterValues(options.parameterGrid),
            'float32',
            8
          )
        : undefined,
      output: {
        pixelCounts: outputs.pixelCounts.view,
        areas: outputs.areas.view,
        perimeters: outputs.perimeters.view
      }
    })
  );
  const result = {
    pixelCounts: await readUint32(outputs.pixelCounts.buffer, capacity),
    areas: Array.from(await readFloat32(outputs.areas.buffer, capacity)),
    perimeters: Array.from(await readFloat32(outputs.perimeters.buffer, capacity))
  };
  rig.destroy();
  return result;
}

const expectRelative = (actual: number[], expected: number[], tolerance: number) => {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(Math.abs(actual[index] - expected[index])).toBeLessThanOrEqual(
      tolerance * Math.max(1e-30, Math.abs(expected[index]))
    );
  }
};

it('GPURasterPatchMetrics keeps coordinate-unit areas and perimeters by default', async () => {
  const reference = getReference('uniform');
  for (const mode of [undefined, 'uniform'] as const) {
    const result = await run(mode);
    expect(result.pixelCounts).toEqual([18, 4, 1, 0]);
    expectRelative(result.areas.slice(0, 3), reference.area.slice(1), 0);
    expectRelative(result.perimeters.slice(0, 3), reference.perimeter.slice(1), 1e-6);
  }
});

it('GPURasterPatchMetrics reports ground square meters and meters with a geographic cellSizeMode', async () => {
  const reference = getReference('geographic');
  const result = await run('geographic');
  expect(result.pixelCounts).toEqual([18, 4, 1, 0]);
  // Metres per degree squared is about 6e9, so compare relative to the float64 reference.
  expectRelative(result.areas.slice(0, 3), reference.area.slice(1), 1e-5);
  expectRelative(result.perimeters.slice(0, 3), reference.perimeter.slice(1), 1e-5);
  expect(result.areas[3]).toBe(0);
  expect(result.perimeters[3]).toBe(0);
  // Sanity: patch 1 spans 6 rows from 60.5 to 55.5 degrees north.
  expect(result.areas[0]).toBeGreaterThan(18 * 5.5e9);
  expect(result.areas[0]).toBeLessThan(18 * 7.5e9);
});

// Independent reference, not the library's scale formula: geodesic area and perimeter of each patch
// rectangle on the WGS84 ellipsoid, with every edge densified along its parallel or meridian. From
// /usr/bin/python3 (NumPy 2.0.2, pyproj 3.6.1):
//   g = Geod(ellps='WGS84'); ring = points every 0.01 degree around (lon0, lat0)-(lon1, lat1)
//   area, perimeter = g.polygon_area_perimeter(lons, lats)
// patch 1 = lon 10..13, lat 55..61; patch 2 = lon 14..16, lat 59..61; patch 3 = lon 17..18, lat 55..56.
const GEODESIC_AREAS = [118492543415.56998, 24865893016.71557, 7035675608.892387];
const GEODESIC_PERIMETERS = [1690831.2906241724, 668814.4795494487, 349052.2993478109];

it('GPURasterPatchMetrics ground values match geodesic areas and perimeters on the ellipsoid', async () => {
  const result = await run('geographic');
  // The scale is taken at the row center, which differs from the geodesic value by about 1e-5
  // for the area of one-degree cells (second order in the cell height). A perimeter face along a
  // parallel is counted at its pixel's row center instead of at its edge latitude: the first-order
  // terms of the north and south edges cancel, the second-order residue is below 1e-4.
  expectRelative(result.areas.slice(0, 3), GEODESIC_AREAS, 5e-5);
  expectRelative(result.perimeters.slice(0, 3), GEODESIC_PERIMETERS, 2e-4);
});

it('GPURasterPatchMetrics reads the row edges from parameters on every encoding', async () => {
  // The graph is compiled for GRID (rows centred at 60.5 to 55.5) but the parameters describe the
  // same grid 30 degrees further south: the ground weights must follow the parameters.
  const moved: GPURasterMetadata = {...GRID, affine: [1, 0, 10, 0, -1, 31]};
  const baked = await run('geographic');
  const literalMoved = await run('geographic', {grid: moved});
  const parameterized = await run('geographic', {parameterGrid: moved});
  expect(literalMoved.areas[0]).not.toBeCloseTo(baked.areas[0], -8);
  expectRelative(parameterized.areas.slice(0, 3), literalMoved.areas.slice(0, 3), 1e-6);
  expectRelative(parameterized.perimeters.slice(0, 3), literalMoved.perimeters.slice(0, 3), 1e-6);
  // The parameters equal to the compiled georeference change nothing.
  const same = await run('geographic', {parameterGrid: GRID});
  expectRelative(same.areas.slice(0, 3), baked.areas.slice(0, 3), 0);
});

it('GPURasterPatchMetrics ground weights stay within their quantization at the poles', async () => {
  // A global 0.1 degree strip around 89.95 degrees north: the fixed-point weights of the last rows
  // have the largest relative error (about 1e-5 with the 2^-26 scale; 4.6e-4 with 2^-20).
  const polar: GPURasterMetadata = {
    width: WIDTH,
    height: HEIGHT,
    affine: [0.1, 0, 10, 0, -0.1, 90],
    pixelInterpretation: 'area'
  };
  const result = await run('geographic', {grid: polar});
  let expectedPatch1 = 0;
  for (let row = 0; row < HEIGHT; row++) {
    const [scaleX, scaleY] = getRasterGroundScale('geographic', 90 - 0.1 * (row + 0.5));
    expectedPatch1 += 3 * scaleX * scaleY * 0.01;
  }
  expectRelative([result.areas[0]], [expectedPatch1], 2e-5);
});

it('RT-102 uniform sheared one-cell patch has determinant area and affine edge perimeter', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const grid: GPURasterMetadata = {
    width: 1,
    height: 1,
    affine: [3, 2, 0, 4, -2, 0],
    pixelInterpretation: 'area'
  };
  const areas = rig.output('float32', 1);
  const perimeters = rig.output('float32', 1);
  try {
    rig.run(
      new GPURasterPatchMetrics({
        grid,
        labels: rig.input(new Uint32Array([1]), 'uint32', 1),
        parameters: rig.input(getGPURasterPatchMetricsParameterValues(grid), 'float32', 8),
        output: {areas: areas.view, perimeters: perimeters.view}
      })
    );
    const area = (await readFloat32(areas.buffer, 1))[0];
    const perimeter = (await readFloat32(perimeters.buffer, 1))[0];
    expect(Number.isFinite(area)).toBe(true);
    expect(Number.isFinite(perimeter)).toBe(true);
    // Hand parallelogram: |3*(-2)-2*4| = 14; 2*(sqrt(3²+4²)+sqrt(2²+(-2)²)).
    expect(area).toBe(14);
    expect(perimeter).toBeCloseTo(15.65685424949238, 5);
  } finally {
    rig.destroy();
  }
});
