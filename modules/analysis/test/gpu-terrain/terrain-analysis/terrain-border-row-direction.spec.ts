// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUTerrainDerivativesParameterValues,
  GPUTerrainDerivatives,
  type GPUTerrainDerivativesProps
} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {
  getGPUTerrainCurvatureParameterValues,
  GPUTerrainCurvature,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureProps
} from '../../../src/gpu-terrain/terrain-curvature/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Every expected value in this file is a closed form of the analytic surface (its true gradient
// and second derivatives), or the arithmetic of the review report for the 'clamp' border. None is
// a copy of the kernel.

const WIDTH = 9;
const HEIGHT = 7;
const PIXEL_COUNT = WIDTH * HEIGHT;
const CELL_SIZE = 10;

/** Elevation from a function of ground `x` (east, metres) and row index. */
function createSurface(getElevation: (x: number, row: number) => number): Float32Array {
  return Float32Array.from({length: PIXEL_COUNT}, (_, pixel) =>
    getElevation((pixel % WIDTH) * CELL_SIZE, Math.floor(pixel / WIDTH))
  );
}

async function runDerivatives(
  device: Device,
  elevation: Float32Array,
  grid: GPURasterMetadata,
  overrides: Partial<GPUTerrainDerivativesProps> = {},
  validityMask?: Uint32Array
): Promise<{slope: number[]; aspect: number[]; validity: number[]}> {
  const graph = new GPUCommandGraph(device, {id: 'terrain-border-derivatives'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = validityMask ? createInputBuffer(device, validityMask) : undefined;
  const slopeBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const aspectBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const validityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUParameterBuffer(device, {
    id: 'derivative-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainDerivativesParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainDerivatives({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', PIXEL_COUNT)
          : undefined
      },
      parameters: settings.importToGraph(graph),
      output: {
        slope: importGraphBuffer(graph, 'slope', slopeBuffer, 'float32', PIXEL_COUNT),
        aspect: importGraphBuffer(graph, 'aspect', aspectBuffer, 'float32', PIXEL_COUNT),
        outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', PIXEL_COUNT)
      },
      ...overrides
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    slope: await readFloat32(slopeBuffer, PIXEL_COUNT),
    aspect: await readFloat32(aspectBuffer, PIXEL_COUNT),
    validity: await readUint32(validityBuffer, PIXEL_COUNT)
  };
  compiled.destroy();
  settings.destroy();
  for (const buffer of [elevationBuffer, slopeBuffer, aspectBuffer, validityBuffer]) {
    buffer.destroy();
  }
  maskBuffer?.destroy();
  return result;
}

async function runCurvature<Kind extends GPUTerrainCurvatureKind>(
  device: Device,
  elevation: Float32Array,
  grid: GPURasterMetadata,
  kinds: readonly Kind[],
  overrides: Partial<GPUTerrainCurvatureProps> = {}
): Promise<Record<Kind, number[]>> {
  const graph = new GPUCommandGraph(device, {id: 'terrain-border-curvature'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputBuffers = kinds.map(() => createOutputBuffer(device, PIXEL_COUNT));
  const settings = new GPUParameterBuffer(device, {
    id: 'curvature-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainCurvatureParameterValues({}, grid)
  });
  graph.add(
    new GPUTerrainCurvature({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      parameters: settings.importToGraph(graph),
      output: {
        curvatures: Object.fromEntries(
          kinds.map((kind, index) => [
            kind,
            importGraphBuffer(graph, kind, outputBuffers[index], 'float32', PIXEL_COUNT)
          ])
        )
      },
      ...overrides
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {} as Record<Kind, number[]>;
  for (const [index, kind] of kinds.entries()) {
    result[kind] = await readFloat32(outputBuffers[index], PIXEL_COUNT);
  }
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  for (const buffer of outputBuffers) buffer.destroy();
  return result;
}

const toDegrees = (radians: number): number => (radians * 180) / Math.PI;

it('GPUTerrainDerivatives keeps the slope of a plane up to the border by default', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  // Gradient 0.2 rising east at 1200 m: slope atan(0.2) = 11.3099 degrees, downslope aspect 270.
  const plane = createSurface(x => 1200 + 0.2 * x);
  const result = await runDerivatives(device, plane, grid);
  for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
    expect(
      Math.abs(result.slope[pixel] - toDegrees(Math.atan(0.2))),
      `slope ${pixel}`
    ).toBeLessThan(2e-3);
    expect(Math.abs(result.aspect[pixel] - 270), `aspect ${pixel}`).toBeLessThan(2e-3);
    expect(result.validity[pixel]).toBe(1);
  }

  // The 'clamp' border stays available by name and halves the gradient across the west and east
  // edges: atan(0.1) = 5.7106 degrees there, 11.3099 in the interior.
  const clamped = await runDerivatives(device, plane, grid, {borderMode: 'clamp'});
  for (let row = 0; row < HEIGHT; row++) {
    expect(Math.abs(clamped.slope[row * WIDTH] - toDegrees(Math.atan(0.1)))).toBeLessThan(2e-3);
    expect(
      Math.abs(clamped.slope[row * WIDTH + WIDTH - 1] - toDegrees(Math.atan(0.1)))
    ).toBeLessThan(2e-3);
    expect(Math.abs(clamped.slope[row * WIDTH + 4] - toDegrees(Math.atan(0.2)))).toBeLessThan(2e-3);
  }
});

it('GPUTerrainDerivatives border slope is exact on a quadratic surface and on a tilted plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  // z = 0.002 x^2: dz/dx = 0.004 x, which is 0 at the west edge and 0.32 at x = 80 m. The
  // one-sided second-order stencil is exact for a quadratic.
  const bowl = await runDerivatives(
    device,
    createSurface(x => 300 + 0.002 * x * x),
    grid
  );
  for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
    const x = (pixel % WIDTH) * CELL_SIZE;
    expect(
      Math.abs(bowl.slope[pixel] - toDegrees(Math.atan(0.004 * x))),
      `bowl ${pixel}`
    ).toBeLessThan(3e-3);
  }
  // A plane tilted on both axes: east gradient 0.2, and 3 m per row toward the south (0.3). Every
  // pixel, corners included, has slope atan(hypot(0.2, 0.3)). Rows advance south and the surface
  // rises south, so it falls toward the north-west: aspect = atan2(-0.2, 0.3) = 326.31 degrees.
  const tilted = await runDerivatives(
    device,
    createSurface((x, row) => 50 + 0.2 * x + 3 * row),
    grid
  );
  for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
    expect(
      Math.abs(tilted.slope[pixel] - toDegrees(Math.atan(Math.hypot(0.2, 0.3)))),
      `tilted ${pixel}`
    ).toBeLessThan(2e-3);
    expect(Math.abs(tilted.aspect[pixel] - 326.3099), `tilted aspect ${pixel}`).toBeLessThan(2e-3);
  }
});

it('GPUTerrainDerivatives one-sided border keeps the validity pattern of clamp', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  const plane = createSurface((x, row) => 10 + 0.2 * x + 2 * row);
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  // Two cells in from the west edge on row 3: the third sample of the quadratic for the ring
  // cells of rows 2 to 4. They fall back to the line through two samples and stay valid.
  mask[3 * WIDTH + 2] = 0;
  const oneSided = await runDerivatives(device, plane, grid, {}, mask);
  const clamped = await runDerivatives(device, plane, grid, {borderMode: 'clamp'}, mask);
  expect(oneSided.validity).toEqual(clamped.validity);
  // The west-edge pixel of row 3 is valid and still exact on the plane (linear fallback).
  expect(oneSided.validity[3 * WIDTH]).toBe(1);
  expect(
    Math.abs(oneSided.slope[3 * WIDTH] - toDegrees(Math.atan(Math.hypot(0.2, 0.2))))
  ).toBeLessThan(2e-3);
});

it('GPUTerrainDerivatives takes north from the grid affine', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Row 0 is the southernmost row (affine[4] = +10) and the surface rises 3 m per row, so it
  // rises to the north: it faces south, aspect 180. Reading the rows as south-advancing would
  // give aspect 0.
  const northGrid = makeGPURasterMetadata({
    width: WIDTH,
    height: HEIGHT,
    cellSize: CELL_SIZE,
    rowDirection: 'north'
  });
  expect(northGrid.affine[4]).toBe(10);
  const risingNorth = createSurface((_, row) => 400 + 3 * row);
  const fromAffine = await runDerivatives(device, risingNorth, northGrid);
  for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
    expect(Math.abs(fromAffine.aspect[pixel] - 180), `aspect ${pixel}`).toBeLessThan(2e-3);
  }
  // The same samples on a north-up grid rise to the south: aspect 0.
  const southGrid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  const northUp = await runDerivatives(device, risingNorth, southGrid);
  for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
    const aspect = northUp.aspect[pixel];
    expect(Math.min(Math.abs(aspect), Math.abs(aspect - 360)), `aspect ${pixel}`).toBeLessThan(
      2e-3
    );
  }
});

it('GPUTerrainCurvature reports zero curvature on a plane up to the border by default', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  const plane = createSurface((x, row) => 1200 + 0.2 * x + 1.5 * row);
  const kinds = ['profile', 'mean', 'plan', 'gaussian', 'laplacian'] as const;
  for (const method of ['evans-young', 'zevenbergen-thorne', 'florinsky'] as const) {
    const result = await runCurvature(device, plane, grid, kinds, {method});
    for (const kind of kinds) {
      for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
        // Elevations near 1200 m carry 1.2e-4 m of float32 spacing; over 10 m cells a second
        // difference of rounding is below 1e-5 1/m.
        expect(Math.abs(result[kind][pixel]), `${method} ${kind} ${pixel}`).toBeLessThan(1e-5);
      }
    }
  }

  // The report's numbers for the old default on gradient 0.2 with 10 m cells, Zevenbergen-Thorne:
  // in the west column p = 0.1 and r = 0.02, so profile = -r / (1 + p^2)^1.5 = -0.019704 and
  // mean = -r / (2 (1 + p^2)^1.5) = -0.009852. 'clamp' still does this, by name.
  const eastPlane = createSurface(x => 1200 + 0.2 * x);
  const clamped = await runCurvature(device, eastPlane, grid, ['profile', 'mean'], {
    method: 'zevenbergen-thorne',
    borderMode: 'clamp'
  });
  const westPixel = 3 * WIDTH;
  expect(Math.abs(clamped.profile[westPixel] - -0.019704)).toBeLessThan(2e-5);
  expect(Math.abs(clamped.mean[westPixel] - -0.009852)).toBeLessThan(2e-5);
  const oneSided = await runCurvature(device, eastPlane, grid, ['profile', 'mean'], {
    method: 'zevenbergen-thorne'
  });
  expect(Math.abs(oneSided.profile[westPixel])).toBeLessThan(1e-5);
  expect(Math.abs(oneSided.mean[westPixel])).toBeLessThan(1e-5);
});

it('GPUTerrainCurvature border second derivatives are exact on a quadratic surface', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  // z = 0.001 x^2 + 0.002 y^2 with y = 10 m per row: r = 0.002, t = 0.004, so the Laplacian
  // r + t is 0.006 1/m at every pixel, border ring and corners included.
  const surface = createSurface(
    (x, row) => 0.001 * x * x + 0.002 * (row * CELL_SIZE) * (row * CELL_SIZE)
  );
  for (const method of ['evans-young', 'zevenbergen-thorne', 'florinsky'] as const) {
    const result = await runCurvature(device, surface, grid, ['laplacian'], {method});
    for (let pixel = 0; pixel < PIXEL_COUNT; pixel++) {
      expect(Math.abs(result.laplacian[pixel] - 0.006), `${method} ${pixel}`).toBeLessThan(2e-6);
    }
  }
});

it('GPUTerrainDerivatives border gradients are the one-sided stencil of GPUFiniteDifference2D on rough terrain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  const surface = createSurface(
    (x, row) => 100 + 7 * Math.sin(0.09 * x) + 5 * Math.cos(0.7 * row + 0.03 * x)
  );
  const result = await runDerivatives(device, surface, grid);
  const getElevation = (column: number, row: number): number => surface[row * WIDTH + column];
  // The reference formula, in float64: luma.gl's second-order one-sided first derivative
  // (-3 f0 + 4 f1 - f2) / (2 h) at the edge sample, read inward along `step`.
  const getOneSided = (first: number, second: number, third: number): number =>
    (-3 * first + 4 * second - third) / (2 * CELL_SIZE);
  // Horn smooths the across-edge derivative with weights 1, 2, 1 along the edge.
  const smooth = (getDerivative: (offset: number) => number): number =>
    (getDerivative(-1) + 2 * getDerivative(0) + getDerivative(1)) / 4;
  // Slope and aspect give back the gradient: magnitude tan(slope), pointing away from the aspect.
  const getGradient = (pixel: number): {east: number; north: number} => {
    const magnitude = Math.tan((result.slope[pixel] * Math.PI) / 180);
    const aspect = (result.aspect[pixel] * Math.PI) / 180;
    return {east: -magnitude * Math.sin(aspect), north: -magnitude * Math.cos(aspect)};
  };
  for (let row = 1; row < HEIGHT - 1; row++) {
    const west = smooth(offset =>
      getOneSided(
        getElevation(0, row + offset),
        getElevation(1, row + offset),
        getElevation(2, row + offset)
      )
    );
    expect(Math.abs(getGradient(row * WIDTH).east - west), `west ${row}`).toBeLessThan(2e-5);
    // At the east edge the inward direction is west, so the eastward derivative changes sign.
    const east = -smooth(offset =>
      getOneSided(
        getElevation(WIDTH - 1, row + offset),
        getElevation(WIDTH - 2, row + offset),
        getElevation(WIDTH - 3, row + offset)
      )
    );
    expect(Math.abs(getGradient(row * WIDTH + WIDTH - 1).east - east), `east ${row}`).toBeLessThan(
      2e-5
    );
  }
  for (let column = 1; column < WIDTH - 1; column++) {
    // Rows advance south: the inward direction from row 0 is south, so north is the negative.
    const top = -smooth(offset =>
      getOneSided(
        getElevation(column + offset, 0),
        getElevation(column + offset, 1),
        getElevation(column + offset, 2)
      )
    );
    expect(Math.abs(getGradient(column).north - top), `top ${column}`).toBeLessThan(2e-5);
    const bottom = smooth(offset =>
      getOneSided(
        getElevation(column + offset, HEIGHT - 1),
        getElevation(column + offset, HEIGHT - 2),
        getElevation(column + offset, HEIGHT - 3)
      )
    );
    expect(
      Math.abs(getGradient((HEIGHT - 1) * WIDTH + column).north - bottom),
      `bottom ${column}`
    ).toBeLessThan(2e-5);
  }
});
