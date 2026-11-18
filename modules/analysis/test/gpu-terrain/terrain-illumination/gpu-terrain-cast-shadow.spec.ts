// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  type GPUTerrainCastShadowSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, createSmoothTerrain, getVisibleDiskFraction} from './terrain-horizon-oracle';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {computeTerrainHorizonSweepBruteForce} from './terrain-horizon-sweep-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeTerrainCastShadowOracle} from './terrain-cast-shadow-oracle';

type CastShadowResult = {
  sunVisibility: number[];
  horizonAngle: number[];
  validity: number[];
};

function createCastShadowFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: {
    maximumRadius?: number;
    rowDirection?: 'south' | 'north';
    cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
    sweepSettings?: Pick<
      GPUTerrainCastShadowSettings,
      'sweepOrigin' | 'sweepReferenceCellSize' | 'sweepSourceGrid'
    >;
  } = {}
) {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'cast-shadow-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const buffers = {
    sunVisibility: createOutputBuffer(device, pixelCount),
    horizonAngle: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  // The contributor takes north from the grid, so the grid states the fixture's row order.
  const defaultGrid = makeGPURasterMetadata({
    width,
    height,
    cellSize: 10,
    rowDirection: options.rowDirection
  });
  const settings = new GPUParameterBuffer(device, {
    id: 'cast-shadow-parameters',
    format: 'float32',
    length: options.sweepSettings
      ? GPU_TERRAIN_CAST_SHADOW_SWEEP_PARAMETER_LENGTH
      : GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
    values: getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees: 0, altitudeDegrees: 0, ...options.sweepSettings},
      defaultGrid
    )
  });
  const contributor = new GPUTerrainCastShadow({
    grid: defaultGrid,
    maximumRadius: options.maximumRadius,
    rowDirection: options.rowDirection,
    cellSizeMode: options.cellSizeMode,
    parameterLayout: options.sweepSettings ? 'sweep' : 'standard',
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      }
    },
    parameters: settings.importToGraph(graph),
    output: {
      sunVisibility: importGraphBuffer(
        graph,
        'sun-visibility',
        buffers.sunVisibility,
        'float32',
        pixelCount
      ),
      horizonAngle: importGraphBuffer(
        graph,
        'horizon',
        buffers.horizonAngle,
        'float32',
        pixelCount
      ),
      outputValidity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', pixelCount)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    contributor,
    /** Packs `values` on `frameGrid` (default: 10 m square cells) and reads the outputs back. */
    async run(
      values: GPUTerrainCastShadowSettings,
      frameGrid: GPURasterMetadata = defaultGrid
    ): Promise<CastShadowResult> {
      // The packer takes no row direction or cell size mode: the contributor owns both.
      settings.write(
        getGPUTerrainCastShadowParameterValues({...options.sweepSettings, ...values}, frameGrid)
      );
      submitGraph(device, compiled, undefined);
      return {
        sunVisibility: await readFloat32(buffers.sunVisibility, pixelCount),
        horizonAngle: await readFloat32(buffers.horizonAngle, pixelCount),
        validity: await readUint32(buffers.validity, pixelCount)
      };
    },
    destroy() {
      compiled.destroy();
      settings.destroy();
      elevationBuffer.destroy();
      for (const buffer of Object.values(buffers)) buffer.destroy();
    }
  };
}

/** Pixel-space direction of the sun, float64, as packed by the settings packer. */
function getSunDirection(azimuthDegrees: number, rowDirection: 'south' | 'north' = 'south') {
  const azimuth = (azimuthDegrees * Math.PI) / 180;
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  return [
    snap(Math.sin(azimuth)),
    snap((rowDirection === 'south' ? -1 : 1) * Math.cos(azimuth))
  ] as [number, number];
}

function expectNaNPattern(actual: number[], expected: number[]): number {
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `NaN at ${index}`).toBe(true);
    } else {
      expect(Number.isFinite(actual[index]), `finite at ${index}`).toBe(true);
      worst = Math.max(worst, Math.abs(actual[index] - value));
    }
  }
  return worst;
}

it('RT-03 cast shadow agrees on a whole and cropped blocker with its supported halo', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fullGrid = makeGPURasterMetadata({width: 9, height: 9, cellSize: 1, rowDirection: 'north'});
  const windowGrid = makeGPURasterMetadata({
    width: 7,
    height: 7,
    cellSize: 1,
    rowDirection: 'north',
    origin: [1, 1]
  });
  const elevation = new Float32Array(81);
  elevation[5 * 9 + 4] = 10;
  const window = new Float32Array(49);
  window[4 * 7 + 3] = 10;
  const sun = {azimuthDegrees: 22.5, altitudeDegrees: 20, angularRadiusDegrees: 0};
  for (const shared of [false, true]) {
    const fullSettings = shared
      ? {
          sweepReferenceCellSize: [1, 1] as const,
          sweepSourceGrid: fullGrid,
          sweepOrigin: [0, 0] as const
        }
      : undefined;
    const windowSettings = shared ? {...fullSettings, sweepOrigin: [1, 1] as const} : undefined;
    const fullFixture = createCastShadowFixture(device, elevation, 9, 9, {
      maximumRadius: 2,
      rowDirection: 'north',
      sweepSettings: fullSettings
    });
    const windowFixture = createCastShadowFixture(device, window, 7, 7, {
      maximumRadius: 2,
      rowDirection: 'north',
      sweepSettings: windowSettings
    });
    try {
      if (shared) {
        expect(fullFixture.contributor.getHaloStage()).toEqual({requiredHalo: 2});
        expect(windowFixture.contributor.getHaloStage()).toEqual({requiredHalo: 2});
      } else {
        expect(() => fullFixture.contributor.getHaloStage()).toThrow(/parameterLayout/);
        expect(() => windowFixture.contributor.getHaloStage()).toThrow(/parameterLayout/);
      }
      const full = await fullFixture.run(sun, fullGrid);
      const cropped = await windowFixture.run(sun, windowGrid);
      const fullPixel = 4 * 9 + 4;
      const windowPixel = 3 * 7 + 3;
      expect(full.horizonAngle[fullPixel]).toBeCloseTo(83.822396, 3);
      expect(cropped.horizonAngle[windowPixel]).toBeCloseTo(shared ? 83.822396 : 0, 3);
      expect(full.sunVisibility[fullPixel]).toBe(0);
      expect(cropped.sunVisibility[windowPixel]).toBe(shared ? 0 : 1);
      expect(full.validity[fullPixel]).toBe(1);
      expect(cropped.validity[windowPixel]).toBe(1);
      for (const [actual, grid, values, sweepSettings] of [
        [full, fullGrid, elevation, fullSettings],
        [cropped, windowGrid, window, windowSettings]
      ] as const) {
        const expected = computeTerrainCastShadowOracle({
          width: grid.width,
          height: grid.height,
          elevation: values,
          parameters: getGPUTerrainCastShadowParameterValues({...sun, ...sweepSettings}, grid),
          maximumRadius: 2,
          rowDirection: 'north',
          parameterLayout: shared ? 'sweep' : 'standard'
        });
        expect(expectNaNPattern(actual.horizonAngle, expected.horizonAngle)).toBeLessThan(2e-3);
        expect(expectNaNPattern(actual.sunVisibility, expected.sunVisibility)).toBeLessThan(1e-6);
        expect(actual.validity).toEqual(expected.validity);
      }
    } finally {
      fullFixture.destroy();
      windowFixture.destroy();
    }
  }
});

it('RT-03 cast shadow uses a shared geographic source row at the distance cutoff', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fullGrid = makeGPURasterMetadata({
    width: 9,
    height: 11,
    origin: [0, 89],
    cellSize: [0.0001, 0.00001]
  });
  const windowGrid = makeGPURasterMetadata({
    width: 5,
    height: 5,
    origin: [0.0002, 88.99998],
    cellSize: [0.0001, 0.00001]
  });
  const elevation = new Float32Array(99);
  elevation[3 * 9 + 5] = 10;
  const window = new Float32Array(25);
  window[1 * 5 + 3] = 10;
  const sweepSettings = {sweepReferenceCellSize: [1, 1] as const, sweepSourceGrid: fullGrid};
  const fullFixture = createCastShadowFixture(device, elevation, 9, 11, {
    maximumRadius: 1,
    cellSizeMode: 'geographic',
    sweepSettings: {...sweepSettings, sweepOrigin: [0, 0]}
  });
  const windowFixture = createCastShadowFixture(device, window, 5, 5, {
    maximumRadius: 1,
    cellSizeMode: 'geographic',
    sweepSettings: {...sweepSettings, sweepOrigin: [2, 2]}
  });
  try {
    const sun = {
      azimuthDegrees: 90,
      altitudeDegrees: 20,
      angularRadiusDegrees: 0,
      maximumDistance: 0.19493955373764038
    };
    const full = await fullFixture.run(sun, fullGrid);
    const cropped = await windowFixture.run(sun, windowGrid);
    for (const [result, pixel] of [
      [full, 3 * 9 + 4],
      [cropped, 1 * 5 + 2]
    ] as const) {
      expect(result.horizonAngle[pixel]).toBeCloseTo(88.88322, 3);
      expect(result.sunVisibility[pixel]).toBe(0);
      expect(result.validity[pixel]).toBe(1);
    }
  } finally {
    fullFixture.destroy();
    windowFixture.destroy();
  }
});

it('RT-03 cast shadow updates placement and sun on one compiled shared-source sweep', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sourceGrid = makeGPURasterMetadata({
    width: 10,
    height: 10,
    cellSize: 1,
    rowDirection: 'north'
  });
  const elevation = new Float32Array(81);
  elevation[4 * 9 + 3] = 10;
  const fixture = createCastShadowFixture(device, elevation, 9, 9, {
    maximumRadius: 2,
    rowDirection: 'north',
    sweepSettings: {
      sweepReferenceCellSize: [1, 1],
      sweepSourceGrid: sourceGrid,
      sweepOrigin: [0, 0]
    }
  });
  try {
    for (const [origin, azimuthDegrees, angle, visibility] of [
      [[0, 0], 22.5, 0, 1],
      [[1, 1], 22.5, 83.822396, 0],
      [[1, 1], 90, 0, 1],
      [[0, 0], 22.5, 0, 1]
    ] as const) {
      const actual = await fixture.run(
        {azimuthDegrees, altitudeDegrees: 20, angularRadiusDegrees: 0, sweepOrigin: origin},
        makeGPURasterMetadata({width: 9, height: 9, cellSize: 1, rowDirection: 'north', origin})
      );
      expect(actual.horizonAngle[3 * 9 + 3]).toBeCloseTo(angle, 3);
      expect(actual.sunVisibility[3 * 9 + 3]).toBe(visibility);
      expect(actual.validity[3 * 9 + 3]).toBe(1);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUTerrainCastShadow matches the float64 brute force along the sun line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 52;
  const height = 44;
  const random = createRandom(4);
  const elevation = createSmoothTerrain(width, height, 6);
  for (let index = 0; index < elevation.length; index++) {
    elevation[index] += (random() - 0.5) * 6;
  }
  for (const hole of [8 * width + 9, 8 * width + 10, 30 * width + 40]) {
    elevation[hole] = NaN;
  }
  const fixture = createCastShadowFixture(device, elevation, width, height);
  const altitudeDegrees = 12;
  const angularRadiusDegrees = 0.5;
  const azimuths = [0, 37, 90, 135, 200, 270, 311];
  const horizons: number[][] = [];
  for (const azimuthDegrees of azimuths) {
    const settings = {azimuthDegrees, altitudeDegrees, angularRadiusDegrees};
    const actual = await fixture.run(settings);
    const expected = computeTerrainHorizonSweepBruteForce({
      width,
      height,
      elevation,
      directionCount: 1,
      directions: [getSunDirection(azimuthDegrees)],
      cellSize: [10, 10]
    });
    expect(expectNaNPattern(actual.horizonAngle, expected.horizon)).toBeLessThan(2e-3);
    // Structure: real occlusion and real visibility (not silent zeros).
    expect(actual.horizonAngle.filter(angle => angle > 3).length).toBeGreaterThan(50);
    const visibility = expected.horizon.map(angle =>
      Number.isNaN(angle)
        ? NaN
        : getVisibleDiskFraction(altitudeDegrees, angle, angularRadiusDegrees)
    );
    expect(expectNaNPattern(actual.sunVisibility, visibility)).toBeLessThan(2e-2);
    expect(actual.sunVisibility.filter(value => value > 0.99).length).toBeGreaterThan(100);
    expect(actual.sunVisibility.filter(value => value < 0.01).length).toBeGreaterThan(20);
    expect(actual.validity).toEqual(expected.validity);
    horizons.push(actual.horizonAngle);
  }
  // Per-frame azimuth changes reuse the single compiled graph and really change the output.
  expect(horizons[0]).not.toEqual(horizons[1]);
  expect(horizons[2]).not.toEqual(horizons[5]);
  fixture.destroy();
});

it('GPUTerrainCastShadow mirrors the sun line for north-advancing rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Only the contributor knows rowDirection 'north'; the brute force walks the north-row pixel
  // direction (sin az, +cos az) directly, which the node spec shows is the same line family.
  const width = 52;
  const height = 44;
  const random = createRandom(9);
  const elevation = createSmoothTerrain(width, height, 3);
  for (let index = 0; index < elevation.length; index++) {
    elevation[index] += (random() - 0.5) * 6;
  }
  const fixture = createCastShadowFixture(device, elevation, width, height, {
    rowDirection: 'north'
  });
  for (const azimuthDegrees of [0, 37, 135, 200, 311]) {
    const actual = await fixture.run({
      azimuthDegrees,
      altitudeDegrees: 12,
      angularRadiusDegrees: 0.5
    });
    const expected = computeTerrainHorizonSweepBruteForce({
      width,
      height,
      elevation,
      directionCount: 1,
      directions: [getSunDirection(azimuthDegrees, 'north')],
      cellSize: [10, 10]
    });
    expect(expectNaNPattern(actual.horizonAngle, expected.horizon)).toBeLessThan(2e-3);
    expect(actual.horizonAngle.filter(angle => angle > 3).length).toBeGreaterThan(50);
  }
  fixture.destroy();
});

it('GPUTerrainCastShadow follows a 45 degree ground sun azimuth on a 60 N geographic grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The settings packer gets no cell size mode; only the contributor is geographic. One 1000 m
  // pillar P = (42, 2) on flat ground and the observer O = (2, 22) (true ground bearing 45.048
  // degrees). python3 -I, WGS84 (a = 6378137, f = 1 / 298.257223563): at the reference row 12
  // (latitude 59.9995) the 45 degree pixel slope is -32823.70774773758 / 65536, so slopeFixed =
  // -32824 and the line from O meets P after 40 major steps. Horizon toward the sun =
  // atan(1000 / (40 * 78.926508719586)) = 17.57561688017635 degrees (row 22 step). The square-cell
  // line packed by the CPU (the pixel diagonal) misses P and would read 0.
  const width = 48;
  const height = 24;
  const elevation = new Float32Array(width * height);
  elevation[2 * width + 42] = 1000;
  const fixture = createCastShadowFixture(device, elevation, width, height, {
    maximumRadius: 46,
    cellSizeMode: 'geographic'
  });
  // 24 rows of 0.001 degrees from 60.012 to 59.988 north.
  const result = await fixture.run(
    {azimuthDegrees: 45, altitudeDegrees: 10, angularRadiusDegrees: 0.5},
    makeGPURasterMetadata({width, height, cellSize: 0.001, origin: [0, 60.012]})
  );
  const observer = 22 * width + 2;
  expect(result.horizonAngle[observer]).toBeCloseTo(17.57561688017635, 3);
  // The whole disk is below the 17.6 degree horizon (acos(-1) / PI in f32).
  expect(result.sunVisibility[observer]).toBeLessThan(1e-6);
  fixture.destroy();
});

it('GPUTerrainCastShadow casts a wall shadow of length H / tan(altitude)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 61;
  const height = 21;
  const elevation = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    elevation[row * width + 30] = 20;
  }
  const fixture = createCastShadowFixture(device, elevation, width, height);
  const altitudeDegrees = 30;
  // Wall of 20 m on 10 m cells: pixel k columns behind it has horizon atan(2 / k); it is shadowed
  // while that exceeds 30 degrees, i.e. k < 2 / tan(30) = 3.46 -> columns 31..33.
  const east = await fixture.run({
    azimuthDegrees: 270,
    altitudeDegrees,
    angularRadiusDegrees: 0
  });
  const row = 10;
  for (let column = 0; column < width; column++) {
    const index = row * width + column;
    const behind = column > 30 && column <= 33;
    expect(east.sunVisibility[index], `column ${column}`).toBe(behind ? 0 : 1);
    if (column > 30) {
      const expectedAngle = (Math.atan(2 / (column - 30)) * 180) / Math.PI;
      expect(east.horizonAngle[index]).toBeCloseTo(expectedAngle, 3);
    }
  }
  // Mirrored: sun in the east shadows the pixels west of the wall.
  const west = await fixture.run({
    azimuthDegrees: 90,
    altitudeDegrees,
    angularRadiusDegrees: 0
  });
  for (let column = 0; column < width; column++) {
    const behind = column < 30 && column >= 27;
    expect(west.sunVisibility[row * width + column], `column ${column}`).toBe(behind ? 0 : 1);
  }
  // A soft disk makes a penumbra at the shadow edge (column 34: horizon 26.6, alt 30, radius 5).
  const soft = await fixture.run({
    azimuthDegrees: 270,
    altitudeDegrees,
    angularRadiusDegrees: 5
  });
  const penumbra = soft.sunVisibility[row * width + 34];
  const edgeAngle = (Math.atan(2 / 4) * 180) / Math.PI;
  expect(penumbra).toBeCloseTo(getVisibleDiskFraction(altitudeDegrees, edgeAngle, 5), 4);
  expect(penumbra).toBeGreaterThan(0.1);
  expect(penumbra).toBeLessThan(0.95);
  fixture.destroy();
});

it('GPUTerrainCastShadow handles a sun below the horizon and a high sun on a plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 20;
  const height = 16;
  const plane = new Float32Array(width * height).fill(1234.5);
  plane[3 * width + 3] = NaN;
  const fixture = createCastShadowFixture(device, plane, width, height);
  const night = await fixture.run({
    azimuthDegrees: 123,
    altitudeDegrees: -10
  });
  const day = await fixture.run({azimuthDegrees: 123, altitudeDegrees: 60});
  for (let index = 0; index < width * height; index++) {
    if (index === 3 * width + 3) {
      expect(Number.isNaN(night.sunVisibility[index])).toBe(true);
      expect(Number.isNaN(day.sunVisibility[index])).toBe(true);
      expect(day.validity[index]).toBe(0);
      continue;
    }
    expect(night.sunVisibility[index]).toBe(0);
    expect(day.sunVisibility[index]).toBe(1);
    expect(day.horizonAngle[index]).toBe(0);
    expect(day.validity[index]).toBe(1);
  }
  fixture.destroy();
});

it('GPUTerrainCastShadow equals GPUSolarShadowMask fed by the sweep horizon at exact azimuths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 40;
  const height = 36;
  const random = createRandom(9);
  const elevation = createSmoothTerrain(width, height, 2);
  for (let index = 0; index < elevation.length; index++) {
    elevation[index] += (random() - 0.5) * 4;
  }
  const pixelCount = width * height;
  const directionCount = 8;
  const horizonResult = computeTerrainHorizonSweepBruteForce({
    width,
    height,
    elevation,
    directionCount,
    cellSize: [10, 10]
  });
  const graph = new GPUCommandGraph(device, {id: 'mask-test'});
  const horizonBuffer = createInputBuffer(device, Float32Array.from(horizonResult.horizon));
  const visibilityBuffer = createOutputBuffer(device, pixelCount);
  const maskSettings = new GPUParameterBuffer(device, {
    id: 'mask-settings',
    format: 'float32',
    length: 8,
    values: getGPUSolarShadowMaskParameterValues({azimuthDegrees: 0, altitudeDegrees: 0})
  });
  graph.add(
    new GPUSolarShadowMask({
      grid: makeGPURasterMetadata({width, height}),
      directionCount,
      horizon: importGraphBuffer(
        graph,
        'horizon',
        horizonBuffer,
        'float32',
        pixelCount * directionCount
      ),
      parameters: maskSettings.importToGraph(graph),
      output: {
        sunVisibility: importGraphBuffer(
          graph,
          'visibility',
          visibilityBuffer,
          'float32',
          pixelCount
        )
      }
    })
  );
  const compiled = graph.compile();
  const fixture = createCastShadowFixture(device, elevation, width, height);
  for (const azimuthDegrees of [0, 90, 135, 270]) {
    const sun = {azimuthDegrees, altitudeDegrees: 9, angularRadiusDegrees: 0.4};
    maskSettings.write(getGPUSolarShadowMaskParameterValues(sun));
    submitGraph(device, compiled, undefined);
    const mask = await readFloat32(visibilityBuffer, pixelCount);
    const cast = await fixture.run(sun);
    expect(cast.sunVisibility.filter(value => value > 0.99).length).toBeGreaterThan(100);
    expect(cast.sunVisibility.filter(value => value < 0.01).length).toBeGreaterThan(20);
    const worst = expectNaNPattern(cast.sunVisibility, mask);
    expect(worst, `azimuth ${azimuthDegrees}`).toBeLessThan(1e-3);
  }
  fixture.destroy();
  compiled.destroy();
  maskSettings.destroy();
  horizonBuffer.destroy();
  visibilityBuffer.destroy();
});
