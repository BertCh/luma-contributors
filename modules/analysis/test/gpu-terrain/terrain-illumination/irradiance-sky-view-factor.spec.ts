// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSolarIrradianceParameterValues,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPUSolarIrradiance
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {
  getGPUTerrainHorizonParameterValues,
  GPUTerrainHorizon,
  type GPUTerrainHorizonAlgorithm,
  type GPUTerrainHorizonFormat
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Expected values here are closed forms for an isotropic sky (Dozier and Frew 1990), not a copy of
// the kernel. Under a horizon at elevation h in every azimuth, the irradiance on a horizontal
// surface is the integral of L cos(t) sin(t) over zenith angles t in [0, 90 - h], which is
// pi L cos^2(h); the open-sky value is pi L. The visible solid angle is 2 pi (1 - sin(h)) of 2 pi.
// At h = 30 degrees: irradiance fraction 0.75, solid-angle fraction 0.5.

const CELL_SIZE = 10;
const WIDTH = 21;
const HEIGHT = 21;
const PIXEL_COUNT = WIDTH * HEIGHT;
const CENTER = 10 * WIDTH + 10;

/** A cone-shaped pit: the centre sees a horizon of exactly `horizonDegrees` along the four axes. */
function createConePit(horizonDegrees: number): Float32Array {
  const tangent = Math.tan((horizonDegrees * Math.PI) / 180);
  return Float32Array.from({length: PIXEL_COUNT}, (_, pixel) => {
    const column = (pixel % WIDTH) - 10;
    const row = Math.floor(pixel / WIDTH) - 10;
    return 100 + tangent * CELL_SIZE * Math.hypot(column, row);
  });
}

type SkyViewRun = {
  skyViewFactor: number[];
  irradianceSkyViewFactor: number[];
  insolation?: number[];
};

async function runSkyView(
  device: Device,
  elevation: Float32Array,
  options: {
    directionCount: number;
    maximumRadius: number;
    algorithm?: GPUTerrainHorizonAlgorithm;
    /** Bind `output.horizon` in this format; omitted leaves the horizon map to the contributor. */
    horizonFormat?: GPUTerrainHorizonFormat;
    /** Chain a `GPUSolarIrradiance` with this diffuse irradiance (one sun row of one hour). */
    diffuseIrradiance?: number;
  }
): Promise<SkyViewRun> {
  const {directionCount} = options;
  const grid = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: CELL_SIZE});
  const graph = new GPUCommandGraph(device, {id: 'irradiance-sky-view-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const skyViewBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const irradianceSkyViewBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const insolationBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const horizonLength =
    options.horizonFormat === 'unorm16'
      ? Math.ceil((PIXEL_COUNT * directionCount) / 2)
      : PIXEL_COUNT * directionCount;
  const horizonBuffer = createOutputBuffer(device, horizonLength);
  const horizonSettings = new GPUParameterBuffer(device, {
    id: 'horizon-parameters',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHorizonParameterValues({}, grid)
  });
  const irradianceSkyView = importGraphBuffer(
    graph,
    'irradiance-sky-view',
    irradianceSkyViewBuffer,
    'float32',
    PIXEL_COUNT
  );
  const horizonView = importGraphBuffer(
    graph,
    'horizon',
    horizonBuffer,
    options.horizonFormat === 'unorm16' ? 'uint32' : 'float32',
    horizonLength
  );
  const needsHorizon = Boolean(options.horizonFormat) || options.diffuseIrradiance !== undefined;
  graph.add(
    new GPUTerrainHorizon({
      grid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      parameters: horizonSettings.importToGraph(graph),
      directionCount,
      maximumRadius: options.maximumRadius,
      algorithm: options.algorithm,
      horizonFormat: options.horizonFormat,
      output: {
        horizon: needsHorizon ? horizonView : undefined,
        skyViewFactor: importGraphBuffer(graph, 'sky-view', skyViewBuffer, 'float32', PIXEL_COUNT),
        irradianceSkyViewFactor: irradianceSkyView
      }
    })
  );
  const sunTableBuffer = createInputBuffer(
    device,
    // One row: sun due south at 60 degrees altitude for one hour, no direct beam.
    new Float32Array([180, 60, 1, 0])
  );
  const irradianceSettings = new GPUParameterBuffer(device, {
    id: 'irradiance-parameters',
    format: 'float32',
    length: GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
    values: getGPUSolarIrradianceParameterValues({
      sampleCount: 1,
      diffuseIrradiance: options.diffuseIrradiance ?? 0
    })
  });
  if (options.diffuseIrradiance !== undefined) {
    graph.add(
      new GPUSolarIrradiance({
        grid,
        directionCount,
        horizonFormat: options.horizonFormat,
        horizon: horizonView,
        sunTable: importGraphBuffer(graph, 'sun-table', sunTableBuffer, 'float32', 4),
        sampleCapacity: 1,
        parameters: irradianceSettings.importToGraph(graph),
        skyViewFactor: irradianceSkyView,
        output: {
          insolation: importGraphBuffer(
            graph,
            'insolation',
            insolationBuffer,
            'float32',
            PIXEL_COUNT
          )
        }
      })
    );
  }
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: SkyViewRun = {
    skyViewFactor: await readFloat32(skyViewBuffer, PIXEL_COUNT),
    irradianceSkyViewFactor: await readFloat32(irradianceSkyViewBuffer, PIXEL_COUNT),
    insolation:
      options.diffuseIrradiance !== undefined
        ? await readFloat32(insolationBuffer, PIXEL_COUNT)
        : undefined
  };
  compiled.destroy();
  horizonSettings.destroy();
  irradianceSettings.destroy();
  for (const buffer of [
    elevationBuffer,
    skyViewBuffer,
    irradianceSkyViewBuffer,
    insolationBuffer,
    horizonBuffer,
    sunTableBuffer
  ]) {
    buffer.destroy();
  }
  return result;
}

it('GPUTerrainHorizon writes cos^2 for irradiance and 1 - sin for solid angle under a uniform horizon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [horizonDegrees, irradianceFraction, solidAngleFraction] of [
    [30, 0.75, 0.5],
    // cos^2(10 deg) = 0.969846, 1 - sin(10 deg) = 0.826352.
    [10, 0.969846, 0.826352]
  ] as const) {
    const elevation = createConePit(horizonDegrees);
    for (const variant of [
      {algorithm: 'march' as const, horizonFormat: undefined, tolerance: 2e-6},
      {algorithm: 'march' as const, horizonFormat: 'float32' as const, tolerance: 2e-6},
      // Quantised angles: half a code step is 2.4e-5 radians.
      {algorithm: 'march' as const, horizonFormat: 'unorm16' as const, tolerance: 3e-5},
      {algorithm: 'sweep' as const, horizonFormat: undefined, tolerance: 2e-6}
    ]) {
      const result = await runSkyView(device, elevation, {
        directionCount: 4,
        maximumRadius: 8,
        algorithm: variant.algorithm,
        horizonFormat: variant.horizonFormat
      });
      const label = `${horizonDegrees} degrees ${variant.algorithm} ${variant.horizonFormat}`;
      expect(
        Math.abs(result.irradianceSkyViewFactor[CENTER] - irradianceFraction),
        label
      ).toBeLessThan(variant.tolerance);
      // The existing output keeps its definition.
      expect(Math.abs(result.skyViewFactor[CENTER] - solidAngleFraction), label).toBeLessThan(2e-6);
    }
  }
});

it('GPUTerrainHorizon irradiance sky-view factor is 1 on flat ground and (1 + cos s) / 2 on a plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const flat = await runSkyView(device, new Float32Array(PIXEL_COUNT).fill(250), {
    directionCount: 16,
    maximumRadius: 6
  });
  expect(flat.irradianceSkyViewFactor.every(value => value === 1)).toBe(true);

  // A plane rising east with gradient 0.5 (slope s = atan(0.5)). Along compass azimuth t the
  // plane is its own horizon with tan(h) = 0.5 sin(t), so sin^2(h) = x^2 / (1 + x^2) with
  // x = max(0.5 sin(t), 0). The mean over all azimuths is (1 - cos(s)) / 2 exactly:
  // (1 / 2 pi) * integral of x^2 / (1 + x^2) over the uphill half circle = (1 - 1 / sqrt(1.25)) / 2.
  const gradient = 0.5;
  const plane = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, pixel) => 100 + gradient * CELL_SIZE * (pixel % WIDTH)
  );
  const directionCount = 16;
  const result = await runSkyView(device, plane, {directionCount, maximumRadius: 6});
  let sineSquaredSum = 0;
  for (let sector = 0; sector < directionCount; sector++) {
    const tangent = Math.max(gradient * Math.sin((2 * Math.PI * sector) / directionCount), 0);
    sineSquaredSum += (tangent * tangent) / (1 + tangent * tangent);
  }
  const sectorMean = 1 - sineSquaredSum / directionCount;
  const cosSlope = 1 / Math.sqrt(1 + gradient * gradient);
  const closedForm = (1 + cosSlope) / 2;
  expect(Math.abs(result.irradianceSkyViewFactor[CENTER] - sectorMean)).toBeLessThan(2e-6);
  // Sixteen sectors integrate the smooth periodic integrand almost exactly.
  expect(Math.abs(sectorMean - closedForm)).toBeLessThan(1e-4);
  expect(Math.abs(result.irradianceSkyViewFactor[CENTER] - closedForm)).toBeLessThan(1e-4);
});

it('GPUSolarIrradiance diffuse energy under a uniform 30 degree horizon is 75% of the open-site value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 100 W/m^2 of diffuse horizontal irradiance for one hour: 100 Wh/m^2 in the open, and
  // 100 * cos^2(30 deg) = 75 Wh/m^2 at the bottom of the pit. The solid-angle sky-view factor
  // would give 50.
  for (const horizonFormat of ['float32', 'unorm16'] as const) {
    const result = await runSkyView(device, createConePit(30), {
      directionCount: 4,
      maximumRadius: 8,
      horizonFormat,
      diffuseIrradiance: 100
    });
    expect(Math.abs(result.insolation![CENTER] - 75), horizonFormat).toBeLessThan(4e-3);
  }
});
