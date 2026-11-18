// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUPointHorizonVisibilityParameterValues,
  GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE,
  GPUPointHorizonVisibility,
  type GPUPointHorizonModelOptions,
  type GPUPointHorizonVisibilitySettings
} from '../../../src/gpu-terrain/point-horizon/index';
import {GPU_TERRAIN_VISIBILITY as V} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createFractalTerrain, makeMercatorTestGrid} from './point-horizon-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type VisibilityConfig = {
  grid: GPURasterMetadata;
  values: Float32Array;
  observers: readonly (readonly [number, number, number])[];
  /** `[column, row, height, observerIndex]` */
  targets: readonly (readonly [number, number, number, number])[];
  settings: GPUPointHorizonVisibilitySettings;
  model: GPUPointHorizonModelOptions;
};

async function runVisibility(device: Device, config: VisibilityConfig) {
  const {grid, values, observers, targets, model} = config;
  const {width, height} = grid;
  const terrain = createInputBuffer(device, values);
  const observerBuffer = createInputBuffer(
    device,
    Float32Array.from(observers.flatMap(([column, row, eyeHeight]) => [column, row, eyeHeight, 0]))
  );
  const targetBuffer = createInputBuffer(device, Float32Array.from(targets.flat()));
  const settings = new GPUParameterBuffer(device, {
    id: 'visibility-settings',
    format: 'float32',
    length: 12,
    values: getGPUPointHorizonVisibilityParameterValues(config.settings, grid)
  });
  const visibility = createOutputBuffer(device, targets.length);
  const details = createOutputBuffer(device, targets.length * 4);
  const graph = new GPUCommandGraph(device, {id: 'point-horizon-visibility-test'});
  graph.add(
    new GPUPointHorizonVisibility({
      ...model,
      grid,
      elevation: {
        id: 'terrain',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'terrain', terrain, 'float32', width * height)
        }
      },
      observers: importGraphBuffer(
        graph,
        'observers',
        observerBuffer,
        'float32x4',
        observers.length
      ),
      targets: importGraphBuffer(graph, 'targets', targetBuffer, 'float32x4', targets.length),
      parameters: settings.importToGraph(graph),
      output: {
        visibility: importGraphBuffer(graph, 'visibility', visibility, 'uint32', targets.length),
        details: importGraphBuffer(graph, 'details', details, 'float32x4', targets.length)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const codes = await readUint32(visibility, targets.length);
  const rows = await readFloat32(details, targets.length * 4);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [terrain, observerBuffer, targetBuffer, visibility, details]) {
    buffer.destroy();
  }
  return {codes, rows};
}

const SIZE = 128;
const MODEL: GPUPointHorizonModelOptions = {
  projection: 'planar',
  maximumDistance: 1000,
  cellSize: 10
};
const GRID = makeGPURasterMetadata({width: SIZE, height: SIZE, cellSize: 10});
const SETTINGS: GPUPointHorizonVisibilitySettings = {};
const RIDGE_COLUMN = 40; // 200 m east of the observer at column 20
const RIDGE_HEIGHT = 30;
const OBSERVER: readonly [number, number, number] = [20, 64, 2];

function createRidgeTerrain(withRidge: boolean, peakGround = 0): Float32Array {
  const values = new Float32Array(SIZE * SIZE);
  if (withRidge) {
    for (let row = 0; row < SIZE; row++) {
      values[row * SIZE + RIDGE_COLUMN] = RIDGE_HEIGHT;
    }
  }
  values[64 * SIZE + 80] = peakGround;
  return values;
}

it('GPUPointHorizonVisibility classifies targets behind a ridge', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = (values: Float32Array, targets: VisibilityConfig['targets'], model = MODEL) =>
    runVisibility(device, {
      grid: GRID,
      values,
      observers: [OBSERVER],
      targets,
      settings: SETTINGS,
      model
    });
  // 600 m east of the observer. The ridge top subtends atan(28 / 200) from the eye.
  const behind = await run(createRidgeTerrain(true), [
    [80, 64, 0, 0], // ground level behind the ridge: hidden
    [80, 64, 86, 0] // tangent (86 - 2) / 600 equals the ridge tangent 28 / 200: marginal
  ]);
  expect(behind.codes[0]).toBe(V.hidden);
  expect(behind.codes[1]).toBe(V.marginal);
  const open = await run(createRidgeTerrain(false), [[80, 64, 100, 0]]);
  expect(open.codes[0]).toBe(V.visible);
  // Details: target angle, occluder angle (ridge), skyline angle.
  const ridgeAngle = (Math.atan(28 / 200) * 180) / Math.PI;
  expect(Math.abs(behind.rows[1] - ridgeAngle)).toBeLessThan(0.05);
  expect(Math.abs(behind.rows[5] - ridgeAngle)).toBeLessThan(0.05);
  expect(behind.rows[0]).toBeLessThan(0);
  // Clearly above the ridge: visible and on the skyline (it is the highest point of its ray).
  const peak = await run(createRidgeTerrain(true, 150), [[80, 64, 50, 0]]);
  expect(peak.codes[0]).toBe(V.visible);
  expect(peak.rows[3]).toBe(1);
  expect(peak.rows[0]).toBeGreaterThan(peak.rows[1]);
  // The ridge itself is a visible target but below the skyline of a higher peak behind it.
  const ridgeTarget = await run(createRidgeTerrain(true, 150), [[RIDGE_COLUMN, 64, 0, 0]]);
  expect(ridgeTarget.codes[0]).toBe(V.visible);
  expect(ridgeTarget.rows[3]).toBe(0);
});

it('GPUPointHorizonVisibility reports noData and outOfRange', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = createRidgeTerrain(false);
  const result = await runVisibility(device, {
    grid: GRID,
    values,
    observers: [OBSERVER, [-5, 10, 2]],
    targets: [
      [80, 64, 100, 0], // visible
      [125, 64, 0, 0], // 1050 m > maximumDistance 1000: outOfRange
      [80, 64, 0, 1], // observer outside the grid: noData
      [80, 64, 0, 7], // no such observer: noData
      [200, 64, 0, 0], // target outside the grid: noData
      [20, 64, 0, 0] // coincident with the observer: visible
    ],
    settings: SETTINGS,
    model: MODEL
  });
  expect(result.codes).toEqual([V.visible, V.outOfRange, V.noData, V.noData, V.noData, V.visible]);
  expect(result.rows[4 * 4]).toBeNaN();
  expect(result.rows[1 * 4]).toBeNaN();
});

it('GPUPointHorizonVisibility web mercator agrees with a planar equivalent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 192;
  const height = 192;
  const values = createFractalTerrain(width, height, 5, 500, 36);
  const worldPixelSize = 512 * 2 ** 12;
  const latitude = (50 * Math.PI) / 180;
  const originY =
    worldPixelSize * (0.5 - Math.asinh(Math.tan(latitude)) / (2 * Math.PI)) - 96 - 0.5;
  const targets: [number, number, number, number][] = [];
  for (const [column, row] of [
    [140, 96],
    [96, 40],
    [60, 150],
    [150, 150],
    [30, 30]
  ]) {
    targets.push([column, row, 5, 0], [column, row, 900, 0]);
  }
  const base = {
    values,
    observers: [[96, 96, 10] as const],
    targets
  };
  const mercator = await runVisibility(device, {
    ...base,
    grid: makeMercatorTestGrid(width, height, worldPixelSize, originY),
    settings: {projection: 'web-mercator', curvatureCoefficient: 2e-5},
    model: {projection: 'web-mercator', maximumDistance: 2500, cellSize: 13, maximumLatitude: 50}
  });
  // At 50 degrees north a Web Mercator pixel is 19.1 m * cos(50) = 12.3 m in both directions.
  const pixelMeters = (40075016.68557849 / worldPixelSize) * Math.cos(latitude);
  const planar = await runVisibility(device, {
    ...base,
    grid: makeGPURasterMetadata({width, height, cellSize: pixelMeters}),
    settings: {curvatureCoefficient: 2e-5},
    model: {projection: 'planar', maximumDistance: 2500, cellSize: 13}
  });
  const nonTrivial = new Set([...mercator.codes, ...planar.codes]);
  expect(nonTrivial.has(V.hidden)).toBe(true);
  expect(nonTrivial.has(V.visible)).toBe(true);
  // Target angles agree to the Mercator-versus-plane scale error at 2 km (about 1e-3 relative).
  for (let index = 0; index < targets.length; index++) {
    if (mercator.codes[index] === V.noData || planar.codes[index] === V.noData) {
      expect(mercator.codes[index]).toBe(planar.codes[index]);
      continue;
    }
    const tolerance = isSoftwareDevice(device) ? 2 : 0.1;
    expect(Math.abs(mercator.rows[index * 4] - planar.rows[index * 4])).toBeLessThan(tolerance);
    if (!isSoftwareDevice(device)) {
      expect(mercator.codes[index]).toBe(planar.codes[index]);
    }
  }
});

it('review C-05: the default ignore stretch is one ground pixel, so a near ridge still occludes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A 30 m ridge 100 m east of the eye (2 m above flat ground at column 20) and a ground-level
  // target 150 m east (column 35). The ridge subtends atan(28 / 100) = 15.6 degrees; the target is
  // at atan(-2 / 150) = -0.76 degrees, so it is hidden. With the old 150 m default the occlusion
  // march stopped at 150 - 150 = 0 m and the target came back visible.
  const values = new Float32Array(SIZE * SIZE);
  for (let row = 0; row < SIZE; row++) {
    values[row * SIZE + 30] = RIDGE_HEIGHT;
  }
  const run = (settings: GPUPointHorizonVisibilitySettings) =>
    runVisibility(device, {
      grid: GRID,
      values,
      observers: [OBSERVER],
      targets: [[35, 64, 0, 0]],
      settings,
      model: MODEL
    });
  // Default: stop = 150 - 10 m (one 10 m pixel along the east-west ray) includes the ridge.
  expect((await run(SETTINGS)).codes[0]).toBe(V.hidden);
  // The mt-image preset reproduces the previous behaviour.
  const preset = await run({
    ...SETTINGS,
    targetIgnoreDistance: GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE
  });
  expect(preset.codes[0]).toBe(V.visible);
});

it('GPUPointHorizonVisibility classes are ordered: a target exactly on the lower band edge is marginal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A target 600 m east at about 6 degrees, behind the ridge at about 7.97 degrees.
  const targets: VisibilityConfig['targets'] = [[80, 64, 65, 0]];
  const run = (toleranceDegrees: number) =>
    runVisibility(device, {
      grid: GRID,
      values: createRidgeTerrain(true),
      observers: [OBSERVER],
      targets,
      // sigmaZ 0 makes the tolerance exactly `toleranceDegrees`.
      settings: {toleranceDegrees, sigmaZ: 0},
      model: MODEL
    });
  // The angles do not depend on the tolerance: read the float32 values the kernel compares.
  const reference = await run(0.02);
  const targetAngle = reference.rows[0];
  const occluderAngle = reference.rows[1];
  expect(reference.codes[0]).toBe(V.hidden);
  expect(occluderAngle - targetAngle).toBeGreaterThan(1);
  // Both angles are within a factor of two, so their float32 difference is exact (Sterbenz) and
  // occluder - tolerance is exactly the target angle: the target sits on the lower band edge.
  const tolerance = Math.fround(occluderAngle - targetAngle);
  expect(Math.fround(occluderAngle - tolerance)).toBe(targetAngle);
  // On the edge: not below the band, so marginal. (The former rule tested |difference| < tolerance
  // and classed this target visible, between hidden and marginal.)
  expect((await run(tolerance)).codes[0]).toBe(V.marginal);
  // One float32 step narrower: the target is below the band. One step wider: inside it.
  const step = tolerance * 2 ** -22;
  expect((await run(Math.fround(tolerance - step))).codes[0]).toBe(V.hidden);
  expect((await run(Math.fround(tolerance + step))).codes[0]).toBe(V.marginal);
  // A band narrower than the gap above the occluder: a target above it by more is visible.
  const above = await runVisibility(device, {
    grid: GRID,
    values: createRidgeTerrain(true),
    observers: [OBSERVER],
    targets: [[80, 64, 120, 0]],
    settings: {toleranceDegrees: 0.02, sigmaZ: 0},
    model: MODEL
  });
  expect(above.codes[0]).toBe(V.visible);
});

it('GPUPointHorizonVisibility measures a Web Mercator tile in WGS84 ground metres, like the raster grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A zoom 12 window (512 px tiles) centred on the equator: one pixel is 2 pi a / worldPixelSize
  // = 19.109 EPSG:3857 metres. Expected distances below are closed forms on the WGS84 ellipsoid,
  // independent of the kernel: the meridian arc (integral of M) north-south and a times the
  // longitude difference along the equator. They are the row scales of analysis-core/contracts/raster-grid
  // (M cos(lat) / a and N cos(lat) / a per EPSG:3857 metre). The mean sphere of 6371008.8 m gives
  // 0.56% more north-south and 0.11% less east-west.
  const size = 256;
  const worldPixelSize = 512 * 2 ** 12;
  const originY = worldPixelSize / 2 - size / 2;
  const grid = makeMercatorTestGrid(size, size, worldPixelSize, originY);
  const semiMajorAxis = 6378137;
  const flattening = 1 / 298.257223563;
  const eccentricitySquared = flattening * (2 - flattening);
  const getLatitude = (row: number): number =>
    Math.atan(Math.sinh(Math.PI * (1 - (2 * (originY + row + 0.5)) / worldPixelSize)));
  const getMeridionalRadius = (latitude: number): number =>
    (semiMajorAxis * (1 - eccentricitySquared)) /
    (1 - eccentricitySquared * Math.sin(latitude) ** 2) ** 1.5;
  // Meridian arc between two rows by Simpson's rule.
  const getMeridianArc = (fromRow: number, toRow: number): number => {
    const start = getLatitude(fromRow);
    const end = getLatitude(toRow);
    const intervalCount = 200;
    const step = (end - start) / intervalCount;
    let sum = getMeridionalRadius(start) + getMeridionalRadius(end);
    for (let index = 1; index < intervalCount; index++) {
      sum += (index % 2 === 1 ? 4 : 2) * getMeridionalRadius(start + index * step);
    }
    return Math.abs((sum * step) / 3);
  };
  const pixelCount = 144;
  const northSouth = getMeridianArc(200, 200 - pixelCount);
  const eastWest = (2 * Math.PI * semiMajorAxis * pixelCount) / worldPixelSize;
  const sphereArc = (2 * Math.PI * 6371008.8 * pixelCount) / worldPixelSize;
  expect(northSouth).toBeCloseTo(2733.3, 0);
  expect(eastWest).toBeCloseTo(2751.7, 0);
  expect(sphereArc).toBeCloseTo(2748.7, 0);
  const values = new Float32Array(size * size);
  const run = (
    observer: readonly [number, number, number],
    target: readonly [number, number, number, number],
    maximumDistance: number
  ) =>
    runVisibility(device, {
      grid,
      values,
      observers: [observer],
      targets: [target],
      settings: {projection: 'web-mercator', maximumDistance, sigmaZ: 0},
      model: {projection: 'web-mercator', maximumDistance: 4000, cellSize: 19, maximumLatitude: 5}
    });
  // North-south: 144 rows due north of the eye. The cut sits between the WGS84 arc and the
  // sphere arc, so the target is in range (the sphere would call it out of range).
  const northObserver = [128, 200, 2] as const;
  const northTarget = [128, 200 - pixelCount, 302, 0] as const;
  const northInside = await run(northObserver, northTarget, northSouth + 5);
  expect(northInside.codes[0]).toBe(V.visible);
  expect((await run(northObserver, northTarget, northSouth - 5)).codes[0]).toBe(V.outOfRange);
  // The target angle uses the same ground distance: atan(300 / 2733.3 m) = 6.2637 degrees
  // (6.2291 on the sphere).
  const northAngle = (Math.atan(300 / northSouth) * 180) / Math.PI;
  expect(Math.abs(northInside.rows[0] - northAngle)).toBeLessThan(
    isSoftwareDevice(device) ? 0.05 : 3e-3
  );
  // East-west along the equator (row 127.5 is the equator): 144 pixels are 144 * 19.109 m of
  // ground, more than the sphere arc, so a cut between the two is out of range.
  const eastObserver = [56, 127.5, 2] as const;
  const eastTarget = [56 + pixelCount, 127.5, 302, 0] as const;
  expect((await run(eastObserver, eastTarget, eastWest - 1.5)).codes[0]).toBe(V.outOfRange);
  const eastInside = await run(eastObserver, eastTarget, eastWest + 1.5);
  expect(eastInside.codes[0]).toBe(V.visible);
  const eastAngle = (Math.atan(300 / eastWest) * 180) / Math.PI;
  expect(Math.abs(eastInside.rows[0] - eastAngle)).toBeLessThan(
    isSoftwareDevice(device) ? 0.05 : 3e-3
  );
});
