// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  getGPUReliefShadingParameterValues,
  GPUReliefShading
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {getGPUTerrainViewshedToleranceParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {getGPUTerrainLineOfSightParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {
  GPUSolarIrradiance,
  getGPUSolarIrradianceParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {resolvePointHorizonModel} from '../../../src/gpu-terrain/point-horizon/point-horizon-march';
import {GPUPointHorizonProfile} from '../../../src/gpu-terrain/point-horizon/gpu-point-horizon-profile';
import {GPUPointHorizonVisibility} from '../../../src/gpu-terrain/point-horizon/gpu-point-horizon-visibility';
import {
  createKernelAuditContext,
  recordShaderModules
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {computePointHorizonProfile} from '../point-horizon/point-horizon-oracle';
import {createBand} from '../terrain-test-utils';
import {POINT_PROFILE_GRID, POINT_PROFILE_ELEVATION} from './illumination-sight-w3-fixtures';

const GRID = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});

it('B3 clockwise-bin GPU fixture has an interior bilinear footprint and a 45 degree rise', () => {
  const model = resolvePointHorizonModel('fixture', POINT_PROFILE_GRID, {
    cellSize: 1,
    minimumDistance: 1,
    maximumDistance: 1,
    azimuthCount: 4,
    firstAzimuthIndex: 1,
    azimuthBinCount: 1,
    traversal: 'march'
  });
  const result = computePointHorizonProfile(
    {
      width: model.width,
      height: model.height,
      projection: model.projection,
      azimuthCount: model.azimuthCount,
      firstAzimuthIndex: model.firstAzimuthIndex,
      azimuthBinCount: model.azimuthBinCount,
      lattice: model.lattice,
      values: POINT_PROFILE_ELEVATION,
      curvature: 0,
      cellSize: [1, 1]
    },
    [[1, 1, 0]]
  );
  expect(Number.isFinite(result.skylineAngle[0])).toBe(true);
  expect(result.skylineAngle[0]).toBeCloseTo(45, 10); // atan(1 m / 1 m).
});

it('B4 preserves a descending contrast step whose encoded endpoints coincide', () => {
  expect(() =>
    getGPUReliefShadingParameterValues(
      {contrastLowElevation: 300, contrastHighElevation: 299.999999, contrastStrength: 1},
      GRID
    )
  ).not.toThrow();
});

it('B6 rejects out-of-range targetIgnoreFraction before either destination is written', () => {
  for (const value of [2, -0.1, 1.00000001]) {
    const target = new Float32Array(12).fill(7);
    expect(() =>
      getGPUTerrainViewshedToleranceParameterValues({targetIgnoreFraction: value}, target)
    ).toThrow(/targetIgnoreFraction/);
    expect(() =>
      getGPUTerrainLineOfSightParameterValues({targetIgnoreFraction: value}, GRID, target)
    ).toThrow(/targetIgnoreFraction/);
    expect(Array.from(target)).toEqual(new Array(12).fill(7));
  }
  for (const value of [0, 1]) {
    expect(getGPUTerrainViewshedToleranceParameterValues({targetIgnoreFraction: value})[3]).toBe(
      value
    );
    expect(getGPUTerrainLineOfSightParameterValues({targetIgnoreFraction: value}, GRID)[9]).toBe(
      value
    );
  }
});

it('S1 validates each standalone tolerance at float32 precision before writing', () => {
  for (const name of [
    'toleranceMeters',
    'tolerancePerKilometer',
    'targetIgnoreDistance',
    'targetIgnoreFraction'
  ]) {
    const target = new Float32Array(4).fill(7);
    expect(() => getGPUTerrainViewshedToleranceParameterValues({[name]: 2 ** 129}, target)).toThrow(
      new RegExp(name)
    );
    expect(Array.from(target)).toEqual([7, 7, 7, 7]);
  }
});

it('S2 permits the exact 2^30 row count whose final lane is 0xffffffff', () => {
  expect(getGPUSolarIrradianceParameterValues({sampleCount: 2 ** 30})[0]).toBe(2 ** 30);
  expect(() => getGPUSolarIrradianceParameterValues({sampleCount: 2 ** 30 + 128})).toThrow(
    /sampleCount/
  );
});

it('S2 includes the known scalar binding offset in the sun-table last-lane bound without allocating', () => {
  const context = createKernelAuditContext('sun-table-address');
  try {
    const capacity = 2 ** 30;
    const limits = context.device.limits as {
      maxBufferSize: number;
      maxStorageBufferBindingSize: number;
    };
    limits.maxBufferSize = 2 ** 36;
    limits.maxStorageBufferBindingSize = 2 ** 36;
    // These are logical graph descriptors; no buffers are allocated or compiled.
    const buffer = context.buffer(capacity * 16 + 4);
    const create = (byteOffset: number) =>
      new GPUSolarIrradiance({
        grid: GRID,
        directionCount: 4,
        horizon: context.view('float32', 36),
        parameters: context.view('float32', 8),
        sampleCapacity: capacity,
        sunTable: context.graph.createDataView(buffer, {
          format: 'float32',
          length: capacity * 4,
          byteOffset
        }),
        output: {sunHours: context.view('float32', 9)}
      });
    expect(() => create(0)).not.toThrow();
    expect(() => create(4)).toThrow(/sunTable.*uint32/);
  } finally {
    context.device.destroy();
  }
});

it('S3 resident weighting dispatch is based on raw enum bits', () => {
  const context = createKernelAuditContext('weighting-bits');
  try {
    const contributor = new GPUReliefShading({
      grid: GRID,
      elevation: createBand(context.graph, 'elevation', 9),
      parameters: context.view('float32', 80),
      output: {hillshade: context.view('float32', 9)}
    });
    const shaders: string[] = [];
    recordShaderModules([...contributor.getCommandNodes(context.graph)], module => {
      if (module.nodeId.includes('hillshade')) shaders.push(module.source);
    });
    for (const source of shaders)
      expect(source).toContain('bitcast<u32>(settings[settingsOffset + 6u])');
  } finally {
    context.device.destroy();
  }
});

it('S5 packed horizon indices remain positive beyond the signed 32-bit boundary', () => {
  const source = readFileSync(
    new URL(
      '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon.ts',
      import.meta.url
    ),
    'utf8'
  );
  const expression = source.match(/const word = words\[(.*?)\];/)![1];
  const getIndex = new Function('element', `return ${expression};`) as (element: number) => number;
  // Evaluate the exact decoder indexing expression without allocating 2^31 output rows.
  expect(getIndex(2 ** 31)).toBe(1073741824);
  expect(getIndex(2 ** 32 + 1)).toBe(2147483648);
});

it('earlier budget fixes capture profile and visibility chunk sizes before caller mutation', () => {
  const context = createKernelAuditContext('captured-chunks');
  try {
    const common = {
      grid: GRID,
      elevation: createBand(context.graph, 'elevation', 9),
      cellSize: 1,
      minimumDistance: 1,
      maximumDistance: 10,
      traversal: 'march' as const,
      observers: context.view('float32x4', 1)
    };
    const profileProps = {
      ...common,
      parameters: context.view('float32', 8),
      azimuthCount: 8,
      azimuthBinCount: 5,
      raysPerDispatch: 5,
      output: {tangent: context.view('float32', 5)}
    };
    const profile = new GPUPointHorizonProfile(profileProps);
    profileProps.raysPerDispatch = 1;
    expect(
      profile.getCommandNodes(context.graph).filter(node => /-march-\d+$/.test(node.id))
    ).toHaveLength(1);
    const visibilityProps = {
      ...common,
      targets: context.view('float32x4', 5),
      parameters: context.view('float32', 12),
      targetsPerDispatch: 5,
      output: {visibility: context.view('uint32', 5)}
    };
    const visibility = new GPUPointHorizonVisibility(visibilityProps);
    visibilityProps.targetsPerDispatch = 1;
    expect(
      visibility.getCommandNodes(context.graph).filter(node => /-classify-\d+$/.test(node.id))
    ).toHaveLength(1);
    const allOutputs = new GPUPointHorizonProfile({
      ...profileProps,
      id: 'all-outputs',
      traversal: 'pyramid',
      raysPerDispatch: 5,
      output: {
        tangent: context.view('float32', 5),
        skylineAngle: context.view('float32', 5),
        distance: context.view('float32', 5),
        samples: context.view('uint32', 5)
      }
    });
    expect(
      allOutputs.getCommandNodes(context.graph).some(node => node.id.endsWith('-skyline-angle'))
    ).toBe(true);
  } finally {
    context.device.destroy();
  }
});
