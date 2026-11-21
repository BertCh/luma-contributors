// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUReliefShading,
  getGPUReliefShadingParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  createKernelAuditContext,
  recordShaderModules
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {createBand} from '../terrain-test-utils';
import {getGPUTerrainDerivativesParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {getGPUTerrainViewshedParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {getGPUTerrainViewshedToleranceParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {getGPUTerrainLineOfSightParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {unpackGPUTerrainHorizonUnorm16} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {getSolarPosition} from '../../../src/gpu-terrain/terrain-illumination/solar-position';
import {getGPUTerrainHorizonParameterValues} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-parameters';
import {resolvePointHorizonModel} from '../../../src/gpu-terrain/point-horizon/point-horizon-march';
import {GPUPointHorizonVisibility} from '../../../src/gpu-terrain/point-horizon/gpu-point-horizon-visibility';
import {
  computeTerrainViewshed,
  computeTerrainViewshedWithTolerance,
  computeTerrainSightLine,
  computeTerrainDerivatives
} from '../terrain-analysis/terrain-analysis-oracle';
import {
  getGPUSolarIrradianceSunTable,
  getGPUSolarIrradianceParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';

const GRID = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});

it('DEF-03 the ordinary relief shader includes resident Imhof dispatch and unsupported-code invalidation', () => {
  const context = createKernelAuditContext('ordinary-imhof');
  try {
    const contributor = new GPUReliefShading({
      grid: GRID,
      elevation: createBand(context.graph, 'elevation', 9),
      parameters: context.view('float32', 80),
      output: {hillshade: context.view('float32', 9)}
    });
    const sources: string[] = [];
    recordShaderModules([...contributor.getCommandNodes(context.graph)], module => {
      if (module.nodeId.endsWith('-hillshade')) sources.push(module.source);
    });
    expect(sources).toHaveLength(1);
    expect(sources[0]).toContain('if (swingWeighting)');
    expect(sources[0]).toContain('(fixedWeighting || aspectWeighting || swingWeighting)');
  } finally {
    context.device.destroy();
  }
});

it('URT-05 direct and tolerance CPU references classify missing heights as noData', () => {
  const terrain = [0, 10, 0];
  const grid = makeGPURasterMetadata({width: 3, height: 1, cellSize: 1});
  const parameters = getGPUTerrainViewshedParameterValues({observer: [0, 0]}, grid);
  const sight = getGPUTerrainLineOfSightParameterValues({}, grid);
  for (const slot of [2, 3]) {
    const invalid = parameters.slice();
    invalid[slot] = NaN;
    expect(computeTerrainViewshed(terrain, undefined, 3, 1, invalid)).toEqual([3, 3, 3]);
    expect(
      computeTerrainViewshedWithTolerance(terrain, undefined, 3, 1, invalid, [0, 0, 0, 0]).map(
        result => result.code
      )
    ).toEqual([3, 3, 3]);
  }
  for (const heights of [
    [NaN, 0],
    [0, NaN]
  ] as const) {
    expect(
      computeTerrainSightLine(terrain, undefined, 3, 1, [0, 0], [2, 0], sight, heights).code
    ).toBe(3);
    expect(
      computeTerrainSightLine(terrain, undefined, 3, 1, [0, 0], [0, 0], sight, heights).code
    ).toBe(3);
  }
  const lighting = getGPUTerrainDerivativesParameterValues({}, GRID);
  lighting[3] = NaN;
  expect(
    computeTerrainDerivatives(new Float32Array(9), undefined, 3, 3, lighting).validity
  ).toEqual(new Array(9).fill(0));
});

it('W3-14 uses identical tolerance bytes for omitted and empty settings', () => {
  expect(getGPUTerrainViewshedToleranceParameterValues()).toEqual(
    getGPUTerrainViewshedToleranceParameterValues({})
  );
});

it('ING-18 distinguishes active solar rows from padded allocation and names the format', () => {
  const start = Date.parse('2026-06-21T12:00:00Z');
  const target = new Float32Array(12).fill(7);
  const table = getGPUSolarIrradianceSunTable({
    longitude: 0,
    latitude: 0,
    start,
    end: start + 60000,
    target
  });
  expect(table.values).toBe(target);
  expect(table.sampleCount).toBe(1);
  expect(table.counts).toEqual({sampleCount: 1, sampleCapacity: 3});
  expect(table.formats).toEqual({sunTable: 'float32'});
  expect(Array.from(target.slice(4))).toEqual(new Array(8).fill(7));
});

it('UNI-19 resolves explicit RVT degrees and clockwise bin selections and rejects retired props', () => {
  const values = getGPUTerrainHorizonParameterValues({rvtDirectionDegrees: 90}, GRID);
  expect(values.length).toBe(12);
  expect(values[8]).toBe(90);
  const retiredDirection = {rvtDirectionDegrees: 90, anisotropyAzimuthDegrees: 90};
  expect(() => getGPUTerrainHorizonParameterValues(retiredDirection, GRID)).toThrow(
    /anisotropyAzimuthDegrees.*removed.*rvtDirectionDegrees/
  );
  const options = {
    maximumDistance: 10,
    minimumDistance: 1,
    cellSize: 1,
    azimuthCount: 8,
    firstAzimuthIndex: 2,
    azimuthBinCount: 3
  };
  const model = resolvePointHorizonModel('bins', GRID, options);
  // Eight full-cycle bins, clockwise from grid north: bin 2 is east (90 degrees).
  expect([model.azimuthCount, model.firstAzimuthIndex, model.azimuthBinCount]).toEqual([8, 2, 3]);
  const retiredFirstBin = {...options, firstAzimuth: 2};
  expect(() => resolvePointHorizonModel('bins', GRID, retiredFirstBin)).toThrow(
    /firstAzimuth.*removed.*firstAzimuthIndex/
  );
  const retiredBinCount = {...options, azimuthSpan: 3};
  expect(() => resolvePointHorizonModel('bins', GRID, retiredBinCount)).toThrow(
    /azimuthSpan.*removed.*azimuthBinCount/
  );
  const context = createKernelAuditContext('visibility-aliases');
  try {
    const props = {
      ...options,
      grid: GRID,
      elevation: createBand(context.graph, 'elevation', 9),
      observers: context.view('float32x4', 1),
      targets: context.view('float32x4', 1),
      parameters: context.view('float32', 12),
      output: {visibility: context.view('uint32', 1)}
    };
    expect(() => new GPUPointHorizonVisibility(props)).not.toThrow();
    expect(() => new GPUPointHorizonVisibility({...props, ...retiredFirstBin})).toThrow(
      /firstAzimuth.*removed.*firstAzimuthIndex/
    );
    const inheritedRetiredBinCount: typeof props = Object.create(retiredBinCount);
    Object.assign(inheritedRetiredBinCount, props);
    expect(() => new GPUPointHorizonVisibility(inheritedRetiredBinCount)).toThrow(
      /azimuthSpan.*removed.*azimuthBinCount/
    );
  } finally {
    context.device.destroy();
  }
});

it('W3-13 rejects active relief ramps that collapse around 100 after float32 encoding', () => {
  const settings = {
    contrastLowElevation: 99.999999,
    contrastHighElevation: 100.000001,
    contrastStrength: 1
  };
  expect(() => getGPUReliefShadingParameterValues(settings, GRID)).toThrow(
    /99\.999999.*100\.000001.*100/
  );
  expect(() =>
    getGPUReliefShadingParameterValues({...settings, contrastStrength: 0}, GRID)
  ).not.toThrow();
  expect(() =>
    getGPUReliefShadingParameterValues(
      {...settings, contrastLowElevation: 100, contrastHighElevation: 100},
      GRID
    )
  ).not.toThrow();
  expect(() =>
    getGPUReliefShadingParameterValues(
      {
        elevationStops: [
          {elevation: 99.999999, color: [0, 0, 0]},
          {elevation: 100.000001, color: [1, 1, 1]}
        ]
      },
      GRID
    )
  ).toThrow(/99\.999999.*100\.000001.*100/);
});

it('ING-09 requires finite float32 controls and exact consumer row counts without allocating a table', () => {
  for (const sampleCount of [16777217, 16777219, 2 ** 30 + 128, 2 ** 32]) {
    expect(() => getGPUSolarIrradianceParameterValues({sampleCount})).toThrow(/sampleCount/);
  }
  // The last row of an exact 2^30-row table uses lane 4 * (2^30 - 1) + 3 = 0xffffffff.
  for (const sampleCount of [0, 16777216, 16777218, 2 ** 30]) {
    expect(getGPUSolarIrradianceParameterValues({sampleCount})[0]).toBe(sampleCount);
  }
  for (const field of ['angularRadiusDegrees', 'diffuseIrradiance']) {
    const target = new Float32Array(8).fill(7);
    expect(() =>
      getGPUSolarIrradianceParameterValues({sampleCount: 1, [field]: 1e40}, target)
    ).toThrow(new RegExp(field));
    expect(Array.from(target)).toEqual(new Array(8).fill(7));
  }
  const start = Date.parse('2026-06-01T12:00:00Z');
  const target = new Float32Array(4).fill(7);
  expect(() =>
    getGPUSolarIrradianceSunTable({
      longitude: 0,
      latitude: 0,
      start,
      end: start + 60000,
      directNormalIrradiance: 1e40,
      target
    })
  ).toThrow(/directNormalIrradiance/);
  expect(Array.from(target)).toEqual([7, 7, 7, 7]);
});

it('UNI-11 rejects invalid scalar and shared-table coordinates without touching the target', () => {
  const start = Date.parse('2026-06-21T12:00:00Z');
  for (const [longitude, latitude, field] of [
    [0, 91, 'latitude'],
    [0, -91, 'latitude'],
    [Infinity, 0, 'longitude'],
    [0, NaN, 'latitude']
  ] as const) {
    expect(() => getSolarPosition(start, longitude, latitude)).toThrow(new RegExp(field));
    const target = new Float32Array(8).fill(7);
    expect(() =>
      getGPUSolarIrradianceSunTable({longitude, latitude, start, end: start + 60000, target})
    ).toThrow(new RegExp(field));
    expect(Array.from(target)).toEqual(new Array(8).fill(7));
  }
  for (const latitude of [-90, 90]) {
    expect(Number.isFinite(getSolarPosition(start, 0, latitude).altitudeDegrees)).toBe(true);
  }
});

it('OUT-02 decodes offset byte readbacks and rejects truncated horizon words', () => {
  const bytes = new Uint8Array([0, 255, 255, 255, 255, 0]).subarray(1, 5);
  // Each 0xffff halfword is exactly the +90 degree endpoint.
  expect(Array.from(unpackGPUTerrainHorizonUnorm16({bytes}, 2))).toEqual([90, 90]);
  expect(Array.from(unpackGPUTerrainHorizonUnorm16(bytes, 2))).toEqual([90, 90]);
  expect(Array.from(unpackGPUTerrainHorizonUnorm16([0xffffffff], 2))).toEqual([90, 90]);
  expect(() => unpackGPUTerrainHorizonUnorm16([0xffffffff], 3)).toThrow(
    /unpackGPUTerrainHorizonUnorm16.*2.*words/
  );
  for (const count of [-1, 1.5, NaN]) {
    expect(() => unpackGPUTerrainHorizonUnorm16([], count)).toThrow(/elementCount/);
  }
  expect(() => unpackGPUTerrainHorizonUnorm16({bytes: bytes.subarray(1)}, 1)).toThrow(/whole/);
});

it('DEF-03 rejects unknown light weighting instead of silently selecting fixed', () => {
  expect(() =>
    getGPUReliefShadingParameterValues({lightWeighting: 'other' as never}, GRID)
  ).toThrow(/fixed.*aspect.*imhof-swing/);
});

it('URT-05 rejects missing and float32-overflowing physical settings before writing', () => {
  for (const value of [NaN, Infinity, 1e40]) {
    for (const name of [
      'observerHeight',
      'targetHeight',
      'maximumDistance',
      'curvatureCoefficient'
    ]) {
      const target = new Float32Array(12).fill(7);
      expect(() =>
        getGPUTerrainViewshedParameterValues({observer: [0, 0], [name]: value}, GRID, target)
      ).toThrow(new RegExp(name));
      expect(Array.from(target)).toEqual(new Array(12).fill(7));
      expect(() => getGPUTerrainLineOfSightParameterValues({[name]: value}, GRID, target)).toThrow(
        new RegExp(name)
      );
      expect(Array.from(target)).toEqual(new Array(12).fill(7));
    }
    for (const name of ['zFactor', 'azimuthDegrees', 'altitudeDegrees']) {
      expect(() => getGPUTerrainDerivativesParameterValues({[name]: value}, GRID)).toThrow(
        new RegExp(name)
      );
    }
  }
});
