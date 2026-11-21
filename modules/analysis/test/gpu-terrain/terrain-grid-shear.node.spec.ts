// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPUGeomorphons} from '../../src/gpu-terrain/geomorphons/gpu-geomorphons';
import {GPUPointHorizonProfile} from '../../src/gpu-terrain/point-horizon/gpu-point-horizon-profile';
import {GPUPointHorizonVisibility} from '../../src/gpu-terrain/point-horizon/gpu-point-horizon-visibility';
import {GPUTerrainCumulativeViewshed} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed';
import {GPUTerrainLineOfSight} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {GPUTerrainViewshed} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {GPUTerrainCurvature} from '../../src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {GPUTerrainPeakSnap} from '../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {GPUTerrainSummits} from '../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {GPUTerrainFlowField} from '../../src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field';
import {GPUReliefShading} from '../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {GPUTerrainCastShadow} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {GPUTerrainHorizon} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {GPUTerrainVectorRuggedness} from '../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {GPUTerrainWeissLandforms} from '../../src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createBand} from './terrain-test-utils';

const WIDTH = 9;
const HEIGHT = 9;
const PIXEL_COUNT = WIDTH * HEIGHT;
const AXIS_ALIGNED_GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10});
// Column axis (a, d) = (10, 0), row axis (b, e) = (10, -10): normalized dot product 0.707, so the
// two diagonals are 10 and 10 * sqrt(5) long, not the 10 * sqrt(3) the per-axis lengths imply.
const SHEARED_GRID: GPURasterMetadata = {...AXIS_ALIGNED_GRID, affine: [10, 10, 0, 0, -10, 90]};
// Column axis (6, 8), row axis (8, -6): perpendicular, both 10 long, rows still advance south.
const ROTATED_GRID: GPURasterMetadata = {...AXIS_ALIGNED_GRID, affine: [6, 8, 0, 8, -6, 90]};

it('RT-102 terrain contributors that measure along both pixel axes reject a sheared grid', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = (length: number = PIXEL_COUNT) =>
    createTransientView(graph, `view-${++instance}`, 'float32', length);
  const words = (length: number = PIXEL_COUNT) =>
    createTransientView(graph, `words-${++instance}`, 'uint32', length);
  const pairs = <Format extends 'float32x2' | 'float32x4'>(format: Format, length: number) =>
    createTransientView(graph, `pairs-${++instance}`, format, length);
  const band = () => createBand(graph, `elevation-${++instance}`, PIXEL_COUNT);
  const factories: Record<string, (grid: GPURasterMetadata) => unknown> = {
    GPUTerrainCurvature: grid =>
      new GPUTerrainCurvature({
        grid,
        elevation: band(),
        parameters: view(12),
        output: {curvatures: {mean: view()}}
      }),
    GPUGeomorphons: grid =>
      new GPUGeomorphons({
        grid,
        searchRadius: 3,
        elevation: band(),
        parameters: view(8),
        output: {forms: words()}
      }),
    GPUTerrainVectorRuggedness: grid =>
      new GPUTerrainVectorRuggedness({
        grid,
        elevation: band(),
        parameters: view(8),
        output: {vectorRuggedness: view()}
      }),
    GPUTerrainWeissLandforms: grid =>
      new GPUTerrainWeissLandforms({
        grid,
        smallScale: {radius: 1, innerRadius: 0},
        largeScale: {radius: 2, innerRadius: 0},
        standardization: 'local',
        elevation: band(),
        parameters: view(8),
        output: {landforms: words()}
      }),
    GPUReliefShading: grid =>
      new GPUReliefShading({
        grid,
        elevation: band(),
        parameters: view(80),
        output: {hillshade: view()}
      }),
    GPUTerrainHorizon: grid =>
      new GPUTerrainHorizon({
        grid,
        elevation: band(),
        parameters: view(8),
        maximumRadius: 3,
        output: {skyViewFactor: view()}
      }),
    GPUTerrainCastShadow: grid =>
      new GPUTerrainCastShadow({
        grid,
        elevation: band(),
        parameters: view(16),
        output: {sunVisibility: view()}
      }),
    GPUTerrainViewshed: grid =>
      new GPUTerrainViewshed({
        grid,
        elevation: band(),
        parameters: view(12),
        output: {visibility: words()}
      }),
    GPUTerrainLineOfSight: grid =>
      new GPUTerrainLineOfSight({
        grid,
        elevation: band(),
        parameters: view(12),
        pairs: pairs('float32x4', 3),
        output: {visibility: words(3), clearance: view(3)}
      }),
    GPUTerrainCumulativeViewshed: grid =>
      new GPUTerrainCumulativeViewshed({
        grid,
        elevation: band(),
        parameters: view(12),
        observers: pairs('float32x2', 3),
        output: {visibleCount: words(), marginalCount: words()}
      }),
    GPUTerrainSummits: grid =>
      new GPUTerrainSummits({
        grid,
        elevation: band(),
        parameters: view(8),
        output: {summitMask: words()}
      }),
    GPUTerrainPeakSnap: grid =>
      new GPUTerrainPeakSnap({
        grid,
        elevation: band(),
        candidates: pairs('float32x2', 2),
        parameters: view(8),
        output: {positions: pairs('float32x2', 2), heights: view(2), snapStatus: words(2)}
      }),
    GPUTerrainFlowField: grid =>
      new GPUTerrainFlowField({
        grid,
        elevation: band(),
        parameters: view(8),
        output: {velocities: pairs('float32x2', PIXEL_COUNT)}
      }),
    GPUPointHorizonProfile: grid =>
      new GPUPointHorizonProfile({
        grid,
        maximumDistance: 100,
        cellSize: 10,
        azimuthCount: 8,
        elevation: band(),
        observers: pairs('float32x4', 1),
        parameters: view(8),
        output: {tangent: view(8)}
      }),
    GPUPointHorizonVisibility: grid =>
      new GPUPointHorizonVisibility({
        grid,
        maximumDistance: 100,
        cellSize: 10,
        elevation: band(),
        observers: pairs('float32x4', 1),
        targets: pairs('float32x4', 2),
        parameters: view(12),
        output: {visibility: words(2)}
      })
  };
  try {
    for (const [name, create] of Object.entries(factories)) {
      expect(() => create(AXIS_ALIGNED_GRID), name).not.toThrow();
      // Rotation with perpendicular axes keeps every per-axis distance exact.
      expect(() => create(ROTATED_GRID), name).not.toThrow();
      expect(() => create(SHEARED_GRID), name).toThrow(
        /grid\.affine must have orthogonal pixel axes/
      );
    }
  } finally {
    device.destroy();
  }
});

import {getGPUTerrainCurvatureParameterValues} from '../../src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {getGPUGeomorphonsParameterValues} from '../../src/gpu-terrain/geomorphons/gpu-geomorphons';
import {getGPUTerrainVectorRuggednessParameterValues} from '../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {getGPUTerrainWeissLandformsParameterValues} from '../../src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms';
import {getGPUReliefShadingParameterValues} from '../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {getGPUTerrainHorizonParameterValues} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {getGPUTerrainCastShadowParameterValues} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {getGPUTerrainViewshedParameterValues} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {getGPUTerrainLineOfSightParameterValues} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {getGPUTerrainSummitsParameterValues} from '../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {getGPUTerrainPeakSnapParameterValues} from '../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {getGPUTerrainFlowFieldParameterValues} from '../../src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field';
import {getGPUPointHorizonProfileParameterValues} from '../../src/gpu-terrain/point-horizon/point-horizon-parameters';
import {getGPUPointHorizonVisibilityParameterValues} from '../../src/gpu-terrain/point-horizon/point-horizon-parameters';

it('RT-102 public per-frame packers reject shear before writing the target', () => {
  const packers: Record<string, (grid: GPURasterMetadata, target: Float32Array) => Float32Array> = {
    getGPUTerrainCurvature: (grid, target) =>
      getGPUTerrainCurvatureParameterValues({}, grid, target),
    getGPUGeomorphons: (grid, target) => getGPUGeomorphonsParameterValues({}, grid, target),
    getGPUTerrainVectorRuggedness: (grid, target) =>
      getGPUTerrainVectorRuggednessParameterValues({}, grid, target),
    getGPUTerrainWeissLandforms: (grid, target) =>
      getGPUTerrainWeissLandformsParameterValues({}, grid, target),
    getGPUReliefShading: (grid, target) => getGPUReliefShadingParameterValues({}, grid, target),
    getGPUTerrainHorizon: (grid, target) => getGPUTerrainHorizonParameterValues({}, grid, target),
    getGPUTerrainCastShadow: (grid, target) =>
      getGPUTerrainCastShadowParameterValues(
        {azimuthDegrees: 0, altitudeDegrees: 45},
        grid,
        target
      ),
    getGPUTerrainViewshed: (grid, target) =>
      getGPUTerrainViewshedParameterValues({observer: [0, 0]}, grid, target),
    getGPUTerrainLineOfSight: (grid, target) =>
      getGPUTerrainLineOfSightParameterValues({}, grid, target),
    getGPUTerrainSummits: (grid, target) =>
      getGPUTerrainSummitsParameterValues({radius: 1}, grid, target),
    getGPUTerrainPeakSnap: (grid, target) =>
      getGPUTerrainPeakSnapParameterValues({radius: 1}, grid, target),
    getGPUTerrainFlowField: (grid, target) =>
      getGPUTerrainFlowFieldParameterValues({wind: [1, 0]}, grid, target),
    getGPUPointHorizonProfile: (grid, target) =>
      getGPUPointHorizonProfileParameterValues({maximumDistance: 100}, grid, target),
    getGPUPointHorizonVisibility: (grid, target) =>
      getGPUPointHorizonVisibilityParameterValues({maximumDistance: 100}, grid, target)
  };
  for (const [name, pack] of Object.entries(packers)) {
    const target = new Float32Array(80).fill(123);
    expect(() => pack(AXIS_ALIGNED_GRID, target), name).not.toThrow();
    expect(() => pack(ROTATED_GRID, target), name).not.toThrow();
    target.fill(123);
    expect(() => pack(SHEARED_GRID, target), name).toThrow(/orthogonal pixel axes/);
    for (const axisLength of [1e-46, 1e39]) {
      const unrepresentable: GPURasterMetadata = {
        ...AXIS_ALIGNED_GRID,
        affine: [axisLength, 0, 0, 0, -10, 90]
      };
      expect(() => pack(unrepresentable, target), name).toThrow(
        /positive and finite after float32/
      );
    }
    expect(Array.from(target), name).toEqual(new Array(80).fill(123));
  }
});

it('RT-102 shared-source horizon packers reject shear in the source before target writes', () => {
  const target = new Float32Array(32).fill(123);
  const settings = {sweepSourceGrid: SHEARED_GRID, sweepReferenceCellSize: [10, 10] as const};
  expect(() => getGPUTerrainHorizonParameterValues(settings, AXIS_ALIGNED_GRID, target)).toThrow(
    /sweepSourceGrid.*orthogonal/
  );
  expect(Array.from(target)).toEqual(new Array(32).fill(123));
  expect(() =>
    getGPUTerrainCastShadowParameterValues(
      {...settings, azimuthDegrees: 0, altitudeDegrees: 45},
      AXIS_ALIGNED_GRID,
      target
    )
  ).toThrow(/sweepSourceGrid.*orthogonal/);
  expect(Array.from(target)).toEqual(new Array(32).fill(123));
});
