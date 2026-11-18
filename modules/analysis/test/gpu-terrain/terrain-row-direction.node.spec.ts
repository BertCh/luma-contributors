// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPUPointHorizonProfile} from '../../src/gpu-terrain/point-horizon/gpu-point-horizon-profile';
import {GPUPointHorizonVisibility} from '../../src/gpu-terrain/point-horizon/gpu-point-horizon-visibility';
import {resolvePointHorizonModel} from '../../src/gpu-terrain/point-horizon/point-horizon-march';
import {GPUTerrainDerivatives} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives';
import {GPUTerrainCurvature} from '../../src/gpu-terrain/terrain-curvature/gpu-terrain-curvature';
import {resolveTerrainRowDirection} from '../../src/gpu-terrain/terrain-curvature/terrain-geomorphometry-utils';
import {GPUReliefShading} from '../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {GPUTerrainCastShadow} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {GPUTerrainHorizon} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {GPUTerrainVectorRuggedness} from '../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createBand} from './terrain-test-utils';

const WIDTH = 6;
const HEIGHT = 5;
const PIXEL_COUNT = WIDTH * HEIGHT;
const SOUTH_GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10});
const NORTH_GRID = makeGPURasterMetadata({
  width: WIDTH,
  height: HEIGHT,
  cellSize: 10,
  rowDirection: 'north'
});

it('the grid affine is the source of the terrain row direction', () => {
  // affine = [a, b, c, d, e, f] with world y = d * column + e * row + f: e < 0 is north-up.
  expect(SOUTH_GRID.affine[4]).toBe(-10);
  expect(NORTH_GRID.affine[4]).toBe(10);
  expect(resolveTerrainRowDirection('test', SOUTH_GRID, undefined)).toBe('south');
  expect(resolveTerrainRowDirection('test', NORTH_GRID, undefined)).toBe('north');
  expect(resolveTerrainRowDirection('test', SOUTH_GRID, 'south')).toBe('south');
  expect(resolveTerrainRowDirection('test', NORTH_GRID, 'north')).toBe('north');
  // The message starts with the contributor ID and names the prop and the grid term.
  expect(() => resolveTerrainRowDirection('my-terrain', NORTH_GRID, 'south')).toThrow(
    /^my-terrain rowDirection 'south' must match the grid: grid\.affine\[4\] is 10, so rows advance north/
  );
  expect(() => resolveTerrainRowDirection('my-terrain', SOUTH_GRID, 'north')).toThrow(
    /^my-terrain rowDirection 'north' must match the grid: grid\.affine\[4\] is -10, so rows advance south/
  );
  expect(() => resolveTerrainRowDirection('test', SOUTH_GRID, 'east' as never)).toThrow(
    /rowDirection must be south or north/
  );
  // A grid rotated a quarter turn has no row term: the prop decides, 'south' by default.
  const quarterTurn: GPURasterMetadata = {...SOUTH_GRID, affine: [0, 10, 0, 10, 0, 0]};
  expect(resolveTerrainRowDirection('test', quarterTurn, undefined)).toBe('south');
  expect(resolveTerrainRowDirection('test', quarterTurn, 'north')).toBe('north');
});

it('eight terrain contributors default rowDirection from the grid and reject a contradiction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = (length: number = PIXEL_COUNT) =>
    createTransientView(graph, `view-${++instance}`, 'float32', length);
  const band = () => createBand(graph, `elevation-${++instance}`, PIXEL_COUNT);
  type RowOptions = {grid: GPURasterMetadata; rowDirection?: 'south' | 'north'};
  const factories: Record<string, (options: RowOptions) => {rowDirection: 'south' | 'north'}> = {
    GPUTerrainDerivatives: options =>
      new GPUTerrainDerivatives({
        ...options,
        elevation: band(),
        parameters: view(8),
        output: {slope: view()}
      }),
    GPUTerrainCurvature: options =>
      new GPUTerrainCurvature({
        ...options,
        elevation: band(),
        parameters: view(12),
        output: {curvatures: {mean: view()}}
      }),
    GPUTerrainVectorRuggedness: options =>
      new GPUTerrainVectorRuggedness({
        ...options,
        elevation: band(),
        parameters: view(8),
        output: {vectorRuggedness: view()}
      }),
    GPUReliefShading: options =>
      new GPUReliefShading({
        ...options,
        elevation: band(),
        parameters: view(80),
        output: {hillshade: view()}
      }),
    GPUTerrainHorizon: options =>
      new GPUTerrainHorizon({
        ...options,
        elevation: band(),
        parameters: view(8),
        maximumRadius: 3,
        output: {skyViewFactor: view()}
      }),
    GPUTerrainCastShadow: options =>
      new GPUTerrainCastShadow({
        ...options,
        elevation: band(),
        parameters: view(16),
        output: {sunVisibility: view()}
      })
  };
  // The point-horizon classes resolve the direction into their model (planar rasters read it).
  const rows = (length: number) =>
    createTransientView(graph, `rows-${++instance}`, 'float32x4', length);
  const pointHorizonModel = {maximumDistance: 100, cellSize: 10} as const;
  // The contributors keep their resolved model private: construct one (it validates the prop against
  // the grid), then resolve the same options with the function its constructor calls.
  factories['GPUPointHorizonProfile'] = options => {
    const contributor = new GPUPointHorizonProfile({
      ...options,
      ...pointHorizonModel,
      azimuthCount: 8,
      elevation: band(),
      observers: rows(1),
      parameters: view(8),
      output: {tangent: view(8)}
    });
    return resolvePointHorizonModel(contributor.id, options.grid, contributor.props);
  };
  factories['GPUPointHorizonVisibility'] = options => {
    const contributor = new GPUPointHorizonVisibility({
      ...options,
      ...pointHorizonModel,
      elevation: band(),
      observers: rows(1),
      targets: rows(2),
      parameters: view(12),
      output: {visibility: createTransientView(graph, `codes-${++instance}`, 'uint32', 2)}
    });
    return resolvePointHorizonModel(contributor.id, options.grid, contributor.props);
  };
  for (const [name, create] of Object.entries(factories)) {
    expect(create({grid: SOUTH_GRID}).rowDirection, name).toBe('south');
    expect(create({grid: NORTH_GRID}).rowDirection, name).toBe('north');
    expect(create({grid: NORTH_GRID, rowDirection: 'north'}).rowDirection, name).toBe('north');
    expect(create({grid: SOUTH_GRID, rowDirection: 'south'}).rowDirection, name).toBe('south');
    expect(() => create({grid: NORTH_GRID, rowDirection: 'south'}), name).toThrow(
      /rowDirection 'south' must match the grid: grid\.affine\[4\] is 10/
    );
    expect(() => create({grid: SOUTH_GRID, rowDirection: 'north'}), name).toThrow(
      /rowDirection 'north' must match the grid: grid\.affine\[4\] is -10/
    );
  }
  device.destroy();
});
