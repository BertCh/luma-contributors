// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDatumGridTransform} from '../../src/geospatial-analysis/gpu-datum-grid-transform';
import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '../../src/geospatial-analysis/gpu-point-buffer';
import {GPUPointCloudStatisticalFilter} from '../../src/pointcloud-analysis/gpu-pointcloud-statistical-filter';
import {GPUPointCloudSurfaces} from '../../src/pointcloud-analysis/gpu-pointcloud-surfaces';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

function createPointBuffer(overrides: Record<string, unknown> = {}, offsetCount = 8) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = createTransientView;
  return () =>
    new GPUPointBuffer({
      points: view(graph, 'points', 'float32x2', 2),
      parameters: view(graph, 'parameters', 'float32', 2),
      offsets: view(graph, 'offsets', 'float32x2', offsetCount),
      verticesPerRing: 4,
      coordinateSystem: 'planar',
      ...overrides
    });
}

it('GPUPointBuffer refuses bad props with named-limit messages', () => {
  expect(createPointBuffer()()).toBeInstanceOf(GPUPointBuffer);
  expect(createPointBuffer({verticesPerRing: 2})).toThrow(
    /^point-buffer verticesPerRing must be an integer in \[3, 4294967295\]$/
  );
  expect(createPointBuffer({verticesPerRing: 5})).toThrow(
    /^point-buffer offsets must have verticesPerRing rows per point$/
  );
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  for (const offsetCount of [9, 10, 11]) {
    const construct = createPointBuffer({verticesPerRing: 5}, offsetCount);
    if (offsetCount === 10) expect(construct).not.toThrow();
    else
      expect(construct).toThrow(/^point-buffer offsets must have verticesPerRing rows per point$/);
  }
  expect(
    () =>
      new GPUPointBuffer({
        points: createTransientView(graph, 'points', 'float32x2', 1),
        parameters: createTransientView(graph, 'parameters', 'float32', 1),
        offsets: createTransientView(graph, 'offsets', 'float32x2', 3),
        verticesPerRing: 3,
        coordinateSystem: 'planar'
      })
  ).toThrow(/^point-buffer parameters must have at least 2 rows$/);
  expect(() => getGPUPointBufferParameterValues(0)).toThrow(
    /^GPUPointBuffer radius must be positive$/
  );
  expect(() => getGPUPointBufferParameterValues(Number.NaN)).toThrow(
    /^GPUPointBuffer radius must be a finite number/
  );
});

function createDatumTransform(
  overrides: Record<string, unknown> = {},
  gridOverrides: Record<string, unknown> = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32x3' | 'uint32'>(name: string, format: Format, length = 4) =>
    createTransientView(graph, name, format, length);
  return () =>
    new GPUDatumGridTransform({
      positions: view('positions', 'float32x3'),
      samples: view('samples', 'float32x3'),
      mask: view('mask', 'uint32'),
      grids: [
        {width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1], ...gridOverrides}
      ],
      operation: 'horizontal',
      output: {positions: view('output', 'float32x3'), status: view('status', 'uint32')},
      ...overrides
    });
}

it('GPUDatumGridTransform takes mask and no longer accepts the validity alias', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const mask = createTransientView(graph, 'mask', 'uint32', 4);
  expect(createDatumTransform({mask})().mask).toBe(mask);
  expect(createDatumTransform({mask: undefined, validity: mask})).toThrow(
    /^datum-grid-transform mask must be given$/
  );
});

it('GPUDatumGridTransform refuses bad props with named-limit messages', () => {
  expect(createDatumTransform({}, {width: 1})).toThrow(
    /^datum-grid-transform grids\[0\]\.width must be an integer in \[2, 4294967295\]$/
  );
  expect(createDatumTransform({}, {sampleOffset: 1})).toThrow(
    /^datum-grid-transform grids\[0\] sampleOffset \+ width \* height must be at most 4$/
  );
  expect(createDatumTransform({}, {origin: [Number.NaN, 0]})).toThrow(
    /^datum-grid-transform grids\[0\]\.origin\[0\] must be a finite number/
  );
  expect(createDatumTransform({}, {spacing: [1, 0]})).toThrow(
    /^datum-grid-transform grids\[0\]\.spacing\[1\] must be nonzero in float32$/
  );
  expect(createDatumTransform({heightUnitMeters: 0})).toThrow(
    /^datum-grid-transform heightUnitMeters must be positive in float32$/
  );
  expect(createDatumTransform({tolerance: -1})).toThrow(
    /^datum-grid-transform tolerance must be positive in float32$/
  );
  expect(createDatumTransform({maximumIterations: 1025})).toThrow(
    /^datum-grid-transform maximumIterations must be an integer in \[1, 1024\]$/
  );
  expect(createDatumTransform({operation: 'vertical'})).toThrow(
    /^datum-grid-transform vertical operation requires sourceVerticalReference and targetVerticalReference$/
  );
});

it('GPUPointCloudStatisticalFilter takes validity and no longer accepts the mask alias', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const props = {
    squaredDistances: view('distances', 'float32', 8),
    counts: view('counts', 'uint32', 4),
    capacity: 2,
    meanDistances: view('means', 'float32', 4)
  };
  const validity = view('validity', 'uint32', 4);
  expect(new GPUPointCloudStatisticalFilter({...props, validity}).validity).toBe(validity);
  expect(() => new GPUPointCloudStatisticalFilter({...props, mask: validity} as never)).toThrow(
    /^gpu-pointcloud-statistical-filter validity must be given$/
  );
});

it('PointCloudRasterGrid rowDirection defaults to north and refuses other values', () => {
  const build = (rowDirection?: string) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = <Format extends 'float32' | 'uint32' | 'float32x3'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    return new GPUPointCloudSurfaces({
      positions: view('positions', 'float32x3', 4),
      classifications: view('classifications', 'uint32', 4),
      grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2, rowDirection: rowDirection as never},
      terrain: view('terrain', 'float32', 4),
      surface: view('surface', 'float32', 4),
      canopy: view('canopy', 'float32', 4),
      coverage: view('coverage', 'uint32', 4)
    });
  };
  expect(build()).toBeInstanceOf(GPUPointCloudSurfaces);
  expect(build('south')).toBeInstanceOf(GPUPointCloudSurfaces);
  expect(() => build('up')).toThrow(
    /^gpu-pointcloud-surfaces grid\.rowDirection must be 'north' or 'south'$/
  );
});
