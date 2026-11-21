// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {resolvePointHorizonModel} from '../../../src/gpu-terrain/point-horizon/point-horizon-march';
import {
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  getGPUPointHorizonProfileParameterValues,
  getGPUPointHorizonVisibilityParameterValues
} from '../../../src/gpu-terrain/point-horizon/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const grid: GPURasterMetadata = {
  width: 8,
  height: 8,
  affine: [5, 0, 0, 0, -5, 0],
  pixelInterpretation: 'area'
};
const options = {maximumDistance: 1000, cellSize: 5, traversal: 'march' as const};

it('retains minimum-two dimensions and delegates excessive cell products to the shared validator', () => {
  expect(() =>
    resolvePointHorizonModel('horizon-grid', {...grid, width: 65536, height: 65536}, options)
  ).toThrow('horizon-grid grid cell count (4294967296) must be at most 4294967295');
  for (const dimension of ['width', 'height'])
    expect(() =>
      resolvePointHorizonModel('horizon-grid', {...grid, [dimension]: 1}, options)
    ).toThrow(/dimensions must be integers of at least 2/);
});

it('rejects rotated or mirrored Web Mercator axes in both constructors and packers', () => {
  // The rotated axes have equal length 5 and dot product zero, but advancing one column
  // changes northing by 3. Latitude is therefore not constant within a row.
  for (const affine of [
    [4, 3, 0, 3, -4, 0],
    [-5, 0, 0, 0, -5, 0],
    [5, 0, 0, 0, 5, 0]
  ] as const) {
    const rotated = {...grid, affine};
    for (const pack of [
      getGPUPointHorizonProfileParameterValues,
      getGPUPointHorizonVisibilityParameterValues
    ]) {
      expect(() => pack({projection: 'web-mercator'}, rotated)).toThrow(
        /web-mercator.*columns increasing east and rows increasing south/
      );
    }
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const common = {
        ...options,
        grid: rotated,
        projection: 'web-mercator' as const,
        elevation: {
          id: 'elevation',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: createTransientView(graph, 'elevation', 'float32', 64)
          }
        },
        observers: createTransientView(graph, 'observers', 'float32x4', 1)
      };
      const profile = {
        ...common,
        parameters: createTransientView(graph, 'profile-settings', 'float32', 8),
        output: {tangent: createTransientView(graph, 'tangent', 'float32', 720)}
      };
      const visibility = {
        ...common,
        parameters: createTransientView(graph, 'visibility-settings', 'float32', 12),
        targets: createTransientView(graph, 'targets', 'float32x4', 1),
        output: {visibility: createTransientView(graph, 'visibility', 'uint32', 1)}
      };
      expect(() => new GPUPointHorizonProfile(profile)).toThrow(
        /web-mercator.*columns increasing east and rows increasing south/
      );
      expect(() => new GPUPointHorizonVisibility(visibility)).toThrow(
        /web-mercator.*columns increasing east and rows increasing south/
      );
    } finally {
      device.destroy();
    }
  }
  expect(() =>
    resolvePointHorizonModel('horizon-grid', grid, {...options, projection: 'web-mercator'})
  ).not.toThrow();
  for (const pack of [
    getGPUPointHorizonProfileParameterValues,
    getGPUPointHorizonVisibilityParameterValues
  ])
    expect(() => pack({projection: 'web-mercator'}, grid)).not.toThrow();
  // The new restriction belongs to the Web Mercator window model, not the planar grid validator.
  expect(() =>
    resolvePointHorizonModel('horizon-grid', {...grid, affine: [4, 3, 0, 3, -4, 0]}, options)
  ).not.toThrow();
});
