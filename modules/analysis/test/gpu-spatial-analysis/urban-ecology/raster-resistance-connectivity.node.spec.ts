// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  addRasterResistanceConnectivityRecipe,
  getRasterResistanceConnectivityParameterValues,
  type GPURasterResistanceConnectivityProps
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {estimateRasterResistanceCommandNodeCount} from '../../../src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `raster-resistance-${serial++}`, format, length);
  return {device, graph, view};
}

function getNodeCount(graph: GPUCommandGraph): number {
  return (graph as unknown as {nodes: unknown[]}).nodes.length;
}

function createProps(
  view: ReturnType<typeof createContext>['view'],
  width: number,
  height: number,
  overrides: Partial<GPURasterResistanceConnectivityProps> = {}
): GPURasterResistanceConnectivityProps {
  const cellCount = width * height;
  return {
    grid: {width, height, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
    resistance: {
      id: 'resistance',
      format: 'float32',
      storage: {kind: 'buffer', values: view('float32', cellCount)},
      validity: view('uint32', cellCount)
    },
    parameters: view('float32', 8),
    source: 0,
    sink: cellCount - 1,
    output: {
      potentials: view('float32', cellCount),
      vertexCurrents: view('float32', cellCount),
      diagnostics: view('float32x4', 1),
      solverStatus: view('uint32x4', 1)
    },
    ...overrides
  };
}

it('raster resistance connectivity names the limit it rejects', () => {
  const {device, graph, view} = createContext();
  try {
    const add =
      (
        width: number,
        height: number,
        overrides: Partial<GPURasterResistanceConnectivityProps> = {}
      ) =>
      () =>
        addRasterResistanceConnectivityRecipe(graph, createProps(view, width, height, overrides));
    expect(add(1025, 1024, {neighborhood: 4})).toThrow(
      /^raster-resistance-connectivity grid cells \(width \* height\) must be at most 1048576$/
    );
    expect(add(1024, 513)).toThrow(
      /^raster-resistance-connectivity edge slots \(grid cells \* 4 for neighborhood 8\) must be at most 2097152$/
    );
    expect(add(4, 4, {sink: 0})).toThrow(
      /^raster-resistance-connectivity source and sink must be different cells$/
    );
    expect(add(4, 4, {source: 16})).toThrow(
      /^raster-resistance-connectivity source must be an integer in \[0, 15\]$/
    );
    expect(add(4, 4, {neighborhood: 6 as 4})).toThrow(
      /^raster-resistance-connectivity neighborhood must be 4 or 8$/
    );
    expect(
      add(4, 4, {
        cellSizeMode: 'geographic',
        grid: {width: 4, height: 4, affine: [1, 0.1, 0, 0, -1, 0], pixelInterpretation: 'area'}
      })
    ).toThrow(/^raster-resistance-connectivity geographic cell sizes need an axis-aligned grid$/);
    expect(add(4, 4, {parameters: view('float32', 7)})).toThrow(
      /^raster-resistance-connectivity parameters must contain at least 8 values$/
    );
    expect(add(4, 4, {maximumIterations: 4097})).toThrow(
      /^raster-resistance-connectivity maximumIterations must be an integer in \[1, 4096\]$/
    );
  } finally {
    device.destroy();
  }
});

it('raster resistance connectivity accepts its largest grids', () => {
  for (const [width, height, neighborhood] of [
    [1024, 1024, 4],
    [1024, 512, 8]
  ] as const) {
    const {device, graph, view} = createContext();
    try {
      expect(() =>
        addRasterResistanceConnectivityRecipe(
          graph,
          createProps(view, width, height, {neighborhood})
        )
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  }
});

it('the raster resistance estimate is never below the command nodes the recipe adds', () => {
  for (const [width, height, neighborhood, maximumIterations, connectivityIterations] of [
    [4, 4, 4, 1, 1],
    [4, 4, 8, 64, 16],
    [1024, 1024, 4, 1, 1],
    [1024, 512, 8, 1024, 128]
  ] as const) {
    const {device, graph, view} = createContext();
    try {
      const before = getNodeCount(graph);
      const result = addRasterResistanceConnectivityRecipe(
        graph,
        createProps(view, width, height, {neighborhood, maximumIterations, connectivityIterations})
      );
      const actual = getNodeCount(graph) - before;
      expect(result.estimatedCommandNodeCount).toBe(
        estimateRasterResistanceCommandNodeCount(maximumIterations, connectivityIterations)
      );
      expect(result.estimatedCommandNodeCount, `${width} x ${height}`).toBeGreaterThanOrEqual(
        actual
      );
      // Every stage is listed: band, edges, invalid-cell count and sum, solve, solve status.
      expect(result.contributors.length).toBeGreaterThanOrEqual(4);
      expect(result.status.stages.map(stage => stage.stage)).toEqual(['resistance', 'solve']);
    } finally {
      device.destroy();
    }
  }
});

it('packs the grid georeference into the per-frame parameters', () => {
  const values = getRasterResistanceConnectivityParameterValues({
    width: 4,
    height: 2,
    affine: [2, 0.5, 10, 0.25, -3, 20],
    pixelInterpretation: 'area'
  });
  expect(Array.from(values)).toEqual([
    2,
    0.5,
    0.25,
    -3,
    Math.fround(Math.hypot(2, 0.25)),
    Math.fround(Math.hypot(0.5, -3)),
    20,
    14
  ]);
  expect(() =>
    getRasterResistanceConnectivityParameterValues(
      {width: 1, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      new Float32Array(7)
    )
  ).toThrow(/target must hold 8 values/);
});
