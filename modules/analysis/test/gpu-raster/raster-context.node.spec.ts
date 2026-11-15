// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {resolveGPURasterContext} from '../../src/analysis-core/raster-grid';
import {
  getGPURasterReprojectionParameterValues,
  GPURasterReprojection
} from '../../src/raster-analysis/gpu-raster-reprojection';
import {
  getGPURasterPatchMetricsParameterValues,
  GPURasterPatchMetrics
} from '../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

const GRID = {
  width: 17,
  height: 3,
  affine: [0.25, 0.125, 500000.3, -0.125, 0.25, 4600000.1] as const,
  pixelInterpretation: 'point' as const
};

it('resolves affine words relative to a float64 local origin without changing row direction', () => {
  const context = resolveGPURasterContext('context', {
    grid: GRID,
    localOrigin: [500000, 4600000]
  });
  expect(context.cellCount).toBe(51);
  expect(context.parameterValues.slice(0, 2)).toEqual(Float32Array.of(0.3, 0.1));
  expect(Array.from(context.parameterValues.slice(2, 7))).toEqual([
    0.25, 0.125, -0.125, 0.25, 0.078125
  ]);
  expect(context.georeference.northEdge).toBe(GRID.affine[5] - 0.125);
  expect(context.cellSizeMode).toBe('uniform');
});

it('rejects invalid shapes, modes, singular affines and rotated ground contexts with named limits', () => {
  for (const grid of [
    {...GRID, width: 0},
    {...GRID, height: 1.5},
    {...GRID, affine: [1, 2, 0, 2, 4, 0] as const},
    {...GRID, affine: [1, 0, Infinity, 0, 1, 0] as const}
  ]) {
    expect(() => resolveGPURasterContext('context', {grid})).toThrow(/^context grid/);
  }
  expect(() =>
    resolveGPURasterContext('context', {grid: GRID, cellSizeMode: 'geographic'})
  ).toThrow(/^context grid.affine must be axis-aligned/);
  expect(() =>
    resolveGPURasterContext('context', {grid: GRID, cellSizeMode: 'unknown' as 'uniform'})
  ).toThrow(/^context cellSizeMode/);
  expect(() => resolveGPURasterContext('context', {grid: GRID, localOrigin: [NaN, 0]})).toThrow(
    /^context localOrigin/
  );
});

it('packs same-CRS origin differences before float32 rounding, with a reusable target', () => {
  const destinationGrid = {
    ...GRID,
    affine: [0.25, 0.125, 500001.925, -0.125, 0.25, 4600001.31875] as const
  };
  const target = new Float32Array(22).fill(777);
  expect(getGPURasterReprojectionParameterValues({sourceGrid: GRID, destinationGrid}, target)).toBe(
    target
  );
  expect(Array.from(target.slice(4, 6))).toEqual([1.625, 1.21875]);
  expect(Array.from(target.slice(20))).toEqual([777, 777]);
  expect(() =>
    getGPURasterReprojectionParameterValues(
      {sourceGrid: GRID, destinationGrid},
      new Float32Array(19)
    )
  ).toThrow(/target must hold 20/);
  expect(getGPURasterPatchMetricsParameterValues(GRID)).toEqual(
    resolveGPURasterContext('context', {grid: GRID}).parameterValues
  );
});

it('rejects affines whose axes become identical when packed to float32', () => {
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 1, 0, 1, 1 + 1e-8, 0] as const,
    pixelInterpretation: 'point' as const
  };
  expect(() => resolveGPURasterContext('context', {grid})).toThrow(
    /^context grid.affine .*invertible in float32/
  );
  for (const settings of [
    {sourceGrid: grid, destinationGrid: GRID},
    {sourceGrid: GRID, destinationGrid: grid}
  ]) {
    expect(() => getGPURasterReprojectionParameterValues(settings)).toThrow(
      /grid.affine .*invertible in float32/
    );
  }
  expect(() => getGPURasterPatchMetricsParameterValues(grid)).toThrow(
    /grid.affine .*invertible in float32/
  );
  // The next representable float32 coefficient keeps the two axes distinct.
  const distinct = {...grid, affine: [1, 1, 0, 1, 1 + 2 ** -23, 0] as const};
  expect(resolveGPURasterContext('context', {grid: distinct}).parameterValues[6]).toBe(2 ** -23);
});

it('binds dynamic affine words for reprojection and every patch-metric measurement mode', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${sequence++}`, format, length);
  const parameters = view('float32', 20);
  const reprojection = new GPURasterReprojection({
    sourceGrid: GRID,
    destinationGrid: GRID,
    parameters,
    input: {id: 'input', format: 'float32', storage: {kind: 'buffer', values: view('float32', 51)}},
    output: {values: view('float32', 53), validity: view('uint32', 53)}
  });
  try {
    const sampler = getParsedKernelSources(reprojection.getCommandNodes(graph)).get(
      'raster-reprojection-sample'
    )!;
    expect(sampler).toContain('readAffine(10u)');
    expect(sampler).not.toContain('500000');
    for (const cellSizeMode of ['uniform', 'geographic', 'web-mercator'] as const) {
      const grid = {...GRID, affine: [1, 0, 0, 0, -1, 61] as const};
      const kernels = getParsedKernelSources(
        new GPURasterPatchMetrics({
          id: cellSizeMode,
          grid,
          cellSizeMode,
          parameters: view('float32', 8),
          labels: view('uint32', 51),
          output: {areas: view('float32', 2), perimeters: view('float32', 2)}
        }).getCommandNodes(graph)
      );
      expect(kernels.get(`${cellSizeMode}-areas`)).toContain('getPixelArea()');
      expect(kernels.get(`${cellSizeMode}-perimeters`)).toContain('getColumnStep()');
    }
    expect(
      () => new GPURasterReprojection({...reprojection.props, parameters: view('float32', 19)})
    ).toThrow(/^raster-reprojection parameters must contain at least 20/);
    expect(
      () =>
        new GPURasterPatchMetrics({
          grid: GRID,
          labels: view('uint32', 51),
          parameters: view('float32', 7),
          output: {areas: view('float32', 2)}
        })
    ).toThrow(/^raster-patch-metrics parameters must contain at least 8/);
  } finally {
    reprojection.destroy();
    device.destroy();
  }
});

it('uses the shared scaled norm for dynamic patch-axis lengths', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1e20, 0, 0, 0, 1e-20, 0] as const,
    pixelInterpretation: 'area' as const
  };
  try {
    expect(() => getGPURasterPatchMetricsParameterValues(grid)).not.toThrow();
    const kernels = getParsedKernelSources(
      new GPURasterPatchMetrics({
        grid,
        cellSizeMode: 'web-mercator',
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        labels: createTransientView(graph, 'labels', 'uint32', 1),
        output: {perimeters: createTransientView(graph, 'perimeters', 'float32', 2)}
      }).getCommandNodes(graph)
    );
    for (const name of ['raster-patch-metrics-faces', 'raster-patch-metrics-perimeters']) {
      const source = kernels.get(name)!;
      expect(source).toContain('convertPlanarDistanceValue(getPlanarDistanceVectorLength(');
      expect(source).not.toContain('length(vec2f(georeference[');
    }
  } finally {
    device.destroy();
  }
});

it.each([
  [1e-25, 0, 0, 0, 1e-25, 0],
  [1e25, 0, 0, 0, 1e25, 0],
  [1, 1, 0, 1, 1 + 1e-8, 0]
] as const)('keeps fixed-affine pixel counts independent of packed precision: %j', (...affine) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine,
    pixelInterpretation: 'area' as const
  };
  try {
    const labels = createTransientView(graph, 'labels', 'uint32', 1);
    const pixelCounts = createTransientView(graph, 'pixel-counts', 'uint32', 2);
    const contributor = new GPURasterPatchMetrics({grid, labels, output: {pixelCounts}});
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(3);
    expect(getParsedKernelSources(nodes).size).toBe(3);
    for (const node of nodes) graph.add(node);
    // A caller opting into packed geometry still needs a representable affine.
    expect(() => getGPURasterPatchMetricsParameterValues(grid)).toThrow(/invertible in float32/);
    expect(
      () =>
        new GPURasterPatchMetrics({
          grid,
          labels,
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          output: {pixelCounts}
        })
    ).toThrow(/invertible in float32/);
  } finally {
    device.destroy();
  }
});
