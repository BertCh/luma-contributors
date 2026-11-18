// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {GPURasterFractionalZonalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {findConstantNonFiniteBitcasts} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

const GRID: GPURasterMetadata = {
  width: 2,
  height: 2,
  affine: [1, 0, 10, 0, -1, 61],
  pixelInterpretation: 'area'
};

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${Math.random()}`, format, length);
  const coverage = {
    zoneIds: view('uint32', 4),
    cellIds: view('uint32', 4),
    fractions: view('float32', 4),
    status: {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    zoneValidity: view('uint32', 1)
  };
  const categorical = (overrides: Record<string, unknown> = {}) =>
    new GPURasterFractionalCategoricalStatistics({
      grid: GRID,
      input: {
        id: 'categories',
        format: 'uint32',
        storage: {kind: 'buffer', values: view('uint32', 4)}
      },
      zoneCount: 1,
      categoryCapacity: 2,
      coverage,
      output: {
        coverageSums: view('float32', 2),
        classAreas: view('float32', 2),
        proportions: view('float32', 2),
        coveredArea: view('float32', 1),
        validArea: view('float32', 1),
        missingArea: view('float32', 1),
        majority: view('uint32', 1),
        minority: view('uint32', 1),
        validity: view('uint32', 1)
      },
      ...overrides
    });
  const distribution = (overrides: Record<string, unknown> = {}) =>
    new GPURasterFractionalZonalDistribution({
      grid: GRID,
      input: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32', 4)}
      },
      zoneCount: 1,
      coverage,
      probabilities: view('float32', 1),
      output: {
        means: view('float32', 1),
        populationVariances: view('float32', 1),
        sampleVariances: view('float32', 1),
        effectiveSampleSizes: view('float32', 1),
        weightSums: view('float32', 1),
        squaredWeightSums: view('float32', 1),
        coveredArea: view('float32', 1),
        validArea: view('float32', 1),
        quantiles: view('float32', 1),
        quantileValidity: view('uint32', 1),
        validity: view('uint32', 1),
        status: view('uint32', 1)
      },
      ...overrides
    });
  const scalar = (overrides: Record<string, unknown> = {}) =>
    new GPURasterFractionalZonalStatistics({
      grid: GRID,
      zoneCount: 1,
      quantity: 'density',
      coverage,
      input: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32', 4)}
      },
      output: {
        totals: view('float32', 1),
        means: view('float32', 1),
        coveredArea: view('float32', 1),
        validArea: view('float32', 1),
        validity: view('uint32', 1)
      },
      ...overrides
    });
  return {graph, view, categorical, distribution, scalar};
}

it('fractional aggregators validate cellSizeMode at construction', () => {
  const {view, categorical, distribution, scalar} = setup();
  const rotated: GPURasterMetadata = {...GRID, affine: [1, 0.2, 10, 0.2, -1, 61]};
  for (const mode of ['uniform', 'geographic', 'web-mercator', undefined]) {
    expect(() => categorical({cellSizeMode: mode})).not.toThrow();
    expect(() => distribution({cellSizeMode: mode})).not.toThrow();
    expect(() => scalar({cellSizeMode: mode})).not.toThrow();
  }
  expect(() => categorical({cellSizeMode: 'planar'})).toThrow(
    /^fractional-categorical-statistics cellSizeMode must be uniform, web-mercator, or geographic/
  );
  expect(() => distribution({cellSizeMode: 'planar'})).toThrow(
    /^raster-fractional-zonal-distribution cellSizeMode must be uniform/
  );
  expect(() => categorical({grid: rotated, cellSizeMode: 'geographic'})).toThrow(
    /^fractional-categorical-statistics geographic cell sizes need an axis-aligned grid/
  );
  expect(() => distribution({grid: rotated, cellSizeMode: 'web-mercator'})).toThrow(
    /^raster-fractional-zonal-distribution web-mercator cell sizes need an axis-aligned grid/
  );
  expect(() => distribution({cellSizeMode: 'geographic', cellAreas: view('float32', 4)})).toThrow(
    /^raster-fractional-zonal-distribution cellSizeMode must be uniform when cellAreas is given/
  );
  expect(() => scalar({cellSizeMode: 'planar'})).toThrow(/cellSizeMode must be uniform/);
  expect(() => scalar({grid: rotated, cellSizeMode: 'geographic'})).toThrow(/axis-aligned/);
  expect(() => scalar({cellSizeMode: 'geographic', cellAreas: view('float32', 4)})).toThrow(
    /cellSizeMode must be uniform when cellAreas is given/
  );
  expect(() => scalar({cellAreas: view('float32', 3)})).toThrow(/cellAreas/);
  expect(() => scalar({cellAreas: view('float32', 5)})).toThrow(/cellAreas/);
  expect(() => scalar({cellAreas: view('uint32', 4)})).toThrow(/cellAreas/);
});

it('R7 explicit scalar areas do not require a representable affine determinant', () => {
  const {scalar, view, graph} = setup();
  expect(() =>
    scalar({
      grid: {...GRID, affine: [1e20, 0, 0, 0, 1e20, 0]},
      cellAreas: view('float32', 4)
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('R3 scalar GPU views accept exactly one offset area and reject one-cell padding', () => {
  const {scalar, view, graph} = setup();
  const grid = {...GRID, width: 1, height: 1};
  expect(() => scalar({grid, cellAreas: view('float32', 2)})).toThrow(/cellAreas/);
  const parent = view('float32', 3);
  const cellAreas = graph.createDataView(parent.buffer, {
    format: 'float32',
    byteOffset: 4,
    length: 1
  });
  expect(() =>
    scalar({
      grid,
      cellAreas,
      input: {
        id: 'single-value',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32', 1)}
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('scalar area kernels avoid constant nonfinite WGSL expressions for all support models', () => {
  for (const mode of ['uniform', 'geographic', 'web-mercator', 'explicit'] as const) {
    const {graph, view, scalar} = setup();
    const contributor = scalar({
      cellSizeMode: mode === 'explicit' ? 'uniform' : mode,
      cellAreas: mode === 'explicit' ? view('float32', 4) : undefined,
      parameters: view('float32', 8)
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const aggregate = sources.find(kernel => kernel.id.endsWith('-aggregate'));
    expect(aggregate, mode).toBeDefined();
    for (const kernel of sources)
      expect(findConstantNonFiniteBitcasts(kernel.source), `${mode}: ${kernel.id}`).toEqual([]);
  }
});
