// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  estimateKeplerDorlingByteLength,
  parseKeplerDorlingSettings,
  type KeplerDorlingSettings
} from '../src/analysis/dorling-cartogram';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {
  parseKeplerAnalysisRecipe,
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {selectKeplerExecutionRoute} from '../src/execution/execution-router';

const settings: KeplerDorlingSettings = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  crs: 'EPSG:32618',
  bounds: [0, 0, 1000, 1000],
  xColumn: 'east',
  yColumn: 'north',
  valueColumn: 'area',
  areaScale: 100,
  maximumIterations: 0,
  overlapTolerance: 1,
  attraction: 0
};
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'dorling',
  source: {datasetId: 'source', revision: 1},
  population: 'selected',
  selectionRevision: 3,
  backendPolicy: 'require-gpu',
  operation: {kind: 'dorling-cartogram', ...settings}
};

test('Dorling recipes retain projected units and bounded controls without external resources', () => {
  expect(
    restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
      resolveDatasetRevision: () => 1,
      resolveSelectionRevision: () => 3
    })
  ).toEqual(recipe);
  const parsed = parseKeplerDorlingSettings(settings);
  expect(parsed.bounds).not.toBe(settings.bounds);
  expect(Object.isFrozen(parsed.bounds)).toBe(true);
  for (const override of [
    {maximumIterations: 257},
    {maximumIterations: -1},
    {maximumIterations: 0.5},
    {areaScale: -1},
    {areaScale: 1e16},
    {overlapTolerance: 1e-50},
    {attraction: 1.1},
    {metric: 'great-circle'},
    {units: 'feet'},
    {crs: ''},
    {bounds: [1, 0, 0, 1]}
  ])
    expect(() =>
      parseKeplerAnalysisRecipe({...recipe, operation: {...recipe.operation, ...override}})
    ).toThrow();
  const input = {
    rowCount: 3,
    chunkCount: 1,
    formats: {east: 'float32', north: 'float32', area: 'float32'}
  };
  const capabilities = {webgpu: true, availableBytes: 1024 * 1024};
  const request = {
    kind: 'analysis' as const,
    recipe,
    input,
    estimatedBytes: estimateKeplerDorlingByteLength(3)
  };
  expect(selectKeplerExecutionRoute(request, capabilities).backend).toBe('luma');
  expect(
    selectKeplerExecutionRoute({...request, input: {...input, rowCount: 65537}}, capabilities)
      .reason
  ).toBe('capacity');
  expect(
    selectKeplerExecutionRoute(
      {...request, input: {...input, formats: {...input.formats, east: 'float64'}}},
      capabilities
    ).reason
  ).toBe('precision');
  expect(() => estimateKeplerDorlingByteLength(65537)).toThrow();
});

test('Dorling dispatch preserves source-aligned caller geometry and radius ownership', () => {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.assign(device.limits, {
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxComputeWorkgroupsPerDimension: 65_535
  });
  const graph = new GPUCommandGraph(device);
  const columns = {
    east: createTransientView(graph, 'east', 'float32', 3),
    north: createTransientView(graph, 'north', 'float32', 3),
    area: createTransientView(graph, 'area', 'float32', 3)
  };
  const radii = createTransientView(graph, 'radii', 'float32', 3);
  const positions = createTransientView(graph, 'positions', 'float32x2', 3);
  const effectiveMask = createTransientView(graph, 'effective-mask', 'uint32', 3);
  const validityMask = createTransientView(graph, 'validity', 'uint32', 3);
  try {
    const result = addKeplerAnalysisRecipe(graph, recipe, {
      source: recipe.source,
      columns,
      validityMask,
      coordinateValidityMask: validityMask,
      mask: validityMask,
      selectionRevision: 3,
      outputs: {values: radii, dorling: {positions, effectiveMask}}
    });
    if (!('radii' in result)) throw new Error('Expected Dorling geometry');
    expect(result.radii).toBe(radii);
    expect(result.positions).toBe(positions);
    expect(result.effectiveMask).toBe(effectiveMask);
    expect(result.sourceIds.length).toBe(3);
    expect(result.metrics.length).toBe(1);
    expect(result.status.length).toBe(4);
    expect(result.parameterByteLength).toBe(0);
    expect(result.spatialContext.frame?.crs).toBe('EPSG:32618');
    result.destroyParameters();
  } finally {
    device.destroy();
  }
});
