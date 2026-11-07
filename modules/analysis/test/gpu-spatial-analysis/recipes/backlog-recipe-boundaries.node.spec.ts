// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addNetworkDisruptionRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {addRasterZonalDistributionRecipe} from '../../../src/gpu-spatial-analysis/recipes/raster-summary-recipes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `backlog-view-${serial++}`, format, length);
  const search = {
    topology: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 3),
      weights: view('float32', 3)
    },
    startNodes: view('uint32', 2),
    targetNodes: view('uint32', 3),
    maximumIterations: 2
  };
  const props = {
    search,
    rowCount: 2,
    edgeMask: view('uint32', 3),
    opportunityWeights: view('float32', 3),
    parameters: view('float32', 4),
    baseline: {costs: view('float32', 6), status: {converged: view('uint32', 1)}}
  };
  return {device, graph, view, props};
}

it('backlog retained routing evidence keeps caller identities and exact convergence words', () => {
  const {device, graph, props} = createContext();
  try {
    const result = addNetworkDisruptionRecipe(graph, props);
    expect(result.outputs.baselineCosts).toBe(props.baseline.costs);
    expect(result.intermediates.topology).toBe(props.search.topology);
    expect(result.intermediates.originNodes).toBe(props.search.startNodes);
    expect(result.intermediates.targetNodes).toBe(props.search.targetNodes);
    expect(
      result.status.stages?.find(stage => stage.stage === 'baseline-routing')?.status.converged
    ).toBe(props.baseline.status.converged);
    expect(result.status.stages?.map(stage => stage.stage)).toEqual([
      'baseline-routing',
      'scenario-routing'
    ]);
  } finally {
    device.destroy();
  }
});

it.each([
  'baseline replacement',
  'scenario alias',
  'wrong baseline dimensions',
  'input alias'
])('backlog disruption rejects %s across recipe stages', boundary => {
  const {device, graph, view, props} = createContext();
  try {
    const parameters = boundary === 'input alias' ? view('float32', 2) : props.parameters;
    expect(() =>
      addNetworkDisruptionRecipe(graph, {
        ...props,
        parameters,
        ...(boundary === 'wrong baseline dimensions'
          ? {baseline: {...props.baseline, costs: view('float32', 3)}}
          : {}),
        outputs:
          boundary === 'baseline replacement'
            ? {baselineCosts: view('float32', 6)}
            : boundary === 'scenario alias'
              ? {scenarioCosts: props.baseline.costs}
              : boundary === 'input alias'
                ? {baselineAccessibility: parameters}
                : undefined
      })
    ).toThrow();
  } finally {
    device.destroy();
  }
});

it('backlog prepared bands keep validity and separate fill versus final failure evidence through composition', () => {
  const {device, graph, view} = createContext();
  try {
    const grid: GPURasterMetadata = {
      width: 6,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    };
    const values = view('float32', 6);
    const validity = view('uint32', 6);
    const prepared = addSurfacePreparationRecipe(graph, {
      grid,
      input: {id: 'source', format: 'float32', storage: {kind: 'buffer', values}},
      maximumDistance: 1,
      filter: {radius: 1, noDataPolicy: 'propagate'},
      outputs: {values: view('float32', 6), validity}
    });
    expect(prepared.band.storage.values).toBe(prepared.outputs.values);
    expect(prepared.band.validity).toBe(validity);
    const stages = prepared.status.stages!;
    expect(stages.map(stage => stage.stage)).toEqual(['fill-nodata', 'prepared-surface']);
    expect(stages[0].status.invalidCount).not.toBe(stages[1].status.invalidCount);
    expect(stages[1].status.invalidCount).toBe(prepared.outputs.unresolvedCount);
    const zones = view('uint32', 6);
    const distribution = addRasterZonalDistributionRecipe(graph, {
      grid,
      input: prepared.band,
      zones,
      zoneCount: 4
    });
    expect(distribution.intermediates.zones).toBe(zones);
    expect(distribution.outputs.zoneIds.length).toBe(4);
    expect(distribution.outputs.validCounts.length).toBe(4);
  } finally {
    device.destroy();
  }
});
