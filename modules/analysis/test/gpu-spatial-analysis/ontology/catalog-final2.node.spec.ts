// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import type {GPUPointInPolygonJoinOutput} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {addCellFlowTopologyRecipe} from '../../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';
import {addSurfacePreparationRecipe} from '../../../src/gpu-spatial-analysis/recipes/surface-preparation-recipe';
import {
  getGPUAnalysisOperator,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';
import {queryGPUSpatialAnalysisOperators} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import type {GPUSpatialAnalysisStatusKind} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-ontology';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `catalog-final2-${serial++}`, format, length);
  return {device, graph, view};
}

function expectDiscovery(
  exportName: string,
  status: GPUSpatialAnalysisStatusKind,
  present: boolean
) {
  for (const query of [queryGPUSpatialAnalysisOperators, queryGPUAnalysisOperators]) {
    expect(
      query({status}).some(operator => operator.exportName === exportName),
      `${exportName}: ${status}`
    ).toBe(present);
  }
}

it('G8 discovers nearest-feature weights flags from its actual outputWeights status', () => {
  const {device, view} = createContext();
  try {
    const weights = new GPUNearestFeatureWeights({
      neighborRows: view('uint32', 2),
      neighborCounts: view('uint32', 2),
      slotCapacity: 1,
      output: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2),
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      }
    });
    for (const field of ['overflow', 'candidateOverflow'] as const) {
      expect(weights.outputWeights.status?.[field]).toBeDefined();
      expectDiscovery('GPUNearestFeatureWeights', field, true);
    }
    expectDiscovery('GPUNearestFeatureWeights', 'requiredCount', false);
  } finally {
    device.destroy();
  }
});

it('G8 discovers compact polygon matches count and requirement ports', () => {
  const {device, view} = createContext();
  try {
    const matches: NonNullable<GPUPointInPolygonJoinOutput['matches']> = {
      ids: view('uint32', 2),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      }
    };
    for (const field of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const) {
      expect(matches.status[field]).toBeDefined();
      expectDiscovery('GPUPointInPolygonJoin', field, true);
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'local',
  'cell',
  'surface'
] as const)('G8 matches discovery to declared %s recipe stages', recipe => {
  const {device, graph, view} = createContext();
  try {
    const declarations = {
      local: () =>
        addLocalVariabilityRecipe(graph, {
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 2),
            weights: view('float32', 2),
            status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
          },
          values: view('float32', 2),
          parameters: view('float32', 8)
        }),
      cell: () =>
        addCellFlowTopologyRecipe(graph, {
          cells: view('uint32x2', 2),
          origins: view('uint32', 1),
          destinations: view('uint32', 1),
          values: view('float32', 1)
        }),
      surface: () =>
        addSurfacePreparationRecipe(graph, {
          grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          input: {
            id: 'source',
            format: 'float32',
            storage: {kind: 'buffer', values: view('float32', 2)}
          },
          maximumDistance: 1
        })
    };
    const names = {
      local: 'addLocalVariabilityRecipe',
      cell: 'addCellFlowTopologyRecipe',
      surface: 'addSurfacePreparationRecipe'
    };
    const result = declarations[recipe]();
    const statusKinds = [
      ...new Set(result.status.stages!.flatMap(stage => Object.keys(stage.status)))
    ].sort();
    expect(statusKinds).toEqual(
      recipe === 'local' ? ['candidateOverflow', 'invalidCount', 'overflow'] : ['invalidCount']
    );
    expect([...getGPUAnalysisOperator(names[recipe])!.status].sort()).toEqual(statusKinds);
    for (const field of ['overflow', 'candidateOverflow', 'invalidCount'] as const) {
      expectDiscovery(names[recipe], field, statusKinds.includes(field));
    }
  } finally {
    device.destroy();
  }
});

it('G8 documents compact polygon overflow at the actual nested status path', () => {
  const guide = readFileSync('docs/api/analysis-catalog.md', 'utf8');
  expect(guide.includes('`matches.status.overflow`')).toBe(true);
  expect(guide.includes('`matches.overflow`')).toBe(false);
});
