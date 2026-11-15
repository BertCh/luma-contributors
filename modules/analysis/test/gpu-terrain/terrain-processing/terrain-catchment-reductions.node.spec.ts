// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {validateTerrainCatchmentReductionSizes} from '../../../src/gpu-terrain/terrain-processing/terrain-catchment-reductions';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function setup(cellCount = 1025) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${identifier++}`, format, length);
  const status = () => ({
    converged: view('uint32', 1),
    iterationLimitReached: view('uint32', 1),
    invalidCount: view('uint32', 1)
  });
  const props = {
    grid: {
      width: cellCount,
      height: 1,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    },
    elevation: view('float32', cellCount),
    basinIds: view('uint32', cellCount),
    basinCount: 4,
    histogramBins: 3,
    streams: view('uint32', cellCount),
    flowDirections: view('uint32', cellCount),
    output: {
      statistics: view('float32', 24),
      hypsometry: view('float32', 12),
      profiles: view('float32', cellCount * 3),
      profileStatus: view('uint32', cellCount),
      status: status()
    }
  };
  return {graph, view, status, props};
}

it('splits a dominant basin into bounded tiles and audits every generated kernel', () => {
  for (const cellSizeMode of ['uniform', 'geographic'] as const) {
    const {graph, props} = setup();
    const nodes = new GPUTerrainCatchmentSummary({...props, cellSizeMode}).getCommandNodes(graph);
    const sources = getParsedKernelSources(nodes);
    const sourceFor = (suffix: string) => {
      const entry = [...sources].find(([name]) =>
        name.includes(`terrain-catchment-summary-${suffix}`)
      );
      expect(entry, suffix).toBeDefined();
      return entry![1];
    };
    // Fails for the old per-basin kernel: even a single huge basin must be distributed.
    expect(sourceFor('metadata-tiles')).toContain('local < 256u');
    expect(sourceFor('sum-tiles')).toContain('local < 256u');
    expect(sourceFor('statistics')).not.toContain('for (var entry');
    expect(sourceFor('hypsometry')).not.toContain('for (var entry');
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source), name).toEqual([]);
    }
  }
});

it('rejects histogram index overflow and scratch above the device binding limit', () => {
  expect(() => validateTerrainCatchmentReductionSizes('summary', 1, 1, 0x80000000)).toThrow(
    'summary basinCount x histogramBins x 3'
  );
  expect(() => validateTerrainCatchmentReductionSizes('summary', 0xffffffff, 1, 256)).toThrow(
    'summary cell tiles x (histogramBins + 1)'
  );
  const {graph, view, props} = setup();
  const basinCount = 4_000_000;
  const contributor = new GPUTerrainCatchmentSummary({
    ...props,
    basinCount,
    histogramBins: 1,
    output: {
      ...props.output,
      statistics: view('float32', basinCount * 6),
      hypsometry: view('float32', basinCount)
    }
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow('basin sum words needs 144000000 bytes');
});

it('validates named scalar status shape, aliasing and graph ownership for both contributors', () => {
  const {graph, view, status, props} = setup(4);
  expect(
    () =>
      new GPUTerrainCatchmentSummary({
        ...props,
        output: {...props.output, status: {converged: view('uint32', 2)}}
      })
  ).toThrow('output.status.converged must be one uint32 row');
  expect(
    () =>
      new GPUTerrainCatchmentSummary({
        ...props,
        output: {...props.output, status: {converged: props.output.profileStatus}}
      })
  ).toThrow();
  const shared = status().converged;
  expect(
    () =>
      new GPUTerrainCatchmentSummary({
        ...props,
        output: {...props.output, status: {converged: shared, iterationLimitReached: shared}}
      })
  ).toThrow();
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const foreign = createTransientView(otherGraph, 'foreign', 'uint32', 1);
  const contributor = new GPUTerrainCatchmentSummary({
    ...props,
    output: {...props.output, status: {converged: foreign}}
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow();
  const tiledProps = {
    grid: props.grid,
    maximumIterations: 2,
    tiles: [
      {
        column: 0,
        row: 0,
        width: 4,
        height: 1,
        owned: [0, 0, 4, 1] as const,
        flowDirections: props.flowDirections,
        runoff: props.elevation,
        streams: props.streams,
        output: {
          accumulation: view('float32', 4),
          status: view('uint32', 4),
          reachIds: view('uint32', 4)
        }
      }
    ],
    output: {status: status()}
  };
  expect(
    () =>
      new GPUTerrainTiledFlow({...tiledProps, output: {status: {invalidCount: view('uint32', 2)}}})
  ).toThrow('output.status.invalidCount must be one uint32 row');
  expect(
    () =>
      new GPUTerrainTiledFlow({
        ...tiledProps,
        output: {status: {converged: shared, iterationLimitReached: shared}}
      })
  ).toThrow();
  expect(() =>
    new GPUTerrainTiledFlow({
      ...tiledProps,
      output: {status: {converged: foreign}}
    }).getCommandNodes(graph)
  ).toThrow();
  for (const [name, source] of getParsedKernelSources(
    new GPUTerrainTiledFlow(tiledProps).getCommandNodes(graph)
  )) {
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source), name).toEqual([]);
  }
});

it('publishes tiled budget exhaustion from the solver gate rather than unresolved cell counts', () => {
  const {graph, view, status, props} = setup(2);
  const nodes = new GPUTerrainTiledFlow({
    grid: props.grid,
    maximumIterations: 128,
    tiles: [
      {
        column: 0,
        row: 0,
        width: 2,
        height: 1,
        owned: [0, 0, 2, 1],
        flowDirections: props.flowDirections,
        runoff: props.elevation,
        streams: props.streams,
        output: {
          accumulation: view('float32', 2),
          status: view('uint32', 2),
          reachIds: view('uint32', 2)
        }
      }
    ],
    output: {status: status()}
  }).getCommandNodes(graph);
  const source = [...getParsedKernelSources(nodes)].find(([name]) =>
    name.includes('terrain-tiled-flow-budget-status-ports')
  )?.[1];
  expect(source).toBeDefined();
  expect(source).toContain('source[sourceOffset + 2u] >= 128u');
  expect(source).toContain('source[sourceOffset + 3u] == 0u');
  const budgetNode = nodes.find(node => node.id === 'terrain-tiled-flow-budget-status-ports');
  const solverGate = nodes.find(node => node.id.startsWith('terrain-tiled-flow-dag-solve-gate-'));
  expect(budgetNode).toBeDefined();
  expect(solverGate).toBeDefined();
});
