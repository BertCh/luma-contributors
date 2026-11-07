// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('terrain roadmap kernels parse within portable binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `terrain-${identifier++}`, format, length);
  const grid = {
    width: 3,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const contributors = [
    ...(['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const).map(
      routing =>
        new GPUTerrainFlowFractions({
          id: routing,
          grid,
          elevation: view('float32', 3),
          flowDirections: view('uint32', 3),
          parameters: view('float32', 8),
          routing,
          output: {fractions: view('float32', 24), edgeLengths: view('float32', 24)}
        })
    ),
    ...(
      [
        'transport',
        'expected-distance',
        'minimum-distance',
        'maximum-distance',
        'fractional-catchment'
      ] as const
    ).map(
      mode =>
        new GPUTerrainBranchingFlow({
          id: mode,
          grid,
          fractions: view('float32', 24),
          values: view('float32', 3),
          mode,
          edgeLengths: view('float32', 24),
          maximumIterations: 2,
          output: {
            values: view('float32', 3),
            status: view('uint32', 3),
            unresolvedCount: view('uint32', 1)
          }
        })
    ),
    new GPUTerrainBreach({
      grid,
      elevation: view('float32', 3),
      pitCells: view('uint32', 1),
      maximumSteps: 2,
      maximumCutDepth: 3,
      elevationDecrement: 0.1,
      burnDepth: view('float32', 3),
      output: {
        elevation: view('float32', 3),
        cutDepth: view('float32', 3),
        modificationMask: view('uint32', 3),
        pathCells: view('uint32', 3),
        pathLengths: view('uint32', 1),
        pitStatus: view('uint32', 1),
        unresolvedCount: view('uint32', 1)
      }
    }),
    new GPUTerrainCatchmentSummary({
      grid,
      elevation: view('float32', 3),
      basinIds: view('uint32', 3),
      basinCount: 1,
      streams: view('uint32', 3),
      flowDirections: view('uint32', 3),
      histogramBins: 2,
      output: {
        statistics: view('float32', 6),
        hypsometry: view('float32', 2),
        profiles: view('float32', 9),
        profileStatus: view('uint32', 3)
      }
    }),
    new GPUTerrainDEMCoregistration({
      grid,
      reference: view('float32', 3),
      moving: view('float32', 3),
      stableMask: view('uint32', 3),
      differenceVariance: view('float32', 3),
      correlationLength: 2,
      output: {
        fit: view('float32', 4),
        differences: view('float32', 6),
        validity: view('uint32', 3),
        volume: view('float32', 3)
      }
    }),
    new GPUTerrainDepressionHierarchy({
      grid,
      elevation: view('float32', 3),
      outlets: view('uint32', 3),
      stages: view('float32', 2),
      nodeCapacity: 3,
      output: {
        nodes: view('uint32', 12),
        levels: view('float32', 6),
        cellNodes: view('uint32', 3),
        curves: view('float32', 12),
        status: view('uint32', 3)
      }
    }),
    new GPUTerrainTiledFlow({
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      maximumIterations: 2,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 3,
          height: 1,
          owned: [0, 0, 3, 1],
          flowDirections: view('uint32', 3),
          runoff: view('float32', 3),
          streams: view('uint32', 3),
          output: {
            accumulation: view('float32', 3),
            status: view('uint32', 3),
            reachIds: view('uint32', 3)
          }
        }
      ]
    })
  ];
  contributors.push(
    new GPUTerrainTiledFlow({
      id: 'branching-tiles',
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      routing: 'branching',
      maximumIterations: 2,
      tiles: [
        {
          column: 0,
          row: 0,
          width: 3,
          height: 1,
          owned: [0, 0, 3, 1],
          flowDirections: view('uint32', 3),
          fractions: view('float32', 24),
          runoff: view('float32', 3),
          streams: view('uint32', 3),
          output: {
            accumulation: view('float32', 3),
            status: view('uint32', 3),
            reachIds: view('uint32', 3)
          }
        }
      ]
    })
  );
  for (const contributor of contributors) {
    const commandNodes = contributor.getCommandNodes(graph);
    if (contributor instanceof GPUTerrainBreach) {
      expect(commandNodes.map(node => node.id)).toContain(`${contributor.id}-verify-paths`);
    }
    const sources = getParsedKernelSources(commandNodes);
    expect(sources.size).toBeGreaterThan(0);
    for (const [name, source] of sources) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source), name).toEqual([]);
    }
  }
});

it('tile ownership rejects missing domains and overlapping interiors before GPU work', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `ownership-${identifier++}`, format, length);
  const tile = (column: number) => ({
    column,
    row: 0,
    width: 2,
    height: 1,
    owned: [0, 0, 2, 1] as const,
    flowDirections: view('uint32', 2),
    runoff: view('float32', 2),
    streams: view('uint32', 2),
    output: {
      accumulation: view('float32', 2),
      status: view('uint32', 2),
      reachIds: view('uint32', 2)
    }
  });
  expect(
    () =>
      new GPUTerrainTiledFlow({
        grid: {width: 4, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        tiles: [tile(0)]
      })
  ).toThrow();
  expect(
    () =>
      new GPUTerrainTiledFlow({
        grid: {width: 4, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        tiles: [tile(0), tile(1)]
      })
  ).toThrow();
});

it('terrain outputs reject aliases and malformed capacity', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `capacity-${identifier++}`, format, length);
  const values = view('float32', 2);
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  expect(
    () =>
      new GPUTerrainBranchingFlow({
        grid,
        fractions: view('float32', 16),
        values,
        mode: 'transport',
        output: {values, status: view('uint32', 2), unresolvedCount: view('uint32', 1)}
      })
  ).toThrow();
  expect(
    () =>
      new GPUTerrainDepressionHierarchy({
        grid,
        elevation: values,
        outlets: view('uint32', 2),
        stages: view('float32', 2),
        nodeCapacity: 2,
        output: {
          nodes: view('uint32', 7),
          levels: view('float32', 4),
          cellNodes: view('uint32', 2),
          curves: view('float32', 8),
          status: view('uint32', 3)
        }
      })
  ).toThrow();
});
