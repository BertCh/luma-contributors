// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUTerrainBreach,
  GPUTerrainBranchingFlow,
  GPUTerrainFlowFractions,
  GPUTerrainDepressionHierarchy,
  GPUTerrainCatchmentSummary,
  GPUTerrainDEMCoregistration,
  GPUTerrainTiledFlow
} from '../../../src/gpu-terrain';
import type {KernelAuditCase} from './kernel-audit-context';

const makeGrid = (width: number) => ({
  width,
  height: 1,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
});

/** Every optional output and routing variant, with resident buffers at their binding-limited capacities. */
export const COVERAGE_TERRAIN_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUTerrainBreach',
    name: 'burn-and-breach-binding-limit',
    build: ({graph, view, maximumRows}) => {
      const count = maximumRows(24);
      graph.add(
        new GPUTerrainBreach({
          grid: makeGrid(count),
          elevation: view('float32', count),
          pitCells: view('uint32', 1),
          maximumSteps: 2,
          maximumCutDepth: 100,
          elevationDecrement: 0.5,
          burnDepth: view('float32', count),
          output: {
            elevation: view('float32', count),
            cutDepth: view('float32', count),
            modificationMask: view('uint32', count),
            pathCells: view('uint32', 3),
            pathLengths: view('uint32', 1),
            pitStatus: view('uint32', 1),
            unresolvedCount: view('uint32', 1)
          }
        })
      );
    }
  },
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
      ({
        exportName: 'GPUTerrainBranchingFlow',
        name: `${mode}-edge-lengths-binding-limit`,
        build: ({graph, view, maximumRows}) => {
          const count = maximumRows(32);
          graph.add(
            new GPUTerrainBranchingFlow({
              grid: makeGrid(count),
              fractions: view('float32', count * 8),
              values: view('float32', count),
              edgeLengths: view('float32', count * 8),
              mode,
              maximumIterations: 2,
              output: {
                values: view('float32', count),
                status: view('uint32', count),
                unresolvedCount: view('uint32', 1)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const).map(
    routing =>
      ({
        exportName: 'GPUTerrainFlowFractions',
        name: `${routing}-ground-lengths-binding-limit`,
        build: ({graph, view, maximumRows}) => {
          const count = maximumRows(32);
          graph.add(
            new GPUTerrainFlowFractions({
              grid: makeGrid(count),
              elevation: view('float32', count),
              flowDirections: view('uint32', count),
              parameters: view('float32', 8),
              routing,
              cellSizeMode: 'geographic',
              output: {
                fractions: view('float32', count * 8),
                edgeLengths: view('float32', count * 8)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  {
    exportName: 'GPUTerrainDepressionHierarchy',
    name: 'all-node-stage-outputs-binding-limit',
    build: ({graph, view, maximumRows}) => {
      const count = maximumRows(16);
      graph.add(
        new GPUTerrainDepressionHierarchy({
          grid: makeGrid(count),
          elevation: view('float32', count),
          outlets: view('uint32', count),
          stages: view('float32', 2),
          nodeCapacity: count,
          output: {
            nodes: view('uint32', count * 4),
            levels: view('float32', count * 2),
            cellNodes: view('uint32', count),
            curves: view('float32', count * 4),
            status: view('uint32', 3)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainCatchmentSummary',
    name: 'profiles-hypsometry-binding-limit',
    build: ({graph, view, maximumRows}) => {
      const count = maximumRows(16);
      graph.add(
        new GPUTerrainCatchmentSummary({
          grid: makeGrid(count),
          elevation: view('float32', count),
          basinIds: view('uint32', count),
          basinCount: 2,
          streams: view('uint32', count),
          flowDirections: view('uint32', count),
          histogramBins: 4,
          maximumSteps: 2,
          output: {
            statistics: view('float32', 12),
            hypsometry: view('float32', 8),
            profiles: view('float32', count * 3),
            profileStatus: view('uint32', count)
          }
        })
      );
    }
  },
  ...([0, 2] as const).map(
    correlationLength =>
      ({
        exportName: 'GPUTerrainDEMCoregistration',
        name: `correlation-${correlationLength}-binding-limit`,
        build: ({graph, view, maximumRows}) => {
          const count = maximumRows(8);
          graph.add(
            new GPUTerrainDEMCoregistration({
              grid: makeGrid(count),
              reference: view('float32', count),
              moving: view('float32', count),
              stableMask: view('uint32', count),
              differenceVariance: view('float32', count),
              correlationLength,
              output: {
                fit: view('float32', 4),
                differences: view('float32', count * 2),
                validity: view('uint32', count),
                volume: view('float32', 3)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...(['d8', 'branching'] as const).map(
    routing =>
      ({
        exportName: 'GPUTerrainTiledFlow',
        name: `${routing}-resident-dag-binding-limit`,
        build: ({graph, view, maximumRows}) => {
          const count = maximumRows(32);
          graph.add(
            new GPUTerrainTiledFlow({
              grid: makeGrid(count),
              routing,
              maximumIterations: 2,
              tiles: [
                {
                  column: 0,
                  row: 0,
                  width: count,
                  height: 1,
                  owned: [0, 0, count, 1],
                  flowDirections: view('uint32', count),
                  ...(routing === 'branching' ? {fractions: view('float32', count * 8)} : {}),
                  runoff: view('float32', count),
                  streams: view('uint32', count),
                  output: {
                    accumulation: view('float32', count),
                    status: view('uint32', count),
                    reachIds: view('uint32', count)
                  }
                }
              ]
            })
          );
        }
      }) satisfies KernelAuditCase
  )
];
