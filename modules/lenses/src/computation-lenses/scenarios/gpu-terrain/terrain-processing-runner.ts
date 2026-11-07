// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {runRasterLens, type RasterLensSession} from '../gpu-raster/raster-lens-harness';
import {runExample as runBreach} from './terrain-breach.example';
import {runExample as runBranchingFlow} from './terrain-branching-flow.example';
import {runExample as runFlowFractions} from './terrain-flow-fractions.example';
import {runExample as runDepressionHierarchy} from './terrain-depression-hierarchy.example';
import {runExample as runCatchmentSummary} from './terrain-catchment-summary.example';
import {runExample as runDEMCoregistration} from './terrain-dem-coregistration.example';
import {runExample as runTiledFlow} from './terrain-tiled-flow.example';

const GRID: GPURasterMetadata = {
  width: 3,
  height: 3,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};

/** Executes actual graph nodes, then captures their GPU inputs, outputs and recorded kernels. */
export async function runTerrainProcessing(device: Device, element: string, scenarioId: string) {
  return runRasterLens({
    device,
    scenarioId,
    domain: 'gpu-terrain',
    label: element,
    grid: GRID,
    coordinateModel: 'projected',
    unit: 'm',
    dataset: {id: 'terrain-processing-synthetic', version: '1'},
    build: session => buildTerrainProcessingScenario(session, element)
  });
}

/** Builds the public example with deterministic synthetic inputs and semantic captures. */
export async function buildTerrainProcessingScenario(
  session: RasterLensSession,
  element: string
): Promise<void> {
  const scalar = (name: string, meaning: string) =>
    session.output(name, 'uint32', {role: 'intermediate', meaning}, 1);
  const category = (name: string, meaning: string) =>
    session.output(name, 'uint32', {role: 'intermediate', geometry: 'raster', meaning});
  const values = (name: string, meaning: string) =>
    session.output(name, 'float32', {role: 'output', geometry: 'raster', meaning});
  const table = (name: string, length: number, meaning: string) =>
    session.output(name, 'float32', {role: 'output', meaning}, length);
  const tableCategory = (name: string, length: number, meaning: string) =>
    session.output(name, 'uint32', {role: 'intermediate', meaning}, length);
  const diamond = Array<number>(72).fill(0);
  diamond[0] = 0.25;
  diamond[2] = 0.75;
  diamond[10] = 1;
  diamond[24] = 1;
  switch (element) {
    case 'GPUTerrainBreach':
      await runBreach(session.graph, {
        grid: GRID,
        elevation: session.input('elevation', 'float32', [2, 5, 0, NaN, NaN, NaN, NaN, NaN, NaN]),
        pitCells: session.input('pitCells', 'uint32', [0], {
          role: 'context',
          meaning: 'Explicit pit cell identifiers'
        }),
        maximumSteps: 2,
        maximumCutDepth: 4,
        elevationDecrement: 0.5,
        output: {
          elevation: values('editedElevation', 'Elevation after the bounded breach (m)'),
          cutDepth: values('cutDepth', 'Excavated depth by terrain cell (m)'),
          modificationMask: category(
            'modificationMask',
            'Bits identify breach, burn and invalid burn edits'
          ),
          pathCells: tableCategory('pathCells', 3, 'Selected pit-to-outlet path cells'),
          pathLengths: tableCategory('pathLengths', 1, 'Number of cells in each selected path'),
          pitStatus: tableCategory('pitStatus', 1, 'Resolved versus exhausted pit search'),
          unresolvedCount: scalar('unresolvedCount', 'Number of unresolved pit searches')
        }
      });
      break;
    case 'GPUTerrainBranchingFlow':
      await runBranchingFlow(session.graph, {
        grid: GRID,
        fractions: session.input('fractions', 'float32', diamond, {
          role: 'input',
          meaning: 'Eight outgoing fractions per cell',
          components: 8
        }),
        values: session.input('injection', 'float32', [8, 0, 0, 0, 0, 0, 0, 0, 0]),
        mode: 'transport',
        output: {
          values: values('transport', 'Transported injected mass; outlet cell four receives eight'),
          status: category('status', 'Per-cell transport readiness and validity'),
          unresolvedCount: scalar('unresolvedCount', 'Unresolved transport dependency count')
        }
      });
      break;
    case 'GPUTerrainFlowFractions':
      await runFlowFractions(session.graph, {
        grid: GRID,
        elevation: session.input('elevation', 'float32', [9, 8, 7, 6, 5, 4, 3, 2, 1]),
        flowDirections: session.input('flowDirections', 'uint32', [4, 4, 4, 4, 4, 4, 1, 1, 0]),
        parameters: session.input('parameters', 'float32', [1, 1, 0, 3, 0, 0, 1.1, 0], false),
        routing: 'mfd-freeman',
        output: {
          fractions: session.output(
            'fractions',
            'float32',
            {
              role: 'output',
              meaning: 'Eight outgoing MFD fractions per terrain cell',
              components: 8
            },
            72
          ),
          edgeLengths: session.output(
            'edgeLengths',
            'float32',
            {
              role: 'output',
              meaning: 'Ground length for each directed neighbor edge (m)',
              components: 8
            },
            72
          )
        }
      });
      break;
    case 'GPUTerrainDepressionHierarchy':
      await runDepressionHierarchy(session.graph, {
        grid: GRID,
        elevation: session.input('elevation', 'float32', [0, 2, 1, NaN, NaN, NaN, NaN, NaN, NaN]),
        outlets: session.input('outlets', 'uint32', Array(9).fill(0)),
        stages: session.input('stages', 'float32', [1, 3], {
          role: 'context',
          meaning: 'Query water elevations (m)'
        }),
        nodeCapacity: 3,
        output: {
          nodes: tableCategory('nodes', 12, 'Parent, minimum cell, outlet and validity per node'),
          levels: table('levels', 6, 'Birth and merge elevation per hierarchy node (m)'),
          cellNodes: session.output('cellNodes', 'uint32', {
            role: 'output',
            geometry: 'raster',
            meaning: 'Unique activation owner of each terrain cell'
          }),
          curves: table('curves', 12, 'Area and volume for each node and query stage'),
          status: tableCategory('status', 3, 'Stored count, required count and hierarchy overflow')
        }
      });
      break;
    case 'GPUTerrainCatchmentSummary':
      await runCatchmentSummary(session.graph, {
        grid: GRID,
        elevation: session.input('elevation', 'float32', [3, 2, 1, 6, 5, 4, 9, 8, 7]),
        basinIds: session.input('basinIds', 'uint32', [0, 0, 0, 1, 1, 1, 1, 1, 1]),
        basinCount: 2,
        streams: session.input('streams', 'uint32', Array(9).fill(1)),
        flowDirections: session.input('flowDirections', 'uint32', [1, 1, 0, 1, 1, 0, 1, 1, 0]),
        histogramBins: 2,
        output: {
          statistics: table(
            'statistics',
            12,
            'Area, mean, extrema, stream length and density per basin'
          ),
          hypsometry: table('hypsometry', 4, 'Cumulative area below each normalized elevation bin'),
          profiles: session.output(
            'profiles',
            'float32',
            {
              role: 'output',
              geometry: 'raster',
              components: 3,
              meaning: 'Downstream distance, elevation and outgoing gradient per stream cell'
            },
            27
          ),
          profileStatus: category('profileStatus', 'Resolved versus truncated stream profiles')
        }
      });
      break;
    case 'GPUTerrainDEMCoregistration':
      await runDEMCoregistration(session.graph, {
        grid: GRID,
        reference: session.input('reference', 'float32', Array(9).fill(0)),
        moving: session.input('moving', 'float32', [2, 2, 2, 100, 5, 2, 2, 2, 2]),
        stableMask: session.input('stableMask', 'uint32', [1, 1, 1, 1, 0, 0, 0, 0, 0]),
        differenceVariance: session.input('differenceVariance', 'float32', Array(9).fill(1)),
        correlationLength: 2,
        output: {
          fit: table('fit', 4, 'Median shift, residual NMAD, stable count and bias standard error'),
          differences: session.output(
            'differences',
            'float32',
            {
              role: 'output',
              geometry: 'raster',
              components: 2,
              meaning: 'Corrected difference and modeled standard deviation per cell'
            },
            18
          ),
          validity: category('validity', 'Valid uncertainty-aware DEM differences'),
          volume: table(
            'volume',
            3,
            'Integrated elevation change, standard deviation and valid area'
          )
        }
      });
      break;
    case 'GPUTerrainTiledFlow': {
      const tiles = [0, 1].map(tileIndex => {
        const prefix = `tile${tileIndex}`;
        return {
          column: 0,
          row: 0,
          width: 3,
          height: 3,
          owned: tileIndex === 0 ? ([0, 0, 1, 3] as const) : ([1, 0, 2, 3] as const),
          flowDirections: session.input(`${prefix}Directions`, 'uint32', Array(9).fill(0), false),
          fractions: session.input(`${prefix}Fractions`, 'float32', diamond, {
            role: 'input',
            meaning: 'Owned and halo fractional routing',
            components: 8
          }),
          runoff: session.input(`${prefix}Runoff`, 'float32', [8, 0, 0, 0, 0, 0, 0, 0, 0]),
          streams: session.input(`${prefix}Streams`, 'uint32', Array(9).fill(1)),
          output: {
            accumulation: values(
              `${prefix}Accumulation`,
              'Transported mass on owned cells; halo results remain invalid'
            ),
            status: category(`${prefix}Status`, 'Tile ownership and dependency readiness'),
            reachIds: category(
              `${prefix}ReachIds`,
              'Global stream-chain origin IDs across tile boundaries'
            )
          }
        };
      });
      await runTiledFlow(session.graph, {grid: GRID, routing: 'branching', tiles});
      break;
    }
    default:
      throw new Error(`Unknown terrain processing contributor ${element}`);
  }
}
