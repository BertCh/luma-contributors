// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {runExample as runRasterTileReprojectionExample} from './raster-tile-reprojection.example';
import type {Device} from '@luma.gl/core';
import {runExample as runRasterFillNoDataExample} from './raster-fill-no-data.example';
import {runExample as runRasterRankFilterExample} from './raster-rank-filter.example';
import {runExample as runRasterReprojectionExample} from './raster-reprojection.example';
import {runExample as runRasterReconstructionExample} from './raster-reconstruction.example';
import {runExample as runRasterFillHolesExample} from './raster-fill-holes.example';
import {runExample as runRasterSkeletonExample} from './raster-skeleton.example';
import {runExample as runRasterMedialAxisExample} from './raster-medial-axis.example';
import {runExample as runRasterWalkingCostExample} from './raster-walking-cost.example';
import {runExample as runRasterWatershedSegmentationExample} from './raster-watershed-segmentation.example';
import {runExample as runRasterFractionalCoverageExample} from './raster-fractional-coverage.example';
import {runExample as runRasterFractionalZonalStatisticsExample} from './raster-fractional-zonal-statistics.example';
import {runExample as runTerrainFlowDistanceExample} from '../gpu-terrain/terrain-flow-distance.example';
import {runExample as runTerrainStreamNetworkExample} from '../gpu-terrain/terrain-stream-network.example';
import type * as raster from '@luma.gl/graph/gpu-raster';
import {
  getGPURasterFractionalCoverageParameterValues,
  getGPURasterWalkingCostParameterValues,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {runRasterLens, type RasterLensSession} from './raster-lens-harness';
import {getRasterOperatorEntries} from '../../../gpu-raster/computation-pack/index';
import {getTerrainOperatorEntries} from '../../../gpu-terrain/computation-pack/index';

/** Executes the actual contributor on a bounded, explicit synthetic grid. */
export async function runExpansion(
  device: Device,
  element: string,
  scenarioId: string,
  domain: string
) {
  const grid: raster.GPURasterMetadata = {
    width: 3,
    height: 3,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const formula = (
    domain === 'gpu-raster' ? getRasterOperatorEntries() : getTerrainOperatorEntries()
  )[element].formula;
  return runRasterLens({
    formulas: 'tex' in formula ? [formula] : [],
    device,
    scenarioId,
    domain,
    label: element,
    grid,
    coordinateModel: 'projected',
    unit: 'm',
    dataset: {id: 'bounded-raster-expansion', version: '1'},
    build: async session => {
      await buildRasterExpansionScenario(session, element, grid);
    }
  });
}

/** Adds the same real contributor and capturable views used by the executable scenario. */
export async function buildRasterExpansionScenario(
  session: RasterLensSession,
  element: string,
  grid: raster.GPURasterMetadata
): Promise<void> {
  const scalar = (name: string) =>
    session.output(name, 'uint32', {role: 'intermediate', meaning: name}, 1);
  const values = () => session.output('values', 'float32');
  const validity = () =>
    session.output('validity', 'uint32', {role: 'intermediate', geometry: 'raster'});
  const source = session.input('input', 'float32', [9, 8, 7, 6, NaN, 4, 3, 2, 1]);
  const band = makeGPURasterBand(source, grid);
  const binary = () => session.input('foreground', 'uint32', [1, 1, 1, 1, 0, 1, 1, 1, 1]);
  switch (element) {
    case 'GPURasterFillNoData':
      await runRasterFillNoDataExample(session.graph, {
        grid,
        input: band,
        maximumDistance: 2,
        method: 'gdal-four-quadrant',
        output: {
          values: values(),
          validity: validity(),
          filled: session.output('filled', 'uint32'),
          status: {invalidCount: scalar('invalidCount')}
        }
      });
      break;
    case 'GPURasterRankFilter':
      await runRasterRankFilterExample(session.graph, {
        grid,
        input: band,
        radius: 1,
        noDataPolicy: 'ignore',
        output: {values: values(), validity: validity()}
      });
      break;
    case 'GPURasterTileReprojection':
      await runRasterTileReprojectionExample(session.graph, {
        sourceGrid: grid,
        destinationGrid: grid,
        sources: [{pixelBounds: [0, 0, 3, 3], input: band}],
        method: 'bilinear',
        output: {values: values(), validity: validity()}
      });
      break;
    case 'GPURasterReprojection':
      await runRasterReprojectionExample(session.graph, {
        sourceGrid: grid,
        destinationGrid: grid,
        input: band,
        method: 'nearest',
        output: {values: values(), validity: validity()}
      });
      break;
    case 'GPURasterReconstruction':
      // connectivity defaults to 8: the corner marker floods the uniform mask through diagonals,
      // reaching the far corner in 2 dilations where connectivity 4 needs 4.
      await runRasterReconstructionExample(session.graph, {
        grid,
        marker: session.input('marker', 'float32', [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        mask: session.input('mask', 'float32', [1, 1, 1, 1, 1, 1, 1, 1, 1]),
        maximumIterations: 12,
        output: {values: values(), validity: validity(), status: {converged: scalar('converged')}}
      });
      break;
    case 'GPURasterFillHoles':
      await runRasterFillHolesExample(session.graph, {
        grid,
        input: binary(),
        maximumIterations: 12,
        output: {
          values: session.output('values', 'uint32'),
          status: {converged: scalar('converged')}
        }
      });
      break;
    case 'GPURasterSkeleton':
      await runRasterSkeletonExample(session.graph, {
        grid,
        input: binary(),
        maximumIterations: 12,
        output: {
          skeleton: session.output('skeleton', 'uint32'),
          status: {converged: scalar('converged')}
        }
      });
      break;
    case 'GPURasterMedialAxis':
      await runRasterMedialAxisExample(session.graph, {
        grid,
        input: binary(),
        output: {
          medialAxis: session.output('medialAxis', 'uint32'),
          radius: session.output('radius', 'float32')
        }
      });
      break;
    case 'GPURasterWalkingCost':
      await runRasterWalkingCostExample(session.graph, {
        grid,
        elevation: band,
        friction: makeGPURasterBand(
          session.input('friction', 'float32', [1, 1, 1, 1, 1, 1, 1, 1, 1]),
          grid
        ),
        sources: session.input('sources', 'uint32', [1, 0, 0, 0, 0, 0, 0, 0, 0]),
        parameters: session.input(
          'parameters',
          'float32',
          Array.from(getGPURasterWalkingCostParameterValues({}, grid)),
          {role: 'context', meaning: 'georeference, friction weight and model coefficients'}
        ),
        maximumIterations: 12,
        output: {costs: values(), validity: validity(), status: {converged: scalar('converged')}}
      });
      break;
    case 'GPURasterWatershedSegmentation':
      await runRasterWatershedSegmentationExample(session.graph, {
        grid,
        input: band,
        markers: session.input('markers', 'uint32', [1, 0, 0, 0, 0, 0, 0, 0, 2]),
        output: {
          labels: session.output('labels', 'uint32'),
          status: {converged: scalar('converged')}
        }
      });
      break;
    case 'GPURasterFractionalCoverage':
    case 'GPURasterFractionalZonalStatistics': {
      const coordinates = session.input(
        'zoneCoordinates',
        'float32',
        [0.5, 0.5, 2.5, 0.5, 2.5, 2.5, 0.5, 2.5],
        {role: 'context', meaning: 'polygon coordinate components'}
      );
      const positions = session.graph.createDataView(coordinates.buffer, {
        format: 'float32x2',
        length: 4
      });
      const coverage = {
        zoneIds: session.output('zoneIds', 'uint32', {role: 'output', meaning: 'compact zoneIds'}),
        cellIds: session.output('cellIds', 'uint32', {role: 'output', meaning: 'compact cellIds'}),
        fractions: session.output('fractions', 'float32', {
          role: 'output',
          meaning: 'compact fractions'
        }),
        status: {
          count: scalar('count'),
          requiredCount: scalar('requiredCount'),
          overflow: scalar('overflow')
        },
        zoneValidity: session.output('zoneValidity', 'uint32', {role: 'intermediate'}, 1)
      };
      await runRasterFractionalCoverageExample(session.graph, {
        grid,
        parameters: session.input(
          'parameters',
          'float32',
          Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
          {role: 'context', meaning: 'affine georeference words'}
        ),
        zones: {
          positions,
          featureOffsets: session.input('featureOffsets', 'uint32', [0, 1], false),
          polygonOffsets: session.input('polygonOffsets', 'uint32', [0, 1], false),
          ringOffsets: session.input('ringOffsets', 'uint32', [0, 4], false)
        },
        output: coverage
      });
      if (element === 'GPURasterFractionalZonalStatistics')
        await runRasterFractionalZonalStatisticsExample(session.graph, {
          grid,
          input: band,
          coverage,
          zoneCount: 1,
          quantity: 'density',
          output: {
            totals: session.output('totals', 'float32', {role: 'output'}, 1),
            means: session.output('means', 'float32', {role: 'output'}, 1),
            coveredArea: session.output('coveredArea', 'float32', {role: 'output'}, 1),
            validArea: session.output('validArea', 'float32', {role: 'output'}, 1),
            validity: session.output('zoneAggregateValidity', 'uint32', {role: 'output'}, 1)
          }
        });
      break;
    }
    case 'GPUTerrainFlowDistance':
    case 'GPUTerrainStreamNetwork': {
      const flowDirections = session.input('flowDirections', 'uint32', [1, 1, 0, 1, 1, 0, 1, 1, 0]);
      const streams = session.input('streams', 'uint32', [1, 1, 1, 0, 0, 0, 0, 0, 0]);
      if (element === 'GPUTerrainFlowDistance')
        await runTerrainFlowDistanceExample(session.graph, {
          grid,
          flowDirections,
          streams,
          output: {
            distance: values(),
            validity: validity(),
            terminalCells: session.output('terminalCells', 'uint32')
          }
        });
      else {
        const coordinateComponents = session.output(
          'reachCoordinates',
          'float32',
          {role: 'output', meaning: 'compact polyline coordinate components'},
          54
        );
        await runTerrainStreamNetworkExample(session.graph, {
          grid,
          flowDirections,
          streams,
          output: {
            sourceCells: session.output('sourceCells', 'uint32', {
              role: 'output',
              meaning: 'compact sourceCells'
            }),
            targetCells: session.output('targetCells', 'uint32', {
              role: 'output',
              meaning: 'compact targetCells'
            }),
            lengths: session.output('reachLengths', 'float32', {
              role: 'output',
              meaning: 'compact reach lengths'
            }),
            vertexOffsets: session.output('vertexOffsets', 'uint32', {role: 'output'}, 10),
            positions: session.graph.createDataView(coordinateComponents.buffer, {
              format: 'float32x2',
              length: 27
            }),
            reachCount: scalar('reachCount'),
            vertexCount: scalar('vertexCount'),
            status: {overflow: scalar('overflow')},
            unresolvedCount: scalar('unresolvedCount')
          }
        });
      }
      break;
    }
    default:
      throw new Error(`Unknown expansion contributor ${element}`);
  }
}
