// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as spatial from '../../../src/gpu-spatial-analysis/index';
import * as network from '../../../src/gpu-network/index';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const GRID: GPURasterMetadata = {
  width: 128,
  height: 128,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
const ROWS = 16384;
function band(context: KernelAuditContext): GPURasterBufferBand<'sint32'> {
  return {
    id: 'calibrated',
    format: 'sint32',
    storage: {kind: 'buffer', values: context.view('sint32', ROWS)},
    validity: context.view('uint32', ROWS),
    noDataValue: -9999,
    scale: 2,
    offset: 1
  };
}
const cases: KernelAuditCase[] = [];
function add(exportName: string, name: string, build: KernelAuditCase['build']) {
  cases.push({exportName, name, build});
}
for (const probabilities of [false, true])
  add(
    'addZonalDistributionRecipe',
    `masked zone rows,${probabilities ? '128quantiles' : 'noquantiles'}`,
    context => {
      spatial.addZonalDistributionRecipe(context.graph, {
        source: {
          kind: 'zones',
          zones: context.view('uint32', ROWS),
          zoneCount: 256,
          mask: context.view('uint32', ROWS)
        },
        values: context.view('float32', ROWS),
        probabilities: probabilities ? context.view('float32', 128) : undefined
      });
    }
  );
add('addZonalDistributionRecipe', 'point-polygon assignment with quantiles and holes', context => {
  spatial.addZonalDistributionRecipe(context.graph, {
    source: {
      kind: 'points',
      join: {
        points: {
          kind: 'points',
          positions: context.view('float32x2', ROWS),
          sourceIds: context.view('uint32', ROWS)
        },
        polygons: {
          kind: 'polygons',
          positions: context.view('float32x2', 4096),
          featureOffsets: context.view('uint32', 257),
          polygonOffsets: context.view('uint32', 513),
          ringOffsets: context.view('uint32', 1025)
        },
        leafCapacity: 64,
        candidateCapacity: 1048576,
        spatialSort: true,
        querySort: true,
        edgeSlabs: true
      }
    },
    values: context.view('float32', ROWS),
    probabilities: context.view('float32', 128)
  });
});
add(
  'addRasterZonalDistributionRecipe',
  'calibrated signed validity nodata,128quantiles',
  context => {
    spatial.addRasterZonalDistributionRecipe(context.graph, {
      grid: GRID,
      input: band(context),
      zones: context.view('uint32', ROWS),
      zoneCount: 256,
      probabilities: context.view('float32', 128)
    });
  }
);
for (const weightMeaning of ['count', 'density'] as const)
  add(
    'addExposureByZoneRecipe',
    `${weightMeaning},explicit areas calibrated masked bands`,
    context => {
      spatial.addExposureByZoneRecipe(context.graph, {
        grid: GRID,
        hazard: band(context),
        population: band(context),
        weightMeaning,
        cellAreas: context.view('float32', ROWS),
        zones: context.view('uint32', ROWS),
        zoneCount: 256,
        threshold: context.view('float32', 1)
      });
    }
  );
add('addRegionalRasterChangeRecipe', '8slices,maskedareaweightedzones,allmetrics', context => {
  spatial.addRegionalRasterChangeRecipe(context.graph, {
    grid: GRID,
    change: {
      slices: context.view('float32', ROWS * 8),
      sliceCount: 8,
      parameters: context.view('float32', 16)
    },
    zones: context.view('uint32', ROWS),
    zoneCount: 256,
    cellAreas: context.view('float32', ROWS),
    threshold: context.view('float32', 1)
  });
});
for (const filter of [false, true])
  add(
    'addSurfacePreparationRecipe',
    `radius256IDW,${filter ? 'radius8rankfilter' : 'nofilter'}`,
    context => {
      spatial.addSurfacePreparationRecipe(context.graph, {
        grid: GRID,
        input: band(context),
        maximumDistance: 256,
        method: 'inverse-distance',
        filter: filter
          ? {radius: 8, quantile: 0.5, borderMode: 'nodata', noDataPolicy: 'ignore'}
          : undefined
      });
    }
  );
add(
  'addInterpolationDiagnosticsRecipe',
  '16neighbors,indexed leave-one-out and allresidualmetrics',
  context => {
    spatial.addInterpolationDiagnosticsRecipe(context.graph, {
      kriging: {
        positions: context.view('float32x2', ROWS),
        values: context.view('float32', ROWS),
        grid: GRID,
        indexGridSize: [128, 128],
        indexBounds: [0, 0, 128, 128],
        maximumNeighborCount: 16,
        parameters: context.view('float32', 12)
      }
    });
  }
);
for (const preparation of [false, true])
  add(
    'addDrainageNetworkRecipe',
    `${preparation ? 'fillandflats' : 'raw'},pourpointsandcompactgeometry`,
    context => {
      spatial.addDrainageNetworkRecipe(context.graph, {
        flow: {
          grid: GRID,
          elevation: band(context),
          parameters: context.view('float32', 16),
          fillDepressions: preparation,
          resolveFlats: preparation,
          maximumFillIterations: 128,
          maximumFlatIterations: 128,
          maximumAccumulationIterations: 128
        },
        pourPoints: context.view('uint32', ROWS),
        maximumSteps: ROWS,
        reachCapacity: ROWS,
        vertexCapacity: ROWS * 2
      });
    }
  );
for (const multipleTesting of ['none', 'benjamini-hochberg', 'benjamini-yekutieli'] as const)
  add('addLocalVariabilityRecipe', `${multipleTesting},sharedmaskandupstreamoverflow`, context => {
    spatial.addLocalVariabilityRecipe(context.graph, {
      weights: {...context.weights(ROWS, 64), status: {overflow: context.view('uint32', 1)}},
      values: context.view('float32', ROWS),
      mask: context.view('uint32', ROWS),
      parameters: context.view('float32', 8),
      residualPower: 1,
      minimumNeighbors: 64,
      multipleTesting
    });
  });
add(
  'addCellFlowTopologyRecipe',
  '16384H3cells65536ODrows,boundaryandconservationoutputs',
  context => {
    spatial.addCellFlowTopologyRecipe(context.graph, {
      cells: context.view('uint32x2', ROWS),
      origins: context.view('uint32', 65536),
      destinations: context.view('uint32', 65536),
      values: context.view('float32', 65536)
    });
  }
);
for (const doubles of [false, true])
  for (const matching of [false, true])
    add(
      'addTrackQualityRecipe',
      `${doubles ? 'double-single' : 'single'}timestamps,all4similaritymetricsanddecision,mapmatching${matching}`,
      context => {
        network.addTrackQualityRecipe(context.graph, {
          trajectory: {
            positions: context.view('float32x2', ROWS),
            timestamps: context.view('float32', ROWS),
            timestampsLow: doubles ? context.view('float32', ROWS) : undefined,
            trackOffsets: context.view('uint32', 257)
          },
          thresholds: context.view('float32', 2),
          similarity: {
            pairA: context.view('uint32', 1024),
            pairB: context.view('uint32', 1024),
            activePairCount: context.view('uint32', 1),
            maximumFrechetVertexCount: 2048,
            frechetEpsilon: context.view('float32', 1),
            lcssTolerance: 2,
            lcssBandRadius: 1073741824,
            densify: 1 / 4096
          },
          mapMatching: matching
            ? {
                topology: {
                  offsets: context.view('uint32', 1025),
                  neighbors: context.view('uint32', 65536),
                  nodePositions: context.view('float32x2', 1024)
                },
                parameters: context.view('float32', 8),
                cellSize: 1,
                bounds: {minimum: [0, 0], maximum: [128, 128]},
                entryCapacity: context.maximumRows(4),
                candidateCount: 8,
                routeNodeBudget: 128
              }
            : undefined
        });
      }
    );
for (const retained of [false, true])
  add(
    'addNetworkDisruptionRecipe',
    `${retained ? 'retainedbaseline' : 'twosearches'},closuresandreplacementcosts,targetsubset`,
    context => {
      network.addNetworkDisruptionRecipe(context.graph, {
        search: {
          topology: {
            offsets: context.view('uint32', 1025),
            neighbors: context.view('uint32', 65536),
            weights: context.view('float32', 65536)
          },
          startNodes: context.view('uint32', 128),
          targetNodes: context.view('uint32', 256),
          maximumIterations: 1024
        },
        rowCount: 128,
        edgeMask: context.view('uint32', 65536),
        replacementCosts: context.view('float32', 65536),
        opportunityWeights: context.view('float32', 256),
        parameters: context.view('float32', 4),
        baseline: retained
          ? {
              costs: context.view('float32', 128 * 256),
              status: {converged: context.view('uint32', 1)}
            }
          : undefined
      });
    }
  );
add('addFacilityScenarioRecipe', 'masked512facilities16384demands,reusablecostmatrix', context => {
  network.addFacilityScenarioRecipe(context.graph, {
    costs: context.view('float32', 512 * ROWS),
    demand: context.view('float32', ROWS),
    supply: context.view('float32', 512),
    facilityMask: context.view('uint32', 512),
    parameters: context.view('float32', 4)
  });
});
add('addClusterAndOutlineRecipe', 'alpha outlines,holes disconnectedarea,8groups', context => {
  spatial.addClusterAndOutlineRecipe(context.graph, {
    points: {
      kind: 'points',
      positions: context.view('float32x2', 4096),
      sourceIds: context.view('uint32', 4096)
    },
    clusteringParameters: context.view('float32', 64),
    gridSize: [64, 64],
    geometryParameters: context.view('float32', 64),
    maximumClusterCount: 8,
    maximumVerticesPerHull: 256,
    hullCapacity: 4096,
    outline: {kind: 'alpha', maximumCircumradius: 10}
  });
});
export const EXPANSION_RECIPE_CASES: readonly KernelAuditCase[] = cases;
