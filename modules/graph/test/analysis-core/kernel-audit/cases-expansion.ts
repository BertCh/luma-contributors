// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as raster from '../../../src/gpu-raster';
import * as terrain from '../../../src/gpu-terrain';
import * as spatial from '../../../src/gpu-spatial-analysis';
import {compileProjectionProgram} from '../../../src/gpu-project/projection-program';
import * as network from '../../../src/gpu-network';
import {
  GPUCrossVariogram,
  GPUSpaceTimeVariogram
} from '../../../src/gpu-dataframe/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/gpu-dataframe/pair-statistics/gpu-marked-point-statistics';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';
import {EXPANSION_RECIPE_CASES} from './cases-expansion-recipes';

/** Maximal bounded shapes are transient declarations, never CPU allocations or GPU execution. */
const GRID: raster.GPURasterMetadata = {
  width: 128,
  height: 128,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
const CELL_COUNT = 16384;
function band(
  context: KernelAuditContext,
  format: 'float32' | 'sint32' = 'float32'
): raster.GPURasterBufferBand {
  return {
    id: 'source',
    format,
    storage: {kind: 'buffer', values: context.view(format, CELL_COUNT)},
    validity: context.view('uint32', CELL_COUNT),
    noDataValue: -9999,
    scale: 2,
    offset: 1
  } as raster.GPURasterBufferBand;
}
function convergence(context: KernelAuditContext) {
  return {converged: context.view('uint32', 1)};
}
function rasterOutput(context: KernelAuditContext) {
  return {
    values: context.view('float32', CELL_COUNT),
    validity: context.view('uint32', CELL_COUNT)
  };
}
const cases: KernelAuditCase[] = [];
function add(exportName: string, name: string, build: KernelAuditCase['build']) {
  cases.push({exportName, name, build});
}

for (const method of ['nearest', 'inverse-distance'] as const)
  add('GPURasterFillNoData', `${method}, radius256, calibrated masked signed band`, context => {
    context.graph.add(
      new raster.GPURasterFillNoData({
        grid: GRID,
        input: band(context, 'sint32'),
        maximumDistance: 256,
        method,
        output: {
          ...rasterOutput(context),
          filled: context.view('uint32', CELL_COUNT),
          unresolvedCount: context.view('uint32', 1)
        }
      })
    );
  });
for (const borderMode of ['clamp', 'nodata'] as const)
  for (const noDataPolicy of ['propagate', 'ignore'] as const)
    add('GPURasterRankFilter', `radius8 ${borderMode} ${noDataPolicy}, every output`, context => {
      context.graph.add(
        new raster.GPURasterRankFilter({
          grid: GRID,
          input: band(context),
          radius: 8,
          quantile: 1,
          borderMode,
          noDataPolicy,
          output: {...rasterOutput(context), changed: context.view('uint32', CELL_COUNT)}
        })
      );
    });
for (const method of ['dilation', 'erosion'] as const)
  for (const connectivity of [4, 8] as const)
    add('GPURasterReconstruction', `${method} ${connectivity}-connected,128 rounds`, context => {
      context.graph.add(
        new raster.GPURasterReconstruction({
          grid: GRID,
          marker: context.view('float32', CELL_COUNT),
          mask: context.view('float32', CELL_COUNT),
          method,
          connectivity,
          maximumIterations: 128,
          output: {...rasterOutput(context), status: convergence(context)}
        })
      );
    });
for (const connectivity of [4, 8] as const)
  add('GPURasterFillHoles', `${connectivity}-connected,128 rounds`, context => {
    context.graph.add(
      new raster.GPURasterFillHoles({
        grid: GRID,
        input: context.view('uint32', CELL_COUNT),
        connectivity,
        maximumIterations: 128,
        output: {values: context.view('uint32', CELL_COUNT), status: convergence(context)}
      })
    );
  });
add('GPURasterSkeleton', 'Zhang-Suen128 rounds on16384cells', context => {
  context.graph.add(
    new raster.GPURasterSkeleton({
      grid: GRID,
      input: context.view('uint32', CELL_COUNT),
      maximumIterations: 128,
      output: {skeleton: context.view('uint32', CELL_COUNT), status: convergence(context)}
    })
  );
});
add('GPURasterMedialAxis', '16384 cells, disk centers and radii', context => {
  context.graph.add(
    new raster.GPURasterMedialAxis({
      grid: GRID,
      input: context.view('uint32', CELL_COUNT),
      output: {
        medialAxis: context.view('uint32', CELL_COUNT),
        radius: context.view('float32', CELL_COUNT)
      }
    })
  );
});
for (const [width, height] of [
  [32768, 1],
  [1, 32768],
  [32768, 1024],
  [1024, 32768]
] as const) {
  add('GPURasterMedialAxis', `exact distance transform ${width}x${height}`, context => {
    const cellCount = width * height;
    context.graph.add(
      new raster.GPURasterMedialAxis({
        grid: {...GRID, width, height},
        input: context.view('uint32', cellCount),
        output: {
          medialAxis: context.view('uint32', cellCount),
          radius: context.view('float32', cellCount)
        }
      })
    );
  });
}
for (const connectivity of [4, 8] as const)
  add(
    'GPURasterWatershedSegmentation',
    `${connectivity}-connected, maximum16384cells and steps`,
    context => {
      context.graph.add(
        new raster.GPURasterWatershedSegmentation({
          grid: GRID,
          input: band(context),
          markers: context.view('uint32', CELL_COUNT),
          connectivity,
          maximumSteps: CELL_COUNT,
          output: {labels: context.view('uint32', CELL_COUNT), status: convergence(context)}
        })
      );
    }
  );
for (const connectivity of [4, 8] as const) {
  add(
    'GPURasterWatershedSegmentation',
    `${connectivity}-connected minimum spanning forest, 1024x1024 cells`,
    context => {
      const cellCount = 1024 * 1024;
      context.graph.add(
        new raster.GPURasterWatershedSegmentation({
          grid: {...GRID, width: 1024, height: 1024},
          input: {
            id: 'forest-elevation',
            format: 'float32',
            storage: {kind: 'buffer', values: context.view('float32', cellCount)},
            validity: context.view('uint32', cellCount),
            noDataValue: -9999,
            scale: 2,
            offset: 1
          },
          markers: context.view('uint32', cellCount),
          connectivity,
          method: 'minimum-spanning-forest',
          maximumIterations: 3,
          output: {labels: context.view('uint32', cellCount), status: convergence(context)}
        })
      );
    }
  );
}
for (const model of [
  {kind: 'tobler'} as const,
  {kind: 'grass', coefficients: [1, 2, 3, 4], downhillSlopeThreshold: -0.2} as const
])
  add('GPURasterWalkingCost', `${model.kind}, masked calibrated inputs,128 rounds`, context => {
    context.graph.add(
      new raster.GPURasterWalkingCost({
        grid: GRID,
        elevation: band(context),
        friction: band(context, 'sint32'),
        sources: context.view('uint32', CELL_COUNT),
        model,
        frictionWeight: 2,
        maximumIterations: 128,
        output: {
          costs: context.view('float32', CELL_COUNT),
          validity: context.view('uint32', CELL_COUNT),
          status: convergence(context)
        }
      })
    );
  });
for (const projection of [false, true])
  for (const method of ['nearest', 'bilinear'] as const)
    for (const noDataPolicy of ['strict', 'renormalize'] as const)
      add(
        'GPURasterReprojection',
        `${method} ${noDataPolicy}, rotated source and destination,projection${projection}`,
        context => {
          context.graph.add(
            new raster.GPURasterReprojection({
              sourceGrid: {...GRID, affine: [1, 0.2, 100, 0.1, -1, 200]},
              destinationGrid: GRID,
              input: band(context, 'sint32'),
              method,
              noDataPolicy,
              inverseProjection: projection
                ? compileProjectionProgram({
                    precision: 'local-f32',
                    destinationOrigin: [10, 20],
                    operations: [{type: 'affine', scale: [2, 2], offset: [10, 20]}]
                  })
                : undefined,
              output: rasterOutput(context)
            })
          );
        }
      );
for (const costs of [false, true])
  add(
    'GPUTerrainFlowDistance',
    `16384cells, stream stop, ${costs ? 'explicit costs' : 'affine distances'}`,
    context => {
      context.graph.add(
        new terrain.GPUTerrainFlowDistance({
          grid: GRID,
          flowDirections: context.view('uint32', CELL_COUNT),
          streams: context.view('uint32', CELL_COUNT),
          transitionCosts: costs ? context.view('float32', CELL_COUNT) : undefined,
          maximumSteps: CELL_COUNT,
          output: {
            distance: context.view('float32', CELL_COUNT),
            validity: context.view('uint32', CELL_COUNT),
            terminalCells: context.view('uint32', CELL_COUNT)
          }
        })
      );
    }
  );
add('GPUTerrainStreamNetwork', '16384cells, all compact geometry and status outputs', context => {
  context.graph.add(
    new terrain.GPUTerrainStreamNetwork({
      grid: GRID,
      flowDirections: context.view('uint32', CELL_COUNT),
      streams: context.view('uint32', CELL_COUNT),
      maximumSteps: CELL_COUNT,
      output: {
        sourceCells: context.view('uint32', CELL_COUNT),
        targetCells: context.view('uint32', CELL_COUNT),
        lengths: context.view('float32', CELL_COUNT),
        positions: context.view('float32x2', CELL_COUNT * 2),
        vertexOffsets: context.view('uint32', CELL_COUNT + 1),
        reachCount: context.view('uint32', 1),
        vertexCount: context.view('uint32', 1),
        status: {overflow: context.view('uint32', 1)},
        unresolvedCount: context.view('uint32', 1)
      }
    })
  );
});
function coverage(context: KernelAuditContext) {
  return {
    zoneIds: context.view('uint32', 4194304),
    cellIds: context.view('uint32', 4194304),
    fractions: context.view('float32', 4194304),
    status: {
      count: context.view('uint32', 1),
      requiredCount: context.view('uint32', 1),
      overflow: context.view('uint32', 1)
    },
    zoneValidity: context.view('uint32', 256)
  };
}
add(
  'GPURasterFractionalCoverage',
  'maximum4194304zone-cell candidates, shell/hole layouts',
  context => {
    context.graph.add(
      new raster.GPURasterFractionalCoverage({
        grid: GRID,
        zones: {
          positions: context.view('float32x2', 4096),
          featureOffsets: context.view('uint32', 257),
          polygonOffsets: context.view('uint32', 513),
          ringOffsets: context.view('uint32', 1025)
        },
        output: coverage(context)
      })
    );
  }
);
for (const quantity of ['extensive', 'intensive', 'density'] as const)
  add(
    'GPURasterFractionalZonalStatistics',
    `${quantity},maximumcoverage capacity, all outputs`,
    context => {
      context.graph.add(
        new raster.GPURasterFractionalZonalStatistics({
          grid: GRID,
          input: band(context, 'sint32'),
          zoneCount: 256,
          coverage: coverage(context),
          quantity,
          output: {
            totals: context.view('float32', 256),
            means: context.view('float32', 256),
            coveredArea: context.view('float32', 256),
            validArea: context.view('float32', 256),
            validity: context.view('uint32', 256)
          }
        })
      );
    }
  );

for (const residualPower of [1, 2] as const)
  add(
    'GPULocalSpatialHeteroscedasticity',
    `residual power${residualPower},mask and upstream overflow`,
    context => {
      const weights = {
        ...context.weights(16384, 64),
        status: {overflow: context.view('uint32', 1)}
      };
      context.graph.add(
        new spatial.GPULocalSpatialHeteroscedasticity({
          weights,
          values: context.view('float32', 16384),
          mask: context.view('uint32', 16384),
          residualPower,
          output: {statistic: context.view('float32', 16384), status: context.view('uint32', 16384)}
        })
      );
    }
  );
add('GPUMultivariateLocalGeary', '32variables,masked16384rows and upstream overflow', context => {
  context.graph.add(
    new spatial.GPUMultivariateLocalGeary({
      weights: {...context.weights(16384, 64), status: {overflow: context.view('uint32', 1)}},
      values: context.view('float32', 16384 * 32),
      variableCount: 32,
      mask: context.view('uint32', 16384),
      output: {statistic: context.view('float32', 16384), status: context.view('uint32', 16384)}
    })
  );
});
for (const temporal of [false, true])
  add(
    'GPUCrossVariogram',
    `${temporal ? 'space-time subclass' : 'spatial'} maximum16384rows128bins,allcomplete-caseinputs`,
    context => {
      const Constructor = temporal ? GPUSpaceTimeVariogram : GPUCrossVariogram;
      context.graph.add(
        new Constructor({
          positions: context.view('float32x2', 16384),
          values: context.view('float32', 16384),
          secondValues: context.view('float32', 16384),
          times: temporal ? context.view('float32', 16384) : undefined,
          mask: context.view('uint32', 16384),
          parameters: context.view('float32', 2),
          lagCount: temporal ? 16 : 128,
          temporalLagCount: temporal ? 8 : 1,
          output: {
            semivariances: context.view('float32', 128),
            pairCounts: context.view('uint32', 128),
            status: context.view('uint32', 128)
          }
        })
      );
    }
  );
for (const edgeCorrection of ['translation', 'none'] as const)
  for (const extras of [false, true])
    add(
      'GPUMarkedPointStatistics',
      `${edgeCorrection},16384rows128radii,${extras ? 'types marks intensities mask' : 'homogeneous unmarked'}`,
      context => {
        context.graph.add(
          new GPUMarkedPointStatistics({
            positions: context.view('float32x2', 16384),
            radiusCount: 128,
            parameters: context.view('float32', 6),
            edgeCorrection,
            types: extras ? context.view('uint32', 16384) : undefined,
            typePair: extras ? [0, 1] : undefined,
            marks: extras ? context.view('float32', 16384) : undefined,
            intensities: extras ? context.view('float32', 16384) : undefined,
            mask: extras ? context.view('uint32', 16384) : undefined,
            output: {
              statistics: context.view('float32', 512),
              pairCounts: context.view('uint32', 128),
              status: context.view('uint32', 128)
            }
          })
        );
      }
    );
function regression(context: KernelAuditContext) {
  return {
    positions: context.view('float32x2', 16384),
    predictors: context.view('float32', 16384 * 11),
    response: context.view('float32', 16384),
    predictorCount: 11,
    parameters: context.view('float32', 13),
    mask: context.view('uint32', 16384),
    offset: context.view('float32', 16384),
    maximumIterations: 128
  };
}
for (const family of ['poisson', 'logistic'] as const)
  for (const kernel of ['gaussian', 'bisquare'] as const)
    add(
      'GPUGeneralizedGeographicallyWeightedRegression',
      `${family} ${kernel},11predictors128iterations16384rows,offsetmask`,
      context => {
        context.graph.add(
          new spatial.GPUGeneralizedGeographicallyWeightedRegression({
            ...regression(context),
            family,
            kernel,
            output: {
              coefficients: context.view('float32', 16384 * 12),
              fitted: context.view('float32', 16384),
              status: {
                codes: context.view('uint32', 16384),
                iterationCount: context.view('uint32', 16384)
              }
            }
          })
        );
      }
    );
for (const kernel of ['gaussian', 'bisquare'] as const)
  for (const selection of [false, true])
    add(
      'GPUMultiscaleGeographicallyWeightedRegression',
      `${kernel},11predictors128rounds,${selection ? '32candidateCVandconditionalSE' : 'fixed bandwidths'}`,
      context => {
        const {offset: _offset, ...inputs} = regression(context);
        const props: spatial.GPUMultiscaleGeographicallyWeightedRegressionProps = {
          ...inputs,
          kernel,
          bandwidthCandidates: selection ? context.view('float32', 12 * 32) : undefined,
          bandwidthCandidateCount: selection ? 32 : undefined,
          noiseVariance: context.view('float32', 1),
          output: {
            coefficients: context.view('float32', 16384 * 12),
            fitted: context.view('float32', 16384),
            selectedBandwidths: context.view('float32', 12),
            conditionalStandardErrors: context.view('float32', 16384 * 12),
            status: {
              codes: context.view('uint32', 16384),
              iterationCount: context.view('uint32', 1)
            }
          }
        };
        context.graph.add(new spatial.GPUMultiscaleGeographicallyWeightedRegression(props));
      }
    );
add(
  'GPUSpatialRegimeRegression',
  '11predictors,maximum4096regimes65536maskedrows,alloutputs',
  context => {
    context.graph.add(
      new spatial.GPUSpatialRegimeRegression({
        predictors: context.view('float32', 65536 * 11),
        response: context.view('float32', 65536),
        regimes: context.view('uint32', 65536),
        mask: context.view('uint32', 65536),
        predictorCount: 11,
        regimeCount: 4096,
        output: {
          coefficients: context.view('float32', 4096 * 12),
          fitted: context.view('float32', 65536),
          counts: context.view('uint32', 4096),
          status: context.view('uint32', 4096)
        }
      })
    );
  }
);
for (const instrumentOrder of [1, 2] as const)
  add(
    'GPUSpatialPanelRegression',
    `instrument order${instrumentOrder},maxsecond-orderpredictors,alloutputs`,
    context => {
      context.graph.add(
        new spatial.GPUSpatialPanelRegression({
          weights: {...context.weights(256, 64), status: {overflow: context.view('uint32', 1)}},
          periodCount: 1024,
          predictorCount: instrumentOrder === 2 ? 5 : 8,
          predictors: context.view('float32', 256 * 1024 * (instrumentOrder === 2 ? 5 : 8)),
          response: context.view('float32', 256 * 1024),
          instrumentOrder,
          output: {
            table: context.view('float32', (instrumentOrder === 2 ? 7 : 10) * 4),
            summary: context.view('float32', 32),
            residuals: context.view('float32', 256 * 1024),
            status: {code: context.view('uint32', 1)}
          }
        })
      );
    }
  );

for (const operation of ['directed-edges', 'vertices'] as const)
  add('GPUH3CellIdentities', `${operation},largestbindinglimitedsixkeyoutput`, context => {
    const rows = Math.floor(context.maximumRows(8) / 6);
    context.graph.add(
      new spatial.GPUH3CellIdentities({
        cells: context.view('uint32x2', rows),
        operation,
        output: context.view('uint32x2', rows * 6)
      })
    );
  });
for (const operation of ['p-center', 'capacitated-p-median'] as const)
  for (const [facilityCount, selectCount] of [
    [32, 4],
    [16, 8]
  ])
    add(
      'GPUBoundedFacilityAllocation',
      `${operation},${facilityCount}facilities${selectCount}selected4096demands`,
      context => {
        context.graph.add(
          new spatial.GPUBoundedFacilityAllocation({
            operation,
            facilityCount,
            selectCount,
            costs: context.view('float32', 4096 * facilityCount),
            demand: context.view('float32', 4096),
            capacities: context.view('float32', facilityCount),
            output: {
              selectedFacilities: context.view('uint32', facilityCount),
              assignments: context.view('uint32', 4096),
              objective: context.view('float32', 1),
              status: context.view('uint32', 3)
            }
          })
        );
      }
    );
add('GPUNetworkStructure', '1024vertices65536slots,allfouroutputs', context => {
  context.graph.add(
    new network.GPUNetworkStructure({
      topology: {offsets: context.view('uint32', 1025), neighbors: context.view('uint32', 65536)},
      reverseEdgeSlots: context.view('uint32', 65536),
      output: {
        strongComponents: context.view('uint32', 1024),
        bridges: context.view('uint32', 65536),
        articulationPoints: context.view('uint32', 1024),
        biconnectedComponents: context.view('uint32', 65536)
      }
    })
  );
});
add('GPUAlternativeRoutes', '256nodes32routes10Mexpansions,heapbindingcapacity', context => {
  const queries = Math.floor(context.maximumRows(4) / (1024 * (2 + 2 * 256)));
  context.graph.add(
    new network.GPUAlternativeRoutes({
      topology: {
        offsets: context.view('uint32', 257),
        neighbors: context.view('uint32', 65536),
        weights: context.view('float32', 65536)
      },
      sources: context.view('uint32', queries),
      targets: context.view('uint32', queries),
      routeCount: 32,
      maximumExpansions: 10000000,
      output: {
        vertices: context.view('uint32', queries * 32 * 256),
        edges: context.view('uint32', queries * 32 * 256),
        lengths: context.view('uint32', queries * 32),
        costs: context.view('float32', queries * 32),
        counts: context.view('uint32', queries),
        complete: context.view('uint32', queries)
      }
    })
  );
});

for (const grouped of [false, true])
  add(
    grouped ? 'GPUGroupConcaveHull' : 'GPUConcaveHull',
    '1024points,densemaximumtriangleandboundarycapacity',
    context => {
      const points = 1024,
        groups = grouped ? 8 : 1,
        triangles = (2 * points - 5) * groups;
      const output = {
        triangles: context.view('uint32x3', triangles),
        triangleValidity: context.view('uint32', triangles),
        boundaryEdges: context.view('uint32x2', triangles * 3),
        edgeValidity: context.view('uint32', triangles * 3),
        status: {converged: context.view('uint32', groups)}
      };
      if (grouped)
        context.graph.add(
          new spatial.GPUGroupConcaveHull({
            positions: context.view('float32x2', points),
            groupIds: context.view('uint32', points),
            groupCount: groups,
            maximumCircumradius: 10,
            output
          })
        );
      else
        context.graph.add(
          new spatial.GPUConcaveHull({
            positions: context.view('float32x2', points),
            maximumCircumradius: 10,
            output
          })
        );
    }
  );
for (const useZOrderHash of [false, true])
  for (const largePolygonMethod of ['monotone', 'none'] as const)
    add(
      'GPUConstrainedDelaunayTriangulation',
      `4096vertices,holes and allmeshmetadata,hash${useZOrderHash},${largePolygonMethod}`,
      context => {
        context.graph.add(
          new spatial.GPUConstrainedDelaunayTriangulation({
            positions: context.view('float32x2', 4096),
            featureOffsets: context.view('uint32', 5),
            polygonOffsets: context.view('uint32', 9),
            ringOffsets: context.view('uint32', 17),
            maximumFlips: 0xffffffff,
            maximumWork: 0xffffffff,
            useZOrderHash,
            largePolygonMethod,
            output: {
              indices: context.view(
                'uint32',
                spatial.getPolygonTriangulationIndexCount(4096, 16, 8)
              ),
              valid: context.view('uint32', 8),
              triangleOffsets: context.view('uint32', 9),
              triangleCount: context.view('uint32', 8),
              status: {converged: context.view('uint32', 8)}
            }
          })
        );
      }
    );

for (const edgesOnly of [false, true])
  add(
    'GPUNetworkCentrality',
    `edge betweenness,${edgesOnly ? 'edgeonly' : 'alloutputs'},forwardreverseCSR`,
    context => {
      const topology = {
        offsets: context.view('uint32', 1025),
        neighbors: context.view('uint32', 65536),
        weights: context.view('float32', 65536)
      };
      context.graph.add(
        new network.GPUNetworkCentrality({
          topology,
          reverseTopology: {
            offsets: context.view('uint32', 1025),
            neighbors: context.view('uint32', 65536),
            weights: context.view('float32', 65536)
          },
          pivotNodes: context.view('uint32', 4),
          costLimit: context.view('float32', 1),
          laneCount: 4,
          maximumIterations: network.GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
          localIterations: network.GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
          output: {
            edgeBetweenness: context.view('float32', 65536),
            betweenness: edgesOnly ? undefined : context.view('float32', 1024),
            closeness: edgesOnly ? undefined : context.view('float32', 4),
            harmonicCloseness: edgesOnly ? undefined : context.view('float32', 4),
            reachedCounts: edgesOnly ? undefined : context.view('uint32', 4),
            status: {converged: context.view('uint32', 1)}
          }
        })
      );
    }
  );
export const EXPANSION_CASES: readonly KernelAuditCase[] = [...cases, ...EXPANSION_RECIPE_CASES];
