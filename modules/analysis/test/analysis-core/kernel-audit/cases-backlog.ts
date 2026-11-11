// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as spatial from '../../../src/gpu-spatial-analysis/index';
import * as raster from '@luma-contributors/analysis/raster-analysis';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import * as network from '../../../src/gpu-network/index';
import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';
import type {KernelAuditCase} from './kernel-audit-context';

const cases: KernelAuditCase[] = [];
const ROWS = 16384;
for (const residualPower of [1, 2] as const) {
  for (const alternative of ['greater', 'lesser', 'two-sided'] as const) {
    cases.push({
      exportName: 'GPULOSHResidualPermutationTest',
      name: `power ${residualPower}, ${alternative}, all outputs and upstream overflow`,
      build(context) {
        context.graph.add(
          new spatial.GPULOSHResidualPermutationTest({
            weights: {...context.weights(ROWS, 64), status: {overflow: context.view('uint32', 1)}},
            values: context.view('float32', ROWS),
            mask: context.view('uint32', ROWS),
            parameters: context.view('uint32', 4),
            maximumPermutations: 999,
            maximumNeighbors: 64,
            multipleTesting: 'bonferroni',
            residualPower,
            alternative,
            output: {
              statistic: context.view('float32', ROWS),
              status: context.view('uint32', ROWS),
              normalizedResiduals: context.view('float32', ROWS),
              inference: {
                exceedances: context.view('uint32', ROWS),
                pseudoPValues: context.view('float32', ROWS),
                adjustedPValues: context.view('float32', ROWS),
                significant: context.view('uint32', ROWS),
                status: {overflow: context.view('uint32', 1)}
              }
            }
          })
        );
      }
    });
  }
}
for (const family of ['poisson', 'logistic'] as const) {
  for (const kernel of ['gaussian', 'bisquare'] as const) {
    cases.push({
      exportName: 'GPUGeneralizedGeographicallyWeightedRegression',
      name: `adaptive ${family} ${kernel}, 128 neighbors, maximum predictors and iterations`,
      build(context) {
        context.graph.add(
          new spatial.GPUGeneralizedGeographicallyWeightedRegression({
            positions: context.view('float32x2', ROWS),
            predictors: context.view('float32', ROWS * 11),
            response: context.view('float32', ROWS),
            offset: context.view('float32', ROWS),
            mask: context.view('uint32', ROWS),
            parameters: context.view('float32', 2),
            predictorCount: 11,
            maximumIterations: 128,
            maximumNeighborCount: 128,
            bandwidthMode: 'adaptive',
            family,
            kernel,
            output: {
              coefficients: context.view('float32', ROWS * 12),
              fitted: context.view('float32', ROWS),
              status: {
                codes: context.view('uint32', ROWS),
                iterationCount: context.view('uint32', ROWS)
              }
            }
          })
        );
      }
    });
  }
}
for (const instrumentOrder of [1, 2] as const) {
  cases.push({
    exportName: 'GPUSpatialPanelRegression',
    name: `unit effects, instrument order ${instrumentOrder}, maximum periods and predictors`,
    build(context) {
      const predictorCount = instrumentOrder === 2 ? 5 : 8;
      const rows = 256 * 1024;
      context.graph.add(
        new spatial.GPUSpatialPanelRegression({
          weights: {...context.weights(256, 64), status: {overflow: context.view('uint32', 1)}},
          periodCount: 1024,
          effects: 'unit',
          predictorCount,
          predictors: context.view('float32', rows * predictorCount),
          response: context.view('float32', rows),
          instrumentOrder,
          output: {
            table: context.view('float32', (predictorCount + 2) * 4),
            summary: context.view('float32', 32),
            residuals: context.view('float32', rows),
            status: {code: context.view('uint32', 1)}
          }
        })
      );
    }
  });
}
const GRID: GPURasterMetadata = {
  width: 128,
  height: 128,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
cases.push({
  exportName: 'GPURasterFillNoData',
  name: 'GDAL four-quadrant, maximum radius, signed calibrated masked band',
  build(context) {
    context.graph.add(
      new raster.GPURasterFillNoData({
        grid: GRID,
        input: {
          id: 'source',
          format: 'sint32',
          storage: {kind: 'buffer', values: context.view('sint32', ROWS)},
          validity: context.view('uint32', ROWS),
          scale: 2,
          offset: 1,
          noDataValue: -9999
        },
        method: 'gdal-four-quadrant',
        maximumDistance: 256,
        output: {
          values: context.view('float32', ROWS),
          validity: context.view('uint32', ROWS),
          filled: context.view('uint32', ROWS),
          status: {invalidCount: context.view('uint32', 1)}
        }
      })
    );
  }
});
for (const method of ['nearest', 'bilinear'] as const) {
  for (const noDataPolicy of ['strict', 'renormalize'] as const) {
    cases.push({
      exportName: 'GPURasterTileReprojection',
      name: `${method} ${noDataPolicy}, overlapping tiles, projection, calibrated inputs`,
      build(context) {
        context.graph.add(
          new raster.GPURasterTileReprojection({
            sourceGrid: GRID,
            destinationGrid: GRID,
            sources: [
              {
                pixelBounds: [0, 0, 128, 128],
                input: {
                  id: 'first',
                  format: 'sint32',
                  storage: {kind: 'buffer', values: context.view('sint32', ROWS)},
                  validity: context.view('uint32', ROWS),
                  scale: 2,
                  offset: 1,
                  noDataValue: -9999
                }
              },
              {
                pixelBounds: [64, 0, 128, 128],
                input: {
                  id: 'second',
                  format: 'float32',
                  storage: {kind: 'buffer', values: context.view('float32', ROWS / 2)},
                  validity: context.view('uint32', ROWS / 2)
                }
              }
            ],
            method,
            noDataPolicy,
            inverseProjection: compileProjectionProgram({
              precision: 'local-f32',
              destinationOrigin: [10, 20],
              operations: [{type: 'affine', scale: [2, 2], offset: [10, 20]}]
            }),
            output: {values: context.view('float32', ROWS), validity: context.view('uint32', ROWS)}
          })
        );
      }
    });
  }
}
for (const maximumFrontierSize of [1024, 65536]) {
  cases.push({
    exportName: 'GPUAlternativeRoutes',
    name: `heap ${maximumFrontierSize}, 256 nodes, 32 routes, binding-limited queries`,
    build(context) {
      const queries = Math.floor(context.maximumRows(4) / (maximumFrontierSize * 514));
      // At the maximum frontier, one 256-node query needs slightly more than 128 MiB.
      // Use the largest admissible node count for that frontier on the core device.
      const nodes = maximumFrontierSize === 65536 ? 255 : 256;
      const queryCount = Math.max(1, queries);
      context.graph.add(
        new network.GPUAlternativeRoutes({
          topology: {
            offsets: context.view('uint32', nodes + 1),
            neighbors: context.view('uint32', nodes * nodes),
            weights: context.view('float32', nodes * nodes)
          },
          sources: context.view('uint32', queryCount),
          targets: context.view('uint32', queryCount),
          routeCount: 32,
          maximumFrontierSize,
          maximumExpansions: 10000000,
          output: {
            vertices: context.view('uint32', queryCount * 32 * nodes),
            edges: context.view('uint32', queryCount * 32 * nodes),
            lengths: context.view('uint32', queryCount * 32),
            costs: context.view('float32', queryCount * 32),
            counts: context.view('uint32', queryCount),
            complete: context.view('uint32', queryCount)
          }
        })
      );
    }
  });
}
export const BACKLOG_CASES: readonly KernelAuditCase[] = cases;
