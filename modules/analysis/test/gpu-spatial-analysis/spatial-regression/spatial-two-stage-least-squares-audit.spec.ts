// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {GPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {
  runOperationGraph,
  makeCPUVector,
  getCPUVectorRows
} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-two-stage-least-squares-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const fixture of fixtures) {
  for (const method of ['classical', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'CLUSTER'] as const) {
    it(`GPU IV independent oracle: ${fixture.name}, order ${fixture.instrumentOrder}, ${method}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const robust = method !== 'classical';
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
          weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          response: {format: 'float32', values: Float32Array.from(fixture.response)},
          ...(method === 'HAC'
            ? {
                kernelOffsets: {
                  format: 'uint32' as const,
                  values: Uint32Array.from(fixture.kernel.offsets)
                },
                kernelNeighbors: {
                  format: 'uint32' as const,
                  values: Uint32Array.from(fixture.kernel.neighbors)
                },
                kernelWeights: {
                  format: 'float32' as const,
                  values: Float32Array.from(fixture.kernel.weights)
                }
              }
            : {}),
          ...(method === 'CLUSTER'
            ? {clusters: {format: 'uint32' as const, values: Uint32Array.from(fixture.clusters)}}
            : {})
        },
        operations: [
          new GPUSpatialTwoStageLeastSquaresOperation({
            rowCount: fixture.response.length,
            predictorCount: fixture.predictorCount,
            instrumentOrder: fixture.instrumentOrder as 1 | 2,
            weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            predictors: 'predictors',
            response: 'response',
            robust: robust
              ? {
                  method,
                  kernel:
                    method === 'HAC'
                      ? {
                          offsets: 'kernelOffsets',
                          neighbors: 'kernelNeighbors',
                          weights: 'kernelWeights'
                        }
                      : undefined,
                  clusters: method === 'CLUSTER' ? 'clusters' : undefined,
                  clusterCount: method === 'CLUSTER' ? fixture.clusterCount : undefined,
                  output: {covariance: 'covariance', status: {code: 'robustStatus'}}
                }
              : undefined,
            output: {
              table: 'table',
              summary: 'summary',
              residuals: 'residuals',
              status: {code: 'code'}
            }
          })
        ],
        outputs: [
          'table',
          'summary',
          'residuals',
          'code',
          ...(robust ? ['covariance', 'robustStatus'] : [])
        ]
      });
      expect(Array.from(result['code'])).toEqual([0]);
      const expectedTable = method === 'classical' ? fixture.table : fixture.robust[method].table;
      // 0.5% budget for float32 normal solves and nested IV projection/covariance/diagnostics.
      expectRegressionRowsClose(result['table'], expectedTable, 5e-3);
      expectRegressionRowsClose(
        result['summary'],
        fixture.summary.map(value => value ?? NaN),
        5e-3
      );
      expectRegressionRowsClose(result['residuals'], fixture.residuals, 5e-3);
      for (const slot of [0, 10, 12]) expect(result['summary'][slot]).toBe(fixture.summary[slot]);
      if (method !== 'classical') {
        expect(Array.from(result['robustStatus'])).toEqual([0]);
        expectRegressionRowsClose(result['covariance'], fixture.robust[method].covariance, 5e-3);
      }
    });
  }
}

for (const flag of ['overflow', 'candidateOverflow', 'complete'] as const) {
  it(`both named IV forms suppress HAC ${flag} and recover on the same graph against NumPy`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[0];
    for (const Operation of [
      GPUSpatialTwoStageLeastSquaresOperation,
      CPUSpatialTwoStageLeastSquaresOperation
    ]) {
      const loss = Uint32Array.of(flag === 'complete' ? 0 : 1);
      const inputs = {
        offsets: makeCPUVector('uint32', Uint32Array.from(fixture.weights.offsets)),
        neighbors: makeCPUVector('uint32', Uint32Array.from(fixture.weights.neighbors)),
        weights: makeCPUVector('float32', Float32Array.from(fixture.weights.weights)),
        predictors: makeCPUVector('float32', Float32Array.from(fixture.predictors)),
        response: makeCPUVector('float32', Float32Array.from(fixture.response)),
        kernelOffsets: makeCPUVector('uint32', Uint32Array.from(fixture.kernel.offsets)),
        kernelNeighbors: makeCPUVector('uint32', Uint32Array.from(fixture.kernel.neighbors)),
        kernelWeights: makeCPUVector('float32', Float32Array.from(fixture.kernel.weights)),
        loss: makeCPUVector('uint32', loss)
      };
      const props = {
        rowCount: 12,
        predictorCount: 2,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        predictors: 'predictors',
        response: 'response',
        robust: {
          method: 'HAC' as const,
          kernel: {
            offsets: 'kernelOffsets',
            neighbors: 'kernelNeighbors',
            weights: 'kernelWeights',
            status: {[flag]: 'loss'}
          },
          output: {covariance: 'covariance', status: {code: 'robustStatus'}}
        },
        output: {table: 'table', summary: 'summary', residuals: 'residuals', status: {code: 'code'}}
      };
      const operation = new Operation(props);
      // Mutation must not disconnect the captured loss graph value or redirect a result.
      Object.assign(props.robust.kernel.status, {[flag]: 'replacement'});
      props.robust.output.status.code = 'replacement-status';
      const graph = new GPUGraph();
      for (const [name, vector] of Object.entries(inputs))
        graph.addInput(
          name,
          gpuGraph.vector(vector.format as 'float32' | 'uint32', {length: vector.length})
        );
      graph.add(operation);
      const names = ['code', 'table', 'summary', 'residuals', 'covariance', 'robustStatus'];
      const compiled = graph.compile(device, {
        outputs: names,
        outputLocations: Object.fromEntries(names.map(name => [name, 'cpu' as const])),
        lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
      });
      try {
        for (const value of flag === 'complete' ? [1, 0, 1] : [1, 0]) {
          const incomplete = flag === 'complete' ? value === 0 : value !== 0;
          inputs.loss = makeCPUVector('uint32', Uint32Array.of(value));
          const outputs = (await compiled.run({inputs})) as Record<string, unknown>;
          const result = Object.fromEntries(
            names.map(name => {
              const output = outputs[name];
              if (!(output instanceof CPUVector)) throw new Error(`${name} must be a CPU output`);
              return [name, getCPUVectorRows(output)];
            })
          );
          expect(Array.from(result['code'])).toEqual([0]);
          expect(Array.from(result['robustStatus'])).toEqual([incomplete ? 4 : 0]);
          // Same 0.5% float32 projection/normal-solve budget as the independent audit above.
          expectRegressionRowsClose(
            result['summary'],
            fixture.summary.map(entry => entry ?? NaN),
            5e-3
          );
          expectRegressionRowsClose(result['residuals'], fixture.residuals, 5e-3);
          if (incomplete) {
            expect(result['covariance'].every(Number.isNaN)).toBe(true);
            for (let row = 0; row < 4; row++) {
              expectRegressionRowsClose(
                result['table'].subarray(row * 4, row * 4 + 1),
                [fixture.table[row * 4]],
                5e-3
              );
              expect(result['table'].subarray(row * 4 + 1, row * 4 + 4).every(Number.isNaN)).toBe(
                true
              );
            }
          } else {
            expectRegressionRowsClose(result['table'], fixture.robust.HAC.table, 5e-3);
            expectRegressionRowsClose(result['covariance'], fixture.robust.HAC.covariance, 5e-3);
          }
        }
      } finally {
        compiled.destroy();
        if ('destroy' in operation) operation.destroy();
      }
    }
  });
}
