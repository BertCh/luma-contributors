// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialTwoStageLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {CPUSpatialTwoStageLeastSquaresOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {
  GPUSpatialTwoStageLeastSquaresOperation,
  type SpatialTwoStageLeastSquaresOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation';
import fixtures from './spatial-two-stage-least-squares-audit-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

// GPUCPUOperation.execute receives borrowed callback chunks, not a CPUVector instance.
function makeCPUVector(format: 'float32' | 'uint32', values: Float32Array | Uint32Array) {
  return {
    format,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

for (const fixture of fixtures) {
  for (const method of ['classical', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'CLUSTER'] as const) {
    it(`IV independent oracle: ${fixture.name}, order ${fixture.instrumentOrder}, ${method}`, () => {
      const result = computeSpatialTwoStageLeastSquaresOnCPU(
        {
          predictors: Float32Array.from(fixture.predictors),
          response: Float32Array.from(fixture.response),
          weights: {
            offsets: Uint32Array.from(fixture.weights.offsets),
            neighbors: Uint32Array.from(fixture.weights.neighbors),
            weights: Float32Array.from(fixture.weights.weights)
          },
          robust:
            method === 'classical'
              ? undefined
              : {
                  method,
                  kernel:
                    method === 'HAC'
                      ? {
                          offsets: Uint32Array.from(fixture.kernel.offsets),
                          neighbors: Uint32Array.from(fixture.kernel.neighbors),
                          weights: Float32Array.from(fixture.kernel.weights)
                        }
                      : undefined,
                  clusters: method === 'CLUSTER' ? Uint32Array.from(fixture.clusters) : undefined,
                  clusterCount: method === 'CLUSTER' ? fixture.clusterCount : undefined
                }
        },
        {predictorCount: fixture.predictorCount, instrumentOrder: fixture.instrumentOrder as 1 | 2}
      );
      expect(Array.from(result.status)).toEqual([0]);
      const expectedTable = method === 'classical' ? fixture.table : fixture.robust[method].table;
      expectRegressionRowsClose(result.table, Float32Array.from(expectedTable), 3e-6);
      expectRegressionRowsClose(
        result.summary,
        Float32Array.from(fixture.summary, value => value ?? NaN),
        3e-6
      );
      expectRegressionRowsClose(result.residuals, Float32Array.from(fixture.residuals), 3e-6);
      if (method !== 'classical') {
        expect(Array.from(result.robust!.status)).toEqual([0]);
        expectRegressionRowsClose(
          result.robust!.covariance,
          Float32Array.from(fixture.robust[method].covariance),
          3e-6
        );
      }
      const differences = {
        table: [result.table, expectedTable],
        summary: [result.summary, fixture.summary],
        residuals: [result.residuals, fixture.residuals],
        ...(method === 'classical'
          ? {}
          : {covariance: [result.robust!.covariance, fixture.robust[method].covariance]})
      };
      for (const [port, [actual, expected]] of Object.entries(differences))
        console.info(
          `${fixture.name}/${fixture.instrumentOrder}/${method} ${port}: ${Math.max(...Array.from(actual!, (value, row) => (expected![row] === null ? 0 : Math.abs(value! - expected![row]!))))}`
        );
    });
  }
}

for (const flag of ['overflow', 'candidateOverflow', 'complete'] as const) {
  it(`HAC ${flag} suppresses only CPU inference and recovers against the NumPy oracle`, () => {
    const fixture = fixtures[0];
    const loss = Uint32Array.of(flag === 'complete' ? 0 : 1);
    const input = {
      predictors: Float32Array.from(fixture.predictors),
      response: Float32Array.from(fixture.response),
      weights: {
        offsets: Uint32Array.from(fixture.weights.offsets),
        neighbors: Uint32Array.from(fixture.weights.neighbors),
        weights: Float32Array.from(fixture.weights.weights)
      },
      robust: {
        method: 'HAC' as const,
        kernel: {
          offsets: Uint32Array.from(fixture.kernel.offsets),
          neighbors: Uint32Array.from(fixture.kernel.neighbors),
          weights: Float32Array.from(fixture.kernel.weights),
          status: {[flag]: loss}
        }
      }
    };
    for (const value of flag === 'complete' ? [1, 0, 1] : [1, 0]) {
      const incomplete = flag === 'complete' ? value === 0 : value !== 0;
      loss[0] = value;
      const result = computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 2});
      expect(Array.from(result.status)).toEqual([0]);
      expect(Array.from(result.robust!.status)).toEqual([incomplete ? 4 : 0]);
      expectRegressionRowsClose(
        result.summary,
        fixture.summary.map(entry => entry ?? NaN),
        3e-6
      );
      expectRegressionRowsClose(result.residuals, fixture.residuals, 3e-6);
      if (incomplete) {
        expect(result.robust!.covariance.every(Number.isNaN)).toBe(true);
        for (let row = 0; row < 4; row++) {
          expectRegressionRowsClose(
            result.table.subarray(row * 4, row * 4 + 1),
            [fixture.table[row * 4]],
            3e-6
          );
          expect(result.table.subarray(row * 4 + 1, row * 4 + 4).every(Number.isNaN)).toBe(true);
        }
      } else {
        expectRegressionRowsClose(result.table, fixture.robust.HAC.table, 3e-6);
        expectRegressionRowsClose(result.robust!.covariance, fixture.robust.HAC.covariance, 3e-6);
      }
    }
  });

  it(`named HAC ${flag} survives repeated snapshots and caller mutation in both operations`, async () => {
    const fixture = fixtures[0];
    const flagPort = {
      complete: 'kernelComplete',
      overflow: 'kernelOverflow',
      candidateOverflow: 'kernelCandidateOverflow'
    }[flag];
    const loss = Uint32Array.of(flag === 'complete' ? 0 : 1);
    const props: SpatialTwoStageLeastSquaresOperationProps = {
      id: 'hac-loss',
      rowCount: 12,
      predictorCount: 2,
      predictors: 'predictors',
      response: 'response',
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      robust: {
        method: 'HAC',
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
    const cpu = new CPUSpatialTwoStageLeastSquaresOperation(props);
    const gpu = new GPUSpatialTwoStageLeastSquaresOperation(props);
    expect(cpu.inputs[flagPort]).toBe('loss');
    expect(gpu.inputs[flagPort]).toBe('loss');
    Object.assign(props.robust!.kernel!.status!, {[flag]: 'replacement'});
    props.robust!.output.status.code = 'replacement-status';
    const inputs = {
      offsets: makeCPUVector('uint32', Uint32Array.from(fixture.weights.offsets)),
      neighbors: makeCPUVector('uint32', Uint32Array.from(fixture.weights.neighbors)),
      weights: makeCPUVector('float32', Float32Array.from(fixture.weights.weights)),
      predictors: makeCPUVector('float32', Float32Array.from(fixture.predictors)),
      response: makeCPUVector('float32', Float32Array.from(fixture.response)),
      kernelOffsets: makeCPUVector('uint32', Uint32Array.from(fixture.kernel.offsets)),
      kernelNeighbors: makeCPUVector('uint32', Uint32Array.from(fixture.kernel.neighbors)),
      kernelWeights: makeCPUVector('float32', Float32Array.from(fixture.kernel.weights)),
      [flagPort]: makeCPUVector('uint32', loss)
    };
    try {
      for (const value of flag === 'complete' ? [1, 0, 1] : [1, 0]) {
        const incomplete = flag === 'complete' ? value === 0 : value !== 0;
        loss[0] = value;
        const result = (await cpu.execute({inputs})) as Record<string, Float32Array | Uint32Array>;
        expect(Array.from(result['code'])).toEqual([0]);
        expect(Array.from(result['robustStatus'])).toEqual([incomplete ? 4 : 0]);
        expectRegressionRowsClose(
          result['summary'],
          fixture.summary.map(entry => entry ?? NaN),
          3e-6
        );
        expectRegressionRowsClose(result['residuals'], fixture.residuals, 3e-6);
        if (incomplete) {
          expect(result['covariance'].every(Number.isNaN)).toBe(true);
          for (let row = 0; row < 4; row++) {
            expectRegressionRowsClose(
              result['table'].subarray(row * 4, row * 4 + 1),
              [fixture.table[row * 4]],
              3e-6
            );
            expect(result['table'].subarray(row * 4 + 1, row * 4 + 4).every(Number.isNaN)).toBe(
              true
            );
          }
        } else {
          expectRegressionRowsClose(result['table'], fixture.robust.HAC.table, 3e-6);
          expectRegressionRowsClose(result['covariance'], fixture.robust.HAC.covariance, 3e-6);
        }
      }
    } finally {
      gpu.destroy();
    }
  });
}
