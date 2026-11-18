// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {
  computeSpatialPanelLagMaximumLikelihoodOnCPU,
  CPUSpatialPanelLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  GPUSpatialPanelLagMaximumLikelihoodOperation,
  type SpatialPanelLagMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation';
import {
  runOperationGraph,
  makeCPUVector,
  getCPUVectorRows
} from '../../utils/mixed-execution-test-utils';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {
  expectPanelLagCovarianceClose,
  expectPanelLagPredictionsClose
} from './panel-lag-comparisons.test-utils';
import fixtures from './spatial-panel-lag-maximum-likelihood-operation-fixtures.json';

const FIXTURE = fixtures[1];
const WEIGHTS = new Float32Array(FIXTURE.weights.weights.length).fill(0.5);
const OUTPUTS = ['coefficients', 'covariance', 'summary', 'status', 'fitted', 'residuals'];

for (const [fitMode, getterStatus] of [
  ['reference', false],
  ['reference', true],
  ['tiled', false],
  ['tiled', true]
] as const) {
  it(`refuses candidate-only weights then recovers on the same compiled panel graph: ${fitMode} getter=${getterStatus}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = {
      weights: {
        offsets: Uint32Array.from(FIXTURE.weights.offsets),
        neighbors: Uint32Array.from(FIXTURE.weights.neighbors),
        weights: WEIGHTS
      },
      predictors: Float32Array.from(FIXTURE.predictors),
      response: Float32Array.from(FIXTURE.response)
    };
    const options = {
      fitMode,
      periodCount: FIXTURE.periodCount,
      predictorCount: FIXTURE.predictorCount,
      effects: 'unit' as const
    };
    const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, {
      ...options,
      fitMode: 'reference'
    });
    expect(expected.status[0]).toBe(0);
    const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
      Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
    );
    for (const Operation of [
      GPUSpatialPanelLagMaximumLikelihoodOperation,
      CPUSpatialPanelLagMaximumLikelihoodOperation
    ]) {
      const graph = new GPUGraph({id: `panel-candidate-${fitMode}`});
      const inputs = {
        offsets: makeCPUVector('uint32', input.weights.offsets),
        neighbors: makeCPUVector('uint32', input.weights.neighbors),
        weights: makeCPUVector('float32', input.weights.weights),
        predictors: makeCPUVector('float32', input.predictors),
        response: makeCPUVector('float32', input.response),
        overflow: makeCPUVector('uint32', Uint32Array.of(0)),
        candidateOverflow: makeCPUVector('uint32', Uint32Array.of(1)),
        ...(fitMode === 'tiled' ? {eigenvalues: makeCPUVector('float32', eigenvalues)} : {})
      };
      for (const [name, values] of Object.entries(inputs))
        graph.addInput(
          name,
          gpuGraph.vector(values.format as 'float32' | 'uint32', {length: values.length})
        );
      let reads = 0;
      let flagName = 'candidateOverflow';
      const weightsStatus = getterStatus
        ? Object.defineProperty({overflow: 'overflow'}, 'candidateOverflow', {
            enumerable: false,
            get() {
              reads++;
              return flagName;
            }
          })
        : {overflow: 'overflow', candidateOverflow: 'candidateOverflow'};
      const operation = new Operation({
        ...options,
        entityCount: FIXTURE.entityCount,
        eigenvalues: fitMode === 'tiled' ? 'eigenvalues' : undefined,
        predictors: 'predictors',
        response: 'response',
        weights: {
          offsets: 'offsets',
          neighbors: 'neighbors',
          weights: 'weights',
          status: weightsStatus
        },
        output: {
          coefficients: 'coefficients',
          covariance: 'covariance',
          summary: 'summary',
          status: 'status',
          fitted: 'fitted',
          residuals: 'residuals'
        }
      });
      flagName = 'lost';
      if (getterStatus) {
        expect(operation.inputs['weightsStatus_candidateOverflow']).toBe('candidateOverflow');
        expect(reads).toBe(1);
      }
      graph.add(operation);
      const compiled = graph.compile(device, {
        outputs: OUTPUTS,
        outputLocations: Object.fromEntries(OUTPUTS.map(name => [name, 'cpu' as const])),
        lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
      });
      try {
        for (const candidateOverflow of [1, 0]) {
          const outputs = (await compiled.run({
            inputs: {
              ...inputs,
              candidateOverflow: makeCPUVector('uint32', Uint32Array.of(candidateOverflow))
            }
          })) as Record<string, unknown>;
          const rows = Object.fromEntries(
            OUTPUTS.map(name => {
              const vector = outputs[name];
              expect(vector).toBeInstanceOf(CPUVector);
              return [name, getCPUVectorRows(vector as CPUVector)];
            })
          );
          expect(Array.from(rows['status'])).toEqual([candidateOverflow ? 1 : 0]);
          if (candidateOverflow) {
            for (const name of OUTPUTS.filter(name => name !== 'status'))
              expect((rows[name] as Float32Array).every(Number.isNaN), name).toBe(true);
          } else {
            expect(rows['fitted'][0]).toBeCloseTo(-0.31592104, 3);
            expectPanelLagPredictionsClose(rows['fitted'], expected.fitted, input.response);
            expectPanelLagPredictionsClose(rows['residuals'], expected.residuals, input.response);
            expectPanelLagCovarianceClose(
              rows['covariance'],
              expected.covariance,
              FIXTURE.predictorCount + 2
            );
          }
        }
      } finally {
        compiled.destroy();
        if (operation instanceof GPUSpatialPanelLagMaximumLikelihoodOperation) operation.destroy();
      }
    }
  });
}

for (const scene of [
  'valid',
  'zero',
  'short',
  'long',
  'nonfinite',
  'outOfRange',
  'oneStep',
  'mutatedProps'
] as const) {
  it(`both panel graph entry points consume the entity spectrum: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const eigenvalues = Float32Array.from({length: FIXTURE.entityCount}, (_, index) =>
      Math.cos((2 * Math.PI * index) / FIXTURE.entityCount)
    );
    if (scene === 'zero' || scene === 'mutatedProps') eigenvalues.fill(0);
    if (scene === 'nonfinite') eigenvalues[2] = NaN;
    if (scene === 'outOfRange') eigenvalues[2] = 1.01;
    const spectrum =
      scene === 'short'
        ? eigenvalues.slice(1)
        : scene === 'long'
          ? Float32Array.from([...eigenvalues, 0])
          : eigenvalues;
    const input = {
      weights: {
        offsets: Uint32Array.from(FIXTURE.weights.offsets),
        neighbors: Uint32Array.from(FIXTURE.weights.neighbors),
        weights: WEIGHTS
      },
      predictors: Float32Array.from(FIXTURE.predictors),
      response: Float32Array.from(FIXTURE.response)
    };
    const options = {
      periodCount: FIXTURE.periodCount,
      predictorCount: FIXTURE.predictorCount,
      effects: 'unit' as const
    };
    for (const Operation of [
      GPUSpatialPanelLagMaximumLikelihoodOperation,
      CPUSpatialPanelLagMaximumLikelihoodOperation
    ]) {
      const operationProps: SpatialPanelLagMaximumLikelihoodOperationProps = {
        ...options,
        id: 'review-panel',
        entityCount: FIXTURE.entityCount,
        eigenvalues: 'eigenvalues',
        maximumSearchIterations: scene === 'oneStep' ? 1 : undefined,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        predictors: 'predictors',
        response: 'response',
        output: {
          coefficients: 'coefficients',
          covariance: 'covariance',
          summary: 'summary',
          status: 'status',
          fitted: 'fitted',
          residuals: 'residuals'
        }
      };
      const operation = new Operation(operationProps);
      if (scene === 'mutatedProps') {
        operationProps.eigenvalues = undefined;
        operationProps.fitMode = 'reference';
        operationProps.effects = 'random';
        operationProps.output.coefficients = 'mutated';
      }
      try {
        const run = () =>
          runOperationGraph(device, {
            inputs: {
              offsets: {format: 'uint32', values: input.weights.offsets},
              neighbors: {format: 'uint32', values: input.weights.neighbors},
              weights: {format: 'float32', values: WEIGHTS},
              predictors: {format: 'float32', values: input.predictors},
              response: {format: 'float32', values: input.response},
              eigenvalues: {format: 'float32', values: spectrum}
            },
            operations: [operation],
            outputs: OUTPUTS
          });
        if (scene === 'short' || scene === 'long') {
          await expect(run()).rejects.toEqual(
            new Error('review-panel eigenvalues must hold one value per entity')
          );
          continue;
        }
        const result = await run();
        if (scene === 'oneStep') {
          expect(Array.from(result['status'])).toEqual([3]);
          expect(result['summary'][4]).toBeCloseTo((1.998 * (Math.sqrt(5) - 1)) / 2, 5);
          expect((result['covariance'] as Float32Array).every(Number.isNaN)).toBe(true);
          expect((result['fitted'] as Float32Array).every(Number.isFinite)).toBe(true);
          continue;
        }
        if (scene === 'nonfinite' || scene === 'outOfRange') {
          expect(Array.from(result['status'])).toEqual([1]);
          for (const port of OUTPUTS.filter(port => port !== 'status')) {
            expect((result[port] as Float32Array).every(Number.isNaN)).toBe(true);
          }
          continue;
        }
        // For the valid ring, the oracle uses the dense CSR determinant, independently of
        // the spectrum. A deliberately zero spectrum must also change the graph result.
        const expected = computeSpatialPanelLagMaximumLikelihoodOnCPU(
          scene === 'zero' || scene === 'mutatedProps' ? {...input, eigenvalues} : input,
          options
        );
        expect(Array.from(result['status'])).toEqual([0]);
        expectRegressionRowsClose(result['coefficients'], expected.coefficients, 0.005);
        // Off-diagonal covariance can approach zero without the model being ill-conditioned.
        // Use the established panel standard-error normalization rather than entrywise error.
        expectPanelLagCovarianceClose(
          result['covariance'],
          expected.covariance,
          FIXTURE.predictorCount + 2
        );
        for (const port of ['fitted', 'residuals'] as const) {
          expectPanelLagPredictionsClose(result[port], expected[port], input.response);
        }
        expectRegressionRowsClose(
          result['summary'].subarray(0, 2),
          expected.summary.subarray(0, 2),
          0.005
        );
        if (scene === 'zero' || scene === 'mutatedProps') {
          const dense = computeSpatialPanelLagMaximumLikelihoodOnCPU(input, options);
          const rhoIndex = FIXTURE.predictorCount + 1;
          // Without a determinant term, rho is the independent within-OLS optimum. A
          // narrow golden-section bracket must not mask an objective rounded flat in f32.
          expect(
            Math.abs(result['coefficients'][rhoIndex] - expected.coefficients[rhoIndex])
          ).toBeLessThan(2e-6);
          expect(
            Math.abs(result['coefficients'][rhoIndex] - dense.coefficients[rhoIndex])
          ).toBeGreaterThan(0.01);
        }
      } finally {
        if (operation instanceof GPUSpatialPanelLagMaximumLikelihoodOperation) operation.destroy();
      }
    }
  });
}
