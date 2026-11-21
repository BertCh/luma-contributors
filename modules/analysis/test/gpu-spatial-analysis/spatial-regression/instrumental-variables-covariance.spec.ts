// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {
  GPUInstrumentalVariablesCovariance,
  GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS,
  type GPUInstrumentalVariablesCovarianceProps,
  type GPUInstrumentalVariablesCovarianceMethod
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-two-stage-least-squares-operation-fixtures.json';
import auditFixtures from './spatial-two-stage-least-squares-audit-fixtures.json';

type CovarianceInput = {
  design: Float32Array;
  coefficientCount: number;
  instruments: Float32Array;
  instrumentCount: number;
  residuals: Float32Array;
  method: GPUInstrumentalVariablesCovarianceMethod;
  clusters?: Uint32Array;
  clusterCount?: number;
  kernel?: {offsets: number[]; neighbors: number[]; weights: number[]};
};

async function runCovariance(
  input: CovarianceInput,
  options: {
    overflow?: Uint32Array;
    candidateOverflow?: Uint32Array;
    mutateAfterConstruction?: boolean;
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs: Record<string, OperationGraphInput> = {
    design: {format: 'float32', values: input.design},
    instruments: {format: 'float32', values: input.instruments},
    residuals: {format: 'float32', values: input.residuals}
  };
  if (input.clusters) inputs['clusters'] = {format: 'uint32', values: input.clusters};
  if (input.kernel) {
    inputs['kernelOffsets'] = {format: 'uint32', values: Uint32Array.from(input.kernel.offsets)};
    inputs['kernelNeighbors'] = {
      format: 'uint32',
      values: Uint32Array.from(input.kernel.neighbors)
    };
    inputs['kernelWeights'] = {format: 'float32', values: Float32Array.from(input.kernel.weights)};
    if (options.overflow) inputs['overflow'] = {format: 'uint32', values: options.overflow};
    if (options.candidateOverflow)
      inputs['candidateOverflow'] = {format: 'uint32', values: options.candidateOverflow};
  }
  return await runOperationGraph(device, {
    inputs,
    operations: [
      new GPUContributorOperation({
        id: 'direct-iv-covariance',
        inputs: Object.fromEntries(Object.keys(inputs).map(name => [name, name])),
        outputs: {
          covariance: {
            value: 'covariance',
            format: 'float32',
            length: (input.coefficientCount + 1) ** 2
          },
          code: {value: 'code', format: 'uint32', length: 1}
        },
        createContributor: getView => {
          const props: GPUInstrumentalVariablesCovarianceProps = {
            ...input,
            design: getView<'float32'>('design'),
            instruments: getView<'float32'>('instruments'),
            residuals: getView<'float32'>('residuals'),
            clusters: input.clusters ? getView<'uint32'>('clusters') : undefined,
            kernel: input.kernel
              ? {
                  offsets: getView<'uint32'>('kernelOffsets'),
                  neighbors: getView<'uint32'>('kernelNeighbors'),
                  weights: getView<'float32'>('kernelWeights'),
                  status: {
                    ...(options.overflow ? {overflow: getView<'uint32'>('overflow')} : {}),
                    ...(options.candidateOverflow
                      ? {candidateOverflow: getView<'uint32'>('candidateOverflow')}
                      : {})
                  }
                }
              : undefined,
            output: {
              covariance: getView<'float32'>('covariance'),
              status: {code: getView<'uint32'>('code')}
            }
          };
          const contributor = new GPUInstrumentalVariablesCovariance(props);
          if (options.mutateAfterConstruction) {
            props.coefficientCount = 1;
            props.instrumentCount = 1;
            props.method = 'HC0';
            props.design = props.residuals;
            props.output.covariance = props.residuals;
            props.output.status = {code: getView<'uint32'>('code')};
            if (props.kernel?.status) {
              delete props.kernel.status.overflow;
              delete props.kernel.status.candidateOverflow;
            }
          }
          return contributor;
        }
      })
    ],
    outputs: ['covariance', 'code']
  });
}

it('direct IV covariance retains its validated model after caller-map mutation against NumPy', async () => {
  const fixture = auditFixtures[0];
  const result = await runCovariance(getFixtureInput('HC1', fixture), {
    mutateAfterConstruction: true
  });
  expect(Array.from(result['code'])).toEqual([0]);
  expect(result['covariance'].length).toBe(fixture.robust.HC1.covariance.length);
  // 2e-4 covers float32 projection, score publication and the dense sandwich reduction.
  fixture.robust.HC1.covariance.forEach((value, row) =>
    expect(Math.abs(result['covariance'][row] - value)).toBeLessThanOrEqual(
      2e-4 * Math.max(Math.abs(value), 1e-3)
    )
  );
});

for (const flag of ['overflow', 'candidateOverflow'] as const) {
  for (const value of [1, 0]) {
    it(`direct IV HAC captures ${flag}=${value} and preserves the NumPy loss/clean contract`, async () => {
      const fixture = auditFixtures[0];
      const result = await runCovariance(getFixtureInput('HAC', fixture), {
        [flag]: Uint32Array.of(value),
        mutateAfterConstruction: true
      });
      expect(Array.from(result['code'])).toEqual([value ? 4 : 0]);
      if (value) expect(result['covariance'].every(Number.isNaN)).toBe(true);
      else
        fixture.robust.HAC.covariance.forEach((expected, row) =>
          expect(Math.abs(result['covariance'][row] - expected)).toBeLessThanOrEqual(
            2e-4 * Math.max(Math.abs(expected), 1e-3)
          )
        );
    });
  }
}

function getFixtureInput(
  method: GPUInstrumentalVariablesCovarianceMethod,
  fixture = fixtures[0]
): CovarianceInput {
  const rowCount = fixture.response.length;
  const predictorCount = fixture.predictorCount;
  const predictors = Array.from({length: rowCount}, (_, row) =>
    fixture.predictors.slice(row * predictorCount, (row + 1) * predictorCount)
  );
  const lag = (values: number[][]) =>
    values.map((row, position) =>
      row.map((_, column) => {
        let total = 0;
        for (
          let slot = fixture.weights.offsets[position];
          slot < fixture.weights.offsets[position + 1];
          slot++
        ) {
          total += fixture.weights.weights[slot] * values[fixture.weights.neighbors[slot]][column];
        }
        return total;
      })
    );
  const laggedPredictors = lag(predictors);
  const twiceLaggedPredictors = lag(laggedPredictors);
  const laggedResponse = lag(fixture.response.map(value => [value]));
  return {
    design: Float32Array.from(
      predictors.flatMap((row, position) => [...row, laggedResponse[position][0]])
    ),
    coefficientCount: predictorCount + 1,
    instruments: Float32Array.from(
      predictors.flatMap((row, position) => [
        ...row,
        ...laggedPredictors[position],
        ...(fixture.instrumentOrder === 2 ? twiceLaggedPredictors[position] : [])
      ])
    ),
    instrumentCount: (fixture.instrumentOrder + 1) * predictorCount,
    residuals: Float32Array.from(fixture.residuals),
    method,
    ...(method === 'CLUSTER'
      ? {clusters: Uint32Array.from(fixture.clusters), clusterCount: fixture.clusterCount}
      : {}),
    ...(method === 'HAC' ? {kernel: fixture.kernel} : {})
  };
}

for (const fixture of [...fixtures.slice(0, 2), ...auditFixtures]) {
  for (const method of ['HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'CLUSTER'] as const) {
    it(`matches the independent NumPy sandwich: ${fixture.response.length} rows, ${fixture.name}, order ${fixture.instrumentOrder}, ${method}`, async () => {
      const result = await runCovariance(getFixtureInput(method, fixture));
      expect(result['code'][0]).toBe(GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.SUCCESS);
      const expected = fixture.robust[method].covariance;
      expected.forEach((value, index) =>
        expect(Math.abs(result['covariance'][index] - value)).toBeLessThan(
          2e-4 * Math.max(1e-3, Math.abs(value))
        )
      );
    });
  }
}

for (const scene of ['oneGroup', 'missingKey', 'outOfRangeKey', 'allMissing', 'nonPSD'] as const) {
  it(`reports the named failure with NaN covariance: ${scene}`, async () => {
    const input = getFixtureInput(scene === 'nonPSD' ? 'HAC' : 'CLUSTER');
    if (scene === 'oneGroup') input.clusters!.fill(0);
    if (scene === 'missingKey') input.clusters![0] = 0xffffffff;
    if (scene === 'outOfRangeKey') input.clusters![0] = input.clusterCount!;
    if (scene === 'allMissing') input.clusters!.fill(0xffffffff);
    if (scene === 'nonPSD') input.kernel = fixtures[0].nonPositiveSemidefiniteKernel;
    const result = await runCovariance(input);
    expect(result['code'][0]).toBe(
      scene === 'nonPSD'
        ? GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.NOT_POSITIVE_SEMIDEFINITE
        : GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.INVALID_CLUSTERS
    );
    expect(result['covariance'].every(Number.isNaN)).toBe(true);
  });
}

for (const method of ['HC2', 'HC3'] as const) {
  it(`rejects unit leverage in ${method} even with a zero residual`, async () => {
    const design = Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0);
    const result = await runCovariance({
      design,
      coefficientCount: 1,
      instruments: design,
      instrumentCount: 1,
      residuals: new Float32Array(8),
      method
    });
    expect(result['code'][0]).toBe(GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.INVALID_LEVERAGE);
    expect(result['covariance'].every(Number.isNaN)).toBe(true);
  });
}

for (const scale of [0, 1e-12, 1, 1e12]) {
  it(`accepts zero and rank-deficient clustered covariance at residual scale ${scale}`, async () => {
    const design = Float32Array.of(-4, -3, -2, -1, 0, 1, 2, 3, 4);
    const residuals = Float32Array.from([1, -2, 2, -2, 1, -1, 2, -1, 0], value => value * scale);
    const clusters = Uint32Array.of(0, 2, 0, 2, 0, 2, 0, 2, 0);
    const result = await runCovariance({
      design,
      coefficientCount: 1,
      instruments: design,
      instrumentCount: 1,
      residuals,
      method: 'CLUSTER',
      clusters,
      clusterCount: 4
    });
    expect(result['code'][0]).toBe(GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.SUCCESS);
    // OLS special case: B=diag(1/9, 1/60). Group score totals (2/3, -1/15)
    // and their negatives give rank one. CR1=2*(9-1)/(9-2)=16/7.
    const expected = [128 / 63, -64 / 315, -64 / 315, 32 / 1575];
    expected.forEach((value, index) => {
      const wanted = value * scale ** 2;
      expect(Math.abs(result['covariance'][index] - wanted)).toBeLessThanOrEqual(
        2e-4 * Math.abs(wanted)
      );
    });
  });
}
