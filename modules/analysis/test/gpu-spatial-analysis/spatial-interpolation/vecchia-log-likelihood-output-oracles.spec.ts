// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUVecchiaLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {GPUVecchiaSpaceTimeLogLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation';
import {createMarginalLikelihoodInput} from './vecchia-output-oracles-fixture';
import {
  createTwoPredecessorLikelihoodInput,
  VECCHIA_TWO_PREDECESSOR_CONDITIONALS
} from './vecchia-output-oracles-fixture';
import {getSpaceTimeLikelihoodProps} from './likelihoods-operation-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph
} from '../spatial-clustering/output-oracle-test-utils';
import {
  createScalarLikelihoodInput,
  getScalarLikelihoodProps,
  VECCHIA_ORACLE_CONDITIONALS,
  VECCHIA_ORACLE_TERMS,
  expectOracleFloats
} from './vecchia-output-oracles-fixture';
import {getLikelihoodGraphInputs} from './likelihoods-operation-test-utils';

it('spatial likelihood contributor and operation overwrite poisoned conditionals and site outputs with scalar Gaussian literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = Object.fromEntries(
    Object.entries(getLikelihoodGraphInputs(createScalarLikelihoodInput())).map(([name, input]) => [
      name,
      {
        ...input,
        values: input.values as Float32Array | Uint32Array,
        length: input.values.length / (name === 'positions' ? 2 : name === 'observations' ? 4 : 1)
      }
    ])
  );
  const harness = createPoisonedOracleGraph(device, inputs, {
    conditionals: {format: 'float32x4', length: 3},
    conditionalStatus: {format: 'uint32', length: 3},
    siteTerms: {format: 'float32', length: 3},
    siteStatus: {format: 'uint32', length: 3},
    logLikelihood: {format: 'float32', length: 2},
    status: {format: 'uint32', length: 3}
  });
  harness.graph.add(
    new GPUVecchiaLogLikelihoodOperation(getScalarLikelihoodProps()).createContributor(
      harness.getView
    )
  );
  const results = await runPoisonedOracleGraph(harness, device, submission => {
    if (submission === 1) harness.buffers['parameters'].write(new Float32Array(8));
  });
  for (const [submission, result] of results.entries()) {
    if (submission === 1) {
      // Covariance scale zero is invalid: every solve fails with code 1 and every
      // likelihood site with code 2. A hard-coded success status must fail this.
      expect(result['conditionalStatus']).toEqual([1, 1, 1]);
      expect(result['siteStatus']).toEqual([2, 2, 2]);
      expect(result['status']).toEqual([3, 0, 0]);
      expect(result['siteTerms'].every(Number.isNaN)).toBe(true);
      continue;
    }
    expectOracleFloats(result['conditionals'], VECCHIA_ORACLE_CONDITIONALS);
    expectOracleFloats(result['siteTerms'], VECCHIA_ORACLE_TERMS);
    expect(result['conditionalStatus']).toEqual([0, 0, 0]);
    expect(result['siteStatus']).toEqual([0, 0, 0]);
    expect(result['status']).toEqual([0, 0, 3]);
    expectOracleFloats(
      [result['logLikelihood'][0] + result['logLikelihood'][1]],
      [-8.203302855575092]
    );
  }
});

it('GPU failed marginal stays failed when caller replaces the captured conditional-status destination', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createMarginalLikelihoodInput(false, 'position');
  const inputs = Object.fromEntries(
    Object.entries(getLikelihoodGraphInputs(input)).map(([name, descriptor]) => [
      name,
      {
        ...descriptor,
        values: descriptor.values as Float32Array | Uint32Array,
        length:
          descriptor.values.length / (name === 'positions' ? 2 : name === 'observations' ? 4 : 1)
      }
    ])
  );
  const harness = createPoisonedOracleGraph(
    device,
    {...inputs, replacementStatus: {format: 'uint32', length: 1, values: Uint32Array.of(0)}},
    {
      conditionals: {format: 'float32x4', length: 1},
      conditionalStatus: {format: 'uint32', length: 1},
      siteTerms: {format: 'float32', length: 1},
      siteStatus: {format: 'uint32', length: 1},
      logLikelihood: {format: 'float32', length: 2},
      status: {format: 'uint32', length: 3}
    }
  );
  const view = harness.getView;
  const output = {
    conditionals: view<'float32x4'>('conditionals'),
    conditionalStatus: view<'uint32'>('conditionalStatus'),
    siteTerms: view<'float32'>('siteTerms'),
    siteStatus: view<'uint32'>('siteStatus'),
    logLikelihood: view<'float32'>('logLikelihood'),
    status: view<'uint32'>('status')
  };
  const contributor = new GPUVecchiaLogLikelihood({
    positions: view('positions'),
    observations: view('observations'),
    parameters: view('parameters'),
    neighborhoodOffsets: view('neighborhoodOffsets'),
    neighborhoods: view('neighborhoods'),
    neighborCount: 1,
    output
  });
  output.conditionalStatus = view('replacementStatus');
  harness.graph.add(contributor);
  for (const result of await runPoisonedOracleGraph(harness, device)) {
    expect(result['conditionalStatus']).toEqual([1]);
    expect(result['siteStatus']).toEqual([2]);
    expect(result['status']).toEqual([1, 0, 0]);
    expect(result['siteTerms'].every(Number.isNaN)).toBe(true);
    expect(result['logLikelihood'].every(Number.isNaN)).toBe(true);
  }
});

it('GPU conditionals publish independent nonunit pivot ratio for two predecessors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = Object.fromEntries(
    Object.entries(getLikelihoodGraphInputs(createTwoPredecessorLikelihoodInput())).map(
      ([name, input]) => [
        name,
        {
          ...input,
          values: input.values as Float32Array | Uint32Array,
          length: input.values.length / (name === 'positions' ? 2 : name === 'observations' ? 4 : 1)
        }
      ]
    )
  );
  const harness = createPoisonedOracleGraph(device, inputs, {
    conditionals: {format: 'float32x4', length: 3},
    conditionalStatus: {format: 'uint32', length: 3},
    siteTerms: {format: 'float32', length: 3},
    siteStatus: {format: 'uint32', length: 3},
    logLikelihood: {format: 'float32', length: 2},
    status: {format: 'uint32', length: 3}
  });
  harness.graph.add(
    new GPUVecchiaLogLikelihoodOperation({
      ...getScalarLikelihoodProps(),
      neighborCount: 2
    }).createContributor(harness.getView)
  );
  for (const result of await runPoisonedOracleGraph(harness, device)) {
    expectOracleFloats(result['conditionals'], VECCHIA_TWO_PREDECESSOR_CONDITIONALS);
    expect(result['conditionals'][11]).toBe(2);
    expect(result['conditionalStatus']).toEqual([0, 0, 0]);
  }
});

for (const spaceTime of [false, true]) {
  for (const invalid of spaceTime
    ? (['none', 'position', 'time'] as const)
    : (['none', 'position'] as const)) {
    it(`GPU marginal ${spaceTime ? 'space-time' : 'spatial'} ${invalid} propagates conditional status`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = createMarginalLikelihoodInput(spaceTime, invalid);
      const inputs = Object.fromEntries(
        Object.entries(getLikelihoodGraphInputs(input, spaceTime)).map(([name, descriptor]) => [
          name,
          {
            ...descriptor,
            values: descriptor.values as Float32Array | Uint32Array,
            length:
              descriptor.values.length /
              (name === 'positions' ? 2 : name === 'observations' ? 4 : 1)
          }
        ])
      );
      const harness = createPoisonedOracleGraph(device, inputs, {
        conditionals: {format: 'float32x4', length: 1},
        conditionalStatus: {format: 'uint32', length: 1},
        siteTerms: {format: 'float32', length: 1},
        siteStatus: {format: 'uint32', length: 1},
        logLikelihood: {format: 'float32', length: 2},
        status: {format: 'uint32', length: 3}
      });
      const operation = spaceTime
        ? new GPUVecchiaSpaceTimeLogLikelihoodOperation({
            ...getSpaceTimeLikelihoodProps(),
            rowCount: 1,
            neighborCount: 1
          })
        : new GPUVecchiaLogLikelihoodOperation({...getScalarLikelihoodProps(), rowCount: 1});
      harness.graph.add(operation.createContributor(harness.getView));
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        expect(result['conditionalStatus']).toEqual([invalid === 'none' ? 0 : 1]);
        expect(result['siteStatus']).toEqual([invalid === 'none' ? 0 : 2]);
        expect(result['status']).toEqual(invalid === 'none' ? [0, 0, 1] : [1, 0, 0]);
        if (invalid === 'none') {
          expectOracleFloats(result['conditionals'], [0, 2, 1, 0]);
          expectOracleFloats(result['siteTerms'], [-2.2655121234846454]);
          expectOracleFloats(
            [result['logLikelihood'][0] + result['logLikelihood'][1]],
            [-2.2655121234846454]
          );
        } else {
          expect(result['siteTerms'].every(Number.isNaN)).toBe(true);
          expect(result['logLikelihood'].every(Number.isNaN)).toBe(true);
        }
      }
    });
  }
}
