// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeVecchiaLogLikelihoodOnCPU,
  CPUVecchiaLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation';
import {
  createScalarLikelihoodInput,
  getScalarLikelihoodProps,
  VECCHIA_ORACLE_CONDITIONALS,
  VECCHIA_ORACLE_TERMS,
  expectOracleFloats
} from './vecchia-output-oracles-fixture';
import {createMarginalLikelihoodInput} from './vecchia-output-oracles-fixture';
import {
  createTwoPredecessorLikelihoodInput,
  VECCHIA_TWO_PREDECESSOR_CONDITIONALS
} from './vecchia-output-oracles-fixture';
import {
  computeVecchiaSpaceTimeLogLikelihoodOnCPU,
  CPUVecchiaSpaceTimeLogLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation';
import {getSpaceTimeLikelihoodProps} from './likelihoods-operation-test-utils';
import {
  getLikelihoodGraphInputs,
  makeLikelihoodCPUInputs
} from './likelihoods-operation-test-utils';

it('spatial Vecchia retains one captured conditional-status view for solve and likelihood after caller mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const originalStatus = createTransientView(graph, 'original-status', 'uint32', 1);
    const replacement = createTransientView(graph, 'replacement-status', 'uint32', 1);
    const output = {
      conditionals: createTransientView(graph, 'conditionals', 'float32x4', 1),
      conditionalStatus: originalStatus,
      siteTerms: createTransientView(graph, 'terms', 'float32', 1),
      siteStatus: createTransientView(graph, 'site-status', 'uint32', 1),
      logLikelihood: createTransientView(graph, 'total', 'float32', 2),
      status: createTransientView(graph, 'status', 'uint32', 3)
    };
    const props = {
      positions: createTransientView(graph, 'positions', 'float32x2', 1),
      observations: createTransientView(graph, 'observations', 'float32x4', 1),
      neighborhoodOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      neighborhoods: createTransientView(graph, 'neighbors', 'uint32', 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      neighborCount: 1,
      output
    };
    const contributor = new GPUVecchiaLogLikelihood(props);
    output.conditionalStatus = replacement;
    props.output = {...output};
    const nodes = contributor.getCommandNodes(graph);
    const siteTerms = nodes.find(node => node.id.endsWith('-site-terms'))!;
    // The likelihood must read the same failed-condition flag the solve wrote,
    // irrespective of a replacement buffer's zero contents.
    expect(
      siteTerms.resources?.some(
        resource => 'buffer' in resource && resource.buffer === originalStatus
      )
    ).toBe(true);
    expect(
      nodes
        .flatMap(node => node.resources ?? [])
        .some(resource => 'buffer' in resource && resource.buffer === replacement)
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('scalar Gaussian oracle covers every conditional component and per-site likelihood', async () => {
  const input = createScalarLikelihoodInput();
  const results = [
    computeVecchiaLogLikelihoodOnCPU(input),
    await new CPUVecchiaLogLikelihoodOperation(getScalarLikelihoodProps()).execute({
      inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
    })
  ];
  for (const result of results) {
    expectOracleFloats(result['conditionals'] as Float32Array, VECCHIA_ORACLE_CONDITIONALS);
    expectOracleFloats(result['siteTerms'] as Float32Array, VECCHIA_ORACLE_TERMS);
    expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([0, 0, 0]);
    expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([0, 0, 0]);
  }
});

it('two-predecessor Gaussian system has nonunit independent pivot ratio and retained count two', async () => {
  const input = createTwoPredecessorLikelihoodInput();
  const results = [
    computeVecchiaLogLikelihoodOnCPU(input),
    await new CPUVecchiaLogLikelihoodOperation({
      ...getScalarLikelihoodProps(),
      neighborCount: 2
    }).execute({inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))})
  ];
  for (const result of results) {
    // Normalized pivots 1 and 1-exp(-2), so a constant-one ratio fails.
    expectOracleFloats(
      result['conditionals'] as Float32Array,
      VECCHIA_TWO_PREDECESSOR_CONDITIONALS
    );
    expect((result['conditionals'] as Float32Array)[11]).toBe(2);
  }
});

it('the independent pivot oracle rejects a constant-one diagnostic writer', () => {
  const constantOne = Float32Array.from(VECCHIA_TWO_PREDECESSOR_CONDITIONALS);
  constantOne[10] = 1;
  expect(() => expectOracleFloats(constantOne, VECCHIA_TWO_PREDECESSOR_CONDITIONALS)).toThrow();
});

for (const spaceTime of [false, true]) {
  it(`excluded marginal ${spaceTime ? 'space-time' : 'spatial'} site stays excluded before conditional failure`, async () => {
    const input = createMarginalLikelihoodInput(spaceTime, 'position', true);
    const operation = spaceTime
      ? new CPUVecchiaSpaceTimeLogLikelihoodOperation({
          ...getSpaceTimeLikelihoodProps(),
          rowCount: 1,
          neighborCount: 1
        })
      : new CPUVecchiaLogLikelihoodOperation({...getScalarLikelihoodProps(), rowCount: 1});
    const results = [
      spaceTime
        ? computeVecchiaSpaceTimeLogLikelihoodOnCPU(input)
        : computeVecchiaLogLikelihoodOnCPU(input),
      await operation.execute({
        inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input, spaceTime))
      })
    ];
    for (const result of results) {
      expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([1]);
      expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([1]);
      expect(Array.from(result['status'] as Uint32Array)).toEqual([0, 1, 0]);
      expect(Array.from(result['siteTerms'] as Float32Array).every(Number.isNaN)).toBe(true);
      expect(Array.from(result['logLikelihood'] as Float32Array).every(Number.isNaN)).toBe(true);
    }
  });
}

for (const spaceTime of [false, true]) {
  for (const invalid of spaceTime
    ? (['none', 'position', 'time'] as const)
    : (['none', 'position'] as const)) {
    it(`marginal ${spaceTime ? 'space-time' : 'spatial'} ${invalid} propagates conditional failure before counting a used site`, async () => {
      const input = createMarginalLikelihoodInput(spaceTime, invalid);
      const props = {
        ...(spaceTime ? getSpaceTimeLikelihoodProps() : getScalarLikelihoodProps()),
        rowCount: 1,
        neighborCount: 1
      };
      const results = [
        spaceTime
          ? computeVecchiaSpaceTimeLogLikelihoodOnCPU(input)
          : computeVecchiaLogLikelihoodOnCPU(input),
        await (spaceTime
          ? new CPUVecchiaSpaceTimeLogLikelihoodOperation({
              ...getSpaceTimeLikelihoodProps(),
              rowCount: 1,
              neighborCount: 1
            })
          : new CPUVecchiaLogLikelihoodOperation(props)
        ).execute({inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input, spaceTime))})
      ];
      for (const result of results) {
        expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([
          invalid === 'none' ? 0 : 1
        ]);
        expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([
          invalid === 'none' ? 0 : 2
        ]);
        expect(Array.from(result['status'] as Uint32Array)).toEqual(
          invalid === 'none' ? [0, 0, 1] : [1, 0, 0]
        );
        if (invalid === 'none') {
          // -0.5*(log(2*pi*2)+2²/2), with no conditioning system to solve.
          expectOracleFloats(result['siteTerms'] as Float32Array, [-2.2655121234846454]);
        } else {
          expect(Array.from(result['siteTerms'] as Float32Array).every(Number.isNaN)).toBe(true);
          expect(Array.from(result['logLikelihood'] as Float32Array).every(Number.isNaN)).toBe(
            true
          );
        }
      }
    });
  }
}

it('invalid covariance scale publishes nonzero conditional and site failure codes', async () => {
  const input = createScalarLikelihoodInput();
  input.parameters.fill(0);
  const result = await new CPUVecchiaLogLikelihoodOperation(getScalarLikelihoodProps()).execute({
    inputs: makeLikelihoodCPUInputs(getLikelihoodGraphInputs(input))
  });
  expect(Array.from(result['conditionalStatus'] as Uint32Array)).toEqual([1, 1, 1]);
  expect(Array.from(result['siteStatus'] as Uint32Array)).toEqual([2, 2, 2]);
  expect(Array.from(result['status'] as Uint32Array)).toEqual([3, 0, 0]);
  expect(Array.from(result['siteTerms'] as Float32Array).every(Number.isNaN)).toBe(true);
});
