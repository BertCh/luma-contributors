// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMantelTestOnCPU,
  CPUMantelTestOperation
} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {
  GPUMantelTestOperation,
  type MantelTestOperationProps
} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {computeMantelOracle, computeSequentialStop} from './space-time-oracle';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: MantelTestOperationProps = {
  id: 'mantel-operation-test',
  rowCount: 3,
  pairs: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', distances: 'distances'},
  times: 'times',
  parameters: 'parameters',
  maximumPermutations: 31,
  output: {statistics: 'statistics', summary: 'summary'}
};
const INPUT = {
  offsets: Uint32Array.of(0, 2, 3, 3),
  neighbors: Uint32Array.of(1, 2, 2),
  weights: Float32Array.of(1, 5, 0),
  distances: Float32Array.of(1, 2, 4),
  times: Float32Array.of(0, 1, 3),
  parameters: getGPUSpaceTimeParameterValues({seed: 2 ** 40 + 123, permutationCount: 31})
};

it('G2 ignores requiredCount and reads non-enumerable loss words explicitly', () => {
  const expected = computeMantelTestOnCPU(INPUT, PROPS);
  const status = {
    overflow: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0),
    requiredCount: Uint32Array.of(3)
  };
  expect(computeMantelTestOnCPU({...INPUT, status}, PROPS)).toEqual(expected);
});

it('G2 recognizes a non-enumerable candidate loss getter', () => {
  const hiddenLoss = Object.defineProperty({}, 'candidateOverflow', {get: () => Uint32Array.of(1)});
  expect(
    computeMantelTestOnCPU({...INPUT, status: hiddenLoss}, PROPS).summary.every(Number.isNaN)
  ).toBe(true);
});

it('G2 captures a changing status-map getter once per CPU call', () => {
  let reads = 0;
  const input = Object.defineProperty({...INPUT}, 'status', {
    get() {
      reads++;
      return reads === 1 ? {candidateOverflow: Uint32Array.of(1)} : {};
    }
  });
  expect(computeMantelTestOnCPU(input, PROPS).summary.every(Number.isNaN)).toBe(true);
  expect(reads).toBe(1);
});

it('G3 Mantel operations keep loss ports and output names after caller mutation and recover', async () => {
  const props: MantelTestOperationProps = {
    ...PROPS,
    pairs: {...PROPS.pairs, status: {candidateOverflow: 'loss'}},
    output: {...PROPS.output}
  };
  const cpu = new CPUMantelTestOperation(props);
  const gpu = new GPUMantelTestOperation(props);
  const vector = <Values extends Float32Array | Uint32Array>(values: Values) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const loss = Uint32Array.of(1);
  const inputs = {
    ...Object.fromEntries(Object.entries(INPUT).map(([key, values]) => [key, vector(values)])),
    pairsCandidateOverflow: vector(loss)
  };
  expect((await cpu.execute({inputs}))['summary']).toEqual(new Float32Array(12).fill(NaN));
  delete props.pairs.status!.candidateOverflow;
  props.output.summary = 'replacement';
  expect((await cpu.execute({inputs}))['summary']).toEqual(new Float32Array(12).fill(NaN));
  const contributor = constructNetworkOperationContributor(gpu, {
    offsets: {format: 'uint32', length: 4},
    neighbors: {format: 'uint32', length: 3},
    weights: {format: 'float32', length: 3},
    distances: {format: 'float32', length: 3},
    times: {format: 'float32', length: 3},
    parameters: {format: 'uint32', length: 4},
    pairsCandidateOverflow: {format: 'uint32', length: 1}
  });
  expect(contributor).toHaveProperty('props.pairs.status.candidateOverflow');
  loss[0] = 0;
  expect((await cpu.execute({inputs}))['summary']).toEqual(
    computeMantelTestOnCPU(INPUT, PROPS).summary
  );
});

it.each([
  'overflow',
  'candidateOverflow'
] as const)('CPU Mantel suppresses %s and recovers with complete pairs', flag => {
  const loss = Uint32Array.of(1);
  const input = {...INPUT, status: {[flag]: loss}};
  const incomplete = computeMantelTestOnCPU(input, PROPS);
  expect(incomplete.summary.every(Number.isNaN)).toBe(true);
  expect(incomplete.statistics.every(Number.isNaN)).toBe(true);
  expect(Array.from(incomplete.permutationCount)).toEqual([0]);
  loss[0] = 0;
  expect(computeMantelTestOnCPU(input, PROPS)).toEqual(computeMantelTestOnCPU(INPUT, PROPS));
  const operation = new GPUMantelTestOperation({
    ...PROPS,
    pairs: {...PROPS.pairs, status: {[flag]: 'loss'}}
  });
  expect(Object.values(operation.inputs)).toContain('loss');
  const contributor = constructNetworkOperationContributor(operation, {
    offsets: {format: 'uint32', length: 4},
    neighbors: {format: 'uint32', length: 3},
    weights: {format: 'float32', length: 3},
    distances: {format: 'float32', length: 3},
    times: {format: 'float32', length: 3},
    parameters: {format: 'uint32', length: 4},
    [flag === 'overflow' ? 'pairsOverflow' : 'pairsCandidateOverflow']: {
      format: 'uint32',
      length: 1
    }
  }) as import('../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test').GPUMantelTest;
  expect(contributor.props.pairs.status?.[flag]?.length).toBe(1);
});

it('matches a closed-form correlation and the independent unscaled Pearson oracle', () => {
  const result = computeMantelTestOnCPU(INPUT, PROPS);
  // Distances [1,2,4], differences [1,3,2]: centered cross product 1,
  // square sums 14/3 and 2, hence r = sqrt(3/28).
  expect(result.statistics[0]).toBe(Math.fround(Math.sqrt(3 / 28)));
  const expected = computeMantelOracle(INPUT, INPUT.times, 2 ** 40 + 123, 31);
  result.statistics.forEach((value, slot) =>
    expect(Math.abs(value - expected[slot])).toBeLessThan(5e-8)
  );
  const observed = expected[0];
  const greater = expected.slice(1).filter(value => value >= observed - 1e-14).length;
  const lesser = expected.slice(1).filter(value => value <= observed + 1e-14).length;
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.pairCount]).toBe(3);
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.greaterCount]).toBe(greater);
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.lesserCount]).toBe(lesser);
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBe(
    Math.fround((greater + 1) / 32)
  );
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBeGreaterThan(0.2);
  expect(result.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBeLessThan(0.95);
});

it('keeps the lower-row first entry and asymmetric pairs, ignoring self/duplicate/invalid slots', () => {
  const dirty = {
    ...INPUT,
    offsets: Uint32Array.of(0, 4, 5, 7),
    neighbors: Uint32Array.of(0, 1, 1, 99, 0, 0, 1),
    weights: new Float32Array(7),
    distances: Float32Array.of(99, 1, 88, 77, 66, 2, 4)
  };
  expect(computeMantelTestOnCPU(dirty, PROPS)).toEqual(computeMantelTestOnCPU(INPUT, PROPS));
});

it('supports transforms, clamped parameter counts and sequential stopping', () => {
  const transform = {constant: 1, power: -1};
  const options = {...PROPS, distanceTransform: transform, timeTransform: transform};
  const expected = computeMantelOracle(INPUT, INPUT.times, 2 ** 40 + 123, 31, {
    distance: transform,
    time: transform
  });
  const result = computeMantelTestOnCPU(INPUT, options);
  result.statistics.forEach((value, slot) =>
    expect(Math.abs(value - expected[slot])).toBeLessThan(5e-8)
  );
  const stopped = computeMantelTestOnCPU(INPUT, {...options, stopAfterExceedances: 3});
  const reference = computeSequentialStop(expected, 31, 3);
  expect(stopped.permutationCount[0]).toBe(reference.permutationsUsed);
  expect(stopped.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBe(
    Math.fround(reference.pValue)
  );
  expect(
    stopped.statistics.subarray(reference.permutationsUsed + 1).every(value => value === 0)
  ).toBe(true);
  const clamped = computeMantelTestOnCPU(
    {...INPUT, parameters: new Uint32Array(4)},
    {...PROPS, maximumPermutations: 3}
  );
  expect(clamped.permutationCount[0]).toBe(1);
  expect([...clamped.statistics].slice(2)).toEqual([0, 0]);
});

it('defines constant and empty pair populations and zero-variance summaries', () => {
  const constant = computeMantelTestOnCPU(
    {...INPUT, times: new Float32Array(3)},
    {...PROPS, stopAfterExceedances: 2}
  );
  expect(constant.statistics.every(Number.isNaN)).toBe(true);
  expect(constant.permutationCount[0]).toBe(0);
  expect(constant.usablePermutationCount[0]).toBe(0);
  expect(constant.summary[GPU_SPACE_TIME_SUMMARY.pairCount]).toBe(3);
  expect(constant.summary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBeNaN();
  expect(Number.isNaN(constant.summary[GPU_SPACE_TIME_SUMMARY.zScore])).toBe(true);
  const empty = computeMantelTestOnCPU({...INPUT, offsets: new Uint32Array(4)}, PROPS);
  expect(empty.statistics.every(Number.isNaN)).toBe(true);
  expect(empty.summary[GPU_SPACE_TIME_SUMMARY.pairCount]).toBe(0);
  expect(empty.usablePermutationCount[0]).toBe(0);
  expect(empty.permutationCount[0]).toBe(0);
});

it('rejects malformed arrays and validates CPU and deferred GPU construction', () => {
  expect(() =>
    computeMantelTestOnCPU({...INPUT, offsets: Uint32Array.of(0, 3, 2, 3)}, PROPS)
  ).toThrow(/offsets/);
  for (const name of ['distances', 'weights'] as const)
    expect(() => computeMantelTestOnCPU({...INPUT, [name]: new Float32Array(2)}, PROPS)).toThrow(
      new RegExp(name)
    );
  expect(() => computeMantelTestOnCPU({...INPUT, parameters: new Uint32Array(2)}, PROPS)).toThrow(
    /parameters/
  );
  expect(() => new CPUMantelTestOperation({...PROPS, rowCount: 1})).toThrow(
    /mantel-operation-test.*rowCount/
  );
  expect(() => new CPUMantelTestOperation({...PROPS, stopAfterExceedances: 2})).toThrow(
    /mantel-operation-test.*permutationCount/
  );
  expect(() => new GPUMantelTestOperation({...PROPS, maximumPermutations: -1})).toThrow(
    /mantel-operation-test.*maximumPermutations/
  );
  const shapes = {
    offsets: {format: 'uint32' as const, length: 4},
    neighbors: {format: 'uint32' as const, length: 3},
    weights: {format: 'float32' as const, length: 3},
    distances: {format: 'float32' as const, length: 3},
    times: {format: 'float32' as const, length: 3},
    parameters: {format: 'uint32' as const, length: 4}
  };
  expect(() =>
    constructNetworkOperationContributor(
      new GPUMantelTestOperation({...PROPS, maximumPermutations: 0}),
      shapes
    )
  ).toThrow(/maximumPermutations/);
  expect(() =>
    constructNetworkOperationContributor(
      new GPUMantelTestOperation({...PROPS, stopAfterExceedances: 2}),
      shapes
    )
  ).toThrow(/permutationCount/);
});
