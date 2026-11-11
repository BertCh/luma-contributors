// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {
  computeSpatialWeightsTracesOnCPU,
  CPUSpatialWeightsTracesOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {
  GPUSpatialWeightsTracesOperation,
  type SpatialWeightsTracesOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import {
  getGPUSpatialWeightsTracesParameterValues,
  getSpatialWeightsTracesProbeValue
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  constructRegressionOperationContributor,
  expectRegressionRowsClose
} from './regression-operation-test-utils';

const WEIGHTS = {
  offsets: Uint32Array.of(0, 3, 4, 5),
  neighbors: Uint32Array.of(0, 1, 99, 2, 0),
  weights: Float32Array.of(1, 2, 999, 3, 4)
};
const PROPS: SpatialWeightsTracesOperationProps = {
  id: 'traces-test',
  rowCount: 3,
  maximumPower: 4,
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  output: {
    powerTraces: 'traces',
    powerTraceStandardErrors: 'errors',
    powerSums: 'sums',
    gramTraces: 'gram',
    infinityNorm: 'norm',
    status: {approximation: 'approximation'}
  }
};

it('keeps two one-row CSR chunks separate and matches the swap-matrix closed form', () => {
  const chunks = [
    {
      rowStart: 0,
      rowCount: 1,
      weights: {
        offsets: Uint32Array.of(0, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(1)
      }
    },
    {
      rowStart: 1,
      rowCount: 1,
      weights: {
        offsets: Uint32Array.of(0, 1),
        neighbors: Uint32Array.of(0),
        weights: Float32Array.of(1)
      }
    }
  ];
  const result = computeSpatialWeightsTracesOnCPU(
    {weights: {rowCount: 2, chunks}},
    {maximumPower: 2, mode: 'exact'}
  );
  // W swaps the two basis vectors, W^2=I, and both powers leave the ones vector fixed.
  expect(Array.from(result.powerTraces)).toEqual([0, 2]);
  expect(Array.from(result.powerSums)).toEqual([2, 2]);
  expect(Array.from(result.powerTraceStandardErrors)).toEqual([0, 0]);
  expect(result.infinityNorm[0]).toBe(1);
  expect(result.approximation[0]).toBe(0);
  const props: SpatialWeightsTracesOperationProps = {
    ...PROPS,
    rowCount: 2,
    maximumPower: 2,
    mode: 'exact',
    includeGramTraces: false,
    weights: {
      rowCount: 2,
      chunks: chunks.map((chunk, index) => ({
        rowStart: chunk.rowStart,
        rowCount: chunk.rowCount,
        weights: {
          offsets: `offsets${index}`,
          neighbors: `neighbors${index}`,
          weights: `weights${index}`
        }
      }))
    },
    output: {...PROPS.output, gramTraces: undefined}
  };
  const operation = new GPUSpatialWeightsTracesOperation(props);
  expect(operation.inputs).toMatchObject({
    chunk0Offsets: 'offsets0',
    chunk1Neighbors: 'neighbors1'
  });
  const contributor = constructRegressionOperationContributor(operation, {
    chunk0Offsets: {format: 'uint32', length: 2},
    chunk0Neighbors: {format: 'uint32', length: 1},
    chunk0Weights: {format: 'float32', length: 1},
    chunk1Offsets: {format: 'uint32', length: 2},
    chunk1Neighbors: {format: 'uint32', length: 1},
    chunk1Weights: {format: 'float32', length: 1}
  });
  expect(contributor).toBeInstanceOf(GPUSpatialWeightsTraces);
  expect('chunks' in (contributor as GPUSpatialWeightsTraces).props.weights).toBe(true);
  expect(new CPUSpatialWeightsTracesOperation(props).contract.outputs).toEqual(
    operation.contract.outputs
  );
  for (const invalid of [
    [chunks[0], {...chunks[1], rowStart: 0}],
    [chunks[0], {...chunks[1], weights: {...chunks[1].weights, offsets: Uint32Array.of(0, 2)}}],
    [chunks[0], {...chunks[1], weights: {...chunks[1].weights, weights: new Float32Array()}}]
  ])
    expect(() =>
      computeSpatialWeightsTracesOnCPU({weights: {rowCount: 2, chunks: invalid}}, {maximumPower: 2})
    ).toThrow(/weights/);
  expect(() =>
    computeSpatialWeightsTracesOnCPU({weights: {rowCount: 2, chunks: []}}, {maximumPower: 2})
  ).toThrow(/weights.chunks/);
  expect(
    () => new CPUSpatialWeightsTracesOperation({...props, weights: {rowCount: 2, chunks: []}})
  ).toThrow(/traces-test.*weights.chunks/);
});

it('matches exhaustive walks, with asymmetric weights, a loop and a dropped neighbor', () => {
  const dense = [
    [1, 2, 0],
    [0, 0, 3],
    [4, 0, 0]
  ];
  for (const mode of ['exact', 'stochastic'] as const) {
    const probeCount = 8;
    const result = computeSpatialWeightsTracesOnCPU(
      {weights: WEIGHTS, parameters: getGPUSpatialWeightsTracesParameterValues({seed: 11})},
      {maximumPower: 4, mode, probeCount, includeGramTraces: true}
    );
    for (let power = 1; power <= 4; power++) {
      let expectedTrace = 0;
      let expectedSum = 0;
      const estimates = new Array<number>(probeCount).fill(0);
      const walk = (start: number, current: number, length: number, weight: number) => {
        if (length === power) {
          expectedSum += weight;
          if (current === start) expectedTrace += weight;
          for (let probe = 0; probe < probeCount; probe++)
            estimates[probe] +=
              weight *
              getSpatialWeightsTracesProbeValue(11, probe, start) *
              getSpatialWeightsTracesProbeValue(11, probe, current);
          return;
        }
        for (let neighbor = 0; neighbor < 3; neighbor++)
          if (dense[current][neighbor] !== 0)
            walk(start, neighbor, length + 1, (weight * dense[current][neighbor]) / 4);
      };
      for (let start = 0; start < 3; start++) walk(start, start, 0, 1);
      const mean = estimates.reduce((sum, value) => sum + value, 0) / probeCount;
      const standardError = Math.sqrt(
        estimates.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
          (probeCount - 1) /
          probeCount
      );
      expectRegressionRowsClose(
        result.powerTraces.subarray(power - 1, power),
        [mode === 'exact' ? expectedTrace : mean],
        1e-7
      );
      expectRegressionRowsClose(result.powerSums.subarray(power - 1, power), [expectedSum], 1e-7);
      expectRegressionRowsClose(
        result.powerTraceStandardErrors.subarray(power - 1, power),
        [mode === 'exact' ? 0 : standardError],
        1e-7
      );
    }
    expect(result.infinityNorm[0]).toBe(4);
    expect(result.approximation[0]).toBe(mode === 'stochastic' ? 1 : 0);
    // Dense quadratic forms, independently of sparse chains:
    // W'W W and (W'W)^2, applied directly to each Philox sign vector.
    const gramPairMatrix = [
      [17, 34, 6],
      [2, 4, 12],
      [36, 0, 0]
    ];
    const gramSquareMatrix = [
      [293, 42, 0],
      [42, 20, 0],
      [0, 0, 81]
    ];
    const quadraticMean = (matrix: number[][]) => {
      let sum = 0;
      for (let probe = 0; probe < probeCount; probe++) {
        for (let first = 0; first < 3; first++) {
          for (let second = 0; second < 3; second++) {
            sum +=
              matrix[first][second] *
              getSpatialWeightsTracesProbeValue(11, probe, first) *
              getSpatialWeightsTracesProbeValue(11, probe, second);
          }
        }
      }
      return sum / probeCount;
    };
    expectRegressionRowsClose(
      result.gramTraces!,
      [
        30,
        mode === 'exact' ? 21 : quadraticMean(gramPairMatrix),
        mode === 'exact' ? 394 : quadraticMean(gramSquareMatrix),
        386
      ],
      1e-7
    );
  }
  // W'W = [[17,2,0],[2,4,0],[0,0,9]]. Its squared trace is 394;
  // tr(W'W W)=17+4=21, and squared column-square sums are 17^2+4^2+9^2=386.
  const exact = computeSpatialWeightsTracesOnCPU(
    {weights: WEIGHTS},
    {maximumPower: 1, includeGramTraces: true}
  );
  expect(Array.from(exact.gramTraces!)).toEqual([30, 21, 394, 386]);
});

it('matches diagonal closed forms in stochastic mode, for every seed and probe count', () => {
  const weights = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(0, 1),
    weights: Float32Array.of(2, 3)
  };
  for (const seed of [0, 42, 0xffffffff]) {
    for (const probeCount of [1, 3, 17]) {
      const result = computeSpatialWeightsTracesOnCPU(
        {weights, parameters: getGPUSpatialWeightsTracesParameterValues({seed})},
        {maximumPower: 3, mode: 'stochastic', probeCount, includeGramTraces: true}
      );
      expectRegressionRowsClose(result.powerTraces, [1 + 2 / 3, 1 + 4 / 9, 1 + 8 / 27], 1e-7);
      expect(Array.from(result.gramTraces!)).toEqual([13, 35, 97, 97]);
      expect(
        result.powerTraceStandardErrors.every(
          probeCount === 1 ? Number.isNaN : value => value === 0
        )
      ).toBe(true);
    }
  }
});

it('accepts zero links and validates malformed CSR and packed seed arrays', () => {
  const result = computeSpatialWeightsTracesOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 0, 0),
        neighbors: new Uint32Array(),
        weights: new Float32Array()
      }
    },
    {maximumPower: 257, includeGramTraces: true}
  );
  expect(result.powerTraces.every(value => value === 0)).toBe(true);
  expect(result.infinityNorm[0]).toBe(1);
  for (const changes of [
    {offsets: Uint32Array.of(1, 2)},
    {offsets: Uint32Array.of(0, 3, 2)},
    {offsets: Uint32Array.of(0, 6)},
    {weights: new Float32Array(4)}
  ])
    expect(() =>
      computeSpatialWeightsTracesOnCPU({weights: {...WEIGHTS, ...changes}}, {maximumPower: 4})
    ).toThrow(/weights/);
  expect(() =>
    computeSpatialWeightsTracesOnCPU(
      {weights: WEIGHTS, parameters: new Uint32Array()},
      {maximumPower: 4}
    )
  ).toThrow(/parameters/);
});

it('checks both constructions and identical graph contracts', () => {
  const shapes = {
    offsets: {format: 'uint32' as const, length: 4},
    neighbors: {format: 'uint32' as const, length: 5},
    weights: {format: 'float32' as const, length: 5}
  };
  for (const changes of [
    {maximumPower: 0},
    {probeCount: 0},
    {tileRowCount: 0},
    {probeBatchSize: 0},
    {includeGramTraces: false}
  ]) {
    expect(() => new CPUSpatialWeightsTracesOperation({...PROPS, ...changes})).toThrow(
      /traces-test/
    );
    expect(() =>
      constructRegressionOperationContributor(
        new GPUSpatialWeightsTracesOperation({...PROPS, ...changes}),
        shapes
      )
    ).toThrow(/traces-test/);
  }
  expect(new CPUSpatialWeightsTracesOperation(PROPS).contract.outputs).toEqual(
    new GPUSpatialWeightsTracesOperation(PROPS).contract.outputs
  );
  for (const Operation of [CPUSpatialWeightsTracesOperation, GPUSpatialWeightsTracesOperation])
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, powerSums: 'offsets'}})
    ).toThrow(/traces-test/);
  expect(() =>
    constructRegressionOperationContributor(new GPUSpatialWeightsTracesOperation(PROPS), shapes)
  ).not.toThrow();
});
