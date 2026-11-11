// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeMaxPRegionsOnCPU,
  CPUMaxPRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {GPU_MAX_P_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {getCSRFromAdjacency} from './azp-batched-mirror';
import {
  constructRegionalizationGPU,
  createMaxPInput,
  createMaxPGrowthInput,
  createMaxPCycleInput,
  createMaxPRefinementInput,
  createMaxPProps
} from './regionalization-operation-test-utils';

it('matches the unique feasible partition and hand-computed two-column objective', () => {
  const input = createMaxPInput();
  const before = input.values.slice();
  for (const maximumIterations of [0, 1]) {
    const result = computeMaxPRegionsOnCPU(input, {...createMaxPProps(), maximumIterations});
    expect(result.labels).toEqual(Uint32Array.of(0, 0, 1, 1));
    // Pair SSD = half the squared difference: (2²+2²)/2 + (4²+6²)/2 = 30.
    expect(result.objective).toEqual(Float32Array.of(30));
    expect(result.status).toEqual(Uint32Array.of(2, 1, 0, 0, 0, 0));
  }
  expect(input.values).toEqual(before);
});

it('uses population standardization and is invariant to column translations and scales', () => {
  const input = createMaxPInput();
  const result = computeMaxPRegionsOnCPU(input, {columnCount: 2, startCount: 4});
  // Original column population variances: 131/4 and 5. SSDs: 10 and 20.
  expect(result.objective[0]).toBeCloseTo(40 / 131 + 4, 6);
  const transformed = input.values.map((value, index) =>
    index % 2 ? value * 100 - 500 : value * 8 + 16
  );
  const translated = computeMaxPRegionsOnCPU(
    {...input, values: transformed},
    {columnCount: 2, startCount: 4}
  );
  expect(translated).toEqual(result);
});

it('grows several steps, assigns enclaves, and selects the first all-core start by Philox minima', () => {
  const input = createMaxPGrowthInput();
  const result = computeMaxPRegionsOnCPU(input, {
    columnCount: 2,
    standardize: false,
    startCount: 8
  });
  expect(result.labels).toEqual(Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1));
  // Each clique has three identical core values and one differing enclave: SSD = 3/4 * 100.
  expect(result.objective[0]).toBe(150);
  // In a clique the first eligible row is its global minimum key. A core seed grows only core
  // rows before completion, so its grown score is zero; an enclave seed has positive score.
  // Thus the first start with two core seeds is the unique tie-order oracle, without simulating
  // any growth, claims or enclave passes.
  const purpose = PHILOX_PURPOSES.maxPRegionsSeedOrder.id;
  const expectedStart = Array.from({length: 8}, (_, start) => start).find(start =>
    [0, 4].every(begin => {
      const rows = [begin, begin + 1, begin + 2, begin + 3];
      const seedRow = rows.reduce((best, row) =>
        getPhilox4x32([row, start, 0, 0], [3, purpose])[0] <
        getPhilox4x32([best, start, 0, 0], [3, purpose])[0]
          ? row
          : best
      );
      return seedRow !== begin + 3;
    })
  );
  expect(expectedStart).toBeDefined();
  expect(result.status).toEqual(Uint32Array.of(2, 1, 0, 0, expectedStart!, 0));
});

it('uses the seed for start selection and bijective equal-distance growth ties', () => {
  const purpose = PHILOX_PURPOSES.maxPRegionsSeedOrder.id;
  let nonzeroSelectionCount = 0;
  for (let seed = 0; seed < 8; seed++) {
    const input = {...createMaxPGrowthInput(), parameters: Uint32Array.of(seed)};
    const expectedStart = Array.from({length: 16}, (_, start) => start).find(start =>
      [0, 4].every(begin => {
        const rows = [begin, begin + 1, begin + 2, begin + 3];
        const seedRow = rows.reduce((best, row) =>
          getPhilox4x32([row, start, 0, 0], [seed, purpose])[0] <
          getPhilox4x32([best, start, 0, 0], [seed, purpose])[0]
            ? row
            : best
        );
        return seedRow !== begin + 3;
      })
    );
    expect(expectedStart).toBeDefined();
    const result = computeMaxPRegionsOnCPU(input, {columnCount: 2, standardize: false});
    expect(result.status[GPU_MAX_P_STATUS.selectedStart]).toBe(expectedStart);
    if (expectedStart! > 0) nonzeroSelectionCount++;

    // In a four-row clique with threshold two, the first seed must take the row with smallest
    // bijective tie. Only two rows remain, which necessarily form the second region. This is a
    // closed-form partition oracle, with no growth loop or claim implementation.
    const rows = [0, 1, 2, 3];
    const seedRow = rows.reduce((best, row) =>
      getPhilox4x32([row, 0, 0, 0], [seed, purpose])[0] <
      getPhilox4x32([best, 0, 0, 0], [seed, purpose])[0]
        ? row
        : best
    );
    const salt = getPhilox4x32([0xffffffff, 0, 0, 0], [seed, purpose])[1];
    const getTie = (row: number) => (Math.imul(row, 0x9e3779b1) ^ salt) >>> 0;
    const chosen = rows
      .filter(row => row !== seedRow)
      .reduce((best, row) => (getTie(row) < getTie(best) ? row : best));
    const remaining = rows.filter(row => row !== seedRow && row !== chosen);
    const otherSeed = remaining.reduce((best, row) =>
      getPhilox4x32([row, 0, 0, 0], [seed, purpose])[0] <
      getPhilox4x32([best, 0, 0, 0], [seed, purpose])[0]
        ? row
        : best
    );
    const expectedLabels = Uint32Array.from(rows, row =>
      row === seedRow || row === chosen ? Number(seedRow > otherSeed) : Number(otherSeed > seedRow)
    );
    const tied = computeMaxPRegionsOnCPU(
      {
        ...createMaxPInput(),
        offsets: Uint32Array.of(0, 0, 1, 3, 6),
        neighbors: Uint32Array.of(0, 0, 1, 0, 1, 2),
        weights: Float32Array.of(1, 2, 3, 4, 5, 6),
        values: new Float32Array(4),
        extensiveValues: new Float32Array(4).fill(1),
        threshold: Float32Array.of(2),
        parameters: Uint32Array.of(seed)
      },
      {standardize: false, startCount: 1}
    );
    expect(tied.labels).toEqual(expectedLabels);
    expect(tied.status).toEqual(Uint32Array.of(2, 1, 0, 0, 0, 0));
    expect(tied.objective[0]).toBe(0);
  }
  expect(nonzeroSelectionCount).toBeGreaterThan(0);
});

it('finishes long growth despite a one-round GPU budget and assigns enclaves', () => {
  const rowCount = 40;
  const adjacency = Array.from({length: rowCount}, (_, row) =>
    [row - 1, row + 1].filter(neighbor => neighbor >= 0 && neighbor < rowCount)
  );
  const topology = getCSRFromAdjacency(adjacency);
  const input = {
    ...createMaxPInput(),
    ...topology,
    weights: new Float32Array(topology.neighbors.length).fill(1),
    values: Float32Array.from({length: rowCount}, (_, row) => row),
    extensiveValues: new Float32Array(rowCount).fill(1),
    threshold: Float32Array.of(21)
  };
  const result = computeMaxPRegionsOnCPU(input, {
    standardize: false,
    startCount: 4,
    maximumGrowthSteps: 1
  });
  expect(result.status[GPU_MAX_P_STATUS.feasible]).toBe(1);
  expect(result.status[GPU_MAX_P_STATUS.regionCount]).toBe(1);
  expect([...new Set(result.labels)]).toEqual([0]);
  // Closed form SSD of 0..n-1: n(n²-1)/12, independent of growth or enclave order.
  expect(result.objective[0]).toBe((rowCount * (rowCount ** 2 - 1)) / 12);
});

it('reports infeasible components, zero threshold singletons and invalid input exactly', () => {
  const input = createMaxPInput();
  const impossible = computeMaxPRegionsOnCPU(
    {...input, threshold: Float32Array.of(5)},
    {columnCount: 2, standardize: false}
  );
  expect(impossible.labels).toEqual(Uint32Array.of(0, 0, 1, 1));
  expect(impossible.objective[0]).toBe(30);
  expect(impossible.status).toEqual(Uint32Array.of(2, 0, 2, 0, 0, 0));
  const singletons = computeMaxPRegionsOnCPU(
    {...input, threshold: Float32Array.of(0)},
    {columnCount: 2}
  );
  expect(singletons.labels).toEqual(Uint32Array.of(0, 1, 2, 3));
  expect(singletons.status).toEqual(Uint32Array.of(4, 1, 0, 0, 0, 0));
  expect(singletons.objective[0]).toBe(0);
  const invalid = computeMaxPRegionsOnCPU(
    {
      ...input,
      values: Float32Array.of(NaN, 1, 2, 3, 10, -1, 14, 5),
      extensiveValues: Float32Array.of(-1, 3, 2, 2),
      neighbors: Uint32Array.of(99, 2)
    },
    {columnCount: 2}
  );
  expect(invalid.labels).toEqual(Uint32Array.of(0, 1, 2, 3));
  expect(invalid.objective[0]).toBe(0);
  expect(invalid.status).toEqual(Uint32Array.of(0, 0, 4, 3, 0, 0));
});

it('accepts isolated rows, unused neighbor capacity and finite distances beyond float32', () => {
  const input = {
    ...createMaxPInput(),
    values: Float32Array.of(-1e30, 1e30),
    offsets: Uint32Array.of(0, 0, 1),
    neighbors: Uint32Array.of(0, 99),
    weights: Float32Array.of(1, 1),
    extensiveValues: Float32Array.of(1, 1),
    threshold: Float32Array.of(2)
  };
  const result = computeMaxPRegionsOnCPU(input, {standardize: false, startCount: 1});
  expect(result.labels).toEqual(Uint32Array.of(0, 0));
  expect(result.status).toEqual(Uint32Array.of(1, 1, 0, 0, 0, 0));
  expect(result.objective[0]).toBe(Infinity);
  const isolated = computeMaxPRegionsOnCPU(
    {
      ...input,
      offsets: Uint32Array.of(0, 0, 0),
      neighbors: new Uint32Array(0),
      weights: new Float32Array(0)
    },
    {standardize: false, startCount: 1}
  );
  expect(isolated.status).toEqual(Uint32Array.of(2, 0, 2, 0, 0, 0));
});

it('refines a construction with real constrained moves and reports successful rounds', () => {
  let improvedCount = 0;
  for (let seed = 0; seed < 8; seed++) {
    const input = createMaxPRefinementInput(seed);
    const options = {standardize: false, startCount: 1};
    const original = computeMaxPRegionsOnCPU(input, options);
    const refined = computeMaxPRegionsOnCPU(input, {...options, maximumIterations: 1});
    expect(refined.status[GPU_MAX_P_STATUS.regionCount]).toBe(
      original.status[GPU_MAX_P_STATUS.regionCount]
    );
    expect(refined.status[GPU_MAX_P_STATUS.feasible]).toBe(
      original.status[GPU_MAX_P_STATUS.feasible]
    );
    expect(refined.objective[0]).toBeLessThanOrEqual(original.objective[0]);
    let pairwiseObjective = 0;
    for (const region of new Set(refined.labels)) {
      const rows = Array.from({length: 9}, (_, row) => row).filter(
        row => refined.labels[row] === region
      );
      expect(rows.reduce((sum, row) => sum + input.extensiveValues[row], 0)).toBeGreaterThanOrEqual(
        3
      );
      for (const first of rows)
        for (const second of rows)
          pairwiseObjective +=
            (input.values[first] - input.values[second]) ** 2 / (2 * rows.length);
    }
    expect(refined.objective[0]).toBe(Math.fround(pairwiseObjective));
    if (refined.status[GPU_MAX_P_STATUS.improvementMoveCount] > 0) {
      improvedCount++;
      expect(refined.objective[0]).toBeLessThan(original.objective[0]);
    }
  }
  expect(improvedCount).toBeGreaterThan(0);
});

it('moves row 4 in seed-two constrained refinement with hand-derived before and after SSD', () => {
  const input = createMaxPRefinementInput(2);
  const options = {standardize: false, startCount: 1, maximumGrowthSteps: 64};
  const original = computeMaxPRegionsOnCPU(input, options);
  const refined = computeMaxPRegionsOnCPU(input, {...options, maximumIterations: 8});
  expect(original.labels).toEqual(Uint32Array.of(0, 1, 1, 0, 2, 2, 0, 2, 2));
  expect(original.status).toEqual(Uint32Array.of(3, 1, 0, 0, 0, 0));
  // Before: region values [0,12,1], [9,2], [5,15,10,4] have pairwise SSDs
  // 266/3, 49/2, 77; total 1141/6. Moving row 4 (value 5, extensive value 2)
  // leaves source extensive 1+2+1=4 >= 3. After: [0,12,1], [9,2,5], [15,10,4]
  // have pairwise SSDs 266/3, 74/3, 182/3; total 174, a gain of 97/6.
  expect(original.objective[0]).toBe(Math.fround(1141 / 6));
  expect(refined.labels).toEqual(Uint32Array.of(0, 1, 1, 0, 1, 2, 0, 2, 2));
  expect(refined.objective[0]).toBe(174);
  expect(refined.objective[0]).toBeLessThan(original.objective[0]);
  expect(refined.status).toEqual(Uint32Array.of(3, 1, 0, 0, 0, 1));
});

it('keeps Max-P refinement inside the same two-hop admissibility rule as AZP', () => {
  const input = createMaxPCycleInput();
  const options = {standardize: false, startCount: 1, maximumGrowthSteps: 64};
  for (const maximumIterations of [0, 8]) {
    const result = computeMaxPRegionsOnCPU(input, {...options, maximumIterations});
    // The zero-extensive row 0 cannot cross to the singleton: its source neighbors 1 and 5
    // cannot reconnect in {1,2,4,5}. The remaining five zeros and one 10 have SSD 500/6.
    expect(result.labels).toEqual(Uint32Array.of(0, 0, 0, 0, 0, 0, 1));
    expect(result.objective[0]).toBe(Math.fround(250 / 3));
    expect(result.status).toEqual(Uint32Array.of(2, 1, 0, 0, 0, 0));
  }
});

it('validates both operation forms at construction or GPU contributor lowering', () => {
  const props = createMaxPProps();
  const input = createMaxPInput();
  expect(new CPUMaxPRegionsOperation(props).contract.outputs).toEqual(
    new GPUMaxPRegionsOperation(props).contract.outputs
  );
  constructRegionalizationGPU(new GPUMaxPRegionsOperation(props), input);
  for (const change of [
    {columnCount: 0},
    {startCount: 0},
    {maximumGrowthSteps: 0},
    {maximumIterations: -1}
  ]) {
    expect(() => new CPUMaxPRegionsOperation({...props, ...change})).toThrow(/^max-p-test /);
    expect(() =>
      constructRegionalizationGPU(new GPUMaxPRegionsOperation({...props, ...change}), input)
    ).toThrow(/^max-p-test /);
  }
  expect(
    () =>
      new CPUMaxPRegionsOperation({
        ...props,
        columnCount: 9,
        startCount: 65,
        maximumGrowthSteps: 4097,
        maximumIterations: 257
      })
  ).not.toThrow();
  expect(() =>
    constructRegionalizationGPU(new GPUMaxPRegionsOperation({...props, columnCount: 9}), input)
  ).toThrow(/columnCount/);
  const badMetadata = {
    ...props,
    weights: {
      ...props.weights,
      information: {rowCount: 5, symmetric: true, transform: 'raw' as const}
    }
  };
  expect(() => new CPUMaxPRegionsOperation(badMetadata)).toThrow(/weights.information.rowCount/);
  expect(() =>
    constructRegionalizationGPU(new GPUMaxPRegionsOperation(badMetadata), input)
  ).toThrow(/weights.information.rowCount/);
  for (const Operation of [CPUMaxPRegionsOperation, GPUMaxPRegionsOperation]) {
    expect(() => new Operation({...props, output: {...props.output, labels: 'values'}})).toThrow(
      /^max-p-test /
    );
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^max-p-test /);
  }
});

it('names malformed arrays in errors', () => {
  const input = createMaxPInput();
  for (const [argument, change] of [
    ['offsets', {offsets: Uint32Array.of(0, 3, 2, 2, 2)}],
    ['weights', {weights: new Float32Array(1)}],
    ['distances', {distances: new Float32Array(1)}],
    ['values', {values: new Float32Array(7)}],
    ['extensiveValues', {extensiveValues: new Float32Array(3)}],
    ['threshold', {threshold: new Float32Array(0)}],
    ['parameters', {parameters: new Uint32Array(0)}]
  ] as const)
    expect(() => computeMaxPRegionsOnCPU({...input, ...change}, {columnCount: 2})).toThrow(
      argument
    );
});
