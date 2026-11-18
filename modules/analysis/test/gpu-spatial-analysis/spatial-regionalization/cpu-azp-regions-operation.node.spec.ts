// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeAZPRegionsOnCPU,
  CPUAZPRegionsOperation,
  type CPUAZPRegionsInput
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {GPU_AZP_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';
import {getCSRFromAdjacency} from './azp-batched-mirror';
import {
  constructRegionalizationGPU,
  createAZPCycleInput,
  createAZPExtensiveRoundingInput,
  createAZPInput,
  createAZPProps
} from './regionalization-operation-test-utils';

// Independent objective oracle: pairwise squared distances / region size equals region SSD.
function getPairwiseObjective(
  values: Float32Array,
  labels: Uint32Array,
  columnCount: number
): number {
  let objective = 0;
  for (let first = 0; first < labels.length; first++)
    for (let second = first + 1; second < labels.length; second++) {
      if (labels[first] !== labels[second]) continue;
      const size = labels.filter(label => label === labels[first]).length;
      for (let column = 0; column < columnCount; column++)
        objective +=
          (values[first * columnCount + column] - values[second * columnCount + column]) ** 2 /
          size;
    }
  return objective;
}

it('matches exhaustive path-cut minimization and applies two independent moves in one round', () => {
  const input = createAZPInput();
  const before = input.initialLabels.slice();
  const result = computeAZPRegionsOnCPU(input, {...createAZPProps(), maximumIterations: 1});
  expect(result.labels).toEqual(Uint32Array.of(0, 1, 1, 1, 2, 3, 3, 3));
  // Every contiguous two-region partition of a path is one of its three cuts.
  let optimum = Infinity;
  for (let firstCut = 1; firstCut < 4; firstCut++)
    for (let secondCut = 5; secondCut < 8; secondCut++) {
      const labels = Uint32Array.from({length: 8}, (_, row) =>
        row < 4 ? Number(row >= firstCut) : 2 + Number(row >= secondCut)
      );
      optimum = Math.min(optimum, getPairwiseObjective(input.values, labels, 2));
    }
  expect(result.objective[0]).toBe(Math.fround(optimum));
  expect(result.objective[0]).toBeCloseTo(1820 / 3, 4);
  expect(result.status).toEqual(Uint32Array.of(1, 0, 1, 0));
  expect(computeAZPRegionsOnCPU(input, {...createAZPProps(), maximumIterations: 2}).status).toEqual(
    Uint32Array.of(1, 1, 0, 0)
  );
  expect(input.initialLabels).toEqual(before);
});

it('honours nonuniform extensive values, inclusive thresholds and minimum region sizes', () => {
  const input = createAZPInput();
  const options = createAZPProps();
  const free = computeAZPRegionsOnCPU(input, options);
  expect(free.labels[1]).toBe(1); // Row zero alone has extensive value two: equality is allowed.
  const constrained = computeAZPRegionsOnCPU(
    {...input, extensiveThreshold: Float32Array.of(8)},
    options
  );
  expect(constrained.labels).toEqual(input.initialLabels);
  expect(constrained.objective[0]).toBe(getPairwiseObjective(input.values, input.initialLabels, 2));
  expect(constrained.status).toEqual(Uint32Array.of(0, 1, 0, 0));
  expect(computeAZPRegionsOnCPU(input, {...options, minimumRegionSize: 2})).toEqual(constrained);
  const changed = computeAZPRegionsOnCPU(
    {...input, extensiveValues: Float32Array.of(0, 7, 1, 3, 4, 2, 6, 1)},
    options
  );
  expect(changed.labels[1]).toBe(0); // The same total extensive value, allocated differently.
  expect(changed.labels[2]).toBe(0); // The other direction remains a valid improving move.
});

it('resolves a conflicting equal-gain pair by seeded row order', () => {
  const input: CPUAZPRegionsInput = {
    ...createAZPInput(),
    offsets: Uint32Array.of(0, 0, 1, 2, 3),
    neighbors: Uint32Array.of(0, 0, 1),
    weights: Float32Array.of(2, 3, 5),
    values: Float32Array.of(0, 10, 0, 10),
    initialLabels: Uint32Array.of(0, 0, 1, 2),
    extensiveValues: undefined,
    extensiveThreshold: undefined
  };
  // Either source row can join an equal-valued singleton with gain 50. They conflict on source
  // region zero, and after the winner leaves the source is a singleton and must stay.
  for (const seed of [0, 1]) {
    const result = computeAZPRegionsOnCPU(
      {...input, parameters: Uint32Array.of(seed)},
      {regionCapacity: 3, standardize: false}
    );
    expect(result.labels).toEqual(
      seed === 0 ? Uint32Array.of(0, 2, 1, 2) : Uint32Array.of(1, 0, 1, 2)
    );
    expect(result.objective[0]).toBe(0);
    expect(result.status).toEqual(Uint32Array.of(1, 1, 0, 0));
  }
});

it('rejects a removal whose reconnecting path leaves the two-hop neighborhood', () => {
  const input = createAZPCycleInput();
  const result = computeAZPRegionsOnCPU(input, {
    regionCapacity: 2,
    standardize: false,
    maximumIterations: 8
  });
  // The induced set {1,2,4,5} has two components. Row 3 reconnects them in the whole
  // region, but is outside the two-hop set, so row 0 must remain. Only row 0 can join region 1.
  expect(result.labels).toEqual(input.initialLabels);
  // One 10 and five zeros have pairwise SSD (5 * 100) / 6 = 250/3.
  expect(result.objective[0]).toBe(Math.fround(250 / 3));
  expect(result.status).toEqual(Uint32Array.of(0, 1, 0, 0));
});

it('retains a unit extensive remainder in float64 before the inclusive threshold test', () => {
  const input = createAZPExtensiveRoundingInput();
  const result = computeAZPRegionsOnCPU(input, {
    regionCapacity: 2,
    standardize: false,
    maximumIterations: 8
  });
  // 1 + 2^24 - 2^24 = 1 exactly in float64: row 1 can leave. Both final regions
  // contain only equal feature values, so SSD is zero and exactly one round makes a move.
  expect(result.labels).toEqual(Uint32Array.of(0, 1, 1));
  expect(result.objective).toEqual(Float32Array.of(0));
  expect(result.status).toEqual(Uint32Array.of(1, 1, 0, 0));
});

it('rejects an improving articulation move while allowing large neighborhoods without caps', () => {
  const topology = getCSRFromAdjacency([[1], [0, 2, 3], [1], [1]]);
  const input: CPUAZPRegionsInput = {
    ...createAZPInput(),
    ...topology,
    weights: new Float32Array(topology.neighbors.length).fill(1),
    values: Float32Array.of(0, 10, 0, 10),
    initialLabels: Uint32Array.of(0, 0, 0, 1),
    extensiveValues: undefined,
    extensiveThreshold: undefined
  };
  const blocked = computeAZPRegionsOnCPU(input, {regionCapacity: 2, standardize: false});
  expect(blocked.labels).toEqual(input.initialLabels);
  expect(blocked.objective[0]).toBe(Math.fround(200 / 3));
  const ringSize = 140;
  const adjacency: number[][] = Array.from({length: ringSize + 2}, () => []);
  const linkRows = (first: number, second: number) => {
    adjacency[first].push(second);
    adjacency[second].push(first);
  };
  for (let row = 1; row <= ringSize; row++) {
    linkRows(0, row);
    linkRows(row, row === ringSize ? 1 : row + 1);
  }
  linkRows(0, ringSize + 1);
  const largeTopology = getCSRFromAdjacency(adjacency);
  const large = computeAZPRegionsOnCPU(
    {
      ...input,
      ...largeTopology,
      weights: new Float32Array(largeTopology.neighbors.length).fill(1),
      values: Float32Array.from({length: ringSize + 2}, (_, row) =>
        row === 0 || row === ringSize + 1 ? 10 : 0
      ),
      initialLabels: Uint32Array.from({length: ringSize + 2}, (_, row) =>
        Number(row === ringSize + 1)
      )
    },
    {regionCapacity: 2, standardize: false}
  );
  expect(large.labels[0]).toBe(1);
  expect(large.objective[0]).toBe(0);
  expect(large.status).toEqual(Uint32Array.of(1, 1, 0, 0));
});

it('standardizes with the population divisor and keeps a constant column exactly zero', () => {
  const input = createAZPInput();
  // Each column's global population variance, computed independently from pairwise distances.
  const variances = [0, 0];
  for (let first = 0; first < 8; first++)
    for (let second = first + 1; second < 8; second++)
      for (let column = 0; column < 2; column++)
        variances[column] +=
          (input.values[first * 2 + column] - input.values[second * 2 + column]) ** 2 / 64;
  const result = computeAZPRegionsOnCPU(input, {...createAZPProps(), standardize: true});
  const columns = [0, 1].map(column =>
    Float32Array.from({length: 8}, (_, row) => input.values[row * 2 + column])
  );
  const expected = columns.reduce(
    (sum, values, column) =>
      sum + getPairwiseObjective(values, result.labels, 1) / variances[column],
    0
  );
  expect(result.objective[0]).toBe(Math.fround(expected));
  const constant = computeAZPRegionsOnCPU(
    {...input, values: new Float32Array(16).fill(1e30)},
    {...createAZPProps(), standardize: true}
  );
  expect(constant.labels).toEqual(input.initialLabels);
  expect(constant.objective[0]).toBe(0);
  expect(constant.status).toEqual(Uint32Array.of(0, 1, 0, 0));
});

it('preserves the explicit GPU gain threshold even under a large column translation', () => {
  const input = createAZPInput();
  const shifted = input.values.map(value => value + 1e6);
  const result = computeAZPRegionsOnCPU({...input, values: shifted}, createAZPProps());
  expect(result.labels).toEqual(input.initialLabels);
  expect(result.objective[0]).toBe(getPairwiseObjective(shifted, input.initialLabels, 2));
  expect(result.status[GPU_AZP_STATUS.converged]).toBe(1);
});

it('counts extra components, invalid labels, bad used slots and nonfinite entries', () => {
  const input = createAZPInput();
  const disconnected = {...input, initialLabels: Uint32Array.of(0, 1, 0, 1, 2, 3, 2, 3)};
  const result = computeAZPRegionsOnCPU(disconnected, createAZPProps());
  expect(result.labels).toEqual(disconnected.initialLabels);
  expect(result.objective[0]).toBe(0);
  expect(result.status).toEqual(Uint32Array.of(0, 0, 0, 4));
  const bad = {
    ...input,
    neighbors: input.neighbors.slice(),
    values: input.values.slice(),
    initialLabels: input.initialLabels.slice()
  };
  bad.neighbors[0] = 99;
  bad.values[0] = NaN;
  bad.initialLabels[0] = 99;
  const invalid = computeAZPRegionsOnCPU(bad, createAZPProps());
  expect(invalid.status).toEqual(Uint32Array.of(0, 0, 0, 3));
  expect(invalid.objective[0]).toBe(0);
  expect(invalid.labels).toEqual(bad.initialLabels);
});

it('validates CPU construction and GPU contributor construction during host lowering', () => {
  const props = createAZPProps();
  const input = createAZPInput();
  expect(new CPUAZPRegionsOperation(props).contract.outputs).toEqual(
    new GPUAZPRegionsOperation(props).contract.outputs
  );
  constructRegionalizationGPU(new GPUAZPRegionsOperation(props), input);
  for (const change of [
    {columnCount: 0},
    {regionCapacity: 1},
    {maximumIterations: 0},
    {minimumRegionSize: 0},
    {extensiveThreshold: undefined}
  ]) {
    expect(() => new CPUAZPRegionsOperation({...props, ...change})).toThrow(/^azp-test /);
    expect(() =>
      constructRegionalizationGPU(new GPUAZPRegionsOperation({...props, ...change}), input)
    ).toThrow(/^azp-test /);
  }
  expect(
    () => new CPUAZPRegionsOperation({...props, columnCount: 9, maximumIterations: 257})
  ).not.toThrow();
  expect(() =>
    constructRegionalizationGPU(
      new GPUAZPRegionsOperation({...props, maximumIterations: 257}),
      input
    )
  ).toThrow(/maximumIterations/);
  for (const Operation of [CPUAZPRegionsOperation, GPUAZPRegionsOperation]) {
    expect(() => new Operation({...props, output: {...props.output, labels: 'values'}})).toThrow(
      /^azp-test /
    );
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^azp-test /);
  }
  const badMetadata = {
    ...props,
    weights: {
      ...props.weights,
      information: {rowCount: 99, symmetric: true, transform: 'raw' as const}
    }
  };
  expect(() => new CPUAZPRegionsOperation(badMetadata)).toThrow(/weights.information.rowCount/);
  expect(() => constructRegionalizationGPU(new GPUAZPRegionsOperation(badMetadata), input)).toThrow(
    /weights.information.rowCount/
  );
});

it('names malformed arrays in errors', () => {
  const input = createAZPInput();
  for (const [argument, change] of [
    ['offsets', {offsets: Uint32Array.of(1, 0)}],
    ['weights', {weights: new Float32Array(1)}],
    ['distances', {distances: new Float32Array(1)}],
    ['values', {values: new Float32Array(1)}],
    ['initialLabels', {initialLabels: new Uint32Array(1)}],
    ['extensiveValues', {extensiveValues: new Float32Array(1)}],
    ['extensiveThreshold', {extensiveThreshold: new Float32Array(0)}],
    ['parameters', {parameters: new Uint32Array(0)}]
  ] as const)
    expect(() => computeAZPRegionsOnCPU({...input, ...change}, createAZPProps())).toThrow(argument);
});
