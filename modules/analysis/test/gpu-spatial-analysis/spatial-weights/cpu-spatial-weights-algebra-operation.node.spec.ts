// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {
  computeSpatialWeightsAlgebraOnCPU,
  CPUSpatialWeightsAlgebraOperation,
  type CPUSpatialWeightsAlgebraRows
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {
  GPUSpatialWeightsAlgebraOperation,
  type SpatialWeightsAlgebraOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';

const SOURCE: CPUSpatialWeightsAlgebraRows = {
  offsets: Uint32Array.of(0, 2, 4, 5, 6),
  neighbors: Uint32Array.of(1, 3, 0, 2, 3, 0),
  weights: Float32Array.of(2, 4, 3, 5, 6, 7),
  distances: Float32Array.of(10, 20, 30, 40, 50, 60)
};
const OUTPUT = {
  offsets: 'offsets',
  neighbors: 'neighbors',
  weights: 'weights',
  status: {overflow: 'overflow', requiredCount: 'requiredCount'}
};
const GRAPH_SOURCE = {
  offsets: 'sourceOffsets',
  neighbors: 'sourceNeighbors',
  weights: 'sourceWeights'
};

const REVIEW_ROWS: CPUSpatialWeightsAlgebraRows = {
  offsets: Uint32Array.of(0, 1, 2),
  neighbors: Uint32Array.of(1, 0),
  weights: Float32Array.of(1, 1),
  distances: Float32Array.of(10, 20)
};
const REVIEW_INFORMATION = {rowCount: 2, transform: 'raw', symmetric: true} as const;
const REVIEW_CONTEXT: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters'
};
const REVIEW_PROPS = {
  id: 'review-algebra',
  operation: 'union',
  rowCount: 2,
  capacity: 2,
  left: {
    ...GRAPH_SOURCE,
    distances: 'sourceDistances',
    information: REVIEW_INFORMATION,
    spatialContext: REVIEW_CONTEXT
  },
  right: {
    offsets: 'rightOffsets',
    neighbors: 'rightNeighbors',
    weights: 'rightWeights',
    distances: 'rightDistances',
    information: REVIEW_INFORMATION,
    spatialContext: REVIEW_CONTEXT
  },
  output: {...OUTPUT, distances: 'distances'}
} satisfies SpatialWeightsAlgebraOperationProps;

it('rejects incompatible declared contexts at CPU construction and GPU host lowering', () => {
  // Identical two-row inputs have a union containing precisely the same two entries.
  const expected = {...REVIEW_ROWS, overflow: Uint32Array.of(0), requiredCount: Uint32Array.of(2)};
  expect(
    computeSpatialWeightsAlgebraOnCPU(
      {left: REVIEW_ROWS, right: REVIEW_ROWS},
      {operation: 'union', capacity: 2, distances: true}
    )
  ).toEqual(expected);
  const device = createNullWebGPUDevice();
  try {
    for (const spatialContext of [
      REVIEW_CONTEXT,
      undefined,
      {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      } as GPUSpatialContext
    ]) {
      const props = {...REVIEW_PROPS, right: {...REVIEW_PROPS.right, spatialContext}};
      const graph = new GPUCommandGraph(device);
      const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
        createTransientView(
          graph,
          port,
          port.endsWith('Weights') ||
            port.endsWith('Distances') ||
            port === 'weights' ||
            port === 'distances'
            ? 'float32'
            : 'uint32',
          port.endsWith('Offsets') || port === 'offsets'
            ? 3
            : port === 'overflow' || port === 'requiredCount'
              ? 1
              : 2
        ) as GraphDataView<Format>;
      if (spatialContext?.coordinateSpace === 'longitude-latitude') {
        expect(() => new CPUSpatialWeightsAlgebraOperation(props)).toThrow(
          /^review-algebra incompatible spatial contexts/
        );
        expect(() =>
          new GPUSpatialWeightsAlgebraOperation(props).createContributor(getView)
        ).toThrow(/^review-algebra incompatible spatial contexts/);
      } else {
        expect(() => new CPUSpatialWeightsAlgebraOperation(props)).not.toThrow();
        expect(() =>
          new GPUSpatialWeightsAlgebraOperation(props).createContributor(getView)
        ).not.toThrow();
      }
    }
  } finally {
    device.destroy();
  }
});

it('validates every metadata field on both algebra operands before execution', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const operand of ['left', 'right'] as const)
      for (const [field, information] of [
        ['rowCount', {...REVIEW_INFORMATION, rowCount: 3}],
        ['transform', {...REVIEW_INFORMATION, transform: 'invalid' as 'raw'}],
        ['symmetric', {...REVIEW_INFORMATION, symmetric: 'invalid' as 'unknown'}]
      ] as const) {
        const props = {...REVIEW_PROPS, [operand]: {...REVIEW_PROPS[operand], information}};
        const expectedError = new RegExp(`^review-algebra ${operand}\\.information\\.${field}`);
        expect(() => new CPUSpatialWeightsAlgebraOperation(props)).toThrow(expectedError);
        const graph = new GPUCommandGraph(device);
        const getView = <Format extends GPUGraphVectorFormat>(
          port: string
        ): GraphDataView<Format> =>
          createTransientView(
            graph,
            port,
            port.endsWith('Weights') ||
              port.endsWith('Distances') ||
              port === 'weights' ||
              port === 'distances'
              ? 'float32'
              : 'uint32',
            port.endsWith('Offsets') || port === 'offsets'
              ? 3
              : port === 'overflow' || port === 'requiredCount'
                ? 1
                : 2
          ) as GraphDataView<Format>;
        expect(() =>
          new GPUSpatialWeightsAlgebraOperation(props).createContributor(getView)
        ).toThrow(expectedError);
      }
  } finally {
    device.destroy();
  }
});

it('checks unary metadata using spatial rows for space-time operands', () => {
  const base = {id: 'review-algebra', rowCount: 2, capacity: 2, output: OUTPUT};
  const modes: SpatialWeightsAlgebraOperationProps[] = [
    {...base, operation: 'higherOrder', weights: REVIEW_PROPS.left, order: 1},
    {...base, operation: 'selfWeight', weights: REVIEW_PROPS.left, selfWeight: 1},
    {...base, operation: 'subgraph', weights: REVIEW_PROPS.left, mask: 'mask'},
    {
      ...base,
      operation: 'spaceTime',
      weights: REVIEW_PROPS.left,
      rowCount: 4,
      periodCount: 2,
      layout: 'period-major',
      lags: [{lag: 0}]
    }
  ];
  const device = createNullWebGPUDevice();
  try {
    for (const props of modes) {
      if (!('weights' in props)) throw new Error('review case requires weights');
      expect(() => new CPUSpatialWeightsAlgebraOperation(props)).not.toThrow();
      const invalidProps = {
        ...props,
        weights: {...props.weights, information: {...REVIEW_INFORMATION, rowCount: 3}}
      };
      expect(() => new CPUSpatialWeightsAlgebraOperation(invalidProps)).toThrow(
        /^review-algebra weights\.information\.rowCount/
      );
      const graph = new GPUCommandGraph(device);
      const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
        createTransientView(
          graph,
          port,
          port.endsWith('Weights') || port.endsWith('Distances') || port === 'weights'
            ? 'float32'
            : 'uint32',
          port === 'offsets'
            ? props.rowCount + 1
            : port.endsWith('Offsets')
              ? 3
              : port === 'overflow' || port === 'requiredCount'
                ? 1
                : 2
        ) as GraphDataView<Format>;
      expect(() =>
        new GPUSpatialWeightsAlgebraOperation(invalidProps).createContributor(getView)
      ).toThrow(/^review-algebra weights\.information\.rowCount/);
    }
  } finally {
    device.destroy();
  }
});

it('truncates an exact-order generation prefix before sorting and computes the uncapped requirement', () => {
  const weights = {
    offsets: Uint32Array.of(0, 2, 3, 5, 5, 5, 5),
    neighbors: Uint32Array.of(1, 2, 5, 3, 4),
    weights: new Float32Array(5).fill(1)
  };
  const result = computeSpatialWeightsAlgebraOnCPU(
    {weights},
    {operation: 'higherOrder', order: 2, capacity: 1}
  );
  // Row 0 reaches 5 through frontier row 1 before reaching 3 and 4 through frontier row 2.
  expect(Array.from(result.neighbors)).toEqual([5]);
  expect(Array.from(result.offsets)).toEqual([0, 1, 1, 1, 1, 1, 1]);
  expect(result.requiredCount[0]).toBe(3);
  expect(result.overflow[0]).toBe(1);
  const props: SpatialWeightsAlgebraOperationProps = {
    rowCount: 6,
    capacity: 1,
    operation: 'higherOrder',
    weights: GRAPH_SOURCE,
    order: 2,
    workCapacity: 1,
    output: OUTPUT
  };
  const operation = new CPUSpatialWeightsAlgebraOperation(props);
  const vector = (format: 'uint32' | 'float32', values: Uint32Array | Float32Array) => ({
    format,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const outputs = operation.execute({
    inputs: {
      sourceOffsets: vector('uint32', weights.offsets),
      sourceNeighbors: vector('uint32', weights.neighbors),
      sourceWeights: vector('float32', weights.weights)
    }
  });
  expect(outputs).toEqual({
    offsets: result.offsets,
    neighbors: result.neighbors,
    weights: result.weights,
    overflow: result.overflow,
    requiredCount: result.requiredCount
  });
});

it('higher orders agree with all-pairs Floyd-Warshall shortest paths, including cycles and cumulative sets', () => {
  const rowCount = SOURCE.offsets.length - 1;
  const distances = Array.from({length: rowCount}, (_, row) =>
    Array.from({length: rowCount}, (_, column) => (row === column ? 0 : Infinity))
  );
  for (let row = 0; row < rowCount; row++)
    for (let slot = SOURCE.offsets[row]; slot < SOURCE.offsets[row + 1]; slot++)
      distances[row][SOURCE.neighbors[slot]] = 1;
  for (let middle = 0; middle < rowCount; middle++)
    for (let row = 0; row < rowCount; row++)
      for (let column = 0; column < rowCount; column++)
        distances[row][column] = Math.min(
          distances[row][column],
          distances[row][middle] + distances[middle][column]
        );
  for (const order of [1, 2, 3, 4, 40])
    for (const cumulative of [false, true]) {
      const result = computeSpatialWeightsAlgebraOnCPU(
        {weights: SOURCE},
        {operation: 'higherOrder', order, cumulative, capacity: 20}
      );
      let count = 0;
      for (let row = 0; row < rowCount; row++) {
        const expected = distances[row].flatMap((distance, column) =>
          column !== row && (cumulative ? distance <= order : distance === order) ? [column] : []
        );
        expect(
          Array.from(result.neighbors.slice(result.offsets[row], result.offsets[row + 1]))
        ).toEqual(expected);
        count += expected.length;
      }
      expect(result.requiredCount[0]).toBe(count);
      expect(result.overflow[0]).toBe(0);
      expect(Array.from(result.weights.slice(0, count))).toEqual(new Array(count).fill(1));
    }
});

it('hand-computed weighted set operations preserve distance provenance and all combination rules', () => {
  const left = {
    offsets: Uint32Array.of(0, 2, 2),
    neighbors: Uint32Array.of(0, 1),
    weights: Float32Array.of(2, 3),
    distances: Float32Array.of(4, 5)
  };
  const right = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(7, 11),
    distances: Float32Array.of(8, 9)
  };
  for (const [weightRule, combined] of [
    ['left', 3],
    ['right', 7],
    ['sum', 10],
    ['min', 3],
    ['max', 7],
    ['product', 21],
    ['binary', 1]
  ] as const) {
    for (const [
      operation,
      expectedOffsets,
      expectedNeighbors,
      expectedWeights,
      expectedDistances
    ] of [
      ['union', [0, 2, 3], [0, 1, 0], [2, combined, 11], [4, 5, 9]],
      ['intersection', [0, 1, 1], [1], [combined], [5]],
      ['difference', [0, 1, 1], [0], [2], [4]],
      ['symmetricDifference', [0, 1, 2], [0, 0], [2, 11], [4, 9]]
    ] as const) {
      const result = computeSpatialWeightsAlgebraOnCPU(
        {left, right},
        {operation, weightRule, capacity: 4, distances: true}
      );
      const count = result.requiredCount[0];
      expect(Array.from(result.offsets)).toEqual(expectedOffsets);
      expect(Array.from(result.neighbors.slice(0, count))).toEqual(expectedNeighbors);
      expect(Array.from(result.weights.slice(0, count))).toEqual(
        weightRule === 'binary' ? new Array(count).fill(1) : expectedWeights
      );
      expect(Array.from(result.distances!.slice(0, count))).toEqual(expectedDistances);
    }
  }
});

it('self replacement, masked subgraphs, groups and overflow match hand-counted rows', () => {
  const replaced = computeSpatialWeightsAlgebraOnCPU(
    {
      weights: {
        ...SOURCE,
        offsets: Uint32Array.of(0, 2, 4, 5, 6),
        neighbors: Uint32Array.of(0, 3, 0, 2, 3, 0)
      },
      selfWeights: Float32Array.of(0, 0.5, 1, 1.5)
    },
    {operation: 'selfWeight', capacity: 12, distances: true}
  );
  expect(Array.from(replaced.offsets)).toEqual([0, 2, 5, 7, 9]);
  expect(Array.from(replaced.weights.slice(0, 9))).toEqual([0, 4, 3, 0.5, 5, 1, 6, 7, 1.5]);
  expect(Array.from(replaced.distances!.slice(0, 9))).toEqual([0, 20, 30, 0, 40, 0, 50, 60, 0]);
  const masked = computeSpatialWeightsAlgebraOnCPU(
    {weights: SOURCE, mask: Uint32Array.of(1, 0, 2, 1)},
    {operation: 'subgraph', capacity: 8, distances: true}
  );
  expect(Array.from(masked.offsets)).toEqual([0, 1, 1, 2, 3]);
  expect(Array.from(masked.neighbors.slice(0, 3))).toEqual([3, 3, 0]);
  expect(Array.from(masked.weights.slice(0, 3))).toEqual([4, 6, 7]);
  for (const capacity of [1, 3, 12]) {
    const grouped = computeSpatialWeightsAlgebraOnCPU(
      {groupIds: Uint32Array.of(1, 0, 1, 0, 0xffffffff)},
      {operation: 'block', groupCount: 2, capacity}
    );
    expect(Array.from(grouped.offsets)).toEqual(
      [0, 1, 2, 3, 4, 4].map(offset => Math.min(offset, capacity))
    );
    expect(Array.from(grouped.neighbors.slice(0, Math.min(4, capacity)))).toEqual(
      [2, 3, 0, 1].slice(0, capacity)
    );
    expect(grouped.requiredCount[0]).toBe(4);
    expect(grouped.overflow[0]).toBe(Number(capacity < 4));
  }
});

it('space-time weights equal the dense Kronecker product under both row permutations', () => {
  const weights = {
    offsets: Uint32Array.of(0, 2, 3),
    neighbors: Uint32Array.of(0, 1, 0),
    weights: Float32Array.of(9, 2, 3)
  };
  const lags = [
    {lag: -1, weight: 0.5, includeSelf: true},
    {lag: 0, weight: 2},
    {lag: 1, weight: 3, includeSelf: true}
  ];
  const spatial = [
    [9, 2],
    [3, 0]
  ];
  for (const layout of ['period-major', 'cell-major'] as const) {
    const result = computeSpatialWeightsAlgebraOnCPU(
      {weights},
      {operation: 'spaceTime', capacity: 40, periodCount: 3, layout, lags}
    );
    const locate = (row: number) =>
      layout === 'period-major' ? [row % 2, Math.floor(row / 2)] : [Math.floor(row / 3), row % 3];
    for (let row = 0; row < 6; row++)
      for (let column = 0; column < 6; column++) {
        const [cell, period] = locate(row);
        const [otherCell, otherPeriod] = locate(column);
        const lag = lags.find(entry => entry.lag === period - otherPeriod);
        const expected = lag
          ? lag.weight * (lag.includeSelf && cell === otherCell ? 1 : spatial[cell][otherCell])
          : 0;
        const slots = Array.from(
          result.neighbors.slice(result.offsets[row], result.offsets[row + 1])
        );
        const position = slots.indexOf(column);
        // A zero spatial weight absent from the CSR has no slot.
        expect(position < 0 ? 0 : result.weights[result.offsets[row] + position]).toBe(expected);
      }
  }
});

it('validates malformed arrays and both operation constructions, deferring GPU checks to lowering', () => {
  const props: SpatialWeightsAlgebraOperationProps = {
    id: 'algebra-test',
    rowCount: 4,
    capacity: 8,
    operation: 'higherOrder',
    weights: GRAPH_SOURCE,
    order: 2,
    output: OUTPUT
  };
  expect(new CPUSpatialWeightsAlgebraOperation(props).contract.outputs).toEqual(
    new GPUSpatialWeightsAlgebraOperation(props).contract.outputs
  );
  expect(() => new CPUSpatialWeightsAlgebraOperation({...props, order: 0})).toThrow(
    /^algebra-test order/
  );
  expect(() => new CPUSpatialWeightsAlgebraOperation({...props, rowCount: 0})).toThrow(
    /^algebra-test rowCount/
  );
  expect(() => new GPUSpatialWeightsAlgebraOperation({...props, capacity: 0})).toThrow(
    /^algebra-test outputs.neighbors.length/
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        port.endsWith('Weights') || port === 'weights' ? 'float32' : 'uint32',
        port.endsWith('Offsets') || port === 'offsets'
          ? 5
          : port === 'overflow' || port === 'requiredCount'
            ? 1
            : 8
      ) as GraphDataView<Format>;
    expect(() =>
      new GPUSpatialWeightsAlgebraOperation({...props, order: 0}).createContributor(getView)
    ).toThrow(/^algebra-test order/);
  } finally {
    device.destroy();
  }
  for (const [field, changed] of [
    ['offsets', {...SOURCE, offsets: Uint32Array.of(0, 3, 2, 5, 6)}],
    ['neighbors', {...SOURCE, neighbors: Uint32Array.of(3, 1, 0, 2, 3, 0)}],
    ['weights', {...SOURCE, weights: new Float32Array(1)}],
    ['distances', {...SOURCE, distances: new Float32Array(1)}]
  ] as const)
    expect(() =>
      computeSpatialWeightsAlgebraOnCPU(
        {weights: changed},
        {operation: 'higherOrder', order: 2, capacity: 8}
      )
    ).toThrow(`weights.${field}`);
  expect(() =>
    computeSpatialWeightsAlgebraOnCPU(
      {weights: SOURCE, mask: Uint32Array.of(1)},
      {operation: 'subgraph', capacity: 8}
    )
  ).toThrow('mask');
  expect(() =>
    computeSpatialWeightsAlgebraOnCPU(
      {weights: SOURCE, selfWeights: Float32Array.of(1)},
      {operation: 'selfWeight', capacity: 8}
    )
  ).toThrow('selfWeights');
});
