// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {
  computeKShortestPathsOnCPU,
  decodeGPUKShortestPaths,
  getGPUKShortestPathsOutputLayout,
  GPUKShortestPathsOperation,
  CPUKShortestPathsOperation,
  GPUKShortestPaths,
  type GPUKShortestPathsReadback,
  type KShortestPathsOperationProps
} from '../../../src/gpu-network/k-shortest-paths/index';
import {setupKShortestPaths} from './k-shortest-paths-test-utils';

it('describes every constructor-accepted output and operation port with unequal dimensions', () => {
  const options = {queryCount: 2, routeCount: 3, maximumPathLength: 4};
  const {props} = setupKShortestPaths(options);
  const layout = getGPUKShortestPathsOutputLayout(options);
  expect(layout).toEqual({
    vertices: {format: 'uint32', length: 24},
    edges: {format: 'uint32', length: 24},
    lengths: {format: 'uint32', length: 6},
    costs: {format: 'float32', length: 6},
    counts: {format: 'uint32', length: 2},
    status: {
      converged: {format: 'uint32', length: 1},
      iterationLimitReached: {format: 'uint32', length: 1},
      candidateOverflow: {format: 'uint32', length: 1},
      iterationCount: {format: 'uint32', length: 1}
    }
  });
  const output = {
    vertices: 'path-vertices',
    edges: 'path-edges',
    lengths: 'path-lengths',
    costs: 'path-costs',
    counts: 'path-counts',
    status: {
      converged: 'batch-converged',
      iterationLimitReached: 'batch-limit',
      candidateOverflow: 'batch-overflow',
      iterationCount: 'batch-rounds'
    }
  };
  const operation = new GPUKShortestPathsOperation({
    ...options,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output
  });
  const views = {...props.output, ...props.output.status};
  const names = {...output, ...output.status};
  for (const [name, descriptor] of Object.entries({...layout, ...layout.status})) {
    if (name === 'status') continue;
    expect(views[name as keyof typeof views]).toMatchObject(descriptor);
    expect(operation.outputs[name]).toEqual({
      ...descriptor,
      value: names[name as keyof typeof names]
    });
  }
  expect(getGPUKShortestPathsOutputLayout({queryCount: 2, routeCount: 3}).vertices.length).toBe(
    384
  );
});

function createReadback(): GPUKShortestPathsReadback {
  // Each view starts two words into its buffer and includes three unused capacity words.
  const unsigned = (values: number[]) => {
    const storage = new Uint32Array(values.length + 7).fill(77);
    storage.set(values, 2);
    return storage.subarray(2, values.length + 5);
  };
  const floating = (values: number[]) => {
    const storage = new Float32Array(values.length + 7).fill(77);
    storage.set(values, 2);
    return storage.subarray(2, values.length + 5);
  };
  const missing = 0xffffffff;
  return {
    vertices: unsigned([
      4,
      5,
      6,
      missing,
      4,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      7,
      9,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing
    ]),
    edges: unsigned([
      8,
      9,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      12,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing,
      missing
    ]),
    lengths: unsigned([3, 1, 0, 2, 0, 0]),
    costs: floating([2.5, 0, Infinity, 4, Infinity, Infinity]),
    counts: unsigned([2, 1]),
    status: {
      converged: unsigned([0]),
      iterationLimitReached: unsigned([1]),
      candidateOverflow: unsigned([1]),
      iterationCount: unsigned([3])
    }
  };
}

it('decodes query-major routes at byte offsets, ignores spare capacity and preserves incomplete status', () => {
  const output = createReadback();
  const decoded = decodeGPUKShortestPaths(output, {
    queryCount: 2,
    routeCount: 3,
    maximumPathLength: 4
  });
  expect(decoded).toEqual({
    queries: [
      {
        count: 2,
        routes: [
          {length: 3, cost: 2.5, vertices: [4, 5, 6], edges: [8, 9]},
          {length: 1, cost: 0, vertices: [4], edges: []}
        ]
      },
      {count: 1, routes: [{length: 2, cost: 4, vertices: [7, 9], edges: [12]}]}
    ],
    status: {
      converged: false,
      iterationLimitReached: true,
      candidateOverflow: true,
      iterationCount: 3
    }
  });
  output.vertices[0] = 99;
  expect(decoded.queries[0].routes[0].vertices[0]).toBe(4);
});

it('rejects truncated layouts, impossible counts or lengths, wrong formats and NaN route costs', () => {
  const options = {queryCount: 2, routeCount: 3, maximumPathLength: 4};
  let output = createReadback();
  expect(() =>
    decodeGPUKShortestPaths({...output, edges: output.edges.subarray(0, 23)}, options)
  ).toThrow(/decodeGPUKShortestPaths output.edges.*24/);
  expect(() =>
    decodeGPUKShortestPaths(
      {...output, costs: new Uint32Array(6) as unknown as Float32Array},
      options
    )
  ).toThrow(/output.costs.*Float32Array/);
  expect(() =>
    decodeGPUKShortestPaths(
      {...output, status: {...output.status, converged: new Uint32Array(0)}},
      options
    )
  ).toThrow(/output.status.converged.*1/);
  output.counts[0] = 4;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/counts\[0\].*3/);
  output = createReadback();
  output.lengths[0] = 5;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/lengths\[0\].*4/);
  output.lengths[0] = 0;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/lengths\[0\].*1/);
  output = createReadback();
  output.costs[0] = NaN;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/costs\[0\].*NaN/);
  output = createReadback();
  output.status.converged[0] = 2;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/status.converged.*1/);
  output = createReadback();
  output.status.iterationCount[0] = 4;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/status.iterationCount.*3/);
});

it('uses the same layout for CPU results including a query with no accepted routes', () => {
  const options = {queryCount: 2, routeCount: 2, maximumPathLength: 3};
  const result = computeKShortestPathsOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 1, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(2)
      },
      sources: Uint32Array.of(0, 99),
      targets: Uint32Array.of(1, 1)
    },
    options
  );
  const decoded = decodeGPUKShortestPaths({...result, status: result}, options);
  expect(decoded.queries).toEqual([
    {count: 1, routes: [{length: 2, cost: 2, vertices: [0, 1], edges: [0]}]},
    {count: 0, routes: []}
  ]);
  expect(decoded.status.converged).toBe(true);
});

it('rejects invalid layout dimensions and uint32-overflowing output lengths', () => {
  for (const options of [
    {queryCount: -1, routeCount: 1},
    {queryCount: 1, routeCount: 1.5},
    {queryCount: 1, routeCount: 1, maximumPathLength: 0},
    {queryCount: 0xffffffff, routeCount: 2}
  ]) {
    expect(() => getGPUKShortestPathsOutputLayout(options)).toThrow(
      /getGPUKShortestPathsOutputLayout/
    );
  }
});

const SINGLE_EDGE_INPUT = {
  topology: {
    offsets: Uint32Array.of(0, 1, 1),
    neighbors: Uint32Array.of(1),
    weights: Float32Array.of(2)
  },
  sources: Uint32Array.of(0),
  targets: Uint32Array.of(1)
};

function getOperationProps(options: {
  queryCount: number;
  routeCount: number;
  maximumPathLength: number;
}): KShortestPathsOperationProps {
  return Object.assign(options, {
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    sources: 'sources',
    targets: 'targets',
    output: {
      vertices: 'vertices',
      edges: 'edges',
      lengths: 'lengths',
      costs: 'costs',
      counts: 'counts',
      status: {
        converged: 'converged',
        iterationLimitReached: 'iterationLimitReached',
        candidateOverflow: 'candidateOverflow',
        iterationCount: 'iterationCount'
      }
    }
  });
}

it('R1 decodes retained CPU paths whose finite float64 cost overflows float32', () => {
  const options = {queryCount: 1, routeCount: 1, maximumPathLength: 3};
  const result = computeKShortestPathsOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 1, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(2 ** 127, 2 ** 127)
      },
      sources: Uint32Array.of(0),
      targets: Uint32Array.of(2)
    },
    options
  );
  expect(Array.from(result.counts)).toEqual([1]);
  expect(result.converged[0]).toBe(1);
  expect(result.costs[0]).toBe(Infinity);
  expect(decodeGPUKShortestPaths({...result, status: result}, options).queries).toEqual([
    {count: 1, routes: [{length: 3, cost: Infinity, vertices: [0, 1, 2], edges: [0, 1]}]}
  ]);
  for (const invalidCost of [NaN, -1, -Infinity]) {
    result.costs[0] = invalidCost;
    expect(() => decodeGPUKShortestPaths({...result, status: result}, options)).toThrow(
      /costs\[0\]/
    );
  }
});

it('R2 keeps inherited path-length getters consistent across layout, both operations, lowering and CPU', () => {
  class Options {
    queryCount = 1;
    routeCount = 1;
    get maximumPathLength() {
      return 2;
    }
  }
  const options = new Options();
  expect(getGPUKShortestPathsOutputLayout(options).vertices.length).toBe(2);
  const result = computeKShortestPathsOnCPU(SINGLE_EDGE_INPUT, options);
  expect(result.vertices).toHaveLength(2);
  const operationProps = getOperationProps(options);
  const operation = new GPUKShortestPathsOperation(operationProps);
  expect(operation.outputs['vertices'].length).toBe(2);
  expect(new CPUKShortestPathsOperation(operationProps).contract.outputs).toEqual(
    operation.contract.outputs
  );
  const {props} = setupKShortestPaths(options);
  const {status, ...columns} = props.output;
  const views = {
    ...props.topology,
    sources: props.sources,
    targets: props.targets,
    ...columns,
    ...status
  };
  expect(() =>
    operation.createContributor(port => views[port as keyof typeof views] as never)
  ).not.toThrow();
  expect(
    decodeGPUKShortestPaths({...result, status: result}, options).queries[0].routes[0]
  ).toEqual({
    length: 2,
    cost: 2,
    vertices: [0, 1],
    edges: [0]
  });
});

it('R2 rejects a 64-slot readback for an inherited 65-slot layout before slicing', () => {
  class Options {
    queryCount = 1;
    routeCount = 1;
    get maximumPathLength() {
      return 65;
    }
  }
  const result = computeKShortestPathsOnCPU(SINGLE_EDGE_INPUT, {routeCount: 1});
  result.lengths[0] = 65;
  expect(() => decodeGPUKShortestPaths({...result, status: result}, new Options())).toThrow(
    /output.vertices.length.*65/
  );
});

it('R2 resolves a getter once before validating and declaring CPU operation dimensions', () => {
  let reads = 0;
  class Options {
    queryCount = 1;
    routeCount = 1;
    get maximumPathLength() {
      return ++reads === 1 ? 2 : 65;
    }
  }
  const operation = new CPUKShortestPathsOperation(getOperationProps(new Options()));
  expect(operation.outputs['vertices']).toMatchObject({length: 2});
  expect(reads).toBe(1);
});

it('R3 names the contributor ID and missing status view instead of dereferencing undefined', () => {
  const {props} = setupKShortestPaths();
  const {candidateOverflow, ...status} = props.output.status;
  expect(
    () =>
      new GPUKShortestPaths({
        ...props,
        id: 'review-ksp',
        output: {...props.output, status: status as typeof props.output.status}
      })
  ).toThrow(/review-ksp output.status.candidateOverflow/);
});

it('R3 names a missing status object in contributor errors', () => {
  const {props} = setupKShortestPaths();
  expect(
    () =>
      new GPUKShortestPaths({
        ...props,
        id: 'review-ksp',
        output: {...props.output, status: undefined as never}
      })
  ).toThrow(/review-ksp output.status/);
});

it('R3 names a missing status object in decoder errors', () => {
  const result = computeKShortestPathsOnCPU(SINGLE_EDGE_INPUT, {routeCount: 1});
  expect(() =>
    decodeGPUKShortestPaths(result as unknown as GPUKShortestPathsReadback, {
      id: 'review-ksp',
      queryCount: 1,
      routeCount: 1
    })
  ).toThrow(/review-ksp output.status/);
});

it('R4 accepts foreign uint32 views while rejecting foreign float32 vertices', () => {
  const options = {queryCount: 1, routeCount: 1, maximumPathLength: 1};
  const result = computeKShortestPathsOnCPU(
    {
      ...SINGLE_EDGE_INPUT,
      targets: Uint32Array.of(0)
    },
    options
  );
  const output = {...result, status: result};
  output.vertices = runInNewContext('Uint32Array.of(8)') as Uint32Array;
  expect(decodeGPUKShortestPaths(output, options).queries[0].routes[0].vertices).toEqual([8]);
  output.vertices = runInNewContext('Float32Array.of(8)') as Uint32Array;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/output.vertices.*Uint32Array/);
  const disguised = Float32Array.of(0.5);
  Object.defineProperty(disguised, Symbol.toStringTag, {value: 'Uint32Array'});
  output.vertices = disguised as unknown as Uint32Array;
  expect(() => decodeGPUKShortestPaths(output, options)).toThrow(/output.vertices.*Uint32Array/);
});

it('R5 rejects overflowing path-slot products even when route rows fit uint32', () => {
  expect(() =>
    getGPUKShortestPathsOutputLayout({
      queryCount: 1,
      routeCount: 0x40000000,
      maximumPathLength: 4
    })
  ).toThrow(/queryCount \* routeCount \* maximumPathLength/);
});

it('describes a zero-query batch with defined scalar status', () => {
  const layout = getGPUKShortestPathsOutputLayout({queryCount: 0, routeCount: 2});
  expect(layout.counts.length).toBe(0);
  expect(layout.vertices.length).toBe(0);
  expect(layout.status.converged.length).toBe(1);
});
