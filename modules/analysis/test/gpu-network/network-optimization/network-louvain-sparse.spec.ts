// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkLouvainOperation,
  type NetworkLouvainOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-louvain-operation';
import {CPUNetworkLouvainOperation} from '../../../src/gpu-network/network-optimization/cpu-network-louvain-operation';
import {
  runOperationGraph,
  type OperationGraphInput,
  type OperationGraphRows
} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/network-louvain-sparse.json';

const OUTPUTS = [
  'communities',
  'trace',
  'summary',
  'rounds',
  'converged',
  'limited',
  'invalid',
  'approximation'
];
const STATUS = {
  iterationCount: 'rounds',
  converged: 'converged',
  iterationLimitReached: 'limited',
  invalidCount: 'invalid',
  approximation: 'approximation'
};

// Float64 versus fixed-tree float32 modularity. Eight spacings at unit magnitude account
// for normalization and cancellation; label/round parity is tested only away from near ties.
const OBJECTIVE_TOLERANCE = 8 * 2 ** -23;

async function compareTwins(
  props: NetworkLouvainOperationProps,
  inputs: Record<string, OperationGraphInput>
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results: Record<string, Record<string, OperationGraphRows>> = {};
  for (const location of ['gpu', 'cpu']) {
    results[location] = await runOperationGraph(device, {
      id: `sparse-louvain-${location}`,
      inputs,
      operations: [
        location === 'gpu'
          ? new GPUNetworkLouvainOperation(props)
          : new CPUNetworkLouvainOperation(props)
      ],
      outputs: OUTPUTS
    });
  }
  for (const output of OUTPUTS.filter(name => name !== 'trace')) {
    expect(Array.from(results['gpu'][output]), output).toEqual(Array.from(results['cpu'][output]));
  }
  const expected = results['cpu']['trace'];
  const actual = results['gpu']['trace'];
  expect(actual.length).toBe(expected.length);
  for (let row = 0; row < expected.length; row++) {
    if (Number.isNaN(expected[row]))
      expect(Number.isNaN(actual[row]), `trace row ${row}`).toBe(true);
    else
      expect(Math.abs(actual[row] - expected[row]), `trace row ${row}`).toBeLessThanOrEqual(
        OBJECTIVE_TOLERANCE
      );
  }
  return results['gpu'];
}

for (const scene of fixture.cases) {
  it(`sparse Louvain twins match every output and NetworkX quality (${scene.name})`, async () => {
    const props: NetworkLouvainOperationProps = {
      id: 'sparse-louvain',
      nodeCount: scene.nodeCount,
      directed: scene.directed,
      maximumLevels: 4,
      maximumIterations: 16,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      parameters: 'parameters',
      output: {
        communities: 'communities',
        objectiveTrace: 'trace',
        summary: 'summary',
        status: STATUS
      }
    };
    const result = await compareTwins(props, {
      offsets: {format: 'uint32', values: Uint32Array.from(scene.offsets)},
      neighbors: {format: 'uint32', values: Uint32Array.from(scene.neighbors)},
      weights: {format: 'float32', values: Float32Array.from(scene.weights)},
      parameters: {format: 'float32', values: Float32Array.of(scene.resolution, 1e-5)}
    });
    expect(result['converged'][0]).toBe(1);
    expect(result['trace'][result['rounds'][0]]).toBeCloseTo(scene.modularity, 6);
    // Independently evaluate the final labels against the original adjacency (including loops,
    // duplicate arcs and masked/tail slots), without using either implementation's totals.
    let total = 0;
    let internal = 0;
    const outgoing = new Float64Array(scene.nodeCount);
    const incoming = new Float64Array(scene.nodeCount);
    for (let source = 0; source < scene.nodeCount; source++) {
      for (let slot = scene.offsets[source]; slot < scene.offsets[source + 1]; slot++) {
        const destination = scene.neighbors[slot];
        if (destination >= scene.nodeCount) continue;
        const weight = scene.weights[slot];
        total += weight;
        outgoing[result['communities'][source]] += weight;
        incoming[result['communities'][destination]] += weight;
        if (result['communities'][source] === result['communities'][destination])
          internal += weight;
      }
    }
    const expected =
      internal / total -
      outgoing.reduce(
        (penalty, strength, community) =>
          penalty + scene.resolution * (strength / total) * (incoming[community] / total),
        0
      );
    expect(Math.abs(result['trace'][result['rounds'][0]] - expected)).toBeLessThanOrEqual(
      OBJECTIVE_TOLERANCE
    );
  });
}

for (const scene of [
  'budget',
  'level-budget',
  'tie-seed',
  'empty',
  'zero-slots',
  'invalid-weight',
  'invalid-offsets',
  'asymmetric',
  'invalid-parameters',
  'high-tolerance',
  'missing-target'
] as const) {
  it(`sparse Louvain handles status, padding and stable labels (${scene})`, async () => {
    const props: NetworkLouvainOperationProps = {
      id: 'sparse-louvain',
      nodeCount: 3,
      directed: scene === 'missing-target',
      seed: scene === 'tie-seed' ? 1 : undefined,
      maximumLevels: scene === 'budget' || scene === 'level-budget' ? 1 : 2,
      maximumIterations: scene === 'budget' ? 1 : 4,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      parameters: 'parameters',
      output: {
        communities: 'communities',
        objectiveTrace: 'trace',
        summary: 'summary',
        status: STATUS
      }
    };
    const offsets =
      scene === 'empty' || scene === 'zero-slots'
        ? Uint32Array.of(0, 0, 0, 0)
        : scene === 'invalid-offsets'
          ? Uint32Array.of(0, 2, 1, 4)
          : Uint32Array.of(0, 1, 2, 2);
    const neighbors =
      scene === 'zero-slots'
        ? new Uint32Array(0)
        : Uint32Array.of(scene === 'missing-target' ? 0xffffffff : 1, 0, 0, 0xffffffff);
    const weights =
      scene === 'zero-slots'
        ? new Float32Array(0)
        : Float32Array.of(
            scene === 'invalid-weight' ? -2 : 2,
            scene === 'asymmetric' ? 1 : 2,
            NaN,
            -1
          );
    const result = await compareTwins(props, {
      offsets: {format: 'uint32', values: offsets},
      neighbors: {format: 'uint32', values: neighbors},
      weights: {format: 'float32', values: weights},
      parameters: {
        format: 'float32',
        values: Float32Array.of(
          scene === 'invalid-parameters' ? -1 : 1,
          scene === 'high-tolerance' ? 1 : 1e-5
        )
      }
    });
    const invalid = [
      'invalid-weight',
      'invalid-offsets',
      'asymmetric',
      'invalid-parameters'
    ].includes(scene);
    expect(result['invalid'][0]).toBe(Number(invalid));
    expect(result['limited'][0]).toBe(Number(scene === 'budget' || scene === 'level-budget'));
    if (scene === 'budget') expect(Array.from(result['communities'])).toEqual([1, 1, 2]);
    if (scene === 'tie-seed') expect(Array.from(result['communities'])).toEqual([0, 0, 2]);
  });
}

it('sparse moving supports a neighborhood larger than one workgroup without a degree table', async () => {
  const nodeCount = 257;
  const offsets = Uint32Array.from({length: nodeCount + 1}, (_, node) =>
    node === 0 ? 0 : 256 + node - 1
  );
  const neighbors = Uint32Array.from({length: 512}, (_, slot) => (slot < 256 ? slot + 1 : 0));
  const result = await compareTwins(
    {
      id: 'sparse-star',
      nodeCount,
      maximumLevels: 1,
      maximumIterations: 2,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      parameters: 'parameters',
      output: {
        communities: 'communities',
        objectiveTrace: 'trace',
        summary: 'summary',
        status: STATUS
      }
    },
    {
      offsets: {format: 'uint32', values: offsets},
      neighbors: {format: 'uint32', values: neighbors},
      weights: {format: 'float32', values: new Float32Array(512).fill(1)},
      parameters: {format: 'float32', values: Float32Array.of(1, 1e-5)}
    }
  );
  expect(result['invalid'][0]).toBe(0);
  expect(result['limited'][0]).toBe(1);
  expect(result['trace'][2]).toBeGreaterThan(result['trace'][0]);
});

for (const seed of [0, 1]) {
  it(`sparse acceptance commits two disjoint improving moves in one round (seed ${seed})`, async () => {
    // Two independent undirected edges: singleton Q=-1/4; paired Q=1/2. Each edge
    // proposes both directions, so reservations allow exactly one move per edge.
    const result = await compareTwins(
      {
        id: 'sparse-acceptance-regression',
        nodeCount: 4,
        seed,
        maximumLevels: 1,
        maximumIterations: 1,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        parameters: 'parameters',
        output: {
          communities: 'communities',
          objectiveTrace: 'trace',
          summary: 'summary',
          status: STATUS
        }
      },
      {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 4)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 3, 2)},
        weights: {format: 'float32', values: Float32Array.of(2, 2, 2, 2)},
        parameters: {format: 'float32', values: Float32Array.of(1, 0)}
      }
    );
    expect(Array.from(result['communities'])).toEqual(seed === 0 ? [1, 1, 3, 3] : [0, 0, 2, 2]);
    expect(Array.from(result['trace'])).toEqual([-0.25, 0.5]);
    expect(Array.from(result['summary'])).toEqual([1, 0, 1, 1]);
    expect(result['rounds'][0]).toBe(1);
    expect(result['converged'][0]).toBe(0);
    expect(result['limited'][0]).toBe(1);
    expect(result['invalid'][0]).toBe(0);
  });
}

it('rejects aliased partition, summary and status buffers on the two-edge GPU fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const offsetsBuffer = device.createBuffer({
    data: Uint32Array.of(0, 1, 2, 3, 4),
    usage: Buffer.STORAGE
  });
  const neighborsBuffer = device.createBuffer({
    data: Uint32Array.of(1, 0, 3, 2),
    usage: Buffer.STORAGE
  });
  const weightsBuffer = device.createBuffer({
    data: Float32Array.of(2, 2, 2, 2),
    usage: Buffer.STORAGE
  });
  const parametersBuffer = device.createBuffer({
    data: Float32Array.of(1, 0),
    usage: Buffer.STORAGE
  });
  const partitionBuffer = device.createBuffer({
    data: Uint32Array.of(1, 1, 3, 3),
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });
  const summaryBuffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  const traceBuffer = device.createBuffer({byteLength: 8, usage: Buffer.STORAGE});
  try {
    const communities = importGraphBuffer(graph, 'partition', partitionBuffer, 'uint32', 4);
    const summary = importGraphBuffer(graph, 'summary', summaryBuffer, 'uint32', 4);
    const objectiveTrace = importGraphBuffer(graph, 'trace', traceBuffer, 'float32', 2);
    const props = {
      id: 'alias-regression',
      nodeCount: 4,
      maximumLevels: 1,
      maximumIterations: 1,
      topology: {
        offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', 5),
        neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', 4),
        weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 4)
      },
      parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'float32', 2)
    };
    // The reviewer's exact alias must fail before publish-summary can replace [1,1,3,3]
    // with [1,0,1,1]. Distinct output buffers still construct normally.
    expect(
      () =>
        new GPUNetworkLouvain({
          ...props,
          output: {communities, objectiveTrace, summary: communities}
        })
    ).toThrow(/alias-regression.*output.*distinct buffers/);
    const status = graph.createDataView(communities.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    expect(
      () =>
        new GPUNetworkLouvain({
          ...props,
          output: {communities, objectiveTrace, summary, status: {converged: status}}
        })
    ).toThrow(/alias-regression.*output.*distinct buffers/);
    expect(
      () => new GPUNetworkLouvain({...props, output: {communities, objectiveTrace, summary}})
    ).not.toThrow();
    const bytes = await partitionBuffer.readAsync();
    expect(Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, 4))).toEqual([1, 1, 3, 3]);
  } finally {
    for (const buffer of [
      offsetsBuffer,
      neighborsBuffer,
      weightsBuffer,
      parametersBuffer,
      partitionBuffer,
      summaryBuffer,
      traceBuffer
    ])
      buffer.destroy();
  }
});
