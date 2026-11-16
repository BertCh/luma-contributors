// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkLouvain,
  getNetworkLouvainStatusNodes
} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
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
import {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {getSparseNetworkLouvainNodes} from '../../../src/gpu-network/network-optimization/network-louvain-sparse';
import {
  createNetworkLouvainQuotientFixture,
  createNetworkLouvainMixedDuplicateFixture,
  type NetworkLouvainScalabilityFixture
} from './network-louvain-scalability-fixtures';
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

for (const nodeCount of [16, 131072]) {
  it(`two-word sparse Louvain preserves planted pairs and deterministic output (${nodeCount} nodes)`, async () => {
    // Cross-half pairs exercise high source/destination words. At 131,072 nodes the old
    // source * nodeCount key wrapped every 32,768 sources and merged unrelated pairs.
    const halfCount = nodeCount / 2;
    const offsets = Uint32Array.from({length: nodeCount + 1}, (_, row) => row);
    const neighbors = Uint32Array.from({length: nodeCount}, (_, row) =>
      row < halfCount ? row + halfCount : row - halfCount
    );
    const weights = new Float32Array(nodeCount).fill(1);
    const props: NetworkLouvainOperationProps = {
      id: 'two-word-louvain',
      nodeCount,
      maximumLevels: 2,
      maximumIterations: 3,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      parameters: 'parameters',
      output: {
        communities: 'communities',
        objectiveTrace: 'trace',
        summary: 'summary',
        status: STATUS
      }
    };
    const inputs: Record<string, OperationGraphInput> = {
      offsets: {format: 'uint32', values: offsets},
      neighbors: {format: 'uint32', values: neighbors},
      weights: {format: 'float32', values: weights},
      parameters: {format: 'float32', values: Float32Array.of(1, 0)}
    };
    const result = await compareTwins(props, inputs);
    const expected = Uint32Array.from({length: nodeCount}, (_, row) =>
      row < halfCount ? row + halfCount : row
    );
    expect(result['communities']).toEqual(expected);
    expect(Array.from(result['summary'])).toEqual([1, 1, 2, 3]);
    expect(result['trace'][0]).toBeCloseTo(-1 / nodeCount, 7);
    expect(result['trace'][3]).toBeCloseTo(1 - 2 / nodeCount, 7);

    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const repeated = await runOperationGraph(device, {
      id: 'two-word-louvain-repeat',
      inputs,
      operations: [new GPUNetworkLouvainOperation(props)],
      outputs: OUTPUTS
    });
    for (const output of OUTPUTS) expect(repeated[output], output).toEqual(result[output]);

    if (nodeCount === 16) {
      const denseWeights = new Float32Array(nodeCount * nodeCount);
      for (let source = 0; source < nodeCount; source++) {
        denseWeights[source * nodeCount + neighbors[source]] = 1;
      }
      const {topology: unusedTopology, ...sharedProps} = props;
      const dense = await runOperationGraph(device, {
        id: 'two-word-louvain-dense-reference',
        inputs: {
          weights: {format: 'float32', values: denseWeights},
          parameters: inputs['parameters']
        },
        operations: [new GPUNetworkLouvainOperation({...sharedProps, weights: 'weights'})],
        outputs: OUTPUTS
      });
      expect(dense['communities']).toEqual(expected);
      expect(dense['trace'][dense['rounds'][0]]).toBeCloseTo(result['trace'][3], 7);
    }
  });
}

/** The old one-word pair representation and 32-bit radix sort, with the unchanged moving schedule. */
function createPackedReferenceOperation(
  props: NetworkLouvainOperationProps
): GPUContributorOperation {
  const metadata = new GPUNetworkLouvainOperation(props);
  return new GPUContributorOperation({
    id: `${metadata.id}-packed-reference`,
    inputs: metadata.inputs,
    outputs: metadata.outputs,
    createContributor: getView => {
      const contributor = new GPUNetworkLouvain({
        ...props,
        weights: undefined,
        topology: {
          offsets: getView<'uint32'>('offsets'),
          neighbors: getView<'uint32'>('neighbors'),
          weights: getView<'float32'>('weights')
        },
        parameters: getView<'float32'>('parameters'),
        output: {
          communities: getView<'uint32'>('communities'),
          objectiveTrace: getView<'float32'>('objectiveTrace'),
          summary: getView<'uint32'>('summary'),
          status: Object.fromEntries(
            Object.keys(STATUS).map(name => [name, getView<'uint32'>(name)])
          )
        }
      });
      const sparseProps = contributor.props;
      if (!sparseProps.topology) throw new Error('packed reference requires CSR');
      return {
        getCommandNodes: graph => [
          ...getSparseNetworkLouvainNodes(graph, contributor.id, sparseProps, 'packed-reference'),
          ...getNetworkLouvainStatusNodes(graph, contributor.id, sparseProps.output)
        ]
      };
    }
  });
}

function getScalabilityProps(
  fixture: NetworkLouvainScalabilityFixture,
  directed = false
): NetworkLouvainOperationProps {
  return {
    id: 'weighted-scalability-louvain',
    nodeCount: fixture.nodeCount,
    directed,
    maximumLevels: 3,
    maximumIterations: 4,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    parameters: 'parameters',
    output: {
      communities: 'communities',
      objectiveTrace: 'trace',
      summary: 'summary',
      status: STATUS
    }
  };
}

function getScalabilityInputs(
  fixture: NetworkLouvainScalabilityFixture,
  minimumGain = 1e-8
): Record<string, OperationGraphInput> {
  return {
    offsets: {format: 'uint32', values: fixture.offsets},
    neighbors: {format: 'uint32', values: fixture.neighbors},
    weights: {format: 'float32', values: fixture.weights},
    parameters: {format: 'float32', values: Float32Array.of(fixture.resolution, minimumGain)}
  };
}

for (const directed of [false, true]) {
  it(`matches packed baseline output bits with mixed duplicate weights at gain thresholds (directed=${directed})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createNetworkLouvainMixedDuplicateFixture();
    const props = getScalabilityProps(fixture, directed);
    // Strong-pair gain at singleton labels: 4/18 - 1/8 = 7/72. Test either side
    // of this boundary by eight f32 spacings, plus the rounded boundary itself.
    const center = new Uint32Array(Float32Array.of(7 / 72).buffer)[0];
    const observedMoves = new Set<boolean>();
    for (const spacingOffset of [-8, 0, 8]) {
      const minimumGain = new Float32Array(Uint32Array.of(center + spacingOffset).buffer)[0];
      const inputs = getScalabilityInputs(fixture, minimumGain);
      const baseline = await runOperationGraph(device, {
        id: 'packed-baseline-bits',
        inputs,
        operations: [createPackedReferenceOperation(props)],
        outputs: OUTPUTS
      });
      const staged = await runOperationGraph(device, {
        id: 'two-word-stage-bits',
        inputs,
        operations: [new GPUNetworkLouvainOperation(props)],
        outputs: OUTPUTS
      });
      for (const output of OUTPUTS) {
        const expected = baseline[output];
        const actual = staged[output];
        // Include the exact float32 trace payloads, signed zero and unused NaN words.
        expect(new Uint32Array(actual.buffer, actual.byteOffset, actual.length), output).toEqual(
          new Uint32Array(expected.buffer, expected.byteOffset, expected.length)
        );
      }
      expect(staged['invalid'][0]).toBe(0);
      observedMoves.add(staged['communities'][0] !== 0);
    }
    expect(observedMoves).toEqual(new Set([false, true]));
  });
}

for (const nodeCount of [65535, 65536, 65537]) {
  for (const directed of [false, true]) {
    it(`matches the CPU route across the sort transition with duplicates and loops (${nodeCount}, directed=${directed})`, async () => {
      const fixture = createNetworkLouvainQuotientFixture(nodeCount, {
        duplicateArcs: true,
        selfLoops: true,
        directed
      });
      const result = await compareTwins(
        getScalabilityProps(fixture, directed),
        getScalabilityInputs(fixture)
      );
      expect(result['invalid'][0]).toBe(0);
      expect(result['converged'][0]).toBe(1);
      const isolateCount = nodeCount % 4;
      for (let isolate = 0; isolate < isolateCount; isolate++) {
        expect(result['communities'][isolate]).toBe(isolate);
      }
      expect(result['communities'][nodeCount - 1]).toBeLessThan(nodeCount);
      expect(result['trace'][result['rounds'][0]]).toBeGreaterThan(result['trace'][0]);
    });
  }
}

it('moves weighted high-ID planted communities after quotient aggregation at 131,072 nodes', async () => {
  const fixture = createNetworkLouvainQuotientFixture(131072);
  const result = await compareTwins(getScalabilityProps(fixture), getScalabilityInputs(fixture));
  expect(Array.from(result['summary'])).toEqual([1, 1, 3, 5]);
  expect(result['trace'][0]).toBeCloseTo(-0.25, 6);
  // Rounds 1–2 stabilize strong pairs. Round 3 moves a quotient vertex and merges
  // two pairs in every component, improving modularity from -1/18 to zero.
  expect(result['trace'][2]).toBeCloseTo(-1 / 18, 6);
  expect(result['trace'][3]).toBeGreaterThan(result['trace'][2] + 0.01);
  expect(result['trace'][5]).toBeCloseTo(0, 6);
  const componentCount = fixture.nodeCount / 4;
  const expected = Uint32Array.from(
    {length: fixture.nodeCount},
    (_, source) => 3 * componentCount + (source % componentCount)
  );
  expect(result['communities']).toEqual(expected);
});
