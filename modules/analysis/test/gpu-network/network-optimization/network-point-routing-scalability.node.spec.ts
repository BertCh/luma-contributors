// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslScanner} from 'wgsl_reflect';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import type {CPUNetworkPointRoutingInput} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {GPUNetworkPointRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing-operation';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getCandidateTruncationNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {GPUNetworkSnapping} from '../../../src/gpu-network/network-accessibility/gpu-network-snapping';
import {getPointRoutingInput} from './network-point-routing-operation-test-utils';
import {
  NETWORK_POINT_ROUTING_SNAPPING_CANDIDATE_CAPACITY,
  setupPointRouting
} from './network-point-routing-scalability-test-utils';

it('builds global-memory CSR routing at 100k edges and bounds the real command count', () => {
  for (const edgeCount of [1, 257, 100000, 16777216]) {
    const {graph, contributor} = setupPointRouting({
      edgeCount,
      turnCount: 1,
      queryCount: 1,
      maximumRelaxations: 1
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => node.workload?.operation === 'GPUNetworkReachability')).toBe(true);
    const sources = getParsedKernelSources(nodes);
    for (const {id: name, source: source, constants} of sources.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source, constants), name).toBeLessThanOrEqual(8192);
    }
  }
});

it('keeps the named reference caps and validates scalable limits, aliasing and command budgets', () => {
  expect(() => setupPointRouting({mode: 'reference', edgeCount: 129})).toThrow(
    /network-point-routing costs rows.*128/
  );
  expect(() => setupPointRouting({mode: 'reference', edgeCount: 128, turnCount: 4097})).toThrow(
    /network-point-routing turns rows.*4096/
  );
  expect(() => setupPointRouting({maximumRelaxations: 1025})).toThrow(
    /network-point-routing maximumRelaxations.*1024/
  );
  expect(() => setupPointRouting({queryCount: 16, maximumRelaxations: 1024})).toThrow(
    /network-point-routing estimated command node count.*32768/
  );
  const {graph, contributor} = setupPointRouting({
    mode: 'reference',
    edgeCount: 128,
    maximumRelaxations: 65536
  });
  expect(contributor.getCommandNodes(graph)).toHaveLength(1);
  const {props} = setupPointRouting();
  expect(
    () =>
      new GPUNetworkPointRouting({
        ...props,
        output: {...props.output, status: props.costs as never}
      })
  ).toThrow(/network-point-routing/);
  expect(
    () =>
      new GPUNetworkPointRouting({
        ...props,
        output: {...props.output, predecessorEdges: props.queries as never}
      })
  ).toThrow(/network-point-routing/);
  const oversized = setupPointRouting({edgeCount: 16777217, turnCount: 0, queryCount: 1});
  expect(() => oversized.contributor.getCommandNodes(oversized.graph)).toThrow(
    /network-point-routing frontier queues.*limit/
  );
  const oversizedOutput = setupPointRouting({
    edgeCount: 1000000,
    queryCount: 34,
    turnCount: 0,
    maximumRelaxations: 0
  });
  expect(() => oversizedOutput.contributor.getCommandNodes(oversizedOutput.graph)).toThrow(
    /network-point-routing predecessorEdges.*limit/
  );
  const empty = setupPointRouting({queryCount: 0});
  expect(empty.contributor.estimatedCommandNodeCount).toBe(0);
  expect(empty.contributor.getCommandNodes(empty.graph)).toEqual([]);
});

/** Execute certification, both reductions, the final predecessor copy and publication on CPU. */
function interpretPublication(
  input: CPUNetworkPointRoutingInput,
  distances: Float32Array,
  predecessors: Uint32Array,
  maximumRelaxations = 4,
  converged = 1
) {
  const edgeCount = input.costs.length;
  const {graph, contributor} = setupPointRouting({
    edgeCount,
    turnCount: input.turns.length / 2,
    queryCount: 1,
    maximumRelaxations
  });
  const kernels = getGPUCommandNodeKernelSources(
    contributor
      .getCommandNodes(graph)
      .filter(node =>
        /query-0-(paths-|partial-costs|target-parent|copy-parents|publish)/.test(node.id)
      )
  );
  const sameEdge =
    input.queries[0] === input.queries[1] && input.fractions[1] >= input.fractions[0];
  const best = sameEdge
    ? (input.fractions[1] - input.fractions[0]) * input.costs[input.queries[0]]
    : Infinity;
  const state = Uint32Array.of(
    0,
    new Uint32Array(Float32Array.of(best).buffer)[0],
    0xffffffff,
    maximumRelaxations,
    0
  );
  const paths = [new Uint32Array(edgeCount * 2), new Uint32Array(edgeCount * 2)];
  const outputPredecessors = new Uint32Array(edgeCount);
  const outputCosts = new Float32Array(1);
  const status = new Uint32Array(1);
  let pathIndex = 0;
  for (const kernel of kernels) {
    let bindings;
    if (kernel.id.endsWith('-paths-initialize'))
      bindings = {0: input.queries, 1: state, 2: predecessors, 3: paths[pathIndex]};
    else if (kernel.id.includes('-paths-jump-')) {
      bindings = {0: paths[pathIndex], 1: paths[1 - pathIndex]};
      pathIndex = 1 - pathIndex;
    } else if (kernel.id.endsWith('-copy-parents'))
      bindings = {
        0: input.queries,
        1: state,
        2: predecessors,
        3: paths[pathIndex],
        4: outputPredecessors
      };
    else if (kernel.id.endsWith('-publish'))
      bindings = {0: state, 1: Uint32Array.of(converged), 2: outputCosts, 3: status};
    else
      bindings = {
        0: input.queries,
        1: input.fractions,
        2: input.turns,
        3: input.costs,
        4: distances,
        5: state,
        6: paths[pathIndex]
      };
    // These kernels have no workgroup communication. Interpret one invocation per workgroup
    // to keep dispatch simulation independent of interpreter workgroup-size handling.
    const workgroupSize = Number(kernel.source.match(/@workgroup_size\((\d+)\)/)?.[1]);
    const source = kernel.source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(`@workgroup_size(${workgroupSize})`, '@workgroup_size(1)')
      .replace(`workgroupIndex * ${workgroupSize}u`, 'workgroupIndex * 1u');
    const interpreter = new WgslExec(new WgslParser().parse(source));
    interpreter.dispatchWorkgroups(
      'main',
      [Number(kernel.constants['INVOCATION_COUNT']), 1, 1],
      {0: bindings},
      {constants: kernel.constants}
    );
  }
  return {
    cost: outputCosts[0],
    status: status[0],
    predecessors: Array.from(outputPredecessors),
    cursor: state[4]
  };
}

it('publishes same-edge return parents and preserves direct-route ties through the final copy', () => {
  const input = {
    turns: Uint32Array.of(0, 1, 1, 2, 2, 0),
    costs: Float32Array.of(8, 4, 6),
    queries: Uint32Array.of(0, 0),
    fractions: Float32Array.of(0.75, 0.25)
  };
  const distances = Float32Array.of(2, 6, 12);
  const predecessors = Uint32Array.of(0xffffffff, 0, 1);
  expect(interpretPublication(input, distances, predecessors)).toEqual({
    cost: 14,
    status: 0,
    predecessors: [2, 0, 1],
    cursor: 1
  });
  expect(
    interpretPublication(
      {...input, fractions: Float32Array.of(0.25, 0.75)},
      distances,
      predecessors
    )
  ).toEqual({cost: 4, status: 0, predecessors: [0xffffffff, 0, 1], cursor: 1});
});

it('publishes the cost-attaining partial predecessor when full-edge float32 costs tie', () => {
  const result = interpretPublication(
    {
      costs: Float32Array.of(2, 1, 67108864, 1),
      turns: Uint32Array.of(3, 1, 3, 0, 1, 2, 0, 2),
      queries: Uint32Array.of(3, 2),
      fractions: Float32Array.of(0, 0)
    },
    Float32Array.of(3, 2, 67108864, 1),
    Uint32Array.of(3, 3, 0, 0xffffffff)
  );
  expect(result).toEqual({cost: 2, status: 0, predecessors: [3, 3, 1, 0xffffffff], cursor: 1});
  // Reconstructed path 3 -> 1 -> 2 pays 1 + 1 + 0, rather than 1 + 2 + 0.
});

it('excludes a destination descendant from tied partial parents to prevent a zero-cost cycle', () => {
  const result = interpretPublication(
    {
      costs: Float32Array.of(0, 0, 0, 1),
      turns: Uint32Array.of(3, 2, 2, 0, 0, 2),
      queries: Uint32Array.of(3, 2),
      fractions: Float32Array.of(0, 0)
    },
    Float32Array.of(1, Infinity, 1, 1),
    Uint32Array.of(2, 0xffffffff, 3, 0xffffffff)
  );
  expect(result).toEqual({
    cost: 1,
    status: 0,
    predecessors: [2, 0xffffffff, 3, 0xffffffff],
    cursor: 1
  });
});

it('keeps a finite partial result reconstructable when the full destination is beyond the frontier budget', () => {
  const edgeCount = 66;
  const costs = new Float32Array(edgeCount).fill(1);
  const turns = new Uint32Array((edgeCount - 1) * 2);
  const distances = new Float32Array(edgeCount).fill(Infinity);
  const predecessors = new Uint32Array(edgeCount).fill(0xffffffff);
  for (let edge = 0; edge < edgeCount - 1; edge++) {
    turns.set([edge, edge + 1], edge * 2);
    distances[edge] = edge + 1;
    if (edge > 0) predecessors[edge] = edge - 1;
  }
  const input = {costs, turns, queries: Uint32Array.of(0, 65), fractions: Float32Array.of(0, 0)};
  const result = interpretPublication(input, distances, predecessors, 1, 0);
  expect(result.cost).toBe(65);
  expect(result.status).toBe(2);
  expect(result.predecessors[65]).toBe(64);
  for (let edge = 1; edge < edgeCount; edge++) expect(result.predecessors[edge]).toBe(edge - 1);
  // A label whose zero-cost predecessor phase did not reach the source cannot be a terminal parent.
  predecessors[64] = 0xffffffff;
  const unresolved = interpretPublication(input, distances, predecessors, 1, 0);
  expect(unresolved.cost).toBe(Infinity);
  expect(unresolved.status).toBe(2);
  expect(unresolved.predecessors[65]).toBe(0xffffffff);
});

it('reuses routing and reachability pipelines across 1, 10 and 100 queries', () => {
  const counts = [1, 10, 100].map(queryCount => {
    const {graph, contributor} = setupPointRouting({
      edgeCount: 100000,
      queryCount,
      maximumRelaxations: 1
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    return {
      all: getGPUCommandNodePipelineKeys(nodes).size,
      routing: getGPUCommandNodePipelineKeys(nodes, {
        idPattern:
          /network-point-routing-(reset|validate|scatter|query-\d+-(seed|paths|partial|target|copy|publish))/
      }).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
  expect(counts[2]).toEqual(counts[0]);
  expect(counts[0].routing).toBe(11);
});

it('checks all generated identifiers against the full W3C reserved-word list', () => {
  // WGSL §16.2 was checked against https://www.w3.org/TR/WGSL/#reserved-words.
  const reservedWords = WGSL_RESERVED_WORDS;
  for (const mode of ['scalable', 'reference'] as const) {
    const {graph, contributor} = setupPointRouting({
      mode,
      edgeCount: 66,
      turnCount: 65,
      maximumRelaxations: 1
    });
    for (const [name, source] of getParsedKernelSources(contributor.getCommandNodes(graph))) {
      const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? [];
      expect(
        identifiers.filter(identifier => reservedWords.has(identifier)),
        name
      ).toEqual([]);
    }
  }
});

it('assembles valid unsorted turns into CSR without losing duplicates or outgoing cost identity', () => {
  const {graph, contributor} = setupPointRouting({
    edgeCount: 4,
    turnCount: 5,
    queryCount: 1,
    maximumRelaxations: 1
  });
  const kernels = getGPUCommandNodeKernelSources(
    contributor
      .getCommandNodes(graph)
      .filter(node => /-(reset|validate-count|reset-cursors|scatter)$/.test(node.id))
  );
  const counts = new Uint32Array(5);
  const invalid = new Uint32Array(1);
  const state = new Uint32Array(5).fill(99);
  const neighbors = new Uint32Array(5);
  const weights = new Float32Array(5);
  const turns = Uint32Array.of(2, 3, 0, 2, 2, 1, 0, 2, 3, 99);
  const costs = Float32Array.of(2, 3, 5, 7);
  const offsets = new Uint32Array(5);
  for (const kernel of kernels) {
    let bindings;
    if (kernel.id.endsWith('-reset'))
      bindings = {0: counts, 1: invalid, 2: neighbors, 3: weights, 4: state};
    else if (kernel.id.endsWith('-validate-count'))
      bindings = {0: turns, 1: costs, 2: counts, 3: invalid};
    else if (kernel.id.endsWith('-reset-cursors')) {
      // The published scan is independently tested upstream. Supply its exclusive sum here.
      for (let index = 1; index < offsets.length; index++)
        offsets[index] = offsets[index - 1] + counts[index - 1];
      expect(Array.from(offsets)).toEqual([0, 2, 2, 4, 4]);
      expect(invalid[0]).toBe(1);
      expect(state[4]).toBe(0);
      bindings = {0: counts};
    } else bindings = {0: turns, 1: costs, 2: offsets, 3: counts, 4: neighbors, 5: weights};
    const interpreter = new WgslExec(
      new WgslParser().parse(kernel.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    );
    interpreter.dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: kernel.constants});
  }
  expect(Array.from(neighbors.slice(0, 4))).toEqual([2, 2, 3, 1]);
  expect(Array.from(weights.slice(0, 4))).toEqual([5, 5, 7, 3]);
});

it('rejects overlapping outputs for every pair, including distinct handles over one physical buffer', () => {
  for (const [first, second] of [
    ['costs', 'status'],
    ['costs', 'predecessorEdges'],
    ['status', 'predecessorEdges']
  ] as const) {
    for (const imported of [false, true]) {
      const {graph, props} = setupPointRouting({edgeCount: 4, queryCount: 2, turnCount: 0});
      const shared = createTransientView(graph, 'shared-outputs', 'uint32', 16);
      const physical = imported
        ? graph.device.createBuffer({byteLength: 64, usage: Buffer.STORAGE})
        : undefined;
      try {
        const output = {...props.output};
        for (const [index, name] of [first, second].entries()) {
          const template = props.output[name];
          const handle = physical
            ? graph.importBuffer(
                {id: `physical-output-${index}`, byteLength: 64, usage: Buffer.STORAGE},
                physical
              )
            : shared.buffer;
          const overlapping = graph.createDataView(handle, {
            format: template.format,
            length: template.length,
            byteOffset: 4
          });
          Object.assign(output, {[name]: overlapping});
        }
        expect(() => new GPUNetworkPointRouting({...props, output})).toThrow(
          /network-point-routing .*aligned binding byte ranges must not overlap/
        );
      } finally {
        physical?.destroy();
      }
    }
  }
});

it('checks aligned view ranges and every scratch category before declaring resources or nodes', () => {
  for (const scenario of [
    'input-prefix',
    'output-prefix',
    'frontier',
    'state',
    'control',
    'dispatch'
  ] as const) {
    const edgeCount =
      scenario === 'input-prefix'
        ? 66
        : scenario === 'output-prefix'
          ? 64
          : scenario === 'frontier'
            ? 65
            : 1;
    const {graph, props} = setupPointRouting({
      edgeCount,
      turnCount: 0,
      queryCount: scenario === 'output-prefix' ? 2 : 1,
      maximumRelaxations: scenario === 'control' ? 1024 : 1
    });
    let values = props;
    if (scenario === 'input-prefix') {
      const padded = createTransientView(graph, 'padded-input', 'float32', 129);
      values = {
        ...props,
        costs: graph.createDataView(padded.buffer, {
          format: 'float32',
          length: edgeCount,
          byteOffset: 252
        })
      };
    }
    if (scenario === 'output-prefix') {
      const padded = createTransientView(graph, 'padded-output', 'uint32', 129);
      values = {
        ...props,
        output: {
          ...props.output,
          predecessorEdges: graph.createDataView(padded.buffer, {
            format: 'uint32',
            length: 128,
            byteOffset: 4
          })
        }
      };
    }
    const contributor = new GPUNetworkPointRouting(values);
    const limit =
      scenario === 'state'
        ? 16
        : scenario === 'control'
          ? 1024
          : scenario === 'dispatch'
            ? 128
            : 512;
    Object.defineProperty(graph.device.limits, 'maxStorageBufferBindingSize', {value: limit});
    const spies = [
      vi.spyOn(graph, 'createTransientBuffer'),
      vi.spyOn(graph, 'createDataView'),
      vi.spyOn(graph, 'add'),
      vi.spyOn(graph, 'addComputePass'),
      vi.spyOn(graph, 'addCopyPass'),
      vi.spyOn(graph, 'addRenderPass')
    ];
    try {
      const expected = {
        'input-prefix': /edgeCosts needs 516 bytes/,
        'output-prefix': /predecessorEdges needs 516 bytes/,
        frontier: /frontier queues needs 520 bytes/,
        state: /query state needs 20 bytes/,
        control: /frontier control needs 8232 bytes/,
        dispatch: /frontier dispatch needs 512 bytes/
      }[scenario];
      expect(() => contributor.getCommandNodes(graph)).toThrow(expected);
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  }
});

it('operation lowering forwards reference mode and the 2048 relaxation budget into executed WGSL', () => {
  const input = getPointRoutingInput();
  const {graph, props} = setupPointRouting({
    edgeCount: 6,
    queryCount: 5,
    turnCount: input.turns.length / 2
  });
  const operation = new GPUNetworkPointRoutingOperation({
    edgeCount: 6,
    queryCount: 5,
    turns: 'turns',
    costs: 'edgeCosts',
    queries: 'queries',
    fractions: 'fractions',
    mode: 'reference',
    maximumRelaxations: 2048,
    output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
  });
  const views: Record<string, GraphDataView> = {
    turns: props.turns,
    edgeCosts: props.costs,
    queries: props.queries,
    fractions: props.fractions,
    ...props.output
  };
  const contributor = operation.createContributor(
    <Format extends GPUGraphVectorFormat>(port: string) => views[port] as GraphDataView<Format>
  );
  expect(contributor).toBeInstanceOf(GPUNetworkPointRouting);
  if (!(contributor instanceof GPUNetworkPointRouting))
    throw new Error('Expected point routing contributor');
  expect(contributor.mode).toBe('reference');
  expect(contributor.maximumRelaxations).toBe(2048);
  const memory = createCPUGraphMemory();
  memory.write(props.turns, input.turns);
  memory.write(props.costs, input.costs);
  memory.write(props.queries, input.queries);
  memory.write(props.fractions, input.fractions);
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readFloat32(props.output.costs))).toEqual([11, 12, 4, 19, Infinity]);
  expect(Array.from(memory.readUint32(props.output.status))).toEqual([0, 0, 0, 0, 0]);
  // Returning same-edge route and direct same-edge route have independent, hand-computed parents.
  expect(memory.readUint32(props.output.predecessorEdges)[6]).toBe(2);
  expect(memory.readUint32(props.output.predecessorEdges)[12]).toBe(0xffffffff);
});

it('accepts shared output allocations with disjoint aligned bindings and rejects overlapping prefixes', () => {
  const {graph, props} = setupPointRouting({edgeCount: 4, queryCount: 2, turnCount: 0});
  const backing = createTransientView(graph, 'packed-outputs', 'uint32', 136);
  const output = {
    costs: graph.createDataView(backing.buffer, {format: 'float32', length: 2, byteOffset: 0}),
    status: graph.createDataView(backing.buffer, {format: 'uint32', length: 2, byteOffset: 256}),
    predecessorEdges: graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 512
    })
  };
  expect(() => new GPUNetworkPointRouting({...props, output})).not.toThrow();
  const status = graph.createDataView(backing.buffer, {format: 'uint32', length: 2, byteOffset: 8});
  // Logical [0,8) and [8,16) writes do not overlap, but both are bound in publication.
  expect(() => new GPUNetworkPointRouting({...props, output: {...output, status}})).toThrow(
    /aligned binding byte ranges must not overlap/
  );
});

it('keeps shader and pipeline counts constant as the frontier round budget grows', () => {
  const counts = [1, 4, 16].map(maximumRelaxations => {
    const {graph, contributor} = setupPointRouting({
      edgeCount: 66,
      queryCount: 2,
      maximumRelaxations
    });
    const nodes = contributor.getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    return {
      shaders: new Set(kernels.map(kernel => kernel.source)).size,
      pipelines: getGPUCommandNodePipelineKeys(nodes).size
    };
  });
  expect(counts[1]).toEqual(counts[0]);
  expect(counts[2]).toEqual(counts[0]);
});

describe('indexed snapping candidate budget', () => {
  for (const {candidateCapacity, complete} of [
    {candidateCapacity: 2, complete: false},
    {candidateCapacity: NETWORK_POINT_ROUTING_SNAPPING_CANDIDATE_CAPACITY, complete: true}
  ]) {
    it(`indexed snapping applies capacity ${candidateCapacity} across all four query points`, () => {
      const {graph, props} = setupPointRouting({edgeCount: 2, queryCount: 2, turnCount: 1});
      const positions = createTransientView(graph, 'snap-points', 'float32x2', 4);
      const nodePositions = createTransientView(graph, 'snap-node-positions', 'float32x2', 3);
      const offsets = createTransientView(graph, 'snap-offsets', 'uint32', 4);
      const neighbors = createTransientView(graph, 'snap-neighbors', 'uint32', 2);
      const radius = createTransientView(graph, 'snap-radius', 'float32', 1);
      const snappedEdges = createTransientView(graph, 'snap-edges', 'uint32', 4);
      const snapFractions = createTransientView(graph, 'snap-fractions', 'float32', 4);
      const overflow = createTransientView(graph, 'snap-overflow', 'uint32', 1);
      const snapping = new GPUNetworkSnapping({
        positions,
        topology: {offsets, neighbors, weights: props.costs, nodePositions},
        candidateCapacity,
        maximumSnapDistance: radius,
        output: {snappedEdges, snapFractions, status: {overflow}}
      });
      const memory = createCPUGraphMemory();
      memory.write(positions, Float32Array.of(2, 0, 10, 0, 2, 0, 6, 0));
      memory.write(nodePositions, Float32Array.of(0, 0, 8, 0, 12, 0));
      memory.write(offsets, Uint32Array.of(0, 1, 2, 2));
      memory.write(neighbors, Uint32Array.of(1, 2));
      memory.write(props.costs, Float32Array.of(8, 4));
      memory.write(radius, Float32Array.of(1));
      // Four independent point/segment projections give one candidate per point: edges
      // [0,1,0,0], fractions [1/4,1/2,1/4,3/4]. Check the real global-capacity status
      // and projection kernels without attempting to interpret the cooperative BVH builder.
      recordWGSLKernelNodes(graph);
      const nodes = snapping.getCommandNodes(graph);
      for (const [name, source] of getParsedKernelSources(nodes)) {
        const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? [];
        expect(
          identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier)),
          name
        ).toEqual([]);
      }
      const findNode = (suffix: string) => {
        const node = nodes.find(value => value.id === `network-snapping-${suffix}`);
        if (!node) throw new Error(`Missing snapping node ${suffix}`);
        return node;
      };
      const finalize = findNode('join-finalize');
      const project = findNode('project');
      const getBinding = (nodeId: string, name: string) => {
        const binding = getWGSLKernelNodeRecord(graph, nodeId)?.bindings.find(
          value => value.name === name
        );
        if (!binding) throw new Error(`Missing ${nodeId} binding ${name}`);
        return binding.view;
      };
      memory.write(getBinding(finalize.id, 'state'), Uint32Array.of(4, 0, 0, 0));
      memory.write(getBinding(finalize.id, 'bvhOverflow'), Uint32Array.of(0));
      memory.write(getBinding(project.id, 'edgeSources'), Uint32Array.of(0, 1));
      memory.write(
        getBinding(project.id, 'joinRows'),
        complete ? Uint32Array.of(0, 1, 0, 0) : Uint32Array.of(0, 1, 0xffffffff, 0xffffffff)
      );
      const candidateCounts = createTransientView(graph, 'candidate-counts-oracle', 'uint32', 4);
      const candidateOffsets = createTransientView(graph, 'candidate-offsets-oracle', 'uint32', 4);
      const truncated = createTransientView(graph, 'candidate-truncated-oracle', 'uint32', 4);
      memory.write(candidateCounts, Uint32Array.of(1, 1, 1, 1));
      memory.write(candidateOffsets, Uint32Array.of(0, 1, 2, 3));
      const truncationNodes = getCandidateTruncationNodes(graph, {
        id: 'test-truncation',
        operation: 'PointRoutingTest',
        counts: candidateCounts,
        offsets: candidateOffsets,
        rowCount: 4,
        candidateCapacity,
        total: getBinding(finalize.id, 'state') as GraphDataView<'uint32'>,
        candidateTruncated: truncated
      });
      executeCommandNodesOnCPU([...truncationNodes, finalize, project], memory);
      expect(Array.from(memory.readUint32(truncated))).toEqual(
        complete ? [0, 0, 0, 0] : [0, 0, 1, 1]
      );
      expect(Array.from(memory.readUint32(overflow))).toEqual([complete ? 0 : 1]);
      expect(Array.from(memory.readUint32(snappedEdges))).toEqual(
        complete ? [0, 1, 0, 0] : [0, 1, 0xffffffff, 0xffffffff]
      );
      expect(Array.from(memory.readFloat32(snapFractions))).toEqual(
        complete ? [0.25, 0.5, 0.25, 0.75] : [0.25, 0.5, -1, -1]
      );
    });
  }
});

it('direct same-edge publication retains predecessors of other reachable edges', () => {
  const input = {
    costs: Float32Array.of(8, 4),
    turns: Uint32Array.of(0, 1),
    queries: Uint32Array.of(0, 0),
    fractions: Float32Array.of(0.25, 0.75)
  };
  // Source remainder is 6. The allowed turn reaches full edge 1 at cost 6 + 4 = 10.
  // The direct terminal cost is (3/4 - 1/4) * 8 = 4, without removing that explored edge.
  expect(
    interpretPublication(input, Float32Array.of(6, 10), Uint32Array.of(0xffffffff, 0))
  ).toEqual({cost: 4, status: 0, predecessors: [0xffffffff, 0], cursor: 1});
  const {graph, props} = setupPointRouting({
    edgeCount: 2,
    queryCount: 1,
    turnCount: 1,
    mode: 'reference'
  });
  const memory = createCPUGraphMemory();
  memory.write(props.costs, input.costs);
  memory.write(props.turns, input.turns);
  memory.write(props.queries, input.queries);
  memory.write(props.fractions, input.fractions);
  executeCommandNodesOnCPU(new GPUNetworkPointRouting(props).getCommandNodes(graph), memory);
  expect(Array.from(memory.readFloat32(props.output.costs))).toEqual([4]);
  expect(Array.from(memory.readUint32(props.output.status))).toEqual([0]);
  expect(Array.from(memory.readUint32(props.output.predecessorEdges))).toEqual([0xffffffff, 0]);
});

it('rejects non-finite and out-of-range interpolated shapes and budgets at construction', () => {
  const {props} = setupPointRouting({edgeCount: 2, queryCount: 1, turnCount: 1});
  for (const mode of ['scalable', 'reference'] as const) {
    for (const value of [NaN, Infinity, -Infinity, -1, 0.5, 2 ** 32]) {
      expect(() => new GPUNetworkPointRouting({...props, mode, maximumRelaxations: value})).toThrow(
        /network-point-routing maximumRelaxations/
      );
      for (const name of ['costs', 'turns', 'queries'] as const) {
        const view = new Proxy(props[name], {
          get(currentView, property, receiver) {
            return property === 'length' ? value : Reflect.get(currentView, property, receiver);
          }
        });
        expect(() => new GPUNetworkPointRouting({...props, mode, [name]: view})).toThrow(
          new RegExp(`network-point-routing ${name} rows`)
        );
      }
    }
  }
});

it('emits only representable numeric literals in routing WGSL at supported boundaries', () => {
  for (const options of [
    {mode: 'scalable' as const, edgeCount: 16777216, turnCount: 1, maximumRelaxations: 1},
    {mode: 'reference' as const, edgeCount: 128, turnCount: 4096, maximumRelaxations: 65536}
  ]) {
    const {graph, contributor} = setupPointRouting({...options, queryCount: 1});
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).filter(
      kernel =>
        kernel.operation === 'GPUNetworkPointRouting' ||
        kernel.operation === 'GPUNetworkReachability'
    );
    for (const kernel of kernels) {
      const tokens = new WgslScanner(stripWGSLComments(kernel.source)).scanTokens();
      for (const [index, token] of tokens.entries()) {
        const literal = token.lexeme;
        if (['int_literal', 'uint_literal'].includes(token.type.name)) {
          const value = BigInt(literal.replace(/[iu]$/, ''));
          const negative = tokens[index - 1]?.lexeme === '-';
          const maximum = literal.endsWith('u')
            ? 0xffffffffn
            : literal.endsWith('i')
              ? negative
                ? 2147483648n
                : 2147483647n
              : negative
                ? 9223372036854775808n
                : 9223372036854775807n;
          expect(value <= maximum, `${kernel.id}: ${literal}`).toBe(true);
        } else if (token.type.name === 'decimal_float_literal') {
          const value = Number(literal.replace(/[fh]$/, ''));
          expect(Number.isFinite(value), `${kernel.id}: ${literal}`).toBe(true);
          expect(Number.isFinite(Math.fround(value)), `${kernel.id}: ${literal}`).toBe(true);
        }
      }
      for (const [name, value] of Object.entries(kernel.constants)) {
        expect(
          Number.isInteger(value) && value >= 0 && value <= 0xffffffff,
          `${kernel.id}: ${name}`
        ).toBe(true);
      }
    }
  }
});
