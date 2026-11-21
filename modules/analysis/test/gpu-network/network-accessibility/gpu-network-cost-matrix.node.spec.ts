// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {CPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {
  GPUNetworkCostMatrixOperation,
  type GPUNetworkCostMatrixOperationProps,
  type NetworkCostMatrixOperationProps
} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUNetworkCostMatrix,
  type GPUNetworkCostMatrixProps
} from '../../../src/gpu-network/network-accessibility/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const NODE_COUNT = 8;
const EDGE_COUNT = 12;

let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  rowCount: number,
  overrides: Partial<GPUNetworkCostMatrixProps> = {},
  columnCount: number = NODE_COUNT
): GPUNetworkCostMatrixProps {
  return {
    id: 'matrix',
    topology: {
      offsets: createView(graph, 'uint32', NODE_COUNT + 1),
      neighbors: createView(graph, 'uint32', EDGE_COUNT),
      weights: createView(graph, 'float32', EDGE_COUNT)
    },
    startNodes: createView(graph, 'uint32', rowCount),
    output: {
      costs: createView(graph, 'float32', rowCount * columnCount),
      status: {converged: createView(graph, 'uint32', 1)}
    },
    laneCount: 2,
    maximumIterations: 3,
    ...overrides
  };
}

function getNodeIds(rowCount: number, targetCount?: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostMatrix(
    createProps(
      graph,
      rowCount,
      targetCount === undefined ? {} : {targetNodes: createView(graph, 'uint32', targetCount)},
      targetCount ?? NODE_COUNT
    )
  );
  return {contributor, nodes: contributor.getCommandNodes(graph)};
}

/** Stable traversal IDs, followed by initialization and parallel convergence reduction. */
const SNAPSHOT_IDS = [
  'matrix-expand-seeds',
  'matrix-batch-info',
  'matrix-batch-0-initialize',
  'matrix-batch-0-seed',
  'matrix-batch-0-relax-0',
  'matrix-batch-0-relax-1',
  'matrix-batch-0-relax-2',
  'matrix-batch-0-finalize',
  'matrix-batch-0-copy',
  'matrix-batch-1-initialize',
  'matrix-batch-1-seed',
  'matrix-batch-1-relax-0',
  'matrix-batch-1-relax-1',
  'matrix-batch-1-relax-2',
  'matrix-batch-1-finalize',
  'matrix-batch-1-copy',
  'matrix-batch-2-initialize',
  'matrix-batch-2-seed',
  'matrix-batch-2-relax-0',
  'matrix-batch-2-relax-1',
  'matrix-batch-2-relax-2',
  'matrix-batch-2-finalize',
  'matrix-batch-2-copy',
  'matrix-converged-initialize',
  'matrix-converged'
];

it('GPUNetworkCostMatrix without targetNodes keeps its node list', () => {
  const {contributor, nodes} = getNodeIds(5);
  expect(nodes.find(node => node.id === 'matrix-converged')?.workload?.maximumInvocationCount).toBe(
    contributor.batchCount
  );
  expect(contributor.columnCount).toBe(NODE_COUNT);
  expect(nodes.map(node => node.id)).toEqual(SNAPSHOT_IDS);
});

it('rejects a JavaScript noded handoff without mandatory live count or piece overflow (B7)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 2);
  const count = createView(graph, 'uint32', 1);
  const overflow = createView(graph, 'uint32', 1);
  for (const missing of ['count', 'overflow']) {
    expect(
      () =>
        new GPUNetworkCostMatrix({
          ...props,
          nodedTopology: {
            topology: props.topology,
            nodes: {status: {count: missing === 'count' ? undefined! : count}},
            pieces: {status: {count, overflow: missing === 'overflow' ? undefined! : overflow}},
            status: {overflow, uncertainCount: overflow}
          },
          output: {...props.output, status: {sourceComplete: createView(graph, 'uint32', 1)}}
        })
    ).toThrow(
      new RegExp(
        missing === 'count'
          ? 'nodedTopology.nodes.status.count'
          : 'nodedTopology.pieces.status.overflow'
      )
    );
  }
});

it('snapshots a topology shared with the noded port only once (S2)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 2);
  let reads = 0;
  const topology = {
    ...props.topology,
    get offsets() {
      reads++;
      return createView(graph, 'uint32', NODE_COUNT + 1);
    }
  };
  const word = () => createView(graph, 'uint32', 1);
  expect(
    () =>
      new GPUNetworkCostMatrix({
        ...props,
        topology,
        nodedTopology: {
          topology,
          nodes: {status: {count: word()}},
          pieces: {status: {count: word(), overflow: word()}},
          status: {overflow: word(), uncertainCount: word()}
        },
        output: {...props.output, status: {sourceComplete: word()}}
      })
  ).not.toThrow();
  expect(reads).toBe(1);
});

it('reuses the dense live-row mask pipeline across 24 batches (S3)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const topology = {
    offsets: createView(graph, 'uint32', 65),
    neighbors: createView(graph, 'uint32', 2),
    weights: createView(graph, 'float32', 2)
  };
  const word = () => createView(graph, 'uint32', 1);
  const contributor = new GPUNetworkCostMatrix({
    topology,
    startNodes: createView(graph, 'uint32', 24),
    laneCount: 1,
    maximumIterations: 1,
    nodedTopology: {
      topology,
      nodes: {status: {count: word()}},
      pieces: {status: {count: word(), overflow: word()}},
      status: {overflow: word(), uncertainCount: word()}
    },
    output: {costs: createView(graph, 'float32', 24 * 64), status: {sourceComplete: word()}}
  });
  const masks = contributor
    .getCommandNodes(graph)
    .filter(node => node.id.includes('-mask-inactive-targets-'));
  expect(masks).toHaveLength(24);
  expect(getGPUCommandNodePipelineKeys(masks, {includeShaderLayout: true}).size).toBe(1);
});

it('preserves legacy typed operation props and rejects unsupported CPU noded evidence (B3, S6)', () => {
  const legacy: NetworkCostMatrixOperationProps = {
    nodeCount: 4,
    rowCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs', status: {converged: 'converged'}}
  };
  const gpuProps: GPUNetworkCostMatrixOperationProps = legacy;
  expect(() => new GPUNetworkCostMatrixOperation(gpuProps)).not.toThrow();
  const noded: GPUNetworkCostMatrixOperationProps = {
    ...legacy,
    nodedTopology: {
      nodeCount: 'live-count',
      pieceCount: 'piece-count',
      piecesOverflow: 'piece-overflow',
      overflow: 'source-overflow'
    },
    output: {costs: 'costs', status: {sourceComplete: 'source-complete'}}
  };
  expect(() => new CPUNetworkCostMatrixOperation(noded)).toThrow(/nodedTopology.*GPU/);
  expect(() => new CPUNetworkCostMatrixOperation({...legacy, output: noded.output})).toThrow(
    /sourceComplete.*GPU/
  );
});

it('masks dense live rows with aligned metadata without clearing packed prefixes (S3)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = createProps(graph, 3, {id: 'offset-mask', laneCount: 1, maximumIterations: 1});
  const storageView = createView(graph, 'float32', 27);
  const costs = graph.createDataView(storageView.buffer, {
    format: 'float32',
    byteOffset: 12,
    length: 24
  });
  const word = () => createView(graph, 'uint32', 1);
  const contributor = new GPUNetworkCostMatrix({
    ...props,
    nodedTopology: {
      topology: props.topology,
      nodes: {status: {count: word()}},
      pieces: {status: {count: word(), overflow: word()}},
      status: {overflow: word(), uncertainCount: word()}
    },
    output: {costs, status: {sourceComplete: word()}}
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
  getParsedKernelSources(nodes);
  const metadata = new Uint32Array(3 * 64);
  const values = new Float32Array(27).fill(913);
  const run = (name: string, storage: Record<string, Uint32Array | Float32Array>) => {
    const record = getWGSLKernelNodeRecord(graph, name)!;
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => [index, storage[binding.name]])
    );
    const source = record.source.replace(/^\s*_\s*=[^;]*;/gm, '');
    // These independent scalar invocations have no barriers or workgroup storage. Dispatch
    // one invocation per interpreted workgroup to avoid the interpreter's divergent-if lane bug.
    const scalarSource = source
      .replace(/@workgroup_size\(\d+\)/, '@workgroup_size(1)')
      .replace(
        /workgroupIndex \* \d+u \+ localInvocationIndex/,
        'workgroupIndex + localInvocationIndex'
      );
    new WgslExec(new WgslParser().parse(scalarSource)).dispatchWorkgroups(
      'main',
      [record.invocationCount, 1, 1],
      {0: bindings},
      {constants: record.constants}
    );
  };
  run('offset-mask-mask-metadata', {metadata});
  for (let batch = 0; batch < 3; batch++) {
    expect(Array.from(metadata.slice(batch * 64, batch * 64 + 2))).toEqual([3, 5]);
    run(`offset-mask-mask-inactive-targets-${batch}`, {
      metadata: metadata.subarray(batch * 64, batch * 64 + 2),
      liveNodeCount: Uint32Array.of(2),
      costs: values
    });
  }
  // Three packed prefix words are outside the matrix. Each of its three capacity-8 rows
  // retains exactly two live columns; six spare columns per row become infinity.
  expect(Array.from(values)).toEqual([
    913,
    913,
    913,
    ...Array.from({length: 3}, () => [
      913,
      913,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity,
      Infinity
    ]).flat()
  ]);
});

it('GPUNetworkCostMatrix validates targetNodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const withTargets = (
    targets: GPUNetworkCostMatrixProps['targetNodes'],
    costsLength: number,
    costsFormat: 'float32' | 'uint32' = 'float32'
  ) => {
    const props = createProps(graph, 4, {targetNodes: targets});
    return new GPUNetworkCostMatrix({
      ...props,
      output: {...props.output, costs: createView(graph, costsFormat, costsLength) as never}
    });
  };
  expect(withTargets(createView(graph, 'uint32', 3), 12).rowCount).toBe(4);
  expect(withTargets(createView(graph, 'uint32', 3), 12).columnCount).toBe(3);
  expect(() => withTargets(createView(graph, 'uint32', 3), 13)).toThrow(/multiple of the target/);
  expect(() => withTargets(createView(graph, 'uint32', 3), 0)).toThrow(/multiple of the target/);
  expect(() => withTargets(createView(graph, 'uint32', 0), 12)).toThrow(/targetNodes/);
  expect(() => withTargets(createView(graph, 'float32', 3) as never, 12)).toThrow(/targetNodes/);
  const shared = createView(graph, 'uint32', 1);
  expect(
    () =>
      new GPUNetworkCostMatrix({
        ...createProps(graph, 4),
        targetNodes: shared,
        output: {costs: createView(graph, 'float32', 4), status: {converged: shared}}
      })
  ).toThrow(/outputs/);
});

it('GPUNetworkCostMatrix with targetNodes searches into scratch and gathers every batch', () => {
  const {contributor, nodes} = getNodeIds(5, 3);
  expect(contributor.rowCount).toBe(5);
  expect(contributor.columnCount).toBe(3);
  expect(contributor.batchCount).toBe(3);
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('matrix-expand-seeds');
  expect(ids).toContain('matrix-batch-info');
  for (const batch of [0, 1, 2]) {
    const seed = ids.indexOf(`matrix-batch-${batch}-seed`);
    const relax = ids.indexOf(`matrix-batch-${batch}-relax-2`);
    const gather = ids.indexOf(`matrix-batch-${batch}-gather`);
    expect(ids.indexOf('matrix-batch-info')).toBeLessThan(seed);
    expect(seed).toBeLessThan(relax);
    expect(relax).toBeLessThan(gather);
    expect(ids).not.toContain(`matrix-batch-${batch}-copy`);
  }
  expect(ids.at(-1)).toBe('matrix-converged');
});

it('GPUNetworkCostMatrix kernels parse and stay within eight storage bindings', () => {
  const sources = getParsedKernelSources(getNodeIds(5, 3).nodes);
  const gatherSources = [...sources.entries()].filter(([id]) => id.includes('gather'));
  expect(gatherSources.length).toBeGreaterThan(0);
  for (const {source} of sources.configurations) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
});

it('GPUNetworkCostMatrix with targetNodes compiles the same pipelines for every batch count', () => {
  const pipelineCount = (rowCount: number) =>
    getGPUCommandNodePipelineKeys(getNodeIds(rowCount, 3).nodes, {includeShaderLayout: true}).size;
  // Fixed laneCount 2: 6 rows are 3 batches, 7 rows are 4 batches.
  expect(getNodeIds(6, 3).contributor.batchCount).toBe(3);
  expect(getNodeIds(7, 3).contributor.batchCount).toBe(4);
  expect(pipelineCount(7)).toBe(pipelineCount(6));
  expect(pipelineCount(2)).toBe(pipelineCount(6));
});

it('retains noded live cardinality and source loss independently of convergence', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = createProps(graph, 2, {targetNodes: createView(graph, 'uint32', 1)}, 1);
  const count = createView(graph, 'uint32', 1);
  const overflow = createView(graph, 'uint32', 1);
  const uncertainCount = createView(graph, 'uint32', 1);
  const pieceOverflow = createView(graph, 'uint32', 1);
  const candidateOverflow = createView(graph, 'uint32', 1);
  const sourceComplete = createView(graph, 'uint32', 1);
  const contributor = new GPUNetworkCostMatrix({
    ...props,
    nodedTopology: {
      topology: props.topology,
      nodes: {status: {count}},
      status: {overflow, uncertainCount},
      pieces: {status: {count, overflow: pieceOverflow, candidateOverflow}}
    },
    output: {...props.output, status: {...props.output.status, sourceComplete}}
  });
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  expect(sources.get('matrix-expand-seeds')).toContain('liveNodeCount');
  expect(sources.get('matrix-batch-0-gather')).toContain('liveNodeCount');
  expect(sources.get('matrix-source-complete')).toContain('sourceComplete');
  expect(contributor.result.values).toBe(props.output.costs);
  expect(contributor.result.sourceComplete).toBe(sourceComplete);
  const run = (name: string, storage: Record<string, Uint32Array | Float32Array>) => {
    const record = getWGSLKernelNodeRecord(graph, name)!;
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => [index, storage[binding.name]])
    );
    new WgslExec(
      new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
    ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
  };
  const expandedSeeds = new Uint32Array(64).fill(99);
  run('matrix-expand-seeds', {
    liveNodeCount: Uint32Array.of(2),
    seedNodes: Uint32Array.of(0, 2),
    expandedSeeds
  });
  // With exactly two live nodes, row 2 is spare capacity, even though NODE_COUNT is eight.
  expect(Array.from(expandedSeeds.slice(0, 2))).toEqual([0, 0xffffffff]);
  const costs = new Float32Array(2).fill(99);
  run('matrix-batch-0-gather', {
    liveNodeCount: Uint32Array.of(2),
    batchInfo: Uint32Array.of(0, 2),
    targetNodes: Uint32Array.of(2),
    scratchCosts: new Float32Array(16),
    costs
  });
  expect(Array.from(costs)).toEqual([Infinity, Infinity]);
  for (const activeLoss of [-1, 0, 1, 2, 3]) {
    const complete = Uint32Array.of(99);
    run('matrix-source-complete', {
      loss0: Uint32Array.of(Number(activeLoss === 0)),
      loss1: Uint32Array.of(Number(activeLoss === 1)),
      loss2: Uint32Array.of(Number(activeLoss === 2)),
      loss3: Uint32Array.of(Number(activeLoss === 3)),
      sourceComplete: complete
    });
    expect(Array.from(complete)).toEqual([Number(activeLoss === -1)]);
  }
});

it.each([
  'overflow',
  'uncertainCount',
  'candidateOverflow'
] as const)('handles omitted optional noded %s evidence (S4)', omitted => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = createProps(graph, 2);
  const word = () => createView(graph, 'uint32', 1);
  const contributor = new GPUNetworkCostMatrix({
    ...props,
    nodedTopology: {
      topology: props.topology,
      nodes: {status: {count: word()}},
      pieces: {
        status: {
          count: word(),
          overflow: word(),
          candidateOverflow: omitted === 'candidateOverflow' ? undefined : word()
        }
      },
      status: {
        overflow: omitted === 'overflow' ? undefined : word(),
        uncertainCount: omitted === 'uncertainCount' ? undefined : word()
      }
    },
    output: {...props.output, status: {sourceComplete: word()}}
  });
  contributor.getCommandNodes(graph);
  const record = getWGSLKernelNodeRecord(graph, 'matrix-source-complete')!;
  const complete = Uint32Array.of(99);
  const bindings = Object.fromEntries(
    record.bindings.map((binding, index) => [
      index,
      binding.name === 'sourceComplete' ? complete : Uint32Array.of(0)
    ])
  );
  new WgslExec(
    new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
  ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
  expect(Array.from(complete)).toEqual([omitted === 'candidateOverflow' ? 1 : 0]);
});
