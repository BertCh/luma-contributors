// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {CPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {GPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {CPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {
  createPathStatusNode,
  createPathCheckTotalsNode,
  createPathSuppressUnrepresentableNode,
  createPathTotalsNode
} from '../../../src/gpu-network/network-analysis/network-path-extraction-passes';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';

it('UNW-01 composes authoritative slots through both named search and extraction forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cpu of [false, true]) {
    const Search = cpu ? CPUNetworkReachabilityOperation : GPUNetworkReachabilityOperation;
    const Extract = cpu ? CPUNetworkPathExtractionOperation : GPUNetworkPathExtractionOperation;
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 2)},
        weights: {format: 'float32', values: Float32Array.of(0.1, 9, 0.2)},
        starts: {format: 'uint32', values: Uint32Array.of(0)},
        targets: {format: 'uint32', values: Uint32Array.of(2)}
      },
      operations: [
        new Search({
          nodeCount: 3,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          startNodes: 'starts',
          output: {
            costs: 'costs',
            predecessors: 'parents',
            predecessorEdges: 'parentEdges',
            status: {converged: 'complete'}
          }
        }),
        new Extract({
          nodeCount: 3,
          targetCapacity: 1,
          predecessors: 'parents',
          predecessorEdges: 'parentEdges',
          costs: 'costs',
          targets: 'targets',
          predecessorsComplete: 'complete',
          output: {
            capacity: 3,
            ids: 'nodes',
            status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
          },
          edges: {
            edgeCount: 3,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              capacity: 2,
              ids: 'edges',
              edgeRows: 'edgeRows',
              status: {count: 'edgeCount', requiredCount: 'edgeRequired', overflow: 'edgeOverflow'}
            }
          }
        })
      ],
      outputs: ['nodes', 'edges', 'edgeRows', 'complete']
    });
    expect(Array.from(result['nodes'])).toEqual([0, 1, 2]);
    expect(Array.from(result['edges'])).toEqual([0, 2]);
    expect(Array.from(result['edgeRows'])).toEqual([0, 2]);
    expect(result['complete'][0]).toBe(1);
  }
});

it('UNW-03 printed composition suppresses an unfinished zero-weight tie forest', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 3)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3)},
      weights: {format: 'float32', values: Float32Array.of(0, 0, 0)},
      starts: {format: 'uint32', values: Uint32Array.of(0)},
      targets: {format: 'uint32', values: Uint32Array.of(3)}
    },
    operations: [
      new GPUNetworkReachabilityOperation({
        nodeCount: 4,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        localIterations: 1,
        maximumTieIterations: 1,
        output: {costs: 'costs', predecessors: 'parents', status: {converged: 'complete'}}
      }),
      new GPUNetworkPathExtractionOperation({
        nodeCount: 4,
        targetCapacity: 1,
        predecessors: 'parents',
        costs: 'costs',
        targets: 'targets',
        predecessorsComplete: 'complete',
        output: {
          capacity: 4,
          ids: 'nodes',
          pathFound: 'found',
          status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
        }
      })
    ],
    outputs: ['complete', 'found', 'count', 'nodes']
  });
  expect(result['complete'][0]).toBe(0);
  expect(result['found'][0]).toBe(0);
  expect(Array.from(result['nodes'].slice(0, result['count'][0]))).toEqual([]);
});

it('DGN-11 executes zero and exact compact capacities through both named forms', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Extract of [CPUNetworkPathExtractionOperation, GPUNetworkPathExtractionOperation])
    for (const capacity of [0, 1]) {
      const result = await runOperationGraph(device, {
        inputs: {
          parents: {format: 'uint32', values: Uint32Array.of(0xffffffff)},
          costs: {format: 'float32', values: Float32Array.of(0)},
          targets: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [
          new Extract({
            nodeCount: 1,
            targetCapacity: 1,
            predecessors: 'parents',
            costs: 'costs',
            targets: 'targets',
            output: {
              capacity,
              ids: 'nodes',
              nodeRows: 'rows',
              pathFound: 'found',
              status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
            }
          })
        ],
        outputs: ['nodes', 'rows', 'count', 'required', 'overflow', 'found']
      });
      expect(result['count'][0]).toBe(capacity);
      expect(result['required'][0]).toBe(1);
      expect(result['overflow'][0]).toBe(1 - capacity);
      expect(result['found'][0]).toBe(1);
    }
});

it('UNW-08 suppresses an unrepresentable final scan addition before publication', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const starts = harness.input('uint32', Uint32Array.of(4294901760));
  const lengths = harness.input('uint32', Uint32Array.of(65536));
  const status = harness.output('uint32', 3),
    total = harness.output('uint32', 1),
    offsets = harness.output('uint32', 2);
  // 4294901760 + 65536 = 2^32: uint32 addition alone would publish zero as a success.
  const read = await harness.run({
    getCommandNodes: graph => [
      createPathStatusNode(graph, {id: 'reset', status}),
      createPathCheckTotalsNode(graph, {id: 'check', starts, lengths, status}),
      createPathSuppressUnrepresentableNode(graph, {id: 'suppress', starts, lengths, status}),
      createPathTotalsNode(graph, {
        id: 'publish',
        targetCapacity: 1,
        nodes: {lengths, starts, total, pathOffsets: offsets}
      })
    ]
  });
  expect(read(status)).toEqual([0, 1, 1]);
  expect(read(total)).toEqual([0]);
  expect(read(offsets)).toEqual([0, 0]);
});

it('UNW-08 admits 65536 targets and vertices with a one-node walk bound on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      parents: {format: 'uint32', values: new Uint32Array(65536).fill(0xffffffff)},
      costs: {format: 'float32', values: new Float32Array(65536)},
      targets: {format: 'uint32', values: new Uint32Array(65536)}
    },
    operations: [
      new GPUNetworkPathExtractionOperation({
        nodeCount: 65536,
        targetCapacity: 65536,
        maximumPathLength: 1,
        predecessors: 'parents',
        costs: 'costs',
        targets: 'targets',
        output: {
          capacity: 0,
          ids: 'nodes',
          status: {
            count: 'count',
            requiredCount: 'required',
            overflow: 'overflow',
            unrepresentable: 'unrepresentable'
          }
        }
      })
    ],
    outputs: ['count', 'required', 'overflow', 'unrepresentable']
  });
  expect(result['count'][0]).toBe(0);
  expect(result['required'][0]).toBe(65536);
  expect(result['overflow'][0]).toBe(1);
  expect(result['unrepresentable'][0]).toBe(0);
});

it('S2 named extraction distinguishes authoritative slots from rounded-label inference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 1)},
      weights: {format: 'float32', values: Float32Array.of(1, 0)},
      starts: {format: 'uint32', values: Uint32Array.of(0)},
      startCosts: {format: 'float32', values: Float32Array.of(2 ** 24)},
      targets: {format: 'uint32', values: Uint32Array.of(1)}
    },
    operations: [
      new GPUNetworkReachabilityOperation({
        nodeCount: 2,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        startCosts: 'startCosts',
        output: {
          costs: 'costs',
          predecessors: 'parents',
          predecessorEdges: 'parentEdges',
          status: {converged: 'complete'}
        }
      }),
      ...([true, false] as const).map(authoritative => {
        const prefix = authoritative ? 'exact' : 'legacy';
        return new GPUNetworkPathExtractionOperation({
          id: prefix,
          nodeCount: 2,
          targetCapacity: 1,
          predecessors: 'parents',
          predecessorEdges: authoritative ? 'parentEdges' : undefined,
          costs: 'costs',
          targets: 'targets',
          predecessorsComplete: 'complete',
          output: {
            capacity: 2,
            ids: `${prefix}Nodes`,
            status: {
              count: `${prefix}Count`,
              requiredCount: `${prefix}Required`,
              overflow: `${prefix}Overflow`
            }
          },
          edges: {
            edgeCount: 2,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              capacity: 1,
              ids: `${prefix}Edges`,
              status: {
                count: `${prefix}EdgeCount`,
                requiredCount: `${prefix}EdgeRequired`,
                overflow: `${prefix}EdgeOverflow`
              }
            }
          }
        });
      })
    ],
    outputs: ['complete', 'exactEdges', 'legacyEdges']
  });
  // 2^24+1 and 2^24+0 both round to 2^24 in f32; only provenance names the zero-cost slot.
  expect(result['complete'][0]).toBe(1);
  expect(Array.from(result['exactEdges'])).toEqual([1]);
  expect(Array.from(result['legacyEdges'])).toEqual([0]);
});

it('S6 rejects negative subnormal edge certificates while permitting either signed zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const slot of [0, 1, 2]) {
    const result = await runOperationGraph(device, {
      inputs: {
        parents: {format: 'uint32', values: Uint32Array.of(0xffffffff, 0)},
        slots: {format: 'uint32', values: Uint32Array.of(0xffffffff, slot)},
        costs: {format: 'float32', values: Float32Array.of(0, 0)},
        targets: {format: 'uint32', values: Uint32Array.of(1)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 3, 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 1, 1)},
        weights: {
          format: 'float32',
          values: new Float32Array(Uint32Array.of(0x80000001, 0, 0x80000000).buffer)
        }
      },
      operations: [
        new GPUNetworkPathExtractionOperation({
          nodeCount: 2,
          targetCapacity: 1,
          predecessors: 'parents',
          predecessorEdges: 'slots',
          costs: 'costs',
          targets: 'targets',
          output: {
            capacity: 2,
            ids: 'nodes',
            status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
          },
          edges: {
            edgeCount: 3,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            output: {
              capacity: 1,
              ids: 'ids',
              edgeRows: 'rows',
              status: {count: 'edgeCount', requiredCount: 'edgeRequired', overflow: 'edgeOverflow'}
            }
          }
        })
      ],
      outputs: ['rows']
    });
    expect(Array.from(result['rows'])).toEqual([slot === 0 ? 0xffffffff : slot]);
  }
});
