// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {
  GPUMapMatchingOperation,
  type MapMatchingOperationProps
} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {CPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {
  GPUNetworkCostTableOperation,
  type NetworkCostTableOperationProps
} from '../../../src/gpu-network/network-cost-table/gpu-network-cost-table-operation';
import {
  CPUNetworkCostTableOperation,
  computeNetworkCostTableOnCPU
} from '../../../src/gpu-network/network-cost-table/cpu-network-cost-table-operation';
import {createOptimizationHarness} from '../network-optimization/optimization-test-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUMapMatchedEdgeVolumes} from '../../../src/gpu-network/flow-aggregation/gpu-map-matched-edge-volumes';
import {createFinalMatchingInput} from './map-matching-final-review-fixture';

const OFFSETS = Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 8);
const NEIGHBORS = Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8);
const WEIGHTS = new Float32Array(8).fill(1);
const NODE_POSITIONS = Float32Array.from({length: 18}, (_, index) =>
  index % 2 === 0 ? index / 2 : 0
);
const POSITIONS = Float32Array.of(0.5, 0, 3.5, 0);
const TRACK_OFFSETS = Uint32Array.of(0, 2);
const PARAMETERS = getGPUMapMatchingParameterValues({
  sigma: 0.1,
  beta: 1,
  searchRadius: 0.1,
  routeFactor: 3,
  routeSlack: 0
});
const BOUNDS = {minimum: [-1, -1] as const, maximum: [9, 1] as const};

it('network:F1 does not count a turn across an unrecovered hop and clears incompleteness on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFinalMatchingInput();
  const harness = createOptimizationHarness(device);
  const positions = harness.input('float32x2', fixture.positions);
  const traversedEdges = {
    edges: harness.output('uint32', 16),
    counts: harness.output('uint32', 1),
    status: {overflow: harness.output('uint32', 1), candidateOverflow: harness.output('uint32', 1)}
  };
  const matching = new GPUMapMatching({
    positions,
    trackOffsets: harness.input('uint32', fixture.trackOffsets),
    parameters: harness.input('float32', fixture.parameters),
    topology: {
      offsets: harness.input('uint32', fixture.topology.offsets),
      neighbors: harness.input('uint32', fixture.topology.neighbors),
      nodePositions: harness.input('float32x2', fixture.topology.nodePositions)
    },
    routeTable: {
      offsets: harness.input('uint32', fixture.routeTable.offsets),
      targets: harness.input('uint32', fixture.routeTable.targets),
      costs: harness.input('float32', fixture.routeTable.costs),
      costLimit: harness.input('float32', fixture.routeTable.costLimit),
      status: {complete: harness.input('uint32', fixture.routeTable.complete)}
    },
    cellSize: 1,
    bounds: {minimum: [-1, -1], maximum: [12, 1]},
    candidateCount: 1,
    routeNodeBudget: 8,
    traversedEdgeCapacity: 16,
    output: {matchedEdges: harness.output('uint32', 2), traversedEdges}
  });
  const transitions = {
    ids: harness.output('uint32', 16),
    status: {
      count: harness.output('uint32', 1),
      overflow: harness.output('uint32', 1),
      candidateOverflow: harness.output('uint32', 1)
    }
  };
  const volumesOutput = {
    edgeVolumes: harness.output('uint32', 11),
    status: {candidateOverflow: harness.output('uint32', 1)},
    transitions
  };
  const volumes = new GPUMapMatchedEdgeVolumes({
    traversedEdges,
    trackCount: 1,
    traversedEdgeCapacity: 16,
    edgeCount: 11,
    transitionPairCapacity: 32,
    output: volumesOutput
  });
  const read = await harness.run(
    {
      getCommandNodes: graph => [
        ...matching.getCommandNodes(graph),
        ...volumes.getCommandNodes(graph)
      ]
    },
    first => {
      expect(first(traversedEdges.edges).slice(0, 3)).toEqual([0, 0xffffffff, 10]);
      expect(first(traversedEdges.status.overflow)).toEqual([0]);
      expect(first(traversedEdges.status.candidateOverflow)).toEqual([1]);
      expect(first(volumesOutput.status.candidateOverflow)).toEqual([1]);
      expect(first(transitions.status.candidateOverflow)).toEqual([1]);
      expect(first(transitions.status.count)).toEqual([0]); // no invented 0->10 movement
      harness.rewrite(positions, Float32Array.of(0.5, 0, 1.5, 0));
    }
  );
  expect(read(traversedEdges.edges).slice(0, 2)).toEqual([0, 1]);
  expect(read(traversedEdges.status.candidateOverflow)).toEqual([0]);
  expect(read(volumesOutput.status.candidateOverflow)).toEqual([0]);
  expect(read(transitions.status.candidateOverflow)).toEqual([0]);
  expect(read(transitions.status.count)).toEqual([1]);
  expect(read(transitions.ids)[0]).toBe(1); // 0*11+1
});

it('network:F6 both operation backends honor the fallback budget when completeness is cleared', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFinalMatchingInput();
  const inputs = {
    points: {format: 'float32x2' as const, values: fixture.positions},
    tracks: {format: 'uint32' as const, values: fixture.trackOffsets},
    parameters: {format: 'float32' as const, values: fixture.parameters},
    offsets: {format: 'uint32' as const, values: fixture.topology.offsets},
    neighbors: {format: 'uint32' as const, values: fixture.topology.neighbors},
    nodes: {format: 'float32x2' as const, values: fixture.topology.nodePositions},
    tableOffsets: {format: 'uint32' as const, values: fixture.routeTable.offsets},
    tableTargets: {format: 'uint32' as const, values: fixture.routeTable.targets},
    tableCosts: {format: 'float32' as const, values: fixture.routeTable.costs},
    cutoff: {format: 'float32' as const, values: fixture.routeTable.costLimit},
    complete: {format: 'uint32' as const, values: fixture.routeTable.complete}
  };
  const props: MapMatchingOperationProps = {
    pointCount: 2,
    trackCount: 1,
    nodeCount: 12,
    edgeCount: 11,
    positions: 'points',
    trackOffsets: 'tracks',
    parameters: 'parameters',
    topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodes'},
    cellSize: 1,
    bounds: {minimum: [-1, -1], maximum: [12, 1]},
    candidateCount: 1,
    routeNodeBudget: 8,
    traversedEdgeCapacity: 16,
    routeTable: {
      entryCount: 144,
      offsets: 'tableOffsets',
      targets: 'tableTargets',
      costs: 'tableCosts',
      costLimit: 'cutoff',
      status: {complete: 'complete'}
    },
    output: {
      matchedEdges: 'edges',
      breaks: 'breaks',
      status: {overflow: 'overflow', candidateOverflow: 'lost', routeOverflow: 'dropped'},
      traversedEdges: {
        edges: 'traversed',
        counts: 'counts',
        status: {overflow: 'traversedOverflow', candidateOverflow: 'traversedLost'}
      }
    }
  };
  for (const Operation of [GPUMapMatchingOperation, CPUMapMatchingOperation])
    for (const complete of [1, 0]) {
      fixture.routeTable.complete[0] = complete;
      props.routeNodeBudget = 8;
      const operation = new Operation(props);
      props.routeNodeBudget = 16; // Validated budget is retained by both deferred operation forms.
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: [
          'edges',
          'breaks',
          'overflow',
          'lost',
          'dropped',
          'traversed',
          'counts',
          'traversedOverflow',
          'traversedLost'
        ]
      });
      expect(Array.from(result['edges'])).toEqual([0, 10]);
      expect(Array.from(result['breaks'])).toEqual(complete ? [0, 0] : [0, 2]);
      expect(result['overflow'][0]).toBe(0);
      expect(result['lost'][0]).toBe(1);
      if (complete) expect(result['dropped'][0]).toBe(0);
      else expect(result['dropped'][0]).toBeGreaterThan(0);
      expect(Array.from(result['traversed'].slice(0, 3))).toEqual([0, 0xffffffff, 10]);
      expect(result['traversedOverflow'][0]).toBe(0);
      expect(result['traversedLost'][0]).toBe(1);
    }
});

for (const mode of ['astar', 'exhausted', 'complete', 'overflow'] as const) {
  it(`chain matching uses A* or a certified table without false breaks (${mode}, contributor)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createOptimizationHarness(device);
    const topology = {
      offsets: harness.input('uint32', OFFSETS),
      neighbors: harness.input('uint32', NEIGHBORS),
      weights: harness.input('float32', WEIGHTS),
      nodePositions: harness.input('float32x2', NODE_POSITIONS)
    };
    const tableOutput = {
      offsets: harness.output('uint32', 10),
      targets: harness.output('uint32', mode === 'overflow' ? 3 : 81),
      costs: harness.output('float32', mode === 'overflow' ? 3 : 81),
      // No separate converged output: complete must track convergence itself.
      status: {complete: harness.output('uint32', 1), requiredCount: harness.output('uint32', 1)}
    };
    const costLimit = harness.input('float32', Float32Array.of(100));
    const table = new GPUNetworkCostTable({
      topology,
      costLimit,
      laneCount: 4,
      maximumIterations: mode === 'exhausted' ? 1 : 16,
      localIterations: 1,
      output: tableOutput
    });
    const output = {
      matchedEdges: harness.output('uint32', 2),
      breaks: harness.output('uint32', 2),
      status: {routeOverflow: harness.output('uint32', 1), overflow: harness.output('uint32', 1)}
    };
    const matcher = new GPUMapMatching({
      topology,
      positions: harness.input('float32x2', POSITIONS),
      trackOffsets: harness.input('uint32', TRACK_OFFSETS),
      parameters: harness.input('float32', PARAMETERS),
      candidateCount: 1,
      cellSize: 1,
      bounds: BOUNDS,
      routeTable: mode === 'astar' ? undefined : {...tableOutput, costLimit},
      output
    });
    const read = await harness.run({
      getCommandNodes: graph => [
        ...(mode === 'astar' ? [] : table.getCommandNodes(graph)),
        ...matcher.getCommandNodes(graph)
      ]
    });
    expect(read(output.matchedEdges)).toEqual([0, 3]);
    expect(read(output.breaks)).toEqual([0, 0]);
    expect(read(output.status.routeOverflow)).toEqual([0]);
    expect(read(output.status.overflow)).toEqual([0]);
    if (mode !== 'astar') {
      expect(read(tableOutput.status.complete)).toEqual([Number(mode === 'complete')]);
      expect(read(tableOutput.status.requiredCount)).toEqual([mode === 'exhausted' ? 17 : 45]);
    }
  });

  it(`chain matching keeps table completeness in both operation forms (${mode})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const tableProps: NetworkCostTableOperationProps = {
      id: 'table',
      nodeCount: 9,
      capacity: mode === 'overflow' ? 3 : 81,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      costLimit: 'cutoff',
      laneCount: 4,
      maximumIterations: mode === 'exhausted' ? 1 : 16,
      localIterations: 1,
      output: {
        offsets: 'tableOffsets',
        targets: 'tableTargets',
        costs: 'tableCosts',
        status: {complete: 'tableComplete', requiredCount: 'tableRequired'}
      }
    };
    const matchProps: MapMatchingOperationProps = {
      id: 'match',
      nodeCount: 9,
      edgeCount: 8,
      pointCount: 2,
      trackCount: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodes'},
      positions: 'points',
      trackOffsets: 'tracks',
      parameters: 'parameters',
      candidateCount: 1,
      cellSize: 1,
      bounds: BOUNDS,
      routeTable:
        mode === 'astar'
          ? undefined
          : {
              entryCount: tableProps.capacity,
              offsets: 'tableOffsets',
              targets: 'tableTargets',
              costs: 'tableCosts',
              costLimit: 'cutoff',
              status: {complete: 'tableComplete'}
            },
      output: {
        matchedEdges: 'edges',
        breaks: 'breaks',
        trackLogLikelihoods: 'likelihood',
        status: {routeOverflow: 'routeOverflow'}
      }
    };
    const outputs = [
      'edges',
      'breaks',
      'likelihood',
      'routeOverflow',
      ...(mode === 'astar' ? [] : ['tableComplete', 'tableRequired'])
    ];
    const inputs = {
      offsets: {format: 'uint32' as const, values: OFFSETS},
      neighbors: {format: 'uint32' as const, values: NEIGHBORS},
      weights: {format: 'float32' as const, values: WEIGHTS},
      nodes: {format: 'float32x2' as const, values: NODE_POSITIONS},
      points: {format: 'float32x2' as const, values: POSITIONS},
      tracks: {format: 'uint32' as const, values: TRACK_OFFSETS},
      parameters: {format: 'float32' as const, values: PARAMETERS},
      cutoff: {format: 'float32' as const, values: Float32Array.of(100)}
    };
    for (const useCPU of [false, true]) {
      const Table = useCPU ? CPUNetworkCostTableOperation : GPUNetworkCostTableOperation;
      const Matcher = useCPU ? CPUMapMatchingOperation : GPUMapMatchingOperation;
      const operations = [
        ...(mode === 'astar' ? [] : [new Table(tableProps)]),
        new Matcher(matchProps)
      ];
      const result = await runOperationGraph(device, {inputs, operations, outputs});
      expect(Array.from(result['edges'])).toEqual([0, 3]);
      expect(Array.from(result['breaks'])).toEqual([0, 0]);
      expect(Array.from(result['routeOverflow'])).toEqual([0]);
      expect(Number.isFinite(result['likelihood'][0])).toBe(true);
      // Both snaps are exact and network length equals displacement, so only two Gaussian
      // log densities contribute (beta = 1 makes the transition constant zero).
      expect(result['likelihood'][0]).toBeCloseTo(-2 * Math.log(0.1 * Math.sqrt(2 * Math.PI)), 5);
      if (mode !== 'astar') {
        expect(Array.from(result['tableComplete'])).toEqual([Number(mode === 'complete')]);
        expect(Array.from(result['tableRequired'])).toEqual([mode === 'exhausted' ? 17 : 45]);
      }
    }
  });
}

it('uses a complete table past the A* budget and reports fallback drops when completeness is cleared', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const topology = {
    offsets: Uint32Array.from({length: 13}, (_, node) => Math.min(node, 11)),
    neighbors: Uint32Array.from({length: 11}, (_, node) => node + 1),
    weights: new Float32Array(11).fill(1)
  };
  const costLimit = Float32Array.of(40);
  const table = computeNetworkCostTableOnCPU({topology, costLimit}, {capacity: 144});
  expect(table.complete[0]).toBe(1);
  const complete = harness.input('uint32', table.complete);
  const output = {
    matchedEdges: harness.output('uint32', 2),
    breaks: harness.output('uint32', 2),
    status: {routeOverflow: harness.output('uint32', 1)}
  };
  const matcher = new GPUMapMatching({
    topology: {
      offsets: harness.input('uint32', topology.offsets),
      neighbors: harness.input('uint32', topology.neighbors),
      nodePositions: harness.input(
        'float32x2',
        Float32Array.from({length: 24}, (_, index) => (index % 2 === 0 ? index / 2 : 0))
      )
    },
    positions: harness.input('float32x2', Float32Array.of(0.5, 0, 10.5, 0)),
    trackOffsets: harness.input('uint32', TRACK_OFFSETS),
    parameters: harness.input(
      'float32',
      getGPUMapMatchingParameterValues({
        sigma: 0.1,
        beta: 1,
        searchRadius: 0.1,
        routeFactor: 3,
        routeSlack: 0
      })
    ),
    candidateCount: 1,
    routeNodeBudget: 8,
    cellSize: 1,
    bounds: {minimum: [-1, -1], maximum: [12, 1]},
    routeTable: {
      offsets: harness.input('uint32', table.offsets),
      targets: harness.input('uint32', table.targets),
      costs: harness.input('float32', table.costs),
      costLimit: harness.input('float32', costLimit),
      status: {complete}
    },
    output
  });
  const read = await harness.run(matcher, first => {
    expect(first(output.matchedEdges)).toEqual([0, 10]);
    expect(first(output.breaks)).toEqual([0, 0]);
    expect(first(output.status.routeOverflow)).toEqual([0]);
    harness.rewrite(complete, Uint32Array.of(0));
  });
  expect(read(output.breaks)).toEqual([0, 2]);
  expect(read(output.status.routeOverflow)[0]).toBeGreaterThan(0);
});

it('a CPU cost table feeds GPU matching at the inclusive float32 cutoff without a false break', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // CSR slot order: 0->1, 1->3, 2->4, 3->2. Ties at the fixes select edges 0 and 2.
  // Interior cost is 0.5 + half an f32 ULP: float64 exceeds the cutoff, while f32 rounds to 0.5.
  // Binary-exact axis lengths also isolate path addition from sqrt/projection approximation;
  // the former decimal geometry already broke in GPU A*-only before consulting the table.
  const halfUlp = 2 ** -25;
  const inputs = {
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1, 2, 3, 4, 4)},
    neighbors: {format: 'uint32' as const, values: Uint32Array.of(1, 3, 4, 2)},
    weights: {format: 'float32' as const, values: Float32Array.of(0.5, 0.5, 0.5, halfUlp)},
    nodes: {
      format: 'float32x2' as const,
      values: Float32Array.of(-0.5, 0, 0, 0, 0.5, halfUlp, 0.5, 0, 1, halfUlp)
    },
    points: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 0.5, halfUlp)},
    tracks: {format: 'uint32' as const, values: TRACK_OFFSETS},
    parameters: {
      format: 'float32' as const,
      values: getGPUMapMatchingParameterValues({
        sigma: 0.01,
        beta: 1,
        searchRadius: 0.01,
        routeFactor: 0,
        routeSlack: 0.5
      })
    },
    cutoff: {format: 'float32' as const, values: Float32Array.of(0.5)}
  };
  const baseProps: MapMatchingOperationProps = {
    id: 'cutoff-match',
    nodeCount: 5,
    edgeCount: 4,
    pointCount: 2,
    trackCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodes'},
    positions: 'points',
    trackOffsets: 'tracks',
    parameters: 'parameters',
    candidateCount: 1,
    cellSize: 0.1,
    bounds: {minimum: [-0.6, -0.1], maximum: [1.1, 0.1]},
    output: {matchedEdges: 'edges', breaks: 'breaks'}
  };
  const astar = await runOperationGraph(device, {
    inputs,
    operations: [new GPUMapMatchingOperation(baseProps)],
    outputs: ['edges', 'breaks']
  });
  expect(Array.from(astar['edges'])).toEqual([0, 2]);
  expect(Array.from(astar['breaks'])).toEqual([0, 0]);
  const mixed = await runOperationGraph(device, {
    inputs,
    operations: [
      new CPUNetworkCostTableOperation({
        id: 'cpu-cutoff-table',
        nodeCount: 5,
        capacity: 25,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        costLimit: 'cutoff',
        output: {
          offsets: 'tableOffsets',
          targets: 'tableTargets',
          costs: 'tableCosts',
          status: {complete: 'tableComplete'}
        }
      }),
      new GPUMapMatchingOperation({
        ...baseProps,
        routeTable: {
          entryCount: 25,
          offsets: 'tableOffsets',
          targets: 'tableTargets',
          costs: 'tableCosts',
          costLimit: 'cutoff',
          status: {complete: 'tableComplete'}
        }
      })
    ],
    outputs: ['edges', 'breaks', 'tableComplete']
  });
  expect(Array.from(mixed['tableComplete'])).toEqual([1]);
  expect(Array.from(mixed['edges'])).toEqual(Array.from(astar['edges']));
  expect(Array.from(mixed['breaks'])).toEqual([0, 0]);
});

it('LIF-01 retains the build cutoff across separate table and matching graphs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1, 2, 3, 3)},
    neighbors: {format: 'uint32' as const, values: Uint32Array.of(1, 2, 3)},
    weights: {format: 'float32' as const, values: Float32Array.of(1, 1, 1)},
    nodes: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)},
    cutoff: {format: 'float32' as const, values: Float32Array.of(0)}
  };
  const tableProps: NetworkCostTableOperationProps = {
    nodeCount: 4,
    capacity: 16,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    costLimit: 'cutoff',
    output: {
      offsets: 'tableOffsets',
      targets: 'tableTargets',
      costs: 'tableCosts',
      builtCostLimit: 'built',
      status: {complete: 'tableComplete'}
    }
  };
  const build = () =>
    runOperationGraph(device, {
      inputs,
      operations: [new GPUNetworkCostTableOperation(tableProps)],
      outputs: ['tableOffsets', 'tableTargets', 'tableCosts', 'built', 'tableComplete']
    });
  const retained = await build();
  inputs.cutoff.values[0] = 10;
  const fresh = await build();
  expect(retained['tableComplete'][0]).toBe(1);
  expect(fresh['tableComplete'][0]).toBe(1);
  expect(retained['built'][0]).toBe(0);
  expect(fresh['built'][0]).toBe(10);
  for (const table of [retained, fresh]) {
    const result = await runOperationGraph(device, {
      inputs: {
        ...inputs,
        tableOffsets: {format: 'uint32', values: table['tableOffsets']},
        tableTargets: {format: 'uint32', values: table['tableTargets']},
        tableCosts: {format: 'float32', values: table['tableCosts']},
        built: {format: 'float32', values: table['built']},
        tableComplete: {format: 'uint32', values: table['tableComplete']},
        points: {format: 'float32x2', values: Float32Array.of(0.5, 0, 2.5, 0)},
        tracks: {format: 'uint32', values: Uint32Array.of(0, 2)},
        parameters: {format: 'float32', values: PARAMETERS}
      },
      operations: [
        new GPUMapMatchingOperation({
          nodeCount: 4,
          edgeCount: 3,
          pointCount: 2,
          trackCount: 1,
          positions: 'points',
          trackOffsets: 'tracks',
          parameters: 'parameters',
          topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodes'},
          candidateCount: 1,
          cellSize: 1,
          bounds: {minimum: [-1, -1], maximum: [4, 1]},
          routeTable: {
            entryCount: 16,
            offsets: 'tableOffsets',
            targets: 'tableTargets',
            costs: 'tableCosts',
            costLimit: 'built',
            status: {complete: 'tableComplete'}
          },
          output: {matchedEdges: 'matched', breaks: 'breaks'}
        })
      ],
      outputs: ['matched', 'breaks']
    });
    expect(Array.from(result['matched'])).toEqual([0, 2]);
    expect(Array.from(result['breaks'])).toEqual([0, 0]);
  }
});
