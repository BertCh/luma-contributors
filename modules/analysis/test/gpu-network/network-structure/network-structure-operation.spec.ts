// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  CPUNetworkStructureOperation,
  computeNetworkStructureOnCPU
} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {
  GPUNetworkStructureOperation,
  type NetworkStructureOperationProps
} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('SCC-only structure twins ignore an unused empty reverse table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: NetworkStructureOperationProps = {
    nodeCount: 2,
    edgeCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    reverseEdgeSlots: 'reverse',
    output: {strongComponents: 'components'}
  };
  const inputs = {
    offsets: {format: 'uint32' as const, values: new Uint32Array([0, 1, 1])},
    neighbors: {format: 'uint32' as const, values: new Uint32Array([1])},
    reverse: {format: 'uint32' as const, values: new Uint32Array(0)}
  };
  const results = [];
  for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: ['components']
    });
    results.push(rows['components']);
    // One edge 0 -> 1 gives two singleton SCCs. Integer labels compare with zero tolerance.
    expect(rows['components']).toEqual(new Uint32Array([0, 1]));
  }
  expect(results[0]).toEqual(results[1]);
});

for (const fixture of ['blocks', 'invalid-slot', 'isolated'] as const) {
  it(`structure twins compare every integer output exactly: ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const offsets = new Uint32Array(fixture === 'blocks' ? [0, 2, 4, 7, 10, 13, 13] : [0, 1, 1]);
    const neighbors = new Uint32Array(
      fixture === 'blocks' ? [1, 2, 0, 2, 1, 0, 3, 2, 4, 4, 3, 3, 4] : [99]
    );
    const reverse = new Uint32Array(
      fixture === 'blocks' ? [2, 5, 0, 4, 3, 1, 7, 6, 10, 11, 8, 9, 12] : [0xffffffff]
    );
    const props: NetworkStructureOperationProps = {
      nodeCount: offsets.length - 1,
      edgeCount: neighbors.length,
      topology: {offsets: 'offsets', neighbors: 'neighbors'},
      reverseEdgeSlots: 'reverse',
      output: {
        strongComponents: 'components',
        bridges: 'bridges',
        articulationPoints: 'articulation',
        biconnectedComponents: 'blocks'
      }
    };
    if (fixture === 'isolated') {
      props.edgeCount = 0;
      props.output = {strongComponents: 'components', articulationPoints: 'articulation'};
    }
    const inputs = {
      offsets: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array([0, 0, 0]) : offsets
      },
      neighbors: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array(0) : neighbors
      },
      reverse: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array(0) : reverse
      }
    };
    const outputs = Object.values(props.output) as string[];
    const results = [];
    for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation])
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    // All outputs are integer identifiers or flags; tolerance is zero. The direct CPU
    // oracle checks expected values independently of how either operation declares its ports.
    const expected = computeNetworkStructureOnCPU({
      offsets: inputs.offsets.values,
      neighbors: inputs.neighbors.values,
      reverseEdgeSlots: inputs.reverse.values
    });
    for (const [port, value] of Object.entries(props.output)) {
      if (value === undefined) continue;
      expect(results[0][value]).toEqual(results[1][value]);
      expect(results[0][value]).toEqual(expected[port as keyof typeof expected]);
    }
  });
}

for (const [mode, strongComponentsMaximumIterations] of [
  ['parallel', 1],
  ['parallel', 32],
  ['exact', 1]
] as const) {
  it(`operation publishes SCC controls and status: ${mode}, budget ${strongComponentsMaximumIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // One 3-cycle and one isolated vertex: CPU Tarjan labels are [0, 0, 0, 3].
    const offsets = new Uint32Array([0, 1, 2, 3, 3]);
    const neighbors = new Uint32Array([1, 2, 0]);
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: offsets},
        neighbors: {format: 'uint32', values: neighbors}
      },
      operations: [
        new GPUNetworkStructureOperation({
          mode,
          strongComponentsMaximumIterations,
          nodeCount: 4,
          edgeCount: 3,
          topology: {offsets: 'offsets', neighbors: 'neighbors'},
          output: {strongComponents: 'labels'},
          strongComponentsStatus: {
            converged: 'converged',
            iterationCount: 'iterations',
            iterationLimitReached: 'exhausted'
          }
        })
      ],
      outputs: ['labels', 'converged', 'iterations', 'exhausted']
    });
    const finished = mode === 'exact' || strongComponentsMaximumIterations > 1;
    expect(result['labels']).toEqual(
      finished
        ? computeNetworkStructureOnCPU({offsets, neighbors}).strongComponents
        : new Uint32Array(4).fill(0xffffffff)
    );
    expect(result['converged']).toEqual(new Uint32Array([Number(finished)]));
    expect(result['exhausted']).toEqual(new Uint32Array([Number(!finished)]));
    expect(result['iterations'][0]).toBeGreaterThanOrEqual(1);
    expect(result['iterations'][0]).toBeLessThanOrEqual(strongComponentsMaximumIterations);
    const cpuResult = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: offsets},
        neighbors: {format: 'uint32', values: neighbors}
      },
      operations: [
        new CPUNetworkStructureOperation({
          mode,
          strongComponentsMaximumIterations,
          nodeCount: 4,
          edgeCount: 3,
          topology: {offsets: 'offsets', neighbors: 'neighbors'},
          output: {strongComponents: 'labels'},
          strongComponentsStatus: {
            converged: 'converged',
            iterationCount: 'iterations',
            iterationLimitReached: 'exhausted'
          }
        })
      ],
      outputs: ['labels', 'converged', 'iterations', 'exhausted']
    });
    expect(cpuResult['labels']).toEqual(result['labels']);
    expect(cpuResult['converged']).toEqual(result['converged']);
    expect(cpuResult['exhausted']).toEqual(result['exhausted']);
    expect(cpuResult['iterations'][0]).toBeGreaterThanOrEqual(1);
    expect(cpuResult['iterations'][0]).toBeLessThanOrEqual(strongComponentsMaximumIterations);
  }, 120000);
}

it('exact structure twins publish all requested SCC status values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: NetworkStructureOperationProps = {
    mode: 'exact',
    strongComponentsMaximumIterations: 1,
    nodeCount: 4,
    edgeCount: 3,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    output: {strongComponents: 'labels'},
    strongComponentsStatus: {
      converged: 'finished',
      iterationCount: 'steps',
      iterationLimitReached: 'limited'
    }
  };
  for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 3)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 0)}
      },
      operations: [new Operation(props)],
      outputs: ['labels', 'finished', 'steps', 'limited']
    });
    expect(rows['labels']).toEqual(Uint32Array.of(0, 0, 0, 3));
    expect(rows['finished']).toEqual(Uint32Array.of(1));
    expect(rows['steps']).toEqual(Uint32Array.of(1));
    expect(rows['limited']).toEqual(Uint32Array.of(0));
  }
});
