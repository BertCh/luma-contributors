// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {CPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {
  GPUNetworkServiceAreasOperation,
  type NetworkServiceAreasOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas-operation';
import {CPUNetworkServiceAreasOperation} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';
import {GPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {CPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const activeIterations of [0, 1, 4])
  it(`R1 both service-area backends bind active cost rounds (${activeIterations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUNetworkServiceAreasOperation, CPUNetworkServiceAreasOperation]) {
      let operation = new Operation({
        nodeCount: 3,
        facilityCapacity: 1,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        facilities: 'facilities',
        activeIterations: 'active',
        maximumIterations: 4,
        maximumLabelIterations: 4,
        maximumTieIterations: 4,
        localIterations: 1,
        output: {
          assignments: 'assignments',
          costs: 'costs',
          facilityNodeCounts: 'counts',
          status: {converged: 'converged', iterationLimitReached: 'limit'}
        }
      });
      if (operation instanceof GPUNetworkServiceAreasOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          facilities: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(activeIterations)}
        },
        operations: [operation],
        outputs: ['costs', 'counts', 'converged', 'limit']
      });
      // A unit chain admits one extra node per one-hop round, never any CPU lookahead.
      expect(Array.from(rows['costs'])).toEqual(
        activeIterations === 0
          ? [0, Infinity, Infinity]
          : activeIterations === 1
            ? [0, 1, Infinity]
            : [0, 1, 2]
      );
      expect(Array.from(rows['counts'])).toEqual([
        activeIterations === 4 ? 3 : activeIterations + 1
      ]);
      expect([rows['converged'][0], rows['limit'][0]]).toEqual(
        activeIterations === 4 ? [1, 0] : [0, 1]
      );
    }
  });

it('R1 both service-area backends retain active binding and phase budgets on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUNetworkServiceAreasOperation, CPUNetworkServiceAreasOperation]) {
    const props: NetworkServiceAreasOperationProps = {
      nodeCount: 3,
      facilityCapacity: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      facilities: 'facilities',
      activeIterations: 'active',
      maximumIterations: 1,
      maximumLabelIterations: 4,
      localIterations: 1,
      output: {
        assignments: 'assignments',
        costs: 'costs',
        facilityNodeCounts: 'counts',
        status: {converged: 'converged', iterationLimitReached: 'limit'}
      }
    };
    let operation = new Operation(props);
    if (operation instanceof GPUNetworkServiceAreasOperation)
      operation = poisonOperationOutputs(operation);
    for (let replay = 0; replay < 2; replay++) {
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          facilities: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [operation],
        outputs: ['costs', 'counts', 'converged', 'limit']
      });
      // Zero active rounds must remain seed-only after caller mutation.
      expect(Array.from(rows['costs'])).toEqual([0, Infinity, Infinity]);
      expect(Array.from(rows['counts'])).toEqual([1]);
      expect([rows['converged'][0], rows['limit'][0]]).toEqual([0, 1]);
      delete props.activeIterations;
      props.maximumIterations = 4;
    }
  }
});

for (const computePredecessors of [true, false])
  it(`R9 service-area backends run only requested tie work (${computePredecessors})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUNetworkServiceAreasOperation, CPUNetworkServiceAreasOperation]) {
      let operation = new Operation({
        nodeCount: 4,
        facilityCapacity: 1,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        facilities: 'facilities',
        maximumIterations: 5,
        maximumLabelIterations: 5,
        maximumTieIterations: 1,
        localIterations: 1,
        output: {
          assignments: 'assignments',
          predecessors: computePredecessors ? 'parents' : undefined,
          facilityNodeCounts: 'counts',
          status: {converged: 'converged', iterationLimitReached: 'limit'}
        }
      });
      if (operation instanceof GPUNetworkServiceAreasOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 3, 3)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2, 3)},
          weights: {format: 'float32', values: Float32Array.of(0, 0, 0)},
          facilities: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [operation],
        outputs: [
          'assignments',
          'counts',
          'converged',
          'limit',
          ...(computePredecessors ? ['parents'] : [])
        ]
      });
      // One tie hop cannot supply parents for nodes 2 and 3; label work is complete.
      expect(Array.from(rows['assignments'])).toEqual([0, 0, 0, 0]);
      expect(Array.from(rows['counts'])).toEqual([4]);
      if (computePredecessors)
        expect(Array.from(rows['parents'])).toEqual([0xffffffff, 0, 0xffffffff, 0xffffffff]);
      expect([rows['converged'][0], rows['limit'][0]]).toEqual(
        computePredecessors ? [0, 1] : [1, 0]
      );
    }
  });

for (const phase of ['label', 'tie', 'complete'] as const)
  it(`R1 service-area operation binds the ${phase} phase independently`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const nodeCount = phase === 'label' ? 4 : 2;
    for (const Operation of [GPUNetworkServiceAreasOperation, CPUNetworkServiceAreasOperation]) {
      let operation = new Operation({
        nodeCount,
        facilityCapacity: 1,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        facilities: 'facilities',
        maximumIterations: 5,
        maximumLabelIterations: phase === 'label' ? 1 : 4,
        maximumTieIterations: phase === 'tie' ? 1 : 2,
        localIterations: 1,
        output: {
          assignments: 'assignments',
          costs: 'costs',
          predecessors: 'parents',
          facilityNodeCounts: 'counts',
          status: {converged: 'converged', iterationLimitReached: 'limit'}
        }
      });
      if (operation instanceof GPUNetworkServiceAreasOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {
            format: 'uint32',
            values: phase === 'label' ? Uint32Array.of(0, 1, 2, 3, 3) : Uint32Array.of(0, 1, 2)
          },
          neighbors: {
            format: 'uint32',
            values: phase === 'label' ? Uint32Array.of(1, 2, 3) : Uint32Array.of(1, 0)
          },
          weights: {
            format: 'float32',
            values: phase === 'label' ? Float32Array.of(1, 1, 1) : Float32Array.of(0, 0)
          },
          facilities: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [operation],
        outputs: ['costs', 'assignments', 'parents', 'counts', 'converged', 'limit']
      });
      // R8: Label budget one leaves nodes 2 and 3 unassigned after costs settled.
      // Neither may get a parent just because their NONE labels match. Tie
      // budget one leaves the cycle's outgoing edge pending despite full parents.
      expect(Array.from(rows['costs'])).toEqual(phase === 'label' ? [0, 1, 2, 3] : [0, 0]);
      expect(Array.from(rows['assignments'])).toEqual(
        phase === 'label' ? [0, 0, 0xffffffff, 0xffffffff] : [0, 0]
      );
      expect(Array.from(rows['counts'])).toEqual([2]);
      expect(Array.from(rows['parents'])).toEqual(
        phase === 'label' ? [0xffffffff, 0, 0xffffffff, 0xffffffff] : [0xffffffff, 0]
      );
      expect([rows['converged'][0], rows['limit'][0]]).toEqual(
        phase === 'complete' ? [1, 0] : [0, 1]
      );
    }
  });

for (const maximumTieIterations of [1, 2])
  it(`R2 both reachability backends distinguish pending cycle work (${maximumTieIterations})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUNetworkReachabilityOperation, CPUNetworkReachabilityOperation]) {
      let operation = new Operation({
        nodeCount: 2,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        maximumIterations: 4,
        localIterations: 1,
        maximumTieIterations,
        output: {
          costs: 'costs',
          predecessors: 'parents',
          unresolvedCount: 'unresolved',
          status: {converged: 'converged', iterationLimitReached: 'limit'}
        }
      });
      if (operation instanceof GPUNetworkReachabilityOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
          weights: {format: 'float32', values: Float32Array.of(0, 0)},
          starts: {format: 'uint32', values: Uint32Array.of(0)}
        },
        operations: [operation],
        outputs: ['parents', 'unresolved', 'converged', 'limit']
      });
      // Node 1 has parent 0 after one hop; its outgoing 1->0 tie remains pending.
      expect(Array.from(rows['parents'])).toEqual([0xffffffff, 0]);
      expect([rows['unresolved'][0], rows['converged'][0], rows['limit'][0]]).toEqual(
        maximumTieIterations === 1 ? [0, 0, 1] : [0, 1, 0]
      );
    }
  });

for (const scene of ['round-limit', 'tie-limit', 'complete', 'zero-active'] as const)
  it(`pins every poisoned reachability stopping and unresolved word (${scene})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Chain 0->1->2: one local hop cannot reach node 2. After zero costs
    // settle, one tie sweep leaves node 2 without a certified predecessor.
    // Four rounds/sweeps settle both links. Zero active rounds execute none.
    for (const Operation of [GPUNetworkReachabilityOperation, CPUNetworkReachabilityOperation]) {
      let operation = new Operation({
        nodeCount: 3,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        startNodes: 'starts',
        activeIterations: 'active',
        maximumIterations: scene === 'round-limit' ? 1 : 4,
        localIterations: 1,
        maximumTieIterations: scene === 'tie-limit' ? 1 : 4,
        output: {
          costs: 'costs',
          predecessors: 'parents',
          unresolvedCount: 'unresolved',
          status: {converged: 'converged', iterationLimitReached: 'limit', iterationCount: 'rounds'}
        }
      });
      if (operation instanceof GPUNetworkReachabilityOperation)
        operation = poisonOperationOutputs(operation);
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(0, 0)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(scene === 'zero-active' ? 0 : 4)}
        },
        operations: [operation],
        outputs: ['unresolved', 'converged', 'limit', 'rounds']
      });
      expect(Array.from(rows['unresolved'])).toEqual([scene === 'tie-limit' ? 1 : 0]);
      expect(Array.from(rows['converged'])).toEqual([scene === 'complete' ? 1 : 0]);
      expect(Array.from(rows['limit'])).toEqual([scene === 'complete' ? 0 : 1]);
      if (scene === 'round-limit' || scene === 'zero-active')
        expect(Array.from(rows['rounds'])).toEqual([scene === 'round-limit' ? 1 : 0]);
    }
  });

it('writes poisoned exact service-area and structure status scalars', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1, 2, 2)},
    neighbors: {format: 'uint32' as const, values: Uint32Array.of(1, 2)},
    weights: {format: 'float32' as const, values: Float32Array.of(1, 1)},
    facilities: {format: 'uint32' as const, values: Uint32Array.of(0)}
  };
  // One finite chain, one facility: all three nodes assigned; no pending cost,
  // label or tie remains after four rounds. Exact SCC traversal runs once and
  // yields three singleton components, regardless of the parallel step budget.
  for (const Operation of [GPUNetworkServiceAreasOperation, CPUNetworkServiceAreasOperation]) {
    let operation = new Operation({
      nodeCount: 3,
      facilityCapacity: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      facilities: 'facilities',
      maximumIterations: 4,
      maximumLabelIterations: 4,
      maximumTieIterations: 4,
      localIterations: 1,
      output: {
        costs: 'costs',
        assignments: 'assignments',
        facilityNodeCounts: 'counts',
        status: {converged: 'converged', iterationLimitReached: 'limit'}
      }
    });
    if (operation instanceof GPUNetworkServiceAreasOperation)
      operation = poisonOperationOutputs(operation);
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [operation],
      outputs: ['counts', 'converged', 'limit']
    });
    expect([rows['counts'][0], rows['converged'][0], rows['limit'][0]]).toEqual([3, 1, 0]);
  }
  for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation]) {
    let operation = new Operation({
      nodeCount: 3,
      edgeCount: 2,
      mode: 'exact',
      strongComponentsMaximumIterations: 1,
      topology: {offsets: 'offsets', neighbors: 'neighbors'},
      output: {strongComponents: 'labels'},
      strongComponentsStatus: {
        converged: 'converged',
        iterationCount: 'rounds',
        iterationLimitReached: 'limit'
      }
    });
    if (operation instanceof GPUNetworkStructureOperation)
      operation = poisonOperationOutputs(operation);
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [operation],
      outputs: ['converged', 'rounds', 'limit']
    });
    expect([rows['converged'][0], rows['rounds'][0], rows['limit'][0]]).toEqual([1, 1, 0]);
  }
});
