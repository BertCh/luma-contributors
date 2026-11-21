// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUFlowAggregation} from '../../../src/gpu-network/flow-aggregation/gpu-flow-aggregation';
import {GPUNetworkCoarsening} from '../../../src/gpu-network/network-coarsening/gpu-network-coarsening';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';

it('NET-125 facility captures controls and destinations while demand contents stay live', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const demand = harness.input('float32', Float32Array.of(1, 1));
  const summary = harness.output('uint32', 4);
  const invalidCount = harness.output('uint32', 1);
  const assignments = harness.output('uint32', 2);
  const props = {
    operation: 'capacitated-p-median' as const,
    facilityCount: 2,
    selectCount: 2,
    maximumCandidates: 4,
    costs: harness.input('float32', Float32Array.of(1, 2, 1, Infinity)),
    demand,
    capacities: harness.input('float32', Float32Array.of(1, 1)),
    threshold: harness.input('float32', Float32Array.of(0)),
    output: {
      selectedFacilities: harness.output('uint32', 2),
      assignments,
      objective: harness.output('float32', 1),
      summary,
      status: {invalidCount}
    }
  };
  const contributor = new GPUNetworkFacilityOptimization(props);
  props.facilityCount = 16;
  props.maximumCandidates = 0;
  props.output.summary = harness.output('uint32', 4);
  props.output.status.invalidCount = harness.output('uint32', 1);
  const read = await harness.run(contributor, first => {
    // Only assignment [1,0] serves both unit demands within the two unit capacities.
    expect(first(summary)).toEqual([1, 1, 1, 4]);
    expect(first(assignments)).toEqual([1, 0]);
    expect(first(invalidCount)).toEqual([0]);
    harness.rewrite(demand, Float32Array.of(-1, 1));
  });
  expect(read(summary)).toEqual([0, 0, 0, 0]);
  expect(read(invalidCount)).toEqual([1]);
  expect(read(assignments)).toEqual([0xffffffff, 0xffffffff]);
});

it('NET-125 aggregation captures zone and compact maps while zone-ID contents stay live', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const origins = harness.input('uint32', Uint32Array.of(0, 0, 1));
  const destinations = harness.input('uint32', Uint32Array.of(1, 1, 2));
  const ids = harness.output('uint32', 3);
  const count = harness.output('uint32', 1);
  const flowCounts = harness.output('uint32', 3);
  const props = {
    zones: {kind: 'ids' as const, zoneCount: 4},
    pairCapacity: 8,
    originZoneIds: origins,
    destinationZoneIds: destinations,
    output: {ids, flowCounts, status: {count, overflow: harness.output('uint32', 1)}}
  };
  const contributor = new GPUFlowAggregation(props);
  props.zones.zoneCount = 64;
  props.pairCapacity = 2;
  props.output.ids = harness.output('uint32', 3);
  props.output.status.count = harness.output('uint32', 1);
  const read = await harness.run(contributor, first => {
    // Packed IDs use origin * 4 + destination; ranking uses the literal pair counts 2 and 1.
    expect(first(ids)).toEqual([1, 6, 0xffffffff]);
    expect(first(flowCounts)).toEqual([2, 1, 0]);
    expect(first(count)).toEqual([2]);
    harness.rewrite(origins, Uint32Array.of(2, 2, 2));
    harness.rewrite(destinations, Uint32Array.of(3, 3, 3));
  });
  expect(read(ids)).toEqual([11, 0xffffffff, 0xffffffff]);
  expect(read(flowCounts)).toEqual([3, 0, 0]);
  expect(read(count)).toEqual([1]);
});

it('NET-125 coarsening captures CSR and optional writers while labels stay live', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createOptimizationHarness(device);
  const labels = harness.input('uint32', Uint32Array.of(0, 0, 1));
  const groupNodeCount = harness.output('uint32', 2);
  const ids = harness.output('uint32', 3);
  const count = harness.output('uint32', 1);
  const edgeCounts = harness.output('uint32', 3);
  const props = {
    directed: true,
    groupCapacity: 2,
    labels,
    topology: {
      offsets: harness.input('uint32', Uint32Array.of(0, 1, 2, 2)),
      neighbors: harness.input('uint32', Uint32Array.of(1, 2))
    },
    output: {
      groupNodeCount,
      edges: {ids, status: {count, overflow: harness.output('uint32', 1)}},
      edgeTargetGroups: harness.output('uint32', 3),
      edgeCounts
    }
  };
  const contributor = new GPUNetworkCoarsening(props);
  props.groupCapacity = 128;
  props.topology.neighbors = harness.input('uint32', Uint32Array.of(0));
  props.output.edges.ids = harness.output('uint32', 3);
  props.output.edges.status.count = harness.output('uint32', 1);
  const read = await harness.run(contributor, first => {
    // Chain 0->1->2: labels [0,0,1] leave only edge row 1 between groups.
    expect(first(groupNodeCount)).toEqual([2, 1]);
    expect(first(count)).toEqual([1]);
    expect(first(edgeCounts)).toEqual([1, 0, 0]);
    harness.rewrite(labels, Uint32Array.of(1, 1, 1));
  });
  expect(read(groupNodeCount)).toEqual([0, 3]);
  expect(read(count)).toEqual([0]);
  expect(read(edgeCounts)).toEqual([0, 0, 0]);
});
