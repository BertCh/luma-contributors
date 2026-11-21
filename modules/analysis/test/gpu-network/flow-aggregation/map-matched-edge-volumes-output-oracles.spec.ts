// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMapMatchedEdgeVolumes,
  type GPUFlowAggregationOutput
} from '../../../src/gpu-network/flow-aggregation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from '../adjacency-matrix/network-output-oracle-test-utils';

it('edge-volume transition status separates top-list overflow from pair-table overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two distinct movements compete for either one top-list row or one hash-table slot.
  for (const transitionPairCapacity of [8, 1]) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const transitions = {
      ids: output('ids', 'uint32'),
      status: {
        count: output('count', 'uint32'),
        overflow: output('overflow', 'uint32'),
        requiredCount: output('required', 'uint32'),
        pairOverflow: output('pair-overflow', 'uint32')
      }
    };
    const edgeVolumes = output('volumes', 'uint32', 8);
    graph.add(
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: input('edges', 'uint32', [2, 5, 7]),
          counts: input('counts', 'uint32', [3]),
          completeness: 'known-complete'
        },
        trackCount: 1,
        traversedEdgeCapacity: 3,
        edgeCount: 8,
        transitionPairCapacity,
        output: {edgeVolumes, status: {candidateOverflow: output('loss', 'uint32')}, transitions}
      })
    );
    try {
      context.run();
      expect(await read(edgeVolumes)).toEqual([0, 0, 1, 0, 0, 1, 0, 1]);
      expect(await read(transitions.status.count)).toEqual([1]);
      expect(await read(transitions.status.requiredCount)).toEqual([
        transitionPairCapacity === 8 ? 2 : 1
      ]);
      expect(await read(transitions.status.overflow)).toEqual([
        transitionPairCapacity === 8 ? 1 : 0
      ]);
      expect(await read(transitions.status.pairOverflow)).toEqual([
        transitionPairCapacity === 1 ? 1 : 0
      ]);
      // Which pair wins a full one-slot table is intentionally unspecified.
      if (transitionPairCapacity === 8) expect(await read(transitions.ids)).toEqual([21]);
    } finally {
      context.destroy();
    }
  }
});
const OPTIONAL_OUTPUTS = [
  'flowOriginZoneIds',
  'flowDestinationZoneIds',
  'flowWeights',
  'drawInstanceCount',
  'zoneOutCountExtent',
  'zoneInCountExtent',
  'pairOverflow',
  'requiredCount'
] as const;

it('edge-volume transitions distinguish incoming and outgoing extents repeated weights and clamped draws', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Four two-edge tracks: 2->5 twice, 2->7 once, 3->7 once. Outgoing counts
  // are {2:3,3:1}; incoming counts are {5:2,7:2}. Three distinct pairs compete
  // for two top rows, ranked (2->5,count2), (2->7,count1), then (3->7,count1).
  for (const selected of [
    undefined,
    'zoneOutCountExtent',
    'zoneInCountExtent',
    'flowWeights',
    'drawInstanceCount',
    'requiredCount'
  ] as const) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const transitions: GPUFlowAggregationOutput = {
      ids: output('ids', 'uint32', 2),
      status: {count: output('count', 'uint32'), overflow: output('overflow', 'uint32')}
    };
    const optional = selected
      ? output(
          'optional',
          selected === 'drawInstanceCount' || selected === 'requiredCount' ? 'uint32' : 'float32',
          selected === 'drawInstanceCount' || selected === 'requiredCount' ? 1 : 2
        )
      : undefined;
    if (selected && optional) {
      if (selected === 'requiredCount') transitions.status.requiredCount = optional as never;
      else Object.assign(transitions, {[selected]: optional});
    }
    const edgeVolumes = output('volumes', 'uint32', 8);
    graph.add(
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: input('edges', 'uint32', [2, 5, 2, 5, 2, 7, 3, 7]),
          counts: input('counts', 'uint32', [2, 2, 2, 2]),
          completeness: 'known-complete'
        },
        trackCount: 4,
        traversedEdgeCapacity: 2,
        edgeCount: 8,
        transitionPairCapacity: 8,
        output: {edgeVolumes, status: {candidateOverflow: output('loss', 'uint32')}, transitions}
      })
    );
    try {
      context.run();
      expect(await read(edgeVolumes)).toEqual([0, 0, 3, 1, 0, 2, 0, 2]);
      expect(await read(transitions.ids)).toEqual([21, 23]);
      expect(await read(transitions.status.count)).toEqual([2]);
      expect(await read(transitions.status.overflow)).toEqual([1]);
      if (optional)
        expect(await read(optional)).toEqual(
          selected === 'zoneOutCountExtent'
            ? [1, 3]
            : selected === 'zoneInCountExtent'
              ? [2, 2]
              : selected === 'flowWeights'
                ? [2, 1]
                : selected === 'requiredCount'
                  ? [3]
                  : [2]
        );
    } finally {
      context.destroy();
    }
  }
});

it('edge-volume transitions have hand-counted pairs with each optional view alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // [2,5,7] has movements 2->5 and 5->7; pair keys at eight zones are 21 and 47.
  // Each occupied zone has count one, so both nonzero extents are [1,1].
  for (const selected of [undefined, ...OPTIONAL_OUTPUTS]) {
    const context = createOracleContext(device);
    const {graph, input, output, read} = context;
    const edgeVolumes = output('volumes', 'uint32', 8);
    const loss = output('loss', 'uint32');
    const transitions: GPUFlowAggregationOutput = {
      ids: output('ids', 'uint32', 3),
      status: {
        count: output('count', 'uint32'),
        overflow: output('overflow', 'uint32')
      }
    };
    const optional = selected
      ? output(
          'optional',
          selected === 'flowWeights' || selected.includes('Extent') ? 'float32' : 'uint32',
          selected.includes('Extent')
            ? 2
            : selected.includes('ZoneIds') || selected === 'flowWeights'
              ? 3
              : 1
        )
      : undefined;
    if (selected && optional) {
      if (selected === 'pairOverflow' || selected === 'requiredCount')
        transitions.status[selected] = optional as never;
      else Object.assign(transitions, {[selected]: optional});
    }
    graph.add(
      new GPUMapMatchedEdgeVolumes({
        traversedEdges: {
          edges: input('edges', 'uint32', [2, 5, 7]),
          counts: input('counts', 'uint32', [3]),
          completeness: 'known-complete'
        },
        trackCount: 1,
        traversedEdgeCapacity: 3,
        edgeCount: 8,
        transitionPairCapacity: 8,
        output: {edgeVolumes, status: {candidateOverflow: loss}, transitions}
      })
    );
    try {
      context.run();
      expect(await read(edgeVolumes)).toEqual([0, 0, 1, 0, 0, 1, 0, 1]);
      expect(await read(loss)).toEqual([0]);
      expect(await read(transitions.ids)).toEqual([21, 47, 4294967295]);
      expect(await read(transitions.status.count)).toEqual([2]);
      expect(await read(transitions.status.overflow)).toEqual([0]);
      if (optional)
        expect(await read(optional)).toEqual(
          selected === 'flowOriginZoneIds'
            ? [2, 5, 4294967295]
            : selected === 'flowDestinationZoneIds'
              ? [5, 7, 4294967295]
              : selected === 'flowWeights'
                ? [1, 1, 0]
                : selected === 'drawInstanceCount' || selected === 'requiredCount'
                  ? [2]
                  : selected === 'pairOverflow'
                    ? [0]
                    : [1, 1]
        );
    } finally {
      context.destroy();
    }
  }
  // Omitting the entire transition group preserves the edge volumes.
  const context = createOracleContext(device);
  const edgeVolumes = context.output('volumes', 'uint32', 8);
  const loss = context.output('loss', 'uint32');
  context.graph.add(
    new GPUMapMatchedEdgeVolumes({
      traversedEdges: {
        edges: context.input('edges', 'uint32', [2, 5, 7]),
        counts: context.input('counts', 'uint32', [3]),
        completeness: 'known-complete'
      },
      trackCount: 1,
      traversedEdgeCapacity: 3,
      edgeCount: 8,
      output: {edgeVolumes, status: {candidateOverflow: loss}}
    })
  );
  try {
    context.run();
    expect(await context.read(edgeVolumes)).toEqual([0, 0, 1, 0, 0, 1, 0, 1]);
    expect(await context.read(loss)).toEqual([0]);
  } finally {
    context.destroy();
  }
});
