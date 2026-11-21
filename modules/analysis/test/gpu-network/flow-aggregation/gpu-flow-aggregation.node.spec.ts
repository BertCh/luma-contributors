// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getResourceBuffer} from '../network-resource-buffers';
import {Buffer, type ComputePass, type CommandEncoder} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getForbiddenFlowWGSLIdentifiers} from '../flow-aggregation/flow-wgsl-review';
import {
  getGPUFlowPairKey,
  getGPUFlowPairZones,
  GPUFlowAggregation,
  type GPUFlowAggregationOutput,
  type GPUFlowAggregationProps
} from '../../../src/gpu-network/flow-aggregation/index';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  createVectorView,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

let defaultViewSerial = 0;

type PropsOverrides = Omit<Partial<GPUFlowAggregationProps>, 'output'> & {
  output?: Partial<GPUFlowAggregationOutput>;
};

function createProps(
  graph: GPUCommandGraph,
  overrides: PropsOverrides = {}
): GPUFlowAggregationProps {
  const createDefault = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `default-${defaultViewSerial++}`, format, length);
  return {
    zones: {kind: 'ids', zoneCount: 4},
    originZoneIds: createDefault('uint32', 10),
    destinationZoneIds: createDefault('uint32', 10),
    pairCapacity: 8,
    ...overrides,
    output: {
      ids: createDefault('uint32', 4),
      status: {
        count: createDefault('uint32', 1),
        overflow: createDefault('uint32', 1)
      },
      ...overrides.output
    }
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => PropsOverrides,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUFlowAggregation(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUFlowAggregation pair keys round trip', () => {
  for (const [origin, destination, zoneCount] of [
    [0, 0, 1],
    [2, 3, 4],
    [65534, 65534, 65535],
    [12, 40000, 65535]
  ]) {
    const key = getGPUFlowPairKey(origin, destination, zoneCount);
    expect(key).toBeLessThan(0xffffffff);
    expect(getGPUFlowPairZones(key, zoneCount)).toEqual([origin, destination]);
  }
});

it('GPUFlowAggregation rejects invalid properties', () => {
  expectThrows(() => ({pairCapacity: 6}), /pairCapacity must be a power of two/);
  expectThrows(() => ({pairCapacity: 0}), /pairCapacity must be a power of two/);
  expectThrows(() => ({maximumProbeCount: 9}), /maximumProbeCount/);
  expectThrows(() => ({zones: {kind: 'ids', zoneCount: 0xffffffff}}), /zone count/);
  expectThrows(() => ({zones: {kind: 'ids', zoneCount: 0}}), /zone count/);
  expectThrows(
    () => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [65536, 65536]}
    }),
    /zone count/
  );
  expectThrows(() => ({originZoneIds: undefined}), /ids zones require/);
  expectThrows(
    graph => ({
      origins: createTransientView(graph, 'origins', 'float32x2', 10)
    }),
    /do not accept origins/
  );
  expectThrows(
    graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10)
    }),
    /require origins and destinations/
  );
  expectThrows(
    graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10)
    }),
    /do not accept zone IDs/
  );
  expectThrows(
    graph => ({
      destinationZoneIds: createTransientView(graph, 'short', 'uint32', 9)
    }),
    /destinationZoneIds length/
  );
  expectThrows(
    graph => ({
      zones: {
        kind: 'hexagon',
        bounds: [0, 0, 1, 1],
        gridSize: [2, 2],
        radius: -1
      },
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10)
    }),
    /radius/
  );
  expectThrows(
    graph => ({
      output: {zoneOutWeights: createTransientView(graph, 'zone-out-weights', 'float32', 4)}
    }),
    /output\.zoneOutWeights requires weights/
  );
  expectThrows(
    graph => ({
      output: {zoneOutCounts: createTransientView(graph, 'zone-out-counts', 'uint32', 3)}
    }),
    /output\.zoneOutCounts length/
  );
  expectThrows(
    graph => ({
      output: {flowCounts: createTransientView(graph, 'flow-counts', 'uint32', 5)}
    }),
    /output\.flowCounts length/
  );
  expectThrows(
    graph => ({
      output: {flowWeights: createTransientView(graph, 'flow-weights', 'float32', 3)}
    }),
    /output\.flowWeights length/
  );
  expectThrows(
    graph => ({
      weights: createTransientView(graph, 'weights', 'float32', 9)
    }),
    /weights length/
  );
  expectThrows(
    graph => ({
      output: {
        ids: createTransientView(graph, 'empty-ids', 'uint32', 0),
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }),
    /at least one row/
  );
});

it('GPUFlowAggregation rejects outputs that share buffers with inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shared = createTransientView(graph, 'shared', 'uint32', 4);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          originZoneIds: createTransientView(graph, 'origin-ids', 'uint32', 4),
          destinationZoneIds: shared,
          output: {
            ids: shared,
            status: {
              count: createTransientView(graph, 'count', 'uint32', 1),
              overflow: createTransientView(graph, 'overflow', 'uint32', 1)
            }
          }
        })
      )
  ).toThrow(/must not share buffers/);
  device.destroy();
});

it('GPUFlowAggregation rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const otherGraph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(createProps(graph));
  expect(() => contributor.getCommandNodes(otherGraph)).toThrow(/target graph/);
  device.destroy();
});

it('GPUFlowAggregation schedules ids zones in order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUFlowAggregation(
    createProps(graph, {
      sumOrder: 'atomic',
      weights: createTransientView(graph, 'weights', 'float32', 10),
      output: {
        flowCounts: createTransientView(graph, 'flow-counts', 'uint32', 4),
        flowOriginZoneIds: createTransientView(graph, 'flow-origin', 'uint32', 4),
        zoneOutCounts: createTransientView(graph, 'zone-out-counts', 'uint32', 4),
        zoneInWeights: createTransientView(graph, 'zone-in-weights', 'float32', 4)
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.every(id => id.startsWith('flow-aggregation-'))).toBe(true);
  expect(new Set(ids).size).toBe(ids.length);
  const prefixes = [
    'flow-aggregation-pair-keys',
    'flow-aggregation-pair-index',
    'flow-aggregation-pair-query',
    'flow-aggregation-pair-counts',
    // The slot order by pair is built before the pair sums, which read it (review B-18).
    'flow-aggregation-slot-keys',
    'flow-aggregation-sort-pairs',
    'flow-aggregation-pair-weights',
    'flow-aggregation-weight-keys',
    'flow-aggregation-sort-weights',
    'flow-aggregation-gather-ids',
    'flow-aggregation-gather-values',
    'flow-aggregation-decode-zones',
    'flow-aggregation-pair-overflow',
    'flow-aggregation-publish',
    'flow-aggregation-zone-out-counts',
    'flow-aggregation-zone-in-weights'
  ];
  let cursor = -1;
  for (const prefix of prefixes) {
    const position = ids.findIndex((id, index) => index > cursor && id.startsWith(prefix));
    expect(position, prefix).toBeGreaterThan(cursor);
    cursor = position;
  }
  expect(ids.some(id => id.includes('time-classify'))).toBe(false);
  expect(ids.some(id => id.includes('-origin-zones'))).toBe(false);
  device.destroy();
});

it('GPUFlowAggregation schedules grid zone and time nodes first', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUFlowAggregation(
    createProps(graph, {
      id: 'flows',
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10),
      timeWindow: {
        timestamps: createTransientView(graph, 'timestamps', 'float32', 10),
        window: createTransientView(graph, 'window', 'float32', 8)
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.slice(0, 6)).toEqual([
    'flows-origin-bases-upload',
    'flows-origin-zones',
    'flows-destination-bases-upload',
    'flows-destination-zones',
    'flows-time-classify',
    'flows-pair-keys'
  ]);
  expect(ids.every(id => id.startsWith('flows-'))).toBe(true);
  device.destroy();
});

it('GPUFlowAggregation validates word time windows and sumOrder', () => {
  const timeWindow = (
    graph: GPUCommandGraph,
    overrides: Partial<GPUFlowAggregationProps['timeWindow'] & object>
  ): GPUFlowAggregationProps['timeWindow'] => ({
    timestamps: createTransientView(graph, 'word-timestamps', 'uint32x2', 10),
    window: createTransientView(graph, 'word-window', 'uint32', 8),
    ...overrides
  });
  expectThrows(() => ({sumOrder: 'fast' as 'sorted'}), /sumOrder must be atomic or sorted/);
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        window: createTransientView(graph, 'float-window', 'float32', 8)
      })
    }),
    /word window/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        window: createTransientView(graph, 'short-word-window', 'uint32', 7)
      })
    }),
    /word window must hold 8 uint32/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        timestampsLow: createTransientView(graph, 'low', 'float32', 10)
      })
    }),
    /timestampsLow requires float32 timestamps/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        timestamps: createTransientView(graph, 'short-word-timestamps', 'uint32x2', 9)
      })
    }),
    /timestamps length/
  );
  expectThrows(
    graph => ({
      timeWindow: {
        timestamps: createTransientView(graph, 'float-timestamps', 'float32', 10),
        window: createTransientView(graph, 'uint-window', 'uint32', 8)
      }
    }),
    /window/
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      timeWindow: timeWindow(graph, {}),
      sumOrder: 'sorted'
    })
  );
  expect(contributor.sumOrder).toBe('sorted');
  expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  const defaultGraph = new GPUCommandGraph(device);
  const defaultContributor = new GPUFlowAggregation(createProps(defaultGraph, {}));
  expect(defaultContributor.sumOrder).toBe('sorted');
  device.destroy();
});

it('GPUFlowAggregation sorted sums add sort, scan, and gather nodes', () => {
  const countNodes = (sumOrder: 'atomic' | 'sorted') => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const createDefault = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${sumOrder}-${name}`, format, length);
    const contributor = new GPUFlowAggregation({
      sumOrder,
      zones: {kind: 'ids', zoneCount: 4},
      originZoneIds: createDefault('origins', 'uint32', 10),
      destinationZoneIds: createDefault('destinations', 'uint32', 10),
      weights: createDefault('weights', 'float32', 10),
      pairCapacity: 16,
      output: {
        ids: createDefault('ids', 'uint32', 4),
        status: {
          count: createDefault('count', 'uint32', 1),
          overflow: createDefault('overflow', 'uint32', 1)
        },
        flowWeights: createDefault('flow-weights', 'float32', 4),
        zoneOutWeights: createDefault('zone-out-weights', 'float32', 4),
        zoneInWeights: createDefault('zone-in-weights', 'float32', 4)
      }
    });
    const count = contributor.getCommandNodes(graph).length;
    device.destroy();
    return count;
  };
  const atomicCount = countNodes('atomic');
  const sortedCount = countNodes('sorted');
  expect(sortedCount).toBeGreaterThan(atomicCount);
  console.log(
    `flow-aggregation nodes (3 weight sums): atomic=${atomicCount} sorted=${sortedCount}`
  );
});

it('GPUFlowAggregation sorted pair sums dispatch one workgroup per stored pair', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUFlowAggregation(
    createProps(graph, {
      weights: createTransientView(graph, 'weights', 'float32', 10),
      output: {flowWeights: createTransientView(graph, 'flow-weights', 'float32', 4)}
    })
  ).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Review B-18: the pair sum is gated by an indirect dispatch written from the stored-pair count,
  // so empty table slots launch no 256-lane workgroups.
  const dispatch = ids.indexOf('flow-aggregation-pair-sum-dispatch');
  const pairSum = ids.indexOf('flow-aggregation-pair-sum');
  expect(dispatch).toBeGreaterThanOrEqual(0);
  expect(dispatch).toBeLessThan(pairSum);
  expect(ids.findIndex(id => id.startsWith('flow-aggregation-sort-pairs'))).toBeLessThan(pairSum);
  expect(nodes[pairSum].condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect(getParsedKernelSources(nodes).get('flow-aggregation-pair-sum')).toContain(
    'slotsByPair[slotsByPairOffset + pairIndex]'
  );
  // Rule 16: a capacity whose 256-lane dispatch cannot fit in uint32 fails at construction.
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          id: 'too-large',
          pairCapacity: 2 ** 25,
          weights: createTransientView(graph, 'weights-2', 'float32', 10)
        })
      )
  ).toThrow(/at most 2\^23/);
  device.destroy();
});

it('GPUFlowAggregation accepts uint32 zone columns beyond the packed-key limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      zones: {kind: 'ids', zoneCount: 100000},
      weights: createTransientView(graph, 'large-weights', 'float32', 10),
      output: {
        flowOriginZoneIds: createTransientView(graph, 'large-origins', 'uint32', 4),
        flowDestinationZoneIds: createTransientView(graph, 'large-destinations', 'uint32', 4),
        flowWeights: createTransientView(graph, 'large-sums', 'float32', 4)
      }
    })
  );
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('flow-aggregation-pair-keys')).not.toContain('origin * ZONE_COUNT');
  expect(sources.get('flow-aggregation-decode-zones')).toContain('sourceOrigins[');
  for (const source of sources.values()) {
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
    expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
  }
  device.destroy();
});

it('GPUFlowAggregation rejects overlapping output fields and status words', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shared = createTransientView(graph, 'overlapping-outputs', 'uint32', 4);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          output: {ids: shared, flowCounts: shared}
        })
      )
  ).toThrow(/writable byte ranges/);
  const status = createTransientView(graph, 'overlapping-status', 'uint32', 1);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          output: {status: {count: status, overflow: status}}
        })
      )
  ).toThrow(/writable byte ranges/);
  device.destroy();
});

it('GPUFlowAggregation rejects aligned binding ranges and derived scratch before graph mutation', () => {
  for (const failure of ['aligned-input', 'zone-offsets', 'pair-scratch'] as const) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'unaligned-flow-input', 'uint32', 257);
    const overrides: PropsOverrides =
      failure === 'aligned-input'
        ? {
            originZoneIds: graph.createDataView(backing.buffer, {
              format: 'uint32',
              byteOffset: 4,
              length: 256
            }),
            destinationZoneIds: createTransientView(graph, 'aligned-destinations', 'uint32', 256)
          }
        : failure === 'zone-offsets'
          ? {
              zones: {kind: 'ids', zoneCount: 256},
              weights: createTransientView(graph, 'offset-weights', 'float32', 10),
              output: {zoneOutWeights: createTransientView(graph, 'zone-sums', 'float32', 256)}
            }
          : {pairCapacity: 512, maximumProbeCount: 1};
    const contributor = new GPUFlowAggregation(createProps(graph, overrides));
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declareView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/flow-aggregation.*binding/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declareView).not.toHaveBeenCalled();
    device.destroy();
  }
});

it('flow chunk shader and pipeline counts stay constant as chunk count grows', () => {
  for (const chunkCount of [2, 8]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const positions = Array.from({length: chunkCount}, (_, index) => {
      const storage = createTransientView(
        graph,
        `position-storage-${index}`,
        'float32x2',
        index + 5
      );
      return graph.createDataView(storage.buffer, {
        format: 'float32x2',
        byteOffset: index * 8,
        length: (index % 2) + 2
      });
    });
    const weightChunks = positions.map((view, index) => {
      const storage = createTransientView(graph, `weight-storage-${index}`, 'float32', index + 5);
      return graph.createDataView(storage.buffer, {
        format: 'float32',
        byteOffset: index * 4,
        length: view.length
      });
    });
    const contributor = new GPUFlowAggregation(
      createProps(graph, {
        zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
        originZoneIds: undefined,
        destinationZoneIds: undefined,
        origins: createVectorView('chunk-origins', 'float32x2', positions),
        destinations: createVectorView('chunk-destinations', 'float32x2', positions),
        weights: createVectorView('chunk-weights', 'float32', weightChunks)
      })
    );
    const nodes = contributor.getCommandNodes(graph);
    for (const pattern of [/origin-zones/, /destination-zones/, /contributions/]) {
      const selected = nodes.filter(node => pattern.test(node.id));
      expect(selected.length).toBe(chunkCount);
      expect(new Set(getParsedKernelSources(selected).values()).size).toBe(1);
      expect(getGPUCommandNodePipelineKeys(selected).size).toBe(1);
    }
    device.destroy();
  }
});

it('flow lexicographic keys declare 30 million rows within default storage binding limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      zones: {kind: 'ids', zoneCount: 100000},
      originZoneIds: createTransientView(graph, 'large-row-origins', 'uint32', 30000000),
      destinationZoneIds: createTransientView(graph, 'large-row-destinations', 'uint32', 30000000),
      maximumProbeCount: 1
    })
  );
  expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
    contributor.estimatedCommandNodeCount
  );
  device.destroy();
});

it('flow rejects adjacent output slices with overlapping aligned bindings and accepts aligned slices', () => {
  for (const stride of [4, 256]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const backing = createTransientView(graph, 'shared-zone-outputs', 'uint32', 65);
    const origin = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    const destination = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: stride,
      length: 1
    });
    const props = createProps(graph, {
      output: {
        ids: createTransientView(graph, 'one-id', 'uint32', 1),
        flowOriginZoneIds: origin,
        flowDestinationZoneIds: destination
      }
    });
    if (stride === 4)
      expect(() => new GPUFlowAggregation(props)).toThrow(/aligned binding byte ranges/);
    else expect(() => graph.add(new GPUFlowAggregation(props))).not.toThrow();
    device.destroy();
  }
});

it('atomic flow sums pack 11000 weight chunks once and stay inside the estimated node budget', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const weight = createTransientView(graph, 'atomic-chunk-weight', 'float32', 1);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      sumOrder: 'atomic',
      zones: {kind: 'ids', zoneCount: 4},
      pairCapacity: 8,
      originZoneIds: createTransientView(graph, 'atomic-chunk-origins', 'uint32', 11000),
      destinationZoneIds: createTransientView(graph, 'atomic-chunk-destinations', 'uint32', 11000),
      weights: createVectorView('atomic-many-weights', 'float32', Array(11000).fill(weight)),
      output: {
        zoneOutWeights: createTransientView(graph, 'atomic-out-weights', 'float32', 4),
        zoneInWeights: createTransientView(graph, 'atomic-in-weights', 'float32', 4)
      }
    })
  );
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.filter(node => node.id.includes('-contributions-chunk-')).length).toBe(11000);
  expect(nodes.filter(node => /-(pair|zone-out|zone-in)-weights-chunk-/.test(node.id))).toEqual([]);
  expect(nodes.length).toBeLessThan(32768);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  device.destroy();
});

it('atomic flow sums reject a chunk configuration above the node budget at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const weight = createTransientView(graph, 'over-budget-weight', 'float32', 1);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          sumOrder: 'atomic',
          originZoneIds: createTransientView(graph, 'over-budget-origins', 'uint32', 33000),
          destinationZoneIds: createTransientView(
            graph,
            'over-budget-destinations',
            'uint32',
            33000
          ),
          weights: createVectorView('over-budget-weights', 'float32', Array(33000).fill(weight)),
          output: {
            zoneOutWeights: createTransientView(graph, 'over-budget-out', 'float32', 4),
            zoneInWeights: createTransientView(graph, 'over-budget-in', 'float32', 4)
          }
        })
      )
  ).toThrow(/flow-aggregation.*command.*32768/);
  device.destroy();
});

it('skewed flow chunks declare their own workgroups and share shader and pipeline constants', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const lengths = [1000000, ...Array(1023).fill(1)];
  const positions = lengths.map((length, index) =>
    createTransientView(graph, `skewed-positions-${index}`, 'float32x2', length)
  );
  const weights = lengths.map((length, index) =>
    createTransientView(graph, `skewed-weights-${index}`, 'float32', length)
  );
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createVectorView('skewed-origins', 'float32x2', positions),
      destinations: createVectorView('skewed-destinations', 'float32x2', positions),
      weights: createVectorView('skewed-weights', 'float32', weights)
    })
  );
  const nodes = contributor.getCommandNodes(graph);
  for (const pattern of [/origin-zones/, /destination-zones/, /contributions/]) {
    const chunks = nodes.filter(node => pattern.test(node.id));
    expect(
      chunks.reduce((total, node) => total + (node.workload?.maximumInvocationCount ?? 0), 0)
    ).toBe(1001023);
    expect(chunks.map(node => node.workload?.maximumWorkgroupCount)).toEqual([
      3907,
      ...Array(1023).fill(1)
    ]);
    const probes = [chunks[0], chunks[1], chunks.at(-1)!];
    expect(new Set(getParsedKernelSources(probes).values()).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(probes).size).toBe(1);
    for (const source of getParsedKernelSources(probes).values())
      expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
  }
  device.destroy();
});

it('the WGSL audit checks the full reserved list and permits sample', () => {
  expect(getForbiddenFlowWGSLIdentifiers('target active filter sample module meta')).toEqual([
    'target',
    'active',
    'filter',
    'module',
    'meta'
  ]);
});

it('flow chunk executables encode each direct dispatch independently of pipeline constants', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.props, '_cachePipelines', {value: false});
  Object.defineProperty(device, 'createComputePipeline', {
    value: () => ({id: 'recorded', hash: '', destroy: () => {}})
  });
  const graph = new GPUCommandGraph(device);
  const weights = [1000000, 1, 300].map((length, index) =>
    createTransientView(graph, `encoded-weights-${index}`, 'float32', length)
  );
  const rows = 1000301;
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      originZoneIds: createTransientView(graph, 'encoded-origins', 'uint32', rows),
      destinationZoneIds: createTransientView(graph, 'encoded-destinations', 'uint32', rows),
      weights: createVectorView('encoded-weight-chunks', 'float32', weights),
      maximumProbeCount: 1
    })
  );
  const nodes = contributor
    .getCommandNodes(graph)
    .filter(node => node.id.includes('-contributions-chunk-'));
  const dispatch = vi.fn();
  const computePass = {
    setPipeline: () => {},
    setBindings: () => {},
    dispatch
  } as unknown as ComputePass;
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.STORAGE});
  for (const node of nodes) {
    if (node.type !== 'compute') throw new Error('expected a contribution compute node');
    const executable = node.compile({device});
    executable.encode({
      computePass,
      parameters: undefined,
      getBuffer: () => buffer,
      commandEncoder: {} as CommandEncoder,
      getTexture: () => {
        throw new Error('no texture');
      },
      getTextureView: () => {
        throw new Error('no texture view');
      },
      getExternalTexture: () => {
        throw new Error('no external texture');
      }
    });
    executable.destroy?.();
  }
  expect(dispatch.mock.calls).toEqual([
    [3907, 1, 1],
    [1, 1, 1],
    [2, 1, 1]
  ]);
  buffer.destroy();
  device.destroy();
});

it('all flow shader and pipeline counts stay constant as unequal weight chunks grow on either sum route', () => {
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const counts: number[][] = [];
    for (const lengths of [
      [15, 1],
      [1, 2, 3, 4, 1, 2, 1, 2]
    ]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const weights = lengths.map((length, index) =>
        createTransientView(graph, `sum-route-weight-${index}`, 'float32', length)
      );
      const contributor = new GPUFlowAggregation(
        createProps(graph, {
          sumOrder,
          originZoneIds: createTransientView(graph, 'sum-route-origins', 'uint32', 16),
          destinationZoneIds: createTransientView(graph, 'sum-route-destinations', 'uint32', 16),
          weights: createVectorView('sum-route-weights', 'float32', weights),
          output: {
            zoneOutWeights: createTransientView(graph, 'sum-route-zone-out', 'float32', 4),
            zoneInWeights: createTransientView(graph, 'sum-route-zone-in', 'float32', 4),
            flowWeights: createTransientView(graph, 'sum-route-flow', 'float32', 4)
          }
        })
      );
      const nodes = contributor.getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      counts.push([new Set(sources.values()).size, getGPUCommandNodePipelineKeys(nodes).size]);
      for (const source of sources.values())
        expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
      device.destroy();
    }
    expect(counts[1]).toEqual(counts[0]);
  }
});

it('every generated flow zone and optional output branch avoids the full W3C reserved list', () => {
  for (const kind of ['grid', 'hexagon'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUFlowAggregation(
      createProps(graph, {
        zones: {
          bounds: createTransientView(graph, 'reserved-bounds', 'float32', 4),
          gridSize: [2, 2],
          activeGridSize: createTransientView(graph, 'reserved-active-grid', 'uint32', 2),
          ...(kind === 'hexagon'
            ? {
                kind: 'hexagon' as const,
                radius: createTransientView(graph, 'reserved-radius', 'float32', 1)
              }
            : {kind: 'grid' as const})
        },
        originZoneIds: undefined,
        destinationZoneIds: undefined,
        origins: createTransientView(graph, 'reserved-positions', 'float32x2', 10),
        destinations: createTransientView(graph, 'reserved-destinations', 'float32x2', 10),
        weights: createTransientView(graph, 'reserved-weights', 'float32', 10),
        mask: createTransientView(graph, 'reserved-mask', 'uint32', 10),
        timeWindow: {
          timestamps: createTransientView(graph, 'reserved-time', 'float32', 10),
          window: createTransientView(graph, 'reserved-window', 'float32', 8)
        },
        output: {
          flowOriginZoneIds: createTransientView(graph, 'reserved-source-output', 'uint32', 4),
          flowDestinationZoneIds: createTransientView(
            graph,
            'reserved-destination-output',
            'uint32',
            4
          ),
          flowCounts: createTransientView(graph, 'reserved-count-output', 'uint32', 4),
          flowWeights: createTransientView(graph, 'reserved-weight-output', 'float32', 4),
          zoneOutCounts: createTransientView(graph, 'reserved-out-counts', 'uint32', 4),
          zoneInCounts: createTransientView(graph, 'reserved-in-counts', 'uint32', 4),
          zoneOutWeights: createTransientView(graph, 'reserved-out-weights', 'float32', 4),
          zoneInWeights: createTransientView(graph, 'reserved-in-weights', 'float32', 4),
          zoneOutCountExtent: createTransientView(graph, 'reserved-out-extent', 'float32', 2),
          zoneInCountExtent: createTransientView(graph, 'reserved-in-extent', 'float32', 2)
        }
      })
    );
    for (const source of getParsedKernelSources(contributor.getCommandNodes(graph)).values()) {
      expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
    }
    device.destroy();
  }
});

it('flow literal bounds and radii reject float32 overflow or radius underflow at construction', () => {
  for (const value of [1e300, -1e300]) {
    expectThrows(
      graph => ({
        zones: {
          kind: 'grid',
          bounds: [Math.min(0, value), 0, Math.max(0, value), 1],
          gridSize: [1, 1]
        },
        originZoneIds: undefined,
        destinationZoneIds: undefined,
        origins: createTransientView(graph, 'literal-origin', 'float32x2', 10),
        destinations: createTransientView(graph, 'literal-destination', 'float32x2', 10)
      }),
      /flow-aggregation bounds literal magnitude/
    );
  }
  for (const radius of [1e300, Number.MIN_VALUE]) {
    expectThrows(
      graph => ({
        zones: {kind: 'hexagon', bounds: [0, 0, 1, 1], gridSize: [1, 1], radius},
        originZoneIds: undefined,
        destinationZoneIds: undefined,
        origins: createTransientView(graph, 'literal-radius-origin', 'float32x2', 10),
        destinations: createTransientView(graph, 'literal-radius-destination', 'float32x2', 10)
      }),
      /flow-aggregation hexagon radius.*float32/
    );
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      zones: {
        kind: 'grid',
        bounds: [-3.4028234663852886e38, 0, 3.4028234663852886e38, 1],
        gridSize: [1, 1]
      },
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'literal-limit-origin', 'float32x2', 10),
      destinations: createTransientView(graph, 'literal-limit-destination', 'float32x2', 10)
    })
  );
  for (const source of getParsedKernelSources(contributor.getCommandNodes(graph)).values()) {
    expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
    expect(source).not.toMatch(/\b(?:NaN|Infinity)\b/);
  }
  device.destroy();
});

it('hexagon axis capacities respect the shared mapper signed cell coordinates', () => {
  for (const gridSize of [
    [2147483521, 1],
    [1, 2147483521]
  ] as const) {
    expectThrows(
      graph => ({
        zones: {kind: 'hexagon', bounds: [0, 0, 1, 1], gridSize, radius: 1},
        originZoneIds: undefined,
        destinationZoneIds: undefined,
        origins: createTransientView(graph, 'signed-cell-origin', 'float32x2', 10),
        destinations: createTransientView(graph, 'signed-cell-destination', 'float32x2', 10)
      }),
      /flow-aggregation hexagon gridSize.*2147483520/
    );
  }
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      zones: {kind: 'hexagon', bounds: [0, 0, 1, 1], gridSize: [2147483520, 1], radius: 1},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'signed-cell-limit-origin', 'float32x2', 10),
      destinations: createTransientView(graph, 'signed-cell-limit-destination', 'float32x2', 10)
    })
  );
  for (const source of getParsedKernelSources(contributor.getCommandNodes(graph)).values()) {
    expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
  }
  device.destroy();
});

it('flow chunk parameter allocations respect maxBufferSize before graph mutation', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 1024});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 8192});
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(graph, 'parameter-weight', 'float32', 1);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      originZoneIds: createTransientView(graph, 'parameter-origins', 'uint32', 5),
      destinationZoneIds: createTransientView(graph, 'parameter-destinations', 'uint32', 5),
      weights: createVectorView('parameter-weights', 'float32', Array(5).fill(chunk))
    })
  );
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const declareView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /flow-aggregation weights chunk parameter buffer byteLength.*1024/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(declareView).not.toHaveBeenCalled();
  device.destroy();
});

it('two chunked flow contributors prefix every generated resource and node with their own ID', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const id of ['review-flows-first', 'review-flows-second']) {
    const chunks = [1, 3, 6].map((length, index) =>
      createTransientView(graph, `${id}-input-${index}`, 'float32', length)
    );
    const contributor = new GPUFlowAggregation(
      createProps(graph, {
        id,
        zones: {kind: 'ids', zoneCount: 100000},
        weights: createVectorView(`${id}-weights`, 'float32', chunks)
      })
    );
    expect(() => graph.add(contributor)).not.toThrow();
  }
  device.destroy();
});

it('flow refuses physical output aliases hidden behind distinct imported handles', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 40, usage: Buffer.STORAGE});
  const first = importGraphBuffer(graph, 'aliased-flow-first', buffer, 'uint32', 4);
  const second = importGraphBuffer(graph, 'aliased-flow-second', buffer, 'uint32', 4);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {output: {zoneOutCounts: first, zoneInCounts: second}})
      )
  ).toThrow(/physical storage.*one graph buffer handle/);
  const origins = importGraphBuffer(graph, 'aliased-flow-input', buffer, 'uint32', 10);
  expect(
    () => new GPUFlowAggregation(createProps(graph, {originZoneIds: origins, output: {ids: first}}))
  ).toThrow(/outputs must not share buffers with inputs/);
  buffer.destroy();
  device.destroy();
});

it('flow refuses output capacities that cannot be emitted as uint32 literals', () => {
  expectThrows(
    graph => ({
      output: {ids: createTransientView(graph, 'oversized-literal-output', 'uint32', 0x100000000)}
    }),
    /flow-aggregation output.ids rows.*4294967295/
  );
});

it('flow validates chunk row totals before declaring packed contribution scratch', () => {
  expectThrows(graph => {
    const chunk = createTransientView(graph, 'inconsistent-weight-chunk', 'float32', 1);
    const weights = createVectorView('inconsistent-weight-total', 'float32', [chunk]);
    Object.defineProperty(weights, 'length', {value: 10});
    return {weights};
  }, /flow-aggregation weights length must equal its chunk total/);
});

it('NET-125 captures nested lattice settings, time ports and output destinations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const bounds: [number, number, number, number] = [0, 0, 2, 2];
  const gridSize: [number, number] = [2, 2];
  let zoneKindReads = 0;
  const zones = {kind: 'grid' as const, bounds, gridSize};
  Object.defineProperty(zones, 'kind', {
    get: () => {
      zoneKindReads++;
      return 'grid';
    }
  });
  const props = createProps(graph, {
    zones,
    originZoneIds: undefined,
    destinationZoneIds: undefined,
    origins: createTransientView(graph, 'capture-origins', 'float32x2', 10),
    destinations: createTransientView(graph, 'capture-destinations', 'float32x2', 10),
    timeWindow: {
      timestamps: createTransientView(graph, 'capture-times', 'float32', 10),
      window: createTransientView(graph, 'capture-window', 'float32', 8)
    }
  });
  try {
    const contributor = new GPUFlowAggregation(props);
    const originalCount = props.output.status.count;
    const originalIds = props.output.ids;
    const originalWindow = props.timeWindow!.window;
    props.pairCapacity = 1024;
    gridSize[0] = 100;
    bounds[2] = 100;
    props.output.ids = createTransientView(graph, 'redirect-ids', 'uint32', 4);
    props.output.status.count = createTransientView(graph, 'redirect-count', 'uint32', 1);
    props.timeWindow!.window = createTransientView(graph, 'redirect-window', 'float32', 8);
    expect(zoneKindReads).toBe(1);
    expect(contributor.props.pairCapacity).toBe(8);
    expect(contributor.props.zones).toMatchObject({gridSize: [2, 2], bounds: [0, 0, 2, 2]});
    expect(contributor.props.output.ids).toBe(originalIds);
    expect(contributor.props.output.status.count).toBe(originalCount);
    expect(contributor.props.timeWindow!.window).toBe(originalWindow);
    expect(contributor.props.origins).toBe(props.origins);
    const nodes = contributor.getCommandNodes(graph);
    expect([contributor.rowCount, contributor.zoneCount]).toEqual([10, 4]);
    const resources = nodes.flatMap(node => (node.resources ?? []).map(getResourceBuffer));
    expect(resources).toContain(originalIds);
    expect(resources).toContain(originalCount);
    expect(resources).toContain(originalWindow);
    expect(resources).not.toContain(props.output.ids);
    expect(resources).not.toContain(props.output.status.count);
    expect(nodes).toHaveLength(20);
  } finally {
    device.destroy();
  }
});

it('B1 captures only ID-zone fields and ignores inactive lattice getters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const zones = {kind: 'ids' as const, zoneCount: 3};
  for (const field of ['gridSize', 'bounds', 'radius', 'activeGridSize']) {
    Object.defineProperty(zones, field, {
      get: () => {
        throw new Error(`inactive ${field}`);
      }
    });
  }
  try {
    const contributor = new GPUFlowAggregation(createProps(graph, {zones}));
    expect(contributor.zoneCount).toBe(3);
    expect(contributor.getCommandNodes(graph)).toHaveLength(15);
  } finally {
    device.destroy();
  }
});

it('B1 preserves the contributor-prefixed diagnostic for an unknown zone kind', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const zones = {kind: 'unknown'} as unknown as GPUFlowAggregationProps['zones'];
    expect(() => new GPUFlowAggregation(createProps(graph, {id: 'unknown-zones', zones}))).toThrow(
      'unknown-zones zones.kind must be grid, hexagon, or ids'
    );
  } finally {
    device.destroy();
  }
});
