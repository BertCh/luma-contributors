// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createHash} from 'node:crypto';
import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {expect, it, vi} from 'vitest';
import {getForbiddenFlowWGSLIdentifiers} from '../flow-aggregation/flow-wgsl-review';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  decodeGPUNetworkCoarseningSummaryWords,
  GPUNetworkCoarsening,
  GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY,
  getGPUNetworkCoarseningMaximumWeightedEdgeCapacity,
  type GPUNetworkCoarseningOutput,
  type GPUNetworkCoarseningProps
} from '../../../src/gpu-network/network-coarsening/index';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCoarseningCSR,
  computeCoarseningOracle,
  dequantize,
  quantize
} from './network-coarsening-oracle';

type PropsOverrides = Omit<Partial<GPUNetworkCoarseningProps>, 'output'> & {
  output?: Partial<GPUNetworkCoarseningOutput>;
};

const NODE_COUNT = 6;
const SLOT_COUNT = 10;
const GROUP_CAPACITY = 4;
const EDGE_CAPACITY = 5;

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  let counter = 0;
  const importView = <Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const rowWords = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    name = `${name}-${counter++}`;
    const buffer = device.createBuffer({
      id: name,
      byteLength: Math.max(length * rowWords, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const props = (overrides: PropsOverrides = {}): GPUNetworkCoarseningProps => ({
    topology: {
      offsets: importView('offsets', 'uint32', NODE_COUNT + 1),
      neighbors: importView('neighbors', 'uint32', SLOT_COUNT)
    },
    labels: importView('labels', 'uint32', NODE_COUNT),
    groupCapacity: GROUP_CAPACITY,
    ...overrides,
    output: {
      groupNodeCount: importView('group-counts', 'uint32', GROUP_CAPACITY),
      edges: {
        ids: importView('edge-ids', 'uint32', EDGE_CAPACITY),
        status: {
          count: importView('edge-count', 'uint32', 1),
          overflow: importView('edge-overflow', 'uint32', 1)
        }
      },
      edgeTargetGroups: importView('edge-targets', 'uint32', EDGE_CAPACITY),
      edgeCounts: importView('edge-counts', 'uint32', EDGE_CAPACITY),
      ...overrides.output
    }
  });
  return {graph, importView, props};
}

it('GPUNetworkCoarsening schedules a fixed node order and omits unused passes', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const minimal = new GPUNetworkCoarsening(props({id: 'coarse'}));
  const minimalIds = minimal.getCommandNodes(graph).map(node => node.id);
  expect(minimalIds[0]).toBe('coarse-zero-stats');
  expect(minimalIds.some(id => id === 'coarse-groups')).toBe(false);
  expect(minimalIds.some(id => id.endsWith('-edge-weights'))).toBe(false);
  expect(minimalIds[minimalIds.length - 1]).toBe('coarse-publish');
  const indexOf = (ids: string[], name: string) => ids.indexOf(`coarse-${name}`);
  expect(indexOf(minimalIds, 'vertices')).toBeGreaterThan(indexOf(minimalIds, 'init-groups'));
  expect(indexOf(minimalIds, 'slots')).toBeGreaterThan(indexOf(minimalIds, 'vertices'));
  expect(indexOf(minimalIds, 'segments')).toBeGreaterThan(indexOf(minimalIds, 'segment-total'));

  const other = createContext(device);
  const full = new GPUNetworkCoarsening(
    other.props({
      id: 'full',
      directed: true,
      topology: {
        offsets: other.importView('offsets', 'uint32', NODE_COUNT + 1),
        neighbors: other.importView('neighbors', 'uint32', SLOT_COUNT),
        weights: other.importView('weights', 'float32', SLOT_COUNT),
        nodePositions: other.importView('positions', 'float32x2', NODE_COUNT)
      },
      nodeMask: other.importView('vertex-mask', 'uint32', NODE_COUNT),
      edgeMask: other.importView('edge-mask', 'uint32', SLOT_COUNT),
      nodeValues: other.importView('values', 'float32', NODE_COUNT),
      output: {
        groupIntraEdgeCount: other.importView('intra-count', 'uint32', GROUP_CAPACITY),
        groupIntraWeight: other.importView('intra-weight', 'float32', GROUP_CAPACITY),
        groupCentroid: other.importView('centroid', 'float32x2', GROUP_CAPACITY),
        groupBounds: other.importView('bounds', 'float32x4', GROUP_CAPACITY),
        groupValueSum: other.importView('value-sum', 'float32', GROUP_CAPACITY),
        edgeWeights: other.importView('edge-weights', 'float32', EDGE_CAPACITY),
        summary: other.importView('summary', 'uint32', 8)
      }
    })
  );
  const fullIds = full.getCommandNodes(other.graph).map(node => node.id);
  expect(fullIds).toContain('full-groups');
  expect(fullIds).toContain('full-edge-weights');
  void importView;
  device.destroy();
});

it('GPUNetworkCoarsening validates props', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const create = (overrides: PropsOverrides) => new GPUNetworkCoarsening(props(overrides));
  const topology = (offsets: number, neighbors: number) => ({
    offsets: importView('o', 'uint32', offsets),
    neighbors: importView('n', 'uint32', neighbors)
  });
  expect(() => create({topology: topology(1, SLOT_COUNT)})).toThrow(/at least two rows/);
  expect(() => create({topology: topology(NODE_COUNT + 1, 0)})).toThrow(/at least one slot/);
  expect(() =>
    create({
      topology: {...topology(NODE_COUNT + 1, SLOT_COUNT), weights: importView('w', 'float32', 3)}
    })
  ).toThrow(/topology\.weights length/);
  expect(() => create({groupCapacity: 0})).toThrow(/groupCapacity/);
  expect(() => create({groupCapacity: 0xffffffff})).toThrow(/groupCapacity/);
  expect(() => create({fixedPointScale: 3})).toThrow(/fixedPointScale/);
  expect(() => create({labels: importView('l-short', 'uint32', 2)})).toThrow(/labels must contain/);
  expect(() => create({output: {groupNodeCount: importView('g-short', 'uint32', 2)}})).toThrow(
    /output\.groupNodeCount must contain exactly 4 rows/
  );
  expect(() => create({output: {edgeTargetGroups: importView('t-short', 'uint32', 2)}})).toThrow(
    /output\.edgeTargetGroups must match output\.edges\.ids length/
  );
  expect(() => create({output: {summary: importView('s-short', 'uint32', 3)}})).toThrow(
    /output\.summary must contain/
  );
  expect(() =>
    create({output: {groupCentroid: importView('c', 'float32x2', GROUP_CAPACITY)}})
  ).toThrow(/require topology\.nodePositions/);
  expect(() =>
    create({output: {groupValueSum: importView('v', 'float32', GROUP_CAPACITY)}})
  ).toThrow(/requires nodeValues/);
  const aliased = importView('aliased', 'uint32', NODE_COUNT);
  expect(() => create({labels: aliased, output: {groupNodeCount: aliased}})).toThrow();
  const foreign = createContext(createNullWebGPUDevice());
  expect(() => new GPUNetworkCoarsening(foreign.props()).getCommandNodes(graph)).toThrow(
    /views must belong to the target graph/
  );
  device.destroy();
});

it('coarsening oracle helpers quantize, wrap and order deterministically', () => {
  expect(quantize(1.5, 4)).toBe(6n);
  expect(quantize(-2.5, 1)).toBe(-2n);
  expect(quantize(3.5, 1)).toBe(4n);
  expect(dequantize(-6n, 4)).toBe(-1.5);
  expect(dequantize(1n << 40n, 1)).toBe(2 ** 40);
  const csr = buildCoarseningCSR(4, [
    [0, 1, 1],
    [1, 0, 1],
    [2, 3, 2],
    [3, 2, 2],
    [0, 0, 5]
  ]);
  const result = computeCoarseningOracle({
    nodeCount: 4,
    csr,
    labels: Uint32Array.from([0, 0, 1, 9]),
    groupCapacity: 3,
    edgeCapacity: 4
  });
  expect(Array.from(result.groupVertexCount)).toEqual([2, 1, 0]);
  expect(Array.from(result.groupIntraEdgeCount)).toEqual([2, 0, 0]);
  expect(result.edges).toEqual([]);
  expect(decodeGPUNetworkCoarseningSummaryWords(result.summary)).toMatchObject({
    liveNodeCount: 4,
    overflowedNodeCount: 1,
    droppedEdgeCount: 1,
    intraEdgeCount: 2,
    groupCount: 2
  });
  expect(result.overflow).toBe(false);
  expect(result.candidateOverflow).toBe(true);
  expect(() => decodeGPUNetworkCoarseningSummaryWords([1, 2])).toThrow(/8 words/);
});

it('GPUNetworkCoarsening gives CSR tail slots the sentinel key and folds its counters', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const nodes = new GPUNetworkCoarsening(props({id: 'tail'})).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Review B-09: slots past offsets[NODE_COUNT] are never written by the per-row slots pass, so
  // the sort keys are cleared to the sentinel first and those slots sort last.
  const clear = ids.indexOf('tail-clear-sort-keys');
  expect(clear).toBeGreaterThanOrEqual(0);
  expect(clear).toBeLessThan(ids.indexOf('tail-slots'));
  const clearResource = nodes[clear].resources?.[0];
  const clearBuffer = clearResource ? getResourceBuffer(clearResource) : undefined;
  expect(clearBuffer?.length).toBe(SLOT_COUNT);
  const slots = nodes[ids.indexOf('tail-slots')];
  expect(slots.resources?.map(resource => getResourceBuffer(resource))).toContain(clearBuffer);
  const sources = getParsedKernelSources(nodes);
  const slotsSource = sources.get('tail-slots') ?? '';
  expect(slotsSource).toContain(
    'let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);'
  );
  // Rule 14: summary words take one atomic per workgroup.
  expect(slotsSource).toContain('workgroupCounters');
  expect(sources.get('tail-vertices')).toContain('workgroupCounters');
});

it('coarsening sorts separate uint32 columns above 65535 groups and uses valid WGSL identifiers', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const contributor = new GPUNetworkCoarsening(
    props({
      groupCapacity: 100000,
      topology: {
        offsets: importView('offsets-large', 'uint32', NODE_COUNT + 1),
        neighbors: importView('neighbors-large', 'uint32', SLOT_COUNT),
        weights: importView('weights-large', 'float32', SLOT_COUNT),
        nodePositions: importView('positions-large', 'float32x2', NODE_COUNT)
      },
      nodeMask: importView('node-mask-large', 'uint32', NODE_COUNT),
      edgeMask: importView('edge-mask-large', 'uint32', SLOT_COUNT),
      nodeValues: importView('values-large', 'float32', NODE_COUNT),
      output: {
        groupNodeCount: importView('counts-large', 'uint32', 100000),
        groupIntraEdgeCount: importView('intra-large', 'uint32', 100000),
        groupIntraWeight: importView('intra-weights-large', 'float32', 100000),
        groupCentroid: importView('centroids-large', 'float32x2', 100000),
        groupBounds: importView('bounds-large', 'float32x4', 100000),
        groupValueSum: importView('sums-large', 'float32', 100000),
        edgeWeights: importView('edge-weights-large', 'float32', EDGE_CAPACITY)
      }
    })
  );
  const nodes = contributor.getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('network-coarsening-slots')).not.toContain('sourceLabel * GROUP_CAPACITY');
  expect(sources.get('network-coarsening-segments')).not.toContain('key / GROUP_CAPACITY');
  for (const source of sources.values()) {
    expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
    expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
  }
  device.destroy();
});

it('coarsening rejects overlapping output fields', () => {
  const device = createNullWebGPUDevice();
  const {importView, props} = createContext(device);
  const shared = importView('shared-output', 'uint32', EDGE_CAPACITY);
  expect(
    () =>
      new GPUNetworkCoarsening(
        props({
          output: {edgeTargetGroups: shared, edgeCounts: shared}
        })
      )
  ).toThrow(/writable byte ranges/);
  device.destroy();
});

it('coarsening checks aligned views and derived accumulators before mutating the graph', () => {
  for (const failure of ['aligned-input', 'group-accumulator', 'slot-columns'] as const) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    const {graph, importView, props} = createContext(device);
    const backing = importView('unaligned', 'uint32', 257);
    const overrides: PropsOverrides =
      failure === 'aligned-input'
        ? {
            labels: graph.createDataView(backing.buffer, {
              format: 'uint32',
              byteOffset: 8,
              length: 255
            }),
            topology: {
              offsets: importView('wide-offsets', 'uint32', 256),
              neighbors: importView('small-neighbors', 'uint32', 10)
            }
          }
        : failure === 'group-accumulator'
          ? {
              groupCapacity: 26,
              output: {groupNodeCount: importView('group-output', 'uint32', 26)}
            }
          : {
              topology: {
                offsets: importView('offsets-pairs', 'uint32', NODE_COUNT + 1),
                neighbors: importView('many-slots', 'uint32', 257)
              }
            };
    const contributor = new GPUNetworkCoarsening(props(overrides));
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const declareView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/network-coarsening.*binding/);
    expect(allocate).not.toHaveBeenCalled();
    expect(declareView).not.toHaveBeenCalled();
    device.destroy();
  }
});

it('coarsening keeps every slot scratch binding scalar-sized at 30 million slots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  const contributor = new GPUNetworkCoarsening({
    topology: {offsets: rows('large-offsets', 7), neighbors: rows('large-neighbors', 30000000)},
    labels: rows('large-labels', 6),
    groupCapacity: 100000,
    edgeMask: rows('large-slot-mask', 30000000),
    output: {
      groupNodeCount: rows('large-group-counts', 100000),
      edges: {
        ids: rows('large-edge-ids', 4),
        status: {count: rows('large-count', 1), overflow: rows('large-overflow', 1)}
      },
      edgeTargetGroups: rows('large-targets', 4),
      edgeCounts: rows('large-counts', 4)
    }
  });
  expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  device.destroy();
});

it('coarsening rejects adjacent aligned-binding overlap and accepts separated shared-buffer outputs', () => {
  for (const stride of [4, 256]) {
    const device = createNullWebGPUDevice();
    const {graph, importView, props} = createContext(device);
    const backing = importView('shared-edge-columns', 'uint32', 129);
    const sources = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 0,
      length: 1
    });
    const targets = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: stride,
      length: 1
    });
    const counts = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 512,
      length: 1
    });
    const original = props();
    const candidate = {
      ...original,
      output: {
        ...original.output,
        edges: {...original.output.edges, ids: sources},
        edgeTargetGroups: targets,
        edgeCounts: counts
      }
    };
    if (stride === 4)
      expect(() => new GPUNetworkCoarsening(candidate)).toThrow(/aligned binding byte ranges/);
    else expect(() => graph.add(new GPUNetworkCoarsening(candidate))).not.toThrow();
    device.destroy();
  }
});

it('both coarsening fixed-point routes avoid every W3C reserved identifier', () => {
  for (const fixedPointScale of [undefined, 65536]) {
    const device = createNullWebGPUDevice();
    const {graph, props} = createContext(device);
    const nodes = new GPUNetworkCoarsening(props({fixedPointScale})).getCommandNodes(graph);
    for (const source of getParsedKernelSources(nodes).values()) {
      expect(getForbiddenFlowWGSLIdentifiers(source)).toEqual([]);
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
    }
    device.destroy();
  }
});

it('coarsening refuses one group above the exported uint32-addressing maximum at construction', () => {
  const device = createNullWebGPUDevice();
  const {props} = createContext(device);
  expect(
    () =>
      new GPUNetworkCoarsening(
        props({groupCapacity: GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY + 1})
      )
  ).toThrow(
    new RegExp(`network-coarsening groupCapacity.*${GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY}`)
  );
  device.destroy();
});

it('coarsening weighted edge capacity respects both buffer and storage binding limits', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 1024});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 8192});
  expect(getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device)).toBe(128);
  for (const capacity of [128, 129]) {
    const {graph, importView, props} = createContext(device);
    const contributor = new GPUNetworkCoarsening(
      props({
        output: {
          edges: {
            ids: importView('buffer-bound-ids', 'uint32', capacity),
            status: {
              count: importView('buffer-bound-count', 'uint32', 1),
              overflow: importView('buffer-bound-overflow', 'uint32', 1)
            }
          },
          edgeTargetGroups: importView('buffer-bound-targets', 'uint32', capacity),
          edgeCounts: importView('buffer-bound-counts', 'uint32', capacity),
          edgeWeights: importView('buffer-bound-weights', 'float32', capacity)
        }
      })
    );
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    if (capacity === 128) expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    else {
      expect(() => contributor.getCommandNodes(graph)).toThrow(
        /network-coarsening.*output.edgeWeights.*128/
      );
      expect(allocate).not.toHaveBeenCalled();
    }
  }
  device.destroy();
});

it('coarsening checks maxBufferSize for derived group scratch before declaring resources', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 1024});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 8192});
  const {graph, importView, props} = createContext(device);
  const contributor = new GPUNetworkCoarsening(
    props({
      groupCapacity: 30,
      output: {groupNodeCount: importView('buffer-group-counts', 'uint32', 30)}
    })
  );
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const declareView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /network-coarsening group accumulator byteLength.*1024/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(declareView).not.toHaveBeenCalled();
  device.destroy();
});

it('coarsening refuses physical output aliases hidden behind distinct imported handles', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  const first = importGraphBuffer(graph, 'aliased-groups-first', buffer, 'uint32', 4);
  const second = importGraphBuffer(graph, 'aliased-groups-second', buffer, 'uint32', 4);
  expect(
    () =>
      new GPUNetworkCoarsening(
        props({output: {groupNodeCount: first, groupIntraEdgeCount: second}})
      )
  ).toThrow(/physical storage.*one graph buffer handle/);
  buffer.destroy();
  device.destroy();
});

it('coarsening bounds node and output capacity literals before creating command nodes', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const original = props();
  expect(
    () =>
      new GPUNetworkCoarsening({
        ...original,
        topology: {
          ...original.topology,
          offsets: createTransientView(graph, 'oversized-node-offsets', 'uint32', 0x100000000)
        },
        labels: createTransientView(graph, 'oversized-node-labels', 'uint32', 0xffffffff)
      })
  ).toThrow(/network-coarsening topology node count.*4294967294/);
  expect(
    () =>
      new GPUNetworkCoarsening({
        ...original,
        output: {
          ...original.output,
          edges: {
            ...original.output.edges,
            ids: createTransientView(graph, 'oversized-edge-capacity', 'uint32', 0x100000000)
          }
        }
      })
  ).toThrow(/network-coarsening output.edges.ids rows.*4294967295/);
  Object.defineProperty(device.limits, 'maxBufferSize', {value: 2 ** 36});
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 2 ** 36});
  expect(getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device)).toBe(2147483647);
  device.destroy();
});

it('NET-125 captures topology descriptors, optional writers and nested compact ports', () => {
  const device = createNullWebGPUDevice();
  const {graph, props, importView} = createContext(device);
  const original = props();
  try {
    const contributor = new GPUNetworkCoarsening(original);
    const originalNeighbors = original.topology.neighbors;
    const originalIds = original.output.edges.ids;
    const originalCount = original.output.edges.status.count;
    original.directed = true;
    original.groupCapacity = 128;
    original.topology.neighbors = importView('redirect-neighbors', 'uint32', 2);
    original.output.edges.ids = importView('redirect-edges', 'uint32', EDGE_CAPACITY);
    original.output.edges.status.count = importView('redirect-count', 'uint32', 1);
    original.output.groupValueSum = importView('redirect-values', 'float32', GROUP_CAPACITY);
    expect(contributor.props.groupCapacity).toBe(4);
    expect(contributor.props.directed).toBeUndefined();
    expect(contributor.props.topology.neighbors).toBe(originalNeighbors);
    expect(contributor.props.output.edges.ids).toBe(originalIds);
    expect(contributor.props.output.edges.status.count).toBe(originalCount);
    expect(contributor.props.output.groupValueSum).toBeUndefined();
    expect(contributor.props.labels).toBe(original.labels);
    const nodes = contributor.getCommandNodes(graph);
    expect([contributor.nodeCount, contributor.groupCapacity, contributor.edgeCapacity]).toEqual([
      6, 4, 5
    ]);
    const resources = nodes.flatMap(node => (node.resources ?? []).map(getResourceBuffer));
    expect(resources).toContain(originalNeighbors);
    expect(resources).toContain(originalIds);
    expect(resources).toContain(originalCount);
    expect(resources).not.toContain(original.output.edges.ids);
    expect(resources).not.toContain(original.output.edges.status.count);
    expect(nodes).toHaveLength(20);
  } finally {
    device.destroy();
  }
});

it('NET-126 pins coarsening reductions, fixed-point encoding and all optional writers before extraction', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const fixedPointScale of [undefined, 65536]) {
      const {graph, props, importView} = createContext(device);
      recordWGSLKernelNodes(graph);
      const contributor = new GPUNetworkCoarsening(
        props({
          fixedPointScale,
          directed: true,
          topology: {
            offsets: importView('pin-offsets', 'uint32', 7),
            neighbors: importView('pin-neighbors', 'uint32', 10),
            weights: importView('pin-weights', 'float32', 10),
            nodePositions: importView('pin-positions', 'float32x2', 6)
          },
          nodeValues: importView('pin-values', 'float32', 6),
          output: {
            groupIntraEdgeCount: importView('pin-intra-counts', 'uint32', 4),
            groupIntraWeight: importView('pin-intra-weights', 'float32', 4),
            groupCentroid: importView('pin-centroids', 'float32x2', 4),
            groupBounds: importView('pin-bounds', 'float32x4', 4),
            groupValueSum: importView('pin-sums', 'float32', 4),
            edgeWeights: importView('pin-edge-weights', 'float32', 5),
            summary: importView('pin-summary', 'uint32', 8)
          }
        })
      );
      const nodes = contributor.getCommandNodes(graph);
      const digest = createHash('sha256')
        .update(JSON.stringify([...getParsedKernelSources(nodes)]))
        .digest('hex');
      const bindings = new Map(
        nodes.flatMap(
          node =>
            getWGSLKernelNodeRecord(graph, node.id)?.bindings.map(
              binding => [`${node.id}:${binding.name}`, binding.view] as const
            ) ?? []
        )
      );
      const {output} = contributor.props;
      const expectedBindings: Record<string, GraphDataView> = {
        'network-coarsening-groups:groupCounts': output.groupNodeCount,
        'network-coarsening-groups:centroidOut': output.groupCentroid!,
        'network-coarsening-groups:boundsOut': output.groupBounds!,
        'network-coarsening-groups:valueOut': output.groupValueSum!,
        'network-coarsening-intra-groups:intraCountOut': output.groupIntraEdgeCount!,
        'network-coarsening-intra-groups:intraWeightOut': output.groupIntraWeight!,
        'network-coarsening-segments:edgeIds': output.edges.ids,
        'network-coarsening-segments:edgeTargets': output.edgeTargetGroups,
        'network-coarsening-segments:edgeCounts': output.edgeCounts,
        'network-coarsening-edge-weights:edgeWeights': output.edgeWeights!,
        'network-coarsening-publish:countOut': output.edges.status.count,
        'network-coarsening-publish:overflowOut': output.edges.status.overflow,
        'network-coarsening-summary:summary': output.summary!
      };
      const assertDeclaration = (bindingViews: ReadonlyMap<string, GraphDataView>) => {
        for (const [name, expectedView] of Object.entries(expectedBindings)) {
          expect(bindingViews.get(name), name).toBe(expectedView);
        }
        expect
          .soft(digest, `fixedPointScale=${fixedPointScale}`)
          .toBe(
            fixedPointScale
              ? '5226c3f191797b985a831cd8fbe6cc841fe81a19298eac392717adfd7406476f'
              : '8fd250fed294bc122b813719fdc5cc522217c865e20878d805907672ed91e36a'
          );
      };
      assertDeclaration(bindings);
      // Same-format optional outputs can be swapped without changing shader text or IDs.
      const swapped = new Map(bindings);
      swapped.set(
        'network-coarsening-groups:valueOut',
        expectedBindings['network-coarsening-intra-groups:intraWeightOut']
      );
      swapped.set(
        'network-coarsening-intra-groups:intraWeightOut',
        expectedBindings['network-coarsening-groups:valueOut']
      );
      expect(() => assertDeclaration(swapped)).toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('coarsening retains optional partition provenance beside the incumbent quotient (CPL-14)', () => {
  const device = createNullWebGPUDevice();
  try {
    const {graph, importView, props} = createContext(device);
    const labelsComplete = importView('labels-complete', 'uint32', 1);
    const published = importView('published-complete', 'uint32', 1);
    const contributor = new GPUNetworkCoarsening(
      props({labelsComplete, output: {labelsComplete: published}})
    );
    expect(contributor.props.labelsComplete).toBe(labelsComplete);
    expect(contributor.partitionProvenance.labelsComplete).toBe(published);
    expect(contributor.getCommandNodes(graph).map(node => node.id)).toContain(
      'network-coarsening-partition-provenance'
    );
    expect(
      () =>
        new GPUNetworkCoarsening(props({labelsComplete: importView('short-complete', 'uint32', 0)}))
    ).toThrow(/labelsComplete/);
    expect(new GPUNetworkCoarsening(props()).partitionProvenance.labelsComplete).toBeUndefined();
  } finally {
    device.destroy();
  }
});
