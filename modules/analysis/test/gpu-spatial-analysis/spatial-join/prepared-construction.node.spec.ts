// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {createSpatialPreparationRetryViews} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-preparation-retry';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSpatialJoinPrepared,
  getGPUSpatialJoinPreparedMemoryPlan
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUBoundsFilter} from '../../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {GPUVertexSnap} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GEO-126 prepared join captures geometry, partition descriptors and dynamic callback identity', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const geometry = {kind: 'points' as const, positions};
  const partition = {id: 0, coreStart: 0, coreEnd: 2, haloStart: 0, haloEnd: 2};
  const partitioning = {
    length: 2,
    partitions: [partition],
    seamOwnership: {kind: 'lowest-partition-id' as const}
  };
  let refresh = false;
  const rebuildWhen = () => refresh;
  const props = {geometry, partitioning, rebuildWhen};
  const prepared = new GPUSpatialJoinPrepared(props);
  try {
    geometry.positions = createTransientView(graph, 'replacement', 'float32x2', 99);
    partition.coreEnd = 99;
    partitioning.partitions.length = 0;
    props.rebuildWhen = () => false;
    expect(prepared.geometry.positions).toBe(positions);
    expect(prepared.featureCount).toBe(2);
    expect(prepared.partitioning!.partitions).toHaveLength(1);
    expect(prepared.partitioning!.partitions[0].coreEnd).toBe(2);
    refresh = true;
    expect(prepared.props.rebuildWhen!(undefined)).toBe(true);
    expect(() => prepared.getCommandNodes(graph)).not.toThrow();
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('GEO-126 bounds filter captures nested output/status destinations and input identities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const bounds = createTransientView(graph, 'bounds', 'float32x4', 2);
  const ids = createTransientView(graph, 'ids', 'uint32', 2);
  const count = createTransientView(graph, 'count', 'uint32', 1);
  const props = {
    bounds,
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      ids,
      status: {count, overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  };
  const contributor = new GPUBoundsFilter(props);
  props.bounds = createTransientView(graph, 'replacement', 'float32x4', 99);
  props.output.ids = createTransientView(graph, 'replacement-ids', 'uint32', 99);
  props.output.status.count = createTransientView(graph, 'replacement-count', 'uint32', 1);
  try {
    expect(contributor.props.bounds).toBe(bounds);
    expect(contributor.props.output.ids).toBe(ids);
    expect(contributor.props.output.status.count).toBe(count);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('GEO-126 vertex snap captures reference and output identities', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const references = createTransientView(graph, 'references', 'float32x2', 2);
  const positions = createTransientView(graph, 'snapped', 'float32x2', 2);
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    referencePositions: references,
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {positions}
  };
  const contributor = new GPUVertexSnap(props);
  props.referencePositions = createTransientView(graph, 'replacement', 'float32x2', 99);
  props.output.positions = createTransientView(graph, 'replacement-out', 'float32x2', 99);
  try {
    expect(contributor.props.referencePositions).toBe(references);
    expect(contributor.props.output.positions).toBe(positions);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B3 prepared storage captures every caller view and preserves caller buffers after destruction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const allocate = <Format extends 'float32x2' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      byteLength: format === 'uint32' ? 4 : 8,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, 1);
  };
  const storage = {
    nodeMinima: allocate('minima', 'float32x2'),
    nodeMaxima: allocate('maxima', 'float32x2'),
    nodeChildren: allocate('children', 'uint32x2'),
    leafIds: allocate('leaves', 'uint32'),
    count: allocate('count', 'uint32'),
    overflow: allocate('overflow', 'uint32')
  };
  const captured = {...storage};
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {kind: 'points', positions: createTransientView(graph, 'positions', 'float32x2', 1)},
    storage
  });
  const destroy = buffers.map(buffer => vi.spyOn(buffer, 'destroy'));
  try {
    storage.leafIds = allocate('replacement-leaves', 'uint32');
    prepared.getCommandNodes(graph);
    prepared.destroy();
    expect(prepared.storage).not.toBe(storage);
    for (const name of [
      'nodeMinima',
      'nodeMaxima',
      'nodeChildren',
      'leafIds',
      'count',
      'overflow'
    ] as const) {
      expect(prepared.storage[name]).toBe(captured[name]);
    }
    for (const callback of destroy) expect(callback).not.toHaveBeenCalled();
  } finally {
    prepared.destroy();
    for (const callback of destroy) callback.mockRestore();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});

it('TSC-01 prepared polygon slabs allocate the layout promised by the planner', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const prepared = new GPUSpatialJoinPrepared({
    geometry: {
      kind: 'polygons',
      positions: createTransientView(graph, 'slab-positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'slab-features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'slab-polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'slab-rings', 'uint32', 2)
    },
    edgeSlabs: true
  });
  try {
    const plan = getGPUSpatialJoinPreparedMemoryPlan(
      {
        kind: 'polygons',
        featureCount: 1,
        vertexCount: 4,
        polygonCount: 1,
        ringCount: 1,
        edgeSlabs: true
      },
      device.limits
    );
    graph.add(prepared);
    // One vertex bitmap word plus two polygon-range words: (1 + 2) * 4 = 12 bytes.
    expect(prepared.edgeSlabStorage!.layout.length * 4).toBe(12);
    expect(prepared.edgeSlabStorage!.layout.length * 4).toBe(
      plan.scratchByteLengths['edge slab layout']
    );
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('TSC-02 retry uses borrowed GPUData and preserves a canonical nonzero-offset view', () => {
  const device = createNullWebGPUDevice();
  const original = new GPUCommandGraph(device);
  const fresh = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    byteLength: 512,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const oldHandle = importGraphBuffer(original, 'old', buffer, 'uint32', 128).buffer;
  const source = original.createDataView(oldHandle, {
    format: 'float32x2',
    length: 2,
    byteOffset: 256,
    byteStride: 16,
    rowByteLength: 8
  });
  const canonical = importGraphBuffer(fresh, 'canonical', buffer, 'uint32', 128);
  const spy = vi.spyOn(fresh, 'importGPUData');
  try {
    const retried = createSpatialPreparationRetryViews(fresh, 'retry', {
      positions: source
    }).positions;
    expect(spy.mock.calls[0][1]).toBeInstanceOf(GPUData);
    spy.mock.calls[0][1].destroy();
    expect(buffer.destroyed).toBe(false);
    expect(retried.buffer).toBe(canonical.buffer);
    for (const field of ['format', 'length', 'byteOffset', 'byteStride', 'rowByteLength'] as const)
      expect(retried[field]).toBe(source[field]);
    expect(() =>
      createSpatialPreparationRetryViews(fresh, 'retry', {
        positions: createTransientView(original, 'transient', 'float32x2', 1)
      })
    ).toThrow(/persistent/);
  } finally {
    spy.mockRestore();
    expect(buffer.destroyed).toBe(false);
    buffer.destroy();
    device.destroy();
  }
});

it('GEO-115 exposes only the successful graph snapshot and clears it on destroy', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'snapshot-source', 'float32x2', 1);
  const sourceIds = createTransientView(graph, 'snapshot-ids', 'uint32', 1);
  const prepared = new GPUSpatialJoinPrepared({geometry: {kind: 'points', positions, sourceIds}});
  try {
    expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
    graph.add(prepared);
    const captured = prepared.getGeometry(graph);
    expect(captured.positions).not.toBe(positions);
    expect(captured.sourceIds).toBe(sourceIds);
    expect(() => prepared.getGeometry(new GPUCommandGraph(device))).toThrow(/graph.add/);
    prepared.destroy();
    expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('W5-01 retry rebinds caller storage without acquiring its lifetime', () => {
  const device = createNullWebGPUDevice();
  const original = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const allocate = <Format extends 'float32x2' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      byteLength: 512,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = importGraphBuffer(original, name, buffer, 'uint32', 128).buffer;
    return original.createDataView(handle, {format, length: 1, byteOffset: 256});
  };
  const geometry = {kind: 'points' as const, positions: allocate('retry-points', 'float32x2')};
  const storage = {
    nodeMinima: allocate('retry-minima', 'float32x2'),
    nodeMaxima: allocate('retry-maxima', 'float32x2'),
    nodeChildren: allocate('retry-children', 'uint32x2'),
    leafIds: allocate('retry-leaves', 'uint32'),
    count: allocate('retry-count', 'uint32'),
    overflow: allocate('retry-overflow', 'uint32')
  };
  const prepared = new GPUSpatialJoinPrepared({id: 'external-retry', geometry, storage});
  const created: Buffer[] = [];
  const createBuffer = device.createBuffer.bind(device);
  const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
    if (
      typeof settings === 'object' &&
      settings !== null &&
      'id' in settings &&
      settings.id === 'external-retry-generation-byte-values'
    )
      throw new Error('injected generation table');
    const buffer = createBuffer(settings);
    created.push(buffer);
    return buffer;
  });
  try {
    try {
      expect(() => original.add(prepared)).toThrow('injected generation table');
    } finally {
      spy.mockRestore();
    }
    expect(created.every(buffer => buffer.destroyed)).toBe(true);
    expect(() => prepared.getGeometry(original)).toThrow(/graph.add/);
    const fresh = new GPUCommandGraph(device);
    const canonical = Object.fromEntries(
      Object.entries(storage).map(([name, view]) => [
        name,
        importGraphBuffer(
          fresh,
          `canonical-${name}`,
          view.buffer.defaultBuffer as Buffer,
          'uint32',
          128
        ).buffer
      ])
    );
    const input = importGraphBuffer(
      fresh,
      'canonical-points',
      geometry.positions.buffer.defaultBuffer as Buffer,
      'uint32',
      128
    );
    fresh.add(prepared);
    for (const [name, view] of Object.entries(prepared.storage))
      if (view) {
        expect(view.buffer).toBe(canonical[name]);
        expect(view.byteOffset).toBe(256);
      }
    const snapshotSource = prepared.getGeometry(fresh);
    expect(snapshotSource.positions.buffer).not.toBe(input.buffer);
    prepared.destroy();
    for (const buffer of buffers) expect(buffer.destroyed).toBe(false);
    expect(prepared.storage.nodeMinima).toBe(storage.nodeMinima);
  } finally {
    spy.mockRestore();
    prepared.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
});
