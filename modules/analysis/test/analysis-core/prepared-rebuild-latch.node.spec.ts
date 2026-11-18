// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialJoinPrepared} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {
  GPUStudyWindowPreparation,
  type GPUStudyWindowPreparationProps
} from '../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {advancePreparedGeometryGeneration} from '../../src/gpu-spatial-analysis/spatial-join/prepared-geometry-snapshot';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

function getControlView(node: GPUCommandNode<{changed: boolean}>, suffix: string) {
  const resource = node.resources?.find(
    resource =>
      'buffer' in resource &&
      resource.buffer instanceof GraphDataView &&
      resource.buffer.buffer.id.endsWith(suffix)
  );
  if (!resource || !('buffer' in resource) || !(resource.buffer instanceof GraphDataView)) {
    throw new Error('Missing persistent generation view ' + suffix);
  }
  expect(resource.buffer.buffer.transient).toBe(false);
  return resource.buffer;
}

for (const form of ['own', 'prototype', 'non-enumerable'] as const) {
  it(`final1 study window retains ${form} summary and capacity properties`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const summary = createTransientView(graph, 'caller-summary', 'float32', 8);
    const props: GPUStudyWindowPreparationProps = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      samplingGrid: [2, 2]
    };
    const descriptors = {
      output: {enumerable: form !== 'non-enumerable', get: () => ({summary})},
      crossingCapacity: {enumerable: form !== 'non-enumerable', get: () => 16},
      slabEntryCapacity: {enumerable: form !== 'non-enumerable', get: () => 16}
    };
    if (form === 'prototype')
      Object.setPrototypeOf(props, Object.defineProperties({}, descriptors));
    else Object.defineProperties(props, descriptors);
    const producer = new GPUStudyWindowPreparation(props);
    try {
      const nodes = producer.getCommandNodes(graph);
      const copy = nodes.find(node => node.id.endsWith('-copy-summary'));
      expect(
        copy?.resources?.some(resource => 'buffer' in resource && resource.buffer === summary)
      ).toBe(true);
      expect(
        getGPUCommandNodeKernelSources(nodes).some(
          kernel =>
            kernel.id.startsWith('study-window-raster') &&
            kernel.source.includes('const CAPACITY: u32 = 16u;')
        )
      ).toBe(true);
    } finally {
      producer.destroy();
      device.destroy();
    }
  });
}

it('prepared geometry generation carries into its high word', () => {
  const generation = new Uint32Array([0xffffffff, 7]);
  advancePreparedGeometryGeneration(generation);
  expect(Array.from(generation)).toEqual([0, 8]);
  advancePreparedGeometryGeneration(generation);
  expect(Array.from(generation)).toEqual([1, 8]);
});

it('retains caller-owned BVH storage when destroying owned geometry snapshots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const view = <Format extends 'float32x2' | 'uint32x2' | 'uint32'>(
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
    nodeMinima: view('minima', 'float32x2'),
    nodeMaxima: view('maxima', 'float32x2'),
    nodeChildren: view('children', 'uint32x2'),
    leafIds: view('leaves', 'uint32'),
    count: view('count', 'uint32'),
    overflow: view('overflow', 'uint32')
  };
  const producer = new GPUSpatialJoinPrepared({
    geometry: {kind: 'points', positions: createTransientView(graph, 'positions', 'float32x2', 1)},
    storage
  });
  try {
    producer.getCommandNodes(graph);
    const destroy = buffers.map(buffer => vi.spyOn(buffer, 'destroy'));
    producer.destroy();
    expect(producer.storage).toBe(storage);
    for (const callback of destroy) expect(callback).not.toHaveBeenCalled();
  } finally {
    producer.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
    vi.restoreAllMocks();
  }
});

for (const kind of ['join', 'window'] as const) {
  for (const request of ['invalidate', 'predicate'] as const) {
    it('ERG-02: ' + kind + ' retains requested GPU generation after discarding ' + request, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph<{changed: boolean}>(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 3);
      const rebuildWhen = (frame: {changed: boolean}) => frame.changed;
      const producer =
        kind === 'join'
          ? new GPUSpatialJoinPrepared({geometry: {kind: 'points', positions}, rebuildWhen})
          : new GPUStudyWindowPreparation({
              positions,
              featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
              polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
              ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
              samplingGrid: [2, 2],
              rebuildWhen
            });
      try {
        const nodes = producer.getCommandNodes(graph);
        const prepare = nodes[0];
        const commit = nodes[nodes.length - 1];
        const requested = getControlView(prepare, '-requested-generation');
        const applied = getControlView(prepare, '-applied-generation');
        const writes: number[][] = [];
        vi.spyOn(requested.buffer.defaultBuffer!, 'write').mockImplementation(values => {
          writes.push(Array.from(values as Uint32Array));
        });
        const writeApplied = vi.spyOn(applied.buffer.defaultBuffer!, 'write');
        const encode = (changed: boolean) => {
          for (const node of nodes) {
            if (node.condition?.source === 'cpu')
              expect(node.condition.evaluate({changed})).toBe(true);
          }
        };
        encode(false);
        if (request === 'invalidate') producer.invalidate();
        encode(request === 'predicate'); // discard all recorded commands
        encode(false); // replacement must still send the unapplied generation
        expect(writes).toEqual([
          [1, 0],
          [2, 0],
          [2, 0]
        ]);
        expect(writeApplied).not.toHaveBeenCalled();
        expect(producer.encodedBuildCount).toBe(2);
        expect(commit.id).toBe(producer.id + '-snapshot-commit');
        for (const node of nodes.slice(0, -1)) expect(commit.dependsOn).toContain(node.id);
        const sources = getGPUCommandNodeKernelSources(nodes);
        const prepareSource = sources.find(source => source.id === prepare.id)!.source;
        const commitSource = sources.find(source => source.id === commit.id)!.source;
        expect(prepareSource).toContain('state[stateOffset] != requestedLow');
        expect(prepareSource).toContain('state[stateOffset + 1u] != requestedHigh');
        expect(commitSource).toContain('state[stateOffset] = state[stateOffset + 2u]');
        expect(commitSource).toContain('state[stateOffset + 1u] = state[stateOffset + 3u]');
        // Published builders replay against the retained geometry, never against live positions.
        const builders = nodes.filter(node => !node.id.includes('-snapshot-'));
        expect(builders.length).toBeGreaterThan(0);
        expect(
          builders.some(node =>
            node.resources?.some(resource => 'buffer' in resource && resource.buffer === positions)
          )
        ).toBe(false);
        expect(nodes.some(node => node.condition?.source === 'gpu')).toBe(true);
      } finally {
        producer.destroy();
        device.destroy();
        vi.restoreAllMocks();
      }
    });
  }
}
