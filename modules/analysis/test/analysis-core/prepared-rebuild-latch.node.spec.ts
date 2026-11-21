// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {getCoreDefaultBuffer} from '../../src/analysis-core/contracts/network-topology';
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

for (const withFeatureRings of [false, true]) {
  it(`retains caller-owned BVH storage and optional rings ${withFeatureRings} when destroying owned geometry snapshots`, () => {
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
      overflow: view('overflow', 'uint32'),
      ...(withFeatureRings ? {featureRings: view('feature-rings', 'uint32x2')} : {})
    };
    const producer = new GPUSpatialJoinPrepared({
      geometry: withFeatureRings
        ? {
            kind: 'polygons',
            positions: createTransientView(graph, 'positions', 'float32x2', 3),
            featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
            polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
            ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
          }
        : {kind: 'points', positions: createTransientView(graph, 'positions', 'float32x2', 1)},
      storage
    });
    try {
      producer.getCommandNodes(graph);
      const destroy = buffers.map(buffer => vi.spyOn(buffer, 'destroy'));
      producer.destroy();
      expect(producer.storage).not.toBe(storage);
      for (const [name, retained] of Object.entries(storage))
        expect(producer.storage[name as keyof typeof storage], name).toBe(retained);
      for (const callback of destroy) expect(callback).not.toHaveBeenCalled();
    } finally {
      producer.destroy();
      for (const buffer of buffers) buffer.destroy();
      device.destroy();
      vi.restoreAllMocks();
    }
  });
}

for (const kind of ['join', 'window'] as const) {
  for (const request of ['invalidate', 'predicate'] as const) {
    it(
      'ERG-02: ' + kind + ' retains requested GPU generation after discarding ' + request,
      async () => {
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
          const requestNode = nodes.find(node => node.id === producer.id + '-snapshot-request')!;
          const prepare = nodes.find(node => node.id === producer.id + '-snapshot-prepare')!;
          if (requestNode.type !== 'copy') throw new Error('snapshot request must be a copy node');
          const requestCopy = requestNode;
          const commit = nodes[nodes.length - 1];
          const requested = getControlView(prepare, '-requested-generation');
          const applied = getControlView(prepare, '-applied-generation');
          const requests: number[][] = [];
          const writeRequested = vi.spyOn(requested.buffer.defaultBuffer!, 'write');
          const writeApplied = vi.spyOn(applied.buffer.defaultBuffer!, 'write');
          const encode = async (changed: boolean) => {
            for (const node of nodes) {
              if (node.condition?.source === 'cpu')
                expect(node.condition.evaluate({changed})).toBe(true);
            }
            const copies: Parameters<CommandEncoder['copyBufferToBuffer']>[0][] = [];
            const executable = requestCopy.compile({device});
            const commandEncoder = device.createCommandEncoder();
            vi.spyOn(commandEncoder, 'copyBufferToBuffer').mockImplementation(copy => {
              copies.push(copy);
            });
            try {
              executable.encode({
                commandEncoder,
                parameters: {changed},
                getBuffer: value => {
                  const handle = 'buffer' in value ? value.buffer : value;
                  if (!handle.defaultBuffer)
                    throw new Error('snapshot copy needs an imported buffer');
                  const buffer = getCoreDefaultBuffer(handle.defaultBuffer);
                  expect(buffer).toBeInstanceOf(Buffer);
                  return buffer;
                },
                getTexture: () => {
                  throw new Error('unexpected texture');
                },
                getTextureView: () => {
                  throw new Error('unexpected texture view');
                },
                getExternalTexture: () => {
                  throw new Error('unexpected external texture');
                }
              });
            } finally {
              commandEncoder.destroy();
              executable.destroy?.();
            }
            expect(copies).toHaveLength(8);
            const words = [0, 0];
            // The immutable source table stores byte b at offset 4*b. Destination word j
            // receives one byte per uint32 slot: reconstruct little-endian generation words.
            for (const [byteIndex, copy] of copies.entries()) {
              expect(copy.destinationBuffer).toBe(requested.buffer.defaultBuffer);
              expect(copy.destinationOffset).toBe(requested.byteOffset + 4 * byteIndex);
              expect(copy.size).toBe(4);
              const selected = await copy.sourceBuffer.readAsync(copy.sourceOffset, copy.size);
              const value = new DataView(
                selected.buffer,
                selected.byteOffset,
                selected.byteLength
              ).getUint32(0, true);
              expect(value).toBeGreaterThanOrEqual(0);
              expect(value).toBeLessThan(256);
              words[Math.floor(byteIndex / 4)] =
                (words[Math.floor(byteIndex / 4)] | (value << (8 * (byteIndex % 4)))) >>> 0;
            }
            requests.push(words);
          };
          await encode(false);
          if (request === 'invalidate') producer.invalidate();
          await encode(request === 'predicate'); // discard all recorded commands
          await encode(false); // replacement must still send the unapplied generation
          expect(requests).toEqual([
            [1, 0],
            [2, 0],
            [2, 0]
          ]);
          expect(writeRequested).not.toHaveBeenCalled();
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
              node.resources?.some(
                resource => 'buffer' in resource && resource.buffer === positions
              )
            )
          ).toBe(false);
          expect(nodes.some(node => node.condition?.source === 'gpu')).toBe(true);
        } finally {
          producer.destroy();
          device.destroy();
          vi.restoreAllMocks();
        }
      }
    );
  }
}
