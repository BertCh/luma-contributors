// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {adaptMultipartNearestUniforms} from './multipart-nearest-executor';
import {
  createPreparedProducerNode,
  createPreparedConsumerNodes,
  getPreparedConsumerExpectations,
  PREPARED_OLD_SQUARE,
  PREPARED_NEW_SQUARE
} from './prepared-consumers-w5-fixture';

useCPUExecutorTestHooks();
for (const retryMode of ['none', 'automatic', 'mapped-buffer', 'mapped-dynamic'] as const)
  for (const edgeSlabs of [false, true])
    it(`GEO-115 W5-01 prepared consumers retain generations retry=${retryMode} edgeSlabs=${edgeSlabs}`, async () => {
      const retry = retryMode !== 'none';
      const device = createNullWebGPUDevice();
      let graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const wrappers: DynamicBuffer[] = [];
      const input = <Format extends 'float32x2' | 'float32' | 'uint32'>(
        name: string,
        format: Format,
        values: Float32Array | Uint32Array
      ): GraphDataView<Format> => {
        const buffer = device.createBuffer({
          byteLength: 512,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffer.write(values, 0);
        buffers.push(buffer);
        const handle = importGraphBuffer(graph, name, buffer, 'uint32', 128).buffer;
        return graph.createDataView(handle, {
          format,
          length: values.length / (format === 'float32x2' ? 2 : 1),
          byteOffset: 0
        });
      };
      let geometry = {
        kind: 'polygons' as const,
        positions: input('positions', 'float32x2', PREPARED_OLD_SQUARE),
        featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
        polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
        ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
        sourceIds: input('ids', 'uint32', Uint32Array.of(99))
      };
      if (retryMode === 'mapped-dynamic') {
        const wrapper = new DynamicBuffer(device, {
          buffer: geometry.positions.buffer.defaultBuffer as Buffer,
          ownsBuffer: false
        });
        wrappers.push(wrapper);
        geometry = {
          ...geometry,
          positions: graph.importGPUData(
            'dynamic-positions',
            new GPUData({
              buffer: wrapper,
              format: 'float32x2',
              length: geometry.positions.length,
              byteOffset: geometry.positions.byteOffset,
              ownsBuffer: false
            })
          )
        };
      }
      const prepared = new GPUSpatialJoinPrepared<void>({id: 'generation', geometry, edgeSlabs});
      try {
        if (retry) {
          const allocate = device.createBuffer.bind(device);
          const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
            if (
              typeof settings === 'object' &&
              settings !== null &&
              'id' in settings &&
              settings.id === 'generation-node-minima'
            )
              throw new Error('injected minima');
            return allocate(settings);
          });
          expect(() => graph.add(prepared)).toThrow('injected minima');
          spy.mockRestore();
          expect(prepared.isDeclaredIn(graph)).toBe(false);
          expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
          graph = new GPUCommandGraph(device);
          // Equivalent ranges share the fresh producer's canonical handle. The device sibling uses offset 256;
          // wgsl_reflect ignores typed-array byteOffset (see scratch offset-probe.mjs).
          geometry = {
            ...geometry,
            ...Object.fromEntries(
              Object.entries(geometry)
                .filter(([, value]) => typeof value === 'object')
                .map(([name, value]) => {
                  const source = value as GraphDataView;
                  const physical = getCoreDefaultBuffer(source.buffer.defaultBuffer!);
                  const handle = retryMode.startsWith('mapped-')
                    ? graph.importBuffer(
                        {
                          id: `canonical-${name}`,
                          byteLength: physical.byteLength,
                          usage: Buffer.STORAGE
                        },
                        physical
                      )
                    : importGraphBuffer(graph, `canonical-${name}`, physical, 'uint32', 128).buffer;
                  return [name, graph.createDataView(handle, source)];
                })
            )
          };
          if (retryMode.startsWith('mapped-')) prepared.setRetryViews(graph, {geometry});
        }
        const producerValues = input('producer-values', 'float32x2', PREPARED_OLD_SQUARE);
        const producer = createPreparedProducerNode(graph, producerValues, geometry.positions);
        const preparation = prepared.getCommandNodes(graph);
        const snapshotSource = preparation.find(
          node => node.id === 'generation-snapshot-positions'
        )!.resources![0];
        expect(
          'buffer' in snapshotSource &&
            'buffer' in snapshotSource.buffer &&
            snapshotSource.buffer.buffer
        ).toBe(geometry.positions.buffer);
        const query = createTransientView(graph, 'query', 'float32x2', 1);
        const radius = input('radius', 'float32', Float32Array.of(1));
        const {nodes, outputs} = createPreparedConsumerNodes(
          graph,
          prepared,
          geometry,
          query,
          radius,
          name => createTransientView(graph, name, 'uint32', 1)
        );
        const different = {
          ...geometry,
          positions: graph.createDataView(geometry.positions.buffer, {
            ...geometry.positions,
            byteOffset: 8
          })
        };
        for (let consumerIndex = 0; consumerIndex < 5; consumerIndex++)
          expect(() =>
            createPreparedConsumerNodes(
              graph,
              prepared,
              different,
              query,
              radius,
              name => createTransientView(graph, `different-${consumerIndex}-${name}`, 'uint32', 1),
              consumerIndex
            )
          ).toThrow(/same.*(views|features)/);
        const memory = createCPUGraphMemory();
        const allNodes = [producer, ...preparation, ...nodes];
        for (const node of allNodes)
          for (const resource of node.resources ?? []) {
            if ('buffer' in resource) {
              const handle = 'buffer' in resource.buffer ? resource.buffer.buffer : resource.buffer;
              if (handle.defaultBuffer)
                new Uint8Array(memory.getBuffer(handle).backing).set(
                  await handle.defaultBuffer.readAsync()
                );
            }
          }
        memory.write(query, Float32Array.of(0, 0));
        memory.write(geometry.positions, PREPARED_NEW_SQUARE); // Producer must restore the old square before capture.
        const run = (hit: boolean, sourceId: number) => {
          const executable = allNodes.map(node => {
            if (node.condition?.source === 'cpu') {
              node.condition.evaluate(undefined);
              return {...node, condition: undefined};
            }
            return node;
          });
          executeCommandNodesOnCPU(adaptMultipartNearestUniforms(executable), memory);
          for (const [name, expected] of Object.entries(
            getPreparedConsumerExpectations(hit, sourceId)
          ))
            expect(Array.from(memory.readUint32(outputs[name])), name).toEqual(expected);
        };
        run(true, 99);
        memory.write(producerValues, PREPARED_NEW_SQUARE);
        memory.write(geometry.ringOffsets, Uint32Array.of(0, 0));
        memory.write(geometry.sourceIds, Uint32Array.of(77));
        run(true, 77); // Coordinates retained, application IDs remain live.
        prepared.invalidate();
        run(false, 77);
        memory.write(geometry.ringOffsets, Uint32Array.of(0, 4));
        memory.write(query, Float32Array.of(10, 10));
        prepared.invalidate();
        run(true, 77);
        prepared.destroy();
        expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
        for (const buffer of buffers) expect(buffer.destroyed).toBe(false);
      } finally {
        prepared.destroy();
        for (const wrapper of wrappers) wrapper.destroy();
        for (const buffer of buffers) buffer.destroy();
        device.destroy();
      }
    });
