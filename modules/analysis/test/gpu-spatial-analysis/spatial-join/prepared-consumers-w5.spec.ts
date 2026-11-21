// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/contracts/network-topology';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createPreparedProducerNode,
  createPreparedConsumerNodes,
  getPreparedConsumerExpectations,
  PREPARED_OLD_SQUARE,
  PREPARED_NEW_SQUARE
} from './prepared-consumers-w5-fixture';

for (const retryMode of ['none', 'automatic', 'mapped-buffer', 'mapped-dynamic'] as const)
  for (const edgeSlabs of [false, true])
    it(`WebGPU GEO-115 W5-01 prepared consumers retry=${retryMode} edgeSlabs=${edgeSlabs}`, async () => {
      const retry = retryMode !== 'none';
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
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
        buffer.write(values, 256);
        buffers.push(buffer);
        const handle = importGraphBuffer(graph, name, buffer, 'uint32', 128).buffer;
        return graph.createDataView(handle, {
          format,
          length: values.length / (format === 'float32x2' ? 2 : 1),
          byteOffset: 256
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
      let compiled: ReturnType<typeof graph.compile> | undefined;
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
          try {
            expect(() => graph.add(prepared)).toThrow('injected minima');
          } finally {
            spy.mockRestore();
          }
          expect(prepared.isDeclaredIn(graph)).toBe(false);
          expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
          graph = new GPUCommandGraph(device);
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
        graph.add(createPreparedProducerNode(graph, producerValues, geometry.positions));
        geometry.positions.buffer.defaultBuffer!.write(
          PREPARED_NEW_SQUARE,
          geometry.positions.byteOffset
        );
        graph.add(prepared);
        const query = input('query', 'float32x2', Float32Array.of(0, 0));
        const radius = input('radius', 'float32', Float32Array.of(1));
        const outputBuffers: Record<string, Buffer> = {};
        const {nodes} = createPreparedConsumerNodes(
          graph,
          prepared,
          geometry,
          query,
          radius,
          name => {
            const buffer = device.createBuffer({
              byteLength: 4,
              usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
            });
            buffers.push(buffer);
            outputBuffers[name] = buffer;
            return importGraphBuffer(graph, name, buffer, 'uint32', 1);
          }
        );
        for (const node of nodes) graph.add(node);
        compiled = graph.compile();
        const run = async (hit: boolean, sourceId: number) => {
          submitGraph(device, compiled!, undefined);
          for (const [name, expected] of Object.entries(
            getPreparedConsumerExpectations(hit, sourceId)
          ))
            expect(await readUint32(outputBuffers[name], 1), name).toEqual(expected);
        };
        await run(true, 99);
        producerValues.buffer.defaultBuffer!.write(PREPARED_NEW_SQUARE, producerValues.byteOffset);
        geometry.ringOffsets.buffer.defaultBuffer!.write(
          Uint32Array.of(0, 0),
          geometry.ringOffsets.byteOffset
        );
        geometry.sourceIds.buffer.defaultBuffer!.write(
          Uint32Array.of(77),
          geometry.sourceIds.byteOffset
        );
        await run(true, 77);
        prepared.invalidate();
        await run(false, 77);
        geometry.ringOffsets.buffer.defaultBuffer!.write(
          Uint32Array.of(0, 4),
          geometry.ringOffsets.byteOffset
        );
        query.buffer.defaultBuffer!.write(Float32Array.of(10, 10), query.byteOffset);
        prepared.invalidate();
        await run(true, 77);
      } finally {
        compiled?.destroy();
        prepared.destroy();
        expect(() => prepared.getGeometry(graph)).toThrow(/graph.add/);
        for (const wrapper of wrappers) wrapper.destroy();
        for (const buffer of buffers) {
          expect(buffer.destroyed).toBe(false);
          buffer.destroy();
        }
      }
    });
