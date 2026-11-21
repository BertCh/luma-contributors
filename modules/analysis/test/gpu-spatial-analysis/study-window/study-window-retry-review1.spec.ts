// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/core/dynamic';
import {GPUData} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_STUDY_WINDOW_SUMMARY} from '../../../src/analysis-core/contracts/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const dynamic of [false, true])
  it(`WebGPU review B2 study-window canonical retry dynamic=${dynamic}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const original = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const wrappers: DynamicBuffer[] = [];
    const input = <Format extends 'float32x2' | 'uint32'>(
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
      const wrapper = new DynamicBuffer(device, {buffer, ownsBuffer: false});
      wrappers.push(wrapper);
      return original.importGPUData(
        name,
        new GPUData({
          buffer: dynamic ? wrapper : buffer,
          format,
          length: values.length / (format === 'float32x2' ? 2 : 1),
          byteOffset: 256,
          ownsBuffer: false
        })
      );
    };
    const inputs = {
      positions: input('positions', 'float32x2', new Float32Array(6)),
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 3))
    };
    const prepared = new GPUStudyWindowPreparation({
      id: 'retry-window',
      ...inputs,
      samplingGrid: [1, 1]
    });
    const allocate = device.createBuffer.bind(device);
    const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
      if (
        typeof settings === 'object' &&
        settings !== null &&
        'id' in settings &&
        settings.id === 'retry-window-edges'
      )
        throw new Error('injected edges');
      return allocate(settings);
    });
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      try {
        expect(() => original.add(prepared)).toThrow('injected edges');
      } finally {
        spy.mockRestore();
      }
      expect(prepared.isDeclaredIn(original)).toBe(false);
      const fresh = new GPUCommandGraph(device);
      const rebind = <Format extends 'float32x2' | 'uint32'>(
        name: string,
        source: GraphDataView<Format>,
        index: number
      ) => {
        const handle = fresh.importBuffer(
          {id: `canonical-${name}`, byteLength: 512, usage: Buffer.STORAGE},
          buffers[index]
        );
        return fresh.createDataView(handle, source);
      };
      const mapped = {
        positions: rebind('positions', inputs.positions, 0),
        featureOffsets: rebind('features', inputs.featureOffsets, 1),
        polygonOffsets: rebind('polygons', inputs.polygonOffsets, 2),
        ringOffsets: rebind('rings', inputs.ringOffsets, 3)
      };
      prepared.setRetryViews(fresh, mapped);
      fresh.add(
        createWGSLKernelNode(fresh, {
          id: 'window-producer',
          operation: 'StudyWindowRetryReview',
          invocationCount: 1,
          bindings: [
            {name: 'positions', view: mapped.positions, type: 'f32', access: 'read_write'}
          ],
          body: 'positions[positionsOffset] = 0.0; positions[positionsOffset + 1u] = 0.0; positions[positionsOffset + 2u] = 2.0; positions[positionsOffset + 3u] = 0.0; positions[positionsOffset + 4u] = 0.0; positions[positionsOffset + 5u] = 2.0;'
        })
      );
      fresh.add(prepared);
      const summaryBuffer = prepared.getViews(fresh).summary.buffer.defaultBuffer as Buffer;
      compiled = fresh.compile();
      for (let submission = 0; submission < 2; submission++) {
        submitGraph(device, compiled, undefined);
        const summary = await readFloat32(summaryBuffer, 8);
        expect(summary).toHaveLength(8);
        expect(summary.every(Number.isFinite)).toBe(true);
        // Hand-worked right triangle: bounds [0,0,2,2], area 2, three edges, perimeter 4+sqrt(8).
        expect(summary.slice(0, 4)).toEqual([0, 0, 2, 2]);
        expect(summary[GPU_STUDY_WINDOW_SUMMARY.area]).toBe(2);
        expect(summary[GPU_STUDY_WINDOW_SUMMARY.edgeCount]).toBe(3);
        expect(summary[GPU_STUDY_WINDOW_SUMMARY.flags]).toBe(0);
        expect(
          Math.abs(summary[GPU_STUDY_WINDOW_SUMMARY.perimeter] - 6.82842712474619)
        ).toBeLessThanOrEqual(2e-6);
      }
    } finally {
      spy.mockRestore();
      compiled?.destroy();
      prepared.destroy();
      for (const wrapper of wrappers) wrapper.destroy();
      for (const buffer of buffers) {
        expect(buffer.destroyed).toBe(false);
        buffer.destroy();
      }
    }
  });
