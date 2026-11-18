// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialJoinPrepared} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialPredicateJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUStudyWindowPreparation} from '../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// Acceptance tests for generation acknowledgement after a complete executed preparation.
// The lead runs these on a device; a missing device must never turn a case green.
for (const kind of ['join', 'window'] as const) {
  for (const discard of ['full', 'clean-prefix', 'predicate', 'submitted-prefix'] as const) {
    it(`R1 ${kind} rebuilds a translated square after discarding ${discard}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const input = <Format extends 'float32x2' | 'uint32'>(
        name: string,
        format: Format,
        values: Float32Array | Uint32Array,
        length: number
      ) => {
        const buffer = device.createBuffer({
          data: values,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
      };
      const output = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => {
        const buffer = device.createBuffer({
          byteLength: length * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
      };
      const positions = input(
        'positions',
        'float32x2',
        new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
        4
      );
      const geometry = {
        positions: positions.view,
        featureOffsets: input('features', 'uint32', new Uint32Array([0, 1]), 2).view,
        polygonOffsets: input('polygons', 'uint32', new Uint32Array([0, 1]), 2).view,
        ringOffsets: input('rings', 'uint32', new Uint32Array([0, 4]), 2).view
      };
      const query = input('query', 'float32x2', new Float32Array([0.5, 0.5]), 1);
      const summary = output('summary', 'float32', 8);
      const count = output('count', 'uint32', 1);
      const leftRows = output('left-rows', 'uint32', 1);
      const rightRows = output('right-rows', 'uint32', 1);
      let changed = false;
      const rebuildWhen = discard === 'predicate' ? () => changed : undefined;
      const preparation =
        kind === 'join'
          ? new GPUSpatialJoinPrepared({geometry: {kind: 'polygons', ...geometry}, rebuildWhen})
          : new GPUStudyWindowPreparation({
              ...geometry,
              samplingGrid: [2, 2],
              output: {summary: summary.view},
              rebuildWhen
            });
      graph.add(preparation);
      if (kind === 'join') {
        graph.add(
          new GPUSpatialPredicateJoin({
            left: {kind: 'points', positions: query.view},
            right: {kind: 'polygons', ...geometry},
            prepared: preparation as GPUSpatialJoinPrepared,
            predicate: 'within',
            candidateCapacity: 1,
            output: {
              pairs: {
                leftRows: leftRows.view,
                rightRows: rightRows.view,
                status: {
                  count: count.view,
                  overflow: output('overflow', 'uint32', 1).view
                }
              }
            }
          })
        );
      }
      const compiled = graph.compile();
      const discardEncoding = (prefix: boolean, submitPrefix = false) => {
        const encoder = device.createCommandEncoder();
        try {
          if (prefix) {
            const execution = compiled.createExecution({
              maximumInvocationCount: 1_000_000,
              maximumNodeCount: 1
            });
            expect(execution.encodeNext(encoder, {parameters: undefined}).nodeCount).toBe(1);
          } else {
            compiled.encode(encoder, {parameters: undefined});
          }
          const commandBuffer = encoder.finish();
          if (submitPrefix) device.submit(commandBuffer);
          commandBuffer.destroy();
        } finally {
          encoder.destroy();
        }
      };
      try {
        submitGraph(device, compiled, undefined);
        if (kind === 'join') expect(await readUint32(count.buffer, 1)).toEqual([1]);
        else expect((await readFloat32(summary.buffer, 8)).slice(0, 4)).toEqual([0, 0, 1, 1]);
        if (discard === 'clean-prefix') discardEncoding(true);
        positions.buffer.write(new Float32Array([10, 0, 11, 0, 11, 1, 10, 1]));
        query.buffer.write(new Float32Array([10.5, 0.5]));
        if (discard === 'predicate') changed = true;
        else preparation.invalidate();
        if (discard === 'submitted-prefix') discardEncoding(true, true);
        else if (discard !== 'clean-prefix') discardEncoding(false);
        changed = false;
        submitGraph(device, compiled, undefined);
        if (kind === 'join') {
          expect(await readUint32(count.buffer, 1)).toEqual([1]);
          expect(await readUint32(leftRows.buffer, 1)).toEqual([0]);
          expect(await readUint32(rightRows.buffer, 1)).toEqual([0]);
        } else {
          expect((await readFloat32(summary.buffer, 8)).slice(0, 4)).toEqual([10, 0, 11, 1]);
        }
      } finally {
        compiled.destroy();
        preparation.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
