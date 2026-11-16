// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUDatumGridTransform,
  GPU_DATUM_GRID_TRANSFORM_STATUS
} from '../../src/geospatial-analysis/gpu-datum-grid-transform';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const direction of ['forward', 'inverse'] as const) {
  for (const split of [false, true]) {
    it(`datum ${direction} ${split ? 'split words' : 'single words'} matches packed rows and constant-grid arithmetic`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const source = new Float32Array([
        0.5,
        0.5,
        10,
        0.25,
        0.25,
        20,
        0.75,
        0.75,
        30,
        2,
        2,
        40,
        NaN,
        0.5,
        50,
        0.625,
        0.5,
        60,
        0.5,
        0.625,
        70
      ]);
      const lowWords = Float32Array.from(
        Array.from({length: 7}, () => [1 / 1024, 1 / 2048, 0.25]).flat()
      );
      const samplesBuffer = createInputBuffer(
        device,
        new Float32Array([0.25, -0.125, 2, 0.25, -0.125, 2, 0.25, -0.125, 2, 0.25, -0.125, 2])
      );
      const maskBuffer = createInputBuffer(device, new Uint32Array([1, 1, 1, 1]));
      buffers.push(samplesBuffer, maskBuffer);
      const samples = importGraphBuffer(graph, 'samples', samplesBuffer, 'float32x3', 4);
      const mask = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 4);
      const routeOutputs: Record<
        string,
        {buffer: Buffer; prefix: number; length: number; integer: boolean}[]
      >[] = [];
      for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
        const recorded: Record<
          string,
          {buffer: Buffer; prefix: number; length: number; integer: boolean}[]
        > = {};
        const rows = <Format extends 'float32x3' | 'uint32'>(
          name: string,
          format: Format,
          values?: Float32Array
        ) => {
          const components = format === 'float32x3' ? 3 : 1;
          const integer = format === 'uint32';
          let rowBase = 0;
          const chunks = lengths.map((length, chunkIndex) => {
            const prefix = chunkIndex + (values ? 1 : 2);
            const buffer = values
              ? (() => {
                  const padded = new Float32Array(prefix + length * components);
                  padded.set(
                    values.subarray(rowBase * components, (rowBase + length) * components),
                    prefix
                  );
                  return createInputBuffer(device, padded);
                })()
              : createOutputBuffer(device, prefix + length * components);
            buffers.push(buffer);
            const backing = importGraphBuffer(
              graph,
              `${name}-${routeIndex}-${chunkIndex}`,
              buffer,
              integer ? 'uint32' : 'float32',
              prefix + length * components
            );
            rowBase += length;
            if (!values) {
              recorded[name] ??= [];
              recorded[name].push({buffer, prefix, length: length * components, integer});
            }
            return graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
          });
          return routeIndex === 0 ? chunks[0] : createVectorView(name, format, chunks);
        };
        graph.add(
          new GPUDatumGridTransform({
            id: `datum-${routeIndex}`,
            direction,
            operation: 'combined',
            sourceVerticalReference: 'source',
            targetVerticalReference: 'target',
            positions: rows('positions', 'float32x3', source),
            positionsLow: split ? rows('positionsLow', 'float32x3', lowWords) : undefined,
            samples,
            mask,
            grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
            output: {
              positions: rows('output', 'float32x3'),
              positionsLow: split ? rows('outputLow', 'float32x3') : undefined,
              status: rows('status', 'uint32')
            }
          })
        );
        routeOutputs.push(recorded);
      }
      const compiled = graph.compile();
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          const results: Record<string, number[]>[] = [];
          for (const fields of routeOutputs) {
            const result: Record<string, number[]> = {};
            for (const [name, chunks] of Object.entries(fields)) {
              result[name] = [];
              for (const {buffer, prefix, length, integer} of chunks)
                result[name].push(
                  ...(await (integer ? readUint32 : readFloat32)(buffer, prefix + length)).slice(
                    prefix
                  )
                );
            }
            results.push(result);
          }
          expect(results[1]).toEqual(results[0]);
          const sign = direction === 'inverse' ? -1 : 1;
          for (let row = 0; row < 7; row++) {
            const status =
              row === 3
                ? GPU_DATUM_GRID_TRANSFORM_STATUS.noCoverage
                : row === 4
                  ? GPU_DATUM_GRID_TRANSFORM_STATUS.invalid
                  : GPU_DATUM_GRID_TRANSFORM_STATUS.success;
            expect(results[0]['status'][row]).toBe(status);
            for (let component = 0; component < 3; component++) {
              const index = row * 3 + component;
              if (status !== 0) {
                expect(results[0]['output'][index]).toBeNaN();
                if (split) expect(results[0]['outputLow'][index]).toBeNaN();
              } else {
                const displacement =
                  (split ? lowWords[index] : 0) + sign * [0.25, -0.125, 2][component];
                expect(results[0]['output'][index]).toBe(
                  split ? source[index] : source[index] + displacement
                );
                if (split) expect(results[0]['outputLow'][index]).toBe(displacement);
              }
            }
          }
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
