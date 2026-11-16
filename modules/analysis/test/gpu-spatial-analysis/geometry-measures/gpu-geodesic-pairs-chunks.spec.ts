// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const configuration of [
  {metric: 'great-circle'},
  {metric: 'rhumb'},
  {metric: 'ellipsoidal'},
  {metric: 'ellipsoidal', ellipsoid: {semiMajorAxis: 7000000, flattening: 0}}
] as const) {
  const {metric} = configuration;
  const ellipsoid = 'ellipsoid' in configuration ? configuration.ellipsoid : undefined;
  const label = ellipsoid ? 'custom zero-flattening ellipsoid' : metric;
  it(`${label} pairs match packed rows exactly and the equatorial closed form`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const origins = Float32Array.from(Array.from({length: 7}, (_, row) => [row, 0]).flat());
    const targets = Float32Array.from(Array.from({length: 7}, (_, row) => [row * 2, 0]).flat());
    const routeOutputs: Record<
      string,
      {buffer: Buffer; prefix: number; length: number; integer: boolean}[]
    >[] = [];
    for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
      const recorded: Record<
        string,
        {buffer: Buffer; prefix: number; length: number; integer: boolean}[]
      > = {};
      const rows = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        name: string,
        format: Format,
        values?: Float32Array
      ) => {
        const components = format === 'float32x2' ? 2 : 1;
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
      const output = {
        distances: rows('distances', 'float32'),
        initialBearings: rows('initial', 'float32'),
        finalBearings: rows('final', 'float32'),
        midpoints: rows('midpoints', 'float32x2'),
        status: {
          ...(metric === 'ellipsoidal' ? {converged: rows('converged', 'uint32')} : {}),
          approximation: rows('approximation', 'uint32'),
          iterationLimitReached: rows('iterationLimitReached', 'uint32')
        }
      };
      graph.add(
        new GPUGeodesicPairs({
          id: `pairs-${routeIndex}`,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric,
            ...(ellipsoid ? {ellipsoid} : {})
          },
          origins: rows('origins', 'float32x2', origins),
          targets: rows('targets', 'float32x2', targets),
          output
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
        const radius = metric === 'ellipsoidal' ? (ellipsoid?.semiMajorAxis ?? 6378137) : 6371008.8;
        for (let row = 0; row < 7; row++) {
          const expected = (row * radius * Math.PI) / 180;
          expect(Math.abs(results[0]['distances'][row] - expected)).toBeLessThanOrEqual(
            Math.max(expected * 3e-6, 1e-6)
          );
          if (row > 0) {
            expect(results[0]['initial'][row]).toBeCloseTo(90, 3);
            expect(results[0]['final'][row]).toBeCloseTo(90, 3);
          }
          expect(results[0]['midpoints'][row * 2]).toBeCloseTo(row * 1.5, 4);
          expect(results[0]['midpoints'][row * 2 + 1]).toBeCloseTo(0, 4);
          expect(results[0]['approximation'][row]).toBe(0);
          expect(results[0]['iterationLimitReached'][row]).toBe(0);
          if (metric === 'ellipsoidal') expect(results[0]['converged'][row]).toBe(1);
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

it('writes disjoint aligned status lanes in one buffer for packed rows and three uneven chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const guard = 0xa5a5a5a5;
  const routes: {buffer: Buffer; expected: Uint32Array; statusRows: number[][]}[] = [];
  const inputValues = new Float32Array(14);
  const targetValues = Float32Array.from(Array.from({length: 7}, (_, row) => [row + 1, 0]).flat());
  for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
    const statusValues = new Uint32Array(lengths.length * 3 * 64).fill(guard);
    const expected = statusValues.slice();
    const buffer = createOutputBuffer(device, statusValues.length);
    buffer.write(statusValues);
    buffers.push(buffer);
    const status = importGraphBuffer(
      graph,
      `shared-status-${routeIndex}`,
      buffer,
      'uint32',
      statusValues.length
    );
    const statusRows = [[], [], []] as number[][];
    const outputs = lengths.map((length, chunkIndex) => {
      return [0, 1, 2].map(fieldIndex => {
        const wordBase = (chunkIndex * 3 + fieldIndex) * 64;
        expected.fill(fieldIndex === 0 ? 1 : 0, wordBase, wordBase + length);
        for (let row = 0; row < length; row++) statusRows[fieldIndex].push(wordBase + row);
        return graph.createDataView(status.buffer, {
          format: 'uint32',
          length,
          byteOffset: wordBase * 4
        });
      });
    });
    const inputRows = (name: string, values: Float32Array) => {
      let rowBase = 0;
      const chunks = lengths.map((length, chunkIndex) => {
        const inputBuffer = createInputBuffer(
          device,
          values.slice(rowBase * 2, (rowBase + length) * 2)
        );
        buffers.push(inputBuffer);
        rowBase += length;
        return importGraphBuffer(
          graph,
          `${name}-${routeIndex}-${chunkIndex}`,
          inputBuffer,
          'float32x2',
          length
        );
      });
      return routeIndex === 0 ? chunks[0] : createVectorView(name, 'float32x2', chunks);
    };
    const outputRows = (fieldIndex: number) => {
      const chunks = outputs.map(fields => fields[fieldIndex]);
      return routeIndex === 0
        ? chunks[0]
        : createVectorView(`status-${fieldIndex}`, 'uint32', chunks);
    };
    graph.add(
      new GPUGeodesicPairs({
        id: `shared-status-pairs-${routeIndex}`,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
        origins: inputRows('origins', inputValues),
        targets: inputRows('targets', targetValues),
        output: {
          status: {
            converged: outputRows(0),
            approximation: outputRows(1),
            iterationLimitReached: outputRows(2)
          }
        }
      })
    );
    routes.push({buffer, expected, statusRows});
  }
  const compiled = graph.compile();
  try {
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      const results: number[][][] = [];
      for (const {buffer, expected, statusRows} of routes) {
        const values = await readUint32(buffer, expected.length);
        // Short equatorial pairs converge without an approximation or exhausted iterations.
        // Check the full allocation too: every alignment-padding guard must survive dispatch.
        expect(values).toEqual(Array.from(expected));
        results.push(statusRows.map(indices => indices.map(index => values[index])));
      }
      expect(results[1]).toEqual(results[0]);
      expect(results[0]).toEqual([
        new Array(7).fill(1),
        new Array(7).fill(0),
        new Array(7).fill(0)
      ]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
