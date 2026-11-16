// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
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
  it(`${label} destinations match packed rows exactly and the equatorial closed form`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const origins = Float32Array.from(Array.from({length: 7}, (_, row) => [row, 0]).flat());
    const radius = metric === 'ellipsoidal' ? (ellipsoid?.semiMajorAxis ?? 6378137) : 6371008.8;
    const distances = Float32Array.from(
      Array.from({length: 7}, (_, row) => (row * radius * Math.PI) / 180)
    );
    const bearings = new Float32Array(7).fill(90);
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
        finalBearings: rows('final', 'float32'),
        destinations: rows('destinations', 'float32x2'),
        status: {
          ...(metric === 'ellipsoidal' ? {converged: rows('converged', 'uint32')} : {}),
          approximation: rows('approximation', 'uint32'),
          iterationLimitReached: rows('iterationLimitReached', 'uint32')
        }
      };
      graph.add(
        new GPUGeodesicDestination({
          id: `destination-${routeIndex}`,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric,
            ...(ellipsoid ? {ellipsoid} : {})
          },
          origins: rows('origins', 'float32x2', origins),
          bearings: rows('bearings', 'float32', bearings),
          distances: rows('distances', 'float32', distances),
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
        for (let row = 0; row < 7; row++) {
          if (row > 0) expect(results[0]['final'][row]).toBeCloseTo(90, 3);
          expect(results[0]['destinations'][row * 2]).toBeCloseTo(row * 2, 4);
          expect(results[0]['destinations'][row * 2 + 1]).toBeCloseTo(0, 4);
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
