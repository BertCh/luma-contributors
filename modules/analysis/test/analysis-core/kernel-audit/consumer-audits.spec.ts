// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUKernelFlowSmoothing} from '../../../src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {expectJoinedFloatAggregates, expectNearestTieResult} from './consumer-audit-expectations';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const mode of [
  'clean',
  'overflow',
  'candidate-overflow',
  'unknown-candidates',
  'legacy-import'
] as const) {
  it(`CPL-09 flow certificate on device: ${mode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array,
      length = values.length
    ) => {
      const buffer = device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const count = input('count', 'uint32', Uint32Array.of(2));
    const complete = input('complete', 'uint32', Uint32Array.of(99));
    const smoothed = input('smoothed', 'float32', new Float32Array(4));
    const contributor = new GPUKernelFlowSmoothing({
      positions: input('positions', 'float32x2', Float32Array.of(0, 0, 1, 0), 2).view,
      locationWeights: input('locations', 'float32', Float32Array.of(1, 1)).view,
      parameters: input('parameters', 'float32', Float32Array.of(0, 0.5, 0, 0)).view,
      flows: {
        kind: 'pairs',
        originZoneIds: input('origins', 'uint32', Uint32Array.of(0, 1)).view,
        destinationZoneIds: input('destinations', 'uint32', Uint32Array.of(1, 0)).view,
        weights: input('weights', 'float32', Float32Array.of(2, 1)).view,
        ...(mode === 'legacy-import'
          ? {count: count.view, completeness: 'known-complete' as const}
          : {
              status: {
                count: count.view,
                overflow: input('overflow', 'uint32', Uint32Array.of(mode === 'overflow' ? 1 : 0))
                  .view,
                ...(mode === 'unknown-candidates'
                  ? {}
                  : {
                      candidateOverflow: input(
                        'candidates',
                        'uint32',
                        Uint32Array.of(mode === 'candidate-overflow' ? 1 : 0)
                      ).view
                    })
              }
            })
      },
      output: {smoothed: smoothed.view, status: {complete: complete.view}}
    });
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      graph.add(contributor);
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const isComplete = mode === 'clean' || mode === 'legacy-import';
      expect(await readUint32(complete.buffer, 1)).toEqual([isComplete ? 1 : 0]);
      if (isComplete) {
        // Distinct zones and p=0 make the smoothing matrix the identity: 0→1=2, 1→0=1.
        const actual = await readFloat32(smoothed.buffer, 4);
        expect(actual.every(Number.isFinite)).toBe(true);
        expect(actual).toEqual([0, 2, 1, 0]);
      }
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const [ties, capacity] of [
  ['lowest-row', 1],
  ['all', 2],
  ['all', 1]
] as const) {
  it(`NAM-03 nearest tied application IDs retain row ordering: ${ties} capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array,
      length = values.length
    ) => {
      const buffer = device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const rows = input('rows', 'uint32', new Uint32Array(capacity));
    const ids = input('ids', 'uint32', new Uint32Array(capacity));
    const count = input('count', 'uint32', new Uint32Array(1));
    const distances = input('distances', 'float32', new Float32Array(capacity));
    const overflow = input('overflow', 'uint32', new Uint32Array(1));
    const contributor = new GPUNearestFeatureJoin({
      queries: {
        kind: 'points',
        positions: input('query', 'float32x2', Float32Array.of(0, 0), 1).view
      },
      features: {
        kind: 'points',
        positions: input('features', 'float32x2', Float32Array.of(-1, 0, 1, 0), 2).view,
        sourceIds: input('source-ids', 'uint32', Uint32Array.of(99, 1)).view
      },
      neighborCount: 1,
      neighborCapacity: capacity,
      ties,
      output: {
        neighborRows: rows.view,
        neighborIds: ids.view,
        neighborCounts: count.view,
        neighborDistances: distances.view,
        status: {overflow: overflow.view}
      }
    });
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      graph.add(contributor);
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const size = ties === 'all' && capacity > 1 ? 2 : 1;
      expectNearestTieResult({
        ties,
        capacity,
        count: (await readUint32(count.buffer, 1))[0],
        rows: await readUint32(rows.buffer, size),
        ids: await readUint32(ids.buffer, size),
        distances: await readFloat32(distances.buffer, size),
        overflow: (await readUint32(overflow.buffer, 1))[0]
      });
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const how of ['left', 'inner'] as const) {
  it(`W3-03 key lookup scale, accumulation and publication submit: ${how}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array,
      length = values.length
    ) => {
      const buffer = device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const sums = input('sum-values', 'float32', new Float32Array(2));
    const means = input('means', 'float32', new Float32Array(2));
    const words = input('sum-words', 'uint32x2', new Uint32Array(4), 2);
    const count = input('compact-count', 'uint32', new Uint32Array(1));
    const compactRows = input('compact-rows', 'uint32', new Uint32Array(2));
    const values = input('values', 'float32', Float32Array.of(1, 2)).view;
    const contributor = new GPUKeyLookup({
      leftKeys: input('left-keys', 'uint32', Uint32Array.of(7, 8)).view,
      rightKeys: input('right-keys', 'uint32', Uint32Array.of(7, 7)).view,
      compactMatchedLeftRows: how === 'inner',
      sumScale: 1024,
      aggregates: [
        {operation: 'sum', column: values, output: sums.view, sums: words.view},
        {operation: 'mean', column: values, output: means.view}
      ],
      output: {
        rightRows: input('right-rows', 'uint32', new Uint32Array(2)).view,
        ...(how === 'inner'
          ? {
              rows: {
                ids: compactRows.view,
                status: {
                  count: count.view,
                  overflow: input('overflow', 'uint32', new Uint32Array(1)).view
                }
              }
            }
          : {})
      }
    });
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      graph.add(contributor);
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      // Key 7 matches [1,2]: sum=3, mean=3/2. Key 8 has missing floats and zero words.
      const actualSums = await readFloat32(sums.buffer, 2);
      const actualMeans = await readFloat32(means.buffer, 2);
      expectJoinedFloatAggregates(actualSums, actualMeans);
      expect(await readUint32(words.buffer, 4)).toEqual([3072, 0, 0, 0]);
      if (how === 'inner') {
        expect(await readUint32(count.buffer, 1)).toEqual([1]);
        expect(await readUint32(compactRows.buffer, 1)).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
