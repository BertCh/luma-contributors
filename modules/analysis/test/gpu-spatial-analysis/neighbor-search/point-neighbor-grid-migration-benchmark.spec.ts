// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Opt-in timings on clustered points for the point neighbor index work: OPTICS end to end, kNN cost
// by k, and space-time search by time bucket count. Median encode-to-readback, in milliseconds.
//   VITE_POINT_NEIGHBOR_BENCHMARK=true npx vitest run --project headless --silent=false <this file>

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {runHDBSCAN} from '../spatial-clustering/hierarchical-density-clustering-harness';
import {
  createClusteredPoints,
  createSeededRandom
} from '../spatial-clustering/spatial-clustering-oracle';

const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_POINT_NEIGHBOR_BENCHMARK'
  ] === 'true';
const WARMUP_COUNT = 2;
const SAMPLE_COUNT = 7;
const EXTENT = 1000;

function createScene(rows: number): Float32Array {
  return createClusteredPoints(7, {
    pointCount: rows,
    blobCount: 4,
    extent: EXTENT,
    blobSigma: 8,
    noiseFraction: 0.1
  });
}

async function timeEncodings(
  device: Device,
  compiled: ReturnType<GPUCommandGraph['compile']>,
  probe: Buffer
): Promise<number> {
  const samples: number[] = [];
  for (let run = 0; run < WARMUP_COUNT + SAMPLE_COUNT; run++) {
    const start = performance.now();
    submitGraph(device, compiled, undefined);
    await readUint32(probe, 1);
    if (run >= WARMUP_COUNT) samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

async function timeSearch(
  device: Device,
  points: Float32Array,
  options: {
    neighborCount?: number;
    radius?: number;
    times?: Float32Array;
    timeRadius?: number;
    timeBucketCount?: number;
    capacity: number;
  }
): Promise<{milliseconds: number; total: number}> {
  const rows = points.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'point-neighbor-benchmark'});
  const buffers: Buffer[] = [];
  const create = <T extends Buffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  const total = create(createOutputBuffer(device, 1));
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: options.times
      ? GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
      : GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues({
      bounds: [0, 0, EXTENT, EXTENT],
      searchRadius: options.radius,
      timeRadius: options.timeRadius,
      timeBounds: options.times ? [0, 1000] : undefined
    })
  });
  const uintOutput = (name: string, length: number) =>
    importGraphBuffer(graph, name, create(createOutputBuffer(device, length)), 'uint32', length);
  const floatOutput = (name: string, length: number) =>
    importGraphBuffer(graph, name, create(createOutputBuffer(device, length)), 'float32', length);
  const columns = options.radius
    ? Math.min(1024, Math.floor(EXTENT / options.radius))
    : Math.ceil(Math.sqrt(rows / 2));
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      new GPUNeighborSearch({
        mode: options.radius ? 'radius' : 'knn',
        neighborCount: options.neighborCount,
        gridSize: [columns, columns],
        positions: importGraphBuffer(
          graph,
          'positions',
          create(createInputBuffer(device, points)),
          'float32x2',
          rows
        ),
        times: options.times
          ? importGraphBuffer(
              graph,
              'times',
              create(createInputBuffer(device, options.times)),
              'float32',
              rows
            )
          : undefined,
        temporalContext: options.times
          ? {unit: 'second', epoch: 0n, encoding: 'float32-relative', axis: {kind: 'irregular'}}
          : undefined,
        timeBucketCount: options.timeBucketCount,
        parameters: parameters.importToGraph(graph),
        output: {
          offsets: uintOutput('offsets', rows + 1),
          neighbors: uintOutput('neighbors', options.capacity),
          weights: floatOutput('weights', options.capacity),
          status: {
            overflow: uintOutput('overflow', 1),
            requiredCount: importGraphBuffer(graph, 'total', total, 'uint32', 1)
          }
        }
      })
    );
    compiled = await graph.compileAsync();
    const milliseconds = await timeEncodings(device, compiled, total);
    return {milliseconds, total: (await readUint32(total, 1))[0]};
  } finally {
    compiled?.destroy();
    parameters.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

async function timeOPTICS(
  device: Device,
  rows: number,
  minimumSamples: number,
  maximumDistance: number
): Promise<number> {
  const graph = new GPUCommandGraph(device, {id: 'optics-benchmark'});
  const buffers: Buffer[] = [];
  const create = <T extends Buffer>(buffer: T): T => {
    buffers.push(buffer);
    return buffer;
  };
  const iterationCount = create(createOutputBuffer(device, 1));
  const uintOutput = (name: string, length: number) =>
    importGraphBuffer(graph, name, create(createOutputBuffer(device, length)), 'uint32', length);
  const floatOutput = (name: string, length: number) =>
    importGraphBuffer(graph, name, create(createOutputBuffer(device, length)), 'float32', length);
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    graph.add(
      new GPUOPTICS({
        positions: importGraphBuffer(
          graph,
          'positions',
          create(createInputBuffer(device, createScene(rows))),
          'float32x2',
          rows
        ),
        minimumSamples,
        maximumDistance,
        extractionDistance: maximumDistance,
        output: {
          ordering: uintOutput('ordering', rows),
          coreDistances: floatOutput('core', rows),
          reachability: floatOutput('reachability', rows),
          predecessors: uintOutput('predecessors', rows),
          labels: uintOutput('labels', rows),
          ports: {
            status: {
              iterationCount: importGraphBuffer(
                graph,
                'iteration-count',
                iterationCount,
                'uint32',
                1
              )
            }
          }
        }
      })
    );
    compiled = await graph.compileAsync();
    return await timeEncodings(device, compiled, iterationCount);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it.skipIf(!RUN_BENCHMARK)(
  'point neighbor index timings on clustered points',
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lines: string[] = [];
    for (const [rows, minimumSamples] of [
      [4096, 5],
      [4096, 64]
    ]) {
      const milliseconds = await timeOPTICS(device, rows, minimumSamples, 60);
      lines.push(
        `optics rows=${rows} minimumSamples=${minimumSamples}: ${milliseconds.toFixed(1)} ms`
      );
    }
    const rows = 100_000;
    const points = createScene(rows);
    // Worst case for a uniform grid over the data extent: one far outlier collapses every other
    // point into very few cells.
    const outlierPoints = Float32Array.from(points);
    outlierPoints[0] = 1e6;
    outlierPoints[1] = 1e6;
    for (const [name, scene] of [
      ['clustered', points],
      ['clustered plus far outlier', outlierPoints]
    ] as const) {
      const result = await runHDBSCAN(device, scene, {
        minimumClusterSize: 50,
        minimumSamples: 10,
        timedEncodings: 3
      });
      lines.push(
        `hdbscan ${name} rows=${rows}: ${result.medianEncodingMilliseconds?.toFixed(1)} ms, clusters ${result.clusterCount}`
      );
    }
    for (const neighborCount of [8, 32, 64, 128, 256]) {
      const result = await timeSearch(device, points, {
        neighborCount,
        capacity: rows * neighborCount
      });
      lines.push(`knn rows=${rows} k=${neighborCount}: ${result.milliseconds.toFixed(1)} ms`);
      expect(result.total).toBeGreaterThan(0);
    }
    const random = createSeededRandom(9);
    const times = Float32Array.from({length: rows}, () => Math.fround(random() * 1000));
    for (const timeBucketCount of [1, 8, 64]) {
      for (const [name, search] of [
        ['radius', {radius: 5, capacity: rows * 256}],
        ['knn', {neighborCount: 16, capacity: rows * 16}]
      ] as const) {
        const result = await timeSearch(device, points, {
          ...search,
          times,
          timeRadius: 25,
          timeBucketCount
        });
        lines.push(
          `space-time ${name} rows=${rows} window=5% buckets=${timeBucketCount}: ${result.milliseconds.toFixed(1)} ms, neighbors ${result.total}`
        );
        expect(result.total).toBeGreaterThan(0);
      }
    }
    // eslint-disable-next-line no-console
    console.log(lines.join('\n'));
  },
  900_000
);
