// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Opt-in bounded performance evidence. Measures upload, graph construction, compilation,
 * first/warm encode+submit+queue completion, changed-input update and optional readback separately.
 * Every OPTICS size completes its full ordering through resident frontier batches.
 */
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Independent f64 CPU baseline for self-inclusive minimumSamples=2, used only by this benchmark. */
function measureCPUOrdering(positions: Float32Array, maximumDistance: number) {
  const started = performance.now();
  const count = positions.length / 2;
  const coreDistances = new Float64Array(count).fill(Infinity);
  const reachability = new Float64Array(count).fill(Infinity);
  const processed = new Uint8Array(count);
  const distance = (first: number, second: number) =>
    Math.hypot(
      positions[first * 2] - positions[second * 2],
      positions[first * 2 + 1] - positions[second * 2 + 1]
    );
  for (let first = 0; first < count; first++) {
    for (let second = 0; second < count; second++) {
      if (first !== second) {
        const value = distance(first, second);
        if (value <= maximumDistance) {
          coreDistances[first] = Math.min(coreDistances[first], value);
        }
      }
    }
  }
  for (let iteration = 0; iteration < count; iteration++) {
    let focus = -1;
    for (let row = 0; row < count; row++) {
      if (!processed[row] && (focus < 0 || reachability[row] < reachability[focus])) {
        focus = row;
      }
    }
    processed[focus] = 1;
    for (let row = 0; row < count; row++) {
      if (processed[row]) {
        continue;
      }
      const value = distance(focus, row);
      if (value <= maximumDistance) {
        reachability[row] = Math.min(reachability[row], Math.max(value, coreDistances[focus]));
      }
    }
  }
  return {milliseconds: performance.now() - started, coreDistances, reachability};
}

async function measureContributor(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  operation: 'optics' | 'sampling',
  rows: number
) {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: `${operation}-${rows}`});
  const columns = Math.ceil(Math.sqrt(rows));
  const coordinates = Float32Array.from(
    Array.from({length: rows}, (_, row) => [
      (row % columns) + (row % 7) / 32,
      Math.floor(row / columns) + (row % 11) / 32
    ]).flat()
  );
  const uploadStart = performance.now();
  const positionsBuffer = createInputBuffer(device, coordinates);
  buffers.push(positionsBuffer);
  const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows);
  const frameBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: rows}, (_, row) => row)
  );
  buffers.push(frameBuffer);
  const frameIds = importGraphBuffer(graph, 'frameIds', frameBuffer, 'uint32', rows);
  const uploadMilliseconds = performance.now() - uploadStart;
  const output = <Format extends 'uint32' | 'float32'>(format: Format, count: number) => {
    const buffer = createOutputBuffer(device, count);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, count),
      buffer
    };
  };
  const sampleCount = Math.min(128, Math.ceil(rows / 4));
  const iterations = rows;
  const status = output('uint32', operation === 'optics' ? 3 : 2);
  const ids = output('uint32', operation === 'optics' ? rows : sampleCount);
  const coreDistances = operation === 'optics' ? output('float32', rows) : undefined;
  const reachability = operation === 'optics' ? output('float32', rows) : undefined;
  const prepareStart = performance.now();
  if (operation === 'optics') {
    graph.add(
      new GPUOPTICS({
        positions,
        minimumSamples: 2,
        maximumDistance: 1.6,
        extractionDistance: 1.1,
        maximumIterations: iterations,
        output: {
          ordering: ids.view,
          coreDistances: coreDistances!.view,
          reachability: reachability!.view,
          predecessors: output('uint32', rows).view,
          labels: output('uint32', rows).view,
          status: status.view
        }
      })
    );
  } else {
    graph.add(
      new GPUSpatialBalancedSample({
        positions,
        frameIds,
        bounds: [0, 0, columns + 1, columns + 1],
        allocations: [{stratumId: 0, sampleCount}],
        seed: 17,
        output: {
          frameIds: ids.view,
          inclusionProbabilities: output('float32', sampleCount).view,
          designWeights: output('float32', sampleCount).view,
          status: status.view
        }
      })
    );
  }
  const preparationMilliseconds = performance.now() - prepareStart;
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compilationMilliseconds = performance.now() - compileStart;
  try {
    await device.handle.queue.onSubmittedWorkDone();
    const elapsed: number[] = [];
    for (let run = 0; run < 4; run++) {
      const started = performance.now();
      submitGraph(device, compiled, undefined);
      await device.handle.queue.onSubmittedWorkDone();
      elapsed.push(performance.now() - started);
    }
    const changedCoordinates = coordinates.map((value, index) =>
      index % 2 === 0 ? value + 1 / 1024 : value
    );
    const updateStart = performance.now();
    positionsBuffer.write(changedCoordinates);
    const updateUploadMilliseconds = performance.now() - updateStart;
    const repeatStart = performance.now();
    submitGraph(device, compiled, undefined);
    await device.handle.queue.onSubmittedWorkDone();
    const changedInputMilliseconds = performance.now() - repeatStart;
    const readStart = performance.now();
    const statusValues = await readUint32(status.buffer, status.view.length);
    const selected = await readUint32(ids.buffer, ids.view.length);
    const readbackMilliseconds = performance.now() - readStart;
    if (operation === 'optics') {
      expect(statusValues).toEqual([iterations, rows === iterations ? 1 : 0, 0]);
      expect(new Set(selected.slice(0, iterations)).size).toBe(iterations);
    } else {
      expect(statusValues).toEqual([sampleCount, 0]);
      expect(new Set(selected).size).toBe(sampleCount);
    }
    const warm = elapsed.slice(1).sort((first, second) => first - second);
    const cpuReference =
      operation === 'optics' ? measureCPUOrdering(changedCoordinates, 1.6) : null;
    if (cpuReference) {
      const actualCore = await readFloat32(coreDistances!.buffer, rows);
      const actualReachability = await readFloat32(reachability!.buffer, rows);
      for (let row = 0; row < rows; row++) {
        expect(actualCore[row]).toBeCloseTo(cpuReference.coreDistances[row], 5);
        if (Number.isFinite(cpuReference.reachability[row]))
          expect(actualReachability[row]).toBeCloseTo(cpuReference.reachability[row], 5);
        else expect(actualReachability[row]).toBe(Infinity);
      }
    }
    const cpuFullOrderingMilliseconds = cpuReference?.milliseconds ?? null;
    return {
      operation,
      rows,
      frontierIterations: operation === 'optics' ? iterations : null,
      sampleCount: operation === 'sampling' ? sampleCount : null,
      uploadMilliseconds,
      preparationMilliseconds,
      compilationMilliseconds,
      firstExecutionMilliseconds: elapsed[0],
      medianWarmExecutionMilliseconds: warm[1],
      updateUploadMilliseconds,
      changedInputMilliseconds,
      readbackMilliseconds,
      declaredScratchBytesUpperBound:
        operation === 'optics'
          ? 12 * rows + 8 * (Math.ceil(rows / 256) + 1)
          : 12 * rows + 8 * sampleCount,
      cpuFullOrderingMilliseconds,
      cpuReference:
        operation === 'optics'
          ? 'independent JS f64 exhaustive minimumSamples=2 full ordering'
          : null
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

it('bounded point-process performance at small and maximum supported frame sizes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const rows of [32, 256, 4096]) {
    for (const operation of ['optics', 'sampling'] as const) {
      results.push(await measureContributor(device, operation, rows));
    }
  }
  console.log(`POINT_PROCESS_BENCHMARK ${JSON.stringify({device: device.info, results})}`);
}, 120000);
