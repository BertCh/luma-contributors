// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointCloudSMRF} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-smrf';
import {GPUPointCloudNeighborhoods} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../utils/benchmark-settings';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
type Settings = ReturnType<typeof getBenchmarkSettings>;

async function measureGraph(
  device: Device,
  graph: GPUCommandGraph,
  validate: () => Promise<void>,
  settings: Settings
) {
  const compileStarted = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStarted;
  const querySet = device.features.has('timestamp-query')
    ? device.createQuerySet({type: 'timestamp', count: 2 * (compiled.stats.nodeOrder.length + 1)})
    : undefined;
  const execute = async () => {
    const started = performance.now();
    const encodeStarted = performance.now();
    submitGraph(device, compiled, undefined);
    const encodeSubmitMilliseconds = performance.now() - encodeStarted;
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    return {wallMilliseconds: performance.now() - started, encodeSubmitMilliseconds};
  };
  try {
    const firstExecution = await execute();
    await validate();
    for (let warmup = 0; warmup < settings.warmupCount; warmup++) await execute();
    const wallSamples = [];
    for (let sample = 0; sample < settings.sampleCount; sample++) wallSamples.push(await execute());
    await validate();
    const profileSamples = [];
    if (querySet) {
      for (let sample = -settings.warmupCount; sample < settings.sampleCount; sample++) {
        const encoder = device.createCommandEncoder({timeProfilingQuerySet: querySet});
        const encoding = compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const timings = await encoding.readTimings();
        if (sample >= 0)
          profileSamples.push({
            cpuEncodeTimeMilliseconds: timings.cpuEncodeTimeMilliseconds,
            gpuStartTimeMilliseconds: timings.gpuStartTimeMilliseconds ?? null,
            gpuEndTimeMilliseconds: timings.gpuEndTimeMilliseconds ?? null,
            nodes: timings.nodes.map(node => ({
              id: node.id,
              hasGPUTimestamps: node.hasGPUTimestamps,
              gpuTimeMilliseconds: node.gpuTimeMilliseconds ?? null,
              cpuEncodeTimeMilliseconds: node.cpuEncodeTimeMilliseconds
            }))
          });
      }
      await validate();
    }
    return {
      compileMilliseconds,
      firstExecution,
      wallSamples,
      wallDistribution: getBenchmarkDistribution(
        wallSamples.map(sample => sample.wallMilliseconds)
      ),
      profileSamples: querySet ? profileSamples : null,
      nodeOrder: compiled.stats.nodeOrder,
      physicalTransientBytes: compiled.stats.physicalTransientResourceBytes,
      logicalTransientBytes: compiled.stats.logicalTransientBytes,
      correctnessStatus: 'passed'
    };
  } finally {
    querySet?.destroy();
    compiled.destroy();
  }
}

async function measureNeighborhoods(
  device: Device,
  sourceCount: number,
  distribution: 'coincident' | 'lattice',
  settings: Settings
) {
  const setupStarted = performance.now();
  const fixture = makeSpatialFixture(device);
  const queryCount = 64;
  const capacity = 4;
  const side = Math.ceil(Math.cbrt(sourceCount));
  const positions = new Float32Array(sourceCount * 3);
  if (distribution === 'lattice') {
    for (let source = 0; source < sourceCount; source++) {
      positions[source * 3] = (source % side) + 0.25;
      positions[source * 3 + 1] = (Math.floor(source / side) % side) + 0.25;
      positions[source * 3 + 2] = Math.floor(source / (side * side)) + 0.25;
    }
  }
  const source = fixture.view('float32x3', positions);
  const queryRows = fixture.view(
    'uint32',
    Uint32Array.from({length: queryCount}, (_, index) => index)
  );
  const indices = fixture.view('uint32', new Uint32Array(queryCount * capacity));
  const squaredDistances = fixture.view('float32', new Float32Array(queryCount * capacity));
  const counts = fixture.view('uint32', new Uint32Array(queryCount));
  const status = fixture.view('uint32', new Uint32Array(queryCount));
  fixture.graph.add(
    new GPUPointCloudNeighborhoods({
      id: 'neighborhoods',
      positions: source.view,
      queryRows: queryRows.view,
      grid:
        distribution === 'coincident'
          ? {origin: [-1, -1, -1], dimensions: [2, 2, 2], cellSize: 1}
          : {origin: [0, 0, 0], dimensions: [side + 1, side + 1, side + 1], cellSize: 1},
      radius: distribution === 'coincident' ? 0.1 : 1.01,
      capacity,
      output: {
        indices: indices.view,
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        status: status.view
      }
    })
  );
  const expectedIndices: number[] = [];
  const expectedDistances: number[] = [];
  const expectedCounts: number[] = [];
  for (let query = 0; query < queryCount; query++) {
    let support: number[];
    let supportCount: number;
    if (distribution === 'coincident') {
      support = Array.from({length: capacity + 1}, (_, index) => index)
        .filter(index => index !== query)
        .slice(0, capacity);
      supportCount = sourceCount - 1;
    } else {
      const coordinate = [
        query % side,
        Math.floor(query / side) % side,
        Math.floor(query / (side * side))
      ];
      support = [];
      for (let axis = 0; axis < 3; axis++) {
        for (const direction of [-1, 1]) {
          const neighbor = coordinate.slice();
          neighbor[axis] += direction;
          if (neighbor.some(value => value < 0 || value >= side)) continue;
          const neighborIndex = (neighbor[2] * side + neighbor[1]) * side + neighbor[0];
          if (neighborIndex < sourceCount) support.push(neighborIndex);
        }
      }
      support.sort((first, second) => first - second);
      supportCount = support.length;
      support = support.slice(0, capacity);
    }
    expectedCounts.push(supportCount);
    for (let slot = 0; slot < capacity; slot++) {
      expectedIndices.push(support[slot] ?? 0xffffffff);
      expectedDistances.push(
        slot < support.length ? (distribution === 'coincident' ? 0 : 1) : Math.fround(3.402823e38)
      );
    }
  }
  const validate = async () => {
    expect(await readIntegers(counts.buffer)).toEqual(expectedCounts);
    expect(await readIntegers(indices.buffer)).toEqual(expectedIndices);
    expect(await readFloats(squaredDistances.buffer)).toEqual(expectedDistances);
    expect(await readIntegers(status.buffer)).toEqual(
      expectedCounts.map(count => (count > capacity ? 2 : 0))
    );
  };
  try {
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    const setupAndUploadMilliseconds = performance.now() - setupStarted;
    return {
      shape: {sourceCount, queryCount, capacity, distribution},
      setupAndUploadMilliseconds,
      inputBytes: positions.byteLength + queryCount * 4,
      outputBytes: queryCount * (capacity * 8 + 8),
      ...(await measureGraph(device, fixture.graph, validate, settings))
    };
  } finally {
    fixture.destroy();
  }
}

async function measureGroundClassification(
  device: Device,
  sourceCount: number,
  settings: Settings
) {
  const setupStarted = performance.now();
  const fixture = makeSpatialFixture(device);
  const side = 256;
  const cellCount = side * side;
  const positions = new Float32Array(sourceCount * 3);
  const isObject = (index: number) => index >= cellCount && index % 17 === 0;
  for (let index = 0; index < sourceCount; index++) {
    positions[index * 3] = (index % side) + 0.25;
    positions[index * 3 + 1] = (Math.floor(index / side) % side) + 0.25;
    positions[index * 3 + 2] = isObject(index) ? 3 : 0;
  }
  const input = fixture.view('float32x3', positions);
  const classifications = fixture.view('uint32', new Uint32Array(sourceCount).fill(6));
  const groundSurface = fixture.view('float32', new Float32Array(cellCount));
  const groundMask = fixture.view('uint32', new Uint32Array(sourceCount));
  const outputClassifications = fixture.view('uint32', new Uint32Array(sourceCount));
  const status = fixture.view('uint32', new Uint32Array(sourceCount));
  fixture.graph.add(
    new GPUPointCloudSMRF({
      id: 'ground-classification',
      positions: input.view,
      classifications: classifications.view,
      grid: {origin: [0, 0], cellSize: 1, width: side, height: side},
      window: 4,
      output: {
        groundSurface: groundSurface.view,
        groundMask: groundMask.view,
        outputClassifications: outputClassifications.view,
        status: status.view
      }
    })
  );
  const validate = async () => {
    // Every cell has a zero-height ground return in the first raster cycle. Later elevated
    // returns cannot alter the minimum surface, so all morphology and terrain stay exactly zero.
    expect((await readFloats(groundSurface.buffer)).every(value => value === 0)).toBe(true);
    expect(
      (await readIntegers(groundMask.buffer)).every(
        (value, index) => value === Number(!isObject(index))
      )
    ).toBe(true);
    expect(
      (await readIntegers(outputClassifications.buffer)).every(
        (value, index) => value === (isObject(index) ? 1 : 2)
      )
    ).toBe(true);
    expect(
      (await readIntegers(status.buffer)).every((value, index) => {
        const column = index % side;
        const row = Math.floor(index / side) % side;
        return value === (column < 8 || row < 8 || column + 8 >= side || row + 8 >= side ? 8 : 0);
      })
    ).toBe(true);
  };
  try {
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    const setupAndUploadMilliseconds = performance.now() - setupStarted;
    return {
      shape: {sourceCount, side, distribution: 'flat-ground-with-elevated-returns', window: 4},
      setupAndUploadMilliseconds,
      inputBytes: sourceCount * 16,
      outputBytes: cellCount * 4 + sourceCount * 12,
      ...(await measureGraph(device, fixture.graph, validate, settings))
    };
  } finally {
    fixture.destroy();
  }
}

it.skipIf(environment?.['VITE_IRREGULAR_GEOSPATIAL_PROFILE'] !== 'true')(
  'point cloud consumers phase attribution',
  {timeout: 900000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    console.log(
      `IRREGULAR_GEOSPATIAL_METADATA ${JSON.stringify({
        settings,
        fixtureVersion: 1,
        device: device.info,
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        timestampQueriesAvailable: device.features.has('timestamp-query'),
        limits: {
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        method:
          'Uninstrumented wall samples include encode/submit/completion and rebuilding indexes/operators. Separate per-node timestamp passes are diagnostic and may change scheduling. Unsupported timestamps are null. Setup combines fixture generation, declaration, allocation and upload. Validation/readback outside timing; cache state uncontrolled. Node count and configured iterations are not executed iteration counts.'
      })}`
    );
    let measured = 0;
    for (const distribution of ['coincident', 'lattice'] as const) {
      for (const sourceCount of [100000, 1000000]) {
        const workload = `neighborhoods-${distribution}-${sourceCount}-64-4`;
        if (!workload.includes(settings.workloadFilter)) continue;
        measured++;
        console.log(
          `IRREGULAR_GEOSPATIAL_MEASUREMENT ${JSON.stringify({workload, ...(await measureNeighborhoods(device, sourceCount, distribution, settings))})}`
        );
      }
    }
    for (const sourceCount of [100000, 1000000]) {
      const workload = `smrf-${sourceCount}`;
      if (!workload.includes(settings.workloadFilter)) continue;
      measured++;
      console.log(
        `IRREGULAR_GEOSPATIAL_MEASUREMENT ${JSON.stringify({workload, ...(await measureGroundClassification(device, sourceCount, settings))})}`
      );
    }
    if (!measured) throw new Error('No irregular geospatial profiling workload matched the filter');
  }
);
