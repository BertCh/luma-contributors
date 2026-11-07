// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointCloudNeighborhoods} from '../../src/pointcloud-analysis/gpu-pointcloud-neighborhoods';
import {GPUPointCloudFeatures} from '../../src/pointcloud-analysis/gpu-pointcloud-features';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../utils/benchmark-settings';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
type Settings = ReturnType<typeof getBenchmarkSettings>;
type Distribution =
  | 'coincident'
  | 'lattice'
  | 'clustered'
  | 'line-ascending'
  | 'line-descending'
  | 'plane';
type NeighborhoodCase = {
  sourceCount: number;
  queryCount: number;
  capacity: number;
  distribution: Distribution;
  indexed: boolean;
  features?: boolean;
};

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
  distribution: Distribution,
  queryCount: number,
  capacity: number,
  strategy: 'serial' | 'cooperative',
  indexed: boolean,
  features: boolean,
  settings: Settings
) {
  const setupStarted = performance.now();
  const fixture = makeSpatialFixture(device);
  const side = Math.ceil(Math.cbrt(sourceCount));
  const positions = new Float32Array(sourceCount * 3);
  if (distribution === 'lattice') {
    for (let source = 0; source < sourceCount; source++) {
      positions[source * 3] = (source % side) + 0.25;
      positions[source * 3 + 1] = (Math.floor(source / side) % side) + 0.25;
      positions[source * 3 + 2] = Math.floor(source / (side * side)) + 0.25;
    }
  }
  if (distribution === 'clustered') {
    for (let source = 0; source < sourceCount; source++) positions[source * 3] = (source % 128) * 4;
  }
  const planarSide = Math.ceil(Math.sqrt(sourceCount));
  if (distribution.startsWith('line-')) {
    for (let source = 0; source < sourceCount; source++) positions[source * 3] = source / 131072;
  }
  if (distribution === 'plane') {
    for (let source = 0; source < sourceCount; source++) {
      positions[source * 3] = (source % planarSide) / 1024;
      positions[source * 3 + 1] = Math.floor(source / planarSide) / 1024;
    }
  }
  const source = fixture.view('float32x3', positions);
  const queryValues = Uint32Array.from({length: queryCount}, (_, index) =>
    distribution === 'line-descending' ? sourceCount - 1 - index : index
  );
  const queryRows = fixture.view('uint32', queryValues);
  const indices = fixture.view('uint32', new Uint32Array(queryCount * capacity));
  const squaredDistances = fixture.view('float32', new Float32Array(queryCount * capacity));
  const counts = fixture.view('uint32', new Uint32Array(queryCount));
  const status = fixture.view('uint32', new Uint32Array(queryCount));
  fixture.graph.add(
    new GPUPointCloudNeighborhoods({
      id: 'neighborhoods',
      positions: source.view,
      queryRows: queryRows.view,
      ...(indexed
        ? {
            grid:
              distribution === 'coincident' ||
              distribution.startsWith('line-') ||
              distribution === 'plane'
                ? {origin: [-1, -1, -1] as const, dimensions: [2, 2, 2] as const, cellSize: 1}
                : distribution === 'clustered'
                  ? {origin: [-1, -1, -1] as const, dimensions: [514, 2, 2] as const, cellSize: 1}
                  : {
                      origin: [0, 0, 0] as const,
                      dimensions: [side + 1, side + 1, side + 1] as const,
                      cellSize: 1
                    }
          }
        : {}),
      radius:
        distribution === 'lattice'
          ? 1.01
          : distribution.startsWith('line-') || distribution === 'plane'
            ? 1
            : 0.1,
      strategy,
      capacity,
      indices: indices.view,
      squaredDistances: squaredDistances.view,
      counts: counts.view,
      status: status.view
    })
  );
  const featureOutputs = features
    ? {
        normals: fixture.view('float32x3', new Float32Array(queryCount * 3)),
        eigenvalues: fixture.view('float32x3', new Float32Array(queryCount * 3)),
        shapes: fixture.view('float32x3', new Float32Array(queryCount * 3)),
        covariance: fixture.view('float32', new Float32Array(queryCount * 6)),
        status: fixture.view('uint32', new Uint32Array(queryCount))
      }
    : undefined;
  if (featureOutputs)
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: source.view,
        indices: indices.view,
        counts: counts.view,
        capacity,
        normals: featureOutputs.normals.view,
        eigenvalues: featureOutputs.eigenvalues.view,
        shapes: featureOutputs.shapes.view,
        covariance: featureOutputs.covariance.view,
        status: featureOutputs.status.view
      })
    );
  const expectedIndices: number[] = [];
  const expectedDistances: number[] = [];
  const expectedCounts: number[] = [];
  for (let query = 0; query < queryCount; query++) {
    const queryRow = queryValues[query];
    let support: number[];
    let supportCount: number;
    if (distribution.startsWith('line-')) {
      supportCount = sourceCount - 1;
      support = Array.from({length: capacity * 2 + 1}, (_, index) => queryRow - capacity + index)
        .filter(index => index >= 0 && index < sourceCount && index !== queryRow)
        .sort(
          (first, second) =>
            Math.abs(first - queryRow) - Math.abs(second - queryRow) || first - second
        )
        .slice(0, capacity);
    } else if (distribution === 'plane') {
      supportCount = sourceCount - 1;
      support = [];
      const queryColumn = queryRow % planarSide;
      const queryLine = Math.floor(queryRow / planarSide);
      for (let line = Math.max(0, queryLine - capacity); line <= queryLine + capacity; line++) {
        for (
          let column = Math.max(0, queryColumn - capacity);
          column <= Math.min(planarSide - 1, queryColumn + capacity);
          column++
        ) {
          const candidate = line * planarSide + column;
          if (candidate < sourceCount && candidate !== queryRow) support.push(candidate);
        }
      }
      const distance = (index: number) =>
        ((index % planarSide) - queryColumn) ** 2 +
        (Math.floor(index / planarSide) - queryLine) ** 2;
      support.sort((first, second) => distance(first) - distance(second) || first - second);
      support = support.slice(0, capacity);
    } else if (distribution === 'coincident') {
      support = Array.from({length: capacity + 1}, (_, index) => index)
        .filter(index => index !== query)
        .slice(0, capacity);
      supportCount = sourceCount - 1;
    } else if (distribution === 'clustered') {
      const cluster = query % 128;
      supportCount = Math.floor((sourceCount - 1 - cluster) / 128);
      support = Array.from(
        {length: Math.min(capacity + 1, supportCount + 1)},
        (_, index) => cluster + index * 128
      )
        .filter(index => index !== query)
        .slice(0, capacity);
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
        slot < support.length
          ? Math.fround(
              [0, 1, 2].reduce(
                (total, axis) =>
                  total +
                  (positions[support[slot] * 3 + axis] - positions[queryRow * 3 + axis]) ** 2,
                0
              )
            )
          : Math.fround(3.402823e38)
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
    if (featureOutputs) {
      const covarianceValues = await readFloats(featureOutputs.covariance.buffer);
      const eigenvalueValues = await readFloats(featureOutputs.eigenvalues.buffer);
      const normalValues = await readFloats(featureOutputs.normals.buffer);
      const shapeValues = await readFloats(featureOutputs.shapes.buffer);
      expect(await readIntegers(featureOutputs.status.buffer)).toEqual(
        expectedCounts.map(count => (count > capacity ? 2 : 0))
      );
      for (let query = 0; query < queryCount; query++) {
        const support = expectedIndices.slice(query * capacity, (query + 1) * capacity);
        const mean = [0, 1].map(
          axis =>
            support.reduce((total, index) => total + positions[index * 3 + axis], 0) / capacity
        );
        const covariance = [
          [0, 0],
          [0, 1],
          [0, 2],
          [1, 1],
          [1, 2],
          [2, 2]
        ].map(
          ([first, second]) =>
            support.reduce(
              (total, index) =>
                total +
                (positions[index * 3 + first] - (mean[first] ?? 0)) *
                  (positions[index * 3 + second] - (mean[second] ?? 0)),
              0
            ) / capacity
        );
        const trace = covariance[0] + covariance[3];
        const difference = Math.sqrt((covariance[0] - covariance[3]) ** 2 + 4 * covariance[1] ** 2);
        const largest = (trace + difference) / 2;
        const middle = (trace - difference) / 2;
        for (let component = 0; component < 6; component++)
          expect(
            Math.abs(covarianceValues[query * 6 + component] - covariance[component])
          ).toBeLessThan(1e-10);
        for (const [component, value] of [largest, middle, 0].entries())
          expect(Math.abs(eigenvalueValues[query * 3 + component] - value)).toBeLessThan(1e-10);
        for (const [component, value] of [0, 0, 1].entries())
          expect(Math.abs(normalValues[query * 3 + component] - value)).toBeLessThan(1e-5);
        for (const [component, value] of [
          (largest - middle) / largest,
          middle / largest,
          0
        ].entries())
          expect(Math.abs(shapeValues[query * 3 + component] - value)).toBeLessThan(2e-5);
      }
    }
  };
  try {
    await (device.handle as GPUDevice).queue.onSubmittedWorkDone();
    const setupAndUploadMilliseconds = performance.now() - setupStarted;
    return {
      shape: {sourceCount, queryCount, capacity, distribution, strategy, indexed, features},
      setupAndUploadMilliseconds,
      inputBytes: positions.byteLength + queryCount * 4,
      outputBytes: queryCount * (capacity * 8 + 8 + (features ? 64 : 0)),
      ...(await measureGraph(device, fixture.graph, validate, settings))
    };
  } finally {
    fixture.destroy();
  }
}

const cases: NeighborhoodCase[] = [
  ...(['line-ascending', 'line-descending'] as const).flatMap(distribution =>
    [4, 64].map(capacity => ({
      sourceCount: 100000,
      queryCount: 8,
      capacity,
      distribution,
      indexed: true
    }))
  ),
  {
    sourceCount: 100000,
    queryCount: 64,
    capacity: 8,
    distribution: 'plane',
    indexed: true,
    features: true
  },
  ...[1, 4, 16, 64].map(capacity => ({
    sourceCount: 100000,
    queryCount: 64,
    capacity,
    distribution: 'coincident' as const,
    indexed: true
  })),
  {
    sourceCount: 1000000,
    queryCount: 64,
    capacity: 4,
    distribution: 'coincident' as const,
    indexed: true
  },
  {
    sourceCount: 1000000,
    queryCount: 1,
    capacity: 4,
    distribution: 'coincident' as const,
    indexed: true
  },
  {
    sourceCount: 100000,
    queryCount: 64,
    capacity: 4,
    distribution: 'coincident' as const,
    indexed: false
  },
  {
    sourceCount: 100000,
    queryCount: 1024,
    capacity: 8,
    distribution: 'lattice' as const,
    indexed: true
  },
  {
    sourceCount: 1000000,
    queryCount: 64,
    capacity: 4,
    distribution: 'lattice' as const,
    indexed: true
  },
  {
    sourceCount: 100000,
    queryCount: 64,
    capacity: 4,
    distribution: 'clustered' as const,
    indexed: true
  },
  {
    sourceCount: 129,
    queryCount: 1,
    capacity: 64,
    distribution: 'coincident' as const,
    indexed: true
  }
];

it.skipIf(environment?.['VITE_NEIGHBORHOOD_STRATEGY_BENCHMARK'] !== 'true')(
  'serial and cooperative exact neighborhood workflows',
  {timeout: 900000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const strategies: ('serial' | 'cooperative')[] =
      environment?.['VITE_NEIGHBORHOOD_COOPERATIVE_FIRST'] === 'true'
        ? ['cooperative', 'serial']
        : ['serial', 'cooperative'];
    console.log(
      `NEIGHBORHOOD_STRATEGY_METADATA ${JSON.stringify({
        settings,
        fixtureVersion: 2,
        strategies,
        device: device.info,
        userAgent: navigator.userAgent,
        timestampQueriesAvailable: device.features.has('timestamp-query'),
        limits: {
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        method:
          'Each strategy includes a fresh grid build and complete exact query in uninstrumented encode/submit/completion samples. GPU timestamps run separately. Allocation/upload, host compilation, first use and validation/readback are outside warm timing. Strategy order is blocked within each case and can reverse across sessions. No controlled cold-cache claim.'
      })}`
    );
    let measured = 0;
    for (const {
      sourceCount,
      queryCount,
      capacity,
      distribution,
      indexed,
      features = false
    } of cases) {
      const workload = `${distribution}-${sourceCount}-${queryCount}-${capacity}-${indexed ? 'grid' : 'all-pairs'}${features ? '-features' : ''}`;
      if (!workload.includes(settings.workloadFilter)) continue;
      measured++;
      for (const strategy of strategies) {
        console.log(
          `NEIGHBORHOOD_STRATEGY_MEASUREMENT ${JSON.stringify({
            workload,
            strategy,
            ...(await measureNeighborhoods(
              device,
              sourceCount,
              distribution,
              queryCount,
              capacity,
              strategy,
              indexed,
              features,
              settings
            ))
          })}`
        );
      }
    }
    if (!measured) throw new Error('No neighborhood strategy workload matched the filter');
  }
);
