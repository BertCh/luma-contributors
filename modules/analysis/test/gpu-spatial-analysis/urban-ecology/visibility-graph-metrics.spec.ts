// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSnapshotArena, GPUSnapshotTap} from '../../../src/computation-record';
import {createVisibilityGraphPivotsNode} from '../../../src/gpu-network/visibility-graph/visibility-graph-pivots';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {addVisibilityGraphRecipe} from '../../../src/gpu-network/visibility-graph/visibility-graph';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  computeVisibilityMetricsOracle,
  VISIBILITY_METRIC_GRAPH_FIXTURES
} from './visibility-graph-metrics-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {publishVisibilityGraphOutput} from './visibility-graph-test-utils';

async function createFixture(name: string) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return new RecipeTestFixture(device, name);
}
function publish(test: RecipeTestFixture, name: string, fields: Record<string, GraphDataView>) {
  return Object.fromEntries(
    Object.entries(fields).map(([field, view]) => [
      field,
      publishVisibilityGraphOutput(test, `${name}-${field}`, view)
    ])
  );
}
async function read(test: RecipeTestFixture, captured: ReturnType<typeof publish>) {
  const values: Record<string, number[]> = {};
  for (const [name, {output, source, components}] of Object.entries(captured)) {
    values[name] = source.format.startsWith('float')
      ? await test.readFloat32(output, source.length * components)
      : await test.readUint32(output, source.length * components);
  }
  return values;
}
function expectValue(actual: number, expected: number) {
  if (Number.isNaN(expected)) expect(actual).toBeNaN();
  else if (!Number.isFinite(expected)) expect(actual).toBe(expected);
  else
    expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(expected)));
}
// Closed-form normalizations, independent of GPU traversal or Brandes accumulation.
function getIntegration(reachable: number, total: number, maximum: number) {
  const size = reachable + 1;
  const mean = reachable > 0 ? total / reachable : NaN;
  const asymmetry = size > 2 ? (2 * (mean - 1)) / (size - 2) : NaN;
  const normalization =
    (2 * (size * (Math.log2((size + 2) / 3) - 1) + 1)) / ((size - 1) * (size - 2));
  const real = size > 2 ? asymmetry / normalization : NaN;
  const pyramid =
    size > 2 && mean > 1
      ? (2 * (size - Math.log2(size) - 1)) / ((size - 1) * (size - 2)) / asymmetry
      : NaN;
  const teklenburg =
    size > 2 && mean > 1 && total - size + 1 > 1
      ? Math.log(0.5 * (size - 2)) / Math.log(total - size + 1)
      : NaN;
  const radius = Math.min(maximum, size - 1);
  const greatest = radius * (size - 0.5 * (radius + 1));
  const penn =
    size > 2 && mean > 1 && greatest > size - 1 ? (greatest - total) / (greatest - size + 1) : NaN;
  return {
    integration: [mean, asymmetry, real, 1 / real],
    extendedIntegration: [pyramid, teklenburg, penn, maximum]
  };
}

it.each([
  1, 3, 32, 255, 256, 512
])('CSR all-source integration and choice match path closed forms at %i points', async count => {
  const test = await createFixture(`csr-all-source-${count}`);
  try {
    const coordinates = new Float32Array(count * 2);
    for (let row = 0; row < count; row++) coordinates[2 * row] = row;
    const positions = test.input('points', coordinates, 'float32x2', count);
    const segments = test.input(
      'walls',
      new Float32Array([10000, 10000, 10001, 10000]),
      'float32x4',
      1
    );
    const result = addVisibilityGraphRecipe(test.graph, {
      id: 'csr',
      mode: 'csr',
      positions,
      segments,
      maximumDistance: 1.1,
      edgeCapacity: count * 2,
      metricsMode: 'exact',
      maximumIterations: count,
      laneCount: count <= 3 ? count : count > 256 ? 64 : count % 32 === 0 ? 32 : 17
    });
    const {status, ...fields} = result.outputs;
    const captured = publish(test, 'csr', {...fields, ...status});
    const dense =
      count <= 256
        ? addVisibilityGraphRecipe(test.graph, {
            id: 'dense',
            mode: 'reference',
            positions,
            segments,
            maximumDistance: 1.1,
            edgeCapacity: count,
            searchStructure: 'brute-force',
            obstacleIndex: 'bvh'
          })
        : undefined;
    const denseCaptured = dense
      ? publish(test, 'dense', {
          metrics: dense.outputs.metrics,
          integration: dense.outputs.integration,
          extendedIntegration: dense.outputs.extendedIntegration,
          normalizedChoice: dense.outputs.normalizedChoice
        })
      : undefined;
    test.run();
    const values = await read(test, captured);
    expect(values['approximation']).toEqual([0]);
    expect(values['converged']).toEqual([1]);
    expect(values['iterationLimitReached']).toEqual([0]);
    for (let row = 0; row < count; row++) {
      const left = row;
      const right = count - 1 - row;
      const depth = (left * (left + 1)) / 2 + (right * (right + 1)) / 2;
      const expectedMetrics = [
        Number(left > 0) + Number(right > 0),
        count - 1,
        depth,
        depth > 0 ? (count - 1) / depth : NaN
      ];
      expectedMetrics.forEach((value, component) =>
        expectValue(values['metrics'][4 * row + component], value)
      );
      expectValue(values['betweenness'][row], left * right);
      expectValue(
        values['normalizedChoice'][row],
        count > 2 ? (2 * left * right) / ((count - 1) * (count - 2)) : NaN
      );
      const expected = getIntegration(count - 1, depth, Math.max(left, right));
      for (const field of ['integration', 'extendedIntegration'] as const)
        expected[field].forEach((value, component) =>
          expectValue(values[field][4 * row + component], value)
        );
    }
    if (denseCaptured) {
      const reference = await read(test, denseCaptured);
      for (const [field, expected] of Object.entries(reference))
        expected.forEach((value, index) => expectValue(values[field][index], value));
    }
  } finally {
    test.destroy();
  }
});

it('100k CSR viewpoints publish sampled integration and normalized choice with independent component oracles', async () => {
  const test = await createFixture('csr-100k-integration');
  try {
    const count = 100000;
    const coordinates = new Float32Array(count * 2);
    // 25k disconnected four-point paths: exact local distances and dependencies have closed forms.
    for (let row = 0; row < count; row++)
      coordinates[2 * row] = Math.floor(row / 4) * 10 + (row % 4);
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', coordinates, 'float32x2', count),
      segments: test.input('walls', new Float32Array([-100, -100, -99, -100]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: count * 2,
      maximumIterations: 8,
      queryChunkSize: 16384
    });
    const captured = publish(test, 'large', {
      pivots: result.pivotNodes,
      metrics: result.outputs.metrics,
      integration: result.outputs.integration,
      normalizedChoice: result.outputs.normalizedChoice,
      betweenness: result.outputs.betweenness,
      maximum: result.outputs.extendedIntegration,
      ...result.status
    });
    test.run();
    const values = await read(test, captured);
    expect(values['approximation']).toEqual([1]);
    expect(values['converged']).toEqual([1]);
    expect(values['iterationLimitReached']).toEqual([0]);
    expect(values['count']).toEqual([150000]);
    const pivots = Array.from({length: 32}, (_, index) => Math.floor((index * count) / 32));
    expect(values['pivots']).toEqual(pivots);
    for (const row of [0, 1, 2, 3, 4, 3124, 3125, 3126, 3127, 50000, 99999]) {
      const localRow = row % 4;
      const component = Math.floor(row / 4);
      const componentPivots = pivots.filter(pivot => Math.floor(pivot / 4) === component);
      const others = componentPivots.filter(pivot => pivot !== row);
      const scale = (count - 1) / (pivots.length - Number(pivots.includes(row)));
      const reached = others.length * scale;
      const total = others.reduce((sum, pivot) => sum + Math.abs(pivot - row), 0) * scale;
      const maximum = Math.max(0, ...others.map(pivot => Math.abs(pivot - row)));
      const expectedMetrics = [
        localRow === 0 || localRow === 3 ? 1 : 2,
        reached,
        total,
        total > 0 ? reached / total : NaN
      ];
      expectedMetrics.forEach((value, componentIndex) =>
        expectValue(values['metrics'][4 * row + componentIndex], value)
      );
      const dependency = componentPivots.reduce(
        (sum, pivot) => sum + (pivot < row ? 3 - localRow : pivot > row ? localRow : 0),
        0
      );
      const choice = (dependency * 0.5 * count) / pivots.length;
      expectValue(values['betweenness'][row], choice);
      expectValue(values['normalizedChoice'][row], (2 * choice) / ((count - 1) * (count - 2)));
      getIntegration(reached, total, maximum).integration.forEach((value, componentIndex) =>
        expectValue(values['integration'][4 * row + componentIndex], value)
      );
      expectValue(values['maximum'][4 * row + 3], maximum);
    }
  } finally {
    test.destroy();
  }
}, 120000);

it('duplicate pivots mask every new metric and recover on re-execution', async () => {
  const test = await createFixture('csr-metric-failure');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const pivotBuffer = test.parameterBuffer('pivots', 'uint32', 2);
    pivotBuffer.write(new Uint32Array([0, 2]));
    const pivots = pivotBuffer.importToGraph(test.graph);
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 8,
      pivotNodes: pivots,
      maximumIterations: 8
    });
    const captured = publish(test, 'failure', {
      metrics: result.outputs.metrics,
      integration: result.outputs.integration,
      extendedIntegration: result.outputs.extendedIntegration,
      normalizedChoice: result.outputs.normalizedChoice,
      invalidCount: result.status.invalidCount,
      converged: result.status.converged
    });
    compiled = test.graph.compile();
    submitGraph(test.device, compiled, undefined);
    expect((await read(test, captured))['converged']).toEqual([1]);
    pivotBuffer.write(new Uint32Array([0, 0]));
    submitGraph(test.device, compiled, undefined);
    const failed = await read(test, captured);
    expect(failed['invalidCount']).toEqual([2]);
    expect(failed['converged']).toEqual([0]);
    for (const field of ['metrics', 'integration', 'extendedIntegration', 'normalizedChoice'])
      expect(failed[field].every(Number.isNaN), field).toBe(true);
    pivotBuffer.write(new Uint32Array([0, 2]));
    submitGraph(test.device, compiled, undefined);
    const recovered = await read(test, captured);
    expect(recovered['invalidCount']).toEqual([0]);
    expect(recovered['converged']).toEqual([1]);
    expectValue(recovered['metrics'][6], 2);
  } finally {
    compiled?.destroy();
    test.destroy();
  }
});

it.each([
  {edgeCapacity: 2, maximumGridVisits: 8192, overflow: 1, candidateOverflow: 0},
  {edgeCapacity: 8, maximumGridVisits: 1, overflow: 0, candidateOverflow: 1}
])('capacity failures mask every integration field and normalized choice: %j', async settings => {
  const test = await createFixture('csr-integration-capacity');
  try {
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([20, 20, 21, 20]), 'float32x4', 1),
      maximumDistance: 1.1,
      ...settings
    });
    const captured = publish(test, 'capacity', {
      metrics: result.outputs.metrics,
      integration: result.outputs.integration,
      extendedIntegration: result.outputs.extendedIntegration,
      normalizedChoice: result.outputs.normalizedChoice,
      overflow: result.status.overflow,
      candidateOverflow: result.status.candidateOverflow,
      converged: result.status.converged
    });
    test.run();
    const values = await read(test, captured);
    expect(values['overflow']).toEqual([settings.overflow]);
    expect(values['candidateOverflow']).toEqual([settings.candidateOverflow]);
    expect(values['converged']).toEqual([0]);
    for (const field of ['metrics', 'integration', 'extendedIntegration', 'normalizedChoice'])
      expect(values[field].every(Number.isNaN), field).toBe(true);
  } finally {
    test.destroy();
  }
});

it.each(
  VISIBILITY_METRIC_GRAPH_FIXTURES
)('CSR $name metrics match independent CPU BFS/Brandes, including shortest-path ties', async fixture => {
  const test = await createFixture(`csr-oracle-${fixture.name}`);
  try {
    const count = fixture.neighbors.length;
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', new Float32Array(fixture.coordinates), 'float32x2', count),
      segments: test.input('walls', new Float32Array([100, 100, 101, 100]), 'float32x4', 1),
      maximumDistance: fixture.maximumDistance,
      edgeCapacity: count * count,
      metricsMode: 'exact'
    });
    const {status, ...fields} = result.outputs;
    const captured = publish(test, 'oracle', {...fields, ...status});
    test.run();
    const values = await read(test, captured);
    const expected = computeVisibilityMetricsOracle(fixture.neighbors);
    for (const field of ['metrics', 'betweenness', 'normalizedChoice'] as const)
      expected[field].forEach((value, index) => expectValue(values[field][index], value));
    for (let row = 0; row < count; row++) {
      const neighbors = values['neighbors']
        .slice(values['offsets'][row], values['offsets'][row + 1])
        .sort((first, second) => first - second);
      expect(neighbors).toEqual([...fixture.neighbors[row]]);
      const normalized = getIntegration(
        expected.metrics[4 * row + 1],
        expected.metrics[4 * row + 2],
        expected.maximumDepth[row]
      );
      for (const field of ['integration', 'extendedIntegration'] as const)
        normalized[field].forEach((value, component) =>
          expectValue(values[field][4 * row + component], value)
        );
    }
    expect(values['converged']).toEqual([1]);
    expect(values['iterationLimitReached']).toEqual([0]);
    expect(values['approximation']).toEqual([0]);
  } finally {
    test.destroy();
  }
});

it.each([
  false,
  true
])('disjoint packed CSR outputs preserve sentinels with consecutive separate-dispatch ranges=%s', async consecutive => {
  const test = await createFixture('csr-packed-outputs');
  try {
    const count = 8;
    const packed = test.output('packed', 'uint32', 2048);
    const sentinel = 0x4badbeef;
    packed.buffer.write(new Uint32Array(2048).fill(sentinel));
    const writtenWords = new Set<number>();
    let nextByteOffset = 256;
    const view = <Format extends 'float32' | 'float32x4' | 'uint32'>(
      format: Format,
      length: number
    ) => {
      const result = test.graph.createDataView(packed.view.buffer, {
        format,
        length,
        byteOffset: nextByteOffset
      });
      for (
        let word = nextByteOffset / 4;
        word < nextByteOffset / 4 + length * (format === 'float32x4' ? 4 : 1);
        word++
      )
        writtenWords.add(word);
      nextByteOffset += 256;
      return result;
    };
    const output = {
      offsets: view('uint32', 9),
      neighbors: view('uint32', 32),
      weights: view('float32', 32),
      degree: view('uint32', 8),
      gridVisits: view('uint32', 8),
      viewpointClassifications: view('uint32', 8),
      betweenness: view('float32', 8),
      closeness: view('float32', 8),
      reachedCounts: view('uint32', 8),
      metrics: view('float32x4', 8),
      integration: view('float32x4', 8),
      extendedIntegration: view('float32x4', 8),
      normalizedChoice: view('float32', 8),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1),
        invalidCount: view('uint32', 1),
        approximation: view('uint32', 1),
        converged: view('uint32', 1),
        iterationLimitReached: view('uint32', 1),
        zeroWeightCycle: view('uint32', 1)
      }
    };
    if (consecutive) {
      for (
        let word = output.metrics.byteOffset / 4;
        word < output.metrics.byteOffset / 4 + count * 4;
        word++
      )
        writtenWords.delete(word);
      output.metrics = test.graph.createDataView(packed.view.buffer, {
        format: 'float32x4',
        length: count,
        byteOffset: output.betweenness.byteOffset + count * 4
      });
      for (
        let word = output.metrics.byteOffset / 4;
        word < output.metrics.byteOffset / 4 + count * 4;
        word++
      )
        writtenWords.add(word);
    }
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input(
        'points',
        Float32Array.from(Array.from({length: count}, (_, row) => [row, 0]).flat()),
        'float32x2',
        count
      ),
      segments: test.input('walls', new Float32Array([100, 100, 101, 100]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 32,
      metricsMode: 'exact',
      output
    });
    const {status, ...fields} = result.outputs;
    const captured = publish(test, 'packed', {...fields, ...status});
    test.run();
    const values = await read(test, captured);
    const expected = computeVisibilityMetricsOracle(
      Array.from({length: count}, (_, row) =>
        [row - 1, row + 1].filter(neighbor => neighbor >= 0 && neighbor < count)
      )
    );
    for (const field of ['metrics', 'betweenness', 'normalizedChoice'] as const)
      expected[field].forEach((value, index) => expectValue(values[field][index], value));
    expect(values['converged']).toEqual([1]);
    expect(values['count']).toEqual([14]);
    const words = await test.readUint32(packed, 2048);
    words.forEach((value, word) => {
      if (!writtenWords.has(word)) expect(value, `sentinel word ${word}`).toBe(sentinel);
    });
  } finally {
    test.destroy();
  }
});

it('every default pivot ID stays exact when pivotIndex * pointCount exceeds uint32', async () => {
  const test = await createFixture('csr-pivot-overflow');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const count = 68640;
    const coordinates = new Float32Array(count * 2);
    for (let row = 0; row < count; row++) coordinates[2 * row] = row * 2;
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', coordinates, 'float32x2', count),
      segments: test.input('walls', new Float32Array([-100, -100, -99, -100]), 'float32x4', 1),
      maximumDistance: 0.5,
      edgeCapacity: 1,
      pivotCount: count,
      laneCount: 156,
      maximumIterations: 1,
      metricsMode: 'exact'
    });
    const captured = publish(test, 'overflow', {
      pivots: result.pivotNodes,
      reachedCounts: result.outputs.reachedCounts,
      betweenness: result.outputs.betweenness,
      normalizedChoice: result.outputs.normalizedChoice,
      metrics: result.outputs.metrics,
      ...result.status
    });
    compiled = test.graph.compile();
    // Eager frontier seeding previously kept hundreds of batch queues live (35.7 GB).
    // Check the actual physical plan before submitting this large composed regression.
    expect(compiled.stats.physicalTransientBytes).toBeLessThan(768 * 1024 * 1024);
    submitGraph(test.device, compiled, undefined);
    const values = await read(test, captured);
    const expected = Array.from({length: count}, (_, index) =>
      Number((BigInt(index) * BigInt(count)) / BigInt(count))
    );
    expect(values['pivots']).toEqual(expected);
    expect(values['pivots'][62573]).toBe(62573);
    expect(values['reachedCounts']).toEqual(Array(count).fill(1));
    expect(values['betweenness']).toEqual(Array(count).fill(0));
    expect(values['normalizedChoice']).toEqual(Array(count).fill(0));
    for (let row = 0; row < count; row++) {
      expect(values['metrics'].slice(4 * row, 4 * row + 3)).toEqual([0, 0, 0]);
      expect(values['metrics'][4 * row + 3]).toBeNaN();
    }
    expect(values['invalidCount']).toEqual([0]);
    expect(values['count']).toEqual([0]);
    expect(values['converged']).toEqual([1]);
    expect(values['iterationLimitReached']).toEqual([0]);
  } finally {
    compiled?.destroy();
    test.destroy();
  }
}, 120000);

it('an insufficient long-path budget publishes partial metrics and resets stopping flags on re-execution', async () => {
  const test = await createFixture('csr-partial-recovery');
  let compiled: ReturnType<typeof test.graph.compile> | undefined;
  try {
    const count = 32;
    const coordinates = Float32Array.from(Array.from({length: count}, (_, row) => [row, 0]).flat());
    const radius = test.parameterBuffer('radius', 'float32', 2);
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', coordinates, 'float32x2', count),
      segments: test.input('walls', new Float32Array([100, 100, 101, 100]), 'float32x4', 1),
      maximumDistance: 1.1,
      parameters: radius.importToGraph(test.graph),
      edgeCapacity: 64,
      pivotNodes: test.input('pivots', new Uint32Array([0]), 'uint32', 1),
      maximumIterations: 1,
      localIterations: 1
    });
    const captured = publish(test, 'partial', {
      metrics: result.outputs.metrics,
      normalizedChoice: result.outputs.normalizedChoice,
      betweenness: result.outputs.betweenness,
      converged: result.status.converged,
      iterationLimitReached: result.status.iterationLimitReached,
      count: result.status.count
    });
    compiled = test.graph.compile();
    for (const maximumDistance of [1.1, 0.5, 1.1]) {
      radius.write(new Float32Array([maximumDistance, 1e-6]));
      submitGraph(test.device, compiled, undefined);
      const values = await read(test, captured);
      const partial = maximumDistance > 1;
      expect(values['converged']).toEqual([Number(!partial)]);
      expect(values['iterationLimitReached']).toEqual([Number(partial)]);
      expect(values['count']).toEqual([partial ? 62 : 0]);
      expect(values['metrics'].filter((_, index) => index % 4 === 0)).toEqual(
        Array.from({length: count}, (_, row) =>
          partial ? (row === 0 || row === count - 1 ? 1 : 2) : 0
        )
      );
      // Last node is unreached within one round, so its estimated depth is zero, versus 31*31 for a completed source-0 sample.
      expect(values['metrics'][4 * (count - 1) + 2]).toBe(0);
      expect(values['betweenness'].every(Number.isFinite)).toBe(true);
      expect(values['normalizedChoice'].every(Number.isFinite)).toBe(true);
      if (!partial) {
        expect(values['betweenness']).toEqual(Array(count).fill(0));
        expect(values['normalizedChoice']).toEqual(Array(count).fill(0));
        expect(values['metrics'].filter((_, index) => index % 4 === 1 || index % 4 === 2)).toEqual(
          Array(count * 2).fill(0)
        );
      } else {
        expect(values['metrics'][6]).toBe(31);
      }
    }
  } finally {
    compiled?.destroy();
    test.destroy();
  }
});

it.each([
  [68640, 68640],
  [1000000, 68640],
  [999983, 999979]
])(
  'isolated systematic pivot kernel matches every BigInt ID for %i points and %i pivots',
  async (pointCount, pivotCount) => {
    const test = await createFixture('isolated-pivot-arithmetic');
    try {
      const pivots = createTransientView(test.graph, 'pivots', 'uint32', pivotCount);
      test.graph.add(
        createVisibilityGraphPivotsNode(test.graph, {
          id: 'pivot-selection',
          operation: 'visibility-pivots',
          pointCount,
          pivots
        })
      );
      const captured = publish(test, 'isolated', {pivots});
      test.run();
      const values = await read(test, captured);
      const expected = Array.from({length: pivotCount}, (_, index) =>
        Number((BigInt(index) * BigInt(pointCount)) / BigInt(pivotCount))
      );
      expect(values['pivots']).toEqual(expected);
    } finally {
      test.destroy();
    }
  },
  120000
);

it('recipe-owned metric snapshots match the independent three-point path oracle', async () => {
  const test = await createFixture('csr-metric-snapshot');
  let arena: GPUSnapshotArena | undefined;
  try {
    const result = addVisibilityGraphRecipe(test.graph, {
      mode: 'csr',
      positions: test.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3),
      segments: test.input('walls', new Float32Array([0, 10, 2, 10]), 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4,
      pivotCount: 3,
      laneCount: 3,
      maximumIterations: 8,
      localIterations: 4
    });
    const views = {
      metrics: result.outputs.metrics,
      integration: result.outputs.integration,
      extendedIntegration: result.outputs.extendedIntegration,
      normalizedChoice: result.outputs.normalizedChoice
    };
    arena = new GPUSnapshotArena(test.device, {
      slots: Object.entries(views).map(([name, view]) => ({
        name,
        byteLength: view.length * view.rowByteLength,
        elementType: 'f32'
      })),
      capacity: 1
    });
    test.graph.add(new GPUSnapshotTap({id: 'metric-snapshot', arena, views}));
    test.run();
    const oracle = computeVisibilityMetricsOracle([[1], [0, 2], [1]]);
    const normalizations = oracle.maximumDepth.map((maximum, row) =>
      getIntegration(oracle.metrics[4 * row + 1], oracle.metrics[4 * row + 2], maximum)
    );
    const expected = {
      ...oracle,
      integration: normalizations.flatMap(value => value.integration),
      extendedIntegration: normalizations.flatMap(value => value.extendedIntegration)
    };
    const captures = await arena.read();
    expect(captures.map(capture => capture.name)).toEqual(Object.keys(views));
    for (const capture of captures) {
      const name = capture.name as keyof typeof views;
      Array.from(capture.values).forEach((value, index) =>
        expectValue(value, expected[name][index])
      );
    }
  } finally {
    test.destroy();
    arena?.destroy();
  }
});
