// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  getGPUSpaceTimeGuardedRelativeTimes,
  GPU_JACQUEZ_MAXIMUM_K,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUJacquezTest,
  GPULocalKnox,
  GPUNearRepeatKnox
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {getSpaceTimeBandWGSL} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-bands';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  countDistinctPipelines,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 300;
const CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 1_700_000_000n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};
let serial = 0;

function makeViews() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `space-time-statistics-${serial++}`, format, length);
  const pairs = {
    offsets: view('uint32', ROWS + 1),
    neighbors: view('uint32', 4 * ROWS),
    weights: view('float32', 4 * ROWS),
    distances: view('float32', 4 * ROWS)
  };
  return {device, graph, view, pairs};
}

type Views = ReturnType<typeof makeViews>;

const DISTANCE_EDGES = [0, 1, 2.5, 4];
const TIME_EDGES = [0, 1, 6];

function makeNearRepeat(views: Views, overrides: Record<string, unknown> = {}) {
  const {view, pairs} = views;
  return new GPUNearRepeatKnox({
    pairs,
    times: view('float32', ROWS),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    distanceEdges: DISTANCE_EDGES,
    timeEdges: TIME_EDGES,
    maximumPermutations: 99,
    output: {
      counts: view('uint32', 100 * 6),
      summary: view('float32', 6 * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH)
    },
    ...overrides
  });
}

function makeLocal(views: Views, overrides: Record<string, unknown> = {}) {
  const {view, pairs} = views;
  return new GPULocalKnox({
    pairs,
    times: view('float32', ROWS),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    maximumPermutations: 99,
    output: {summary: view('float32', ROWS * GPU_LOCAL_KNOX_SUMMARY_LENGTH)},
    ...overrides
  });
}

function makeJacquez(views: Views, overrides: Record<string, unknown> = {}, permutations = 99) {
  const {view, pairs} = views;
  return new GPUJacquezTest({
    neighbors: pairs,
    times: view('float32', ROWS),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    k: 4,
    maximumPermutations: permutations,
    output: {
      statistics: view('uint32', permutations + 1),
      summary: view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
    },
    ...overrides
  });
}

describe('getGPUSpaceTimeGuardedRelativeTimes', () => {
  const origin = 1_700_000_000;

  it('subtracts the origin in double precision', () => {
    // Raw epoch seconds round to 128 s in float32; the relative values are exact.
    const times = [origin + 3, origin, origin + 5.5, origin + 1200];
    const result = getGPUSpaceTimeGuardedRelativeTimes(times, 1);
    expect(Array.from(result.times)).toEqual([3, 0, 5.5, 1200]);
    expect(result.origin).toBe(origin);
    expect(result.temporalContext).toBeUndefined();
  });

  it('throws when the float32 spacing exceeds a quarter of the narrowest band', () => {
    // A span of 20,000,000 s is between 2^24 and 2^25, so the float32 spacing is 2 s.
    const span = 20_000_000;
    const times = [origin, origin + span / 2, origin + span];
    expect(() => getGPUSpaceTimeGuardedRelativeTimes(times, 1)).toThrow(
      /float32 resolution 2 exceeds 0.25/
    );
    // 2 s of spacing is at most a quarter of an 8 s band.
    expect(getGPUSpaceTimeGuardedRelativeTimes(times, 8).resolution).toBe(2);
    expect(() => getGPUSpaceTimeGuardedRelativeTimes(times, 7.9)).toThrow(/exceeds/);
  });

  it('rejects a non-positive or non-finite narrowest width', () => {
    for (const width of [0, -1, Number.NaN, Infinity]) {
      expect(() => getGPUSpaceTimeGuardedRelativeTimes([1, 2], width)).toThrow(
        /narrowest time width/
      );
    }
  });

  it('declares the float32 context with the epoch moved to the origin', () => {
    const times = [origin + 7.5, origin + 2.25, origin + 40];
    const fromUnix = getGPUSpaceTimeGuardedRelativeTimes(times, 1, {
      temporalContext: {unit: 'second', epoch: 'unix', timeZone: 'UTC'}
    });
    // The minimum is rounded down to a whole tick so the epoch stays an Int64.
    expect(fromUnix.origin).toBe(origin + 2);
    expect(Array.from(fromUnix.times)).toEqual([5.5, 0.25, 38]);
    expect(fromUnix.temporalContext).toEqual({
      unit: 'second',
      epoch: BigInt(origin + 2),
      encoding: 'float32-relative',
      timeZone: 'UTC',
      axis: {kind: 'irregular'}
    });
    const fromOrigin = getGPUSpaceTimeGuardedRelativeTimes(times, 1, {
      origin: origin + 1,
      temporalContext: {unit: 'millisecond', epoch: 5n}
    });
    expect(fromOrigin.temporalContext?.epoch).toBe(5n + BigInt(origin + 1));
    expect(fromOrigin.temporalContext?.unit).toBe('millisecond');
    expect(() =>
      getGPUSpaceTimeGuardedRelativeTimes([10, 20.5], 1, {
        origin: 1.5,
        temporalContext: {unit: 'second', epoch: 'unix'}
      })
    ).toThrow(/whole number of ticks/);
  });

  it('produces a context the contributors accept', () => {
    const result = getGPUSpaceTimeGuardedRelativeTimes([origin, origin + 10], 1, {
      temporalContext: {unit: 'second', epoch: 'unix'}
    });
    expect(makeLocal(makeViews(), {temporalContext: result.temporalContext}).temporalContext).toBe(
      result.temporalContext
    );
  });
});

describe('space-time statistics temporalContext', () => {
  const makers = {
    GPUNearRepeatKnox: makeNearRepeat,
    GPULocalKnox: makeLocal,
    GPUJacquezTest: makeJacquez
  } as const;
  for (const [name, make] of Object.entries(makers)) {
    it(`${name} exposes a float32-relative context and rejects other encodings`, () => {
      expect(make(makeViews(), {temporalContext: CONTEXT}).temporalContext).toBe(CONTEXT);
      expect(make(makeViews()).temporalContext).toBeUndefined();
      for (const encoding of ['int64-words', 'double-single'] as const) {
        expect(() =>
          make(makeViews(), {
            temporalContext: {...CONTEXT, encoding, epoch: 'unix'}
          })
        ).toThrow(new RegExp(`does not support ${encoding} times; it reads float32-relative`));
      }
      // float32 cannot hold absolute time: a Unix epoch is refused by the context contract.
      expect(() => make(makeViews(), {temporalContext: {...CONTEXT, epoch: 'unix'}})).toThrow(
        /float32-relative times need a bigint origin epoch/
      );
      expect(() =>
        make(makeViews(), {temporalContext: {...CONTEXT, unit: 'minute' as never}})
      ).toThrow(/temporal unit minute must be one of/);
    });
  }
});

describe('GPUNearRepeatKnox validation', () => {
  it('derives the cell layout and rejects bad band edges', () => {
    const table = makeNearRepeat(makeViews());
    expect([table.distanceBands, table.timeBands, table.cells]).toEqual([3, 2, 6]);
    for (const edges of [[1], [0, 1, 1], [0, 2, 1], [-1, 1], [0, Number.NaN]]) {
      expect(() => makeNearRepeat(makeViews(), {distanceEdges: edges})).toThrow(/distanceEdges/);
      expect(() => makeNearRepeat(makeViews(), {timeEdges: edges})).toThrow(/timeEdges/);
    }
    // Edges that round together in float32 would make an empty band.
    expect(() => makeNearRepeat(makeViews(), {distanceEdges: [1, 1 + 1e-9]})).toThrow(
      /strictly ascending as float32/
    );
    const seventeen = Array.from({length: 18}, (_, index) => index);
    expect(() => makeNearRepeat(makeViews(), {distanceEdges: seventeen})).toThrow(/2 to 17 edges/);
    // 16 by 16 bands is the largest table: 256 cells.
    const largest = makeViews();
    const largestTable = makeNearRepeat(largest, {
      distanceEdges: seventeen.slice(0, 17),
      timeEdges: seventeen.slice(0, 17),
      output: {
        counts: largest.view('uint32', 100 * 256),
        summary: largest.view('float32', 256 * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH)
      }
    });
    expect(largestTable.cells).toBe(256);
  });

  it('requires distances and sized, separate outputs', () => {
    const views = makeViews();
    const {distances: _distances, ...withoutDistances} = views.pairs;
    expect(() => makeNearRepeat(views, {pairs: withoutDistances})).toThrow(
      /pairs.distances is required/
    );
    const {view} = views;
    expect(() =>
      makeNearRepeat(views, {
        output: {counts: view('uint32', 599), summary: view('float32', 48)}
      })
    ).toThrow(/output.counts needs \(maximumPermutations \+ 1\) \* cells rows/);
    expect(() =>
      makeNearRepeat(views, {
        output: {counts: view('uint32', 600), summary: view('float32', 47)}
      })
    ).toThrow(/output.summary needs cells \* 8 rows/);
    expect(() => makeNearRepeat(views, {maximumPermutations: 65537})).toThrow(
      /maximumPermutations must be an integer in \[1, 65536\]/
    );
  });

  it('describes its inference as one greater test per cell', () => {
    expect(
      makeNearRepeat(makeViews()).getInferenceMetadata({seed: 3, permutationCount: 500})
    ).toEqual({
      alternative: 'greater',
      permutationCount: 99,
      seed: 3,
      multipleTesting: 'none',
      testedRowsDefinition:
        'one test per cell of distance band by time band over all unordered event pairs listed in the spatial pair list'
    });
  });

  it('writes band lookups with the inclusive-upper convention', () => {
    const source = getSpaceTimeBandWGSL('Distance', [0, 1, 2.5, 4]);
    expect(source).toContain('fn getDistanceBand(value: f32) -> u32');
    expect(source).toContain('if (value < 0.0 || value > 4.0)');
    expect(source).toContain('if (value > 1.0) { band = 1u; }');
    expect(source).toContain('if (value > 2.5) { band = 2u; }');
    expect(source).not.toContain('band = 3u');
    // A float32-inexact edge is written as its float32 value.
    expect(getSpaceTimeBandWGSL('Time', [0, 0.1, 1])).toContain(`value > ${Math.fround(0.1)}`);
  });
});

describe('GPULocalKnox and GPUJacquezTest validation', () => {
  it('sizes the local summary', () => {
    const views = makeViews();
    expect(() =>
      makeLocal(views, {
        output: {summary: views.view('float32', ROWS * GPU_LOCAL_KNOX_SUMMARY_LENGTH - 1)}
      })
    ).toThrow(/output.summary needs rows \* 6 rows/);
  });

  it('checks k against the rows and the maximum', () => {
    for (const k of [0, 1.5, GPU_JACQUEZ_MAXIMUM_K + 1]) {
      expect(() => makeJacquez(makeViews(), {k})).toThrow(/k must be an integer in \[1, 256\]/);
    }
    expect(makeJacquez(makeViews(), {k: 256}).rows).toBe(ROWS);
    const small = makeViews();
    const offsets = small.view('uint32', 6);
    expect(() =>
      makeJacquez(small, {
        neighbors: {...small.pairs, offsets},
        times: small.view('float32', 5),
        k: 5
      })
    ).toThrow(/needs more than k rows/);
  });

  it('validates stopAfterExceedances like GPUKnoxTest', () => {
    const views = makeViews();
    const output = (permutationCount?: boolean) => ({
      statistics: views.view('uint32', 100),
      summary: views.view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH),
      permutationCount: permutationCount ? views.view('uint32', 1) : undefined
    });
    expect(() => makeJacquez(views, {stopAfterExceedances: 100, output: output(true)})).toThrow(
      /stopAfterExceedances must be an integer in \[1, 99\]/
    );
    expect(() => makeJacquez(views, {stopAfterExceedances: 5, output: output()})).toThrow(
      /stopAfterExceedances needs output.permutationCount/
    );
    expect(() => makeJacquez(views, {stopAfterExceedances: 5, output: output(true)})).not.toThrow();
  });
});

describe('space-time statistics kernels', () => {
  const builders: Record<string, (views: Views) => ReadonlyArray<{id: string}>> = {
    GPUNearRepeatKnox: views => {
      const test = makeNearRepeat(views);
      return test.getCommandNodes(views.graph);
    },
    GPULocalKnox: views => makeLocal(views).getCommandNodes(views.graph),
    GPUJacquezTest: views => makeJacquez(views).getCommandNodes(views.graph),
    'GPUJacquezTest with stopping': views => {
      const permutationCount = views.view('uint32', 1);
      return makeJacquez(views, {
        stopAfterExceedances: 10,
        output: {
          statistics: views.view('uint32', 100),
          summary: views.view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH),
          permutationCount
        }
      }).getCommandNodes(views.graph);
    }
  };
  for (const [name, build] of Object.entries(builders)) {
    it(`${name} kernels fit the core limits`, () => {
      const views = makeViews();
      const sources = getGPUCommandNodeKernelSources(build(views));
      views.device.destroy();
      expect(sources.length).toBeGreaterThan(2);
      for (const {id, source} of sources) {
        expect(getStorageBindingCount(source), `${id} bindings`).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
        expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_WORKGROUP_STORAGE_LIMIT_BYTES
        );
      }
    });
  }

  it('GPUNearRepeatKnox reaches the workgroup barriers from uniform control flow', () => {
    const views = makeViews();
    const sources = getGPUCommandNodeKernelSources(
      makeNearRepeat(views).getCommandNodes(views.graph)
    );
    views.device.destroy();
    const histogram = sources.find(({id}) => id.endsWith('-histogram'))!.source;
    expect(histogram).toContain('var<workgroup> histogram: array<atomic<u32>, 6>;');
    const body = histogram.slice(histogram.indexOf('fn main('));
    expect(body.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    // The kernel body has no early return before a barrier (the wrapper's own return is on the
    // workgroup index, uniform across a workgroup, and precedes the body).
    const kernelBody = body.slice(body.indexOf('let slot = index'));
    expect(kernelBody.slice(0, kernelBody.lastIndexOf('workgroupBarrier()'))).not.toContain(
      'return'
    );
  });

  it('GPUJacquezTest builds the same pipelines for N and N + 1 stopping batches', () => {
    const build = (permutations: number) => {
      const views = makeViews();
      const nodes = makeJacquez(
        views,
        {
          stopAfterExceedances: 5,
          maximumPermutations: permutations,
          output: {
            statistics: views.view('uint32', permutations + 1),
            summary: views.view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH),
            permutationCount: views.view('uint32', 1)
          }
        },
        permutations
      ).getCommandNodes(views.graph);
      return {nodes, views};
    };
    const three = build(3 * 64 - 1);
    const four = build(3 * 64);
    expect(four.nodes.length - three.nodes.length).toBe(3);
    expect(countDistinctPipelines(four.nodes)).toBe(countDistinctPipelines(three.nodes));
  });
});
