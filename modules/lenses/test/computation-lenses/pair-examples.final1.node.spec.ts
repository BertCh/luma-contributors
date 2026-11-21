// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, beforeEach, expect, test, vi} from 'vitest';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
// Other specs run the real examples with isolate:false. Load fresh modules after installing
// this file's recording mocks rather than retaining another spec's cached core bindings.
beforeEach(() => {
  vi.resetModules();
  probes.join.length = 0;
  probes.gather.length = 0;
  probes.histograms.length = 0;
});
afterEach(() => vi.resetModules());
const gatherExample = (
  ...parameters: Parameters<
    typeof import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example').runExample
  >
) =>
  import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example').then(
    module => module.runExample(...parameters)
  );
const predicateExample = (
  ...parameters: Parameters<
    typeof import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.example').runExample
  >
) =>
  import(
    '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.example'
  ).then(module => module.runExample(...parameters));
const preparedExample = (
  ...parameters: Parameters<
    typeof import('../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-join-prepared.example').runExample
  >
) =>
  import(
    '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-join-prepared.example'
  ).then(module => module.runExample(...parameters));

const probes = vi.hoisted(() => ({
  join: [] as Record<string, unknown>[],
  gather: [] as Record<string, unknown>[],
  histograms: [] as Record<string, unknown>[]
}));
vi.mock('@luma.gl/graph/gpu-core', () => ({
  createTransientView: (_graph: unknown, id: string, format: string, length: number) => ({
    id,
    format,
    length
  }),
  GPUHistogram: class {
    constructor(props: Record<string, unknown>) {
      probes.histograms.push(props);
    }
  }
}));
vi.mock('@luma-contributors/analysis/analysis-core', async () => ({
  createWGSLKernelNode: (_graph: unknown, props: unknown) => props,
  validateRequirement: (
    await vi.importActual<typeof import('@luma-contributors/analysis/analysis-core')>(
      '@luma-contributors/analysis/analysis-core'
    )
  ).validateRequirement
}));
vi.mock('@luma-contributors/analysis/gpu-spatial-analysis', () => ({
  GPUSpatialPredicateJoin: class {
    constructor(props: Record<string, unknown>) {
      probes.join.push(props);
    }
  },
  GPUPairGather: class {
    constructor(props: Record<string, unknown>) {
      probes.gather.push(props);
    }
  },
  GPUSpatialJoinPrepared: class {},
  GPUSpatialJoinCandidates: class {}
}));

const view = (length: number) => ({format: 'uint32', length});
function inputs() {
  return {
    parks: {featureOffsets: view(2)},
    tracts: {featureOffsets: view(3)},
    pairs: {leftRows: view(2), rightRows: view(2), status: {count: view(1), overflow: view(1)}},
    parkAreas: view(1),
    spatialContext: {coordinateSpace: 'planar'},
    candidateCapacity: 1
  };
}
const createGraph = () => ({add: vi.fn()}) as unknown as GPUCommandGraph<undefined>;
test('F4 example completes an omitted producer candidate flag and shares the completed port', async () => {
  const graph = createGraph();
  const input = inputs();
  await gatherExample(graph, input as never);
  const join = probes.join.at(-1)! as {output: {pairs: {status: {candidateOverflow?: unknown}}}};
  const gather = probes.gather.at(-1)!;
  expect(join.output.pairs.status.candidateOverflow).toBeDefined();
  expect(gather['pairs']).toBe(join.output.pairs);
});
test('the public pair example captures getter-backed pair and status fields once', async () => {
  const graph = createGraph();
  const input = inputs();
  const leftRows = input.pairs.leftRows;
  const count = input.pairs.status.count;
  const requiredCount = view(1);
  const candidateOverflow = view(1);
  let rowReads = 0;
  let requiredReads = 0;
  let candidateReads = 0;
  Object.defineProperty(input.pairs, 'leftRows', {
    get() {
      if (++rowReads > 1) throw new Error('leftRows was read more than once');
      return leftRows;
    }
  });
  input.pairs.status = Object.assign(
    Object.create({
      get requiredCount() {
        requiredReads++;
        return requiredCount;
      },
      get candidateOverflow() {
        candidateReads++;
        return candidateOverflow;
      }
    }),
    input.pairs.status
  );
  await gatherExample(graph, input as never);
  const {pairs} = probes.join.at(-1)!['output'] as {
    pairs: {
      leftRows: unknown;
      status: {count: unknown; requiredCount: unknown; candidateOverflow: unknown};
    };
  };
  expect([rowReads, requiredReads, candidateReads]).toEqual([1, 1, 1]);
  expect(pairs.leftRows).toBe(leftRows);
  expect(pairs.status.requiredCount).toBe(requiredCount);
  expect(pairs.status.candidateOverflow).toBe(candidateOverflow);
  expect(Object.isFrozen(pairs)).toBe(true);
  expect(Object.isFrozen(pairs.status)).toBe(true);
  input.pairs.status.count = view(0);
  expect(pairs.status.count).toBe(count);
  expect(probes.gather.at(-1)!['pairs']).toBe(pairs);
});
for (const [name, example] of [
  ['gather', gatherExample],
  ['predicate', predicateExample],
  ['prepared', preparedExample]
] as const) {
  test(`F5 ${name} histogram consumes a GPU mask of the published pair prefix`, async () => {
    const graph = createGraph();
    const input = {...inputs(), candidatePairs: inputs().pairs};
    await example(graph, input as never);
    expect(probes.histograms).toHaveLength(1);
    const mask = probes.histograms[0]['mask'];
    expect(mask).toMatchObject({length: 2});
    const kernels = vi.mocked(graph.add).mock.calls.map(
      call =>
        call[0] as unknown as {
          invocationCount?: number;
          bindings?: {name: string; view: unknown}[];
          body?: string;
        }
    );
    expect(
      kernels.some(
        kernel =>
          kernel.invocationCount === 2 &&
          kernel.bindings?.some(
            binding => binding.name === 'pairCount' && binding.view === input.pairs.status.count
          ) &&
          kernel.bindings?.some(binding => binding.name === 'mask' && binding.view === mask) &&
          kernel.body?.includes('index < pairCount[')
      )
    ).toBe(true);
    const membership = kernels.find(kernel =>
      kernel.bindings?.some(binding => binding.name === 'mask' && binding.view === mask)
    );
    // select(falseValue, trueValue, predicate): reversing the arms selects spare slots.
    expect(membership?.body).toBe(
      'mask[maskOffset + index] = select(0u, 1u, index < pairCount[pairCountOffset]);'
    );
  });
}
