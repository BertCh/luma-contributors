// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it, vi} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import * as cpuJoin from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  createCPUSpatialPredicateJoinOperation,
  createSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {computePointInPolygonCountsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {createSpatialJoinLiveTruncationNode} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-live-input';
import {addSpatialPredicateUnmatchedNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-unmatched';
import {getSpatialDwithinWorkgroupWGSL} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-dwithin-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('B1 oversized live count suppresses emitted unmatched flags then recovers on valid replay', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const view = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const live = view('live'),
    truncated = view('truncated', 2),
    state = view('state', 6);
  const nodes: GPUCommandNode[] = [];
  const clear = createSpatialJoinLiveTruncationNode(graph, 'review', live, truncated);
  addSpatialPredicateUnmatchedNodes(graph, nodes, {
    id: 'review',
    leftCount: 2,
    leftFeatureCount: live,
    candidateCapacity: 1,
    finishOnly: true,
    candidatePairs: createTransientView(graph, 'pairs', 'uint32x2', 1),
    flags: view('flags'),
    leftMatchCounts: view('counts', 2),
    state,
    candidateTruncated: truncated,
    unmatched: {ids: view('ids', 2), status: {count: view('count'), overflow: view('overflow')}}
  });
  const classify = nodes.find(node => node.id === 'review-unmatched-flags')!;
  const destination = classify.resources!.find(
    resource => 'buffer' in resource && resource.usage !== 'storage-read'
  )!;
  if (!('buffer' in destination)) throw new Error('missing unmatched flags');
  const memory = createCPUGraphMemory();
  try {
    for (const [count, expected] of [
      [3, [0, 0]],
      [1, [1, 0]]
    ] as const) {
      memory.write(live, Uint32Array.of(count));
      memory.write(truncated, Uint32Array.of(count > 2 ? 1 : 0, count > 2 ? 1 : 0));
      executeCommandNodesOnCPU([clear, classify], memory);
      expect(Array.from(memory.readUint32(destination.buffer as GraphDataView))).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});

it('B2 retains the existing seven-read point hierarchy budget', () => {
  let reads = 0;
  const featureOffsets = new Proxy(Uint32Array.of(0, 1), {
    get(target, key) {
      if (typeof key === 'string' && /^\d+$/.test(key)) reads++;
      return Reflect.get(target, key, target);
    }
  });
  const result = computePointInPolygonCountsOnCPU(
    {
      points: {kind: 'points', positions: Float32Array.of(0.25, 0.25), featureOffsets},
      polygons: {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 1, 0, 0, 1),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 3)
      }
    },
    {candidateCapacity: 1, matchCapacity: 1}
  );
  expect(Array.from(result.pointFeatureRows)).toEqual([0]);
  expect(reads).toBeLessThanOrEqual(7);
});

it('B3 retains a typed contributor guard', () => {
  const spec = readFileSync(new URL('./named-point-join-w5.node.spec.ts', import.meta.url), 'utf8');
  expect(spec).toContain('contributor instanceof GPUPointInPolygonJoin');
});

it('S1 pure CPU captures all producer status getters once before validation and row traversal', () => {
  const reads = {overflow: 0, candidateOverflow: 0, complete: 0};
  const geometry = {kind: 'points' as const, positions: Float32Array.of(1, 1)};
  const result = cpuJoin.computeSpatialPredicateJoinOnCPU(
    geometry,
    geometry,
    Object.create({
      predicate: 'intersects',
      candidateCapacity: 1,
      pairCapacity: 1,
      leftStatus: {
        get overflow() {
          reads.overflow++;
          return 0;
        },
        get candidateOverflow() {
          reads.candidateOverflow++;
          return 0;
        },
        get complete() {
          return ++reads.complete < 3 ? 1 : 0;
        }
      }
    })
  );
  // The first captured certificate is complete and both identical points match.
  expect(result.status.count).toBe(1);
  expect(result.unmatchedStatus.count).toBe(0);
  expect(reads).toEqual({overflow: 1, candidateOverflow: 1, complete: 1});
});

it('S2 zero-radius branch precedes the all-lane geometry finiteness scans', () => {
  const side = {kind: 'lines', prefix: 'left', vertexCount: 2} as const;
  const source = getSpatialDwithinWorkgroupWGSL(side, {...side, prefix: 'right'}, 0).split(
    'fn dwithinWorkgroupPair'
  )[1];
  expect(source.indexOf('if (limit == 0.0)')).toBeLessThan(source.indexOf('leftFirst(l)'));
});

const rows = (values: Float32Array | Uint32Array, components = 1) => ({
  format: components === 2 ? ('float32x2' as const) : ('uint32' as const),
  length: values.length / components,
  chunks: [{values, length: values.length / components}],
  pack: () => values
});
const base = () => ({
  left: {kind: 'points' as const, positions: {value: 'left', length: 3}},
  right: {kind: 'points' as const, positions: {value: 'right', length: 1}},
  predicate: 'intersects' as const,
  candidateCapacity: 3,
  pairCapacity: 1,
  output: {
    leftRows: 'leftRows',
    rightRows: 'rightRows',
    count: 'count',
    requiredCount: 'requiredCount',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow'
  }
});
it('S3 named CPU only pads requested shorter truncation output and reuses full-length output', async () => {
  const compute = vi.spyOn(cpuJoin, 'computeSpatialPredicateJoinOnCPU');
  const copies = vi.spyOn(Uint32Array, 'from');
  try {
    for (const requested of [false, true])
      for (const live of [1, 3]) {
        const input = base();
        const operation = createCPUSpatialPredicateJoinOperation({
          ...input,
          leftFeatureCount: 'live',
          output: {...input.output, ...(requested ? {candidateTruncated: 'truncated'} : {})}
        });
        const result = await operation.execute({
          inputs: {
            leftPositions: rows(Float32Array.of(0, 0, 1, 1, 2, 2), 2),
            rightPositions: rows(Float32Array.of(0, 0), 2),
            leftFeatureCount: rows(Uint32Array.of(live))
          }
        });
        expect(copies).not.toHaveBeenCalled();
        if (requested) {
          expect(Array.from(result['truncated'] as Uint32Array)).toEqual([0, 0, 0]);
          if (live === 3)
            expect(result['truncated']).toBe(compute.mock.results.at(-1)!.value.candidateTruncated);
        }
      }
  } finally {
    compute.mockRestore();
    copies.mockRestore();
  }
});

it('S5 both factories reject separate unmatched diagnostics without inner unmatchedRows', () => {
  for (const factory of [
    createSpatialPredicateJoinOperation,
    createCPUSpatialPredicateJoinOperation
  ])
    for (const how of ['inner', 'anti'] as const) {
      const input = base();
      const output =
        how === 'inner'
          ? input.output
          : {
              unmatchedRows: 'unmatchedRows',
              count: 'count',
              requiredCount: 'requiredCount',
              overflow: 'overflow',
              candidateOverflow: 'candidateOverflow'
            };
      expect(() =>
        factory({
          ...input,
          how,
          output: {...output, unmatchedRequiredCount: 'unmatchedRequiredCount'}
        })
      ).toThrow(/separate unmatched diagnostics require inner output.unmatchedRows/);
    }
});
