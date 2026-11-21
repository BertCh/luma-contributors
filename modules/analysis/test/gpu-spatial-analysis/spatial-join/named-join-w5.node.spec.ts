// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createCPUSpatialPredicateJoinOperation,
  createSpatialPredicateJoinOperation,
  type SpatialPredicateJoinOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {
  computeSpatialPredicateJoinOnCPU,
  isPointWithinSegmentDistance
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {WgslParser} from 'wgsl_reflect';
import {getCPUPointJoinOrientation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-join-geometry';
import {validatePointJoinOffsets} from '../../../src/gpu-spatial-analysis/spatial-join/point-join-operation-utils';
import {getSpatialDwithinWorkgroupWGSL} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-dwithin-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';

function rows(values: Float32Array | Uint32Array, components = 1) {
  const length = values.length / components;
  return {
    format: components === 2 ? ('float32x2' as const) : ('uint32' as const),
    length,
    chunks: [{values, length}],
    pack: () => values
  };
}
const descriptor = (value: string, length: number) => ({value, length});
function props(): SpatialPredicateJoinOperationProps {
  return {
    left: {kind: 'points', positions: descriptor('points', 3)},
    right: {kind: 'points', positions: descriptor('targets', 2)},
    predicate: 'intersects',
    candidateCapacity: 6,
    pairCapacity: 1,
    output: {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    }
  };
}
const inputs = () => ({
  leftPositions: rows(Float32Array.of(0, 0, 0, 0, 9, 9), 2),
  rightPositions: rows(Float32Array.of(0, 0, 0, 0), 2)
});

it('UGE-02 named live count shrinks and preserves a genuine empty through every polygon terminal', async () => {
  const input: SpatialPredicateJoinOperationProps = {
    ...props(),
    how: 'anti',
    leftFeatureCount: 'live',
    leftStatus: {overflow: 'loss', candidateOverflow: 'candidateLoss', complete: 'complete'},
    left: {
      kind: 'polygons',
      positions: descriptor('squares', 8),
      featureOffsets: descriptor('features', 4),
      polygonOffsets: descriptor('polygons', 3),
      ringOffsets: descriptor('rings', 3)
    },
    right: {kind: 'points', positions: descriptor('target', 1)},
    pairCapacity: 3,
    output: {
      unmatchedRows: 'unmatchedRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateTruncated: 'truncated'
    }
  };
  const operation = createCPUSpatialPredicateJoinOperation(input);
  const run = (live: number, loss = 0, candidateLoss = 0, complete = 1) =>
    operation.execute({
      inputs: {
        leftPositions: rows(Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 8, 0, 10, 0, 10, 2, 8, 2), 2),
        leftFeatureOffsets: rows(Uint32Array.of(0, 1, 1, 2)),
        leftPolygonOffsets: rows(Uint32Array.of(0, 1, 2)),
        leftRingOffsets: rows(Uint32Array.of(0, 4, 8)),
        rightPositions: rows(Float32Array.of(1, 1), 2),
        leftFeatureCount: rows(Uint32Array.of(live)),
        leftStatusOverflow: rows(Uint32Array.of(loss)),
        leftStatusCandidateOverflow: rows(Uint32Array.of(candidateLoss)),
        leftStatusComplete: rows(Uint32Array.of(complete))
      }
    });
  // Square row 0 contains the point; row 1 is genuinely empty; row 2 misses.
  for (const [live, loss, expected] of [
    [3, 0, [1, 2]],
    [1, 0, []],
    [2, 0, [1]],
    [2, 1, []],
    [2, 0, [1]]
  ] as const) {
    const result = await run(live, loss);
    expect(Array.from(result['count'] as Uint32Array)).toEqual([expected.length]);
    expect(Array.from(result['unmatchedRows'] as Uint32Array).slice(0, expected.length)).toEqual(
      expected
    );
    expect(Array.from(result['candidateOverflow'] as Uint32Array)).toEqual([loss]);
    expect((result['truncated'] as Uint32Array).length).toBe(3);
  }
  for (const [live, candidateLoss, complete] of [
    [2, 1, 1],
    [2, 0, 0],
    [4, 0, 1]
  ] as const) {
    const result = await run(live, 0, candidateLoss, complete);
    expect(Array.from(result['count'] as Uint32Array)).toEqual([0]);
    expect(Array.from(result['candidateOverflow'] as Uint32Array)).toEqual([1]);
  }
  expect(Array.from((await run(0))['count'] as Uint32Array)).toEqual([0]);
});

it('UGE-12 inner unmatched membership precedes pair truncation and excludes candidate unknowns', async () => {
  for (const candidateCapacity of [1, 6]) {
    const input = props();
    input.candidateCapacity = candidateCapacity;
    input.output = {
      ...input.output,
      unmatchedRows: 'unmatchedRows',
      unmatchedCount: 'unmatchedCount',
      unmatchedRequiredCount: 'unmatchedRequiredCount',
      unmatchedOverflow: 'unmatchedOverflow',
      unmatchedCandidateOverflow: 'unmatchedCandidateOverflow'
    };
    const result = await createCPUSpatialPredicateJoinOperation(input).execute({inputs: inputs()});
    // Rows 0 and 1 match two points each before pair capacity 1; row 2 is a true miss.
    expect(Array.from(result['leftRows'] as Uint32Array)).toEqual([0]);
    expect(Array.from(result['unmatchedRows'] as Uint32Array)).toEqual([2]);
    expect(Array.from(result['unmatchedCount'] as Uint32Array)).toEqual([1]);
    expect(Array.from(result['unmatchedRequiredCount'] as Uint32Array)).toEqual([1]);
    expect(Array.from(result['unmatchedCandidateOverflow'] as Uint32Array)).toEqual([
      Number(candidateCapacity === 1)
    ]);
  }
});

it('W5-02 named and direct plans agree across crossover, multipart and explicit engines', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const vertexCount of [31, 32, 33])
      for (const multipart of [false, true])
        for (const engine of [undefined, 'fast', 'relate'] as const) {
          const graph = new GPUCommandGraph(device);
          const input = props();
          input.engine = engine;
          input.left = {
            kind: 'lines',
            positions: descriptor('left', vertexCount),
            lineOffsets: descriptor('leftLines', 2),
            ...(multipart ? {featureOffsets: descriptor('leftFeatures', 2)} : {})
          };
          input.right = {
            kind: 'lines',
            positions: descriptor('right', vertexCount),
            lineOffsets: descriptor('rightLines', 2)
          };
          const operation = createSpatialPredicateJoinOperation(input);
          const views = Object.fromEntries(
            Object.entries(operation.outputs).map(([port, output]) => [
              port,
              createTransientView(graph, port, output.format, output.length)
            ])
          );
          const left = {
            kind: 'lines' as const,
            positions: createTransientView(graph, 'left', 'float32x2', vertexCount),
            lineOffsets: createTransientView(graph, 'left-lines', 'uint32', 2),
            ...(multipart
              ? {featureOffsets: createTransientView(graph, 'left-features', 'uint32', 2)}
              : {})
          };
          const right = {
            kind: 'lines' as const,
            positions: createTransientView(graph, 'right', 'float32x2', vertexCount),
            lineOffsets: createTransientView(graph, 'right-lines', 'uint32', 2)
          };
          views['leftPositions'] = left.positions;
          views['leftLineOffsets'] = left.lineOffsets;
          if (left.featureOffsets) views['leftFeatureOffsets'] = left.featureOffsets;
          views['rightPositions'] = right.positions;
          views['rightLineOffsets'] = right.lineOffsets;
          const contributor = operation.createContributor(
            port => views[port] as never
          ) as GPUSpatialPredicateJoin;
          const direct = new GPUSpatialPredicateJoin({
            left,
            right,
            engine,
            predicate: 'intersects',
            candidateCapacity: 6,
            output: {
              pairs: {
                leftRows: views['leftRows'] as never,
                rightRows: views['rightRows'] as never,
                status: {count: views['count'] as never, overflow: views['overflow'] as never}
              }
            }
          });
          expect(contributor.usesRelateEngine).toBe(direct.usesRelateEngine);
          expect(contributor.usesRelateEngine).toBe(
            engine === 'relate' || (engine === undefined && !multipart && vertexCount >= 32)
          );
          operation.destroy();
        }
  } finally {
    device.destroy();
  }
});

it('KI-rr-joins-R5 zero radius scans the intersection once per workgroup', () => {
  const side = {kind: 'points', prefix: 'left', vertexCount: 1} as const;
  const source = getSpatialDwithinWorkgroupWGSL(side, {...side, prefix: 'right'}, 0);
  expect(source).toMatch(/if \(lane == 0u\) \{\s*if \(pairIntersects\(l, r\)\)/);
  new WgslParser().parse(source);
});

it('UGE-02 rejects malformed CPU producer status and both named forms retain dwithin output rejection', () => {
  for (const value of [Number.NaN, Infinity, -1, 1.5])
    expect(() =>
      computeSpatialPredicateJoinOnCPU(
        {kind: 'points', positions: Float32Array.of(0, 0)},
        {kind: 'points', positions: Float32Array.of(0, 0)},
        {...props(), leftStatus: {overflow: value}}
      )
    ).toThrow(/leftStatus.overflow/);
  for (const factory of [
    createSpatialPredicateJoinOperation,
    createCPUSpatialPredicateJoinOperation
  ])
    expect(() =>
      factory({
        ...props(),
        predicate: 'dwithin',
        distance: 1,
        output: {...props().output, relate: 'relate'}
      })
    ).toThrow(/output.relate.*topological predicate/);
});

it('KI-rr-joins-R7 clear point comparisons avoid BigInt but inclusive boundaries use it', () => {
  const exact = vi.spyOn(DataView.prototype, 'getBigUint64');
  try {
    expect(isPointWithinSegmentDistance([0, 0], [1, 1], [1, 1], 2)).toBe(true);
    expect(isPointWithinSegmentDistance([0, 0], [3, 3], [3, 3], 2)).toBe(false);
    expect(exact).not.toHaveBeenCalled();
    expect(isPointWithinSegmentDistance([0, 0], [1, 0], [1, 0], 1)).toBe(true);
    expect(exact).toHaveBeenCalled();
  } finally {
    exact.mockRestore();
  }
  for (const [start, end] of [
    [
      [0, 0],
      [10068795, 13425060]
    ],
    [
      [10068795, 13425060],
      [0, 0]
    ]
  ] as const)
    expect(isPointWithinSegmentDistance([2, 1], start, end, 1)).toBe(true);
});

it('W5-04 shared compact offsets and orientation preserve empty and exact contracts', () => {
  validatePointJoinOffsets('offsets', 'rows', Uint32Array.of(0), 0);
  validatePointJoinOffsets('offsets', 'rows', Uint32Array.of(0, 0, 1, 1), 1);
  expect(() => validatePointJoinOffsets('offsets', 'rows', Uint32Array.of(0, 0), 1)).toThrow(
    /terminal offset/
  );
  expect(getCPUPointJoinOrientation(0, 0, 1, 1, 2, 2)).toBe(0);
  expect(getCPUPointJoinOrientation(0, 0, 1, 1, 2, 2 + 2 ** -51)).toBe(1);
  expect(Number.isNaN(getCPUPointJoinOrientation(0, 0, 1, 1, Infinity, 2))).toBe(true);
  let reads = 0;
  const offsets = new Proxy(Uint32Array.of(0, 0, 1, 1), {
    get(target, property) {
      if (typeof property === 'string' && /^\d+$/.test(property)) reads++;
      return Reflect.get(target, property, target);
    }
  });
  validatePointJoinOffsets('offsets', 'rows', offsets, 1);
  // At most one initial-offset read plus one read of each offset in a single traversal.
  expect(reads).toBeLessThanOrEqual(5);
});
