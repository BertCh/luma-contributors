// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import * as featureRelate from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-relate';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {
  computeSpatialPredicateJoinOnCPU,
  computeSpatialRelateOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  createMultipartCPUGeometry,
  createMultipartOperationInputs
} from './multipart-predicates-test-utils';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {
  REVIEW_DISTANCE_THRESHOLD,
  REVIEW_TRANSLATED_DISTANCE,
  REVIEW_NARROW_HOLE,
  REVIEW_EQUAL_SQUARE
} from './multipart-review-fixtures';
import {runMultipartScalar} from './multipart-scalar-test-utils';
import {getSpatialPredicateWGSL} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl';
import {getSpatialRelateWGSL} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-wgsl';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';

it('R1 includes the exact 6.5 threshold with the nearest point in a later part', () => {
  // Segment length=13; cross product magnitude=84.5; perpendicular distance=84.5/13=6.5.
  for (const [distance, count] of [
    [6.5, 1],
    [6.49, 0]
  ]) {
    const result = computeSpatialPredicateJoinOnCPU(
      createMultipartCPUGeometry(REVIEW_DISTANCE_THRESHOLD.left),
      createMultipartCPUGeometry(REVIEW_DISTANCE_THRESHOLD.right),
      {predicate: 'dwithin', distance, candidateCapacity: 1, pairCapacity: 1}
    );
    expect(result.status).toEqual({count, requiredCount: count, overflow: 0, candidateOverflow: 0});
    if (count) {
      expect(Array.from(result.leftRows)).toEqual([0]);
      expect(Array.from(result.rightRows)).toEqual([0]);
    }
  }
});

it('R3 preserves the exterior interval through a 2^-28 polygon hole', () => {
  const fixture = REVIEW_NARROW_HOLE;
  const matrix = runMultipartScalar(
    fixture.left,
    fixture.right,
    getSpatialRelateWGSL,
    'pairRelate(0u, 0u)'
  );
  expect(matrix >>> 31).toBe(0);
  expect(formatGPUSpatialRelate(matrix)).toBe('1010FF212');
  expect(
    runMultipartScalar(
      fixture.left,
      fixture.right,
      (left, right) => getSpatialPredicateWGSL(left, right, 'within', 0),
      'select(0u, 1u, pairMatches(0u, 0u))'
    )
  ).toBe(0);
});

it('R3 exact integer primitives preserve signs and crossing order', () => {
  const fixture = REVIEW_NARROW_HOLE;
  for (const [expression, expected] of [
    ['spatialSign(spatialInteger(2.0))', 1],
    ['spatialSign(spatialNegate(spatialInteger(2.0)))', -1],
    ['spatialSign(spatialSubtract(spatialInteger(1.0), spatialInteger(2.0)))', -1],
    ['spatialSign(spatialMultiply(spatialInteger(1.0), spatialInteger(2.0)))', 1],
    // n(n-2) - (n-1)^2 = -1, exactly, even though the f32 products round together.
    [
      'spatialSign(spatialSubtract(spatialMultiply(spatialInteger(16777215.0), spatialInteger(16777213.0)), spatialMultiply(spatialInteger(16777214.0), spatialInteger(16777214.0))))',
      -1
    ],
    [
      'spatialCrossingOrder(vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, -1.0), vec2f(0.0, 1.0), vec2f(0.000000003725290298461914, -1.0), vec2f(0.000000003725290298461914, 1.0))',
      -1
    ]
  ] as const) {
    let actual: number;
    try {
      actual = runMultipartScalar(
        fixture.left,
        fixture.right,
        getSpatialRelateWGSL,
        `u32(1i + ${expression})`
      );
    } catch (error) {
      throw new Error(expression, {cause: error});
    }
    expect(actual, expression).toBe(expected + 1);
  }
});

it('R4 equal translated one-ulp squares retain area cells', () => {
  const fixture = REVIEW_EQUAL_SQUARE;
  const matrix = runMultipartScalar(
    fixture.left,
    fixture.right,
    getSpatialRelateWGSL,
    'pairRelate(0u, 0u)'
  );
  expect(matrix >>> 31).toBe(0);
  expect(formatGPUSpatialRelate(matrix)).toBe('2FFF1FFF2');
});

it('R3/R4 CPU forms match the independent matrices and named predicates', () => {
  for (const [fixture, matrix, within, crosses, equals] of [
    [REVIEW_NARROW_HOLE, '1010FF212', 0, 1, 0],
    [REVIEW_EQUAL_SQUARE, '2FFF1FFF2', 1, 0, 1]
  ] as const) {
    const left = createMultipartCPUGeometry(fixture.left),
      right = createMultipartCPUGeometry(fixture.right);
    expect(formatGPUSpatialRelate(computeSpatialRelateOnCPU(left, right))).toBe(matrix);
    for (const [predicate, count] of [
      ['within', within],
      ['crosses', crosses],
      ['equals', equals]
    ] as const) {
      const result = computeSpatialPredicateJoinOnCPU(left, right, {
        predicate,
        relate: true,
        candidateCapacity: 1,
        pairCapacity: 1
      });
      expect(result.status).toEqual({
        count,
        requiredCount: count,
        overflow: 0,
        candidateOverflow: 0
      });
      if (count) {
        expect(Array.from(result.leftRows)).toEqual([0]);
        expect(Array.from(result.rightRows)).toEqual([0]);
        expect(formatGPUSpatialRelate(result.relate[0])).toBe(matrix);
      }
    }
  }
});

it('R5 both operation factories reject dwithin matrix output and float32 range overflow at construction', () => {
  const {left, right} = createMultipartOperationInputs(
    REVIEW_DISTANCE_THRESHOLD.left,
    REVIEW_DISTANCE_THRESHOLD.right
  );
  const output = {
    leftRows: 'leftRows',
    rightRows: 'rightRows',
    count: 'count',
    requiredCount: 'requiredCount',
    overflow: 'overflow',
    candidateOverflow: 'candidateOverflow'
  };
  for (const factory of [
    createSpatialPredicateJoinOperation,
    createCPUSpatialPredicateJoinOperation
  ]) {
    expect(() =>
      factory({
        id: 'review-invalid-matrix',
        left,
        right,
        predicate: 'dwithin',
        distance: 1,
        candidateCapacity: 1,
        pairCapacity: 1,
        output: {...output, relate: 'relate'}
      })
    ).toThrow(/review-invalid-matrix.*output\.relate/);
    expect(() =>
      factory({
        id: 'review-invalid-distance',
        left,
        right,
        predicate: 'dwithin',
        distance: 1e39,
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      })
    ).toThrow(/review-invalid-distance.*distance/);
  }
});

it('R6 multipart point/polygon planning matches 256-lane execution and declared bytes', () => {
  for (const candidateCapacity of [16_777_216, 1]) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 1024 ** 3, maxBufferSize: 1024 ** 3}
    });
    try {
      const graph = new GPUCommandGraph(device);
      const positions = (name: string, length: number) =>
        createTransientView(graph, name, 'float32x2', length);
      const integers = (name: string, length = 1) =>
        createTransientView(graph, name, 'uint32', length);
      const join = new GPUSpatialPredicateJoin({
        id: 'review-plan',
        left: {
          kind: 'points',
          positions: positions('points', 2),
          featureOffsets: integers('point-features', 2)
        },
        right: {
          kind: 'polygons',
          positions: positions('polygon', 4),
          featureOffsets: integers('features', 2),
          polygonOffsets: integers('polygons', 2),
          ringOffsets: integers('rings', 2)
        },
        predicate: 'intersects',
        engine: 'relate',
        candidateCapacity,
        output: {
          pairs: {
            leftRows: integers('left'),
            rightRows: integers('right'),
            status: {count: integers('count'), overflow: integers('overflow')}
          }
        }
      });
      if (candidateCapacity > 1) {
        expect(join.candidatePages.capacity).toBe(16_777_215);
        expect(join.candidatePages.pageCount).toBe(2);
      }
      join.getCommandNodes(graph);
      if (candidateCapacity === 1) {
        const buffers = (
          graph as unknown as {
            buffers: Map<string, {id: string; byteLength: number; transient: boolean}>;
          }
        ).buffers;
        const declared = [...buffers.values()]
          .filter(buffer => buffer.transient && buffer.id.startsWith('review-plan-'))
          .reduce((sum, buffer) => sum + buffer.byteLength, 0);
        // 192 scratch bytes plus the graph-owned one-word pair candidate-loss destination.
        expect(declared).toBe(196);
        expect(
          join.getCapacityPlan({candidateCount: 1, requiredCount: 1}).estimatedPeakTransientBytes
        ).toBe(196);
      }
    } finally {
      device.destroy();
    }
  }
});

it('R7 non-finite feature suppression is independent of part order', () => {
  for (const positions of [
    [0, 0, NaN, 0],
    [NaN, 0, 0, 0]
  ]) {
    const left = {kind: 'points' as const, positions, featureOffsets: [0, 2]};
    const right = {kind: 'points' as const, positions: [0, 0]};
    expect(runMultipartScalar(left, right, getSpatialRelateWGSL, 'pairRelate(0u, 0u)')).toBe(0);
    expect(() =>
      computeSpatialPredicateJoinOnCPU(
        createMultipartCPUGeometry(left),
        createMultipartCPUGeometry(right),
        {predicate: 'intersects', candidateCapacity: 1, pairCapacity: 1}
      )
    ).toThrow(/CPUFeatureJoin positions must be finite/);
  }
});

it('R8 a 240-vertex intersects operation skips the exact arrangement unless relate output is wired', async () => {
  const geometry = {
    kind: 'lines' as const,
    positions: Array.from({length: 240}, (_, row) => [row, row % 2]).flat(),
    lineOffsets: [0, 240]
  };
  const {left, right, inputs} = createMultipartOperationInputs(geometry, geometry);
  const arrangement = vi.spyOn(featureRelate, 'computeFeatureRelate');
  try {
    const operation = createCPUSpatialPredicateJoinOperation({
      left,
      right,
      predicate: 'intersects',
      candidateCapacity: 1,
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
    const vectors = Object.fromEntries(
      Object.entries(operation.inputs).map(([port, name]) => {
        const {format, values} = inputs[name];
        const length = values.length / (format === 'float32x2' ? 2 : 1);
        return [port, {format, length, chunks: [{values, length}], pack: () => values}];
      })
    );
    const result = (await operation.execute({inputs: vectors})) as Record<string, Uint32Array>;
    expect(Array.from(result['count'])).toEqual([1]);
    expect(Array.from(result['leftRows'])).toEqual([0]);
    expect(Array.from(result['rightRows'])).toEqual([0]);
    expect(arrangement).not.toHaveBeenCalled();
    for (const options of [
      {predicate: 'dwithin' as const, distance: 0},
      {predicate: 'intersects' as const, how: 'anti' as const}
    ]) {
      const short = computeSpatialPredicateJoinOnCPU(
        createMultipartCPUGeometry(geometry),
        createMultipartCPUGeometry(geometry),
        {...options, candidateCapacity: 1, pairCapacity: 1}
      );
      expect(short.status.count).toBe(options.predicate === 'dwithin' ? 1 : 0);
      expect(short.relate.length).toBe(0);
    }
    expect(arrangement).not.toHaveBeenCalled();
    const withMatrix = computeSpatialPredicateJoinOnCPU(
      createMultipartCPUGeometry(geometry),
      createMultipartCPUGeometry(geometry),
      {predicate: 'intersects', relate: true, candidateCapacity: 1, pairCapacity: 1}
    );
    expect(arrangement).toHaveBeenCalledTimes(1);
    expect(withMatrix.status.count).toBe(1);
    expect(formatGPUSpatialRelate(withMatrix.relate[0])).toBe('1FFF0FFF2');
  } finally {
    arrangement.mockRestore();
  }
});

it('R2 translated disjoint point has positive distance in scalar WGSL', () => {
  const fixture = REVIEW_TRANSLATED_DISTANCE;
  expect(
    runMultipartScalar(
      fixture.left,
      fixture.right,
      (left, right) => getSpatialPredicateWGSL(left, right, 'intersects', 0),
      'select(0u, 1u, pairMatches(0u, 0u))'
    )
  ).toBe(0);
  for (const [distance, count] of [
    [0, 0],
    [0.02, 0],
    [0.04, 1]
  ])
    expect(
      runMultipartScalar(
        fixture.left,
        fixture.right,
        (left, right) => getSpatialPredicateWGSL(left, right, 'dwithin', distance),
        'select(0u, 1u, pairMatches(0u, 0u))'
      ),
      `distance ${distance}`
    ).toBe(count);
});
