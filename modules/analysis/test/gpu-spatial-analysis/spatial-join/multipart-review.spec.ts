// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  type SpatialPredicateJoinOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createMultipartOperationInputs,
  type MultipartFixtureGeometry
} from './multipart-predicates-test-utils';
import {
  REVIEW_DISTANCE_THRESHOLD,
  REVIEW_TRANSLATED_DISTANCE,
  REVIEW_NARROW_HOLE,
  REVIEW_EQUAL_SQUARE
} from './multipart-review-fixtures';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';

async function checkReviewTopology(
  fixture: {left: MultipartFixtureGeometry; right: MultipartFixtureGeometry},
  predicate: SpatialPredicateJoinOperationProps['predicate'],
  engine: SpatialPredicateJoinOperationProps['engine'],
  count: number,
  matrix?: string,
  includeCPU = true
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, inputs} = createMultipartOperationInputs(fixture.left, fixture.right);
  for (const factory of includeCPU
    ? [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation]
    : [createSpatialPredicateJoinOperation]) {
    const output = {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateTruncated: 'candidateTruncated',
      ...(matrix ? {relate: 'relate'} : {})
    };
    const result = await runOperationGraph(device, {
      inputs,
      operations: [
        factory({left, right, predicate, engine, candidateCapacity: 1, pairCapacity: 1, output})
      ],
      outputs: Object.values(output)
    });
    for (const key of ['count', 'requiredCount'])
      expect(Array.from(result[key]), `${predicate} ${engine}`).toEqual([count]);
    for (const key of ['overflow', 'candidateOverflow', 'candidateTruncated'])
      expect(Array.from(result[key])).toEqual([0]);
    if (count) {
      for (const key of ['leftRows', 'rightRows']) expect(Array.from(result[key])).toEqual([0]);
      if (matrix) expect(formatGPUSpatialRelate(result['relate'][0])).toBe(matrix);
    }
  }
}

it('R3 both engines preserve a narrow polygon hole interval', async () => {
  for (const engine of ['fast', 'relate'] as const) {
    await checkReviewTopology(REVIEW_NARROW_HOLE, 'within', engine, 0);
    await checkReviewTopology(
      REVIEW_NARROW_HOLE,
      'crosses',
      engine === 'fast' ? 'auto' : engine,
      1
    );
    await checkReviewTopology(REVIEW_NARROW_HOLE, 'intersects', engine, 1, '1010FF212');
  }
}, 120_000);

it('R4 equal translated one-ulp squares retain equals, within and the area matrix', async () => {
  for (const engine of ['fast', 'relate'] as const) {
    await checkReviewTopology(REVIEW_EQUAL_SQUARE, 'within', engine, 1);
    await checkReviewTopology(
      REVIEW_EQUAL_SQUARE,
      'equals',
      engine === 'fast' ? 'auto' : engine,
      1,
      '2FFF1FFF2'
    );
    await checkReviewTopology(REVIEW_EQUAL_SQUARE, 'intersects', engine, 1, '2FFF1FFF2');
  }
}, 120_000);

it('R7 both engines suppress a non-finite multipart feature in either part order', async () => {
  // The documented invalid-feature rule, independent of implementation: no feature containing
  // a non-finite coordinate may match, even when another part is an exact hit.
  for (const positions of [
    [0, 0, NaN, 0],
    [NaN, 0, 0, 0]
  ])
    for (const engine of ['fast', 'relate'] as const)
      await checkReviewTopology(
        {
          left: {kind: 'points', positions, featureOffsets: [0, 2]},
          right: {kind: 'points', positions: [0, 0]}
        },
        'intersects',
        engine,
        0,
        undefined,
        false
      );
});

it('R7 prepared refinement suppresses an invalid feature even when a finite part is a candidate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const positions of [
    [0, 0, NaN, 0],
    [NaN, 0, 0, 0]
  ])
    for (const engine of ['fast', 'relate'] as const) {
      const graph = new GPUCommandGraph(device),
        buffers: Buffer[] = [];
      const inputBuffer = (values: Float32Array | Uint32Array) => {
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return buffer;
      };
      const left = {
        kind: 'points' as const,
        positions: importGraphBuffer(
          graph,
          'left-input',
          inputBuffer(Float32Array.of(0, 0)),
          'float32x2',
          1
        )
      };
      const right = {
        kind: 'points' as const,
        positions: importGraphBuffer(
          graph,
          'right-input',
          inputBuffer(Float32Array.from(positions)),
          'float32x2',
          2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'right-offsets',
          inputBuffer(Uint32Array.of(0, 2)),
          'uint32',
          2
        )
      };
      const output = (name: string) => {
        const buffer = createOutputBuffer(device, 1);
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
      };
      const leftRows = output('leftRows'),
        rightRows = output('rightRows'),
        count = output('count'),
        requiredCount = output('requiredCount'),
        overflow = output('overflow'),
        candidateOverflow = output('candidateOverflow'),
        truncated = output('truncated');
      const prepared = new GPUSpatialJoinPrepared({geometry: right});
      graph.add(prepared);
      graph.add(
        new GPUSpatialPredicateJoin({
          left,
          right,
          prepared,
          predicate: 'intersects',
          engine,
          candidateCapacity: 1,
          output: {
            pairs: {
              leftRows: leftRows.view,
              rightRows: rightRows.view,
              status: {
                count: count.view,
                requiredCount: requiredCount.view,
                overflow: overflow.view,
                candidateOverflow: candidateOverflow.view
              }
            },
            candidateTruncated: truncated.view
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const destination of [count, requiredCount, overflow, candidateOverflow, truncated])
          expect(await readUint32(destination.buffer, 1)).toEqual([0]);
      } finally {
        compiled.destroy();
        prepared.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
});

it('R1 both distance engines include the exact threshold in a later point part', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, inputs} = createMultipartOperationInputs(
    REVIEW_DISTANCE_THRESHOLD.left,
    REVIEW_DISTANCE_THRESHOLD.right
  );
  for (const engine of ['fast', 'relate'] as const)
    for (const [distance, count] of [
      [6.5, 1],
      [6.49, 0]
    ]) {
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateTruncated: 'candidateTruncated'
      };
      const operation = createSpatialPredicateJoinOperation({
        left,
        right,
        predicate: 'dwithin',
        distance,
        engine,
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      });
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(output)
      });
      for (const key of ['count', 'requiredCount'])
        expect(Array.from(result[key]), `${engine} ${distance}`).toEqual([count]);
      for (const key of ['overflow', 'candidateOverflow', 'candidateTruncated'])
        expect(Array.from(result[key])).toEqual([0]);
      if (count)
        for (const key of ['leftRows', 'rightRows']) expect(Array.from(result[key])).toEqual([0]);
    }
});

it('R2 both distance engines keep translated disjoint geometry outside 0 and 0.02', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, inputs} = createMultipartOperationInputs(
    REVIEW_TRANSLATED_DISTANCE.left,
    REVIEW_TRANSLATED_DISTANCE.right
  );
  // Shapely distance=0.03031695312954162; each threshold is far from float32 error.
  for (const engine of ['fast', 'relate'] as const)
    for (const [distance, count] of [
      [0, 0],
      [0.02, 0],
      [0.04, 1]
    ]) {
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      };
      const result = await runOperationGraph(device, {
        inputs,
        operations: [
          createSpatialPredicateJoinOperation({
            left,
            right,
            predicate: 'dwithin',
            engine,
            distance,
            candidateCapacity: 1,
            pairCapacity: 1,
            output
          })
        ],
        outputs: Object.values(output)
      });
      for (const key of ['count', 'requiredCount'])
        expect(Array.from(result[key]), `${engine} ${distance}`).toEqual([count]);
      for (const key of ['overflow', 'candidateOverflow'])
        expect(Array.from(result[key])).toEqual([0]);
      if (count)
        for (const key of ['leftRows', 'rightRows']) expect(Array.from(result[key])).toEqual([0]);
    }
});
