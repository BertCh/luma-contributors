// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {OFFSET_REVIEW_CASES} from './offset-review-fixtures';

it.each(OFFSET_REVIEW_CASES)('GPU offset review: $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = fixture.raw.length / 2;
  const result = await runOperationGraph(device, {
    inputs: {
      original: {format: 'float32x2', values: Float32Array.from(fixture.original)},
      raw: {format: 'float32x2', values: Float32Array.from(fixture.raw)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, count)}
    },
    operations: [
      new GPUOffsetCurveComponentsOperation({
        lineCount: 1,
        original: {positions: 'original', lineOffsets: 'offsets'},
        raw: {positions: 'raw', lineOffsets: 'offsets'},
        distance: fixture.distance,
        tolerance: 0,
        intersectionCapacity: 8,
        segmentCapacity: 32,
        vertexCapacity: 32,
        componentCapacity: 8,
        output: {
          positions: 'positions',
          components: 'components',
          sourceEdges: 'sourceEdges',
          status: 'status'
        }
      })
    ],
    outputs: ['positions', 'components', 'sourceEdges', 'status']
  });
  const vertices = fixture.expected.length / 2;
  expect(Array.from(result['status'])).toEqual([
    Number(vertices > 0),
    vertices,
    Number(vertices > 0),
    vertices,
    0,
    0,
    0,
    0
  ]);
  expect(Array.from(result['positions'].slice(0, fixture.expected.length))).toEqual(
    fixture.expected
  );
});
