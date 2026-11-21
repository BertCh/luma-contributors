// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUOffsetCurveComponentsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  expectCompleteObliqueOffset,
  PROJECTED_OBLIQUE_OFFSET_CASES
} from './offset-curve-components-test-utils';

const cases = [
  ...PROJECTED_OBLIQUE_OFFSET_CASES.map(fixture => ({...fixture, distance: 5})),
  ...PROJECTED_OBLIQUE_OFFSET_CASES.map(fixture => ({
    ...fixture,
    name: `reversed ${fixture.name}`,
    distance: -5,
    originalPositions: Float32Array.of(
      fixture.originalPositions[2],
      fixture.originalPositions[3],
      fixture.originalPositions[0],
      fixture.originalPositions[1]
    ),
    expectedPositions: Float32Array.of(
      fixture.expectedPositions[2],
      fixture.expectedPositions[3],
      fixture.expectedPositions[0],
      fixture.expectedPositions[1]
    )
  }))
];

it.each(cases)('GPU retains the complete oblique offset: $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: {
      originalPositions: {format: 'float32x2', values: fixture.originalPositions},
      originalOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)},
      rawPositions: {format: 'float32x2', values: fixture.expectedPositions},
      rawOffsets: {format: 'uint32', values: Uint32Array.of(0, 2)}
    },
    operations: [
      new GPUOffsetCurveComponentsOperation({
        id: 'projected-oblique-offset',
        lineCount: 1,
        distance: fixture.distance,
        tolerance: fixture.tolerance,
        original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
        raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
        intersectionCapacity: 16,
        segmentCapacity: 16,
        vertexCapacity: 8,
        componentCapacity: 4,
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
  expectCompleteObliqueOffset(result, fixture.expectedPositions);
});
