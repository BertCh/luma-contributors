// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGeometryCleanup,
  getGPUGeometryCleanupParameterValues
} from '../../../src/gpu-spatial-analysis/geometry-edit/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Three distinct vertices form points/a line; the valid GeoArrow polygon adds
// the closing (0,0) row. Without snapping all 3/3/4 coordinate rows survive.
// Snapping to grid 10 makes every vertex (0,0): points retain all three,
// a line retains both endpoints (even when coincident), and the polygon ring
// collapses (fewer than three vertices). geometry-cleanup-status-oracles.oracle.py
// verifies the line with Shapely. Tolerances 0 and 0.5 select the parallel and
// sequential repeat-removal paths; both must publish these same cardinalities.
for (const geometryType of ['points', 'lines', 'polygons'] as const) {
  for (const capacity of [1, 4]) {
    it(`cleanup status has exact vertex and collapse cardinalities (${geometryType}, capacity ${capacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const positions = input(
        'positions',
        'float32x2',
        geometryType === 'polygons'
          ? Float32Array.of(0, 0, 1, 0, 0, 1, 0, 0)
          : Float32Array.of(0, 0, 1, 0, 0, 1)
      );
      const parameters = input('parameters', 'float32', getGPUGeometryCleanupParameterValues());
      const status = {
        count: output('count', 'uint32', 1),
        requiredCount: output('required', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1)
      };
      const collapsedCount = output('collapsed', 'uint32', 1);
      const destination = output('points', 'float32x2', capacity);
      const offsets = input(
        'offsets',
        'uint32',
        Uint32Array.of(0, geometryType === 'polygons' ? 4 : 3)
      );
      const outputOffsets = output('out-offsets', 'uint32', 2);
      graph.add(
        new GPUGeometryCleanup(
          geometryType === 'points'
            ? {positions, parameters, geometryType, output: {positions: destination, status}}
            : geometryType === 'lines'
              ? {
                  positions,
                  parameters,
                  geometryType,
                  lineOffsets: offsets,
                  output: {positions: destination, lineOffsets: outputOffsets, status}
                }
              : {
                  positions,
                  parameters,
                  geometryType,
                  ringOffsets: offsets,
                  polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
                  featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
                  output: {
                    positions: destination,
                    ringOffsets: outputOffsets,
                    status: {...status, collapsedCount}
                  }
                }
        )
      );
      const compiled = graph.compile();
      try {
        for (const [gridSize, tolerance] of [
          [0, 0],
          [10, 0],
          [0, 0.5],
          [10, 0.5],
          [0, 0]
        ]) {
          fixture.writeInput(
            parameters,
            getGPUGeometryCleanupParameterValues({gridSize, tolerance})
          );
          const actual = await fixture.submitTwice(compiled, () =>
            fixture.readRows([
              status.count,
              status.requiredCount,
              status.overflow,
              ...(geometryType === 'polygons' ? [collapsedCount] : [])
            ])
          );
          expect(actual).toEqual(
            gridSize === 0 || geometryType === 'points'
              ? geometryType === 'polygons'
                ? capacity === 1
                  ? [1, 4, 1, 0]
                  : [4, 4, 0, 0]
                : capacity === 1
                  ? [1, 3, 1]
                  : [3, 3, 0]
              : geometryType === 'lines'
                ? capacity === 1
                  ? [1, 2, 1]
                  : [2, 2, 0]
                : [0, 0, 0, 1]
          );
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
