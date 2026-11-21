// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePolygonTriangulationOnCPU} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {computeConstrainedDelaunayTriangulationOnCPU} from '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';

for (const compute of [
  computePolygonTriangulationOnCPU,
  computeConstrainedDelaunayTriangulationOnCPU
]) {
  it(`${compute.name} preserves canonical empty polygon counts and terminal offsets`, () => {
    // One empty MultiPolygon has zero polygons; one empty Polygon has one empty polygon.
    for (const polygonCount of [0, 1]) {
      const result = compute(
        {
          positions: new Float32Array(),
          ringOffsets: Uint32Array.of(0),
          polygonOffsets: new Uint32Array(polygonCount + 1),
          featureOffsets: Uint32Array.of(0, polygonCount)
        },
        {indexCapacity: 0, storage: 'compact'}
      );
      expect(Array.from(result.indices)).toEqual([]);
      expect(Array.from(result.triangleCount)).toEqual(new Array(polygonCount).fill(0));
      expect(Array.from(result.triangleOffsets)).toEqual(new Array(polygonCount + 1).fill(0));
    }
  });

  for (const [name, offsets] of [
    ['ringOffsets', [1, 4]],
    ['ringOffsets', [0, 3]],
    ['ringOffsets', [0, 999]],
    ['polygonOffsets', [0, 0]],
    ['featureOffsets', [0, 0]]
  ] as const) {
    it(`${compute.name} rejects compact ${name} = ${offsets.join(',')}`, () => {
      const input = {
        positions: Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4),
        ringOffsets: Uint32Array.of(0, 4),
        polygonOffsets: Uint32Array.of(0, 1),
        featureOffsets: Uint32Array.of(0, 1),
        [name]: Uint32Array.from(offsets)
      };
      expect(() => compute(input, {indexCapacity: 6, storage: 'compact'})).toThrow(
        new RegExp(name)
      );
    });
  }
}
