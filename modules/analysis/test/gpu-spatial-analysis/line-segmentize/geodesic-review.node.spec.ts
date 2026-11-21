// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {createArcEmitNode} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createFlatLines,
  segmentizeLines,
  tessellateGreatCircleArcs
} from './line-segmentize-oracle';

it('arc emission unwraps the target itself instead of reconstructing it from the source (B3)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const node = createArcEmitNode(graph, {
    id: 'arc-emit',
    operation: 'GPUGreatCircleArcs',
    radius: 1,
    sources: createTransientView(graph, 'sources', 'float32x2', 1),
    targets: createTransientView(graph, 'targets', 'float32x2', 1),
    counts: createTransientView(graph, 'counts', 'uint32', 1),
    starts: createTransientView(graph, 'starts', 'uint32', 1),
    outputPositions: createTransientView(graph, 'positions', 'float32x2', 5)
  });
  const source = getGPUCommandNodeKernelSources([node])[0].source;
  expect(source).toContain('geodesicUnwrapTarget(source, destination)');
  expect(source).not.toContain('source.x + geodesicLongitudeDelta(source.x, destination.x)');
  expect(source).toContain('destination.x - 360.0 * turn');
  expect(() => new WgslReflect(source)).not.toThrow();
});

for (const route of ['arcs', 'segmentize'] as const) {
  it(`${route} reference applies the antipodal interior contract without changing endpoints (S1, S2)`, () => {
    for (const [start, end, invalid] of [
      [[-30, 20], [150, -20], true],
      [[0, 0], [Math.fround(179.99998), 0], true],
      [[0, 0], [Math.fround(179.9998), 0], false]
    ] as const) {
      const result =
        route === 'arcs'
          ? tessellateGreatCircleArcs([start], [end], {
              maximumSegmentLength: 0,
              minimumSegments: 4,
              maximumSegmentCount: 4,
              radius: 1
            })
          : segmentizeLines(createFlatLines([[Array.from(start), Array.from(end)]]), {
              maximumSegmentLength: Math.PI / 3.9,
              maximumPieces: 4,
              spherical: true
            });
      expect(result.positions).toHaveLength(5);
      expect(result.positions[0]).toEqual(start);
      // Both representations of the half-turn are allowed for the unwrapped terminal longitude.
      expect(result.positions[4][1]).toBe(end[1]);
      expect(Math.abs(result.positions[4][0] - end[0]) % 360).toBe(0);
      for (const vertex of result.positions.slice(1, 4)) {
        if (invalid) {
          expect(vertex[0]).toBeNaN();
          expect(vertex[1]).toBeNaN();
        } else {
          expect(vertex.every(Number.isFinite)).toBe(true);
        }
      }
    }
  });
}
