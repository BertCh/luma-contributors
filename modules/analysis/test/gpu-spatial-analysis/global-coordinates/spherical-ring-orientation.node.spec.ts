// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {OFF_AXIS_CLOCKWISE_TRIANGLE} from './off-axis-ring-fixture';

it('keeps the packed off-axis ring orientation under direct and reciprocal f32 division', () => {
  // Arithmetic probe only, not device evidence. Independent Decimal oracle is in the fixture.
  const rounded = Math.fround;
  const dot = (first: number[], second: number[]) =>
    rounded(
      rounded(rounded(first[0] * second[0]) + rounded(first[1] * second[1])) +
        rounded(first[2] * second[2])
    );
  const cross = (first: number[], second: number[]) => [
    rounded(rounded(first[1] * second[2]) - rounded(first[2] * second[1])),
    rounded(rounded(first[2] * second[0]) - rounded(first[0] * second[2])),
    rounded(rounded(first[0] * second[1]) - rounded(first[1] * second[0]))
  ];
  for (const reciprocal of [false, true])
    for (const reversed of [false, true]) {
      const points = reversed
        ? [...OFF_AXIS_CLOCKWISE_TRIANGLE].reverse()
        : OFF_AXIS_CLOCKWISE_TRIANGLE;
      const sum = points.reduce(
        (first, second) => first.map((value, axis) => rounded(value + second[axis])),
        [0, 0, 0]
      );
      const pole = sum.map(value =>
        rounded(value * rounded(1 / rounded(Math.sqrt(dot(sum, sum)))))
      );
      const divide = (numerator: number, denominator: number) =>
        reciprocal
          ? rounded(numerator * rounded(1 / denominator))
          : rounded(numerator / denominator);
      const anchor = points[0];
      const project = (point: number[]) => {
        const difference = point.map((value, axis) => rounded(value - anchor[axis]));
        const correction = divide(dot(pole, difference), dot(pole, anchor));
        return difference.map((value, axis) =>
          divide(rounded(value - rounded(anchor[axis] * correction)), dot(pole, point))
        );
      };
      const signedArea = dot(pole, cross(project(points[1]), project(points[2])));
      expect(Number.isFinite(signedArea)).toBe(true);
      expect(Math.abs(signedArea - (reversed ? 1 : -1) * 3.827013771774872e-10)).toBeLessThan(
        2e-16
      );
      for (const antipodal of [false, true])
        for (const chartOffset of [0, -10]) {
          const raw = [1, 1 + chartOffset * 1e-6, 1];
          const rawQuery = raw.map(value =>
            rounded(((antipodal ? -1 : 1) * value) / Math.hypot(...raw))
          );
          const point = rawQuery.map(value =>
            divide(value, rounded(Math.sqrt(dot(rawQuery, rawQuery))))
          );
          const tangent = (vertex: number[]) => {
            const sign = dot(point, vertex) >= 0 ? 1 : -1;
            return cross(
              point,
              vertex.map((value, axis) => rounded(value - sign * point[axis]))
            );
          };
          const winding = points.reduce((total, first, row) => {
            const firstTangent = tangent(first);
            const secondTangent = tangent(points[(row + 1) % points.length]);
            return rounded(
              total +
                rounded(
                  Math.atan2(
                    dot(point, cross(firstTangent, secondTangent)),
                    dot(firstTangent, secondTangent)
                  )
                )
            );
          }, 0);
          expect(Number.isFinite(winding)).toBe(true);
          expect(
            Math.abs(winding - (reversed ? 1 : -1) * (antipodal ? -1 : 1) * 2 * Math.PI)
          ).toBeLessThan(1e-5);
        }
    }
});

it('uses local projected differences for off-axis polygon orientation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUSphericalPolygonContains({
      positions: createTransientView(graph, 'positions', 'float32x3', 3),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      queries: createTransientView(graph, 'queries', 'float32x3', 2),
      polygonIndices: createTransientView(graph, 'indices', 'uint32', 2),
      output: {
        contains: createTransientView(graph, 'contains', 'uint32', 2),
        status: createTransientView(graph, 'status', 'uint32', 2)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels[0].source).toContain('sphericalRingSignedArea(begin, end, center)');
    expect(kernels[kernels.length - 1].source).toContain('cross(point, firstDifference)');
    for (const kernel of kernels)
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});
