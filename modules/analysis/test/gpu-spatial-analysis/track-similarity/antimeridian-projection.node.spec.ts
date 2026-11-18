// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl';
import {getNetworkMetricWGSL} from '../../../src/gpu-network/network-spatial-context';
import {getLineGeodesicMetricWGSL} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-kernels';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

// Equatorial distances in degrees: all expectations follow from the wrapped arc endpoints.
const CASES = [
  {point: [0, 0], start: [100, 0], end: [-100, 0], fraction: 0, distance: 100},
  {point: [0, 0], start: [-100, 0], end: [100, 0], fraction: 0, distance: 100},
  {point: [-60, 0], start: [100, 0], end: [-100, 0], fraction: 1, distance: 40},
  {point: [-60, 0], start: [-100, 0], end: [100, 0], fraction: 0, distance: 40},
  {point: [180, 0], start: [100, 0], end: [-100, 0], fraction: 0.5, distance: 0},
  {
    point: [0, 0],
    start: [179.99, 0],
    end: [-179.99, 0],
    fraction: 0,
    distance: Math.fround(179.99)
  },
  {point: [180, 0], start: [179.99, 0], end: [-179.99, 0], fraction: 0.5, distance: 0},
  {point: [100, 0], start: [100, 0], end: [100, 0], fraction: 0, distance: 0},
  {point: [0, 1], start: [-0.0002, 0], end: [0.0002, 0], fraction: 0.5, distance: 1},
  {point: [0, 1], start: [0.0002, 0], end: [-0.0002, 0], fraction: 0.5, distance: 1}
];

for (const variant of ['shared', 'network', 'line', 'shared-rounded-angle'] as const) {
  it(`projects one wrapped arc and preserves endpoint/interior distances for ${variant} callers`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const points = createTransientView(graph, 'points', 'float32x2', CASES.length);
      const starts = createTransientView(graph, 'starts', 'float32x2', CASES.length);
      const ends = createTransientView(graph, 'ends', 'float32x2', CASES.length);
      const output = createTransientView(graph, 'output', 'float32x2', CASES.length);
      const midpointDistances = createTransientView(graph, 'midpoints', 'float32', CASES.length);
      const radius = 180 / Math.PI;
      const baseDeclarations =
        variant === 'network'
          ? getNetworkMetricWGSL(radius)
          : variant === 'line'
            ? getLineGeodesicMetricWGSL(radius)
            : GEODESIC_WGSL;
      // Metal has confirmed midpoint loss despite the strict angular-distance comparison.
      // Model a small relative difference between those angle algorithms deterministically;
      // candidate ordering must not depend on converting the common haversine quantity twice.
      const declarations =
        variant === 'shared-rounded-angle'
          ? baseDeclarations.replace(
              'return radius * geodesicCentralAngle(a, b);',
              'let angle = geodesicCentralAngle(a, b); return radius * select(angle, angle * 0.9999995, angle < 0.02);'
            )
          : baseDeclarations;
      if (variant === 'shared-rounded-angle') expect(declarations).not.toBe(baseDeclarations);
      const fraction = variant === 'network' ? 'networkSegmentFraction' : 'geodesicSegmentFraction';
      const distance =
        variant === 'network'
          ? 'networkSegmentDistance(point, start, end, fraction)'
          : variant === 'line'
            ? 'lineGeodesicSegmentDistance(point, start, end)'
            : `geodesicSegmentDistance(point, start, end, fraction, ${radius})`;
      const node = createWGSLKernelNode(graph, {
        id: `projection-${variant}`,
        operation: 'TestAntimeridianProjection',
        invocationCount: CASES.length,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'starts', view: starts, type: 'f32', access: 'read'},
          {name: 'ends', view: ends, type: 'f32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'},
          {name: 'midpoints', view: midpointDistances, type: 'f32', access: 'read_write'}
        ],
        declarations,
        body: `let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
let start = vec2<f32>(starts[startsOffset + 2u * index], starts[startsOffset + 2u * index + 1u]);
let end = vec2<f32>(ends[endsOffset + 2u * index], ends[endsOffset + 2u * index + 1u]);
let fraction = ${fraction}(point, start, end);
output[outputOffset + 2u * index] = fraction;
output[outputOffset + 2u * index + 1u] = ${distance};
midpoints[midpointsOffset + index] = geodesicSegmentDistance(point, start, end, 0.5, ${radius});`
      });
      const memory = createCPUGraphMemory();
      memory.write(points, Float32Array.from(CASES.flatMap(testCase => testCase.point)));
      memory.write(starts, Float32Array.from(CASES.flatMap(testCase => testCase.start)));
      memory.write(ends, Float32Array.from(CASES.flatMap(testCase => testCase.end)));
      executeCommandNodesOnCPU([node], memory);
      const actual = memory.readFloat32(output);
      // A caller-supplied midpoint is still evaluated at that fraction, not clamped to an
      // endpoint; network time interpolation and simplification's time-ratio metric need this.
      expect(memory.readFloat32(midpointDistances)[0]).toBeCloseTo(180, 4);
      CASES.forEach((testCase, index) => {
        expect(actual[index * 2], `case ${index} fraction`).toBe(testCase.fraction);
        expect(Number.isFinite(actual[index * 2 + 1]), `case ${index} finite distance`).toBe(true);
        expect(
          Math.abs(actual[index * 2 + 1] - testCase.distance),
          `case ${index} distance`
        ).toBeLessThan(3e-5);
      });
    } finally {
      device.destroy();
    }
  });
}
