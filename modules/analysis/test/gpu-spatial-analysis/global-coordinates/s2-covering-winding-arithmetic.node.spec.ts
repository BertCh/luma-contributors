// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  Float32CoveringPredicate,
  createFineCoveringRectangle
} from './s2-covering-float32-predicate';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {computeS2PolygonCoveringOnCPU} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function getSurface(value: number): number {
  return value >= 0.5 ? (4 * value * value - 1) / 3 : (1 - 4 * (1 - value) ** 2) / 3;
}

// Analytic gnomonic bounds: only grid square [23000,23001] x [21000,21001] fits
// strictly in [22999.6,23001.4] x [20999.6,21001.4]. Independent Hilbert encoding
// gives 0x1123fdc040000000. This arithmetic probe is separate from device qualification.
it('retains every interior level-15 corner under direct and reciprocal f32 normalization (W3-01)', () => {
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
  const getPosition = (first: number, second: number) => {
    const raw = [1, getSurface(first / 32768), getSurface(second / 32768)];
    return raw.map(value => rounded(value / Math.hypot(...raw)));
  };
  const vertices = [
    [22999.6, 20999.6],
    [23001.4, 20999.6],
    [23001.4, 21001.4],
    [22999.6, 21001.4]
  ].map(([first, second]) => getPosition(first, second));
  const result = computeS2PolygonCoveringOnCPU(
    {positions: Float32Array.from(vertices.flat()), ringOffsets: Uint32Array.of(0, 4)},
    {minimumLevel: 15, maximumLevel: 15, mode: 'interior', capacity: 4}
  );
  expect(Array.from(result.counts)).toEqual([1, 1]);
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.cells)).toEqual([0x40000000, 0x1123fdc0, 0, 0, 0, 0, 0, 0]);
  const oldInsideCounts: number[] = [];
  for (const reciprocal of [false, true]) {
    const divide = (first: number, second: number) =>
      reciprocal ? rounded(first * rounded(1 / second)) : rounded(first / second);
    let oldInsideCount = 0;
    for (const [first, second] of [
      [23000, 21000],
      [23001, 21000],
      [23001, 21001],
      [23000, 21001]
    ]) {
      const shaderSurface = (coordinate: number) => {
        const surface = rounded(coordinate / 32768);
        return divide(rounded(rounded(rounded(4 * surface) * surface) - 1), 3);
      };
      const unnormalized = [1, shaderSurface(first), shaderSurface(second)];
      const raw = unnormalized.map(value =>
        divide(value, rounded(Math.sqrt(dot(unnormalized, unnormalized))))
      );
      const point = raw.map(value => divide(value, rounded(Math.sqrt(dot(raw, raw)))));
      let winding = 0;
      let oldWinding = 0;
      for (let edge = 0; edge < 4; edge++) {
        const start = vertices[edge];
        const end = vertices[(edge + 1) % 4];
        oldWinding = rounded(
          oldWinding +
            rounded(
              Math.atan2(
                dot(raw, cross(start, end)),
                rounded(dot(start, end) - rounded(dot(raw, start) * dot(raw, end)))
              )
            )
        );
        const tangent = (vertex: number[]) =>
          cross(
            point,
            vertex.map((value, axis) =>
              rounded(value - (dot(point, vertex) >= 0 ? point[axis] : -point[axis]))
            )
          );
        const startTangent = tangent(start);
        const endTangent = tangent(end);
        winding = rounded(
          winding +
            rounded(
              Math.atan2(dot(point, cross(startTangent, endTangent)), dot(startTangent, endTangent))
            )
        );
      }
      if (oldWinding > Math.PI) oldInsideCount++;
      expect(Number.isFinite(winding)).toBe(true);
      expect(Math.abs(winding - 2 * Math.PI)).toBeLessThan(1e-5);
    }
    oldInsideCounts.push(oldInsideCount);
  }
  expect(oldInsideCounts).toEqual([0, 0]);
});

it('uses local tangent winding on both declared traversal routes (W3-01)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUS2PolygonCovering({
        positions: createTransientView(graph, 'positions', 'float32x3', 4),
        ringOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        maximumLevel: 15,
        capacity: 4,
        maximumVisitedCells: 512,
        refinement,
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 4),
          counts: createTransientView(graph, 'counts', 'uint32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      });
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const predicateSources = sources.filter(kernel =>
        kernel.source.includes('fn coverRingContribution(')
      );
      expect(predicateSources.length).toBeGreaterThan(0);
      for (const kernel of predicateSources) {
        expect(kernel.source).toContain('cross(direction,firstDifference)');
        expect(kernel.source).not.toContain('dot(first,second)-dot(point,first)*dot(point,second)');
        expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('pins full fine-cell classification and simple-ring validation under both f32 division routes (review B1/B2)', () => {
  // Independent gnomonic rectangles contain precisely their central grid square. Keys are from
  // xy-to-Hilbert encoding; neither classification nor validation supplies these expectations.
  const scenes = [
    {first: 23000, second: 21000, key: 0x1123fdc040000000n},
    {first: 24503, second: 19621, key: 0x10d2c5c240000000n},
    {first: 19170, second: 31311, key: 0x12c46378c0000000n}
  ];
  for (const reciprocal of [false, true])
    for (const scene of scenes) {
      const stable = new Float32CoveringPredicate(reciprocal, true, true);
      const unstable = new Float32CoveringPredicate(reciprocal, false);
      const ring = createFineCoveringRectangle(scene.first, scene.second);
      const corners = stable.cellVertices(scene.first, scene.second);
      expect(stable.classify(ring, corners)).toEqual([true, true]);
      expect(stable.validate(ring)).toBe(true);
      expect(stable.validate([ring[0], ring[2], ring[1], ring[3]])).toBe(false);
      if (scene.first === 19170) expect(unstable.validate(ring)).toBe(false);
      else expect(unstable.classify(ring, corners)[1]).toBe(false);
      const result = computeS2PolygonCoveringOnCPU(
        {positions: Float32Array.from(ring.flat()), ringOffsets: Uint32Array.of(0, 4)},
        {minimumLevel: 15, maximumLevel: 15, mode: 'interior', capacity: 4}
      );
      expect(Array.from(result.status)).toEqual([0]);
      expect(Array.from(result.counts)).toEqual([1, 1]);
      expect(BigInt(result.cells[0]) | (BigInt(result.cells[1]) << 32n)).toBe(scene.key);
    }
});

it('uses local edge normals in classification and both ring validators (review B1/B2)', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const refinement of ['serial', 'level-synchronous'] as const) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUS2PolygonCovering({
        positions: createTransientView(graph, 'positions', 'float32x3', 4),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
        maximumLevel: 15,
        capacity: 4,
        maximumVisitedCells: 512,
        refinement,
        output: {
          cells: createTransientView(graph, 'cells', 'uint32x2', 4),
          counts: createTransientView(graph, 'counts', 'uint32x2', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      });
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const geometry = sources.filter(kernel => kernel.source.includes('coverOnArc('));
      expect(geometry.length).toBeGreaterThan(0);
      for (const kernel of geometry) {
        expect(kernel.source.includes('fn coverEdgeCross(')).toBe(true);
        const covering = kernel.source.slice(kernel.source.indexOf('fn coverPosition('));
        expect(/cross\(first,\s*second\)/.test(covering)).toBe(false);
        expect(/cross\(third,\s*fourth\)/.test(covering)).toBe(false);
      }
    }
  } finally {
    device.destroy();
  }
});
