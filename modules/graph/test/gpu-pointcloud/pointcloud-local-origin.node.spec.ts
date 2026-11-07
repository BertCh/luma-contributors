// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {
  GPUPointCloudChange,
  GPUPointCloudICP,
  getPointCloudAbsoluteTransform,
  getPointCloudLocalOrigin,
  getPointCloudLocalTransform,
  makePointCloudLocalPositions
} from '@luma-contributors/graph/gpu-pointcloud';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const f = Math.fround;

describe('point-cloud local origin', () => {
  const positions = [1000000.123, 4600456.911, 25.5, 1000100.456, 4600500.321, 30.25];

  it('keeps millimetre precision at 1e6 m only when positions are recentered', () => {
    const origin = getPointCloudLocalOrigin(positions);
    const local = makePointCloudLocalPositions(positions, origin);
    for (let index = 0; index < positions.length; index++) {
      expect(Math.abs(local[index] + origin[index % 3] - positions[index])).toBeLessThan(1e-3);
      // Absolute float32 is quantized to the ulp (0.0625 m near 1e6, 0.5 m near 4.6e6).
    }
    expect(Math.abs(f(positions[0]) - positions[0])).toBeGreaterThan(1e-3);
    expect(Math.abs(f(positions[1]) - positions[1])).toBeGreaterThan(1e-3);
  });

  it('round-trips rigid transforms between absolute and local frames', () => {
    const origin: [number, number, number] = [1000050, 4600478, 27];
    const angle = 0.3;
    const absolute = [
      Math.cos(angle),
      Math.sin(angle),
      0,
      0,
      -Math.sin(angle),
      Math.cos(angle),
      0,
      0,
      0,
      0,
      1,
      0,
      12.5,
      -3.25,
      0.5,
      1
    ];
    const local = getPointCloudLocalTransform(absolute, origin);
    // A point maps identically in both frames.
    const point = [1000060, 4600470, 28];
    const apply = (matrix: number[], value: number[]) =>
      [0, 1, 2].map(
        row =>
          matrix[row] * value[0] +
          matrix[4 + row] * value[1] +
          matrix[8 + row] * value[2] +
          matrix[12 + row]
      );
    const absoluteResult = apply(absolute, point);
    const localResult = apply(
      local,
      point.map((value, axis) => value - origin[axis])
    ).map((value, axis) => value + origin[axis]);
    for (let axis = 0; axis < 3; axis++)
      expect(localResult[axis]).toBeCloseTo(absoluteResult[axis], 6);
    const restored = getPointCloudAbsoluteTransform(local, origin);
    for (let index = 0; index < 16; index++)
      expect(restored[index]).toBeCloseTo(absolute[index], 6);
  });

  it('applies the origin to ICP initial transforms and exposes absolute readback', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `origin-${serial++}`, format, length);
    const origin: [number, number, number] = [1e6, 4.6e6, 0];
    const icp = new GPUPointCloudICP({
      source: view('float32x3', 4),
      target: view('float32x3', 4),
      maximumDistance: 1,
      origin,
      initialTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1],
      iterations: 2,
      transform: view('float32', 16),
      history: view('float32x3', 2),
      inlierMask: view('uint32', 4),
      status: view('uint32', 1)
    });
    // Pure translation is origin independent for identity rotation.
    expect(icp.getAbsoluteTransform([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])[12]).toBe(2);
  });

  it('caps brute-force ICP and change work and points to the grid option', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `cap-${serial++}`, format, length);
    const large = 70000;
    expect(
      () =>
        new GPUPointCloudICP({
          source: view('float32x3', large),
          target: view('float32x3', large),
          maximumDistance: 1,
          iterations: 1,
          transform: view('float32', 16),
          history: view('float32x3', 1),
          inlierMask: view('uint32', large),
          status: view('uint32', 1)
        })
    ).toThrow();
    expect(
      () =>
        new GPUPointCloudChange({
          reference: view('float32x3', large),
          comparison: view('float32x3', large),
          cores: view('float32x3', large),
          normals: view('float32x3', large),
          alignment: 'already-aligned',
          radius: 1,
          halfDepth: 1,
          changes: view('float32x3', large),
          roughness: view('float32x3', large),
          counts: view('uint32', large * 2),
          status: view('uint32', large)
        })
    ).toThrow();
  });
});
