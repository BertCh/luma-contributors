// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudICP,
  GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// Point-to-plane ICP on a room corner (three mutually orthogonal faces), which constrains all six
// rigid parameters. The expected transform is the one the scene was built with, in float64; the
// kernel's own arithmetic is not repeated here. Before the moments were taken about the centroid
// of the matched source points and the pivot test was scaled by the diagonal, all three scenes
// returned GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE (math review finding core-numerics-01).

type Scene = {
  name: string;
  /** Centre of the room corner. */
  center: [number, number, number];
  /** Edge length of each square face. */
  extent: number;
};

const SCENES: Scene[] = [
  {name: 'a 10 m room 500 m from the origin', center: [500, 300, 0], extent: 10},
  {name: 'a centred room with a 10 km extent', center: [0, 0, 0], extent: 10000},
  {name: 'a centred 30 m room in millimetres', center: [0, 0, 0], extent: 30000}
];

const POINT_COUNT = 3000;
/** Rotation about the vertical axis through the room corner, radians. */
const ROTATION_ANGLE = 0.002;
/** Translation as a fraction of the extent: 2 cm, -1 cm and 1.5 cm for the 10 m room. */
const TRANSLATION_FRACTION = [0.002, -0.001, 0.0015];

function createUniformGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return state / 4294967296;
  };
}

for (const scene of SCENES) {
  it(`point-to-plane ICP solves ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const random = createUniformGenerator(12345);
      const cosine = Math.cos(ROTATION_ANGLE);
      const sine = Math.sin(ROTATION_ANGLE);
      const translation = TRANSLATION_FRACTION.map(fraction => fraction * scene.extent);
      const targetValues: number[] = [];
      const normalValues: number[] = [];
      const sourceValues: number[] = [];
      for (let point = 0; point < POINT_COUNT; point++) {
        const face = point % 3;
        const first = (random() - 0.5) * scene.extent;
        const second = (random() - 0.5) * scene.extent;
        const local =
          face === 0 ? [0, first, second] : face === 1 ? [first, 0, second] : [first, second, 0];
        targetValues.push(...local.map((value, axis) => value + scene.center[axis]));
        normalValues.push(Number(face === 0), Number(face === 1), Number(face === 2));
        // target = rotation * (source - center) + center + translation, so the source is the
        // inverse rotation of the target about the corner, minus the translation.
        const horizontal = local[0] - translation[0];
        const vertical = local[1] - translation[1];
        sourceValues.push(
          cosine * horizontal + sine * vertical + scene.center[0],
          -sine * horizontal + cosine * vertical + scene.center[1],
          local[2] - translation[2] + scene.center[2]
        );
      }
      const source = fixture.view('float32x3', new Float32Array(sourceValues));
      const target = fixture.view('float32x3', new Float32Array(targetValues));
      const normals = fixture.view('float32x3', new Float32Array(normalValues));
      const transform = fixture.view('float32', new Float32Array(16));
      const iterations = 12;
      const history = fixture.view('float32x3', new Float32Array(3 * iterations));
      const mask = fixture.view('uint32', new Uint32Array(POINT_COUNT));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudICP({
          source: source.view,
          target: target.view,
          targetNormals: normals.view,
          metric: 'point-to-plane',
          maximumDistance: 0.02 * scene.extent,
          iterations,
          translationTolerance: 1e-4 * scene.extent,
          rotationTolerance: 1e-4,
          output: {
            transform: transform.view,
            history: history.view,
            inlierMask: mask.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      const [code] = await readIntegers(status.buffer);
      expect(code).not.toBe(GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE);
      expect(code).toBe(0);
      const actual = await readFloats(transform.buffer);
      // Rotation columns of a rotation by ROTATION_ANGLE about the vertical axis.
      const expectedRotation = [cosine, sine, 0, -sine, cosine, 0, 0, 0, 1];
      for (let column = 0; column < 3; column++) {
        for (let row = 0; row < 3; row++) {
          expect(
            Math.abs(actual[column * 4 + row] - expectedRotation[column * 3 + row])
          ).toBeLessThan(2e-5);
        }
      }
      // The corner itself must land on corner + translation. Evaluated in float64 from the
      // float32 transform, so the tolerance is the coordinate resolution, not the lever arm.
      const moved = [0, 1, 2].map(
        row =>
          actual[row] * scene.center[0] +
          actual[4 + row] * scene.center[1] +
          actual[8 + row] * scene.center[2] +
          actual[12 + row]
      );
      for (let axis = 0; axis < 3; axis++) {
        expect(
          Math.abs(moved[axis] - (scene.center[axis] + translation[axis])) / scene.extent
        ).toBeLessThan(5e-5);
      }
      const inliers = await readIntegers(mask.buffer);
      expect(inliers.reduce((sum, value) => sum + value, 0)).toBe(POINT_COUNT);
    } finally {
      fixture.destroy();
    }
  });
}

it('point-to-plane ICP still reports a single tilted plane as degenerate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // One plane with normal (1, 1, 1) / sqrt(3) leaves two translations and one rotation free.
    // No column of the system is exactly zero, so the rank loss has to be found by the pivot test.
    const random = createUniformGenerator(2024);
    const component = Math.sqrt(1 / 3);
    const firstAxis = [Math.SQRT1_2, -Math.SQRT1_2, 0];
    const secondAxis = [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)];
    const targetValues: number[] = [];
    const normalValues: number[] = [];
    const sourceValues: number[] = [];
    for (let point = 0; point < POINT_COUNT; point++) {
      const first = (random() - 0.5) * 10;
      const second = (random() - 0.5) * 10;
      for (let axis = 0; axis < 3; axis++) {
        const value = 5 + first * firstAxis[axis] + second * secondAxis[axis];
        targetValues.push(value);
        sourceValues.push(value - 0.01 * component);
        normalValues.push(component);
      }
    }
    const transform = fixture.view('float32', new Float32Array(16));
    const history = fixture.view('float32x3', new Float32Array(12));
    const mask = fixture.view('uint32', new Uint32Array(POINT_COUNT));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: fixture.view('float32x3', new Float32Array(sourceValues)).view,
        target: fixture.view('float32x3', new Float32Array(targetValues)).view,
        targetNormals: fixture.view('float32x3', new Float32Array(normalValues)).view,
        metric: 'point-to-plane',
        maximumDistance: 0.2,
        iterations: 4,
        output: {
          transform: transform.view,
          history: history.view,
          inlierMask: mask.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([GPU_POINT_CLOUD_ICP_STATUS_DEGENERATE]);
    // The last good iterate is the identity the run started from.
    expect(await readFloats(transform.buffer)).toEqual([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
    ]);
  } finally {
    fixture.destroy();
  }
});
