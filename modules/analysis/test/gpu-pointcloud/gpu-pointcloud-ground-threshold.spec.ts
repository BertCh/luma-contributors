// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudGround} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

// The stage threshold of GPUPointCloudGround is a documented convention that differs from equation
// 7 of Zhang et al. (2003) (math review finding core-numerics-06). This spec pins the documented
// numbers: with radii [1, 2], cellSize 1, slope 0.1 and initialThreshold 0.2 the thresholds are
// 0.3 and 0.3 (Zhang: 0.2 and 0.4). Expected flags are worked out by hand from the openings.

const WIDTH = 16;

it('the ground stage threshold is initial + slope * cellSize * radius increment at every stage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const surface = new Float32Array(WIDTH * WIDTH);
    const expectedGround = new Uint32Array(WIDTH * WIDTH).fill(1);
    // A one-cell bump is removed by the first opening (3 by 3 window), so its residual meets the
    // first threshold, 0.3: 0.25 stays ground (Zhang's 0.2 would reject it), 0.35 is rejected.
    surface[3 * WIDTH + 3] = 0.25;
    surface[3 * WIDTH + 11] = 0.35;
    expectedGround[3 * WIDTH + 11] = 0;
    // A 3 by 3 block survives the first opening (residual 0) and is removed by the second (5 by
    // 5 window), so its residual meets the second threshold, 0.3: 0.25 stays ground and 0.35 is
    // rejected (Zhang's 0.4 would keep it).
    for (let row = 9; row <= 11; row++) {
      for (let column = 2; column <= 4; column++) {
        surface[row * WIDTH + column] = 0.25;
      }
      for (let column = 10; column <= 12; column++) {
        surface[row * WIDTH + column] = 0.35;
        expectedGround[row * WIDTH + column] = 0;
      }
    }
    const positions = new Float32Array(WIDTH * WIDTH * 3);
    surface.forEach((value, cell) => {
      positions.set([(cell % WIDTH) + 0.5, Math.floor(cell / WIDTH) + 0.5, value], cell * 3);
    });
    const ground = fixture.view('uint32', new Uint32Array(WIDTH * WIDTH));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions: fixture.view('float32x3', positions).view,
        classifications: fixture.view('uint32', new Uint32Array(WIDTH * WIDTH).fill(1)).view,
        grid: {origin: [0, 0], cellSize: 1, width: WIDTH, height: WIDTH},
        radii: [1, 2],
        initialThreshold: 0.2,
        slope: 0.1,
        maximumThreshold: 1,
        minimumSurface: fixture.view('float32', surface).view,
        coverage: fixture.view('uint32', new Uint32Array(WIDTH * WIDTH).fill(1)).view,
        output: {
          groundMask: ground.view,
          outputClassifications: fixture.view('uint32', new Uint32Array(WIDTH * WIDTH)).view,
          status: fixture.view('uint32', new Uint32Array(WIDTH * WIDTH)).view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(ground.buffer)).toEqual(Array.from(expectedGround));
  } finally {
    fixture.destroy();
  }
});
