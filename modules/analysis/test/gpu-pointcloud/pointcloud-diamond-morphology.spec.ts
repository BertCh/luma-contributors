// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {getPointCloudDiamondMorphologyNodes} from '../../src/pointcloud-analysis/operations/pointcloud-diamond-morphology';
import {getPointCloudKernelNodes} from '../../src/pointcloud-analysis/operations/pointcloud-kernel-nodes';
import {makeSpatialFixture, readFloats} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {createSeededRandom} from '../utils/wgsl-module-harness';

// Parity of the SMRF windows, which delegate to the published GPURasterErosion and
// GPURasterDilation, with the definition: the minimum or maximum over every cell of the grid
// within a Manhattan distance of `radius`. The reference visits the whole diamond per cell, so it
// shares nothing with the sequence of at most radius-8 passes the helper emits.

const WIDTH = 37;
const HEIGHT = 23;

function getReferenceWindow(
  values: Float32Array,
  radius: number,
  operation: 'min' | 'max'
): number[] {
  const result: number[] = [];
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      let extreme = operation === 'min' ? Infinity : -Infinity;
      for (let otherRow = 0; otherRow < HEIGHT; otherRow++) {
        for (let otherColumn = 0; otherColumn < WIDTH; otherColumn++) {
          if (Math.abs(otherRow - row) + Math.abs(otherColumn - column) <= radius) {
            const value = values[otherRow * WIDTH + otherColumn];
            extreme = operation === 'min' ? Math.min(extreme, value) : Math.max(extreme, value);
          }
        }
      }
      result.push(extreme);
    }
  }
  return result;
}

for (const operation of ['min', 'max'] as const) {
  it(`the SMRF ${operation} window equals the clipped Manhattan diamond for radii 1 to 37`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const random = createSeededRandom(operation === 'min' ? 311 : 312);
      // Rough terrain with a few isolated extremes and one large magnitude.
      const values = Float32Array.from({length: WIDTH * HEIGHT}, (_, cell) => {
        const column = cell % WIDTH;
        const row = Math.floor(cell / WIDTH);
        return 3 * Math.sin(column / 4) * Math.cos(row / 3) + 2 * (random() - 0.5);
      });
      values[5 * WIDTH + 7] = 40;
      values[17 * WIDTH + 30] = -55;
      values[0] = -1e6;
      values[WIDTH * HEIGHT - 1] = 2.5e7;
      const input = fixture.view('float32', values);
      const radii = [1, 3, 8, 9, 18, 37];
      const outputs = radii.map(radius => {
        const result = getPointCloudDiamondMorphologyNodes(
          fixture.graph,
          `window-${radius}`,
          input.view,
          {width: WIDTH, height: HEIGHT},
          radius,
          operation
        );
        // The result is a transient view: one more kernel copies it to a caller-owned buffer.
        const published = fixture.view('float32', new Float32Array(WIDTH * HEIGHT));
        fixture.graph.add({
          getCommandNodes: target => [
            ...result.nodes,
            ...getPointCloudKernelNodes(target, {
              id: `window-${radius}-publish`,
              length: WIDTH * HEIGHT,
              inputs: {source: result.output},
              outputs: {published: published.view},
              body: 'writePublished(index, readSource(index));'
            })
          ]
        });
        return published;
      });
      await fixture.run();
      for (const [index, radius] of radii.entries()) {
        expect(await readFloats(outputs[index].buffer), `radius ${radius}`).toEqual(
          getReferenceWindow(values, radius, operation)
        );
      }
      // A radius of 0 emits nothing and returns the input view.
      const identity = getPointCloudDiamondMorphologyNodes(
        fixture.graph,
        'window-0',
        input.view,
        {width: WIDTH, height: HEIGHT},
        0,
        operation
      );
      expect(identity.nodes).toEqual([]);
      expect(identity.output).toBe(input.view);
    } finally {
      fixture.destroy();
    }
  });
}
