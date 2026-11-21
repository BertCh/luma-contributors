// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addBuildingAdjacencyRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-adjacency';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  'complete',
  'candidate',
  'runs',
  'vertices',
  'vertex-count'
] as const)('B2 adjacency consumes optional upstream loss evidence (%s)', async failure => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'adjacency-evidence');
  try {
    const scalar = (name: string, value: number) =>
      fixture.input(name, new Uint32Array([value]), 'uint32', 1);
    const complete = fixture.output('complete', 'uint32', 1);
    const pairs = fixture.output('pairs', 'float32x4', 4);
    const components = fixture.output('components', 'uint32', 2);
    addBuildingAdjacencyRecipe(fixture.graph, {
      sharedPaths: {
        runs: {
          positions: fixture.input('positions', new Float32Array([0, 0, 0, 3]), 'float32x2', 2),
          offsets: fixture.input('offsets', new Uint32Array([0, 2]), 'uint32', 2),
          leftSourceIds: scalar('left', 0),
          rightSourceIds: scalar('right', 1),
          forward: scalar('forward', 1)
        },
        // Deliberately omit aggregate overflow: the complete set of component evidence certifies coverage.
        status: {
          count: scalar('count', 1),
          uncertainCount: scalar('uncertain', 0),
          requiredCount: scalar('required', failure === 'runs' ? 2 : 1),
          candidateOverflow: scalar('candidate', Number(failure === 'candidate')),
          vertexCount: scalar('vertex-count', failure === 'vertex-count' ? 3 : 2),
          requiredVertexCount: scalar('vertex-required', failure === 'vertices' ? 3 : 2)
        }
      },
      lineBuildingIds: fixture.input('owners', new Uint32Array([0, 1]), 'uint32', 2),
      orientations: fixture.input('angles', new Float32Array([0, 0]), 'float32', 2),
      output: {complete: complete.view, pairs: pairs.view, components: components.view}
    });
    fixture.run();
    expect(await fixture.readUint32(complete, 1)).toEqual([Number(failure === 'complete')]);
    const actual = await fixture.readFloat32(pairs, 16);
    expect(actual).toHaveLength(16);
    if (failure === 'complete') {
      // One vertical shared run of length 3 joins buildings 0 and 1. Equal orientations
      // differ by zero; off-diagonal adjacency is 1 and every pair is valid.
      expect(actual.every(Number.isFinite)).toBe(true);
      expect(actual).toEqual([0, 0, 0, 1, 3, 0, 1, 1, 3, 0, 1, 1, 0, 0, 0, 1]);
      expect(await fixture.readUint32(components, 2)).toEqual([0, 0]);
    } else {
      for (let pair = 0; pair < 4; pair++)
        expect(actual.slice(4 * pair, 4 * pair + 4)).toEqual([NaN, NaN, NaN, 0]);
      expect(await fixture.readUint32(components, 2)).toEqual([0xffffffff, 0xffffffff]);
    }
  } finally {
    fixture.destroy();
  }
});
