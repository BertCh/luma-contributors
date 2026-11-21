// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addVisibilityGraphRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph';
import {decodeGPUVisibilityGraphAdjacencyCSRStatus} from '../../../src/gpu-spatial-analysis/urban-morphology/visibility-graph-csr';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it('STA-09 retries candidate coverage before final directed slot capacity on the same query', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'visibility-retry');
  try {
    const positions = fixture.input('points', new Float32Array([0, 0, 1, 0, 2, 0]), 'float32x2', 3);
    const segments = fixture.input('wall', new Float32Array([20, 20, 21, 20]), 'float32x4', 1);
    const captures = [];
    for (const [id, maximumGridVisits, edgeCapacity] of [
      ['coverage-lost', 1, 8],
      ['covered-small-output', 8192, 2],
      ['covered-fits', 8192, 4]
    ] as const) {
      const ports = {
        count: fixture.output(`${id}-count`, 'uint32', 1),
        requiredCount: fixture.output(`${id}-required`, 'uint32', 1),
        overflow: fixture.output(`${id}-overflow`, 'uint32', 1),
        candidateOverflow: fixture.output(`${id}-candidate`, 'uint32', 1),
        invalidCount: fixture.output(`${id}-invalid`, 'uint32', 1)
      };
      addVisibilityGraphRecipe(fixture.graph, {
        id,
        mode: 'csr',
        positions,
        segments,
        maximumDistance: 1.1,
        maximumGridVisits,
        edgeCapacity,
        output: {
          status: Object.fromEntries(Object.entries(ports).map(([name, port]) => [name, port.view]))
        }
      });
      captures.push(ports);
    }
    fixture.run();
    const decoded = [];
    for (const ports of captures) {
      const words = [];
      for (const port of Object.values(ports)) {
        const value = await fixture.readUint32(port, 1);
        expect(value).toHaveLength(1);
        expect(Number.isInteger(value[0])).toBe(true);
        words.push(value[0]);
      }
      decoded.push(decodeGPUVisibilityGraphAdjacencyCSRStatus(Uint32Array.from(words)));
    }
    expect(decoded[0].candidateOverflow).toBe(1);
    expect(decoded[0].requiredCount).toBe(0xffffffff);
    expect(decoded[0].requirement).toEqual({kind: 'unknown'});
    // The path has undirected pairs (0,1) and (1,2): exactly four directed slots.
    expect(decoded[1]).toMatchObject({
      count: 0,
      overflow: 1,
      candidateOverflow: 0,
      requirement: {kind: 'exact', value: 4}
    });
    expect(decoded[2]).toMatchObject({
      count: 4,
      overflow: 0,
      candidateOverflow: 0,
      requirement: {kind: 'exact', value: 4}
    });
  } finally {
    fixture.destroy();
  }
});
