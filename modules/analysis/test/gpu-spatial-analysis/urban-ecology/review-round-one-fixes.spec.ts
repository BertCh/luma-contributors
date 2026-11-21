// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {addBuildingAdjacencyRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-adjacency';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  'metrics',
  'reference',
  'boundary'
] as const)('R1 B4 %s provenance patches retain zero-based index space', async route => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `provenance-${route}`);
  try {
    const patchClasses = fixture.input('input-categories', new Uint32Array([0, 1]), 'uint32', 2);
    const labels = fixture.input('input-labels', new Uint32Array([0, 1, 2, 1, 0]), 'uint32', 5);
    const first = fixture.output('first-result', 'float32x4', 2),
      second = fixture.output('second-result', 'float32x4', 2);
    const common = {
      patchClasses,
      width: 5,
      height: 1,
      cellSize: [2, 3] as const,
      connectivity: 4 as const
    };
    const denseLabels = {
      labels,
      grid: {
        width: 5,
        height: 1,
        affine: [2, 0, 0, 0, 3, 0] as const,
        pixelInterpretation: 'area' as const
      }
    };
    if (route === 'metrics') {
      const result = addLandscapeMetricsRecipe(fixture.graph, {
        ...common,
        id: 'first',
        denseLabels,
        classCount: 2,
        output: {patchMetrics: first.view}
      });
      addLandscapeMetricsRecipe(fixture.graph, {
        ...common,
        id: 'second',
        patches: result.provenance.patches,
        classCount: 2,
        output: {patchMetrics: second.view}
      });
    } else if (route === 'reference') {
      const result = addPatchProximityRecipe(fixture.graph, {
        ...common,
        id: 'first',
        mode: 'reference',
        denseLabels,
        edgeDepth: 0,
        searchRadius: 0,
        output: {metrics: first.view}
      });
      addPatchProximityRecipe(fixture.graph, {
        ...common,
        id: 'second',
        mode: 'reference',
        patches: result.provenance.patches,
        edgeDepth: 0,
        searchRadius: 0,
        output: {metrics: second.view}
      });
    } else {
      const result = addPatchProximityRecipe(fixture.graph, {
        ...common,
        id: 'first',
        mode: 'boundary',
        denseLabels,
        edgeDepth: 0,
        searchRadius: 0,
        output: {metrics: first.view}
      });
      addPatchProximityRecipe(fixture.graph, {
        ...common,
        id: 'second',
        mode: 'boundary',
        patches: result.provenance.patches,
        edgeDepth: 0,
        searchRadius: 0,
        output: {metrics: second.view}
      });
    }
    fixture.run();
    // Two cells of patch 1 and one of patch 2, each of area 2*3 = 6.
    for (const output of [first, second]) {
      const values = await fixture.readFloat32(output, 8);
      expect(values).toHaveLength(8);
      expect([values[0], values[4]].every(Number.isFinite)).toBe(true);
      expect([values[0], values[4]]).toEqual([12, 6]);
    }
  } finally {
    fixture.destroy();
  }
});

it.each([
  'reference',
  'boundary'
] as const)('R1 earlier %s cell-center singleton reports a missing neighbor', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `singleton-${mode}`);
  try {
    const metrics = fixture.output('result', 'float32x4', 1);
    const common = {
      patches: fixture.input('input-owners', new Uint32Array([0]), 'uint32', 1),
      patchClasses: fixture.input('input-categories', new Uint32Array([0]), 'uint32', 1),
      width: 1,
      height: 1,
      cellSize: [1, 1] as const,
      edgeDepth: 0,
      searchRadius: 0,
      distanceModel: 'cell-center' as const,
      output: {metrics: metrics.view}
    };
    addPatchProximityRecipe(fixture.graph, {...common, mode});
    fixture.run();
    const values = await fixture.readFloat32(metrics, 4);
    expect(values).toHaveLength(4);
    expect(Number.isNaN(values[2])).toBe(true);
    expect(values[3]).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('R1 earlier fitting shared-path count alone cannot certify complete adjacency', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'uncertified');
  try {
    const complete = fixture.output('result', 'uint32', 1);
    addBuildingAdjacencyRecipe(fixture.graph, {
      sharedPaths: {
        runs: {
          positions: fixture.input(
            'input-positions',
            new Float32Array([0, 0, 0, 1]),
            'float32x2',
            2
          ),
          offsets: fixture.input('input-offsets', new Uint32Array([0, 2]), 'uint32', 2),
          leftSourceIds: fixture.input('input-left', new Uint32Array([0]), 'uint32', 1),
          rightSourceIds: fixture.input('input-right', new Uint32Array([1]), 'uint32', 1),
          forward: fixture.input('input-forward', new Uint32Array([1]), 'uint32', 1)
        },
        status: {count: fixture.input('input-count', new Uint32Array([1]), 'uint32', 1)}
      },
      lineBuildingIds: fixture.input('input-buildings', new Uint32Array([0, 1]), 'uint32', 2),
      orientations: fixture.input('input-orientations', new Float32Array([0, 0]), 'float32', 2),
      output: {complete: complete.view}
    });
    fixture.run();
    expect(await fixture.readUint32(complete, 1)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
