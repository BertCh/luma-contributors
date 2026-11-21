// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {declareLandscapeDenseLabelsFixture} from './landscape-dense-labels-test-utils';
import {mirrorViews, type Mirrored} from './urban-ecology-output-test-utils';
import {addBuildingAdjacencyRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-adjacency';
import {addPatchProximityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity';

it.each([
  false,
  true
])('R1 B8 queued dense metrics fixture constructs with reconcile=%s', reconcile => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'dense-construction');
  try {
    expect(() => declareLandscapeDenseLabelsFixture(fixture, 'metrics', reconcile)).not.toThrow();
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('R1 B9 mirrored alias views share one caller allocation', () => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'mirror');
  try {
    const source = createTransientView(fixture.graph, 'source', 'float32x4', 1);
    const mirrored = mirrorViews(
      fixture,
      {measures: source, localMeasures: source},
      '',
      new Map<string, Mirrored>()
    ) as {measures: unknown; localMeasures: unknown};
    expect(mirrored.measures === mirrored.localMeasures).toBe(true);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('R1 earlier shared-path certificate is uncertified without loss evidence', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  try {
    addBuildingAdjacencyRecipe(graph, {
      sharedPaths: {
        runs: {
          positions: view('float32x2', 2),
          offsets: view('uint32', 2),
          leftSourceIds: view('uint32', 1),
          rightSourceIds: view('uint32', 1),
          forward: view('uint32', 1)
        },
        status: {count: view('uint32', 1)}
      },
      lineBuildingIds: view('uint32', 2),
      orientations: view('float32', 2)
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.some(kernel => kernel.id.endsWith('-uncertified-shared-paths'))).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R1 earlier reference singleton tests the infinity word', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    addPatchProximityRecipe(graph, {
      patches: createTransientView(graph, 'owners', 'uint32', 1),
      patchClasses: createTransientView(graph, 'categories', 'uint32', 1),
      width: 1,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 0,
      distanceModel: 'cell-center'
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.some(kernel => /bitcast<u32>\(nearest\)==0x7f800000u/.test(kernel.source))).toBe(
      true
    );
  } finally {
    device.destroy();
  }
});
