// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import dbscanScene from '../../../../website/src/gpu/demos/dbscan';
import {GraphRunner} from '../../../../website/src/gpu/demos/kit';
import {DBSCAN_PARAMETERS, readParameterValues} from '../../../../website/src/gpu/demos/parameters';
import {defineOperationRun} from '../../../../website/src/gpu/runs/define-run';
import {parameters as recipeParameters} from '../../../../website/src/gpu/runs/entries/recipes-trajectories/cluster-and-outline-recipe.parameters';
import type {GpuSceneContext} from '../../../../website/src/gpu/types';

vi.mock('../../../../website/src/gpu/demos/kit', async importOriginal => {
  const kit = await importOriginal<typeof import('../../../../website/src/gpu/demos/kit')>();
  class TestLayer {
    setInstanceCount() {}
    update() {}
    draw() {}
    destroy() {}
  }
  return {...kit, DemoLayer: TestLayer, DotLayer: TestLayer, SegmentListLayer: TestLayer};
});

test.each([
  ['DBSCAN demo', DBSCAN_PARAMETERS],
  ['cluster-and-outline run', recipeParameters]
] as const)('%s rejects its retired setting before applying defaults', (_name, parameters) => {
  for (const raw of [{minimumPoints: 7}, {minimumPoints: 7, minimumSamples: 8}] as const) {
    expect(() => readParameterValues(parameters, raw)).toThrow(
      'use minimumSamples instead of minimumPoints'
    );
  }
  expect(readParameterValues(parameters, {minimumSamples: 8})['minimumSamples']).toBe(8);
});

test.each([
  'DBSCAN demo',
  'operation run'
])('%s rejects a mixed parameter update atomically', async name => {
  const isDemo = name === 'DBSCAN demo';
  const deviceBuffers: {write: ReturnType<typeof vi.fn>; destroy(): void}[] = [];
  const device = {
    type: 'webgpu',
    createBuffer() {
      const buffer = {write: vi.fn(), destroy() {}};
      deviceBuffers.push(buffer);
      return buffer;
    },
    beginRenderPass: () => ({end() {}}),
    getDefaultCanvasContext: () => ({getDrawingBufferSize: () => [320, 200]}),
    submit() {}
  } as unknown as GpuSceneContext['device'];
  const graphRunner = vi.spyOn(GraphRunner, 'create').mockImplementation(async (_, id, build) => {
    if (id === 'run-operation') build({} as GPUCommandGraph<void>);
    return {run() {}, destroy() {}} as GraphRunner;
  });
  vi.stubGlobal('requestAnimationFrame', () => 1);
  vi.stubGlobal('cancelAnimationFrame', () => {});

  const initial = {epsilon: 3.125, minimumSamples: 8};
  const observeUpdate = vi.fn();
  // The generic run uses the recipe controls and an empty GPU graph.
  const scene = isDemo
    ? dbscanScene
    : defineOperationRun({
        parameters: recipeParameters,
        createInputs: () => undefined,
        build: () => ({outputs: null, update: values => observeUpdate({...values})}),
        draw: []
      });
  try {
    const instance = await scene({
      device,
      canvas: {} as HTMLCanvasElement,
      width: 320,
      height: 200,
      theme: 'light',
      reducedMotion: true,
      seed: 1,
      parameters: initial
    });
    try {
      // The demo uploads positions first, then its parameters. Words 4 and 5 are epsilon
      // in world units and the self-inclusive minimum sample count.
      const writeParameters = isDemo ? deviceBuffers[1].write : observeUpdate;
      const assertParameters = () => {
        if (isDemo) {
          const words = writeParameters.mock.lastCall?.[0] as Float32Array;
          expect(Array.from(words.slice(4, 6))).toEqual([0.03125, 8]);
        } else {
          expect(writeParameters).toHaveBeenLastCalledWith(initial);
        }
      };
      instance.render(0);
      assertParameters();
      expect(() =>
        instance.setParameters!({epsilon: 4, minimumPoints: 7, minimumSamples: 13})
      ).toThrow('use minimumSamples instead of minimumPoints');

      // Force a fresh upload without writing either slider so partial mutations become visible.
      writeParameters.mockClear();
      instance.setParameters!({});
      instance.render(1);
      expect(writeParameters).toHaveBeenCalledTimes(1);
      assertParameters();
    } finally {
      instance.destroy();
    }
  } finally {
    graphRunner.mockRestore();
    vi.unstubAllGlobals();
  }
});
