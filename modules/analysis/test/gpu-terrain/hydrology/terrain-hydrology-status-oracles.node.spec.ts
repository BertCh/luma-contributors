// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {checkTerrainCPUStatus} from '../terrain-status-oracles-test-utils';
import {getTerrainHydrologyStatusOracles} from './terrain-hydrology-status-oracles-fixtures';
import {executeHANDPublication} from './hand-publication-fixture';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';

for (const oracle of getTerrainHydrologyStatusOracles()) {
  it(`CPU hydrology independent status oracle: ${oracle.name}`, async () => {
    await checkTerrainCPUStatus(oracle);
  });
}

it('watershed publication verifies original receivers before accepting a pointer fixpoint', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUTerrainWatersheds({
    grid: {width: 3, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
    maximumIterations: 2,
    flowDirections: createTransientView(graph, 'directions', 'uint32', 3),
    output: {
      labels: createTransientView(graph, 'labels', 'uint32', 3),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  }).getCommandNodes(graph);
  try {
    const publication = getRasterPipelines(nodes.filter(node => node.id.endsWith('-labels')))[0];
    if (!publication) throw new Error('Missing watershed publication');
    const memory: Record<string, Uint32Array> = {
      flowDirections: Uint32Array.of(1, 16, 0),
      pointer: Uint32Array.of(0, 0, 2),
      labels: new Uint32Array(3).fill(99),
      converged: Uint32Array.of(1)
    };
    const bindings = Object.fromEntries(
      new WgslReflect(publication.source).storage.map(binding => [
        binding.binding,
        memory[binding.name]
      ])
    );
    new WgslExec(new WgslParser().parse(publication.source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: {INVOCATION_COUNT: 3, WGSL_KERNEL_DISPATCH_X: 1, WGSL_KERNEL_DISPATCH_Y: 1}}
    );
    // Cells 0/1 cycle without a pour point; cell 2 is a genuine terminal.
    expect(Array.from(memory['labels'])).toEqual([0xfffffffe, 0xfffffffe, 2]);
    expect(Array.from(memory['converged'])).toEqual([0]);
  } finally {
    device.destroy();
  }
});

for (const [directionEncoding, directions] of [
  ['esri', [1, 16]],
  ['whitebox', [2, 32]],
  ['taudem', [1, 5]]
] as const) {
  it(`D8 HAND production publication resolves the first stream: ${directionEncoding} (MRG-05)`, () => {
    for (const includeDrainage of [true, false]) {
      const input = {
        directionEncoding,
        directions: [...directions],
        pointer: [0, 0],
        includeDrainage
      };
      const cycle = executeHANDPublication({...input, streams: [0, 0]});
      expect(cycle.heights).toEqual([NaN, NaN]);
      expect(cycle.converged).toBe(0);
      const result = executeHANDPublication({...input, streams: [1, 0]});
      // Inclusive stream 0: elevations [5,3] minus stream elevation 5 give [0,-2].
      expect(result.heights.every(Number.isFinite)).toBe(true);
      expect(result.heights).toEqual([0, -2]);
      expect(result.converged).toBe(1);
      if (includeDrainage) {
        expect(cycle.drainage).toEqual([0xfffffffe, 0xfffffffe]);
        expect(result.drainage).toEqual([0, 0]);
      }
    }
  });
}
