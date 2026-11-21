// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createIsochroneOracleHarness, expectOracleValues} from './isochrones-output-oracle-utils';

it('exposes similarity cap evidence and usable Hausdorff without map matching (STA-10)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const recipe = addTrackQualityRecipe(harness.graph, {
    trajectory: {
      positions: harness.input('float32x2', [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1]),
      timestamps: harness.input('float32', [0, 1, 2, 0, 1, 2]),
      trackOffsets: harness.input('uint32', [0, 3, 6]),
      temporalContext: {
        encoding: 'float32-relative',
        unit: 'second',
        epoch: 0n,
        axis: {kind: 'irregular'}
      }
    },
    thresholds: harness.input('float32', [100, 100]),
    similarity: {
      pairA: harness.input('uint32', [0]),
      pairB: harness.input('uint32', [1]),
      maximumFrechetVertexCount: 2
    }
  });
  const stage = recipe.status.stages.find(entry => entry.stage === 'similarity');
  expect(stage?.layout?.kind).toBe('per-pair-bits');
  expect(stage?.layout?.view === recipe.outputs.similarity!.pairStatus).toBe(true);
  expect(recipe.outputs.matching).toBeUndefined();
  for (const [name, source, format, type] of [
    ['pairStatus', stage!.layout!.view, 'uint32', 'u32'],
    ['hausdorff', recipe.outputs.similarity!.hausdorff, 'float32', 'f32'],
    ['frechet', recipe.outputs.similarity!.frechet, 'float32', 'f32']
  ] as const) {
    const destination = harness.output(name, format, 1);
    harness.graph.add(
      createWGSLKernelNode(harness.graph, {
        id: `read-${name}`,
        operation: 'track-quality-stage-test',
        invocationCount: 1,
        bindings: [
          {name: 'source', view: source, type, access: 'read'},
          {name: 'destination', view: destination, type, access: 'read_write'}
        ],
        body: 'destination[destinationOffset] = source[sourceOffset];'
      })
    );
  }
  const result = await harness.run();
  // Parallel three-vertex tracks at separation 1; 3 > cap 2, independent of Hausdorff.
  expectOracleValues(result['pairStatus'], [2]);
  expectOracleValues(result['hausdorff'], [1]);
  expect(result['frechet']).toEqual([NaN]);
});
