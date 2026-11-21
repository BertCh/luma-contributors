// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addPopulationSupportAlignmentRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import {computePopulationSupportAlignmentOnCPU} from '../../../src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it('reads the alignment certificate and diagnostic counts on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'alignment-result-replay');
  let compiled: ReturnType<typeof fixture.graph.compile> | undefined;
  try {
    const populations = new Float32Array([2, 8, 6, 4, 3, 7, 5, 5, 1, 9, 4, 6]);
    const targetIds = new Uint32Array([10, 20, 30, 70, 80, 90]);
    const sourceIds = fixture.output('source-ids', 'uint32', 6);
    sourceIds.buffer.write(new Uint32Array([20, 10, 10, 40, 50, 60]));
    const recipe = addPopulationSupportAlignmentRecipe(fixture.graph, {
      populations: fixture.input('populations', populations, 'float32x2', 6),
      sourceIds: sourceIds.view,
      targetIds: fixture.input('target-ids', targetIds, 'uint32', 6)
    });
    const readout = fixture.output('stage-readout', 'uint32', 4);
    // Separate copy nodes exercise the offset of each scalar alias, including nonzero words.
    for (const [index, stage] of recipe.status.stages.entries()) {
      fixture.graph.add(
        createWGSLKernelNode(fixture.graph, {
          id: `stage-readout-${index}`,
          operation: 'alignment-status-test',
          invocationCount: 1,
          bindings: [
            {
              name: 'source',
              view: stage.stage === 'alignment' ? stage.status.complete : stage.status.invalidCount,
              type: 'u32',
              access: 'read'
            },
            {name: 'destination', view: readout.view, type: 'u32', access: 'read_write'}
          ],
          body: `destination[destinationOffset + ${index}u] = source[sourceOffset];`
        })
      );
    }
    compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    // Missing targets 30, 70, 80 and 90, ambiguous target 10, and source-only rows 40, 50 and 60.
    expect(await fixture.readUint32(readout, 4)).toEqual([4, 1, 3, 0]);
    const reference = computePopulationSupportAlignmentOnCPU({
      populations,
      sourceIds: new Uint32Array([20, 10, 10, 40, 50, 60]),
      targetIds
    });
    expect(Array.from(reference.status).slice(1)).toEqual([4, 1, 3]);
    sourceIds.buffer.write(targetIds);
    submitGraph(device, compiled, undefined);
    expect(await fixture.readUint32(readout, 4)).toEqual([0, 0, 0, 1]);
  } finally {
    compiled?.destroy();
    fixture.destroy();
    device.destroy();
  }
});
