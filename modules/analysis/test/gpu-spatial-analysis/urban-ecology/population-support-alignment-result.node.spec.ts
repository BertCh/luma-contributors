// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateGPURecipeStatusPort} from '../../../src/analysis-core/contracts/status';
import {addPopulationSupportAlignmentRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual';
import type {GPURecipeResult} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  'omitted',
  'partial',
  'full',
  'offset'
] as const)('publishes canonical recipe status and retains packed words with %s outputs', mode => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packedStorage = createTransientView(graph, 'packed-storage', 'uint32x4', 17);
    const packedStatus = graph.createDataView(packedStorage.buffer, {
      format: 'uint32x4',
      length: 1,
      byteOffset: mode === 'offset' ? 256 : 0
    });
    const suppliedOutputs = {
      populations: createTransientView(graph, 'aligned', 'float32x2', 2),
      alignmentCode: createTransientView(graph, 'codes', 'uint32', 2),
      matchedMask: createTransientView(graph, 'matched', 'uint32', 2),
      status: packedStatus
    };
    const recipe = addPopulationSupportAlignmentRecipe(graph, {
      populations: createTransientView(graph, 'populations', 'float32x2', 2),
      sourceIds: createTransientView(graph, 'sources', 'uint32', 2),
      targetIds: createTransientView(graph, 'targets', 'uint32', 2),
      output:
        mode === 'omitted'
          ? undefined
          : mode === 'partial'
            ? {matchedMask: suppliedOutputs.matchedMask}
            : suppliedOutputs
    });
    expect(recipe.intermediates).toEqual({});
    const canonical: GPURecipeResult = recipe;
    expect(canonical.outputs).toBe(recipe.outputs);
    expect(recipe.status.stages.map(stage => stage.stage)).toEqual([
      'missing-targets',
      'ambiguous-targets',
      'excluded-sources'
    ]);
    expect(() => validateGPURecipeStatusPort('alignment-recipe', recipe.status)).not.toThrow();
    expect(recipe.packedStatus.format).toBe('uint32x4');
    if (mode === 'full' || mode === 'offset') expect(recipe.packedStatus).toBe(packedStatus);
    for (const [index, stage] of recipe.status.stages.entries()) {
      expect(stage.status.invalidCount.format).toBe('uint32');
      expect(stage.status.invalidCount.length).toBe(1);
      expect(stage.status.invalidCount.buffer).toBe(recipe.packedStatus.buffer);
      expect(stage.status.invalidCount.byteOffset).toBe(
        recipe.packedStatus.byteOffset + (index + 1) * 4
      );
    }
    expect(recipe.contributors).toHaveLength(1);
  } finally {
    device.destroy();
  }
});
