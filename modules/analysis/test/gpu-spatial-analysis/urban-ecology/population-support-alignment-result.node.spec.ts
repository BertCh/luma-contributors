// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateGPURecipeStatusPort} from '../../../src/analysis-core/contracts/status';
import {
  addPopulationSupportAlignmentRecipe,
  GPUPopulationSupportAlignment,
  type GPUPopulationSupportAlignmentProps,
  type GPUPopulationSupportAlignmentRecipeProps
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import type {GPURecipeResult} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  'omitted',
  'partial',
  'full',
  'offset'
] as const)('publishes all four status words as named stages with %s outputs', mode => {
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
    const recipeProps: GPUPopulationSupportAlignmentRecipeProps = {
      populations: createTransientView(graph, 'populations', 'float32x2', 2),
      sourceIds: createTransientView(graph, 'sources', 'uint32', 2),
      targetIds: createTransientView(graph, 'targets', 'uint32', 2),
      output:
        mode === 'omitted'
          ? undefined
          : mode === 'partial'
            ? {matchedMask: suppliedOutputs.matchedMask}
            : suppliedOutputs
    };
    const recipe = addPopulationSupportAlignmentRecipe(graph, recipeProps);
    expect(recipe.intermediates).toEqual({});
    const canonical: GPURecipeResult = recipe;
    expect(canonical.outputs).toBe(recipe.outputs);
    expect(recipe.status.stages.map(stage => stage.stage)).toEqual([
      'missing-targets',
      'ambiguous-targets',
      'excluded-sources',
      'alignment'
    ]);
    expect(() => validateGPURecipeStatusPort('alignment-recipe', recipe.status)).not.toThrow();
    expect(recipe).not.toHaveProperty('packedStatus');
    const [missing, ambiguous, excluded, alignment] = recipe.status.stages;
    const words = [
      alignment.status.complete,
      missing.status.invalidCount,
      ambiguous.status.invalidCount,
      excluded.status.invalidCount
    ];
    const [complete] = words;
    if (mode === 'full' || mode === 'offset') {
      expect(complete.buffer).toBe(packedStatus.buffer);
      expect(complete.byteOffset).toBe(packedStatus.byteOffset);
    } else {
      expect(complete.byteOffset).toBe(0);
    }
    for (const [index, word] of words.entries()) {
      expect(word.format).toBe('uint32');
      expect(word.length).toBe(1);
      expect(word.buffer).toBe(complete.buffer);
      expect(word.byteOffset).toBe(complete.byteOffset + index * Uint32Array.BYTES_PER_ELEMENT);
    }
    expect(recipe.contributors).toHaveLength(1);
  } finally {
    device.destroy();
  }
});

it('constructs the contributor with its public props and requires outputs there', () => {
  function build(props: GPUPopulationSupportAlignmentProps) {
    return new GPUPopulationSupportAlignment(props);
  }
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const recipeProps: GPUPopulationSupportAlignmentRecipeProps = {
      populations: createTransientView(graph, 'populations', 'float32x2', 1),
      sourceIds: createTransientView(graph, 'source-ids', 'uint32', 1),
      targetIds: createTransientView(graph, 'target-ids', 'uint32', 1)
    };
    expect(
      build({
        ...recipeProps,
        output: {
          populations: createTransientView(graph, 'aligned', 'float32x2', 1),
          alignmentCode: createTransientView(graph, 'codes', 'uint32', 1),
          matchedMask: createTransientView(graph, 'matched', 'uint32', 1),
          status: createTransientView(graph, 'status', 'uint32x4', 1)
        }
      })
    ).toBeInstanceOf(GPUPopulationSupportAlignment);
    expect(() => {
      // @ts-expect-error Recipe props may omit views that the contributor requires.
      new GPUPopulationSupportAlignment(recipeProps);
    }).toThrow('population-support-alignment output.populations is required');
  } finally {
    device.destroy();
  }
});
