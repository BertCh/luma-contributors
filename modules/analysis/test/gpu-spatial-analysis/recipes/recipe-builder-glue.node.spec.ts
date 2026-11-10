// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  addRecipeCountNodes,
  RecipeBuilder
} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('RecipeBuilder.addGlue lists a named glue stage as a contributor and adds its nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const builder = new RecipeBuilder(graph);
    const values = createTransientView(graph, 'glue-values', 'uint32', 4);
    const node = createWGSLKernelNode(graph, {
      id: 'glue-fill',
      operation: 'Test',
      bindings: [{name: 'values', view: values, type: 'u32', access: 'read_write'}],
      invocationCount: 4,
      body: 'values[valuesOffset + index] = index;'
    });
    builder.addGlue('fill', node);
    expect(builder.glueStages).toEqual([{stage: 'fill', nodes: [node]}]);
    expect(builder.contributors).toHaveLength(1);
    expect(builder.contributors[0].getCommandNodes(graph)).toEqual([node]);
  } finally {
    device.destroy();
  }
});

it('addRecipeCountNodes adds a row-parallel kernel and a reduction per count', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const builder = new RecipeBuilder(graph);
    const validity = createTransientView(graph, 'count-validity', 'uint32', 1000);
    const missing = createTransientView(graph, 'count-missing', 'uint32', 1);
    addRecipeCountNodes(builder, {
      id: 'count-test',
      operation: 'Test',
      rowCount: 1000,
      bindings: [{name: 'validity', view: validity, type: 'u32', access: 'read'}],
      counts: [
        {name: 'missing', output: missing, value: 'u32(validity[validityOffset + index] == 0u)'}
      ]
    });
    expect(builder.glueStages.map(entry => entry.stage)).toEqual(['count-test']);
    // One glue producer for the kernel plus one published GPUReduction for the count.
    expect(builder.contributors.length).toBe(2);
  } finally {
    device.destroy();
  }
});
