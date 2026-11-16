// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {RecipeTestFixture} from '../recipes/recipe-harness';

export function publishBuildingTessellationOutput(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView
) {
  const output = fixture.output(`readback-${name}`, source.format as 'float32', source.length);
  const components = source.format.endsWith('x4') ? 4 : source.format.endsWith('x2') ? 2 : 1;
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: `readback-${name}`,
      operation: 'local-tessellation-readback',
      invocationCount: source.length * components,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}
