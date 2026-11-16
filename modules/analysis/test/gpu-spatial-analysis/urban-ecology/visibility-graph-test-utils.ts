// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {RecipeTestFixture} from '../recipes/recipe-harness';

/** Keeps readback resources separate from recipe-owned outputs with the same field names. */
export function publishVisibilityGraphOutput(
  fixture: RecipeTestFixture,
  name: string,
  source: GraphDataView
) {
  const components = source.format.endsWith('x4') ? 4 : source.format.endsWith('x2') ? 2 : 1;
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  const output = fixture.output(`readback-${name}`, source.format as 'float32', source.length);
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: `readback-${name}`,
      operation: 'visibility-csr-readback',
      invocationCount: source.length * components,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return {output, source, components};
}
