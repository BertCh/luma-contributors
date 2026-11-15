// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64arithmetic} from '@luma.gl/shaders';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  compileProjectionProgram,
  invertProjectionProgram,
  WEB_MERCATOR_EARTH_RADIUS,
  type ProjectionProgram
} from '@luma.gl/graph/gpu-project';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Fixed WGS84 projected-metre frame shared by geographic flow maps and FTLE stencils. @internal */
export const LAGRANGIAN_GEOGRAPHIC_PROJECTION: ProjectionProgram = {
  precision: 'local-f32',
  operations: [
    {type: 'unit', factor: Math.PI / 180},
    {type: 'web-mercator', arithmetic: 'float32', radius: WEB_MERCATOR_EARTH_RADIUS}
  ]
};

/** Resolve an explicitly projected geographic mode; no geodesic metric is implied. @internal */
export function resolveLagrangianSpatialContext(id: string, context?: GPUSpatialContext) {
  return resolveGPUSpatialContext(id, context, {
    coordinateSpaces: ['planar', 'longitude-latitude'],
    metricsByCoordinateSpace: {planar: ['none', 'native'], 'longitude-latitude': ['none']}
  });
}

/**
 * Embed the published callable shaders, sharing one packed parameter binding. Only their storage
 * declarations/names are adapted to createWGSLKernelNode, which owns binding declarations. All
 * coordinate arithmetic and domain rejection remain in the published projection implementation.
 * @internal
 */
export function createLagrangianProjectionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string
) {
  const forward = compileProjectionProgram(LAGRANGIAN_GEOGRAPHIC_PROJECTION);
  const inverse = compileProjectionProgram(
    invertProjectionProgram(LAGRANGIAN_GEOGRAPHIC_PROJECTION)
  );
  const forwardWords = forward.packParameters();
  const inverseWords = inverse.packParameters();
  const words = new Uint32Array(forwardWords.length + inverseWords.length);
  words.set(forwardWords);
  words.set(inverseWords, forwardWords.length);
  const forwardShader = forward.getShader({namespace: 'flowForward'});
  const inverseShader = inverse.getShader({
    namespace: 'flowInverse',
    parameterOffset: forwardWords.length
  });
  const adaptSource = (shader: typeof forwardShader) =>
    shader.source
      .replace(`@group(0) @binding(auto) var<storage, read> ${shader.bindingName}: array<u32>;`, '')
      .replaceAll(shader.bindingName, 'projectionParameters')
      .replaceAll('projectionParameters[', 'projectionParameters[projectionParametersOffset + ');
  const view = createTransientView(graph, `${id}-projection-parameters`, 'uint32', words.length);
  return {
    binding: {name: 'projectionParameters', view, type: 'u32', access: 'read'} as const,
    // Use the public arithmetic module identity recognised by the core kernel helper. Its
    // integer define below disables uniforms, exactly as in the published shader's module.
    modules: [fp64arithmetic],
    defines: forwardShader.defines,
    declarations: `${adaptSource(forwardShader)}\n${adaptSource(inverseShader)}`,
    forward: forwardShader.entryPoint,
    inverse: inverseShader.entryPoint,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-projection-parameters-initialize`,
        operation: 'LagrangianGeographicProjection',
        invocationCount: words.length,
        bindings: [{name: 'projectionParameters', view, type: 'u32', access: 'read_write'}],
        declarations: `const WORDS = array<u32, ${words.length}>(${Array.from(words, word => `${word}u`).join(', ')});`,
        body: 'projectionParameters[projectionParametersOffset + index] = WORDS[index];'
      })
    ]
  };
}
