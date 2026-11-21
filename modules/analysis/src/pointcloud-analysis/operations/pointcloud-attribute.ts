// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

export type PointCloudAttributeFormat = 'float32' | 'float32x2' | 'float32x3' | 'float32x4';

export function flattenPointCloudAttribute<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<PointCloudAttributeFormat>,
  id = 'pointcloud-attribute'
) {
  const components = getPointCloudAttributeComponents(view.format);
  // Attribute rows must be tightly packed; reinterpretation preserves the caller's byte offset.
  validateRequirement(
    id,
    view.byteStride === components * 4,
    'attribute views must be tightly packed'
  );
  return graph.createDataView(view.buffer, {
    format: 'float32',
    length: view.length * components,
    byteOffset: view.byteOffset
  });
}

export function getPointCloudAttributeComponents(format: PointCloudAttributeFormat) {
  return Number(format.match(/x([234])$/)?.[1] ?? 1);
}

/**
 * Wraps a WGSL expression in parentheses unless it is a plain identifier or numeric literal, so a
 * spliced argument such as `index + 1u` keeps its meaning next to `*` or a `.x` swizzle.
 */
function groupWGSLExpression(expression: string): string {
  return /^\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+[uif]?)\s*$/.test(expression)
    ? expression.trim()
    : `(${expression})`;
}

/**
 * WGSL that reads row `index` of a tightly packed attribute through `read<name>(word)`. `index`
 * may be any `u32` expression; it is parenthesised before it is scaled.
 */
export function readPointCloudAttribute(name: string, index: string, components: number) {
  const row = groupWGSLExpression(index);
  const values = Array.from(
    {length: components},
    (_, component) => `read${name}(${row} * ${components}u + ${component}u)`
  );
  return components === 1 ? values[0] : `vec${components}f(${values.join(', ')})`;
}

/**
 * WGSL that writes `value` to row `index` through `write<name>(word, value)`. `index` and `value`
 * may be any expressions; each is parenthesised before it is scaled or swizzled.
 */
export function writePointCloudAttribute(
  name: string,
  index: string,
  value: string,
  components: number
) {
  const row = groupWGSLExpression(index);
  const vector = groupWGSLExpression(value);
  return Array.from(
    {length: components},
    (_, component) =>
      `write${name}(${row} * ${components}u + ${component}u, ${vector}${components === 1 ? '' : `.${'xyzw'[component]}`});`
  ).join('\n');
}
