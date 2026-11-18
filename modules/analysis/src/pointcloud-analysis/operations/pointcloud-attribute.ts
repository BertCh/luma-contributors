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

export function readPointCloudAttribute(name: string, index: string, components: number) {
  const values = Array.from(
    {length: components},
    (_, component) => `read${name}(${index} * ${components}u + ${component}u)`
  );
  return components === 1 ? values[0] : `vec${components}f(${values.join(', ')})`;
}

export function writePointCloudAttribute(
  name: string,
  index: string,
  value: string,
  components: number
) {
  return Array.from(
    {length: components},
    (_, component) =>
      `write${name}(${index} * ${components}u + ${component}u, ${value}${components === 1 ? '' : `.${'xyzw'[component]}`});`
  ).join('\n');
}
