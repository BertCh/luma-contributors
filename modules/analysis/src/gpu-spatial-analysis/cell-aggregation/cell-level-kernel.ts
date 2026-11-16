// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {CELL_ROW_WORKGROUP_SIZE} from './cell-row-kernel';

/** Eight layout/copy fields, one invocation count and binding offsets fit this fixed 128-byte slot. */
export const CELL_LEVEL_PARAMETER_WORD_COUNT = 32;
/** Prevent uint32 invocation-index wrapping in a padded multidimensional dispatch. */
const CELL_LEVEL_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;
const CELL_LEVEL_FIELDS = [
  'ROW_COUNT',
  'CAPACITY',
  'CELL_WIDTH',
  'CELL_LOW_BIT',
  'CELL_HEADER',
  'CELL_RESOLUTION',
  'SOURCE_RESOLUTION',
  'DESTINATION_ROW_BASE'
] as const;
/** Each storage binding needs one offset word in the fixed cell-level parameter slot. */
export const CELL_LEVEL_MAXIMUM_BINDING_COUNT =
  CELL_LEVEL_PARAMETER_WORD_COUNT - 1 - CELL_LEVEL_FIELDS.length;
export type CellLevelValues = Partial<Record<(typeof CELL_LEVEL_FIELDS)[number], number>>;

/**
 * Cell-table kernel whose level shape and binding prefixes are runtime parameters. The caller
 * preflights CELL_LEVEL_PARAMETER_WORD_COUNT words before declaring any graph resources.
 * @internal
 */
export function createCellLevelKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>,
  values: CellLevelValues
): {node: GPUCommandNode<Parameters>; uploads: readonly GPUCommandNode<Parameters>[]} {
  const parameterValues = new Uint32Array(CELL_LEVEL_PARAMETER_WORD_COUNT);
  parameterValues[0] = props.invocationCount;
  CELL_LEVEL_FIELDS.forEach((field, index) => {
    parameterValues[index + 1] = values[field] ?? 0;
  });
  validateCapacity(
    props.id,
    'cell level parameter words',
    1 + CELL_LEVEL_FIELDS.length + props.bindings.length,
    1,
    CELL_LEVEL_PARAMETER_WORD_COUNT
  );
  props.bindings.forEach((binding, index) => {
    parameterValues[1 + CELL_LEVEL_FIELDS.length + index] = getViewElementOffset(binding.view);
  });
  const parameters = createSpatialStatisticsParameterNodes(
    graph,
    `${props.id}-parameters`,
    parameterValues
  );
  const bindings = [
    {
      name: 'cellParameters',
      view: parameters.parameters,
      type: 'u32' as const,
      access: 'read' as const
    },
    ...props.bindings
  ];
  let declarations = props.declarations ?? '';
  let body = props.body;
  const getters: string[] = [];
  for (const [index, field] of CELL_LEVEL_FIELDS.entries()) {
    const expression = `cellLevelField${index}()`;
    getters.push(`fn cellLevelField${index}() -> u32 { return cellParameters[${index + 1}]; }`);
    declarations = declarations.replace(new RegExp(`\\b${field}\\b`, 'g'), expression);
    body = body.replace(new RegExp(`\\b${field}\\b`, 'g'), expression);
  }
  props.bindings.forEach((binding, index) => {
    const expression = `cellBindingOffset${index}()`;
    getters.push(
      `fn cellBindingOffset${index}() -> u32 { return cellParameters[${1 + CELL_LEVEL_FIELDS.length + index}]; }`
    );
    declarations = declarations.replace(new RegExp(`\\b${binding.name}Offset\\b`, 'g'), expression);
    body = body.replace(new RegExp(`\\b${binding.name}Offset\\b`, 'g'), expression);
  });
  body = body.replace(/\bINVOCATION_COUNT\b/g, 'cellParameters[0]');
  const source = `${bindings
    .map(
      (binding, location) =>
        `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n')}
${getters.join('\n')}
${declarations}
@compute @workgroup_size(${CELL_ROW_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${CELL_LEVEL_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CELL_ROW_WORKGROUP_SIZE}u + localInvocationIndex;
  ${props.guardIndex === false ? '' : 'if (index >= cellParameters[0]) { return; }'}
  ${body}
}`;
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    CELL_ROW_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  return {
    uploads: parameters.nodes,
    node: createGPUComputeCommandNode<Parameters>({
      id: props.id,
      resources: bindings.map(binding => ({
        buffer: binding.view,
        usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
      })),
      workload: {
        operation: props.operation,
        ...(props.variant ? {variant: props.variant} : {}),
        commandCount: 1,
        maximumInvocationCount: props.invocationCount,
        maximumWorkgroupCount: layout.x * layout.y * layout.z
      },
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: props.id,
          source,
          shaderLayout: {
            bindings: bindings.map((binding, location) => ({
              name: binding.name,
              group: 0,
              location,
              type: binding.access === 'read' ? 'read-only-storage' : 'storage'
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: Object.fromEntries(
                bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
              ),
              ...layout
            }),
          destroy: () => kernel.destroy()
        };
      }
    })
  };
}
