// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET,
  type GraphChunkGroupKernelProps
} from '../../analysis-core/kernels/graph-chunk-groups';

/** Runtime CSR bounds shared by every chunk and tile of a statistics row kernel. @internal */
export const SPATIAL_STATISTICS_ROW_DISPATCH_WGSL = `var<private> CAPACITY: u32;
var<private> WEIGHTS_ROW_START: u32;`;

/** Dispatches a CSR row range with no chunk-dependent source or pipeline constants. @internal */
export function createSpatialStatisticsRowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GraphChunkGroupKernelProps<Parameters> & {weightsRowStart?: number}
): GPUCommandNode<Parameters> {
  return createGraphChunkGroupKernelNode(graph, {
    ...props,
    parameterValues: [props.weightsRowStart ?? props.rowBase ?? 0],
    declarations: `${SPATIAL_STATISTICS_ROW_DISPATCH_WGSL}\n${props.declarations ?? ''}`,
    body: `CAPACITY = arrayLength(&neighbors) - neighborsOffset;
  WEIGHTS_ROW_START = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  { let index = index + getChunkParameter(0u);
  ${props.body}
  }`
  });
}
