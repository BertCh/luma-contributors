// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Shared kernel helper's default cooperative size; padding must be guarded in whole workgroups. */
const STATISTICS_COOPERATIVE_WORKGROUP_SIZE = 256;

/**
 * Protect cooperative statistics scratch from extra workgroups in a bounded 2D/3D dispatch.
 * guardIndex:false preserves barriers, but also disables the shared helper's logical row guard.
 * Out-of-range storage writes may clamp to the final partial, so every such group must return.
 * Overrides are uniform and preserve the caller's existing shader/pipeline cardinality.
 * @internal
 */
export function createStatisticsKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize ?? STATISTICS_COOPERATIVE_WORKGROUP_SIZE;
  return createWGSLKernelNode(graph, {
    ...props,
    body:
      props.guardIndex === false
        ? `if (workgroupIndex >= (INVOCATION_COUNT - 1u) / ${workgroupSize}u + 1u) { return; }\n${props.body}`
        : props.body
  });
}
