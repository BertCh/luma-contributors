// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {
  GPUCommandGraph,
  GPUCommandGraphGPUIndirectCondition,
  GraphDataView
} from '../gpu-core/gpu-command-graph';
import {createWGSLDispatchGate} from '../utils/wgsl-kernel-nodes';

/**
 * GPU-written indirect dispatch gate for statically unrolled convergence loops. Once a one-thread
 * node runs {@link GPUGraphAlgorithmsIterationGate.disarmWGSL}, every later gated iteration pass
 * launches zero workgroups, so the recorded iteration budget costs no GPU time after convergence
 * and never reads state back to the CPU. @internal
 */
export type GPUGraphAlgorithmsIterationGate = {
  /** Transient dispatch triples; bind as `dispatch` (read_write) in the arming and disarming nodes. */
  view: GraphDataView<'uint32'>;
  /** WGSL statements that publish every bounded dispatch layout. Requires `dispatchOffset`. */
  armWGSL: string;
  /** WGSL statements that zero every dispatch. Requires `dispatchOffset`. */
  disarmWGSL: string;
  /** Returns the indirect condition that gates the node `nodeId` on dispatch shape `slot`. */
  getCondition: (nodeId: string, slot: number) => GPUCommandGraphGPUIndirectCondition;
};

/** Creates a gate with one dispatch triple per `[invocationCount, workgroupSize]` shape. @internal */
export function createGPUGraphAlgorithmsIterationGate<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; shapes: readonly (readonly [number, number])[]}
): GPUGraphAlgorithmsIterationGate {
  const gate = createWGSLDispatchGate<Parameters>(commandGraph, props);
  return {
    view: gate.view,
    armWGSL: gate.armWGSL,
    disarmWGSL: gate.disarmWGSL,
    // WGSLDispatchGate conditions are always GPU-indirect; its type merely widens to the CPU form.
    getCondition: (nodeId, slot) =>
      ({
        ...gate.getGate(slot).condition,
        id: `${nodeId}-gate`
      }) as GPUCommandGraphGPUIndirectCondition
  };
}
