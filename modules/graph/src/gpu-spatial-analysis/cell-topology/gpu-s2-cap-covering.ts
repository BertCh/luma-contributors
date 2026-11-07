// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
export type GPUS2CapCoveringProps = {
  id?: string;
  /** Unit center xyz, cap radius radians [0, pi]. */
  caps: GraphDataView<'float32x4'>;
  /** Uniform covering level 0–8. Each invocation examines 6 * 4^level cells. */
  level: number;
  mode?: 'exterior' | 'interior';
  capacity: number;
  /** Exterior expands / interior shrinks by this angular safety margin, default 1e-5. */
  tolerance?: number;
  output: {
    cells: GraphDataView<'uint32x2'>;
    /** (written, required) per cap. */
    counts: GraphDataView<'uint32x2'>;
    /** 0 complete, 1 invalid cap, 2 insufficient capacity. */
    status: GraphDataView<'uint32'>;
  };
};
/** Bounded uniform-level spherical-cap covering. No host readback or hidden refinement scheduler. */
export class GPUS2CapCovering implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUS2CapCoveringProps) {
    this.id = props.id ?? 's2-cap-covering';
    validatePackedView(props.caps, ['float32x4'], this.id);
    validatePackedView(props.output.cells, ['uint32x2'], this.id);
    validatePackedView(props.output.counts, ['uint32x2'], this.id);
    validatePackedView(props.output.status, ['uint32'], this.id);
    // Uniform enumeration is intentionally bounded; this is not an adaptive region coverer.
    assert(Number.isInteger(props.level) && props.level >= 0 && props.level <= 8);
    assert(
      Number.isInteger(props.capacity) &&
        props.capacity >= 1 &&
        props.capacity <= 6 * 4 ** props.level
    );
    assert(
      props.output.cells.length === props.caps.length * props.capacity &&
        props.output.counts.length === props.caps.length &&
        props.output.status.length === props.caps.length
    );
    assert(
      Number.isFinite(props.tolerance ?? 1e-5) &&
        (props.tolerance ?? 1e-5) >= 1e-6 &&
        (props.tolerance ?? 1e-5) <= 1e-2
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.caps]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.caps, ...Object.values(props.output)]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2CapCovering',
        variant: props.mode ?? 'exterior',
        invocationCount: props.caps.length,
        bindings: [
          {name: 'caps', view: props.caps, type: 'f32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${dggs.source}\n${S2_ANALYSIS_WGSL}`,
        body: `let offset = capsOffset + index * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
var failure = 0u; var count = 0u;
if ((bitcast<u32>(dot(center, center)) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u || abs(dot(center, center) - 1.0) > 1e-4 || radius < 0.0 || radius > 3.1415928) { failure = 1u; }
if (failure == 0u) {
  for (var candidate = 0u; candidate < ${6 * 4 ** props.level}u; candidate++) {
    let face = candidate / ${4 ** props.level}u; let position = candidate % ${4 ** props.level}u;
    let cell = dggs_u64_shift_left(vec2u(0u, face), 61u) | dggs_u64_shift_left(vec2u(0u, position), ${61 - 2 * props.level}u) | dggs_u64_shift_left(vec2u(0u, 1u), ${60 - 2 * props.level}u);
    let selected = ${props.mode === 'interior' ? `3.141592653589793 - s2AnalysisDistance(cell, -center) <= radius - ${props.tolerance ?? 1e-5}` : `s2AnalysisDistance(cell, center) <= radius + ${props.tolerance ?? 1e-5}`};
    if (selected) {
      if (count < ${props.capacity}u) { let output = cellsOffset + (index * ${props.capacity}u + count) * 2u; cells[output] = cell.y; cells[output + 1u] = cell.x; }
      count += 1u;
    }
  }
}
for (var slot = min(count, ${props.capacity}u); slot < ${props.capacity}u; slot++) { let output = cellsOffset + (index * ${props.capacity}u + slot) * 2u; cells[output] = 0u; cells[output + 1u] = 0u; }
if (count > ${props.capacity}u) { failure |= 2u; }
counts[countsOffset + index * 2u] = min(count, ${props.capacity}u); counts[countsOffset + index * 2u + 1u] = count; status[statusOffset + index] = failure;`
      })
    ];
  }
}
