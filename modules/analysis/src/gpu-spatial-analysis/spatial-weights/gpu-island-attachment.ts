// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  describeGPUSpatialWeights,
  validateGPUSpatialWeights,
  validateGPUSpatialWeightsOutput,
  getGPUSpatialWeightsViews,
  getGPUSpatialWeightsStatusViews
} from './spatial-weights';

export type GPUIslandAttachmentProps = {
  id?: string;
  weights: GPUSpatialWeights;
  /** Planar point representatives, e.g. explicitly chosen polygon centroids. */
  positions: GraphDataView<'float32x2'>;
  /** Add reciprocal edges by default. False adds only island -> nearest. */
  reciprocal?: boolean;
  output: GPUSpatialWeightsOutput & {
    /** Per row: selected nearest row, or 0xffffffff when no attachment was needed/possible. */
    attachment: GraphDataView<'uint32'>;
    /** Per written edge: 1 added by this operation, 0 copied. */
    added: GraphDataView<'uint32'>;
  };
};

/**
 * Attach original empty CSR rows to their nearest finite planar representative (ties: lowest ID),
 * retaining original weights and adding binary edges. Existing CSR must satisfy sorted-unique,
 * finite positive off-diagonal input invariants; zero-weight-only rows are not treated as empty.
 * Selection is simultaneous against the ORIGINAL graph, so two islands can attach to each other.
 * Output is sorted, unique and deterministic under truncation. Never silently row-standardizes.
 * Nonfinite representatives cannot acquire an attachment, but existing edges are preserved.
 */
export class GPUIslandAttachment implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  constructor(readonly props: GPUIslandAttachmentProps) {
    this.id = props.id ?? 'island-attachment';
    const rows = validateGPUSpatialWeights(this.id, props.weights);
    validateCapacity(this.id, 'rows', rows, 1, 65535);
    if (
      validateGPUSpatialWeightsOutput(this.id, props.output) !== rows ||
      props.positions.length !== rows ||
      props.output.attachment.length !== rows ||
      props.output.added.length !== props.output.neighbors.length
    )
      throw new Error(`${this.id}: shape mismatch`);
    if (props.output.distances) throw new Error(`${this.id}: distances output is not supported`);
    validatePackedView(props.positions, ['float32x2'], this.id);
    validatePackedView(props.output.attachment, ['uint32'], this.id);
    validatePackedView(props.output.added, ['uint32'], this.id);
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...getGPUSpatialWeightsViews(props.output),
        ...getGPUSpatialWeightsStatusViews(props.output),
        props.output.attachment,
        props.output.added
      ],
      [...getGPUSpatialWeightsViews(props.weights), props.positions]
    );
    this.outputWeights = describeGPUSpatialWeights(props.output, {
      information: {
        rowCount: rows,
        transform: 'raw',
        symmetric:
          props.reciprocal !== false && props.weights.information?.symmetric === true
            ? true
            : 'unknown',
        hasSelfWeights: false,
        provenance: {source: 'GPUIslandAttachment', criterion: 'nearest-original-island'}
      },
      status: props.output.status
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.positions.length;
    const capacity = props.output.neighbors.length;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rows);
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'u32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const node = (
      name: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUIslandAttachment',
        variant: name,
        invocationCount,
        bindings,
        declarations,
        body
      });
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsViews(props.weights),
      props.positions,
      ...getGPUSpatialWeightsViews(props.output),
      ...getGPUSpatialWeightsStatusViews(props.output),
      props.output.attachment,
      props.output.added
    ]);
    const existingBindings = [
      binding('offsets', props.weights.offsets),
      binding('neighbors', props.weights.neighbors),
      binding('attachment', props.output.attachment)
    ];
    const edgeFunction = `fn addedEdge(row: u32, other: u32) -> bool { return row != other && (attachment[attachmentOffset + row] == other ${props.reciprocal === false ? '' : '|| attachment[attachmentOffset + other] == row'}); }
fn findEdge(row: u32, other: u32) -> u32 {
 var begin = offsets[offsetsOffset + row]; var end = offsets[offsetsOffset + row + 1u];
 while (begin < end) { let middle = begin + (end - begin) / 2u; let candidate = neighbors[neighborsOffset + middle]; if (candidate == other) { return middle; } if (candidate < other) { begin = middle + 1u; } else { end = middle; } }
 return 0xffffffffu;
}`;
    return [
      node(
        'nearest',
        rows,
        [
          binding('positions', props.positions, 'read', 'f32'),
          binding('offsets', props.weights.offsets),
          binding('attachment', props.output.attachment, 'read_write')
        ],
        `var nearest = 0xffffffffu; var bestDifference = vec2<f32>(0.0);
let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
if (offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u] && isFiniteBits(position.x) && isFiniteBits(position.y)) {
 for (var other = 0u; other < ${rows}u; other++) { let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
 if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
 // Halving before subtraction prevents overflow for opposite finite coordinates. Compare both
 // distances at their shared scale so squared norms neither overflow nor collapse to zero.
 // Keep the scale multiplier normal even on devices that flush subnormal factors.
 let difference = position * 0.5 - candidate * 0.5;
 let scale = max(max(abs(difference.x), abs(difference.y)), max(abs(bestDifference.x), abs(bestDifference.y)));
 var closer = nearest == 0xffffffffu;
 if (scale > 0.0) { let exponent = clamp(i32((bitcast<u32>(scale) >> 23u) & 255u) - 127, -120, 120); let normalizer = bitcast<f32>(u32(127 - exponent) << 23u); let normalized = difference * normalizer; let bestNormalized = bestDifference * normalizer; closer = closer || dot(normalized, normalized) < dot(bestNormalized, bestNormalized); }
 if (closer) { bestDifference = difference; nearest = other; }
 }
 }
}
attachment[attachmentOffset + index] = nearest;`,
        IS_FINITE_BITS_WGSL
      ),
      node(
        'counts',
        rows,
        [...existingBindings, binding('counts', counts, 'read_write')],
        `var count = 0u;
for (var other = 0u; other < ${rows}u; other++) { if (findEdge(index, other) != 0xffffffffu || addedEdge(index, other)) { count++; } }
counts[countsOffset + index] = count;`,
        edgeFunction
      ),
      node(
        'offsets',
        1,
        [
          binding('counts', counts),
          binding('outputOffsets', props.output.offsets, 'read_write'),
          binding('overflow', props.output.status.overflow, 'read_write'),
          ...(props.output.status.requiredCount
            ? [binding('required', props.output.status.requiredCount, 'read_write')]
            : [])
        ],
        `var total = 0u;
for (var row = 0u; row < ${rows}u; row++) { outputOffsets[outputOffsetsOffset + row] = min(total, ${capacity}u); total += counts[countsOffset + row]; }
outputOffsets[outputOffsetsOffset + ${rows}u] = min(total, ${capacity}u);
overflow[overflowOffset] = select(0u, 1u, total > ${capacity}u);
${props.output.status.requiredCount ? 'required[requiredOffset] = total;' : ''}`
      ),
      node(
        'write',
        rows,
        [
          ...existingBindings,
          binding('weights', props.weights.weights, 'read', 'f32'),
          binding('outputOffsets', props.output.offsets),
          binding('outputNeighbors', props.output.neighbors, 'read_write'),
          binding('outputWeights', props.output.weights, 'read_write', 'f32'),
          binding('added', props.output.added, 'read_write')
        ],
        `var destination = outputOffsets[outputOffsetsOffset + index]; let end = outputOffsets[outputOffsetsOffset + index + 1u];
for (var other = 0u; other < ${rows}u && destination < end; other++) { let original = findEdge(index, other);
 if (original != 0xffffffffu || addedEdge(index, other)) { var weight = 1.0; if (original != 0xffffffffu) { weight = weights[weightsOffset + original]; }
 outputNeighbors[outputNeighborsOffset + destination] = other; outputWeights[outputWeightsOffset + destination] = weight; added[addedOffset + destination] = select(0u, 1u, original == 0xffffffffu); destination++; }
}`,
        edgeFunction
      )
    ];
  }
}
