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
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

export type GPUCanonicalPolygonEqualityProps = {
  id?: string;
  left: GPUPolygonGeometryFields;
  right: GPUPolygonGeometryFields;
  /** Absolute per-coordinate tolerance in the normalized coordinate space, default zero. */
  tolerance?: number;
  /** Equal=1, unequal=0, nonfinite=2 per pair of normalized features. */
  output: GraphDataView<'uint32'>;
};

/** Structural coordinate equality for GPUNormalizePolygons outputs, without hashing. */
export class GPUCanonicalPolygonEquality implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUCanonicalPolygonEqualityProps;
  constructor(props: GPUCanonicalPolygonEqualityProps) {
    this.id = props.id ?? 'canonical-polygon-equality';
    this.props = props;
    if (!(Number.isFinite(props.tolerance ?? 0) && (props.tolerance ?? 0) >= 0)) {
      throw new Error('Invalid equality tolerance');
    }
    for (const geometry of [props.left, props.right]) {
      validateSinglePackedView(geometry.positions, ['float32x2'], this.id);
      for (const offsets of [
        geometry.ringOffsets,
        geometry.polygonOffsets,
        geometry.featureOffsets
      ]) {
        validateSinglePackedView(offsets, ['uint32'], this.id);
      }
    }
    validateSinglePackedView(props.output, ['uint32'], this.id);
    if (
      props.left.featureOffsets.length !== props.right.featureOffsets.length ||
      props.output.length !== props.left.featureOffsets.length - 1
    ) {
      throw new Error('Canonical equality feature count mismatch');
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output],
      [...Object.values(props.left), ...Object.values(props.right)]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...Object.values(props.left),
      ...Object.values(props.right),
      props.output
    ]);
    const spans = createTransientView(graph, `${id}-spans`, 'uint32x4', props.output.length);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-structure`,
        operation: 'GPUCanonicalPolygonEquality',
        variant: 'structure',
        invocationCount: props.output.length,
        bindings: [
          {name: 'leftFeatures', view: props.left.featureOffsets, type: 'u32', access: 'read'},
          {name: 'rightFeatures', view: props.right.featureOffsets, type: 'u32', access: 'read'},
          {name: 'leftPolygons', view: props.left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'rightPolygons', view: props.right.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'leftRings', view: props.left.ringOffsets, type: 'u32', access: 'read'},
          {name: 'rightRings', view: props.right.ringOffsets, type: 'u32', access: 'read'},
          {name: 'spans', view: spans, type: 'u32', access: 'read_write'}
        ],
        body: `let leftFirst = leftFeatures[leftFeaturesOffset + index]; let rightFirst = rightFeatures[rightFeaturesOffset + index];
let leftCount = leftFeatures[leftFeaturesOffset + index + 1u] - leftFirst; let rightCount = rightFeatures[rightFeaturesOffset + index + 1u] - rightFirst;
var equal = leftCount == rightCount;
for (var polygon = 0u; polygon < min(leftCount, rightCount); polygon++) {
  let leftRing = leftPolygons[leftPolygonsOffset + leftFirst + polygon]; let rightRing = rightPolygons[rightPolygonsOffset + rightFirst + polygon];
  let leftLength = leftPolygons[leftPolygonsOffset + leftFirst + polygon + 1u] - leftRing; let rightLength = rightPolygons[rightPolygonsOffset + rightFirst + polygon + 1u] - rightRing;
  if (leftLength != rightLength) { equal = false; }
  for (var ring = 0u; ring < min(leftLength, rightLength); ring++) {
    if (leftRings[leftRingsOffset + leftRing + ring + 1u] - leftRings[leftRingsOffset + leftRing + ring] != rightRings[rightRingsOffset + rightRing + ring + 1u] - rightRings[rightRingsOffset + rightRing + ring]) { equal = false; }
  }
}
let leftStart = leftRings[leftRingsOffset + leftPolygons[leftPolygonsOffset + leftFirst]];
let rightStart = rightRings[rightRingsOffset + rightPolygons[rightPolygonsOffset + rightFirst]];
let leftEnd = leftRings[leftRingsOffset + leftPolygons[leftPolygonsOffset + leftFirst + leftCount]];
spans[spansOffset + index * 4u] = leftStart; spans[spansOffset + index * 4u + 1u] = rightStart;
spans[spansOffset + index * 4u + 2u] = leftEnd - leftStart; spans[spansOffset + index * 4u + 3u] = select(0u, 1u, equal);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-coordinates`,
        operation: 'GPUCanonicalPolygonEquality',
        variant: 'coordinates',
        invocationCount: props.output.length,
        bindings: [
          {name: 'left', view: props.left.positions, type: 'f32', access: 'read'},
          {name: 'right', view: props.right.positions, type: 'f32', access: 'read'},
          {name: 'spans', view: spans, type: 'u32', access: 'read'},
          {name: 'output', view: props.output, type: 'u32', access: 'read_write'}
        ],
        body: `var result = spans[spansOffset + index * 4u + 3u];
if (result == 1u) {
  let leftStart = spans[spansOffset + index * 4u] * 2u; let rightStart = spans[spansOffset + index * 4u + 1u] * 2u;
  for (var component = 0u; component < spans[spansOffset + index * 4u + 2u] * 2u; component++) {
    let leftValue = left[leftOffset + leftStart + component]; let rightValue = right[rightOffset + rightStart + component];
    if (((bitcast<u32>(leftValue) & 0x7f800000u) == 0x7f800000u) || ((bitcast<u32>(rightValue) & 0x7f800000u) == 0x7f800000u)) { result = 2u; break; }
    if (abs(leftValue - rightValue) > ${getWGSLFloatLiteral(props.tolerance ?? 0)}) { result = 0u; }
  }
}
output[outputOffset + index] = result;`
      })
    ];
  }
}
