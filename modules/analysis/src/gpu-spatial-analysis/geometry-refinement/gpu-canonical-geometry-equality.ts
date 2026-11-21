// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRefinementShaderNumber} from './refinement-storage-validation';
import {
  getCanonicalEqualityMetricWGSL,
  validateCanonicalEqualityMetric
} from './canonical-equality-metric';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCanonicalGeometryTree} from './gpu-normalize-geometry';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUCanonicalGeometryEqualityProps = {
  id?: string;
  left: GPUCanonicalGeometryTree;
  right: GPUCanonicalGeometryTree;
  /** Root node IDs in the left and right normalized trees. Only successful normalization may be compared. */
  pairs: GraphDataView<'uint32x2'>;
  /** Absolute tolerance in normalized coordinate units; default zero. */
  tolerance?: number;
  /** Distance metric in normalized coordinate units. Default: per-coordinate (legacy behavior). */
  toleranceMetric?: 'per-coordinate' | 'euclidean';
  /** Per pair: equal=1, unequal=0, nonfinite/invalid traversal=2. Source IDs are not geometry. */
  output: GraphDataView<'uint32'>;
};

/** Structural equality of normalized geometry trees, including nested empty collections. */
export class GPUCanonicalGeometryEquality implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUCanonicalGeometryEqualityProps;
  constructor(props: GPUCanonicalGeometryEqualityProps) {
    this.id = props.id ?? 'canonical-geometry-equality';
    props = {...props, left: {...props.left}, right: {...props.right}};
    this.props = props;
    validateCanonicalEqualityMetric(this.id, props.toleranceMetric);
    validateRefinementShaderNumber(this.id, 'tolerance', props.tolerance ?? 0);
    for (const tree of [props.left, props.right]) {
      validateSinglePackedView(tree.positions, ['float32x2'], this.id);
      validateSinglePackedView(tree.nodes, ['uint32x4'], this.id);
      validateSinglePackedView(tree.children, ['uint32'], this.id);
    }
    validateSinglePackedView(props.pairs, ['uint32x2'], this.id);
    validateSinglePackedView(props.output, ['uint32'], this.id);
    validateRequirement(
      this.id,
      props.output.length === props.pairs.length,
      `output.length must equal pairs.length (${props.pairs.length})`
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output],
      [...Object.values(props.left), ...Object.values(props.right), props.pairs]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          ...Object.values(props.left),
          ...Object.values(props.right),
          props.pairs,
          props.output
        ].map((view, index) => [`binding.${index}`, view])
      )
    );
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUCanonicalGeometryEquality',
        invocationCount: props.pairs.length,
        bindings: [
          {name: 'leftPositions', view: props.left.positions, type: 'f32', access: 'read'},
          {name: 'leftNodes', view: props.left.nodes, type: 'u32', access: 'read'},
          {name: 'leftChildren', view: props.left.children, type: 'u32', access: 'read'},
          {name: 'rightPositions', view: props.right.positions, type: 'f32', access: 'read'},
          {name: 'rightNodes', view: props.right.nodes, type: 'u32', access: 'read'},
          {name: 'rightChildren', view: props.right.children, type: 'u32', access: 'read'},
          {name: 'pairs', view: props.pairs, type: 'u32', access: 'read'},
          {name: 'result', view: props.output, type: 'u32', access: 'read_write'}
        ],
        declarations: getCanonicalEqualityMetricWGSL(props.tolerance ?? 0, props.toleranceMetric),
        body: `var left = pairs[pairsOffset + index * 2u]; var right = pairs[pairsOffset + index * 2u + 1u];
var leftStack: array<u32, 64>; var rightStack: array<u32, 64>; var cursorStack: array<u32, 64>;
var depth = 0u; var code = 1u; var visits = 0u;
loop {
  visits++;
  if (left >= ${props.left.nodes.length}u || right >= ${props.right.nodes.length}u || visits > ${props.left.nodes.length + props.right.nodes.length + 1}u) { code = 2u; break; }
  let leftKind = leftNodes[leftNodesOffset + left * 4u]; let rightKind = rightNodes[rightNodesOffset + right * 4u];
  let leftFirst = leftNodes[leftNodesOffset + left * 4u + 1u]; let rightFirst = rightNodes[rightNodesOffset + right * 4u + 1u];
  let leftCount = leftNodes[leftNodesOffset + left * 4u + 2u]; let rightCount = rightNodes[rightNodesOffset + right * 4u + 2u];
  if (leftKind != rightKind || leftCount != rightCount) { code = 0u; break; }
  if (leftKind == 0u || leftKind == 2u || leftKind == 3u) {
    if (leftFirst > ${props.left.positions.length}u || leftCount > ${props.left.positions.length}u - leftFirst || rightFirst > ${props.right.positions.length}u || rightCount > ${props.right.positions.length}u - rightFirst) { code = 2u; break; }
    for (var coordinate = 0u; coordinate < leftCount * 2u; coordinate++) {
      let leftValue = leftPositions[leftPositionsOffset + leftFirst * 2u + coordinate]; let rightValue = rightPositions[rightPositionsOffset + rightFirst * 2u + coordinate];
      if ((bitcast<u32>(leftValue) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(rightValue) & 0x7f800000u) == 0x7f800000u) { code = 2u; break; }
      if (coordinate % 2u == 0u && !withinTolerance(vec2f(leftValue,leftPositions[leftPositionsOffset + leftFirst * 2u + coordinate + 1u]),vec2f(rightValue,rightPositions[rightPositionsOffset + rightFirst * 2u + coordinate + 1u]))) { code = 0u; }
    }
    if (code != 1u) { break; }
  } else {
    if (leftKind > 7u || leftFirst > ${props.left.children.length}u || leftCount > ${props.left.children.length}u - leftFirst || rightFirst > ${props.right.children.length}u || rightCount > ${props.right.children.length}u - rightFirst) { code = 2u; break; }
    if (leftCount > 0u) {
      if (depth == 64u) { code = 2u; break; }
      leftStack[depth] = left; rightStack[depth] = right; cursorStack[depth] = 1u; depth++;
      left = leftChildren[leftChildrenOffset + leftFirst]; right = rightChildren[rightChildrenOffset + rightFirst]; continue;
    }
  }
  var advanced = false;
  loop {
    if (depth == 0u) { break; }
    let leftParent = leftStack[depth - 1u]; let rightParent = rightStack[depth - 1u]; let cursor = cursorStack[depth - 1u];
    if (cursor < leftNodes[leftNodesOffset + leftParent * 4u + 2u]) {
      cursorStack[depth - 1u]++; left = leftChildren[leftChildrenOffset + leftNodes[leftNodesOffset + leftParent * 4u + 1u] + cursor]; right = rightChildren[rightChildrenOffset + rightNodes[rightNodesOffset + rightParent * 4u + 1u] + cursor]; advanced = true; break;
    }
    depth--;
  }
  if (!advanced) { break; }
}
result[resultOffset + index] = code;`
      })
    ];
  }
}
