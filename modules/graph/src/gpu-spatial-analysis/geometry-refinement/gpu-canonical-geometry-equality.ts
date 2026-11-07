// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPUCanonicalGeometryTree} from './gpu-normalize-geometry';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

export type GPUCanonicalGeometryEqualityProps = {
  id?: string;
  left: GPUCanonicalGeometryTree;
  right: GPUCanonicalGeometryTree;
  /** Root node IDs in the left and right normalized trees. Only successful normalization may be compared. */
  pairs: GraphDataView<'uint32x2'>;
  tolerance?: number;
  /** Per pair: equal=1, unequal=0, nonfinite/invalid traversal=2. Source IDs are not geometry. */
  output: GraphDataView<'uint32'>;
};

/** Structural equality of normalized geometry trees, including nested empty collections. */
export class GPUCanonicalGeometryEquality implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUCanonicalGeometryEqualityProps;
  constructor(props: GPUCanonicalGeometryEqualityProps) {
    this.id = props.id ?? 'canonical-geometry-equality';
    this.props = props;
    for (const tree of [props.left, props.right]) {
      validatePackedView(tree.positions, ['float32x2'], this.id);
      validatePackedView(tree.nodes, ['uint32x4'], this.id);
      validatePackedView(tree.children, ['uint32'], this.id);
    }
    validatePackedView(props.pairs, ['uint32x2'], this.id);
    validatePackedView(props.output, ['uint32'], this.id);
    if (
      props.output.length !== props.pairs.length ||
      !(Number.isFinite(props.tolerance ?? 0) && (props.tolerance ?? 0) >= 0)
    )
      throw new Error('Invalid canonical equality contract');
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
    validateGraphViewsBelongToGraph(id, graph, [
      ...Object.values(props.left),
      ...Object.values(props.right),
      props.pairs,
      props.output
    ]);
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
      if (abs(leftValue - rightValue) > ${getWGSLFloatLiteral(props.tolerance ?? 0)}) { code = 0u; }
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
