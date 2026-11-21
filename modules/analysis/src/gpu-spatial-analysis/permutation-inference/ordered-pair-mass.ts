// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/**
 * Ordered off-diagonal mass in scaled units. Each binary merge combines disjoint row sets:
 * pair(A union B) = pair(A) + pair(B) + 2 sum(A) sum(B).
 * No nearly equal moments are subtracted; each row-sized binding stays at four bytes per row.
 * @internal
 */
export function getOrderedPairMassNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rows: number;
    bindings: WGSLKernelBinding[];
    value: string;
    declarations?: string;
    destination: WGSLKernelBinding;
    write: string;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rows} = props;
  let sums = createTransientView(graph, `${id}-sums`, 'float32', rows);
  let pairs = createTransientView(graph, `${id}-pairs`, 'float32', rows);
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-values`,
      operation,
      invocationCount: rows,
      bindings: [
        ...props.bindings,
        {name: 'sums', view: sums, type: 'f32', access: 'read_write'},
        {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'}
      ],
      declarations: props.declarations,
      body: `sums[sumsOffset + index] = ${props.value}; pairs[pairsOffset + index] = 0.0;`
    })
  ];
  for (let count = rows, level = 0; count > 1; level++) {
    const nextCount = Math.ceil(count / 2);
    const nextSums = createTransientView(graph, `${id}-sums-${level}`, 'float32', nextCount);
    const nextPairs = createTransientView(graph, `${id}-pairs-${level}`, 'float32', nextCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge-${level}`,
        operation,
        invocationCount: nextCount,
        bindings: [
          {name: 'sums', view: sums, type: 'f32', access: 'read'},
          {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
          {name: 'nextSums', view: nextSums, type: 'f32', access: 'read_write'},
          {name: 'nextPairs', view: nextPairs, type: 'f32', access: 'read_write'}
        ],
        body: `let left = index * 2u;
  let leftSum = sums[sumsOffset + left];
  var rightSum = 0.0; var rightPairs = 0.0;
  if (left + 1u < ${count}u) { rightSum = sums[sumsOffset + left + 1u]; rightPairs = pairs[pairsOffset + left + 1u]; }
  nextSums[nextSumsOffset + index] = leftSum + rightSum;
  nextPairs[nextPairsOffset + index] = pairs[pairsOffset + left] + rightPairs + 2.0 * leftSum * rightSum;`
      })
    );
    sums = nextSums;
    pairs = nextPairs;
    count = nextCount;
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-store`,
      operation,
      invocationCount: 1,
      bindings: [{name: 'pairs', view: pairs, type: 'f32', access: 'read'}, props.destination],
      body: `let pairMass = pairs[pairsOffset]; ${props.write}`
    })
  );
  return nodes;
}
