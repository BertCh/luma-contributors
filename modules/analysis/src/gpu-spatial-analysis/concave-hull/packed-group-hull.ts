// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUGroupConcaveHullPackedOutput} from './gpu-concave-hull';

/** Compacts the existing group-major temporary geometry without multiplying by group count. @internal */
export function createPackedGroupHullNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    groupCount: number;
    triangles: GraphDataView<'uint32x3'>;
    triangleValidity: GraphDataView<'uint32'>;
    edges: GraphDataView<'uint32x2'>;
    edgeValidity: GraphDataView<'uint32'>;
    groupOffsets: GraphDataView<'uint32'>;
    groupCounts: GraphDataView<'uint32'>;
    groupConverged: GraphDataView<'uint32'>;
    output: GPUGroupConcaveHullPackedOutput;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const triangleCapacity = props.triangles.length;
  for (const role of ['triangle', 'edge'] as const) {
    const isTriangle = role === 'triangle';
    const input = isTriangle ? props.triangles : props.edges;
    const validity = isTriangle ? props.triangleValidity : props.edgeValidity;
    const output = isTriangle ? props.output.triangles : props.output.boundaryEdges;
    const outputValidity = isTriangle ? props.output.triangleValidity : props.output.edgeValidity;
    const offsets = isTriangle
      ? props.output.groups.triangleOffsets
      : props.output.groups.edgeOffsets;
    const counts = isTriangle ? props.output.groups.triangleCounts : props.output.groups.edgeCounts;
    const components = isTriangle ? 3 : 2;
    const multiplier = isTriangle ? 1 : 3;
    const prefix = createTransientView(
      graph,
      `${props.id}-${role}-prefix`,
      'uint32',
      validity.length
    );
    if (validity.length > 0) {
      nodes.push(
        ...new GPUScan({
          id: `${props.id}-${role}-scan`,
          input: validity,
          output: prefix,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
    }
    const prefixDeclarations = `const CAPACITY: u32 = ${validity.length}u;
fn prefixAt(row: u32) -> u32 {
  if (CAPACITY == 0u) { return 0u; }
  if (row >= CAPACITY) { return prefix[prefixOffset + ${Math.max(validity.length - 1, 0)}u] + validity[validityOffset + ${Math.max(validity.length - 1, 0)}u]; }
  return prefix[prefixOffset + row];
}`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${props.id}-${role}-group-ranges`,
        operation: 'GPUGroupConcaveHull',
        invocationCount: props.groupCount,
        bindings: [
          {name: 'groupOffsets', view: props.groupOffsets, type: 'u32', access: 'read'},
          {name: 'groupCounts', view: props.groupCounts, type: 'u32', access: 'read'},
          {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
          {name: 'validity', view: validity, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        declarations: prefixDeclarations,
        body: `let start = min(groupOffsets[groupOffsetsOffset + index], ${triangleCapacity}u);
let end = start + min(groupCounts[groupCountsOffset + index], ${triangleCapacity}u - start);
let first = prefixAt(${multiplier}u * start);
offsets[offsetsOffset + index] = first;
counts[countsOffset + index] = prefixAt(${multiplier}u * end) - first;`
      })
    );
    if (validity.length > 0) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${props.id}-${role}-clear-packed`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: output.length,
          bindings: [
            {name: 'outputRows', view: output, type: 'u32', access: 'read_write'},
            {name: 'outputValidity', view: outputValidity, type: 'u32', access: 'read_write'}
          ],
          body: `for (var component = 0u; component < ${components}u; component++) {
  outputRows[outputRowsOffset + index * ${components}u + component] = 0u;
}
outputValidity[outputValidityOffset + index] = 0u;`
        }),
        createWGSLKernelNode(graph, {
          id: `${props.id}-${role}-compact-packed`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: validity.length,
          bindings: [
            {name: 'inputRows', view: input, type: 'u32', access: 'read'},
            {name: 'validity', view: validity, type: 'u32', access: 'read'},
            {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
            {name: 'outputRows', view: output, type: 'u32', access: 'read_write'},
            {name: 'outputValidity', view: outputValidity, type: 'u32', access: 'read_write'}
          ],
          body: `if (validity[validityOffset + index] == 0u) { return; }
let destination = prefix[prefixOffset + index];
for (var component = 0u; component < ${components}u; component++) {
  outputRows[outputRowsOffset + destination * ${components}u + component] = inputRows[inputRowsOffset + index * ${components}u + component];
}
outputValidity[outputValidityOffset + destination] = 1u;`
        })
      );
    }
    if (!isTriangle) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${props.id}-publish-packed-status`,
          operation: 'GPUGroupConcaveHull',
          invocationCount: 1,
          bindings: [
            {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
            {name: 'validity', view: validity, type: 'u32', access: 'read'},
            {name: 'count', view: props.output.status.count, type: 'u32', access: 'read_write'},
            {
              name: 'requiredCount',
              view: props.output.status.requiredCount,
              type: 'u32',
              access: 'read_write'
            },
            {
              name: 'overflow',
              view: props.output.status.overflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          declarations: prefixDeclarations,
          body: `let total = prefixAt(CAPACITY);
count[countOffset] = total;
requiredCount[requiredCountOffset] = total;
overflow[overflowOffset] = 0u;`
        })
      );
    }
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${props.id}-publish-converged`,
      operation: 'GPUGroupConcaveHull',
      invocationCount: props.groupCount,
      bindings: [
        {name: 'source', view: props.groupConverged, type: 'u32', access: 'read'},
        {name: 'converged', view: props.output.status.converged, type: 'u32', access: 'read_write'}
      ],
      body: 'converged[convergedOffset + index] = source[sourceOffset + index];'
    })
  );
  return nodes;
}
