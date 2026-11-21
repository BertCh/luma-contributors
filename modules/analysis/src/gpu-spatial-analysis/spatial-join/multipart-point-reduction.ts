// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUSort,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode, createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Distinct owner/polygon counting uses two stable uint32 sorts, never a dense cross product. @internal */
export function getMultipartPointCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    owners: GraphDataView<'uint32'>;
    polygons: GraphDataView<'uint32'>;
    queryCount: number;
    polygonCount: number;
    polygonMembershipCounts?: GraphDataView<'uint32'>;
    pointMatchCounts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, owners, polygons} = props;
  const sortedPolygons = createTransientView(graph, `${id}-polygon-sort`, 'uint32', owners.length),
    polygonOwners = createTransientView(graph, `${id}-polygon-owners`, 'uint32', owners.length);
  const sortedOwners = createTransientView(graph, `${id}-owner-sort`, 'uint32', owners.length),
    ownerPolygons = createTransientView(graph, `${id}-owner-polygons`, 'uint32', owners.length);
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (props.pointMatchCounts)
    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear-match-counts`,
        operation: 'GPUPointInPolygonJoin',
        view: props.pointMatchCounts,
        type: 'u32',
        value: '0u'
      })
    );
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort-polygons`,
      keys: polygons,
      values: owners,
      outputKeys: sortedPolygons,
      outputValues: polygonOwners
    }).getCommandNodes(graph),
    ...new GPUSort({
      id: `${id}-sort-owners`,
      keys: polygonOwners,
      values: sortedPolygons,
      outputKeys: sortedOwners,
      outputValues: ownerPolygons
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-count-distinct`,
      operation: 'GPUPointInPolygonJoin',
      bindings: [
        {name: 'owners', view: sortedOwners, type: 'u32', access: 'read'},
        {name: 'polygons', view: ownerPolygons, type: 'u32', access: 'read'},
        ...(props.polygonMembershipCounts
          ? [
              {
                name: 'counts',
                view: props.polygonMembershipCounts,
                type: 'atomic<u32>' as const,
                access: 'read_write' as const
              }
            ]
          : []),
        ...(props.pointMatchCounts
          ? [
              {
                name: 'matches',
                view: props.pointMatchCounts,
                type: 'atomic<u32>' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: owners.length,
      body: `let owner = owners[ownersOffset + index]; let polygon = polygons[polygonsOffset + index];
  if (owner >= ${props.queryCount}u || polygon >= ${props.polygonCount}u) { return; }
  if (index > 0u && owners[ownersOffset + index - 1u] == owner && polygons[polygonsOffset + index - 1u] == polygon) { return; }
  ${props.polygonMembershipCounts ? 'atomicAdd(&counts[countsOffset + polygon], 1u);' : ''}
  ${props.pointMatchCounts ? 'atomicAdd(&matches[matchesOffset + owner], 1u);' : ''}`
    })
  );
  return nodes;
}
