// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAdjacencyMatrixOrder: a node order that groups communities (then sorts by a tie key such as
// degree). The matrix shows the effect: with this order the communities line up as diagonal blocks.

import {GPUAdjacencyMatrix, GPUAdjacencyMatrixOrder} from '@luma-contributors/analysis/gpu-network';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCommunityGraph, uploadAdjacency} from './shared/graphs';
import {parameters} from './adjacency-matrix-order.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Adjacency matrix drawn in the order the operation produced.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    const graph = createCommunityGraph(makeRandom(seed), world, Math.round(values.communities));
    const order = builder.allocate('order', 'uint32', graph.nodeCount);
    builder.graph.add(
      new GPUAdjacencyMatrixOrder({
        groups: builder.upload('groups', 'uint32', graph.groups).view,
        tieKeys: builder.upload('degrees', 'uint32', graph.degrees).view,
        output: {order: order.view}
      })
    );
    const counts = builder.allocate('counts', 'uint32', values.resolution * values.resolution);
    builder.graph.add(
      new GPUAdjacencyMatrix({
        topology: uploadAdjacency(builder, graph),
        directed: true,
        order: order.view,
        resolution: values.resolution,
        output: {counts: counts.view}
      })
    );
    return {outputs: {counts, resolution: values.resolution}};
  },
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.counts,
      columns: outputs.resolution,
      rows: outputs.resolution,
      ramp: 'sequential',
      range: [0, 6]
    }
  ]
});
