// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAdjacencyMatrix: the network as a density matrix, rows and columns are nodes in `order`, each
// cell counts the edges between two bins of nodes. Grouped order exposes the communities as blocks.

import {GPUAdjacencyMatrix, GPUAdjacencyMatrixOrder} from '@luma-contributors/analysis/gpu-network';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createCommunityGraph, uploadAdjacency} from './shared/graphs';
import {parameters} from './adjacency-matrix.parameters';

const RESOLUTIONS = [32, 64, 128];

export default defineOperationRun({
  parameters,
  caption: 'Edge counts between node bins; communities become blocks when nodes are grouped.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    // A fresh seeded stream per build keeps every rebuild of the same settings identical.
    const graph = createCommunityGraph(makeRandom(seed), world, Math.round(values.communities));
    const resolution = RESOLUTIONS[values.resolution];
    const order = builder.allocate('order', 'uint32', graph.nodeCount);
    if (values.order === 1) {
      builder.graph.add(
        new GPUAdjacencyMatrixOrder({
          groups: builder.upload('groups', 'uint32', graph.groups).view,
          output: {order: order.view}
        })
      );
    } else {
      order.write(Uint32Array.from({length: graph.nodeCount}, (_, node) => node));
    }
    const counts = builder.allocate('counts', 'uint32', resolution * resolution);
    const maximumCount = builder.allocate('maximum-count', 'uint32', 1);
    builder.graph.add(
      new GPUAdjacencyMatrix({
        topology: uploadAdjacency(builder, graph),
        directed: true,
        order: order.view,
        resolution,
        output: {counts: counts.view, maximumCount: maximumCount.view}
      })
    );
    return {outputs: {counts}};
  },
  rebuildOn: ['order', 'resolution', 'communities'],
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.counts,
      columns: RESOLUTIONS[values.resolution],
      rows: RESOLUTIONS[values.resolution],
      ramp: 'sequential',
      range: [0, 6]
    }
  ]
});
