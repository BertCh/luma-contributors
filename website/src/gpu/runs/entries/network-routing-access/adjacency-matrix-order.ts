// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAdjacencyMatrixOrder: a node order that groups communities (then sorts by a tie key such as
// degree). The matrix shows the effect: with this order the communities line up as diagonal blocks.

import {GPUAdjacencyMatrix, GPUAdjacencyMatrixOrder} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createBandText, createMatrixNote} from './shared/band-text';
import {createBandedGraph, uploadAdjacency} from './shared/graphs';
import {parameters} from './adjacency-matrix-order.parameters';

export default defineOperationRun({
  parameters,
  caption:
    'Adjacency matrix in input order (left of the curtain) and in the order the operation produced (right).',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
    const graph = createBandedGraph(Math.round(values.communities));
    const topology = uploadAdjacency(builder, graph);
    const order = builder.allocate('order', 'uint32', graph.nodeCount);
    builder.graph.add(
      new GPUAdjacencyMatrixOrder({
        groups: builder.upload('groups', 'uint32', graph.groups).view,
        tieKeys: builder.upload('degrees', 'uint32', graph.degrees).view,
        output: {order: order.view}
      })
    );
    const inputOrder = builder.upload(
      'input-order',
      'uint32',
      Uint32Array.from({length: graph.nodeCount}, (_, node) => node)
    );
    const size = values.resolution * values.resolution;
    const orderedCounts = builder.allocate('ordered-counts', 'uint32', size);
    const inputCounts = builder.allocate('input-counts', 'uint32', size);
    builder.graph.add(
      new GPUAdjacencyMatrix({
        id: 'matrix-ordered',
        topology,
        directed: true,
        order: order.view,
        resolution: values.resolution,
        output: {counts: orderedCounts.view}
      })
    );
    builder.graph.add(
      new GPUAdjacencyMatrix({
        id: 'matrix-input',
        topology,
        directed: true,
        order: inputOrder.view,
        resolution: values.resolution,
        output: {counts: inputCounts.view}
      })
    );
    return {
      outputs: {orderedCounts, inputCounts, resolution: values.resolution, graph},
      update: () => undefined
    };
  },
  rebuildOn: ['resolution', 'communities'],
  // Swipe: the curtain separates the matrix in input order from the matrix the operation ordered.
  interaction: {
    kind: 'swipe',
    hint: 'swipe the order',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `input order | grouped order (${Math.round(values.communities)} bands)`
    })
  },
  draw: ({outputs, interaction, values, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      {
        layer: 'raster',
        values: outputs.inputCounts,
        columns: outputs.resolution,
        rows: outputs.resolution,
        ramp: 'sequential',
        range: [0, 3],
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'raster',
        values: outputs.orderedCounts,
        columns: outputs.resolution,
        rows: outputs.resolution,
        ramp: 'sequential',
        range: [0, 3],
        clip: {side: 'right', x: curtain}
      },
      // Input order has no blocks to name; the grouped half outlines and numbers its bands.
      {layer: 'text', clip: {side: 'left', x: curtain}, ...createMatrixNote()},
      {
        layer: 'text',
        clip: {side: 'right', x: curtain},
        ...createBandText(outputs.graph, world, Math.round(values.communities))
      }
    ];
  }
});
