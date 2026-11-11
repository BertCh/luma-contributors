// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAdjacencyMatrix: the network as a density matrix, rows and columns are nodes in `order`, each
// cell counts the edges between two bins of nodes. Grouped order exposes the communities as blocks.

import {GPUAdjacencyMatrix, GPUAdjacencyMatrixOrder} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {createMatrixNote, createBandText} from './shared/band-text';
import {createBandedGraph, uploadAdjacency} from './shared/graphs';
import {parameters} from './adjacency-matrix.parameters';

const RESOLUTIONS = [32, 64, 128];

export default defineOperationRun({
  parameters,
  caption:
    'Chicago commuting links between tracts (either direction, most links between bands left out), grouped into north-to-south bands.',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
    const graph = createBandedGraph(Math.round(values.communities));
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
    return {outputs: {counts, nodeCount: graph.nodeCount, graph}};
  },
  rebuildOn: ['order', 'resolution', 'communities'],
  // Probe: the cell under the pointer is a block of tract pairs; its row and column bands are outlined.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.66, 0.34],
    resolve(pointer, {outputs, values, world, peek}) {
      const resolution = RESOLUTIONS[values.resolution];
      const bounds = [0, 0, world.width, 1] as const;
      const cell = getRasterCell(resolution, resolution, bounds, pointer.x, pointer.y);
      const [centerX, centerY] = getRasterCellCenter(
        resolution,
        resolution,
        bounds,
        cell.column,
        cell.row
      );
      const halfWidth = world.width / resolution / 2;
      const halfHeight = 1 / resolution / 2;
      const perCell = outputs.nodeCount / resolution;
      const counts = peek(outputs.counts);
      const span = (index: number) =>
        `${Math.floor(index * perCell)}-${Math.floor((index + 1) * perCell) - 1}`;
      return {
        spotlight: null,
        glyphs: [
          {
            shape: 'box',
            minimum: [0, centerY - halfHeight],
            maximum: [world.width, centerY + halfHeight],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [centerX - halfWidth, 0],
            maximum: [centerX + halfWidth, 1],
            dashed: true
          },
          {
            shape: 'box',
            minimum: [centerX - halfWidth, centerY - halfHeight],
            maximum: [centerX + halfWidth, centerY + halfHeight]
          }
        ],
        readout: `rows ${span(cell.row)} x cols ${span(cell.column)} · ${counts ? counts[cell.index] : 0} links`
      };
    }
  },
  draw: ({outputs, values, world}) => [
    {
      layer: 'raster',
      values: outputs.counts,
      columns: RESOLUTIONS[values.resolution],
      rows: RESOLUTIONS[values.resolution],
      ramp: 'sequential',
      range: [0, 3]
    },
    {
      layer: 'text',
      ...(values.order === 1
        ? createBandText(outputs.graph, world, Math.round(values.communities))
        : createMatrixNote())
    }
  ]
});
