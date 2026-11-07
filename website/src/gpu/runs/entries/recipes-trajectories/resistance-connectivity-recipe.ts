// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addResistanceConnectivityRecipe on a 16 by 16 resistance grid with a barrier wall that has one
// gap. Nodes and edges are colored by the electrical potential from source (left) to sink (right):
// the potential drops across the wall except through the gap, where current concentrates.

import {addResistanceConnectivityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addReadout} from './lib/readout';
import {parameters} from './resistance-connectivity-recipe.parameters';

const SIDE = 16;

export default defineOperationRun({
  parameters,
  caption: 'Potential across a resistance grid with a barrier wall.',
  createInputs: ({world}) => {
    const positions = Float32Array.from({length: SIDE * SIDE * 2}, (_, index) => {
      const node = Math.floor(index / 2);
      const column = node % SIDE;
      const row = Math.floor(node / SIDE);
      return index % 2 === 0
        ? (0.08 + 0.84 * (column / (SIDE - 1))) * world.width
        : 0.08 + 0.84 * (row / (SIDE - 1));
    });
    const edges: number[] = [];
    for (let row = 0; row < SIDE; row++) {
      for (let column = 0; column < SIDE; column++) {
        const node = row * SIDE + column;
        if (column < SIDE - 1) edges.push(node, node + 1);
        if (row < SIDE - 1) edges.push(node, node + SIDE);
      }
    }
    return {positions, edges: Uint32Array.from(edges)};
  },
  build(builder, {positions, edges}, values) {
    const edgeCount = edges.length / 2;
    const conductances = builder.allocate('conductances', 'float32', edgeCount);
    const recipe = addResistanceConnectivityRecipe(builder.graph, {
      edges: builder.upload('edges', 'uint32x2', edges).view,
      conductances: conductances.view,
      vertexCount: SIDE * SIDE,
      source: values.source * SIDE,
      sink: values.sink * SIDE + SIDE - 1,
      maximumIterations: 512
    });
    return {
      outputs: {
        potentials: addReadout(builder, 'potentials', recipe.outputs.potentials),
        nodes: builder.upload('nodes', 'float32x2', positions),
        edges: builder.upload('edge-pairs', 'uint32x2', edges)
      },
      update: current =>
        conductances.write(
          Float32Array.from({length: edgeCount}, (_, edge) => {
            const column = Math.max(edges[edge * 2] % SIDE, edges[edge * 2 + 1] % SIDE);
            const row = Math.floor(edges[edge * 2] / SIDE);
            // A wall at the middle columns, open between rows 6 and 9.
            const isWall = column >= 8 && column <= 8 && (row < 6 || row > 9);
            return isWall ? 1 - current.barrier : 1;
          })
        )
    };
  },
  rebuildOn: ['source', 'sink'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.potentials,
      radius: 3.5,
      range: [0, 1]
    },
    {
      layer: 'dots',
      positions: outputs.nodes,
      values: outputs.potentials,
      radius: 5.5,
      range: [0, 1]
    }
  ]
});
