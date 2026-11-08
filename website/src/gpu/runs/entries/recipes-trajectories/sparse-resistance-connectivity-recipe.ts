// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSparseResistanceConnectivityRecipe on a 48 by 32 lattice (1,536 vertices) with a wall of
// zero-conductance edges that has one gap. Current enters on the left and leaves at the sink: edges
// are colored by the current through each vertex, dots by the potential. A short iteration budget
// leaves the solve unconverged and the picture blotchy.

import {addSparseResistanceConnectivityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addReadout} from './lib/readout';
import {parameters} from './sparse-resistance-connectivity-recipe.parameters';

const COLUMNS = 48;
const ROWS = 32;
const WALL_COLUMN = 24;
const SOURCE_ROW = 6;
const ITERATIONS = [16, 64, 1024];

export default defineOperationRun({
  parameters,
  caption: 'Current finds the gap in a wall across a 1,536-vertex lattice.',
  createInputs: ({random, world}) => {
    const positions = new Float32Array(COLUMNS * ROWS * 2);
    for (let node = 0; node < COLUMNS * ROWS; node++) {
      positions[node * 2] = (0.06 + 0.88 * ((node % COLUMNS) / (COLUMNS - 1))) * world.width;
      positions[node * 2 + 1] = 0.06 + 0.88 * (Math.floor(node / COLUMNS) / (ROWS - 1));
    }
    const edges: number[] = [];
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const node = row * COLUMNS + column;
        if (column < COLUMNS - 1) edges.push(node, node + 1);
        if (row < ROWS - 1) edges.push(node, node + COLUMNS);
      }
    }
    const edgeCount = edges.length / 2;
    // Seeded habitat quality in [0.4, 1].
    const quality = Float32Array.from({length: edgeCount}, () => 0.4 + 0.6 * random());
    return {positions, edges: Uint32Array.from(edges), quality};
  },
  build(builder, {positions, edges, quality}, values) {
    const edgeCount = edges.length / 2;
    const conductances = builder.allocate('conductances', 'float32', edgeCount);
    const recipe = addSparseResistanceConnectivityRecipe(builder.graph, {
      edges: builder.upload('edges', 'uint32x2', edges).view,
      conductances: conductances.view,
      vertexCount: COLUMNS * ROWS,
      source: SOURCE_ROW * COLUMNS,
      sink: values.sinkRow * COLUMNS + COLUMNS - 1,
      maximumIterations: ITERATIONS[values.maximumIterations]
    });
    return {
      outputs: {
        potentials: addReadout(builder, 'potentials', recipe.outputs.potentials),
        vertexCurrents: addReadout(builder, 'vertex-currents', recipe.outputs.vertexCurrents),
        terminals: builder.upload(
          'terminals',
          'float32x2',
          Float32Array.of(
            positions[SOURCE_ROW * COLUMNS * 2],
            positions[SOURCE_ROW * COLUMNS * 2 + 1],
            positions[(values.sinkRow * COLUMNS + COLUMNS - 1) * 2],
            positions[(values.sinkRow * COLUMNS + COLUMNS - 1) * 2 + 1]
          )
        ),
        nodes: builder.upload('nodes', 'float32x2', positions),
        edges: builder.upload('edge-pairs', 'uint32x2', edges)
      },
      update: current => {
        const gapStart = 12 - Math.floor(current.gapWidth / 2);
        conductances.write(
          Float32Array.from({length: edgeCount}, (_, edge) => {
            const first = edges[edge * 2];
            const second = edges[edge * 2 + 1];
            const crossesWall = first % COLUMNS === WALL_COLUMN - 1 && second === first + 1;
            const row = Math.floor(first / COLUMNS);
            const isInGap = row >= gapStart && row < gapStart + current.gapWidth;
            return crossesWall && !isInGap ? 0 : quality[edge];
          })
        );
      }
    };
  },
  rebuildOn: ['sinkRow', 'maximumIterations'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.vertexCurrents,
      ramp: 'sequential',
      radius: 2,
      range: [0, 0.35]
    },
    {
      layer: 'dots',
      positions: outputs.nodes,
      values: outputs.potentials,
      ramp: 'sequential',
      radius: 3
    },
    {layer: 'dots', positions: outputs.terminals, radius: 8, color: [0.2, 0.9, 1, 1]}
  ]
});
