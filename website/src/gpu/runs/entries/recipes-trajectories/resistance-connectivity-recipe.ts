// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addResistanceConnectivityRecipe on a 16 by 16 resistance grid with a barrier wall that has one
// gap. Nodes and edges are colored by the electrical potential from source (left) to sink (right):
// the potential drops across the wall except through the gap, where current concentrates.

import {addResistanceConnectivityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {getBuiltFraction} from './lib/land-cover';
import {addReadout} from './lib/readout';
import {parameters} from './resistance-connectivity-recipe.parameters';

const SIDE = 16;
const WALL_COLUMN = 8;
const GAP_START = 6;
const GAP_ROWS = 4;

export default defineOperationRun({
  parameters,
  caption: 'Current through a resistance grid with a barrier wall: it funnels through the gap.',
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
    return {
      positions,
      edges: Uint32Array.from(edges),
      world,
      wall: {column: WALL_COLUMN, gapStart: GAP_START}
    };
  },
  build(builder, inputs, values) {
    const {positions, edges} = inputs;
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
        currents: addReadout(builder, 'currents', recipe.outputs.currents),
        midpoints: builder.upload(
          'edge-midpoints',
          'float32x2',
          Float32Array.from({length: edgeCount * 2}, (_, index) => {
            const edge = Math.floor(index / 2);
            const axis = index % 2;
            return (
              (positions[edges[edge * 2] * 2 + axis] + positions[edges[edge * 2 + 1] * 2 + axis]) /
              2
            );
          })
        ),
        nodes: builder.upload('nodes', 'float32x2', positions),
        edges: builder.upload('edge-pairs', 'uint32x2', edges)
      },
      update: current =>
        conductances.write(
          Float32Array.from({length: edgeCount}, (_, edge) => {
            const column = Math.max(edges[edge * 2] % SIDE, edges[edge * 2 + 1] % SIDE);
            const row = Math.floor(edges[edge * 2] / SIDE);
            // A wall across the grid, open for GAP_ROWS rows from `gapStart`.
            const {column: wallColumn, gapStart} = inputs.wall;
            const isWall = column === wallColumn && (row < gapStart || row >= gapStart + GAP_ROWS);
            const conductance = 1 - 0.6 * getBuiltFraction(column / (SIDE - 1), row / (SIDE - 1));
            return isWall ? conductance * (1 - current.barrier) : conductance;
          })
        )
    };
  },
  rebuildOn: ['source', 'sink'],
  // The pointer places the wall: x its column, y the center of its gap. Current finds the gap.
  interaction: {
    kind: 'place',
    hint: 'move the wall gap',
    rest: [0.05 + ((WALL_COLUMN - 2) / (SIDE - 5)) * 0.9, 0.5],
    resolve(pointer, {inputs}) {
      const column = clamp(
        Math.round(2 + ((pointer.unitX - 0.05) / 0.9) * (SIDE - 5)),
        2,
        SIDE - 3
      );
      const center = clamp(1 - pointer.unitY, 0, 1);
      const gapStart = clamp(
        Math.round(center * (SIDE - 1) - (GAP_ROWS - 1) / 2),
        0,
        SIDE - GAP_ROWS
      );
      inputs.wall = {column, gapStart};
      const wallX = (0.08 + 0.84 * ((column - 0.5) / (SIDE - 1))) * inputs.world.width;
      const rowY = (row: number) => 0.08 + 0.84 * (row / (SIDE - 1));
      const gapBottom = rowY(gapStart - 0.5);
      const gapTop = rowY(gapStart + GAP_ROWS - 0.5);
      return {
        rerun: true,
        glyphs: [
          {shape: 'line', from: [wallX, 0.04], to: [wallX, gapBottom]},
          {shape: 'line', from: [wallX, gapTop], to: [wallX, 0.96]},
          {shape: 'line', from: [wallX, gapBottom], to: [wallX, gapTop], dashed: true}
        ],
        readout: `wall at column ${column} · gap rows ${gapStart}-${gapStart + GAP_ROWS - 1}`
      };
    }
  },
  // The current through every edge is a dot at its midpoint: the wall is an empty band, and the
  // current funnels through the gap. The potential is only the dim backdrop.
  draw: ({outputs}) => [
    {
      layer: 'segments',
      context: true,
      nodes: outputs.nodes,
      edges: outputs.edges,
      values: outputs.potentials,
      radius: 2,
      range: [0, 1]
    },
    {
      layer: 'dots',
      positions: outputs.midpoints,
      values: outputs.currents,
      ramp: 'sequential',
      range: [0, 0.12],
      fade: true,
      sizeByValue: true,
      radius: 9
    }
  ]
});
