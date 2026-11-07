// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addCellFlowTopologyRecipe on a patch of H3 cells: origin-destination flows are totaled per cell.
// Fill is the outgoing total, rings are the incoming total (bigger and brighter = more arrivals).

import {
  addCellFlowTopologyRecipe,
  GPUPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill} from './lib/cells';
import {parameters} from './cell-flow-topology-recipe.parameters';

const COLUMNS = 13;
const ROWS = 8;
const RESOLUTION = 5;

export default defineOperationRun({
  parameters,
  caption: 'Outgoing (fill) and incoming (dots) flow per H3 cell.',
  createInputs: ({random, world}) => ({
    world,
    // One point per lattice node, close to the H3 cell spacing; each becomes a cell.
    lattice: Float32Array.from({length: COLUMNS * ROWS * 2}, (_, index) => {
      const node = Math.floor(index / 2);
      const column = node % COLUMNS;
      const row = Math.floor(node / COLUMNS);
      return index % 2 === 0 ? 0.08 + column * 0.135 + (row % 2) * 0.068 : 0.1 + row * 0.115;
    }),
    draws: Float32Array.from({length: 800}, random)
  }),
  build(builder, {lattice, draws}, values) {
    const cellCount = COLUMNS * ROWS;
    const pairCount = values.pairCount;
    const cells = builder.allocate('cells', 'uint32x2', cellCount);
    builder.graph.add(
      new GPUPointToCell({
        family: 'h3',
        resolution: RESOLUTION,
        positions: builder.upload('lattice', 'float32x2', lattice).view,
        output: {cells: cells.view}
      })
    );
    const origins = new Uint32Array(pairCount);
    const destinations = new Uint32Array(pairCount);
    for (let pair = 0; pair < pairCount; pair++) {
      const origin = Math.floor(draws[pair * 2] * cellCount);
      // Mostly nearby destinations in the lattice (a few columns and rows away).
      const column = Math.min(
        COLUMNS - 1,
        Math.max(0, (origin % COLUMNS) + Math.round((draws[pair * 2 + 1] - 0.5) * 6))
      );
      const row = Math.min(
        ROWS - 1,
        Math.max(
          0,
          Math.floor(origin / COLUMNS) + Math.round((draws[(pair * 2 + 401) % 800] - 0.5) * 4)
        )
      );
      origins[pair] = origin;
      destinations[pair] = row * COLUMNS + column;
    }
    const flowValues = builder.allocate('flow-values', 'float32', pairCount);
    const outgoingTotals = builder.allocate('outgoing', 'float32', cellCount);
    const incomingTotals = builder.allocate('incoming', 'float32', cellCount);
    const invalid = builder.allocate('invalid', 'uint32', 1);
    addCellFlowTopologyRecipe(builder.graph, {
      cells: cells.view,
      origins: builder.upload('origins', 'uint32', origins).view,
      destinations: builder.upload('destinations', 'uint32', destinations).view,
      values: flowValues.view,
      outputs: {
        outgoingTotals: outgoingTotals.view,
        incomingTotals: incomingTotals.view,
        status: {invalidCount: invalid.view}
      }
    });
    return {
      outputs: {outgoingTotals, incomingTotals, fill: addCellFill(builder, 'h3', cells)},
      update: current =>
        flowValues.write(
          Float32Array.from({length: pairCount}, (_, pair) => {
            const eastward = (destinations[pair] % COLUMNS) - (origins[pair] % COLUMNS);
            return Math.max(0.1, 1 + draws[(pair * 3) % 800] * 4 + current.pull * eastward * 0.5);
          })
        )
    };
  },
  rebuildOn: ['pairCount'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.outgoingTotals,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential'
    },
    {
      layer: 'dots',
      positions: outputs.fill.centers,
      values: outputs.incomingTotals,
      ramp: 'diverging',
      radius: 6
    }
  ]
});
