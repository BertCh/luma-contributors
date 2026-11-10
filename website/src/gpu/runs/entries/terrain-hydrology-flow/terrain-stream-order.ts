// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainStreamOrder on the D8 directions and stream mask of GPUTerrainFlow: every stream cell
// gets its Strahler order, headwaters 1. The threshold slider is a GPUTerrainFlow prop: it changes the stream mask the order is ranked on.

import {GPUTerrainStreamOrder} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-stream-order.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  getTerrainCell,
  getTerrainCellCenter,
  getTerrainContextLayer
} from './shared/terrain';

// Index is the Strahler order: 0 is no stream; orders 1 to 5 step from pale blue to deep magenta.
const ORDER_COLORS = [
  [0, 0, 0, 0],
  [0.4, 0.75, 1, 1],
  [0.1, 0.5, 0.95, 1],
  [0.45, 0.2, 0.85, 1],
  [0.85, 0.1, 0.5, 1],
  [0.6, 0, 0.15, 1]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Strahler order of each stream cell: pale blue is a headwater (order 1), deep red the main stem.',
  createInputs: context => ({
    ...createTerrainInputs(context),
    focus: {order: 0, source: null as Uint32Array | null}
  }),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    const streamOrder = builder.allocate('stream-order', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainStreamOrder({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        streams: flow.streams.view,
        output: {streamOrder: streamOrder.view}
      })
    );
    // The cells of the hovered order, in that order's color; 0 elsewhere.
    const focusOrder = builder.allocate('focus-order', 'uint32', terrain.cellCount);
    return {
      outputs: {filledElevation: flow.filledElevation, streamOrder, focusOrder},
      update: values => flow.writeStreamThreshold(values.streamThreshold)
    };
  },
  // The pointer snaps to the nearest stream cell, and every cell of its Strahler order is lit.
  interaction: {
    kind: 'probe',
    hint: 'hover a stream',
    rest: [0.5, 0.52],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const order = peek(outputs.streamOrder) as Uint32Array | null;
      if (!order) return {};
      const cell = getTerrainCell(inputs, world, pointer.x, pointer.y);
      let nearest = -1;
      let nearestDistance = 7;
      for (let offsetRow = -6; offsetRow <= 6; offsetRow++) {
        for (let offsetColumn = -6; offsetColumn <= 6; offsetColumn++) {
          const column = cell.column + offsetColumn;
          const row = cell.row + offsetRow;
          if (column < 0 || row < 0 || column >= inputs.columns || row >= inputs.rows) continue;
          const candidate = row * inputs.columns + column;
          const distance = Math.hypot(offsetColumn, offsetRow);
          if (order[candidate] > 0 && order[candidate] < 100 && distance < nearestDistance) {
            nearest = candidate;
            nearestDistance = distance;
          }
        }
      }
      const focusedOrder = nearest < 0 ? 0 : order[nearest];
      if (inputs.focus.order !== focusedOrder || inputs.focus.source !== order) {
        outputs.focusOrder.write(
          Uint32Array.from(order, value => (focusedOrder > 0 && value === focusedOrder ? value : 0))
        );
        inputs.focus = {order: focusedOrder, source: order};
      }
      if (nearest < 0) return {focus: 0, spotlight: null};
      let streamCells = 0;
      let sameOrder = 0;
      for (const value of order) {
        if (value > 0 && value < 100) streamCells++;
        if (value === focusedOrder) sameOrder++;
      }
      return {
        focus: focusedOrder,
        spotlight: null,
        glyphs: [{shape: 'dot', at: getTerrainCellCenter(inputs, world, nearest)}],
        readout: `order ${focusedOrder} · ${Math.round((100 * sameOrder) / streamCells)}% of stream cells`
      };
    }
  },
  draw: ({inputs, outputs, interaction}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.streamOrder,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'categorical',
      categories: ORDER_COLORS,
      alpha: interaction.focus ? 0.3 : 1
    },
    {
      layer: 'raster',
      values: outputs.focusOrder,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'categorical',
      categories: ORDER_COLORS
    }
  ]
});
