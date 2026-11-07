// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainStreamOrder on the D8 directions and stream mask of GPUTerrainFlow: every stream cell
// gets its Strahler order, headwaters 1. The threshold slider is a GPUTerrainFlow prop: it changes the stream mask the order is ranked on.

import {GPUTerrainStreamOrder} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-stream-order.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

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
  createInputs: context => createTerrainInputs(context),
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
    return {
      outputs: {filledElevation: flow.filledElevation, streamOrder},
      update: values => flow.writeStreamThreshold(values.streamThreshold)
    };
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.streamOrder,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'categorical',
      categories: ORDER_COLORS
    }
  ]
});
