// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainFlowDistance on the D8 directions of GPUTerrainFlow: how far each cell's flow path runs
// downstream, either to the first stream cell or to the outlet.

import {GPUTerrainFlowDistance} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-flow-distance.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

export default defineOperationRun({
  parameters,
  caption: 'Downstream path length from every cell to the channel (or the outlet).',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain, values) {
    const flow = addTerrainFlow(builder, terrain);
    const distance = builder.allocate('distance', 'float32', terrain.cellCount);
    const validity = builder.allocate('validity', 'uint32', terrain.cellCount);
    const terminalCells = builder.allocate('terminal-cells', 'uint32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainFlowDistance({
        grid: terrain.grid,
        flowDirections: flow.flowDirections.view,
        streams: values.streams === 1 ? flow.streams.view : undefined,
        maximumSteps: terrain.cellCount,
        output: {
          distance: distance.view,
          validity: validity.view,
          terminalCells: terminalCells.view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, distance},
      update: current => flow.writeStreamThreshold(current.streamThreshold)
    };
  },
  rebuildOn: ['streams'],
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.distance,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      alpha: 0.9,
      smoothing: true
    }
  ]
});
