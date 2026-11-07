// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHeightAboveDrainage (HAND) on the filled DEM, D8 directions and stream mask of
// GPUTerrainFlow: each cell's height above the stream cell its flow path reaches first.

import {GPUTerrainHeightAboveDrainage} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-height-above-drainage.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

export default defineOperationRun({
  parameters,
  caption: 'Height above the nearest stream: low cells near streams light up.',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain);
    const heightAboveDrainage = builder.allocate('hand', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainHeightAboveDrainage({
        grid: terrain.grid,
        elevation: makeGPURasterBand(flow.filledElevation.view, terrain.grid, {id: 'filled'}),
        flowDirections: flow.flowDirections.view,
        streams: flow.streams.view,
        output: {heightAboveDrainage: heightAboveDrainage.view}
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, heightAboveDrainage},
      update: values => flow.writeStreamThreshold(values.streamThreshold)
    };
  },
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.heightAboveDrainage,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, 0.2],
      alpha: 0.85,
      smoothing: true
    }
  ]
});
