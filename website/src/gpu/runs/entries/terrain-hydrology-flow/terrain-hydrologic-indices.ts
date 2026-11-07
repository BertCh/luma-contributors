// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHydrologicIndices on the filled DEM and area accumulation of GPUTerrainFlow: slope,
// specific catchment area, wetness index and stream power. All four views are written; the
// slider picks which one is drawn, and minimumSlope clamps flat cells before the division.

import {
  getGPUTerrainHydrologicIndicesParameterValues,
  GPUTerrainHydrologicIndices
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-hydrologic-indices.parameters';
import {addTerrainFlow, createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

export default defineOperationRun({
  parameters,
  caption: 'Wetness peaks in flat, high-accumulation valleys; stream power on steep channels.',
  createInputs: context => createTerrainInputs(context),
  build(builder, terrain) {
    const flow = addTerrainFlow(builder, terrain, {accumulationUnits: 'area'});
    flow.writeStreamThreshold(25);
    const parameterBuffer = builder.allocate('index-parameters', 'float32', 8);
    const slope = builder.allocate('slope', 'float32', terrain.cellCount);
    const specificCatchmentArea = builder.allocate('catchment-area', 'float32', terrain.cellCount);
    const wetnessIndex = builder.allocate('wetness', 'float32', terrain.cellCount);
    const streamPowerIndex = builder.allocate('stream-power', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainHydrologicIndices({
        grid: terrain.grid,
        elevation: makeGPURasterBand(flow.filledElevation.view, terrain.grid, {id: 'filled'}),
        accumulation: flow.accumulation.view,
        parameters: parameterBuffer.view,
        output: {
          slope: slope.view,
          specificCatchmentArea: specificCatchmentArea.view,
          wetnessIndex: wetnessIndex.view,
          streamPowerIndex: streamPowerIndex.view
        }
      })
    );
    return {
      outputs: {
        filledElevation: flow.filledElevation,
        indices: [wetnessIndex, streamPowerIndex, slope, specificCatchmentArea]
      },
      update: values =>
        parameterBuffer.write(
          getGPUTerrainHydrologicIndicesParameterValues(
            {minimumSlope: values.minimumSlope},
            terrain.grid
          )
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.indices[Math.round(values.output)],
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      alpha: 0.9,
      smoothing: true
    }
  ]
});
