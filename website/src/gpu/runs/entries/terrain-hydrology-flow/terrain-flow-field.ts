// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainFlowField deflects a uniform wind over the DEM, writing a (u, v) per cell. The vectors
// are shown through GPULineIntegralConvolution, which smears noise along them, over the relief.

import {
  getGPUTerrainFlowFieldParameterValues,
  GPUTerrainFlowField
} from '@luma-contributors/analysis/gpu-terrain';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues,
  GPULineIntegralConvolution,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {aimAround} from '../../interaction';
import {parameters} from './terrain-flow-field.parameters';
import {createTerrainInputs, getTerrainContextLayer} from './shared/terrain';

const OUTPUT_SCALE = 2;

export default defineOperationRun({
  parameters,
  caption: 'Wind bends around ridges and speeds over summits; noise is streaked along the vectors.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    // Row 0 has the smallest y here, so the vectors, the noise and the raster share one frame.
    const grid = makeGPURasterMetadata({
      width: terrain.columns,
      height: terrain.rows,
      cellSize: terrain.cellSize,
      rowDirection: 'north'
    });
    return {...terrain, grid};
  },
  build(builder, terrain) {
    const {columns, rows, cellCount, cellSize, grid} = terrain;
    const elevation = builder.upload('dem', 'float32', terrain.elevation);
    const flowFieldParameters = builder.allocate('flow-field-parameters', 'float32', 8);
    const velocities = builder.allocate('velocities', 'float32x2', cellCount);
    builder.graph.add(
      new GPUTerrainFlowField({
        grid,
        elevation: makeGPURasterBand(elevation.view, grid, {id: 'dem'}),
        parameters: flowFieldParameters.view,
        output: {velocities: velocities.view}
      })
    );
    const outputCount = cellCount * OUTPUT_SCALE * OUTPUT_SCALE;
    const licParameters = builder.allocate('lic-parameters', 'float32', 12);
    const licWords = builder.allocate('lic-words', 'uint32', 4);
    const values = builder.allocate('values', 'float32', outputCount);
    const speeds = builder.allocate('speeds', 'float32', outputCount);
    builder.graph.add(
      new GPULineIntegralConvolution({
        velocities: velocities.view,
        fieldWidth: columns,
        fieldHeight: rows,
        grid: makeGPURasterMetadata({
          width: columns * OUTPUT_SCALE,
          height: rows * OUTPUT_SCALE
        }),
        stepCount: 20,
        parameters: licParameters.view,
        uint32Parameters: licWords.view,
        output: {values: values.view, speeds: speeds.view}
      })
    );
    return {
      outputs: {elevation, values, speeds},
      update: current => {
        const angle = (current.windDirection * Math.PI) / 180;
        flowFieldParameters.write(
          getGPUTerrainFlowFieldParameterValues(
            {
              wind: [Math.cos(angle), Math.sin(angle)],
              verticalExaggeration: current.verticalExaggeration
            },
            grid
          )
        );
        const settings = {
          fieldPlacement: {origin: [0, 0], cellSize: [cellSize, cellSize]} as const,
          outputPlacement: {
            origin: [0, 0],
            cellSize: [cellSize / OUTPUT_SCALE, cellSize / OUTPUT_SCALE]
          } as const,
          stepLength: 0.5
        };
        licParameters.write(getGPULineIntegralConvolutionParameterValues(settings));
        licWords.write(getGPULineIntegralConvolutionUint32ParameterValues({seed: 0}));
      }
    };
  },
  // The pointer is the wind: its compass angle around the middle of the map is windDirection, and
  // the farther out it sits the more the relief is exaggerated.
  interaction: aimAround(parameters, {
    hint: 'aim the wind',
    angle: 'windDirection',
    convention: 'math',
    distance: {parameter: 'verticalExaggeration'},
    readout: values =>
      `wind ${values.windDirection} deg · verticalExaggeration ${values.verticalExaggeration}`
  }),
  draw: ({inputs, outputs}) => [
    getTerrainContextLayer(inputs, outputs.elevation),
    {
      // Speed relative to the free-stream wind (1): blue slows, red speeds up.
      layer: 'raster',
      values: outputs.speeds,
      columns: inputs.columns * OUTPUT_SCALE,
      rows: inputs.rows * OUTPUT_SCALE,
      ramp: 'diverging',
      range: [0, 2],
      alpha: 0.55,
      smoothing: true
    },
    {
      layer: 'raster',
      values: outputs.values,
      columns: inputs.columns * OUTPUT_SCALE,
      rows: inputs.rows * OUTPUT_SCALE,
      ramp: 'sequential',
      range: [0.2, 0.8],
      alpha: 0.4
    }
  ]
});
