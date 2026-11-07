// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIsobands on a seeded surface: filled contour bands as triangles, each carrying its band index
// (the color). The triangle buffer it wrote is drawn directly as filled polygons.

import {
  getGPUIsobandsParameterValues,
  GPUIsobands
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './isobands.parameters';
import {createIsoInputs, getLevels, LEVEL_CAPACITY} from './shared/iso';

export default defineOperationRun({
  parameters,
  caption: 'Filled bands between breaks, one color each.',
  createInputs: context => createIsoInputs(context),
  build(builder, inputs) {
    const triangleCapacity = inputs.cellCount * 8;
    const values = builder.upload('values', 'float32', inputs.valuesBottomUp);
    const display = builder.upload('display', 'float32', inputs.valuesTopDown);
    const breaks = builder.allocate('breaks', 'float32', LEVEL_CAPACITY);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const triangles = builder.allocate('triangles', 'float32x2', triangleCapacity * 3);
    const triangleBands = builder.allocate('triangle-bands', 'uint32', triangleCapacity);
    builder.graph.add(
      new GPUIsobands({
        grid: inputs.grid,
        values: values.view,
        breaks: breaks.view,
        parameters: parameterBuffer.view,
        output: {
          triangles: triangles.view,
          triangleBands: triangleBands.view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    // The triangles are not indexed: vertex i of triangle t is row 3t + i, and each triangle owns
    // its band value.
    const identity = (length: number) => Uint32Array.from({length}, (_, index) => index);
    const vertexIndices = builder.upload(
      'vertex-indices',
      'uint32',
      identity(triangleCapacity * 3)
    );
    const owners = builder.upload('owners', 'uint32', identity(triangleCapacity));
    return {
      outputs: {display, triangles, triangleBands, vertexIndices, owners},
      update: current => {
        breaks.write(getLevels(current.interval, current.interval));
        parameterBuffer.write(
          getGPUIsobandsParameterValues(
            {breakCount: Math.round(current.breakCount), extent: inputs.extent},
            inputs.grid
          )
        );
        // Triangles past the written count stay zero-area and invisible.
        triangles.write(new Float32Array(triangleCapacity * 6));
        triangleBands.write(new Uint32Array(triangleCapacity));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.display,
      columns: inputs.columns,
      rows: inputs.rows
    },
    {
      layer: 'polygons',
      positions: outputs.triangles,
      indices: outputs.vertexIndices,
      owners: outputs.owners,
      values: outputs.triangleBands,
      triangleCount: outputs.owners.length,
      alpha: 0.85
    }
  ]
});
