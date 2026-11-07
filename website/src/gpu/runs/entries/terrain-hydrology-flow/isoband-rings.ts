// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIsobandRings chains the boundary edges of every isoband into closed rings (shells and holes).
// The ring vertices it wrote are drawn as a dotted outline over the surface.

import {
  getGPUIsobandsParameterValues,
  GPUIsobandRings
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './isoband-rings.parameters';
import {createIsoInputs, getLevels, LEVEL_CAPACITY} from './shared/iso';

const RING_CAPACITY = 600;
const VERTEX_CAPACITY = 20000;

export default defineOperationRun({
  parameters,
  caption: 'Closed shells and holes assembled from the band boundaries.',
  createInputs: context => createIsoInputs(context, 32),
  build(builder, inputs) {
    const values = builder.upload('values', 'float32', inputs.valuesBottomUp);
    const display = builder.upload('display', 'float32', inputs.valuesTopDown);
    const breaks = builder.allocate('breaks', 'float32', LEVEL_CAPACITY);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const positions = builder.allocate('positions', 'float32x2', VERTEX_CAPACITY);
    builder.graph.add(
      new GPUIsobandRings({
        grid: inputs.grid,
        values: values.view,
        breaks: breaks.view,
        parameters: parameterBuffer.view,
        edgeCapacity: VERTEX_CAPACITY,
        output: {
          ringOffsets: builder.allocate('ring-offsets', 'uint32', RING_CAPACITY + 1).view,
          positions: positions.view,
          ringIsHole: builder.allocate('ring-is-hole', 'uint32', RING_CAPACITY).view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view,
            edgeOverflow: builder.allocate('edge-overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {display, positions},
      update: next => {
        breaks.write(getLevels(next.interval, next.interval));
        parameterBuffer.write(
          getGPUIsobandsParameterValues(
            {breakCount: Math.round(next.breakCount), extent: inputs.extent},
            inputs.grid
          )
        );
        // Vertices past the written ones stay out of the picture.
        positions.write(new Float32Array(VERTEX_CAPACITY * 2).fill(Number.NaN));
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.display,
      columns: inputs.columns,
      rows: inputs.rows,
      smoothing: true
    },
    {layer: 'dots', positions: outputs.positions, radius: 3.2, color: [1, 0.9, 0.4, 1]}
  ]
});
