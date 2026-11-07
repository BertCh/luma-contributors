// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIsolines on a seeded surface: marching-squares contour segments at the chosen levels, drawn as
// the line list it wrote (the segment count is the GPU-written scalar). Levels and their count are
// per-frame writes, so sliding them never recompiles.

import {
  getGPUIsolinesParameterValues,
  GPUIsolines
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './isolines.parameters';
import {createIsoInputs, getLevels, LEVEL_CAPACITY} from './shared/iso';

const SEGMENT_CAPACITY = 20000;

export default defineOperationRun({
  parameters,
  caption: 'Contour lines at fixed levels over a surface.',
  createInputs: context => createIsoInputs(context),
  build(builder, inputs) {
    const values = builder.upload('values', 'float32', inputs.valuesBottomUp);
    const display = builder.upload('display', 'float32', inputs.valuesTopDown);
    const levels = builder.allocate('levels', 'float32', LEVEL_CAPACITY);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const segments = builder.allocate('segments', 'float32x4', SEGMENT_CAPACITY);
    const segmentCount = builder.allocate('segment-count', 'uint32', 1);
    builder.graph.add(
      new GPUIsolines({
        grid: inputs.grid,
        values: values.view,
        levels: levels.view,
        parameters: parameterBuffer.view,
        output: {
          segments: segments.view,
          segmentLevels: builder.allocate('segment-levels', 'uint32', SEGMENT_CAPACITY).view,
          status: {
            count: segmentCount.view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {display, segments, segmentCount},
      update: current => {
        levels.write(getLevels(current.interval, current.interval));
        parameterBuffer.write(
          getGPUIsolinesParameterValues(
            {levelCount: Math.round(current.levelCount), extent: inputs.extent},
            inputs.grid
          )
        );
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
    {
      layer: 'lines',
      segments: outputs.segments,
      count: outputs.segmentCount,
      radius: 1.6,
      color: [1, 1, 1, 0.95]
    }
  ]
});
