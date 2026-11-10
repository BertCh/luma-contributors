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
import {
  createIsoInputs,
  getLevels,
  getLevelsAround,
  LEVEL_CAPACITY,
  sampleValue
} from './shared/iso';

const SEGMENT_CAPACITY = 20000;

export default defineOperationRun({
  parameters,
  caption: 'Contour lines at fixed levels over a surface.',
  createInputs: context => ({
    ...createIsoInputs(context),
    /** The field value under the pointer: the level the contours are anchored on. */
    cursorValue: null as number | null
  }),
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
    // The contour through the cursor is a second run of the operation with that one level.
    const cursorLevel = builder.allocate('cursor-level', 'float32', LEVEL_CAPACITY);
    const cursorParameters = builder.allocate('cursor-parameters', 'float32', 12);
    const cursorSegments = builder.allocate('cursor-segments', 'float32x4', SEGMENT_CAPACITY);
    const cursorCount = builder.allocate('cursor-count', 'uint32', 1);
    builder.graph.add(
      new GPUIsolines({
        id: 'cursor-isolines',
        grid: inputs.grid,
        values: values.view,
        levels: cursorLevel.view,
        parameters: cursorParameters.view,
        output: {
          segments: cursorSegments.view,
          segmentLevels: builder.allocate('cursor-segment-levels', 'uint32', SEGMENT_CAPACITY).view,
          status: {
            count: cursorCount.view,
            overflow: builder.allocate('cursor-overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {display, segments, segmentCount, cursorSegments, cursorCount},
      update: current => {
        const anchor = inputs.cursorValue;
        levels.write(
          anchor === null
            ? getLevels(current.interval, current.interval)
            : getLevelsAround(anchor, current.interval)
        );
        parameterBuffer.write(
          getGPUIsolinesParameterValues(
            {levelCount: Math.round(current.levelCount), extent: inputs.extent},
            inputs.grid
          )
        );
        cursorLevel.write(Float32Array.of(anchor ?? current.interval));
        cursorParameters.write(
          getGPUIsolinesParameterValues({levelCount: 1, extent: inputs.extent}, inputs.grid)
        );
      }
    };
  },
  // The pointer reads the field: the contour through the value under it is drawn in color, and the
  // other levels are spaced from it by `interval`.
  interaction: {
    kind: 'level',
    hint: 'pick a level',
    rest: [0.55, 0.45],
    resolve(pointer, {inputs, values}) {
      const value = Math.round(sampleValue(inputs, pointer.x, pointer.y) * 10) / 10;
      const changed = inputs.cursorValue !== value;
      inputs.cursorValue = value;
      return {
        rerun: changed,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `level ${value.toFixed(1)} · interval ${values.interval}`
      };
    }
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
      radius: 1.1,
      color: [1, 1, 1, 0.75]
    },
    {
      layer: 'lines',
      segments: outputs.cursorSegments,
      count: outputs.cursorCount,
      radius: 2.4,
      color: [1, 0.78, 0.2, 1]
    }
  ]
});
