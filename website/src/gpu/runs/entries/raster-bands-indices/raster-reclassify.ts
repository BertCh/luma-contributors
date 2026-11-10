// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterReclassify: elevation binned through an ascending break table into classes. The active
// break count, interval closure and the break values are per-frame data, so sliders never recompile.

import {
  getGPURasterReclassifyParameterValues,
  GPURasterReclassify
} from '@luma-contributors/analysis/raster-analysis';
import {bindAxes} from '../../interaction';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {parameters} from './raster-reclassify.parameters';
import {drawRaster} from './support/bands';

const MAXIMUM_BREAK_COUNT = 7;

const PAD = bindAxes(parameters, {
  kind: 'level',
  hint: 'move the first break and count',
  x: 'firstBreak',
  y: 'breakCount'
});

export default defineOperationRun({
  parameters,
  caption: 'A continuous raster in; one class per cell out.',
  createInputs: ({world}) => createElevationRaster(world),
  build(builder, raster) {
    const cellCount = raster.values.length;
    const source = builder.upload('values', 'float32', raster.values);
    const breaks = builder.allocate('breaks', 'float32', MAXIMUM_BREAK_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const classes = builder.allocate('classes', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterReclassify({
        values: source.view,
        breaks: breaks.view,
        parameters: parameterBuffer.view,
        output: {classes: classes.view}
      })
    );
    return {
      outputs: {source, classes},
      update: current => {
        // Evenly spaced ascending breaks from breaks[0] up to 0.95.
        const spacing = (0.95 - current.firstBreak) / Math.max(1, current.breakCount - 1);
        breaks.write(
          Float32Array.from(
            {length: MAXIMUM_BREAK_COUNT},
            (_, index) => current.firstBreak + index * spacing
          )
        );
        parameterBuffer.write(
          getGPURasterReclassifyParameterValues({
            breakCount: current.breakCount
          })
        );
      }
    };
  },
  // The pointer is a pad over the break table: breaks[0] across, breakCount up.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      return {
        ...base,
        ...{
          glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
          readout: `breaks[0] ${current.firstBreak.toFixed(2)} | ${current.breakCount + 1} classes`
        }
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    drawRaster(inputs, outputs.source, {context: true}),
    // Ordered classes: a stepped sequential ramp over the class index, low to high.
    drawRaster(inputs, outputs.classes, {
      ramp: 'sequential',
      range: [0, values.breakCount],
      bands: values.breakCount + 1
    })
  ]
});
