// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterMedialAxis finds the ridge of centers of maximal inscribed discs of a binary shape and
// the radius of each disc. Axis cells are drawn colored by their radius (a conditional pass keeps
// only axis cells), so the axis reads as a width profile of the shape.

import {
  getGPURasterConditionalParameterValues,
  GPURasterConditional,
  GPURasterMedialAxis
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './medial-axis.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';

export default defineOperationRun({
  parameters,
  caption: 'A binary shape in; the medial axis and its disc radius out.',
  createInputs: ({random, world}) => createFractalRaster(random, world, 72, 3),
  build(builder, inputs) {
    const cellCount = inputs.values.length;
    const shape = builder.allocate('shape', 'uint32', cellCount);
    const medialAxis = builder.allocate('medial-axis', 'uint32', cellCount);
    const radius = builder.allocate('radius', 'float32', cellCount);
    const axisRadius = builder.allocate('axis-radius', 'float32', cellCount);
    const parameterBuffer = builder.upload(
      'conditional-parameters',
      'float32',
      getGPURasterConditionalParameterValues({constantB: Number.NaN})
    );
    builder.graph.add(
      new GPURasterMedialAxis({
        grid: makeGridOfRaster(inputs),
        input: shape.view,
        output: {medialAxis: medialAxis.view, radius: radius.view}
      })
    );
    builder.graph.add(
      new GPURasterConditional({
        cellCount,
        mask: medialAxis.view,
        a: radius.view,
        parameters: parameterBuffer.view,
        output: {values: axisRadius.view}
      })
    );
    const shapeValues = builder.allocate('shape-values', 'float32', cellCount);
    return {
      outputs: {shapeValues, axisRadius},
      update: current => {
        const mask = inputs.values.map(value => (value > current.threshold ? 1 : 0));
        shape.write(Uint32Array.from(mask));
        shapeValues.write(mask);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shapeValues, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.axisRadius, {ramp: 'sequential'})
  ]
});
