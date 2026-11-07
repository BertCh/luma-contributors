// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSkeleton thins a binary shape (land above a threshold) to a one-pixel-wide skeleton
// by iterated peeling. Only skeleton cells are drawn (a conditional pass turns the mask into
// values and nodata), over the shape. With too few `maximumIterations` the shape is only
// partly eroded.

import {
  getGPURasterConditionalParameterValues,
  GPURasterConditional,
  GPURasterSkeleton
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createFractalRaster} from '../../inputs';
import {parameters} from './skeleton.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';

export default defineOperationRun({
  parameters,
  rebuildOn: ['maximumIterations'],
  caption: 'A binary shape in; its one-pixel skeleton out.',
  createInputs: ({random, world}) => createFractalRaster(random, world, 72, 3),
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const shape = builder.allocate('shape', 'uint32', cellCount);
    const skeleton = builder.allocate('skeleton', 'uint32', cellCount);
    const converged = builder.allocate('converged', 'uint32', 1);
    const parameterBuffer = builder.upload(
      'conditional-parameters',
      'float32',
      getGPURasterConditionalParameterValues({constantA: 1, constantB: Number.NaN})
    );
    const skeletonValues = builder.allocate('skeleton-values', 'float32', cellCount);
    builder.graph.add(
      new GPURasterSkeleton({
        grid: makeGridOfRaster(inputs),
        input: shape.view,
        maximumIterations: values.maximumIterations,
        output: {skeleton: skeleton.view, status: {converged: converged.view}}
      })
    );
    builder.graph.add(
      new GPURasterConditional({
        cellCount,
        mask: skeleton.view,
        parameters: parameterBuffer.view,
        output: {values: skeletonValues.view}
      })
    );
    const shapeValues = builder.allocate('shape-values', 'float32', cellCount);
    return {
      outputs: {shapeValues, skeletonValues},
      update: current => {
        const mask = inputs.values.map(value => (value > current.threshold ? 1 : 0));
        shape.write(Uint32Array.from(mask));
        shapeValues.write(mask);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shapeValues, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.skeletonValues, {ramp: 'sequential', range: [0, 1]})
  ]
});
