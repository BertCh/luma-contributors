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
import {formatNumber} from '../../interaction';
import {parameters} from './skeleton.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {getCellAt, getCellCenter} from './support/gestures';
import {createShapeRaster, stampDisc} from './support/shapes';

const DISC_RADIUS = 5;

export default defineOperationRun({
  parameters,
  rebuildOn: ['maximumIterations'],
  caption: 'A binary shape in; its one-pixel skeleton out.',
  createInputs: ({random, world}) => {
    const raster = createShapeRaster(random, world, 56, {
      fieldWeight: 0.7,
      stretch: 1.5,
      radius: 0.5
    });
    // The disc is a pointer-owned input of the shape; it starts at the rest position.
    return {
      ...raster,
      stamp: [Math.floor(raster.columns * 0.45), Math.floor(raster.rows * 0.5)] as [number, number]
    };
  },
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
        const mask = Uint32Array.from(inputs.values, value => (value > current.threshold ? 1 : 0));
        stampDisc(mask, inputs.columns, inputs.rows, inputs.stamp[0], inputs.stamp[1], DISC_RADIUS);
        shape.write(mask);
        shapeValues.write(Float32Array.from(mask));
      }
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'stamp land into the shape',
    rest: [0.45, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const {column, row} = getCellAt(inputs, world, pointer);
      inputs.stamp = [column, row];
      const skeleton = peek(outputs.skeletonValues);
      let count = 0;
      if (skeleton) for (const value of skeleton) if (Number.isFinite(value)) count++;
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'ring',
            at: getCellCenter(inputs, world, column, row),
            radius: (DISC_RADIUS * world.height) / inputs.rows,
            dashed: true
          }
        ],
        readout: `disc at [${column}, ${row}] | skeleton ${formatNumber(count, 0)} cells`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shapeValues, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.skeletonValues, {ramp: 'sequential', range: [0, 1]})
  ]
});
