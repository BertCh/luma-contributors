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
import {formatNumber} from '../../interaction';
import {parameters} from './medial-axis.parameters';
import {drawRaster, makeGridOfRaster} from './support/bands';
import {getCellAt, getCellCenter} from './support/gestures';
import {createShapeRaster} from './support/shapes';

export default defineOperationRun({
  parameters,
  caption: 'A binary shape in; the medial axis and its disc radius out.',
  createInputs: ({random, world}) =>
    createShapeRaster(random, world, 72, {fieldWeight: 0.62, stretch: 1.35}),
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
  // The pointer probes the axis: the nearest axis cell's maximal inscribed disc, read from the
  // radius the operation wrote.
  interaction: {
    kind: 'probe',
    hint: 'probe the axis',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, world, peek}) => {
      const radius = peek(outputs.axisRadius);
      if (!radius) return {readout: 'medial axis'};
      const cellWidth = world.width / inputs.columns;
      const cellHeight = world.height / inputs.rows;
      const center = getCellAt(inputs, world, pointer);
      let best = -1;
      let bestDistance = Infinity;
      for (let index = 0; index < radius.length; index++) {
        if (!Number.isFinite(radius[index])) continue;
        const dx = ((index % inputs.columns) - center.column) * cellWidth;
        const dy = (Math.floor(index / inputs.columns) - center.row) * cellHeight;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      }
      if (best < 0) return {readout: 'no axis'};
      const column = best % inputs.columns;
      const row = Math.floor(best / inputs.columns);
      const at = getCellCenter(inputs, world, column, row);
      return {
        glyphs: [
          {shape: 'dot', at},
          {shape: 'ring', at, radius: radius[best] * cellHeight}
        ],
        readout: `axis [${column}, ${row}] | radius ${formatNumber(radius[best], 1)} cells`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.shapeValues, {context: true, range: [0, 1]}),
    drawRaster(inputs, outputs.axisRadius, {ramp: 'sequential', range: [0, 14]})
  ]
});
