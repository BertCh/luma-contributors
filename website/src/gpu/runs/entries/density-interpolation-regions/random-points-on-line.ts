// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURandomPointsOnLine samples `counts[i]` points along line i, uniform by length; points are
// colored by their `fractions` output (position along the line) over the faint input polylines.

import {
  getGPUDotDensityParameterValues,
  GPURandomPointsOnLine
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findNearestOnPolyline} from '../../pick';
import {parameters} from './random-points-on-line.parameters';
import {createSpreadPolylines} from './support/geometry';

const CAPACITY = 2400;
const LINE_COUNT = 12;
const BOOST = 3;

export default defineOperationRun({
  parameters,
  caption:
    'Points uniform by length along each line (the hovered line asks for three times as many); color is the fraction along the line.',
  createInputs: ({random, world}) => {
    const lines = createSpreadPolylines(random, world, LINE_COUNT, 14);
    return {...lines, focus: {row: Math.floor(LINE_COUNT / 2)}};
  },
  build(builder, lines, values) {
    const {focus} = lines;
    const lineCount = lines.offsets.length - 1;
    const counts = builder.allocate('counts', 'uint32', lineCount);
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
    const samples = builder.allocate('samples', 'float32x2', CAPACITY);
    const fractions = builder.allocate('fractions', 'float32', CAPACITY);
    const ids = builder.allocate('ids', 'uint32', CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPURandomPointsOnLine({
        positions: builder.upload('positions', 'float32x2', lines.vertices).view,
        lineOffsets: builder.upload('line-offsets', 'uint32', lines.offsets).view,
        counts: counts.view,
        parameters: parameterBuffer.view,
        output: {
          positions: samples.view,
          fractions: fractions.view,
          points: {ids: ids.view, status: {count: count.view, overflow: overflow.view}}
        }
      })
    );
    return {
      outputs: {
        samples,
        fractions,
        segments: builder.upload('segments', 'float32x4', lines.segments)
      },
      // counts and the seed are per-frame inputs: the hovered line asks for more points.
      update: current => {
        counts.write(
          new Uint32Array(lineCount).map((_, row) =>
            row === focus.row ? current.pointsPerLine * BOOST : current.pointsPerLine
          )
        );
        parameterBuffer.write(getGPUDotDensityParameterValues({seed: current.seed}));
      }
    };
  },
  // The line nearest the pointer draws BOOST times as many points; the others keep `counts`.
  interaction: {
    kind: 'probe',
    hint: 'hover a line',
    rest: ({inputs, toUnit}) => {
      const start = inputs.offsets[inputs.focus.row];
      const middle = start + ((inputs.offsets[inputs.focus.row + 1] - start) >> 1);
      return toUnit([inputs.vertices[middle * 2], inputs.vertices[middle * 2 + 1]]);
    },
    resolve(pointer, {inputs, values}) {
      let row = -1;
      let nearest = Infinity;
      for (let line = 0; line < inputs.offsets.length - 1; line++) {
        const start = inputs.offsets[line];
        const hit = findNearestOnPolyline(
          inputs.vertices,
          pointer.x,
          pointer.y,
          start,
          inputs.offsets[line + 1] - start
        );
        if (hit && hit.distance < nearest) {
          nearest = hit.distance;
          row = line;
        }
      }
      const changed = row !== inputs.focus.row;
      inputs.focus.row = row;
      if (row < 0) return {rerun: changed};
      const start = inputs.offsets[row];
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'polyline',
            points: inputs.vertices.subarray(start * 2, inputs.offsets[row + 1] * 2)
          }
        ],
        readout: `line ${row} · counts ${values.pointsPerLine * BOOST} · others ${values.pointsPerLine}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', positions: outputs.samples, values: outputs.fractions, range: [0, 1], radius: 4}
  ]
});
