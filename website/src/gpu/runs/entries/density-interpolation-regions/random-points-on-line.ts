// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURandomPointsOnLine samples `counts[i]` points along line i, uniform by length; points are
// colored by their `fractions` output (position along the line) over the faint input polylines.

import {
  getGPUDotDensityParameterValues,
  GPURandomPointsOnLine
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createPolylines} from '../../inputs';
import {parameters} from './random-points-on-line.parameters';

const CAPACITY = 2000;

export default defineOperationRun({
  parameters,
  caption: 'Points uniform by length along each line; color is the fraction along the line.',
  createInputs: ({random, world}) => createPolylines(random, world, 20, 14),
  build(builder, lines, values) {
    const lineCount = lines.offsets.length - 1;
    const counts = builder.upload(
      'counts',
      'uint32',
      new Uint32Array(lineCount).fill(values.pointsPerLine)
    );
    const parameterBuffer = builder.allocate('parameters', 'uint32', 8);
    parameterBuffer.write(getGPUDotDensityParameterValues({seed: values.seed}));
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
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', positions: outputs.samples, values: outputs.fractions, range: [0, 1], radius: 4}
  ]
});
