// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURoseStatistic bins the directions of line segments (left, the input) into a rose histogram
// (right, the `bins` output). Axial mode folds opposite directions together.

import {GPURoseStatistic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {parameters} from './rose-statistic.parameters';

const SEGMENT_COUNT = 400;

export default defineOperationRun({
  parameters,
  caption: 'Segment directions summarized as a rose histogram.',
  createInputs: ({world}) => ({world}),
  build(builder, {world}, values) {
    const random = makeRandom(8);
    const starts = new Float32Array(SEGMENT_COUNT * 2);
    const ends = new Float32Array(SEGMENT_COUNT * 2);
    const segments = new Float32Array(SEGMENT_COUNT * 4);
    for (let index = 0; index < SEGMENT_COUNT; index++) {
      const normal = Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
      // A dominant direction plus a weaker, opposite one (visible only in axial mode).
      const angle = (random() < 0.7 ? 0.6 : 0.6 + Math.PI) + normal * values.concentration;
      const x = random() * 0.5 * world.width * 0.9 + 0.02;
      const y = 0.05 + random() * 0.9;
      const length = 0.02 + random() * 0.03;
      starts.set([x, y], index * 2);
      ends.set([x + Math.cos(angle) * length, y + Math.sin(angle) * length], index * 2);
      segments.set(
        [...starts.slice(index * 2, index * 2 + 2), ...ends.slice(index * 2, index * 2 + 2)],
        index * 4
      );
    }
    const bins = builder.allocate('bins', 'float32', values.binCount);
    builder.graph.add(
      new GPURoseStatistic({
        starts: builder.upload('starts', 'float32x2', starts).view,
        ends: builder.upload('ends', 'float32x2', ends).view,
        circular: {
          mode: values.mode === 0 ? 'directional' : 'axial',
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: values.binCount
        },
        output: {summary: builder.allocate('summary', 'float32', 5).view, bins: bins.view}
      })
    );
    return {outputs: {bins, segments: builder.upload('segments', 'float32x4', segments)}};
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.segments,
      count: SEGMENT_COUNT,
      color: [0.8, 0.85, 1, 0.8]
    },
    {layer: 'chart', kind: 'bars', values: outputs.bins, rectangle: [0.55, 0.1, 0.97, 0.9]}
  ]
});
