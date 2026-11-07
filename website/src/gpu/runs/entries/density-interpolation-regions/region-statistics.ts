// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURegionStatistics summarizes the points inside a region (circle or rectangle) without a
// readback: the bright dots are the selected rows (the `mask` output) and the inset histogram is
// the `summary` output, both written by the operation on every region change.

import {GPURegionStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {parameters} from './region-statistics.parameters';
import {addSlice} from './support/gather';

const HEADER_LENGTH = 8;
const CIRCLE_SEGMENTS = 64;

export default defineOperationRun({
  parameters,
  caption: 'Selected points and the histogram of their values, recomputed as the region moves.',
  createInputs: ({random, world}) => ({
    world,
    points: createUniformPoints(random, world, 3000, 0.2)
  }),
  build(builder, {world, points}, current) {
    const summary = builder.allocate('summary', 'uint32', HEADER_LENGTH + current.binCount);
    const mask = builder.allocate('mask', 'uint32', points.count);
    const isRadius = current.kind === 0;
    const region = builder.allocate('region', 'float32', isRadius ? 3 : 4);
    builder.graph.add(
      new GPURegionStatistics({
        region: isRadius
          ? {kind: 'radius', circle: region.view}
          : {kind: 'rectangle', bounds: region.view},
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        values: builder.upload('values', 'float32', points.values).view,
        histogram: {binCount: current.binCount, domain: [0, 1]},
        output: {summary: summary.view, mask: mask.view}
      })
    );
    const outline = builder.allocate('outline', 'float32x4', CIRCLE_SEGMENTS);
    return {
      outputs: {
        mask,
        histogram: addSlice(builder, summary, HEADER_LENGTH, current.binCount, 'histogram'),
        positions: builder.upload('points', 'float32x2', points.positions),
        outline
      },
      update: values => {
        const centerX = values.centerX * world.width;
        const centerY = 0.5;
        if (isRadius) region.write(Float32Array.of(centerX, centerY, values.size));
        else {
          region.write(
            Float32Array.of(
              centerX - values.size,
              centerY - values.size,
              centerX + values.size,
              centerY + values.size
            )
          );
        }
        const segments = new Float32Array(CIRCLE_SEGMENTS * 4);
        for (let index = 0; index < CIRCLE_SEGMENTS; index++) {
          const corners = isRadius
            ? [index, index + 1].map(step => [
                centerX + Math.cos((step / CIRCLE_SEGMENTS) * Math.PI * 2) * values.size,
                centerY + Math.sin((step / CIRCLE_SEGMENTS) * Math.PI * 2) * values.size
              ])
            : [index, index + 1].map(step => {
                const side = Math.floor(((step % CIRCLE_SEGMENTS) / CIRCLE_SEGMENTS) * 4);
                const along = (((step % CIRCLE_SEGMENTS) / CIRCLE_SEGMENTS) * 4) % 1;
                const points = [
                  [-1, -1 + 2 * along],
                  [-1 + 2 * along, 1],
                  [1, 1 - 2 * along],
                  [1 - 2 * along, -1]
                ][side];
                return [centerX + points[0] * values.size, centerY + points[1] * values.size];
              });
          segments.set([...corners[0], ...corners[1]], index * 4);
        }
        outline.write(segments);
      }
    };
  },
  rebuildOn: ['kind', 'binCount'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.6, alpha: 0.5},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.mask,
      range: [0, 1],
      ramp: 'sequential',
      fade: true,
      radius: 3
    },
    {layer: 'lines', context: true, segments: outputs.outline, color: [1, 1, 1, 1]},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.histogram,
      range: 'auto',
      rectangle: [0.62, 0.06, 0.97, 0.3],
      ramp: 'sequential'
    }
  ]
});
