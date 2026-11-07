// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURegionStatisticsReadback draining a real GPURegionStatistics summary to the CPU. The region
// (left) is summarized on the GPU. After each run `encodeRead` copies the summary into a free slot
// of the readback ring `reads` times, the tickets are read and decoded, and the CPU numbers are
// drawn beside the region: the histogram as bars, mean, minimum and maximum as lines, the share of
// selected points as a bar, and the ring's slots, with the calls that found it full in red.

import {
  GPURegionStatistics,
  GPURegionStatisticsReadback
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {ShapeBatch} from '../../record-demo';
import {parameters} from './region-statistics-readback.parameters';

const HEADER_LENGTH = 8;
const CIRCLE_SEGMENTS = 64;
const MAXIMUM_SLOTS = 6;

export default defineOperationRun({
  parameters,
  caption: 'The region summary read back through the ring: histogram, statistics, slots.',
  createInputs: ({random, world}) => {
    // The points fill a square on the left; the readback is drawn on the right.
    const pointsWorld = {width: Math.min(world.width * 0.55, 1), height: 1};
    return {
      world,
      pointsWorld,
      points: createUniformPoints(random, pointsWorld, 3000, 0.2)
    };
  },
  build(builder, {pointsWorld, points}, values) {
    const summary = builder.allocate('summary', 'uint32', HEADER_LENGTH + values.binCount);
    const mask = builder.allocate('mask', 'uint32', points.count);
    const region = builder.allocate('region', 'float32', 3);
    builder.graph.add(
      new GPURegionStatistics({
        region: {kind: 'radius', circle: region.view},
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        values: builder.upload('values', 'float32', points.values).view,
        histogram: {binCount: values.binCount, domain: [0, 1]},
        output: {summary: summary.view, mask: mask.view}
      })
    );
    const readback = builder.own(
      new GPURegionStatisticsReadback(builder.device, {
        binCount: values.binCount,
        slotCount: values.slotCount
      })
    );
    const outline = builder.allocate('outline', 'float32x4', CIRCLE_SEGMENTS);
    return {
      outputs: {
        summary,
        mask,
        readback,
        positions: builder.upload('points', 'float32x2', points.positions),
        outline
      },
      update: current => {
        const centerX = pointsWorld.width * 0.5;
        region.write(Float32Array.of(centerX, 0.5, current.size));
        const segments = new Float32Array(CIRCLE_SEGMENTS * 4);
        for (let index = 0; index < CIRCLE_SEGMENTS; index++) {
          const angles = [index, index + 1].map(step => (step / CIRCLE_SEGMENTS) * Math.PI * 2);
          segments.set(
            [
              centerX + Math.cos(angles[0]) * current.size,
              0.5 + Math.sin(angles[0]) * current.size,
              centerX + Math.cos(angles[1]) * current.size,
              0.5 + Math.sin(angles[1]) * current.size
            ],
            index * 4
          );
        }
        outline.write(segments);
      }
    };
  },
  rebuildOn: ['binCount', 'slotCount'],
  async afterRun({device, outputs, inputs, values, upload}) {
    const {readback} = outputs;
    const {world, pointsWorld, points} = inputs;
    // Ask for `reads` copies before reading any: the ring has only `slotCount` slots.
    const encoder = device.createCommandEncoder({id: 'region-readback'});
    const tickets = [];
    let dropped = 0;
    for (let call = 0; call < values.reads; call++) {
      const ticket = readback.encodeRead(encoder, outputs.summary.buffer);
      if (ticket) tickets.push(ticket);
      else dropped++;
    }
    device.submit(encoder.finish());
    const [statistics] = await Promise.all(tickets.map(ticket => readback.read(ticket)));

    // Chart rectangle in world units, and the statistics along its x axis (domain 0 to 1).
    const chart = {x0: pointsWorld.width + 0.12, x1: world.width - 0.05, y0: 0.2, y1: 0.66};
    const toX = (value: number) =>
      chart.x0 + Math.min(Math.max(value, 0), 1) * (chart.x1 - chart.x0);
    const lines = new ShapeBatch();
    lines.addSegment(toX(statistics.mean), chart.y0 - 0.02, toX(statistics.mean), chart.y1 + 0.03);
    for (const value of [statistics.minimum, statistics.maximum]) {
      lines.addSegment(toX(value), chart.y0 - 0.06, toX(value), chart.y0 - 0.01);
    }
    lines.addSegment(chart.x0, chart.y0 - 0.01, chart.x1, chart.y0 - 0.01);

    // The ring: one square per slot, acquired ones filled; calls that found it full in red below.
    const ring = new ShapeBatch();
    const square = Math.min(0.07, (chart.x1 - chart.x0) / (MAXIMUM_SLOTS + 1));
    const acquired = tickets.length;
    for (let slot = 0; slot < values.slotCount; slot++) {
      ring.addRectangle(
        {
          x0: chart.x0 + slot * square * 1.15,
          x1: chart.x0 + slot * square * 1.15 + square,
          y0: 0.84,
          y1: 0.84 + square
        },
        slot < acquired ? 1 : 0
      );
    }
    for (let call = 0; call < dropped; call++) {
      ring.addRectangle(
        {
          x0: chart.x0 + call * square * 1.15,
          x1: chart.x0 + call * square * 1.15 + square,
          y0: 0.74,
          y1: 0.74 + square * 0.6
        },
        2
      );
    }
    // The share of points the region selected, as a bar under the chart.
    const share = new ShapeBatch();
    share.addRectangle(
      {
        x0: chart.x0,
        x1: chart.x0 + ((chart.x1 - chart.x0) * statistics.selectedCount) / points.count,
        y0: 0.06,
        y1: 0.11
      },
      1
    );
    const outline = new ShapeBatch();
    outline.addOutline({x0: chart.x0, x1: chart.x1, y0: 0.06, y1: 0.11});
    const histogram = Float32Array.from(statistics.histogram);
    return {
      histogram: upload('histogram', 'float32', histogram.length ? histogram : new Float32Array(1)),
      histogramMaximum: Math.max(...histogram, 1),
      chartRectangle: [chart.x0 / world.width, chart.y0, chart.x1 / world.width, chart.y1] as const,
      lines: lines.uploadSegments(upload, 'statistics'),
      lineCount: lines.segmentCount,
      ring: ring.uploadShapes(upload, 'ring'),
      share: share.uploadShapes(upload, 'share'),
      shareOutline: outline.uploadSegments(upload, 'share-outline'),
      shareOutlineCount: outline.segmentCount
    };
  },
  draw: ({outputs, extra, theme}) => [
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
    {layer: 'lines', context: true, segments: outputs.outline, color: [...theme.ink, 1]},
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.histogram,
      range: [0, extra.histogramMaximum],
      rectangle: extra.chartRectangle,
      color: [...theme.categories[1], 1]
    },
    {
      layer: 'lines',
      segments: extra.lines,
      count: extra.lineCount,
      radius: 1.6,
      color: [...theme.ink, 1]
    },
    {
      layer: 'polygons',
      positions: extra.ring.positions,
      indices: extra.ring.indices,
      owners: extra.ring.owners,
      values: extra.ring.categories,
      triangleCount: extra.ring.triangleCount,
      categories: [theme.muted, theme.categories[3], theme.categories[2]]
    },
    {
      layer: 'polygons',
      positions: extra.share.positions,
      indices: extra.share.indices,
      owners: extra.share.owners,
      values: extra.share.categories,
      triangleCount: extra.share.triangleCount,
      categories: [theme.muted, theme.categories[4]]
    },
    {
      layer: 'lines',
      context: true,
      segments: extra.shareOutline,
      count: extra.shareOutlineCount,
      radius: 1
    }
  ]
});
