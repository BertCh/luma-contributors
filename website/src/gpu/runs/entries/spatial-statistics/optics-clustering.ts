// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOPTICS: orders the points by density reachability and extracts clusters with the xi method.
// Colors are the cluster labels; dots too sparse for any cluster are muted. The bars underneath are
// the reachability plot: each point's reachability distance in the order OPTICS visited them, so
// clusters are the valleys. Hovering a point follows its predecessor chain on the map and marks its
// place in the plot.

import {GPUOPTICS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './optics-clustering.parameters';

const POINT_COUNT = 500;
const PLOT_BAND = 0.3;
const PLOT = [0.04, 0.04, 0.96, 0.26] as const;
const MAXIMUM_CHAIN = 40;

export default defineOperationRun({
  parameters,
  caption: 'OPTICS reachability plot and the clusters extracted with the xi method.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(
      random,
      {width: world.width, height: world.height - PLOT_BAND},
      POINT_COUNT,
      4,
      0.12
    );
    for (let row = 0; row < POINT_COUNT; row++) points.positions[row * 2 + 1] += PLOT_BAND;
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const ordering = builder.allocate('ordering', 'uint32', points.count);
    const coreDistances = builder.allocate('core-distances', 'float32', points.count);
    const reachability = builder.allocate('reachability', 'float32', points.count);
    const predecessors = builder.allocate('predecessors', 'uint32', points.count);
    const labels = builder.allocate('labels', 'uint32', points.count);
    const status = builder.allocate('status', 'uint32', 3);
    builder.graph.add(
      new GPUOPTICS({
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        minimumSamples: values.minimumSamples,
        maximumDistance: values.maximumDistance,
        extractionMethod: 'xi',
        xi: values.xi,
        output: {
          ordering: ordering.view,
          coreDistances: coreDistances.view,
          reachability: reachability.view,
          predecessors: predecessors.view,
          labels: labels.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        labels,
        ordering,
        reachability,
        predecessors,
        // Render-only: reachability in visiting order, for the plot.
        plot: builder.allocate('plot', 'float32', POINT_COUNT),
        positions: builder.upload('points', 'float32x2', points.positions)
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.6, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const ordering = peek(outputs.ordering);
      const reachability = peek(outputs.reachability);
      const predecessors = peek(outputs.predecessors);
      const labels = peek(outputs.labels);
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (!ordering || !reachability || !predecessors || !labels) return {};
      const plot = new Float32Array(POINT_COUNT);
      const order = new Uint32Array(POINT_COUNT);
      for (let rank = 0; rank < POINT_COUNT; rank++) {
        const point = ordering[rank];
        order[point] = rank;
        const distance = reachability[point];
        plot[rank] = Number.isFinite(distance)
          ? Math.min(distance, values.maximumDistance)
          : values.maximumDistance;
      }
      outputs.plot.write(plot);
      const floor = PLOT[1] * world.height;
      const glyphs: RunGlyph[] = [
        {
          shape: 'line',
          from: [PLOT[0] * world.width, floor],
          to: [PLOT[2] * world.width, floor]
        }
      ];
      if (row < 0) return {glyphs};
      const rank = order[row];
      const markerX = (PLOT[0] + ((rank + 0.5) / POINT_COUNT) * (PLOT[2] - PLOT[0])) * world.width;
      glyphs.push({shape: 'line', from: [markerX, floor], to: [markerX, PLOT[3] * world.height]});
      // The chain of predecessors that reached this point, nearest first.
      const chain: number[] = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]];
      let current = row;
      for (let step = 0; step < MAXIMUM_CHAIN; step++) {
        const previous = predecessors[current];
        if (previous === 0xffffffff || previous >= POINT_COUNT) break;
        chain.push(inputs.positions[previous * 2], inputs.positions[previous * 2 + 1]);
        current = previous;
      }
      if (chain.length > 2) glyphs.push({shape: 'polyline', points: chain});
      glyphs.push({shape: 'crosshair', at: [chain[0], chain[1]], size: 12});
      const distance = reachability[row];
      const label = labels[row];
      return {
        glyphs,
        readout: `order ${rank} · reachability ${Number.isFinite(distance) ? distance.toFixed(3) : 'none'} · ${
          label === 0xffffffff ? 'no cluster' : `cluster ${label}`
        }`
      };
    }
  },
  draw: ({outputs, values, theme}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 4.5},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.plot,
      range: [0, values.maximumDistance],
      rectangle: PLOT,
      color: [...theme.categories[0], 0.85]
    }
  ]
});
