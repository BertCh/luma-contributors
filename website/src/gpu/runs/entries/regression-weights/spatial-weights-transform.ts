// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTransform on a distance-band graph. Left: points colored by the sum of their
// transformed weights (1 for `row`). Right: every transformed weight against its link distance,
// which traces the kernel profile for `kernel`.

import {GPUSpatialWeightsTransform} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import {chooseSticky, createPointIndex, getChartWorldX} from '../../pick';
import {parameters} from './spatial-weights-transform.parameters';
import {
  addLinkSegments,
  addRowSums,
  getRowLinkGlyphs,
  getSegmentTargets,
  uploadWeightsBuffers,
  stretchPoints
} from './shared/weights';

const OPERATIONS = ['row', 'binary', 'kernel', 'double', 'variance', 'symmetrize'] as const;
const KERNELS = ['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform'] as const;
const BAND_RADIUS = 0.1;
const CHART_RECTANGLE = [0.62, 0.14, 0.96, 0.86] as const;

export default defineOperationRun({
  parameters,
  caption: 'Transformed weights per link (right) and their row sums (left).',
  createInputs: ({random, world}) => {
    const mapWorld = {width: world.width * 0.55, height: world.height};
    const points = stretchPoints(createUniformPoints(random, mapWorld, 220, 0.1), mapWorld);
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadWeightsBuffers(
      builder,
      'band',
      createDistanceBandWeights(points.positions, BAND_RADIUS)
    );
    const transformed = builder.allocate('transformed', 'float32', band.capacity);
    builder.graph.add(
      new GPUSpatialWeightsTransform({
        operation: OPERATIONS[values.operation],
        weights: band.input,
        output: transformed.view,
        kernel: KERNELS[values.kernel],
        bandwidth: values.bandwidth
      })
    );
    return {
      outputs: {
        positions,
        distances: band.distances!,
        transformed,
        rowSums: addRowSums(builder, 'transformed', band, transformed),
        links: addLinkSegments(builder, 'band', band, positions)
      }
    };
  },
  rebuildOn: ['operation', 'kernel', 'bandwidth'],
  // The pointer picks a point: its links, a ring at the bandwidth and the row sum of its weights.
  // A rule on the kernel profile marks the same bandwidth.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const links = getSegmentTargets(peek(outputs.links), from);
      const rowSums = peek(outputs.rowSums);
      const ruleX = getChartWorldX(
        CHART_RECTANGLE,
        world,
        Math.min(
          1,
          (OPERATIONS[values.operation] === 'kernel' ? values.bandwidth : 0) / BAND_RADIUS
        )
      );
      return {
        focus: row,
        glyphs: [
          {shape: 'ring', at: from, radius: values.bandwidth, dashed: true},
          ...getRowLinkGlyphs(from, links),
          {
            shape: 'line',
            from: [ruleX, CHART_RECTANGLE[1] * world.height],
            to: [ruleX, CHART_RECTANGLE[3] * world.height],
            dashed: true
          }
        ],
        readout: `${OPERATIONS[values.operation]} · bandwidth ${values.bandwidth} · row sum ${rowSums ? rowSums[row].toFixed(2) : '-'}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.rowSums,
      radius: 3.4
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.transformed,
      xValues: outputs.distances,
      xRange: [0, BAND_RADIUS],
      rectangle: CHART_RECTANGLE,
      radius: 2
    }
  ]
});
