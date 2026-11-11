// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineSmooth runs Chaikin corner cutting: each pass replaces every corner by two points at
// `ratio` along its edges. The coarse polyline is the muted context; the smoothed lines are output.

import {
  getGPULineSmoothParameterValues,
  GPULineSmooth
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  addPackedSegments,
  allocateLineOutput,
  createLineSet,
  createPackedSegments,
  findNearestLine,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './line-smooth.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Chaikin corner cutting: coarse polylines (muted) and their smoothed output.',
  createInputs: ({random, world}) => createLineSet(random, world, 3, 7),
  build(builder, lines, values) {
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const result = allocateLineOutput(builder, 'smoothed', 4000, lines.lineCount);
    const parameterBuffer = builder.allocate('smooth-parameters', 'float32', 4);
    builder.graph.add(
      new GPULineSmooth({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        iterationCount: values.iterationCount,
        parameters: parameterBuffer.view,
        output: result.output
      })
    );
    const drawing = addPackedSegments(builder, 'smoothed', result.positions, result.lineOffsets);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(lines.positions, lines.lineOffsets, false)
    );
    const originalPoints = builder.upload('original-points', 'float32x2', lines.positions);
    return {
      outputs: {original, originalPoints, ...drawing},
      update: current =>
        parameterBuffer.write(getGPULineSmoothParameterValues({ratio: current.ratio}))
    };
  },
  rebuildOn: ['iterationCount'],
  // How far along a coarse edge the pointer projects, measured from the nearer corner, is the cut
  // ratio: the two Chaikin points of that edge sit at ratio and 1 - ratio.
  interaction: {
    kind: 'reach',
    hint: 'slide along an edge',
    rest: ({inputs, toUnit}) => {
      const ratio = findParameter(parameters, 'ratio').default;
      const row = inputs.lineOffsets[0] + 2;
      const positions = inputs.positions;
      return toUnit([
        positions[row * 2] + (positions[row * 2 + 2] - positions[row * 2]) * ratio,
        positions[row * 2 + 1] + (positions[row * 2 + 3] - positions[row * 2 + 1]) * ratio
      ]);
    },
    resolve(pointer, {inputs, values}) {
      const hit = findNearestLine(inputs, pointer.x, pointer.y);
      if (!hit) return {};
      const ratio = snapToParameter(
        findParameter(parameters, 'ratio'),
        Math.min(hit.fraction, 1 - hit.fraction)
      );
      const positions = inputs.positions;
      const x0 = positions[hit.row * 2];
      const y0 = positions[hit.row * 2 + 1];
      const x1 = positions[hit.row * 2 + 2];
      const y1 = positions[hit.row * 2 + 3];
      const near: [number, number] = [x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio];
      const far: [number, number] = [x1 + (x0 - x1) * ratio, y1 + (y0 - y1) * ratio];
      return {
        values: {ratio},
        glyphs: [
          {shape: 'line', from: near, to: far, dashed: true},
          {shape: 'ring', at: hit.fraction < 0.5 ? near : far, radius: 0.02},
          {shape: 'dot', at: hit.fraction < 0.5 ? far : near}
        ],
        readout: `ratio ${ratio.toFixed(2)} · ${values.iterationCount} iterations`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original},
    {layer: 'dots', context: true, positions: outputs.originalPoints, radius: 5},
    {layer: 'lines', segments: outputs.segments, color: OUTPUT_COLOR, radius: 3.2}
  ]
});
