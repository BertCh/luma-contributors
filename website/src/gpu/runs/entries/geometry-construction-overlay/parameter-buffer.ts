// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUParameterBuffer carries a per-frame value into a compiled graph. Here the tolerance of a line
// simplification is written into one on every slider move: the graph is compiled once and only
// re-encoded, so dragging is cheap. A noisy coastline is the muted input; kept vertices are the
// output.

import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification,
  GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addSelectedSegments,
  createPackedSegments,
  createRowLines,
  OUTPUT_COLOR
} from './shared/lines';
import {parameters} from './parameter-buffer.parameters';

const VERTEX_COUNT = 260;

export default defineOperationRun({
  parameters,
  caption: 'The tolerance lives in a GPUParameterBuffer; the graph is compiled once.',
  createInputs: ({random, world}) => {
    const phases = Array.from({length: 6}, () => random() * Math.PI * 2);
    const positions = new Float32Array(VERTEX_COUNT * 2);
    for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
      const angle = (vertex / (VERTEX_COUNT - 1)) * Math.PI * 2;
      let radius = 0.36;
      for (let harmonic = 0; harmonic < phases.length; harmonic++) {
        radius += (0.06 / (harmonic + 1)) * Math.sin((harmonic + 2) * angle + phases[harmonic]);
      }
      radius += (random() - 0.5) * 0.012;
      positions[vertex * 2] = world.width / 2 + Math.cos(angle) * radius * 1.25;
      positions[vertex * 2 + 1] = 0.5 + Math.sin(angle) * radius;
    }
    return {positions, lineOffsets: Uint32Array.of(0, VERTEX_COUNT)};
  },
  build(builder, line) {
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const importance = builder.allocate('importance', 'float32', VERTEX_COUNT);
    const ids = builder.allocate('kept-ids', 'uint32', VERTEX_COUNT);
    const count = builder.allocate('kept-count', 'uint32', 1);
    const overflow = builder.allocate('kept-overflow', 'uint32', 1);
    const keptMask = builder.allocate('kept-mask', 'uint32', VERTEX_COUNT);
    // The real parameter buffer of the operation (an application-owned, per-frame storage buffer).
    const parameterBuffer = new GPUParameterBuffer(builder.device, {
      id: 'tolerance',
      format: 'float32',
      length: 4
    });
    builder.graph.add(
      new GPULineSimplification({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        importance: importance.view,
        parameters: parameterBuffer.importToGraph(builder.graph),
        output: {
          ids: ids.view,
          outputMask: keptMask.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const rowLines = builder.upload('row-lines', 'uint32', createRowLines(line.lineOffsets));
    const simplified = addSelectedSegments(builder, 'simplified', positions, rowLines, ids, count);
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(line.positions, line.lineOffsets, false)
    );
    return {
      outputs: {original, simplified, positions, keptMask},
      update: current =>
        parameterBuffer.write(
          getGPULineSimplificationParameterValues({tolerance: current.tolerance})
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4},
    {layer: 'lines', segments: outputs.simplified, color: OUTPUT_COLOR, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.keptMask,
      ramp: 'sequential',
      range: [0.5, 1],
      radius: 4.5
    }
  ]
});
