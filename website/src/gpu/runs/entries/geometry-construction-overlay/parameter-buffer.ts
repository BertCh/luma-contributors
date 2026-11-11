// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUParameterBuffer carries a per-frame value into a compiled graph. Here the tolerance of a line
// simplification is written into one on every slider move: the graph is compiled once and only
// re-encoded, so dragging is cheap. The shore of Lake Superior is the muted input; kept vertices are the
// output.

import {createGreatLakeRings} from '../../../data/hydrography';
import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {defineOperationRun} from '../../define-run';
import {findParameter} from '../../interaction';
import {getLogarithmicReach, getLogarithmicReachDistance} from './shared/reach';
import {
  addSelectedSegments,
  createPackedSegments,
  createRowLines,
  findNearestLine,
  findRestBesideLines,
  OUTPUT_COLOR
} from './shared/lines';
import {createBadge} from './shared/badge';
import {parameters} from './parameter-buffer.parameters';

// Distance from the shore, in world units, at which the tolerance reaches its maximum. The tolerance
// grows logarithmically with the distance, so a hand near the shore picks a fine tolerance and one
// far from it a coarse one. The span is inside the lake's reach, so the whole range of the
// parameter is reachable from within the picture.
const REACH_SPAN = 0.12;

const toleranceParameter = findParameter(parameters, 'tolerance');

export default defineOperationRun({
  parameters,
  caption: 'The tolerance lives in a GPUParameterBuffer; the graph is compiled once.',
  createInputs: ({world}) => {
    const [shore] = createGreatLakeRings(world, 'Lake Superior');
    const positions = Float32Array.from([...shore, shore[0]].flat());
    // Calls of `build` and of the parameter buffer's `write` made by this scene, for the badge.
    const counters = {builds: 0, writes: 0};
    return {positions, lineOffsets: Uint32Array.of(0, positions.length / 2), world, counters};
  },
  build(builder, line) {
    line.counters.builds++;
    const vertexCount = line.positions.length / 2;
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const importance = builder.allocate('importance', 'float32', vertexCount);
    const ids = builder.allocate('kept-ids', 'uint32', vertexCount);
    const count = builder.allocate('kept-count', 'uint32', 1);
    const overflow = builder.allocate('kept-overflow', 'uint32', 1);
    const keptMask = builder.allocate('kept-mask', 'uint32', vertexCount);
    // The real parameter buffer of the operation (an application-owned, per-frame storage buffer).
    const parameterBuffer = builder.own(
      new GPUParameterBuffer(builder.device, {
        id: 'tolerance',
        format: 'float32',
        length: 4
      })
    );
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
      outputs: {
        original,
        simplified,
        positions,
        keptMask,
        keptCount: count,
        counters: line.counters
      },
      update: current => {
        line.counters.writes++;
        parameterBuffer.write(
          getGPULineSimplificationParameterValues({tolerance: current.tolerance})
        );
      }
    };
  },
  // The distance from the pointer to the shoreline is the tolerance, written into the parameter
  // buffer each frame: the graph is compiled once and only re-encoded.
  interaction: {
    kind: 'reach',
    hint: 'move off the shore',
    rest: ({inputs, toUnit}) =>
      toUnit(
        findRestBesideLines(
          inputs,
          getLogarithmicReachDistance(toleranceParameter, toleranceParameter.default, REACH_SPAN),
          (x, y) => findNearestLine(inputs, x, y)?.distance ?? 0,
          inputs.world,
          1e-4
        )
      ),
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestLine(inputs, pointer.x, pointer.y);
      if (!hit) return {};
      const tolerance = getLogarithmicReach(toleranceParameter, hit.distance, REACH_SPAN);
      const keptCount = peek(outputs.keptCount)?.[0];
      return {
        values: {tolerance},
        glyphs: [
          {shape: 'ring', at: hit.point, radius: tolerance, dashed: true},
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: hit.point}
        ],
        readout: `tolerance ${tolerance.toFixed(3)} · compiled once${keptCount === undefined ? '' : ` · kept ${keptCount}`}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 4},
    {layer: 'lines', segments: outputs.simplified, color: OUTPUT_COLOR, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.keptMask,
      ramp: 'sequential',
      range: [0.5, 1],
      radius: 4.5
    },
    {
      layer: 'text',
      ...createBadge({
        lines: [
          `tolerance ${values.tolerance.toFixed(3)}`,
          `this graph: built ${outputs.counters.builds}\u00d7`,
          `parameter buffer written ${outputs.counters.writes}\u00d7`
        ]
      })
    }
  ]
});
