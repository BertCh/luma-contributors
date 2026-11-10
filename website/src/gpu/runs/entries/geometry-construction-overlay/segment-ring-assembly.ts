// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegmentRingAssembly chains loose directed segments into closed rings. The input is the
// shuffled, slightly jittered edge list of adjacent Census tracts (muted). With
// `cancelOpposingSegments`, the two opposite copies of every shared edge cancel and one outer ring
// remains; without it each cell closes on its own. Endpoints join when they are within
// `vertexTolerance`, so jitter beyond the tolerance leaves rings open.

import {GPUSegmentRingAssembly} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createTractCoverage} from './shared/tract-coverage';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {addPackedSegments, findNearestLine} from './shared/lines';
import {createRingSegments} from './shared/polygons';
import {parameters} from './segment-ring-assembly.parameters';

function getHash(index: number): number {
  const value = Math.sin(index * 12.9898 + 4.1) * 43758.5453;
  return value - Math.floor(value);
}

export default defineOperationRun({
  parameters,
  caption: 'Loose segments assembled into rings; one color per assembled ring.',
  createInputs: ({random, world}) => {
    const coverage = createTractCoverage(random, world, 3);
    const segments = createRingSegments(coverage);
    const segmentCount = segments.length / 4;
    const order = Array.from({length: segmentCount}, (_, index) => index).sort(
      (first, second) => getHash(first) - getHash(second)
    );
    const getSegments = (jitter: number) =>
      Float32Array.from({length: segments.length}, (_, slot) => {
        const row = Math.floor(slot / 4);
        const source = order[row];
        return (
          segments[source * 4 + (slot % 4)] +
          (getHash(row * 4 + (slot % 4) + 99) - 0.5) * 2 * jitter
        );
      });
    return {segmentCount, getSegments};
  },
  build(builder, inputs, values) {
    const initial = inputs.getSegments(values.jitter);
    const endpoints = builder.upload('endpoints', 'float32x4', initial);
    const ringCapacity = inputs.segmentCount;
    const ringOffsets = builder.allocate('ring-offsets', 'uint32', ringCapacity + 1);
    const positions = builder.allocate(
      'ring-positions',
      'float32x2',
      inputs.segmentCount + ringCapacity
    );
    const count = builder.allocate('ring-count', 'uint32', 1);
    const overflow = builder.allocate('ring-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUSegmentRingAssembly({
        endpoints: endpoints.view,
        vertexTolerance: values.vertexTolerance,
        cancelOpposingSegments: values.cancelOpposingSegments === 1,
        normalizeWinding: true,
        output: {
          ringOffsets: ringOffsets.view,
          positions: positions.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'rings', positions, ringOffsets, {lineCount: count});
    return {
      outputs: {endpoints, ringOffsets, positions, count, ...drawing},
      update: current => endpoints.write(inputs.getSegments(current.jitter))
    };
  },
  rebuildOn: ['cancelOpposingSegments', 'vertexTolerance'],
  // The pointer picks an assembled ring: it is drawn bright, with a vertexTolerance circle at each
  // joint where two loose segments were chained.
  interaction: {
    kind: 'probe',
    hint: 'hover a ring',
    resolve(pointer, {outputs, values, peek}) {
      const ringPositions = peek(outputs.positions);
      const offsets = peek(outputs.ringOffsets);
      const count = peek(outputs.count)?.[0];
      if (!ringPositions || !offsets || !count) return {};
      const hit = findNearestLine(
        {positions: ringPositions, lineOffsets: offsets.subarray(0, count + 1)},
        pointer.x,
        pointer.y
      );
      if (!hit) return {};
      const first = offsets[hit.line];
      const end = offsets[hit.line + 1];
      const segmentCount = end - first - 1;
      // The joint nearest to the pointer: the two loose segments chained there lie within
      // vertexTolerance of each other, so a circle of that radius marks the join.
      let joint = first;
      let jointDistance = Infinity;
      for (let row = first; row < end - 1; row++) {
        const distance = Math.hypot(
          ringPositions[row * 2] - pointer.x,
          ringPositions[row * 2 + 1] - pointer.y
        );
        if (distance < jointDistance) {
          jointDistance = distance;
          joint = row;
        }
      }
      const before = joint > first ? joint - 1 : end - 2;
      const after = joint + 1;
      const at = (row: number) => [ringPositions[row * 2], ringPositions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: at(before), to: at(joint)},
        {shape: 'line', from: at(joint), to: at(after)},
        {
          shape: 'ring',
          at: at(joint),
          radius: Math.max(values.vertexTolerance, 0.012),
          dashed: true
        }
      ];
      return {
        focus: hit.line,
        glyphs,
        readout: `joint ${joint - first} of ${segmentCount} · tolerance ${values.vertexTolerance.toFixed(4)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.endpoints, radius: 1.5},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 4.5
    }
  ]
});
