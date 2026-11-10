// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegmentSnap snaps the line's endpoints to nearby reference vertices and inserts every
// reference vertex within the tolerance of a segment. The bright path is the output the
// operation wrote (its vertex count comes from the output counts); dots are the references.

import {GPUSegmentSnap} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findNearestOnPolyline} from '../../pick';
import {parameters} from './segment-snap.parameters';
import {createIdentityIds, createSingleLine} from './shared/polygon-set';

const MAXIMUM_VERTICES = 400;

export default defineOperationRun({
  parameters,
  caption: 'Reference vertices near the line are inserted into it.',
  createInputs: ({random, world}) => {
    const line = createSingleLine(random, world, 14);
    const references: number[] = [];
    for (let index = 0; index < 40; index++) {
      const segment = Math.floor(random() * (line.count - 1));
      const amount = random();
      const x =
        line.positions[segment * 2] * (1 - amount) + line.positions[segment * 2 + 2] * amount;
      const y =
        line.positions[segment * 2 + 1] * (1 - amount) + line.positions[segment * 2 + 3] * amount;
      references.push(x + (random() - 0.5) * 0.2, y + (random() - 0.5) * 0.2);
    }
    const referencePositions = Float32Array.from(references);
    // The carried reference starts as the one lying just inside the default tolerance.
    let carriedIndex = 0;
    let carriedGap = Infinity;
    for (let index = 0; index < references.length / 2; index++) {
      const distance =
        findNearestOnPolyline(line.positions, references[index * 2], references[index * 2 + 1])
          ?.distance ?? 0;
      const gap = Math.abs(distance - 0.8 * parameters[0].default);
      if (gap < carriedGap) {
        carriedGap = gap;
        carriedIndex = index;
      }
    }
    return {
      line,
      references: referencePositions,
      carriedIndex,
      carried: [referencePositions[carriedIndex * 2], referencePositions[carriedIndex * 2 + 1]] as [
        number,
        number
      ]
    };
  },
  build(builder, inputs, values) {
    const {line, references} = inputs;
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const referencePositions = builder.allocate('references', 'float32x2', references.length / 2);
    const carriedDot = builder.allocate('carried', 'float32x2', 1);
    const isInserted = builder.allocate('is-inserted', 'uint32', references.length / 2);
    const snapped = builder.allocate('snapped', 'float32x2', MAXIMUM_VERTICES);
    const sourceEdges = builder.allocate('source-edges', 'uint32', MAXIMUM_VERTICES);
    const counts = builder.allocate('counts', 'uint32x4', 1);
    builder.graph.add(
      new GPUSegmentSnap({
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        referencePositions: referencePositions.view,
        tolerance: values.tolerance,
        maximumVerticesPerLine: MAXIMUM_VERTICES,
        output: {positions: snapped.view, sourceEdges: sourceEdges.view, counts: counts.view}
      })
    );
    return {
      outputs: {
        positions,
        referencePositions,
        isInserted,
        carriedDot,
        snapped,
        counts,
        ids: builder.upload('ids', 'uint32', createIdentityIds(MAXIMUM_VERTICES))
      },
      update: current => {
        const moved = Float32Array.from(references);
        moved.set(inputs.carried, inputs.carriedIndex * 2);
        referencePositions.write(moved);
        carriedDot.write(Float32Array.from(inputs.carried));
        isInserted.write(
          Uint32Array.from({length: moved.length / 2}, (_, index) =>
            (findNearestOnPolyline(line.positions, moved[index * 2], moved[index * 2 + 1])
              ?.distance ?? Infinity) <= current.tolerance
              ? 1
              : 0
          )
        );
      }
    };
  },
  // The pointer carries one reference vertex; it is inserted only while it lies within the tolerance.
  interaction: {
    kind: 'drag',
    hint: 'drag a reference vertex',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.references[inputs.carriedIndex * 2],
        inputs.references[inputs.carriedIndex * 2 + 1]
      ]),
    resolve(pointer, {inputs, values, peek, outputs}) {
      inputs.carried = [pointer.x, pointer.y];
      const hit = findNearestOnPolyline(inputs.line.positions, pointer.x, pointer.y);
      const isInserted = (hit?.distance ?? Infinity) <= values.tolerance;
      const vertexCount = peek(outputs.counts)?.[0];
      return {
        rerun: true,
        glyphs: [
          {shape: 'ring', at: inputs.carried, radius: values.tolerance, dashed: true},
          ...(hit
            ? [{shape: 'line' as const, from: inputs.carried, to: hit.point, dashed: true}]
            : []),
          {shape: 'crosshair', at: inputs.carried}
        ],
        readout: `tolerance ${values.tolerance.toFixed(3)} · ${isInserted ? 'inserted' : 'left out'}${vertexCount === undefined ? '' : ` · ${vertexCount} vertices`}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: inputs.line.count
    },
    {layer: 'paths', nodes: outputs.snapped, ids: outputs.ids, count: outputs.counts, radius: 2},
    {
      layer: 'dots',
      positions: outputs.referencePositions,
      values: outputs.isInserted,
      categories: [
        [0.55, 0.6, 0.72],
        [1, 0.6, 0.2]
      ],
      radius: 5
    },
    {layer: 'dots', positions: outputs.carriedDot, radius: 9, color: [1, 0.6, 0.2, 1]}
  ]
});
