// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSegmentSnap snaps the line's endpoints to nearby reference vertices and inserts every
// reference vertex within the tolerance of a segment. The bright path is the output the
// operation wrote (its vertex count comes from the output counts); dots are the references.

import {GPUSegmentSnap} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createIdentityIds, createSingleLine} from './shared/polygon-set';
import {parameters} from './segment-snap.parameters';

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
      references.push(x + (random() - 0.5) * 0.08, y + (random() - 0.5) * 0.08);
    }
    return {line, references: Float32Array.from(references)};
  },
  build(builder, {line, references}, values) {
    const positions = builder.upload('positions', 'float32x2', line.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const referencePositions = builder.upload('references', 'float32x2', references);
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
        snapped,
        counts,
        ids: builder.upload('ids', 'uint32', createIdentityIds(MAXIMUM_VERTICES))
      }
    };
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
    {layer: 'dots', positions: outputs.referencePositions, radius: 3, color: [1, 0.6, 0.2, 1]}
  ]
});
