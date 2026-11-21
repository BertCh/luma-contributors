// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsAlgebra combines two weights: k nearest neighbors (blue) and a distance band
// (green). The thick orange links underneath are the combined weights the operation wrote.

import {GPUSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createKnnWeights, createUniformPoints} from '../../inputs';
import {createLegend} from '../../text';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './spatial-weights-algebra.parameters';
import {
  addLinkSegments,
  allocateWeights,
  getRowLinkGlyphs,
  getSegmentTargets,
  uploadWeightsBuffers,
  stretchPoints
} from './shared/weights';

const OPERATIONS = ['union', 'intersection', 'difference', 'symmetric-difference'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Left (blue) and right (green) weights, and their combination (orange).',
  createInputs: ({random, world}) => {
    const points = stretchPoints(createUniformPoints(random, world, 70, 0.1), world);
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const left = uploadWeightsBuffers(
      builder,
      'left',
      createKnnWeights(points.positions, values.k)
    );
    const right = uploadWeightsBuffers(
      builder,
      'right',
      createDistanceBandWeights(points.positions, values.radius)
    );
    const combined = allocateWeights(
      builder,
      'combined',
      points.count,
      left.capacity + right.capacity
    );
    builder.graph.add(
      new GPUSpatialWeightsAlgebra({
        operation: OPERATIONS[values.operation],
        left: left.input,
        right: right.input,
        output: combined.port
      })
    );
    return {
      outputs: {
        positions,
        leftLinks: addLinkSegments(builder, 'left', left, positions),
        rightLinks: addLinkSegments(builder, 'right', right, positions),
        combinedLinks: addLinkSegments(builder, 'combined', combined, positions)
      }
    };
  },
  // The pointer picks a point: its band (ring), its kNN links (dashed) and the combined links it keeps.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const kept = getSegmentTargets(peek(outputs.combinedLinks), from);
      const nearest = getSegmentTargets(peek(outputs.leftLinks), from);
      const band = getSegmentTargets(peek(outputs.rightLinks), from);
      return {
        focus: row,
        glyphs: [
          {shape: 'ring', at: from, radius: values.radius, dashed: true},
          ...getRowLinkGlyphs(from, nearest, {dashed: true, dots: false}),
          ...getRowLinkGlyphs(from, kept)
        ],
        readout: `${OPERATIONS[values.operation]} · kNN ${nearest.length}, band ${band.length} → ${kept.length}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'lines',
      segments: outputs.combinedLinks,
      color: [1, 0.55, 0.2, 0.85],
      radius: 3
    },
    {
      layer: 'lines',
      segments: outputs.leftLinks,
      color: [0.35, 0.6, 1, 0.55],
      radius: 1
    },
    {
      layer: 'lines',
      segments: outputs.rightLinks,
      color: [0.35, 0.85, 0.5, 0.65],
      radius: 1
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 4},
    {
      layer: 'text',
      ...createLegend({
        corner: 'top-left',
        entries: [
          {label: 'left: k nearest', swatch: [0.35, 0.6, 1]},
          {label: 'right: distance band', swatch: [0.35, 0.85, 0.5]},
          {label: `combined: ${OPERATIONS[values.operation]}`, swatch: [1, 0.55, 0.2]}
        ]
      })
    }
  ]
});
