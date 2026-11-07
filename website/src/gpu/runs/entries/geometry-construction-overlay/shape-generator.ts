// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUShapeGenerator writes a circle, sector or ellipse ring around every center. The segment count
// and radius scale are per-frame parameters, so smoothness changes without recompiling. Rings are
// colored by feature; dots are the vertices the operation wrote, squares of small dots the centers.

import {
  getGPUShapeGeneratorParameterValues,
  GPUShapeGenerator,
  getGPUShapeVertexCount
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, HIGHLIGHT_COLOR} from './shared/lines';
import {parameters} from './shape-generator.parameters';

const SHAPES = ['circle', 'sector', 'ellipse'] as const;
const FEATURE_COUNT = 8;
const MAXIMUM_SEGMENTS = 48;

export default defineOperationRun({
  parameters,
  caption: 'Rings generated around centers: circles, sectors or rotated ellipses.',
  createInputs: ({random, world}) => ({
    centers: Float32Array.from({length: FEATURE_COUNT * 2}, (_, slot) => {
      const feature = Math.floor(slot / 2);
      const column = feature % 4;
      const row = Math.floor(feature / 4);
      return slot % 2 === 0
        ? ((column + 0.5) / 4) * world.width
        : (row + 0.5) / 2 + (random() - 0.5) * 0.04;
    }),
    radii: Float32Array.from({length: FEATURE_COUNT}, () => 0.07 + 0.06 * random()),
    ellipseRadii: Float32Array.from({length: FEATURE_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? 0.1 + 0.05 * random() : 0.05 + 0.03 * random()
    ),
    bearings: Float32Array.from({length: FEATURE_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? 40 * (slot / 2) : 40 * (slot / 2 - 0.5) + 100 + 120 * random()
    ),
    rotations: Float32Array.from({length: FEATURE_COUNT}, (_, feature) => feature * 25)
  }),
  build(builder, inputs, values) {
    const shape = SHAPES[values.shape];
    const maximumVertices = getGPUShapeVertexCount(shape, MAXIMUM_SEGMENTS);
    const centers = builder.upload('centers', 'float32x2', inputs.centers);
    const radii = builder.upload('radii', 'float32', inputs.radii);
    const ellipseRadii = builder.upload('ellipse-radii', 'float32x2', inputs.ellipseRadii);
    const bearings = builder.upload('bearings', 'float32x2', inputs.bearings);
    const rotations = builder.upload('rotations', 'float32', inputs.rotations);
    const positions = builder.allocate(
      'ring-positions',
      'float32x2',
      FEATURE_COUNT * maximumVertices
    );
    const offsets = builder.allocate('ring-offsets', 'uint32', FEATURE_COUNT + 1);
    const parameterBuffer = builder.allocate('shape-parameters', 'float32', 2);
    const common = {
      centers: centers.view,
      maximumSegments: MAXIMUM_SEGMENTS,
      parameters: parameterBuffer.view,
      output: {positions: positions.view, offsets: offsets.view}
    };
    builder.graph.add(
      new GPUShapeGenerator(
        shape === 'ellipse'
          ? {shape, radii: ellipseRadii.view, rotations: rotations.view, ...common}
          : shape === 'sector'
            ? {shape, radii: radii.view, bearings: bearings.view, ...common}
            : {shape, radii: radii.view, ...common}
      )
    );
    const drawing = addPackedSegments(builder, 'rings', positions, offsets, {
      closed: shape !== 'sector'
    });
    return {
      outputs: {centers, ...drawing},
      update: current =>
        parameterBuffer.write(
          getGPUShapeGeneratorParameterValues({
            segmentCount: current.segmentCount,
            radiusScale: current.radiusScale
          })
        )
    };
  },
  rebuildOn: ['shape'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.4
    },
    {layer: 'dots', positions: outputs.points, color: HIGHLIGHT_COLOR, radius: 2.6},
    {layer: 'dots', context: true, positions: outputs.centers, radius: 3}
  ]
});
