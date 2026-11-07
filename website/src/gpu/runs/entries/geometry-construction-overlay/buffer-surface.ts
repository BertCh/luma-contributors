// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBufferSurface grows a clean polygon surface of a given distance around lines or polygons. The
// input geometry is muted; the colored rings are the merged buffer the operation wrote.

import {
  getGPUOffsetCurveParameterValues,
  GPUBufferSurface
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createPackedSegments, createZigzagLines} from './shared/lines';
import {
  addPolygonDrawing,
  allocateOverlayOutput,
  createBlobSet,
  createRingSegments,
  getPolygonPort,
  uploadPolygonSet
} from './shared/polygons';
import {parameters} from './buffer-surface.parameters';

const JOIN_STYLES = ['round', 'mitre', 'bevel'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Buffer polygons of a given distance around lines or polygons (input muted).',
  createInputs: ({random, world}) => ({
    lines: createZigzagLines(random, world, 2, 8, 0.13),
    blobs: createBlobSet(random, world, {
      rows: 1,
      columns: 3,
      vertexCount: 7,
      irregularity: 0.5,
      fill: 0.55
    })
  }),
  build(builder, inputs, values) {
    const isLines = values.geometry === 0;
    const result = allocateOverlayOutput(builder, 'buffer', {
      vertices: 8192,
      rings: 256,
      boundary: 4096
    });
    const parameterBuffer = builder.allocate('buffer-parameters', 'float32', 4);
    const linePositions = builder.upload('line-positions', 'float32x2', inputs.lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lines.lineOffsets);
    const blobs = uploadPolygonSet(builder, 'blobs', inputs.blobs);
    builder.graph.add(
      new GPUBufferSurface({
        geometry: isLines
          ? {kind: 'lines', positions: linePositions.view, lineOffsets: lineOffsets.view}
          : getPolygonPort(blobs),
        parameters: parameterBuffer.view,
        joinStyle: JOIN_STYLES[values.joinStyle],
        capStyle: 'round',
        quadSegments: 8,
        vertexTolerance: 1e-4,
        capacity: {intersections: 4096, nodedSegments: 16384},
        output: result.output
      })
    );
    const drawing = addPolygonDrawing(builder, 'buffer', result.polygons, result.count);
    const input = builder.upload(
      'input',
      'float32x4',
      isLines
        ? createPackedSegments(inputs.lines.positions, inputs.lines.lineOffsets, false)
        : createRingSegments(inputs.blobs)
    );
    return {
      outputs: {input, ...drawing},
      update: current =>
        parameterBuffer.write(getGPUOffsetCurveParameterValues({distance: current.distance}))
    };
  },
  rebuildOn: ['joinStyle', 'geometry'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 3
    },
    {layer: 'lines', context: true, segments: outputs.input, radius: 2.5, alpha: 1}
  ]
});
