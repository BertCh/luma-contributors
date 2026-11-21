// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBufferSurface grows a clean polygon surface of a given distance around lines or polygons. The
// input geometry is muted; the colored rings are the merged buffer the operation wrote.

import {
  getGPUOffsetCurveParameterValues,
  GPUBufferSurface
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {
  createBendSet,
  createPackedSegments,
  findNearestLine,
  findRestBesideLines
} from './shared/lines';
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
    world,
    lines: createBendSet(world, 2, 16),
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
    const drawing = addPolygonDrawing(builder, 'buffer', result.polygons, result.count, {
      status: result.output.status
    });
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
  // The distance from the pointer to the nearest input line or ring is the buffer distance, so the
  // edge of the merged surface passes under the cursor.
  interaction: {
    kind: 'reach',
    hint: 'move off the line',
    rest: ({inputs, toUnit}) => {
      const distance = findParameter(parameters, 'distance').default;
      return toUnit(
        findRestBesideLines(
          inputs.lines,
          distance,
          (x, y) => findNearestLine(inputs.lines, x, y)?.distance ?? 0,
          inputs.world
        )
      );
    },
    resolve(pointer, {inputs, values}) {
      const isLines = values.geometry === 0;
      const hit = findNearestLine(
        isLines
          ? inputs.lines
          : {positions: inputs.blobs.positions, lineOffsets: inputs.blobs.ringOffsets},
        pointer.x,
        pointer.y,
        !isLines
      );
      if (!hit) return {};
      const distance = snapToParameter(findParameter(parameters, 'distance'), hit.distance);
      return {
        values: {distance},
        glyphs: [
          {shape: 'ring', at: hit.point, radius: distance, dashed: true},
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: hit.point}
        ],
        readout: `distance ${distance.toFixed(3)} · ${JOIN_STYLES[values.joinStyle]} joins`
      };
    }
  },
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
