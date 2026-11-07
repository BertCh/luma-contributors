// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineClipByPolygon keeps the parts of lines inside (or outside) polygons. Muted lines are the
// input; the colored pieces are what the operation kept, one color per piece.

import {GPULineClipByPolygon} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  allocatePieceOutput,
  createLineSet,
  createPackedSegments
} from './shared/lines';
import {
  createBlobSet,
  createRingSegments,
  getPolygonPort,
  uploadPolygonSet
} from './shared/polygons';
import {createSeededRandom} from './shared/triangulation';
import {parameters} from './line-clip-by-polygon.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Lines clipped by polygons: the pieces kept inside or outside are colored.',
  createInputs: ({random, world}) => ({
    lines: createLineSet(random, world, 5, 14, 1.2),
    getBlobs: (shift: number) =>
      createBlobSet(createSeededRandom(31), world, {
        rows: 2,
        columns: 3,
        vertexCount: 14,
        irregularity: 0.3,
        fill: 0.7,
        shiftX: shift,
        shiftY: shift * 0.5
      })
  }),
  build(builder, inputs, values) {
    const blobs = inputs.getBlobs(values.shift);
    const polygons = uploadPolygonSet(builder, 'polygons', blobs);
    const positions = builder.upload('positions', 'float32x2', inputs.lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', inputs.lines.lineOffsets);
    const result = allocatePieceOutput(builder, 'clipped', 256, 2048);
    builder.graph.add(
      new GPULineClipByPolygon({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        polygons: getPolygonPort(polygons),
        mode: values.mode === 0 ? 'inside' : 'outside',
        intersectionCapacity: 2048,
        candidateCapacity: 4096,
        output: {pieces: result.pieces}
      })
    );
    const drawing = addPackedSegments(builder, 'clipped', result.positions, result.offsets, {
      lineCount: result.count
    });
    const original = builder.upload(
      'original',
      'float32x4',
      createPackedSegments(inputs.lines.positions, inputs.lines.lineOffsets, false)
    );
    const outline = builder.allocate('outline', 'float32x4', createRingSegments(blobs).length / 4);
    return {
      outputs: {original, outline, ...drawing},
      update: current => {
        const next = inputs.getBlobs(current.shift);
        polygons.positions.write(next.positions);
        outline.write(createRingSegments(next));
      }
    };
  },
  rebuildOn: ['mode'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 2},
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      color: [0.5, 0.6, 0.8, 0.7],
      radius: 1.4
    },
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 5
    }
  ]
});
