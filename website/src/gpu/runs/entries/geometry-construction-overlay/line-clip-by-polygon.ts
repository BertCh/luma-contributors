// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineClipByPolygon keeps the parts of lines inside (or outside) polygons. Muted lines are the
// input; the colored pieces are what the operation kept, one color per piece.

import {GPULineClipByPolygon} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
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
    world,
    drag: [0, 0] as [number, number],
    lines: createLineSet(random, world, 8, 14),
    getBlobs: (shiftX: number, shiftY: number) =>
      createBlobSet(createSeededRandom(31), world, {
        rows: 2,
        columns: 3,
        vertexCount: 14,
        irregularity: 0.3,
        fill: 0.95,
        shiftX,
        shiftY
      })
  }),
  build(builder, inputs, values) {
    const blobs = inputs.getBlobs(values.shift, values.shift * 0.5);
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
        const next = inputs.getBlobs(
          current.shift + inputs.drag[0],
          current.shift * 0.5 + inputs.drag[1]
        );
        polygons.positions.write(next.positions);
        outline.write(createRingSegments(next));
      }
    };
  },
  rebuildOn: ['mode'],
  // The pointer carries the polygons: their displacement from the center, in cells, is added to the
  // slider's shift and the lines are clipped against the new positions.
  interaction: {
    kind: 'drag',
    hint: 'drag the polygons',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, values}) {
      const center: [number, number] = [inputs.world.width / 2, inputs.world.height / 2];
      inputs.drag = [
        clamp((pointer.x - center[0]) / (inputs.world.width / 3), -0.7, 0.7),
        clamp((pointer.y - center[1]) / (inputs.world.height / 2), -0.7, 0.7)
      ];
      return {
        rerun: true,
        glyphs: [
          {shape: 'arrow', from: center, to: [pointer.x, pointer.y]},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `keep ${values.mode === 0 ? 'inside' : 'outside'} · shift ${(values.shift + inputs.drag[0]).toFixed(2)}, ${(values.shift * 0.5 + inputs.drag[1]).toFixed(2)} cells`
      };
    }
  },
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
      radius: 6
    }
  ]
});
