// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURectangleClip cuts wandering lines and parcel polygons to an axis-aligned rectangle. Muted lines are the input,
// the outlined box is the per-frame clip rectangle, and the bright edges are the clipped
// positions the operation wrote, linked by the output line offsets.

import {
  getGPURectangleClipParameterValues,
  GPURectangleClip
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {createCrossingLines, createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {addBoxSegments, addOutputEdges} from './shared/shape-segments';
import {parameters} from './rectangle-clip.parameters';

const CENTER_X = findParameter(parameters, 'centerX');
const CENTER_Y = findParameter(parameters, 'centerY');

export default defineOperationRun({
  parameters,
  caption:
    'Lines and parcels clipped to the rectangle; the pieces inside come from the output buffers.',
  createInputs: ({seed, world}) => ({
    world,
    lines: createCrossingLines(makeRandom(seed), world, 4, 5, 16, 0.08),
    parcels: createPolygonSet(makeRandom(seed + 1), world, {rows: 2, columns: 3, fill: 0.7})
  }),
  build(builder, {lines, parcels, world}) {
    const positions = builder.upload('line-positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const capacity = lines.vertexCount * 2;
    const clipped = builder.allocate('clipped', 'float32x2', capacity);
    const clippedOffsets = builder.allocate('clipped-offsets', 'uint32', capacity);
    const pieceCount = builder.allocate('count', 'uint32', 1);
    builder.graph.add(
      new GPURectangleClip({
        geometryType: 'lines',
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          positions: clipped.view,
          lineOffsets: clippedOffsets.view,
          status: {
            count: builder.allocate('line-count', 'uint32', 1).view,
            vertexCount: pieceCount.view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    // The same rectangle cuts the parcels: rings that leave it are trimmed, rings outside vanish.
    const parcelBuffers = uploadPolygonSet(builder, parcels);
    const ringCapacity = parcels.positions.length / 2 + (parcels.ringOffsets.length - 1) * 8;
    const clippedRings = builder.allocate('clipped-rings', 'float32x2', ringCapacity);
    const clippedRingOffsets = builder.allocate(
      'clipped-ring-offsets',
      'uint32',
      parcels.ringOffsets.length
    );
    builder.graph.add(
      new GPURectangleClip({
        id: 'parcel-clip',
        geometryType: 'polygons',
        positions: parcelBuffers.positions.view,
        featureOffsets: parcelBuffers.featureOffsets.view,
        polygonOffsets: parcelBuffers.polygonOffsets.view,
        ringOffsets: parcelBuffers.ringOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          positions: clippedRings.view,
          lineOffsets: clippedRingOffsets.view,
          status: {
            count: builder.allocate('parcel-count', 'uint32', 1).view,
            overflow: builder.allocate('parcel-overflow', 'uint32', 1).view
          }
        }
      })
    );
    const box = builder.allocate('box', 'float32x4', 1);
    return {
      outputs: {
        parcelOutline: parcelBuffers.outline,
        clippedRings,
        ringEdges: addOutputEdges(builder, 'rings', clippedRingOffsets, ringCapacity, true),
        box,
        pieceCount,
        boxSegments: addBoxSegments(builder, 'box', box),
        clipped,
        edges: addOutputEdges(builder, 'clipped', clippedOffsets, capacity, false),
        inputSegments: builder.upload(
          'input-segments',
          'float32x4',
          Float32Array.from([...lines.firstSegments, ...lines.secondSegments])
        )
      },
      update: values => {
        const half = values.size * 0.5;
        const settings = {
          minX: values.centerX - half * world.height * 1.4,
          minY: values.centerY - half,
          maxX: values.centerX + half * world.height * 1.4,
          maxY: values.centerY + half
        };
        parameterBuffer.write(getGPURectangleClipParameterValues(settings));
        box.write(Float32Array.of(settings.minX, settings.minY, settings.maxX, settings.maxY));
      }
    };
  },
  // The pointer is the center of the clip rectangle; its size stays the slider.
  interaction: {
    kind: 'window',
    hint: 'move the clip window',
    rest: ({world, toUnit}) => toUnit([CENTER_X.default, CENTER_Y.default * world.height]),
    resolve(pointer, {outputs, values, peek}) {
      const centerX = snapToParameter(CENTER_X, pointer.x);
      const centerY = snapToParameter(CENTER_Y, pointer.y);
      const pieces = peek(outputs.pieceCount)?.[0];
      return {
        values: {centerX, centerY},
        glyphs: [{shape: 'crosshair', at: [centerX, centerY]}],
        readout: `size ${values.size.toFixed(2)} · ${pieces ?? '...'} vertices kept`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.inputSegments},
    {
      layer: 'lines',
      context: true,
      segments: outputs.parcelOutline,
      count: inputs.parcels.outlineCount
    },
    {layer: 'lines', segments: outputs.boxSegments, color: [1, 0.6, 0.2, 1]},
    {
      layer: 'segments',
      nodes: outputs.clippedRings,
      edges: outputs.ringEdges,
      color: [0.4, 0.75, 1, 1],
      radius: 2.2
    },
    {layer: 'segments', nodes: outputs.clipped, edges: outputs.edges, radius: 2.6}
  ]
});
