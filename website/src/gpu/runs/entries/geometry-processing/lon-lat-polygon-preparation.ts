// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULonLatPolygonPreparation makes longitude/latitude polygons planar-safe: it densifies
// great-circle edges, splits a polygon at the antimeridian and caps a pole-enclosing ring. The
// muted outlines are the raw input on an equirectangular map (the seam-crossing box and the polar
// triangle look wrong there); the colored rings are the prepared output, one color per ring.

import {
  getGPULonLatPolygonPreparationParameterValues,
  GPULonLatPolygonPreparation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createRingEdges} from './shared/polygon-set';
import {addEquirectangular, addOutputEdges, addRingCategories} from './shared/shape-segments';
import {parameters} from './lon-lat-polygon-preparation.parameters';

// Three features of one ring each: a box over the antimeridian, a polar triangle, a plain box.
const POSITIONS = Float32Array.of(
  170,
  5,
  -160,
  5,
  -160,
  35,
  170,
  35,
  -150,
  70,
  -30,
  70,
  90,
  70,
  -60,
  -30,
  -20,
  -30,
  -20,
  -5,
  -60,
  -5
);
const RING_OFFSETS = Uint32Array.of(0, 4, 7, 11);
const POLYGON_OFFSETS = Uint32Array.of(0, 1, 2, 3);
const FEATURE_OFFSETS = Uint32Array.of(0, 1, 2, 3);
const OUTPUT_VERTICES = 4096;

export default defineOperationRun({
  parameters,
  caption: 'Edges densified, the antimeridian polygon split, the polar ring capped.',
  createInputs: ({world}) => ({world}),
  build(builder, {world}, values) {
    const positions = builder.upload('positions', 'float32x2', POSITIONS);
    const featureOffsets = builder.upload('feature-offsets', 'uint32', FEATURE_OFFSETS);
    const polygonOffsets = builder.upload('polygon-offsets', 'uint32', POLYGON_OFFSETS);
    const ringOffsets = builder.upload('ring-offsets', 'uint32', RING_OFFSETS);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const prepared = builder.allocate('prepared', 'float32x2', OUTPUT_VERTICES);
    const preparedRings = builder.allocate('prepared-rings', 'uint32', 2 * RING_OFFSETS.length - 1);
    builder.graph.add(
      new GPULonLatPolygonPreparation({
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'none',
          edgeModel: 'great-circle'
        },
        parameters: parameterBuffer.view,
        maximumPieceCount: 1024,
        antimeridian: values.antimeridian === 1 ? 'reject' : 'split',
        poleRings: values.poleRings === 1 ? 'reject' : 'cap',
        output: {
          positions: prepared.view,
          featureOffsets: builder.allocate('prepared-features', 'uint32', FEATURE_OFFSETS.length)
            .view,
          polygonOffsets: builder.allocate(
            'prepared-polygons',
            'uint32',
            2 * POLYGON_OFFSETS.length - 1
          ).view,
          ringOffsets: preparedRings.view,
          ringFlags: builder.allocate('ring-flags', 'uint32', RING_OFFSETS.length - 1).view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {
        input: addEquirectangular(builder, 'input', positions, world),
        inputEdges: builder.upload('input-edges', 'uint32x2', createRingEdges(RING_OFFSETS)),
        prepared: addEquirectangular(builder, 'prepared', prepared, world),
        preparedEdges: addOutputEdges(builder, 'prepared', preparedRings, OUTPUT_VERTICES, true),
        categories: addRingCategories(builder, 'prepared', preparedRings, OUTPUT_VERTICES)
      },
      update: current =>
        parameterBuffer.write(
          getGPULonLatPolygonPreparationParameterValues('great-circle', {
            maximumSegmentDegrees: current.maximumSegmentDegrees
          })
        )
    };
  },
  rebuildOn: ['antimeridian', 'poleRings'],
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.input, edges: outputs.inputEdges},
    {
      layer: 'segments',
      nodes: outputs.prepared,
      edges: outputs.preparedEdges,
      values: outputs.categories,
      radius: 2.4
    }
  ]
});
