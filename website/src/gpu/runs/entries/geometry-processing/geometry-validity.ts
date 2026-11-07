// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryValidity writes a bitmask per polygon. Some polygons are damaged (bow-ties, holes
// outside their shell, repeated vertices) and every other polygon winds the wrong way for the
// default convention. Each polygon's edges take the color of its validity class: valid, self-intersection, crossing
// or outside hole, or wrong orientation.

import {GPUGeometryValidity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  createFeatureOfVertex,
  createPolygonSet,
  injectDefects,
  uploadPolygonSet
} from './shared/polygon-set';
import {addFeatureValues, addKernel} from './shared/shape-segments';
import {parameters} from './geometry-validity.parameters';

const ORIENTATIONS = ['counter-clockwise-shell', 'clockwise-shell', 'ignore'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Polygon edges colored by validity class: valid, self-intersecting, hole outside, orientation.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {
      hasHoles: true,
      rows: 3,
      columns: 5,
      mixedWinding: true
    })
  }),
  build(builder, {set}, values) {
    const polygons = uploadPolygonSet(builder, set);
    const mask = builder.allocate('mask', 'uint32', set.count);
    builder.graph.add(
      new GPUGeometryValidity({
        polygons: {
          kind: 'polygons',
          positions: polygons.positions.view,
          featureOffsets: polygons.featureOffsets.view,
          polygonOffsets: polygons.polygonOffsets.view,
          ringOffsets: polygons.ringOffsets.view
        },
        orientation: ORIENTATIONS[values.orientation],
        intersectionCapacity: 4096,
        output: {
          mask: mask.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    // Collapse the bitmask into a few classes: 0 valid, 1 self-intersection, 2 hole or ring
    // crossing, 3 orientation, 4 anything else.
    const classes = builder.allocate('classes', 'uint32', set.count);
    addKernel(builder, {
      id: 'validity-classes',
      invocationCount: set.count,
      bindings: [
        {name: 'mask', buffer: mask, access: 'read'},
        {name: 'classes', buffer: classes, access: 'read_write'}
      ],
      body: `let bits = mask[maskOffset + index];
  var result = 4u;
  if (bits == 0u) { result = 0u; }
  else if ((bits & 16u) != 0u) { result = 1u; }
  else if ((bits & 96u) != 0u) { result = 2u; }
  else if (bits == 128u) { result = 3u; }
  classes[classesOffset + index] = result;`
    });
    const featureOfVertex = builder.upload(
      'feature-of-vertex',
      'uint32',
      createFeatureOfVertex(set)
    );
    return {
      outputs: {
        ...polygons,
        vertexMasks: addFeatureValues(builder, 'class', classes, featureOfVertex)
      },
      update: current =>
        polygons.positions.write(
          injectDefects(set, current.defects, ['bow-tie', 'hole-outside', 'repeated-vertex'])
        )
    };
  },
  rebuildOn: ['orientation'],
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.vertexMasks,
      radius: 2.4
    }
  ]
});
