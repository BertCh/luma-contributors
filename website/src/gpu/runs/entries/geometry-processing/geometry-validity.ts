// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryValidity writes a bitmask per polygon. Some polygons are damaged (bow-ties, holes
// outside their shell, repeated vertices) and every other polygon winds the wrong way for the
// default convention. Each polygon's edges take the color of its validity class: valid, self-intersection, crossing
// or outside hole, or wrong orientation.

import {GPUGeometryValidity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {
  createFeatureOfVertex,
  createPolygonSet,
  getShellRows,
  injectDefects,
  uploadPolygonSet
} from './shared/polygon-set';
import {findRingCrossings} from './shared/pointer-support';
import {addFeatureValues, addKernel} from './shared/shape-segments';
import {parameters} from './geometry-validity.parameters';

const ORIENTATIONS = ['counter-clockwise-shell', 'clockwise-shell', 'ignore'] as const;
const DEFECT_KINDS = ['bow-tie', 'hole-outside', 'repeated-vertex'] as const;
const BIT_NAMES: ReadonlyArray<[number, string]> = [
  [1, 'nonFinite'],
  [2, 'unclosedRing'],
  [4, 'shortRing'],
  [8, 'repeatedVertex'],
  [16, 'selfIntersection'],
  [32, 'crossingRings'],
  [64, 'holeOutsideShell'],
  [128, 'badOrientation']
];

export default defineOperationRun({
  parameters,
  caption:
    'Polygon edges colored by validity class: valid, self-intersecting, hole outside, orientation.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {
      hasHoles: true,
      rows: 3,
      columns: 5,
      mixedWinding: true
    });
    return {set, index: createPointIndex(set.centers), focus: {feature: -1}};
  },
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
        mask,
        vertexMasks: addFeatureValues(builder, 'class', classes, featureOfVertex)
      },
      update: current => polygons.positions.write(injectDefects(set, current.defects, DEFECT_KINDS))
    };
  },
  rebuildOn: ['orientation'],
  // The pointer picks a polygon; its bitmask is decoded and the crossing is marked.
  interaction: {
    kind: 'probe',
    hint: 'hover a polygon',
    rest: ({inputs, toUnit}) => toUnit([inputs.set.centers[6], inputs.set.centers[7]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const feature = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.feature);
      inputs.focus.feature = feature;
      if (feature < 0) return {};
      const {set} = inputs;
      const positions = injectDefects(set, values.defects, DEFECT_KINDS);
      const {first, count} = getShellRows(set, feature);
      const bits = peek(outputs.mask)?.[feature];
      const names =
        bits === undefined
          ? ['...']
          : BIT_NAMES.filter(([bit]) => (bits & bit) !== 0).map(([, name]) => name);
      return {
        focus: feature,
        glyphs: [
          {
            shape: 'polyline',
            points: positions.subarray(first * 2, (first + count) * 2),
            closed: true
          },
          ...findRingCrossings(positions, first, count).map(at => ({shape: 'dot' as const, at}))
        ],
        readout: bits === 0 ? 'valid' : names.join(' · ')
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.vertexMasks,
      categories: [
        [0.435, 0.545, 0.722],
        [1, 0.42, 0.42],
        [0.96, 0.835, 0.28],
        [0.31, 0.82, 0.77],
        [0.78, 0.57, 0.92]
      ],
      radius: 2.4
    }
  ]
});
