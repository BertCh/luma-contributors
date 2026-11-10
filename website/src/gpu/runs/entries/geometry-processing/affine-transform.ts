// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAffineTransform rotates, scales and skews every polygon about its own centroid in one
// per-frame matrix. The muted outlines are the input; the bright edges are the output positions.

import {
  getGPUAffineTransformParameterValues,
  GPUAffineTransform
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {aimAround} from '../../interaction';
import {createPolygonSet, uploadPolygonSet} from './shared/polygon-set';

import {parameters} from './affine-transform.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Each polygon rotated, scaled and skewed about its centroid.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {hasHoles: true})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const transformed = builder.allocate(
      'transformed',
      'float32x2',
      inputs.set.positions.length / 2
    );
    builder.graph.add(
      new GPUAffineTransform({
        geometryType: 'polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        origins: ['centroid'],
        parameters: parameterBuffer.view,
        output: {positions: transformed.view}
      })
    );
    return {
      outputs: {...polygons, transformed},
      update: values =>
        parameterBuffer.write(
          getGPUAffineTransformParameterValues({
            rotate: values.rotate,
            scale: values.scale,
            skew: [values.skew, 0],
            origin: 'centroid'
          })
        )
    };
  },
  // The pointer's angle around the picture center is the rotation and its distance the scale.
  interaction: aimAround(parameters, {
    hint: 'aim the rotation',
    angle: 'rotate',
    distance: {parameter: 'scale', reach: 0.6},
    readout: values => `rotate ${values.rotate}° · scale ${values.scale.toFixed(2)}`
  }),
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'segments', nodes: outputs.transformed, edges: outputs.edges, radius: 1.8}
  ]
});
