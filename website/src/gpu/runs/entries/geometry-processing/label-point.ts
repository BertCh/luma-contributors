// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULabelPoint finds the interior point farthest from each polygon's edge. Polygons are filled by
// that distance; the white dots are the label anchors the operation wrote.

import {GPULabelPoint} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {parameters} from './label-point.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Label anchors at the widest part of each polygon.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {irregularity: 0.2, bend: 0.6})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const points = builder.allocate('points', 'float32x2', inputs.set.count);
    const distances = builder.allocate('distances', 'float32', inputs.set.count);
    builder.graph.add(
      new GPULabelPoint({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        initialGridSize: values.initialGridSize,
        refinementRounds: values.refinementRounds,
        output: {points: points.view, distances: distances.view}
      })
    );
    return {
      outputs: {...polygons, points, distances},
      update: current =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            irregularity: 0.2,
            bend: current.bend
          })
        )
    };
  },
  rebuildOn: ['initialGridSize', 'refinementRounds'],
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.distances,
      triangleCount: inputs.set.fillTriangleCount
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'dots', positions: outputs.points, radius: 5, color: [...theme.ink, 1]}
  ]
});
