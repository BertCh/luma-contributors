// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULabelPoint finds the interior point farthest from each polygon's edge. Polygons are filled by
// that distance; the white dots are the label anchors the operation wrote.

import {GPULabelPoint} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {createPointIndex} from '../../pick';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {createStaleGuard, readFresh} from './shared/pointer-support';
import {parameters} from './label-point.parameters';

const ROUNDS = findParameter(parameters, 'refinementRounds');

export default defineOperationRun({
  parameters,
  caption: 'Label anchors at the widest part of each polygon.',
  buildCache: 5,
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {irregularity: 0.2, bend: 0.6});
    return {
      seed,
      world,
      set,
      index: createPointIndex(set.centers),
      pointGuard: createStaleGuard(),
      distanceGuard: createStaleGuard()
    };
  },
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
        output: {points: points.view, publishedCenterClearance: distances.view}
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
  // The pointer's x is the number of refinement rounds (five prebuilt values); the hovered polygon shows its anchor.
  interaction: {
    kind: 'time',
    hint: 'refine the search',
    rest: [getFractionOf(ROUNDS, ROUNDS.default), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const refinementRounds = getParameterAt(ROUNDS, pointer.unitX);
      const row = inputs.index.nearest(pointer.x, pointer.y);
      const key = `${values.initialGridSize}/${values.refinementRounds}`;
      const points = readFresh(inputs.pointGuard, key, peek(outputs.points));
      const distances = readFresh(inputs.distanceGuard, key, peek(outputs.distances));
      const hasAnchor = points && distances && row >= 0;
      return {
        settle: {refinementRounds},
        glyphs: hasAnchor
          ? [
              {shape: 'ring', at: [points[row * 2], points[row * 2 + 1]], radius: distances[row]},
              {shape: 'dot', at: [points[row * 2], points[row * 2 + 1]]}
            ]
          : [],
        readout:
          refinementRounds === values.refinementRounds
            ? `refinementRounds ${refinementRounds} · distance ${hasAnchor ? distances[row].toFixed(3) : '...'}`
            : `refinementRounds ${values.refinementRounds} -> ${refinementRounds}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.distances,
      triangleCount: inputs.set.fillTriangleCount,
      range: [0, 0.1]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'dots', positions: outputs.points, radius: 5, color: [...theme.ink, 1]}
  ]
});
