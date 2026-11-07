// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryMeasures over star-shaped parcels with holes. Each polygon is filled by its measured
// area (or perimeter) and a dot marks the centroid the same dispatch wrote.

import {GPUGeometryMeasures} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet, writePolygonSet} from './shared/polygon-set';
import {parameters} from './geometry-measures.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Area, perimeter and centroid of every parcel in one dispatch.',
  createInputs: ({seed, world}) => ({
    seed,
    world,
    set: createPolygonSet(makeRandom(seed), world, {hasHoles: true})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const areas = builder.allocate('areas', 'float32', inputs.set.count);
    const lengths = builder.allocate('lengths', 'float32', inputs.set.count);
    const centroids = builder.allocate('centroids', 'float32x2', inputs.set.count);
    builder.graph.add(
      new GPUGeometryMeasures({
        geometryType: 'polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        output: {areas: areas.view, lengths: lengths.view, centroids: centroids.view}
      })
    );
    return {
      outputs: {...polygons, areas, lengths, centroids},
      update: values =>
        writePolygonSet(
          polygons,
          createPolygonSet(makeRandom(inputs.seed), inputs.world, {
            hasHoles: true,
            irregularity: values.irregularity
          })
        )
    };
  },
  draw: ({inputs, outputs, values, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: values.column === 0 ? outputs.areas : outputs.lengths,
      triangleCount: inputs.set.fillTriangleCount,
      ramp: 'sequential'
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'dots', positions: outputs.centroids, radius: 4, color: [...theme.ink, 1]}
  ]
});
