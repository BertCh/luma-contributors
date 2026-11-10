// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryMeasures over star-shaped parcels with holes. Each polygon is filled by its measured
// area (or perimeter) and a dot marks the centroid the same dispatch wrote.

import {GPUGeometryMeasures} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {
  applyVertexDrag,
  createPolygonSet,
  uploadPolygonSet,
  writePolygonSet
} from './shared/polygon-set';
import {carryNearestVertex, createDragFields, getHeroRest} from './shared/pointer-support';
import {parameters} from './geometry-measures.parameters';

const OPTIONS = {hasHoles: true, rows: 2, columns: 3};

export default defineOperationRun({
  parameters,
  caption: 'Area, perimeter and centroid of every parcel in one dispatch.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {...OPTIONS, irregularity: 0.4});
    return {seed, world, set, ...createDragFields(set, 2)};
  },
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
      update: values => {
        const next = createPolygonSet(makeRandom(inputs.seed), inputs.world, {
          ...OPTIONS,
          irregularity: values.irregularity
        });
        applyVertexDrag(next, inputs.drag);
        inputs.set = next;
        writePolygonSet(polygons, next);
      }
    };
  },
  // The pointer carries a shell vertex; the parcel's area, perimeter and centroid follow.
  interaction: {
    kind: 'drag',
    hint: 'drag a parcel corner',
    rest: ({inputs, toUnit}) => toUnit(getHeroRest(inputs)),
    resolve(pointer, {inputs, outputs, peek}) {
      const carried = carryNearestVertex(inputs, pointer, 0, 0.4);
      const feature = inputs.drag?.feature ?? 0;
      const areas = peek(outputs.areas);
      const lengths = peek(outputs.lengths);
      const centroids = peek(outputs.centroids);
      return {
        rerun: true,
        glyphs: [
          ...carried.glyphs,
          ...(centroids
            ? [
                {
                  shape: 'ring' as const,
                  at: [centroids[feature * 2], centroids[feature * 2 + 1]] as const,
                  radius: 0.02
                }
              ]
            : [])
        ],
        readout: `area ${areas ? areas[feature].toFixed(3) : '...'} · perimeter ${lengths ? lengths[feature].toFixed(2) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs, values, theme}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: values.column === 0 ? outputs.areas : outputs.lengths,
      triangleCount: inputs.set.fillTriangleCount,
      ramp: 'sequential',
      range: values.column === 0 ? [0, 0.09] : [0.8, 2.2]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'dots', positions: outputs.centroids, radius: 4, color: [...theme.ink, 1]}
  ]
});
