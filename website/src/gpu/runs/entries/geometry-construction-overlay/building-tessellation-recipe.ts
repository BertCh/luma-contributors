// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addBuildingTessellationRecipe builds morphological tessellation: the boundary of every building
// is sampled, the samples' Voronoi cells are clipped to an enclosure, and the cells of one building
// are dissolved into its territory (the space closest to it). Footprints are muted; each building's
// territory ring is drawn in its own color from the polygons the recipe wrote.

import {addBuildingTessellationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getViewBuffer} from './shared/kernel';
import {addPolygonDrawing} from './shared/polygons';
import {createCityLayout, getBuildings} from './shared/urban';
import {parameters} from './building-tessellation-recipe.parameters';

const PALETTE = [
  [0.35, 0.66, 1, 1],
  [1, 0.7, 0.3, 1],
  [1, 0.42, 0.42, 1],
  [0.31, 0.82, 0.77, 1],
  [0.55, 0.85, 0.35, 1],
  [0.78, 0.58, 0.93, 1]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Each building gets the territory closest to it; colored outlines are those territories.',
  createInputs: ({random, world}) => ({
    layout: createCityLayout(random, world, 3, 2, 0.06),
    world
  }),
  build(builder, inputs, values) {
    const buildings = getBuildings(inputs.layout, values.fill);
    const segments = builder.upload('walls', 'float32x4', buildings.segments);
    const owners = builder.upload('wall-owners', 'uint32', buildings.segmentBuildingIds);
    const margin = 0.02;
    const enclosure = builder.upload(
      'enclosure',
      'float32x2',
      Float32Array.of(
        margin,
        margin,
        inputs.world.width - margin,
        margin,
        inputs.world.width - margin,
        1 - margin,
        margin,
        1 - margin
      )
    );
    const result = addBuildingTessellationRecipe(builder.graph, {
      segments: segments.view,
      segmentBuildingIds: owners.view,
      buildingCount: buildings.count,
      enclosure: enclosure.view,
      sampleSpacing: values.sampleSpacing,
      sampleCapacity: 128,
      vertexTolerance: 1e-3,
      capacity: {intersections: 4096, nodedSegments: 4096, vertices: 1024, rings: 64}
    });
    const territories = result.outputs.map((output, building) =>
      addPolygonDrawing(
        builder,
        `territory-${building}`,
        {
          positions: getViewBuffer(output.geometry.positions),
          featureOffsets: getViewBuffer(output.geometry.featureOffsets),
          polygonOffsets: getViewBuffer(output.geometry.polygonOffsets),
          ringOffsets: getViewBuffer(output.geometry.ringOffsets)
        },
        getViewBuffer(output.status.count),
        {closed: true}
      )
    );
    const walls = builder.allocate('wall-segments', 'float32x4', buildings.segments.length / 4);
    return {
      outputs: {walls, territories},
      update: current => {
        const next = getBuildings(inputs.layout, current.fill);
        segments.write(next.segments);
        walls.write(next.segments);
      }
    };
  },
  rebuildOn: ['sampleSpacing'],
  draw: ({outputs}) => [
    ...outputs.territories.map((territory, building) => ({
      layer: 'lines' as const,
      segments: territory.segments,
      color: PALETTE[building % PALETTE.length],
      radius: 2.6
    })),
    {layer: 'lines', context: true, segments: outputs.walls, radius: 3}
  ]
});
