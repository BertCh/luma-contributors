// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addBuildingTessellationRecipe builds morphological tessellation: the boundary of every building
// is sampled, the samples' Voronoi cells are clipped to an enclosure, and the cells of one building
// are dissolved into its territory (the space closest to it). Footprints are muted; each building's
// territory ring is drawn in its own color from the polygons the recipe wrote.

import {addBuildingTessellationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
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
  createInputs: ({world}) => {
    const layout = createCityLayout(world, 10, true);
    return {
      layout,
      world,
      index: createPointIndex(Float32Array.from(layout.centers.flat())),
      carried: {building: -1, offset: [0, 0] as [number, number]}
    };
  },
  build(builder, inputs, values) {
    const buildings = getBuildings(inputs.layout, values.fill);
    const segments = builder.upload('walls', 'float32x4', buildings.segments);
    const owners = builder.upload('wall-owners', 'uint32', buildings.segmentBuildingIds);
    const all = inputs.layout.rings.flat();
    const left = Math.min(...all.map(point => point[0])) - 0.07;
    const right = Math.max(...all.map(point => point[0])) + 0.07;
    const bottom = Math.min(...all.map(point => point[1])) - 0.07;
    const top = Math.max(...all.map(point => point[1])) + 0.07;
    const enclosure = builder.upload(
      'enclosure',
      'float32x2',
      Float32Array.of(left, bottom, right, bottom, right, top, left, top)
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
        {closed: true, status: output.status}
      )
    );
    const walls = builder.allocate('wall-segments', 'float32x4', buildings.segments.length / 4);
    return {
      outputs: {walls, territories},
      update: current => {
        const next = getBuildings(inputs.layout, current.fill, 0, inputs.carried);
        segments.write(next.segments);
        walls.write(next.segments);
      }
    };
  },
  rebuildOn: ['sampleSpacing'],
  // The pointer carries the nearest building (it leans toward the cursor and lets go at a distance);
  // the territories are recomputed around it each frame.
  interaction: {
    kind: 'drag',
    hint: 'move a building',
    rest: ({inputs, toUnit}) => toUnit(inputs.layout.centers[0]),
    resolve(pointer, {inputs}) {
      const building = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.carried.building);
      if (building < 0) return {};
      const [centerX, centerY] = inputs.layout.centers[building];
      const deltaX = pointer.x - centerX;
      const deltaY = pointer.y - centerY;
      const weight = clamp((0.3 - Math.hypot(deltaX, deltaY)) / 0.22, 0, 1);
      const offset: [number, number] = [deltaX * weight, deltaY * weight];
      const changed =
        building !== inputs.carried.building ||
        offset[0] !== inputs.carried.offset[0] ||
        offset[1] !== inputs.carried.offset[1];
      inputs.carried.building = building;
      inputs.carried.offset = offset;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'polyline',
            points: inputs.layout.rings[building].flatMap(([x, y]) => [
              x + offset[0],
              y + offset[1]
            ]),
            closed: true
          },
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `building ${building} · territories re-form`
      };
    }
  },
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
