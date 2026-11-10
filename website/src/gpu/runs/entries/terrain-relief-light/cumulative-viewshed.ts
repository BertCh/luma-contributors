// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCumulativeViewshed on a Grand Canyon DEM: how many of a set of observers (towers on
// summits spread over the frame) can see each cell. The observer count is a construction prop;
// their height and range are per-frame values, and so are the tower positions, which the pointer
// carries.

import {
  getGPUTerrainLineOfSightParameterValues,
  GPUTerrainCumulativeViewshed
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  findSpreadLocalMaxima,
  getCellOfWorldPosition,
  getCellWorldPosition,
  getWorldPerMeter,
  uploadTerrain
} from './support/terrain';

import {parameters} from './cumulative-viewshed.parameters';

// Index is the number of towers that see the cell; cells nobody sees stay clear.
const COUNT_COLORS = [
  [0, 0, 0, 0],
  [0.99, 0.8, 0.3],
  [0.96, 0.6, 0.2],
  [0.9, 0.4, 0.2],
  [0.8, 0.2, 0.3],
  [0.6, 0.1, 0.4],
  [0.4, 0.1, 0.5],
  [0.25, 0.1, 0.55],
  [0.15, 0.1, 0.6]
] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['observerCount'],
  caption: 'Elevation and several towers in; how many towers see each cell out.',
  createInputs: ({world}) => {
    const terrain = createTerrainInput(world);
    const towers = findSpreadLocalMaxima(terrain, 8, 9).map(
      index => [index % terrain.columns, Math.floor(index / terrain.columns)] as [number, number]
    );
    // The first tower is the one the pointer carries; `towers` holds the current cells.
    return {...terrain, home: towers.map(tower => [...tower]), towers};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const observerCount = Math.min(values.observerCount, terrain.towers.length);
    const observers = builder.allocate('observers', 'float32x2', observerCount);
    const towers = builder.allocate('towers', 'float32x2', observerCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibleCount = builder.allocate('visible-count', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainCumulativeViewshed({
        grid,
        elevation: band,
        observers: observers.view,
        parameters: parameterBuffer.view,
        output: {visibleCount: visibleCount.view}
      })
    );
    return {
      outputs: {elevation, visibleCount, towers},
      update: current => {
        const cells = terrain.towers.slice(0, observerCount);
        observers.write(Float32Array.from(cells.flat()));
        towers.write(
          Float32Array.from(
            cells.flatMap(([column, row]) =>
              getCellWorldPosition(terrain, terrain.world, column, row)
            )
          )
        );
        parameterBuffer.write(
          getGPUTerrainLineOfSightParameterValues(
            {observerHeight: current.observerHeight, maximumDistance: current.maximumDistance},
            grid
          )
        );
      }
    };
  },
  // The pointer carries the first tower; the rest stay on their summits.
  interaction: {
    kind: 'drag',
    hint: 'carry a tower',
    rest: ({inputs, world}) => {
      const [x, y] = getCellWorldPosition(inputs, world, inputs.home[0][0], inputs.home[0][1]);
      return [x / world.width, 1 - y / world.height];
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      inputs.towers[0] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const [column, row] = inputs.towers[0];
      const counts = peek(outputs.visibleCount);
      let overlapShare = 0;
      if (counts) {
        let seenByTwo = 0;
        for (const count of counts) if (count >= 2) seenByTwo++;
        overlapShare = Math.round((100 * seenByTwo) / counts.length);
      }
      const position = getCellWorldPosition(inputs, world, column, row);
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'ring',
            at: position,
            radius: values.maximumDistance * getWorldPerMeter(inputs, world),
            dashed: true
          },
          {shape: 'crosshair', at: position}
        ],
        readout: counts
          ? `tower 1 [${column}, ${row}] · seen by 2+ towers: ${overlapShare}%`
          : `tower 1 [${column}, ${row}] · sees ${values.maximumDistance} m`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.visibleCount, {
      ramp: 'categorical',
      categories: COUNT_COLORS,
      alpha: 0.85
    }),
    {layer: 'dots', positions: outputs.towers, radius: 6, color: [1, 1, 1, 1]}
  ]
});
