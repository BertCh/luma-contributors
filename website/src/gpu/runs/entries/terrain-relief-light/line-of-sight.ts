// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainLineOfSight on a Grand Canyon DEM: one observer and many targets, each pair tested against
// the terrain between them. The clearance (meters above the terrain along the sight line; negative
// when blocked) colors each target. Positions and heights are per-frame values.

import {
  getGPUTerrainLineOfSightParameterValues,
  GPUTerrainLineOfSight
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getObserverRest, getObserverUnderPointer} from './support/pointer';
import {
  createTerrainInput,
  drawElevationContext,
  findSpreadLocalMaxima,
  getCellOfPercentage,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './line-of-sight.parameters';

const TARGET_COUNT = 28;

export default defineOperationRun({
  parameters,
  caption: 'Elevation and observer-target pairs in; the clearance of every sight line out.',
  createInputs: ({world}) => {
    const terrain = createTerrainInput(world);
    // Targets are summits spread over the frame, so blocked and clear sight lines both occur.
    const targets = findSpreadLocalMaxima(terrain, TARGET_COUNT, 5).map(index => [
      index % terrain.columns,
      Math.floor(index / terrain.columns)
    ]);
    return {...terrain, targets};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const targetCount = terrain.targets.length;
    const pairs = builder.allocate('pairs', 'float32x4', targetCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibility = builder.allocate('visibility', 'uint32', targetCount);
    const clearance = builder.allocate('clearance', 'float32', targetCount);
    const targetPositions = builder.upload(
      'target-positions',
      'float32x2',
      Float32Array.from(
        terrain.targets.flatMap(([column, row]) =>
          getCellWorldPosition(terrain, terrain.world, column, row)
        )
      )
    );
    const observerDot = builder.allocate('observer-dot', 'float32x2', 1);
    builder.graph.add(
      new GPUTerrainLineOfSight({
        grid,
        elevation: band,
        pairs: pairs.view,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view, clearance: clearance.view}
      })
    );
    return {
      outputs: {elevation, clearance, targetPositions, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        const [observerX, observerY] = getCellWorldPosition(terrain, terrain.world, column, row);
        pairs.write(
          Float32Array.from(
            terrain.targets.flatMap(([targetColumn, targetRow]) => [
              column,
              row,
              targetColumn,
              targetRow
            ])
          )
        );
        observerDot.write(Float32Array.of(observerX, observerY));
        parameterBuffer.write(
          getGPUTerrainLineOfSightParameterValues(
            {observerHeight: current.observerHeight, targetHeight: current.targetHeight},
            grid
          )
        );
      }
    };
  },
  // The pointer is the observer; a sight line is drawn to every target it can see.
  interaction: {
    kind: 'place',
    hint: 'move the observer',
    rest: getObserverRest(parameters),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const observer = getObserverUnderPointer(inputs, world, pointer);
      const clearance = peek(outputs.clearance);
      const glyphs: RunGlyph[] = [{shape: 'crosshair', at: observer.position}];
      let clearCount = 0;
      inputs.targets.forEach(([column, row], index) => {
        if (!clearance || !(clearance[index] >= 0)) return;
        clearCount++;
        glyphs.push({
          shape: 'line',
          from: observer.position,
          to: getCellWorldPosition(inputs, world, column, row)
        });
      });
      return {
        values: {observerColumn: observer.observerColumn, observerRow: observer.observerRow},
        glyphs,
        readout: clearance
          ? `observer [${observer.column}, ${observer.row}] · ${clearCount} of ${inputs.targets.length} clear`
          : `observer [${observer.column}, ${observer.row}] · ${values.observerHeight} m up`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {
      layer: 'dots',
      positions: outputs.targetPositions,
      values: outputs.clearance,
      ramp: 'diverging',
      range: [-80, 80],
      radius: 9
    },
    {layer: 'dots', positions: outputs.observerDot, radius: 6, color: [1, 1, 1, 1]}
  ]
});
