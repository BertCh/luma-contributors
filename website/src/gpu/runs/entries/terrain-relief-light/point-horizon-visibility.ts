// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointHorizonVisibility on a Grand Canyon DEM: is each summit visible from the observer? The
// observer's horizon is marched once per direction and every target is compared against it. Targets
// are colored by the code the GPU wrote (visible or hidden); the lines show the pairs.

import {
  getGPUPointHorizonVisibilityParameterValues,
  GPUPointHorizonVisibility
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
  uploadTerrain,
  VISIBILITY_COLORS
} from './support/terrain';
import {parameters} from './point-horizon-visibility.parameters';

const TARGET_COUNT = 14;

export default defineOperationRun({
  parameters,
  caption: 'Elevation, an observer and summit targets in; visible or hidden per target out.',
  createInputs: ({world}) => {
    const terrain = createTerrainInput(world);
    const peaks = findSpreadLocalMaxima(terrain, TARGET_COUNT, 7).map(index => [
      index % terrain.columns,
      Math.floor(index / terrain.columns)
    ]);
    return {...terrain, peaks};
  },
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const count = terrain.peaks.length;
    const observers = builder.allocate('observers', 'float32x4', 1);
    const targets = builder.allocate('targets', 'float32x4', count);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const visibility = builder.allocate('visibility', 'uint32', count);
    const details = builder.allocate('details', 'float32x4', count);
    const peakPositions = builder.upload(
      'peak-positions',
      'float32x2',
      Float32Array.from(
        terrain.peaks.flatMap(([column, row]) =>
          getCellWorldPosition(terrain, terrain.world, column, row)
        )
      )
    );
    const observerDot = builder.allocate('observer-dot', 'float32x2', 1);
    builder.graph.add(
      new GPUPointHorizonVisibility({
        grid,
        elevation: band,
        observers: observers.view,
        targets: targets.view,
        cellSize: terrain.cellSize,
        maximumDistance: 3600,
        parameters: parameterBuffer.view,
        output: {visibility: visibility.view, details: details.view}
      })
    );
    return {
      outputs: {elevation, visibility, peakPositions, observerDot},
      update: current => {
        const [column, row] = getCellOfPercentage(
          terrain,
          current.observerColumn,
          current.observerRow
        );
        const observer = getCellWorldPosition(terrain, terrain.world, column, row);
        observers.write(Float32Array.of(column, row, current.observerHeight, 0));
        targets.write(
          Float32Array.from(
            terrain.peaks.flatMap(([peakColumn, peakRow]) => [
              peakColumn,
              peakRow,
              current.targetHeight,
              0
            ])
          )
        );
        observerDot.write(Float32Array.from(observer));
        parameterBuffer.write(
          getGPUPointHorizonVisibilityParameterValues({maximumDistance: 3600}, grid)
        );
      }
    };
  },
  // The pointer is the observer; a sight line is drawn to every summit it can see.
  interaction: {
    kind: 'place',
    hint: 'move the observer',
    rest: getObserverRest(parameters),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const observer = getObserverUnderPointer(inputs, world, pointer);
      const visibility = peek(outputs.visibility);
      const glyphs: RunGlyph[] = [{shape: 'crosshair', at: observer.position}];
      let visibleCount = 0;
      inputs.peaks.forEach(([column, row], index) => {
        if (!visibility || visibility[index] !== 1) return;
        visibleCount++;
        glyphs.push({
          shape: 'line',
          from: observer.position,
          to: getCellWorldPosition(inputs, world, column, row)
        });
      });
      return {
        values: {observerColumn: observer.observerColumn, observerRow: observer.observerRow},
        glyphs,
        readout: visibility
          ? `observer [${observer.column}, ${observer.row}] · ${visibleCount} of ${inputs.peaks.length} summits visible`
          : `observer [${observer.column}, ${observer.row}] · ${values.observerHeight} m up`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    {
      layer: 'dots',
      positions: outputs.peakPositions,
      values: outputs.visibility,
      ramp: 'categorical',
      categories: VISIBILITY_COLORS,
      radius: 13
    },
    {layer: 'dots', positions: outputs.observerDot, radius: 7, color: [1, 1, 1, 1]}
  ]
});
