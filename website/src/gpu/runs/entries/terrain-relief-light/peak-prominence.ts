// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainPeakProminence on a Grand Canyon DEM: for every local maximum, how far it rises above the
// saddle that connects it to higher ground, from a GPU union-find over the elevation order. The
// operation has no props, so the sliders reshape the surface and pick the output shown.

import {GPUTerrainPeakProminence} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  createCellCenters,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './peak-prominence.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; prominence and key saddle height at every peak out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world, 28);
    return {
      ...terrain,
      noise: Float32Array.from(terrain.elevation, () => random()),
      centers: createCellCenters(terrain)
    };
  },
  build(builder, terrain) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const centers = builder.upload('centers', 'float32x2', terrain.centers);
    const prominence = builder.allocate('prominence', 'float32', cellCount);
    const keySaddle = builder.allocate('key-saddle', 'float32', cellCount);
    const parentPeak = builder.allocate('parent-peak', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainPeakProminence({
        grid,
        elevation: band,
        output: {
          prominence: prominence.view,
          keySaddle: keySaddle.view,
          parentPeak: parentPeak.view
        }
      })
    );
    return {
      outputs: {elevation, prominence, keySaddle, parentPeak, centers},
      update: current => {
        const surface = addTerrainNoise(terrain, terrain.noise, current.noise);
        for (let index = 0; index < surface.length; index++) surface[index] *= current.relief;
        elevation.write(surface);
      }
    };
  },
  // The pointer picks the nearest peak; a line runs to the parent peak it is measured against.
  interaction: {
    kind: 'probe',
    hint: 'hover a peak',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const prominence = peek(outputs.prominence);
      const keySaddle = peek(outputs.keySaddle);
      const parentPeak = peek(outputs.parentPeak);
      if (!prominence || !keySaddle || !parentPeak) return {readout: 'prominence of each peak'};
      let nearest = -1;
      let nearestDistance = Infinity;
      for (let index = 0; index < prominence.length; index++) {
        if (!(prominence[index] > 0)) continue;
        const [x, y] = getCellWorldPosition(
          inputs,
          world,
          index % inputs.columns,
          Math.floor(index / inputs.columns)
        );
        const distance = Math.hypot(x - pointer.x, y - pointer.y);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = index;
        }
      }
      if (nearest < 0) return {readout: 'no peaks'};
      const position = getCellWorldPosition(
        inputs,
        world,
        nearest % inputs.columns,
        Math.floor(nearest / inputs.columns)
      );
      const parent = parentPeak[nearest];
      const glyphs: RunGlyph[] = [{shape: 'crosshair', at: position}];
      if (parent !== nearest && parent < prominence.length) {
        glyphs.push({
          shape: 'line',
          from: position,
          to: getCellWorldPosition(
            inputs,
            world,
            parent % inputs.columns,
            Math.floor(parent / inputs.columns)
          ),
          dashed: true
        });
      }
      return {
        glyphs,
        readout: `prominence ${Math.round(prominence[nearest])} m · key saddle ${Math.round(keySaddle[nearest])} m`
      };
    }
  },
  draw: ({outputs, inputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    // Cells that are not peaks have zero prominence and fade out.
    {
      layer: 'dots',
      positions: outputs.centers,
      values: values.output === 0 ? outputs.prominence : outputs.keySaddle,
      fade: true,
      sizeByValue: true,
      range: values.output === 0 ? [0, 60] : [650, 2300],
      radius: 10
    }
  ]
});
