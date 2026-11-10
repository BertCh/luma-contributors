// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainSummits on a Grand Canyon DEM: a cell is a summit when it is the highest within the radius
// and drops by at least minimumDrop on every side. The radius and drop are per-frame values;
// the border rule is a construction prop.

import {
  getGPUTerrainSummitsParameterValues,
  GPUTerrainSummits
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellWorldPosition,
  getWorldPerMeter,
  uploadTerrain
} from './support/terrain';

import {parameters} from './terrain-summits.parameters';

export default defineOperationRun({
  parameters,
  rebuildOn: ['incompleteNeighborhood'],
  caption: "Elevation in; the summit mask and each summit's drop out.",
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world, 30);
    return {...terrain, noise: Float32Array.from(terrain.elevation, () => random())};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const summitMask = builder.allocate('summit-mask', 'uint32', cellCount);
    const drop = builder.allocate('drop', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainSummits({
        grid,
        elevation: band,
        maximumRadiusPixels: 16,
        incompleteNeighborhood: values.incompleteNeighborhood === 1 ? 'ignore' : 'reject',
        parameters: parameterBuffer.view,
        output: {summitMask: summitMask.view, drop: drop.view}
      })
    );
    return {
      outputs: {elevation, drop, summitMask},
      update: current => {
        elevation.write(addTerrainNoise(terrain, terrain.noise, current.noise));
        parameterBuffer.write(
          getGPUTerrainSummitsParameterValues(
            {radius: current.radius, minimumDrop: current.minimumDrop},
            grid
          )
        );
      }
    };
  },
  // The pointer height is the radius (low is small); the ring at the cursor and around every
  // summit is the neighborhood within which a summit must be the highest cell.
  interaction: {
    kind: 'reach',
    hint: 'size the radius',
    rest: [0.5, 1 - getFractionOf(findParameter(parameters, 'radius'), 350)],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const radius = getParameterAt(findParameter(parameters, 'radius'), 1 - pointer.unitY);
      const worldRadius = radius * getWorldPerMeter(inputs, world);
      const mask = peek(outputs.summitMask);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: [pointer.x, pointer.y], radius: worldRadius},
        {shape: 'dot', at: [pointer.x, pointer.y]}
      ];
      let summitCount = 0;
      if (mask) {
        for (let index = 0; index < mask.length; index++) {
          if (mask[index] !== 1) continue;
          summitCount++;
          if (summitCount <= 24) {
            glyphs.push({
              shape: 'ring',
              at: getCellWorldPosition(
                inputs,
                world,
                index % inputs.columns,
                Math.floor(index / inputs.columns)
              ),
              radius: worldRadius,
              dashed: true
            });
          }
        }
      }
      return {
        values: {radius},
        glyphs,
        readout: `radius ${radius} m${mask ? ` · ${summitCount} summits` : ''}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.summitMask, {
      ramp: 'categorical',
      categories: [
        [0, 0, 0, 0],
        [0.95, 0.45, 0.1, 1]
      ]
    })
  ]
});
