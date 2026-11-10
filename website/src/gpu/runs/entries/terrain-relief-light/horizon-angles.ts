// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainHorizon on a Grand Canyon DEM: the horizon angle in every direction around every cell,
// reduced here to the sky view factor and the positive and negative openness. The directions and
// search radius are construction props; the z factor is written per frame.

import {
  getGPUTerrainHorizonParameterValues,
  GPUTerrainHorizon
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './horizon-angles.parameters';

// Sky view factor is a fraction; the openness angles are degrees.
const HORIZON_RANGES = [
  [0.5, 1],
  [60, 100],
  [60, 100]
] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['directionCount', 'maximumRadius'],
  caption: 'Elevation in; how much sky each cell sees, from its horizon in every direction, out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const horizon = builder.allocate('horizon', 'float32', cellCount * values.directionCount);
    const skyViewFactor = builder.allocate('sky-view-factor', 'float32', cellCount);
    const positiveOpenness = builder.allocate('positive-openness', 'float32', cellCount);
    const negativeOpenness = builder.allocate('negative-openness', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainHorizon({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        directionCount: values.directionCount,
        maximumRadius: values.maximumRadius,
        output: {
          horizon: horizon.view,
          skyViewFactor: skyViewFactor.view,
          positiveOpenness: positiveOpenness.view,
          negativeOpenness: negativeOpenness.view
        }
      })
    );
    return {
      outputs: {elevation, horizon, skyViewFactor, positiveOpenness, negativeOpenness},
      update: current =>
        parameterBuffer.write(getGPUTerrainHorizonParameterValues({zFactor: current.zFactor}, grid))
    };
  },
  // The pointer is a cell: a polar star shows its horizon, one spoke per direction.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const index = row * inputs.columns + column;
      const horizon = peek(outputs.horizon);
      const skyView = peek(outputs.skyViewFactor);
      const openness = peek(outputs.positiveOpenness);
      const points: number[] = [];
      if (horizon) {
        for (let direction = 0; direction < values.directionCount; direction++) {
          const angle = horizon[index * values.directionCount + direction];
          const azimuth = (2 * Math.PI * direction) / values.directionCount;
          const length = world.height * 0.14 * (0.2 + 0.8 * Math.min(Math.max(angle / 90, 0), 1));
          points.push(
            pointer.x + Math.sin(azimuth) * length,
            pointer.y + Math.cos(azimuth) * length
          );
        }
      }
      return {
        glyphs: [
          ...(points.length > 0 ? [{shape: 'polyline' as const, points, closed: true}] : []),
          {shape: 'dot' as const, at: [pointer.x, pointer.y] as const}
        ],
        readout:
          skyView && openness
            ? `sky view ${skyView[index].toFixed(2)} · openness ${Math.round(openness[index])}° · ${values.directionCount} directions`
            : `${values.directionCount} directions · ${values.maximumRadius} cells`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(
      inputs,
      [outputs.skyViewFactor, outputs.positiveOpenness, outputs.negativeOpenness][values.output],
      {range: HORIZON_RANGES[values.output], smoothing: true}
    )
  ]
});
