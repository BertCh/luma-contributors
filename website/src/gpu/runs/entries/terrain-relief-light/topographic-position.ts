// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainTopographicPosition on a Grand Canyon DEM: the largest standardized deviation from the local
// mean over several window radii, and the radius at which it occurs. Radii are construction props.

import {GPUTerrainTopographicPosition} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './topographic-position.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; the strongest position signal over three scales out.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const deviation = builder.allocate('deviation', 'float32', cellCount);
    const radius = builder.allocate('radius', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainTopographicPosition({
        grid,
        elevation: band,
        scales: [
          {radius: values.smallRadius},
          {radius: Math.round((values.smallRadius + values.largeRadius) / 2)},
          {radius: values.largeRadius}
        ],
        output: {maximumDeviation: deviation.view, maximumDeviationRadius: radius.view}
      })
    );
    return {outputs: {elevation, deviation, radius}};
  },
  // The pointer is a cell: three rings are the scales compared there, the winning one solid.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const index = row * inputs.columns + column;
      const radii = [
        values.smallRadius,
        Math.round((values.smallRadius + values.largeRadius) / 2),
        values.largeRadius
      ];
      const deviation = peek(outputs.deviation);
      const winner = peek(outputs.radius);
      const worldPerCell = getWorldPerCell(inputs, world);
      const at: [number, number] = [pointer.x, pointer.y];
      return {
        glyphs: [
          ...radii.map(radius => ({
            shape: 'ring' as const,
            at,
            radius: radius * worldPerCell,
            dashed: !winner || winner[index] !== radius
          })),
          {shape: 'dot' as const, at}
        ],
        readout:
          deviation && winner
            ? `deviation ${deviation[index].toFixed(1)} at radius ${winner[index]} cells`
            : `scales ${radii.join(', ')} cells`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    values.output === 0
      ? drawGrid(inputs, outputs.deviation, {ramp: 'diverging', range: [-3, 3], smoothing: true})
      : drawGrid(inputs, outputs.radius)
  ]
});
