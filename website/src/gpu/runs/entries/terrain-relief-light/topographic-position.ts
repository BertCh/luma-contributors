// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainTopographicPosition on a synthetic DEM: the largest standardized deviation from the local
// mean over several window radii, and the radius at which it occurs. Radii are construction props.

import {GPUTerrainTopographicPosition} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createTerrainInput, drawElevationContext, drawGrid, uploadTerrain} from './support/terrain';
import {parameters} from './topographic-position.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; the strongest position signal over three scales out.',
  createInputs: ({random, world}) => createTerrainInput(random, world),
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
  draw: ({inputs, outputs, values}) => [
    drawElevationContext(inputs, outputs.elevation),
    values.output === 0
      ? drawGrid(inputs, outputs.deviation, {ramp: 'diverging', smoothing: true})
      : drawGrid(inputs, outputs.radius)
  ]
});
