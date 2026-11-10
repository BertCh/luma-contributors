// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainWeissLandforms on a Grand Canyon DEM: ten landform classes from topographic position at two
// scales (canyons, drainages, plains, slopes, ridges, mountain tops). The slope threshold is per
// frame; the two radii and the standardization are construction props.

import {
  getGPUTerrainWeissLandformsParameterValues,
  GPUTerrainWeissLandforms
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createMaskState, drawClassDimmer, writeClassMask} from './support/class-highlight';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './weiss-landforms.parameters';

// Index is the Weiss class (GPU_TERRAIN_WEISS_LANDFORMS); 0 marks an invalid cell and stays clear.
// Valleys are blue, ridges red and orange, plains pale, as in the geomorphons run.
const LANDFORM_NAMES = [
  'invalid',
  'canyon',
  'midslope drainage',
  'upland drainage',
  'u-shaped valley',
  'plain',
  'open slope',
  'upper slope',
  'local ridge',
  'midslope ridge',
  'mountain top'
];
const LANDFORM_COLORS = [
  [0, 0, 0, 0],
  [0.1, 0.1, 0.35, 1],
  [0.15, 0.35, 0.8, 1],
  [0.55, 0.8, 0.9, 1],
  [0.2, 0.65, 0.6, 1],
  [0.8, 0.8, 0.78, 1],
  [0.95, 0.93, 0.7, 1],
  [0.98, 0.7, 0.2, 1],
  [0.95, 0.4, 0.15, 1],
  [0.8, 0.2, 0.15, 1],
  [0.55, 0.05, 0.1, 1]
] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['smallScale', 'largeScale', 'standardization'],
  caption: 'Elevation in; ten Weiss landform classes out.',
  createInputs: ({world}) => ({...createTerrainInput(world), maskState: createMaskState()}),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const landforms = builder.allocate('landforms', 'uint32', cellCount);
    builder.graph.add(
      new GPUTerrainWeissLandforms({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        smallScale: {radius: values.smallScale},
        largeScale: {radius: values.largeScale},
        standardization: values.standardization === 1 ? 'local' : 'global',
        output: {landforms: landforms.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32', cellCount);
    return {
      outputs: {elevation, landforms, highlight},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainWeissLandformsParameterValues(
            {slopeThresholdDegrees: current.slopeThresholdDegrees},
            grid
          )
        )
    };
  },
  // The pointer is a cell: every cell of its landform class stays lit, the rest is dimmed, and the
  // two rings are the scales the class is decided from.
  interaction: {
    kind: 'probe',
    hint: 'hover a landform',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const landforms = peek(outputs.landforms);
      const worldPerCell = getWorldPerCell(inputs, world);
      const at: [number, number] = [pointer.x, pointer.y];
      const glyphs = [
        {shape: 'ring' as const, at, radius: values.smallScale * worldPerCell, dashed: true},
        {shape: 'ring' as const, at, radius: values.largeScale * worldPerCell},
        {shape: 'dot' as const, at}
      ];
      if (!landforms)
        return {
          spotlight: null,
          glyphs,
          readout: `scales ${values.smallScale} and ${values.largeScale} cells`
        };
      const selected = landforms[row * inputs.columns + column];
      writeClassMask(
        outputs.highlight,
        landforms,
        selected === 0 ? -1 : selected,
        inputs.maskState
      );
      let share = 0;
      for (const landform of landforms) if (landform === selected) share++;
      return {
        spotlight: null,
        glyphs,
        readout: `${LANDFORM_NAMES[selected] ?? 'class ' + selected} · ${Math.round((100 * share) / landforms.length)}% of cells`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.landforms, {
      ramp: 'categorical',
      categories: LANDFORM_COLORS,
      alpha: 0.9
    }),
    drawClassDimmer(inputs, outputs.highlight)
  ]
});
