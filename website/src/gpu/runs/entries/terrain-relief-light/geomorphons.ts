// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeomorphons on a Grand Canyon DEM: ten landform classes (flat, peak, ridge, shoulder, spur, slope,
// hollow, footslope, valley, pit) from line-of-sight patterns in eight directions. The flat
// threshold is per frame; the search radius and comparison rule are construction props.

import {
  getGPUGeomorphonsParameterValues,
  GPUGeomorphons,
  type GPUGeomorphonComparison
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createMaskState, drawClassDimmer, writeClassMask} from './support/class-highlight';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './geomorphons.parameters';

// Index is the landform code (GPU_GEOMORPHON_FORMS); 0 marks an invalid edge cell, drawn muted.
const FORM_COLORS = [
  [0.45, 0.47, 0.52, 0.35],
  [0.8, 0.8, 0.78, 1],
  [0.75, 0.1, 0.1, 1],
  [0.95, 0.4, 0.15, 1],
  [0.98, 0.7, 0.2, 1],
  [0.7, 0.8, 0.25, 1],
  [0.95, 0.93, 0.7, 1],
  [0.55, 0.8, 0.9, 1],
  [0.2, 0.65, 0.6, 1],
  [0.15, 0.35, 0.8, 1],
  [0.1, 0.1, 0.35, 1]
] as const;

/** Box-blurs the elevation so landforms are large and coherent instead of single-cell noise. */
function smoothElevation(terrain: ReturnType<typeof createTerrainInput>, passes: number): void {
  const {columns, rows, elevation} = terrain;
  for (let pass = 0; pass < passes; pass++) {
    const source = elevation.slice();
    for (let row = 1; row < rows - 1; row++) {
      for (let column = 1; column < columns - 1; column++) {
        let sum = 0;
        for (let offset = 0; offset < 9; offset++) {
          sum += source[(row + Math.floor(offset / 3) - 1) * columns + column + (offset % 3) - 1];
        }
        elevation[row * columns + column] = sum / 9;
      }
    }
  }
}

const FORM_NAMES = [
  'edge',
  'flat',
  'peak',
  'ridge',
  'shoulder',
  'spur',
  'slope',
  'hollow',
  'footslope',
  'valley',
  'pit'
];

// Column and row steps of the eight search directions.
const DIRECTIONS = [
  [0, -1],
  [1, -1],
  [1, 0],
  [1, 1],
  [0, 1],
  [-1, 1],
  [-1, 0],
  [-1, -1]
] as const;

const COMPARISONS: GPUGeomorphonComparison[] = ['anglev1', 'anglev2', 'anglev2-distance'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['searchRadius', 'comparison'],
  caption: 'Elevation in; one of ten landform classes per cell out.',
  createInputs: ({world}) => {
    const terrain = createTerrainInput(world);
    smoothElevation(terrain, 3);
    return {...terrain, maskState: createMaskState()};
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const forms = builder.allocate('forms', 'uint32', cellCount);
    builder.graph.add(
      new GPUGeomorphons({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        searchRadius: values.searchRadius,
        skipRadius: 1,
        comparison: COMPARISONS[values.comparison],
        output: {forms: forms.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32', cellCount);
    return {
      outputs: {elevation, forms, highlight},
      update: current =>
        parameterBuffer.write(
          getGPUGeomorphonsParameterValues(
            {flatThresholdDegrees: current.flatThresholdDegrees},
            grid
          )
        )
    };
  },
  // The pointer is a cell: eight spokes run out to the searchRadius, and every cell of the same
  // landform form stays lit while the rest is dimmed.
  interaction: {
    kind: 'probe',
    hint: 'hover a landform',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const glyphs: RunGlyph[] = [];
      let higher = 0;
      let lower = 0;
      const base = inputs.elevation[row * inputs.columns + column];
      const thresholdTangent = Math.tan((values.flatThresholdDegrees * Math.PI) / 180);
      for (const [stepColumn, stepRow] of DIRECTIONS) {
        let zenith = -Infinity;
        let nadir = Infinity;
        let reached = 0;
        for (let step = 2; step <= values.searchRadius; step++) {
          const sampleColumn = column + stepColumn * step;
          const sampleRow = row + stepRow * step;
          if (
            sampleColumn < 0 ||
            sampleRow < 0 ||
            sampleColumn >= inputs.columns ||
            sampleRow >= inputs.rows
          ) {
            break;
          }
          const tangent =
            (inputs.elevation[sampleRow * inputs.columns + sampleColumn] - base) /
            (step * inputs.cellSize * Math.hypot(stepColumn, stepRow));
          zenith = Math.max(zenith, tangent);
          nadir = Math.min(nadir, tangent);
          reached = step;
        }
        if (reached > 0) {
          if (zenith > thresholdTangent && zenith >= -nadir) higher++;
          else if (-nadir > thresholdTangent) lower++;
          glyphs.push({
            shape: 'line',
            from: getCellWorldPosition(inputs, world, column, row),
            to: getCellWorldPosition(
              inputs,
              world,
              column + stepColumn * reached,
              row + stepRow * reached
            )
          });
        }
      }
      glyphs.push({shape: 'dot', at: getCellWorldPosition(inputs, world, column, row)});
      const forms = peek(outputs.forms);
      if (!forms) return {spotlight: null, glyphs, readout: `${higher} higher, ${lower} lower`};
      const selected = forms[row * inputs.columns + column];
      writeClassMask(outputs.highlight, forms, selected === 0 ? -1 : selected, inputs.maskState);
      return {
        spotlight: null,
        glyphs,
        readout: `${FORM_NAMES[selected] ?? 'form'} · ${higher} up ${lower} down · ${values.searchRadius} cells`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.forms, {ramp: 'categorical', categories: FORM_COLORS, alpha: 0.9}),
    drawClassDimmer(inputs, outputs.highlight)
  ]
});
