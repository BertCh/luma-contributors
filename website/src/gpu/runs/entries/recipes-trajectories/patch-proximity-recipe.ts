// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPatchProximityRecipe: core area of every patch (cells deeper than edgeDepth inside their
// patch, bright) over the class map. Thin and small patches have no core.

import {addPatchProximityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {createLandscape} from './lib/landscape';
import {addReadout} from './lib/readout';
import {parameters} from './patch-proximity-recipe.parameters';

const COLUMNS = 48;
const ROWS = 27;

export default defineOperationRun({
  parameters,
  buildCache: 5,
  caption: 'Patch core area: cells deeper than edgeDepth inside their patch.',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
    const landscape = createLandscape(COLUMNS, ROWS, values.grain, 4);
    const recipe = addPatchProximityRecipe(builder.graph, {
      patches: builder.upload('patches', 'uint32', landscape.patches).view,
      patchClasses: builder.upload('patch-classes', 'uint32', landscape.patchClasses).view,
      width: COLUMNS,
      height: ROWS,
      cellSize: [landscape.cellSizeMeters, landscape.cellSizeMeters],
      // Cells for raster erosion, ground units (projected meters) for the physical model.
      edgeDepth:
        values.coreModel === 1 ? values.edgeDepth : values.edgeDepth * landscape.cellSizeMeters,
      searchRadius: 300,
      coreModel: values.coreModel === 1 ? 'raster-erosion' : 'physical'
    });
    return {
      outputs: {
        classes: builder.upload('classes', 'uint32', landscape.classes),
        core: addReadout(builder, 'core', recipe.outputs.core)
      },
      update: () => undefined
    };
  },
  rebuildOn: ['edgeDepth', 'coreModel', 'grain'],
  // The pointer's x is the edge depth, drawn as a ring of that many cells; it settles when the
  // pointer rests, over five cached builds.
  interaction: {
    kind: 'window',
    hint: 'set the edge depth',
    rest: ({toUnit, world}) => {
      const [, unitY] = toUnit([0, world.height / 2]);
      return [0.05 + ((parameters[0].default - 1) / 4) * 0.9, unitY];
    },
    resolve(pointer, {values, world}) {
      const edgeDepth = clamp(Math.round(1 + ((pointer.unitX - 0.05) / 0.9) * 4), 1, 5);
      const cellSize = world.height / ROWS;
      return {
        settle: {edgeDepth},
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: edgeDepth * cellSize},
          {shape: 'crosshair', at: [pointer.x, pointer.y], size: 5}
        ],
        readout: `edgeDepth ${edgeDepth} cells · ${['physical', 'raster-erosion'][values.coreModel]}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.classes, columns: COLUMNS, rows: ROWS, alpha: 0.45},
    {
      layer: 'raster',
      values: outputs.core,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'sequential',
      range: [0, 1],
      fade: true
    }
  ]
});
