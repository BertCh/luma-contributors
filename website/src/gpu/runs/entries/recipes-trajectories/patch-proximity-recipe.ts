// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPatchProximityRecipe: core area of every patch (cells deeper than edgeDepth inside their
// patch, bright) over the class map. Thin and small patches have no core.

import {addPatchProximityRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createLandscape, createSeededRandom} from './lib/landscape';
import {addReadout} from './lib/readout';
import {parameters} from './patch-proximity-recipe.parameters';

const COLUMNS = 48;
const ROWS = 27;

export default defineOperationRun({
  parameters,
  caption: 'Patch core area: cells deeper than edgeDepth inside their patch.',
  createInputs: ({random}) => ({seed: Math.floor(random() * 1e9)}),
  build(builder, {seed}, values) {
    const landscape = createLandscape(
      createSeededRandom(seed),
      COLUMNS,
      ROWS,
      values.fragmentation,
      4
    );
    const recipe = addPatchProximityRecipe(builder.graph, {
      patches: builder.upload('patches', 'uint32', landscape.patches).view,
      patchClasses: builder.upload('patch-classes', 'uint32', landscape.patchClasses).view,
      width: COLUMNS,
      height: ROWS,
      cellSize: [30, 30],
      // Cells for raster erosion, ground units (cells of 30) for the physical model.
      edgeDepth: values.coreModel === 1 ? values.edgeDepth : values.edgeDepth * 30,
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
  rebuildOn: ['edgeDepth', 'coreModel', 'fragmentation'],
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
