// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPycnophylactic turns zone totals into a smooth surface whose cells still sum to each zone's
// total. The muted raster shows tract totals before redistribution.

import {GPUPycnophylactic} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {createTracts} from '../../inputs';
import {parameters} from './pycnophylactic.parameters';
import {createRunGrid} from './support/grid';
import {addTractZones, addZoneGather, countTractCells} from './support/zones';

export default defineOperationRun({
  parameters,
  caption: 'Chicago tract populations redistributed to cells while preserving each tract total.',
  createInputs: ({world}) => ({grid: createRunGrid(world, 40), world}),
  build(builder, {grid, world}, values) {
    const tracts = createTracts(makeRandom(11), world, Math.sqrt(values.zoneCount));
    const zones = addTractZones(builder, grid, tracts);
    const totals = builder.upload('totals', 'float32', tracts.population);
    // The totals are drawn per cell, on the same range as the redistributed surface.
    const cells = countTractCells(grid, tracts);
    let rangeMaximum = 0;
    tracts.population.forEach((total, tract) => {
      rangeMaximum = Math.max(rangeMaximum, total / Math.max(1, cells[tract]));
    });
    const surface = builder.allocate('surface', 'float32', grid.columns * grid.rows);
    builder.graph.add(
      new GPUPycnophylactic({
        grid: grid.grid,
        zones: zones.view,
        zoneCount: tracts.count,
        totals: totals.view,
        maximumIterations: values.maximumIterations,
        kernel: values.kernel === 0 ? 'rook' : 'box',
        output: {mass: surface.view}
      })
    );
    return {
      outputs: {
        surface,
        rangeMaximum,
        totals: addZoneGather(
          builder,
          zones,
          totals,
          'tract-totals',
          builder.upload('tract-cells', 'float32', cells)
        )
      }
    };
  },
  // A curtain: the tract totals spread evenly over their cells on the left, the pycnophylactic
  // surface on the right, on one range.
  interaction: {
    kind: 'swipe',
    hint: 'swipe totals and surface',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `tract totals | after ${values.maximumIterations} iterations`
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    const range = [0, outputs.rangeMaximum] as const;
    return [
      {
        layer: 'raster',
        values: outputs.totals,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows,
        range,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'raster',
        values: outputs.surface,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows,
        range,
        smoothing: true,
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
