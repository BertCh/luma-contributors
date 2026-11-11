// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRasterResistanceConnectivityRecipe on a constructed resistance raster (a modeled scenario):
// a road with one crossing, a lake with no data and a rough background. The raster is the circuit
// (every cell with data is a node, every neighbouring pair an edge); unit current enters at the
// chosen habitat patch and leaves at the ground. The bright map is the current through each cell.

import {
  addRasterResistanceConnectivityRecipe,
  getRasterResistanceConnectivityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './raster-resistance-connectivity-recipe.parameters';

const ROWS = 32;
/** Habitat patches on the left edge: the five places the unit current can enter. */
const PATCH_ROWS = [0.12, 0.3, 0.5, 0.7, 0.88];
const PATCH_COLUMN = 0.07;
const GROUND = [0.93, 0.5] as const;
const ROAD_COLUMN = 0.56;
const CROSSING = [0.3, 0.42] as const;

function getBackground(u: number, v: number): number {
  const wave =
    Math.sin(u * 9.1 + v * 3.7) * Math.cos(v * 7.3 - u * 2.1) + Math.sin(u * 4.3 - v * 6.1);
  return 1.6 + 0.9 * wave;
}

export default defineOperationRun({
  parameters,
  buildCache: 10,
  caption:
    'Modeled scenario: a resistance raster with a road, one crossing and a lake of no-data cells; unit current from a habitat patch to the ground.',
  createInputs: ({world}) => {
    const columns = Math.round(ROWS * world.width);
    const grid = makeGPURasterMetadata({
      width: columns,
      height: ROWS,
      cellSize: 1,
      origin: [0, ROWS]
    });
    const roadHalfWidth = 1.1 / columns;
    const cellOf = (u: number, v: number) =>
      Math.min(ROWS - 1, Math.floor(v * ROWS)) * columns +
      Math.min(columns - 1, Math.floor(u * columns));
    const isBarrier = new Uint8Array(columns * ROWS);
    const base = new Float32Array(columns * ROWS);
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < columns; column++) {
        const u = (column + 0.5) / columns;
        const v = (row + 0.5) / ROWS;
        const cell = row * columns + column;
        base[cell] = Math.max(1, getBackground(u, v));
        const onRoad = Math.abs(u - (ROAD_COLUMN + 0.05 * (v - 0.5))) < roadHalfWidth;
        const inCrossing = v > CROSSING[0] && v < CROSSING[1];
        isBarrier[cell] = onRoad && !inCrossing ? 1 : 0;
        // The lake has no data: its cells leave the circuit.
        if (Math.hypot((u - 0.32) / 0.12, (v - 0.74) / 0.15 / 1.0) < 1) base[cell] = Number.NaN;
      }
    }
    // Rows are drawn top-down; the world y of a patch is 1 - v.
    const patches = PATCH_ROWS.map(v => cellOf(PATCH_COLUMN, v));
    const patchPositions = Float32Array.from(
      PATCH_ROWS.flatMap(v => [PATCH_COLUMN * world.width, 1 - v])
    );
    for (const patch of patches) base[patch] = 1;
    const sink = cellOf(GROUND[0], GROUND[1]);
    base[sink] = 1;
    return {
      columns,
      grid,
      world,
      base,
      isBarrier,
      patches,
      sink,
      patchPositions,
      patchIndex: createPointIndex(patchPositions),
      chosen: parameters[0].default as number
    };
  },
  build(builder, inputs, values) {
    const {grid, columns} = inputs;
    const cellCount = columns * ROWS;
    const resistance = builder.allocate('resistance', 'float32', cellCount);
    const parameterBuffer = builder.allocate('raster-resistance-parameters', 'float32', 8);
    const potentials = builder.allocate('potentials', 'float32', cellCount);
    const vertexCurrents = builder.allocate('vertex-currents', 'float32', cellCount);
    const diagnostics = builder.allocate('diagnostics', 'float32x4', 1);
    const solverStatus = builder.allocate('solver-status', 'uint32x4', 1);
    addRasterResistanceConnectivityRecipe(builder.graph, {
      grid,
      resistance: makeGPURasterBand(resistance.view, grid, {id: 'resistance'}),
      parameters: parameterBuffer.view,
      neighborhood: values.neighborhood === 0 ? 4 : 8,
      source: inputs.patches[values.source],
      sink: inputs.sink,
      maximumIterations: 1024,
      output: {
        potentials: potentials.view,
        vertexCurrents: vertexCurrents.view,
        diagnostics: diagnostics.view,
        solverStatus: solverStatus.view
      }
    });
    return {
      outputs: {resistance, vertexCurrents, diagnostics, solverStatus},
      update: current => {
        parameterBuffer.write(getRasterResistanceConnectivityParameterValues(grid));
        resistance.write(
          Float32Array.from(inputs.base, (value, cell) =>
            inputs.isBarrier[cell] ? current.barrierResistance : value
          )
        );
      }
    };
  },
  rebuildOn: ['source', 'neighborhood'],
  // The pointer chooses which habitat patch feeds the circuit. The source is a compile-time prop,
  // so the choice is applied once the pointer rests.
  interaction: {
    kind: 'place',
    hint: 'pick a source patch',
    rest: ({toUnit, inputs}) => {
      const index = parameters[0].default as number;
      return toUnit([inputs.patchPositions[index * 2], inputs.patchPositions[index * 2 + 1]]);
    },
    resolve(pointer, {inputs, outputs, values, peek, world}) {
      inputs.chosen = chooseSticky(inputs.patchIndex, pointer.x, pointer.y, inputs.chosen, 0.1);
      const patch = inputs.chosen;
      const at: [number, number] = [
        inputs.patchPositions[patch * 2],
        inputs.patchPositions[patch * 2 + 1]
      ];
      const diagnostics = peek(outputs.diagnostics);
      const status = peek(outputs.solverStatus);
      const isCurrent = patch === values.source;
      const readout =
        isCurrent && diagnostics && status
          ? status[0] === 1
            ? `effective resistance ${diagnostics[0].toPrecision(3)}`
            : 'not converged'
          : `source patch ${patch + 1}`;
      return {
        settle: {source: patch},
        glyphs: [
          {shape: 'ring', at, radius: 0.045},
          {shape: 'dot', at: [GROUND[0] * world.width, 1 - GROUND[1]]}
        ],
        readout
      };
    }
  },
  // Input: the resistance raster, dark where the current meets high resistance. Output: the
  // current through every cell, transparent where none flows.
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.resistance,
      columns: inputs.columns,
      rows: ROWS,
      ramp: 'grey',
      range: [1, 4]
    },
    {
      layer: 'raster',
      values: outputs.vertexCurrents,
      columns: inputs.columns,
      rows: ROWS,
      ramp: 'sequential',
      range: [0, 0.12],
      fade: true
    }
  ]
});
