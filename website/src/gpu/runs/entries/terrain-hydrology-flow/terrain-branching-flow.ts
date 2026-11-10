// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainBranchingFlow over the eight-neighbor fractions of GPUTerrainFlowFractions (MFD on the
// filled DEM): transport of one unit of water released at a cell, expected / minimum / maximum flow
// distance to an outlet, or the chance that a cell drains to the outlets on the upper half of the
// border.

import {
  GPUTerrainBranchingFlow,
  GPUTerrainFlowFractions
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './terrain-branching-flow.parameters';
import {
  addTerrainFlow,
  createTerrainInputs,
  getTerrainCell,
  getTerrainCellCenter,
  getTerrainContextLayer
} from './shared/terrain';

/** Mass released at the source: shares of it are what fans out. */
const RELEASED_MASS = 10;
const MODES = [
  'transport',
  'expected-distance',
  'minimum-distance',
  'maximum-distance',
  'fractional-catchment'
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Mass fans out and rejoins, or distance and outlet probability, through the fractions.',
  createInputs: context => {
    const terrain = createTerrainInputs(context);
    // Outlet probability: 1 on the upper half of the border, 0 elsewhere.
    const outletProbability = new Float32Array(terrain.cellCount);
    for (let cell = 0; cell < terrain.cellCount; cell++) {
      outletProbability[cell] = Math.floor(cell / terrain.columns) < terrain.rows / 2 ? 1 : 0;
    }
    // The default release cell: the highest ground in the western part of the map.
    let sourceCell = 0;
    for (let cell = 0; cell < terrain.cellCount; cell++) {
      const column = cell % terrain.columns;
      const row = Math.floor(cell / terrain.columns);
      if (
        column > terrain.columns * 0.15 &&
        column < terrain.columns * 0.45 &&
        row > terrain.rows * 0.2 &&
        row < terrain.rows * 0.8 &&
        terrain.elevation[cell] > terrain.elevation[sourceCell]
      ) {
        sourceCell = cell;
      }
    }
    return {
      ...terrain,
      world: context.world,
      outletProbability,
      defaultSourceCell: sourceCell,
      sourceCell
    };
  },
  build(builder, terrain, values) {
    const mode = MODES[Math.round(values.mode)];
    const flow = addTerrainFlow(builder, terrain);
    const fractions = builder.allocate('fractions', 'float32', terrain.cellCount * 8);
    const edgeLengths = builder.allocate('edge-lengths', 'float32', terrain.cellCount * 8);
    builder.graph.add(
      new GPUTerrainFlowFractions({
        grid: terrain.grid,
        elevation: flow.filledElevation.view,
        flowDirections: flow.flowDirections.view,
        parameters: flow.parameters.view,
        routing: 'mfd-freeman',
        output: {fractions: fractions.view, edgeLengths: edgeLengths.view}
      })
    );
    const source = builder.allocate('source', 'float32', terrain.cellCount);
    const result = builder.allocate('result', 'float32', terrain.cellCount);
    builder.graph.add(
      new GPUTerrainBranchingFlow({
        grid: terrain.grid,
        fractions: fractions.view,
        values: source.view,
        mode,
        edgeLengths: mode === 'transport' ? undefined : edgeLengths.view,
        output: {
          values: result.view,
          status: builder.allocate('status', 'uint32', terrain.cellCount).view,
          unresolvedCount: builder.allocate('unresolved-count', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {filledElevation: flow.filledElevation, result},
      update: current => {
        flow.writeSettings({flowExponent: current.flowExponent});
        if (mode === 'transport') {
          // One unit of water released at the source cell.
          const released = new Float32Array(terrain.cellCount);
          released[terrain.sourceCell] = RELEASED_MASS;
          source.write(released);
        } else {
          source.write(
            mode === 'fractional-catchment'
              ? terrain.outletProbability
              : new Float32Array(terrain.cellCount).fill(1)
          );
        }
      }
    };
  },
  rebuildOn: ['mode'],
  // The pointer releases the water: in transport mode the unit of mass starts in the cell under it
  // and fans out and rejoins as it follows the fractions downhill.
  interaction: {
    kind: 'place',
    hint: 'release water',
    rest: ({inputs, toUnit}) =>
      toUnit(getTerrainCellCenter(inputs, inputs.world, inputs.defaultSourceCell)),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const cell = getTerrainCell(inputs, world, pointer.x, pointer.y);
      const changed = inputs.sourceCell !== cell.index;
      inputs.sourceCell = cell.index;
      const at = getTerrainCellCenter(inputs, world, cell.index);
      const result = peek(outputs.result);
      let readout = `source [${cell.column}, ${cell.row}]`;
      if (result && values.mode === 0) {
        // Cells that carry at least 1% of the released mass: how far the plume has fanned out.
        let plume = 0;
        for (const value of result) if (value > RELEASED_MASS * 0.01) plume++;
        readout += ` · plume ${plume} cells`;
      } else if (result && values.mode !== 0) {
        readout = `${MODES[Math.round(values.mode)]} here ${result[cell.index].toFixed(2)}`;
      }
      return {
        rerun: changed && values.mode === 0,
        glyphs: [{shape: 'crosshair', at}],
        readout
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    getTerrainContextLayer(inputs, outputs.filledElevation),
    {
      layer: 'raster',
      values: outputs.result,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: values.mode === 0 ? [0, RELEASED_MASS * 0.2] : values.mode === 4 ? [0, 1] : 'auto',
      fade: values.mode === 0,
      alpha: 0.9,
      smoothing: true
    }
  ]
});
