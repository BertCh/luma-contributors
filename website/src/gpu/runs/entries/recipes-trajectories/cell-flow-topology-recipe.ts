// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addCellFlowTopologyRecipe on a patch of H3 cells: origin-destination flows are totaled per cell.
// Fill is the outgoing total, rings are the incoming total (bigger and brighter = more arrivals).

import {
  addCellFlowTopologyRecipe,
  GPUPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {COMMUTE_FLOWS, getCommutePositions} from '../../../data/commute-flows';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createPointIndex} from '../../pick';
import {addCellFill} from './lib/cells';
import {parameters} from './cell-flow-topology-recipe.parameters';

const COLUMNS = 13;
const ROWS = 8;
const RESOLUTION = 5;

export default defineOperationRun({
  parameters,
  caption: 'Chicago home-to-work flows aggregated to a local display lattice.',
  createInputs: ({world}) => {
    const lattice = Float32Array.from({length: COLUMNS * ROWS * 2}, (_, index) => {
      const node = Math.floor(index / 2);
      const column = node % COLUMNS;
      const row = Math.floor(node / COLUMNS);
      return index % 2 === 0 ? 0.08 + column * 0.135 + (row % 2) * 0.068 : 0.1 + row * 0.115;
    });
    return {tractPositions: getCommutePositions(world), lattice, index: createPointIndex(lattice)};
  },
  build(builder, {lattice, tractPositions}, values) {
    const cellCount = COLUMNS * ROWS;
    const pairCount = values.pairCount;
    const cells = builder.allocate('cells', 'uint32x2', cellCount);
    builder.graph.add(
      new GPUPointToCell({
        family: 'h3',
        resolution: RESOLUTION,
        positions: builder.upload('lattice', 'float32x2', lattice).view,
        output: {cells: cells.view}
      })
    );
    const origins = new Uint32Array(pairCount);
    const destinations = new Uint32Array(pairCount);
    // Place each tract in the nearest display-lattice cell. The topology recipe works on
    // cell indices, while GPUPointToCell supplies the cell keys drawn by GPUCellGeometry.
    const tractCells = Uint32Array.from({length: tractPositions.length / 2}, (_, tract) => {
      let nearest = 0;
      let distance = Infinity;
      for (let cell = 0; cell < cellCount; cell++) {
        const candidate = Math.hypot(
          tractPositions[tract * 2] - lattice[cell * 2],
          tractPositions[tract * 2 + 1] - lattice[cell * 2 + 1]
        );
        if (candidate < distance) {
          nearest = cell;
          distance = candidate;
        }
      }
      return nearest;
    });
    for (let pair = 0; pair < pairCount; pair++) {
      const [home, work] = COMMUTE_FLOWS[pair];
      origins[pair] = tractCells[home];
      destinations[pair] = tractCells[work];
    }
    const flowValues = builder.allocate('flow-values', 'float32', pairCount);
    const outgoingTotals = builder.allocate('outgoing', 'float32', cellCount);
    const incomingTotals = builder.allocate('incoming', 'float32', cellCount);
    const invalid = builder.allocate('invalid', 'uint32', 1);
    addCellFlowTopologyRecipe(builder.graph, {
      cells: cells.view,
      origins: builder.upload('origins', 'uint32', origins).view,
      destinations: builder.upload('destinations', 'uint32', destinations).view,
      values: flowValues.view,
      output: {
        outgoingTotals: outgoingTotals.view,
        incomingTotals: incomingTotals.view,
        status: {invalidCount: invalid.view}
      }
    });
    return {
      outputs: {
        outgoingTotals,
        incomingTotals,
        fill: addCellFill(builder, 'h3', cells),
        origins,
        destinations,
        flowValues
      },
      update: current =>
        flowValues.write(
          Float32Array.from({length: pairCount}, (_, pair) => {
            const eastward = (destinations[pair] % COLUMNS) - (origins[pair] % COLUMNS);
            return COMMUTE_FLOWS[pair][2] * Math.max(0.1, 1 + (current.pull * eastward) / COLUMNS);
          })
        )
    };
  },
  rebuildOn: ['pairCount'],
  // The pointer picks a cell: its outgoing flows (solid) and incoming flows (dashed).
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: ({toUnit, inputs}) => toUnit([inputs.lattice[2 * 45], inputs.lattice[2 * 45 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const cell = inputs.index.nearest(pointer.x, pointer.y);
      const from: [number, number] = [inputs.lattice[cell * 2], inputs.lattice[cell * 2 + 1]];
      const outgoing = new Map<number, number>();
      const incoming = new Map<number, number>();
      for (let pair = 0; pair < outputs.origins.length; pair++) {
        const origin = outputs.origins[pair];
        const destination = outputs.destinations[pair];
        if (origin === destination) continue;
        if (origin === cell) outgoing.set(destination, (outgoing.get(destination) ?? 0) + 1);
        if (destination === cell) incoming.set(origin, (incoming.get(origin) ?? 0) + 1);
      }
      const glyphs: RunGlyph[] = [{shape: 'dot', at: from}];
      for (const [partners, dashed] of [
        [outgoing, false],
        [incoming, true]
      ] as const) {
        for (const partner of [...partners.keys()].slice(0, 10)) {
          glyphs.push({
            shape: 'line',
            from,
            to: [inputs.lattice[partner * 2], inputs.lattice[partner * 2 + 1]],
            dashed
          });
        }
      }
      const totalOut = peek(outputs.outgoingTotals);
      const totalIn = peek(outputs.incomingTotals);
      return {
        glyphs,
        readout: `cell ${cell} · out ${totalOut ? Math.round(totalOut[cell]) : '...'} · in ${totalIn ? Math.round(totalIn[cell]) : '...'}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.outgoingTotals,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential'
    },
    {
      layer: 'dots',
      positions: outputs.fill.centers,
      values: outputs.incomingTotals,
      ramp: 'sequential',
      radius: 6
    }
  ]
});
