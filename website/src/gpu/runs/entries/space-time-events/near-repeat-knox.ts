// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNearRepeatKnox on GOES lightning flashes over the Midwest. The table on the right is the operation's
// Knox ratio per (distance band, time band) cell, observed pairs over the mean under permuted times,
// on a log2 diverging ramp: rows are distance bands (nearest at the top), columns time bands
// (shortest at the left). Dashed boxes mark cells with pseudo p <= 0.05. Hovering a cell draws its
// flash pairs on the map: storms make flashes near in space also near in time.

import {
  getGPUSpaceTimeParameterValues,
  GPUNearRepeatKnox,
  GPU_NEAR_REPEAT_CELL_SUMMARY,
  GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {createDistanceBandWeights, uploadSpatialWeights, type CsrWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {createLightningFlashes, formatValue, shuffleTimes} from './_lib/events';
import {addKernelPass} from './_lib/kernel';
import {createCellTable, getCellBox, getTableCell, uploadCellTable} from './_lib/table';
import {parameters} from './near-repeat-knox.parameters';

const BAND_COUNT = 5;
const FLASH_COUNT = 900;
const PERMUTATIONS = 199;
const SIGNIFICANCE = 0.05;
/** Pairs drawn for the hovered cell; the readout counts all of them. */
const MAXIMUM_DRAWN_PAIRS = 160;

type Inputs = ReturnType<typeof createLightningFlashes> & {
  table: ReturnType<typeof createCellTable>;
};
type Outputs = {
  positions: RunBuffer;
  ratio: RunBuffer;
  summary: RunBuffer;
  tablePositions: RunBuffer;
  tableIndices: RunBuffer;
  tableOwners: RunBuffer;
  pairs: CsrWeights;
};

function getEdges(step: number): number[] {
  return Array.from({length: BAND_COUNT + 1}, (_, index) => index * step);
}

/** Band of `value` for edges `[0, step, ...]`: band 0 holds 0 and `(0, step]`; -1 past the last. */
function getBand(value: number, step: number): number {
  const band = Math.max(0, Math.ceil(value / step) - 1);
  return band < BAND_COUNT ? band : -1;
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024 (900 of 4,096 sampled flashes): Knox ratio per distance band (rows, near at top) and time band (columns, short at left); dashed cells p ≤ 0.05, uncorrected.',
  createInputs: ({world}) => ({
    ...createLightningFlashes([0, 0, world.width * 0.56, world.height], FLASH_COUNT),
    table: createCellTable(BAND_COUNT, BAND_COUNT, [
      world.width * 0.6,
      0.12,
      world.width * 0.96,
      0.88
    ])
  }),
  build(builder, events, values) {
    const distanceEdges = getEdges(values.distanceBand / events.kilometersPerUnit);
    const timeEdges = getEdges(values.timeBand);
    const pairs = createDistanceBandWeights(events.positions, distanceEdges[BAND_COUNT]);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const cellCount = BAND_COUNT * BAND_COUNT;
    const summary = builder.allocate(
      'summary',
      'float32',
      cellCount * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH
    );
    builder.graph.add(
      new GPUNearRepeatKnox({
        pairs: uploadSpatialWeights(builder, pairs),
        times: times.view,
        parameters: parameterBuffer.view,
        distanceEdges,
        timeEdges,
        maximumPermutations: PERMUTATIONS,
        output: {
          counts: builder.allocate('counts', 'uint32', (PERMUTATIONS + 1) * cellCount).view,
          summary: summary.view
        }
      })
    );
    // log2 of the Knox ratio per cell for the diverging ramp; NaN stays NaN (not drawn).
    const ratio = builder.allocate('ratio', 'float32', cellCount);
    addKernelPass(builder, {
      id: 'near-repeat-ratio',
      invocationCount: cellCount,
      bindings: [
        {name: 'summary', buffer: summary, access: 'read'},
        {name: 'ratio', buffer: ratio, access: 'read_write'}
      ],
      body: `let value = summary[summaryOffset + index * ${GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH}u + ${GPU_NEAR_REPEAT_CELL_SUMMARY.knoxRatio}u];
  ratio[ratioOffset + index] = log2(max(value, 0.01));`
    });
    const table = uploadCellTable(builder, events.table);
    return {
      outputs: {
        positions: builder.upload('positions', 'float32x2', events.positions),
        ratio,
        summary,
        pairs,
        tablePositions: table.positions,
        tableIndices: table.indices,
        tableOwners: table.owners
      },
      update: current => {
        times.write(shuffleTimes(events.hours, current.observedShare));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({seed: 5, permutationCount: PERMUTATIONS})
        );
      }
    };
  },
  rebuildOn: ['distanceBand', 'timeBand'],
  buildCache: 5,
  interaction: {
    kind: 'probe',
    hint: 'hover a table cell',
    rest: [0.66, 0.2],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {table} = inputs;
      const cell = getTableCell(table, pointer.x, pointer.y);
      const summary = peek(outputs.summary);
      const glyphs: RunGlyph[] = [];
      if (summary) {
        for (let index = 0; index < BAND_COUNT * BAND_COUNT; index++) {
          const pValue =
            summary[
              index * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH +
                GPU_NEAR_REPEAT_CELL_SUMMARY.pseudoPGreater
            ];
          if (pValue <= SIGNIFICANCE) {
            const box = getCellBox(table, Math.floor(index / BAND_COUNT), index % BAND_COUNT);
            glyphs.push({shape: 'box', ...box, dashed: true});
          }
        }
      }
      glyphs.push({shape: 'box', ...getCellBox(table, cell.row, cell.column)});
      // The pairs of the hovered cell, from the same pair list and band rule as the operation.
      const times = shuffleTimes(inputs.hours, values.observedShare);
      const distanceStep = values.distanceBand / inputs.kilometersPerUnit;
      const {pairs} = outputs;
      let pairCount = 0;
      for (let row = 0; row < pairs.rowCount; row++) {
        for (let slot = pairs.offsets[row]; slot < pairs.offsets[row + 1]; slot++) {
          const other = pairs.neighbors[slot];
          if (other <= row) continue;
          if (getBand(pairs.distances[slot], distanceStep) !== cell.row) continue;
          if (getBand(Math.abs(times[row] - times[other]), values.timeBand) !== cell.column)
            continue;
          pairCount++;
          if (pairCount > MAXIMUM_DRAWN_PAIRS) continue;
          glyphs.push({
            shape: 'line',
            from: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            to: [inputs.positions[other * 2], inputs.positions[other * 2 + 1]]
          });
        }
      }
      const base = cell.index * GPU_NEAR_REPEAT_CELL_SUMMARY_LENGTH;
      const distance = `${cell.row * values.distanceBand}–${(cell.row + 1) * values.distanceBand} km`;
      const lag = `${cell.column * values.timeBand}–${(cell.column + 1) * values.timeBand} h`;
      return {
        focus: cell.index,
        glyphs,
        readout: `${formatValue(
          summary?.[base + GPU_NEAR_REPEAT_CELL_SUMMARY.knoxRatio],
          1
        )}x expected · ${distance}, ${lag} · ${pairCount} pairs · p ${formatValue(summary?.[base + GPU_NEAR_REPEAT_CELL_SUMMARY.pseudoPGreater], 3)}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {layer: 'dots', positions: outputs.positions, radius: 2.2, color: [...theme.muted, 0.9]},
    {
      layer: 'polygons',
      positions: outputs.tablePositions,
      indices: outputs.tableIndices,
      owners: outputs.tableOwners,
      values: outputs.ratio,
      triangleCount: inputs.table.triangleCount,
      ramp: 'diverging',
      range: [-1.5, 1.5]
    }
  ]
});
