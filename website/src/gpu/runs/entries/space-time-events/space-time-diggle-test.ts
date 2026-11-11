// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpaceTimeDiggleTest on the Kīlauea summit earthquakes. The table on the right is the
// observed D(u, v) of each class standardized against the permutations of the event times,
// (D - permutationMean) / permutationStandardDeviation: rows are distance classes (near at the
// top), columns time-lag classes (short at the left); red is more space-time clustering than the
// shuffled times give. The test statistic U in the chart is the plain sum of D over the classes
// (as splancs stmctest), not the sum of the standardized cells. Hovering a cell draws the event
// pairs of that class on the map.

import {
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues,
  GPUSpaceTimeDiggleTest,
  GPU_SPACE_TIME_DIGGLE_SUMMARY,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createKilaueaEvents, formatValue, shuffleTimes} from './_lib/events';
import {addKernelPass} from './_lib/kernel';
import {drawReferenceHistogram} from './_lib/permutation-chart';
import {createCellTable, getCellBox, getTableCell, uploadCellTable} from './_lib/table';
import {drawPanel, type Panel, uploadPanel} from './_lib/panel';
import {parameters} from './space-time-diggle-test.parameters';

const CLASS_COUNT = 6;
const PERMUTATIONS = 99;
const WINDOW_DAYS = 121;
const MAXIMUM_DRAWN_PAIRS = 160;
const HISTOGRAM_BINS = 24;
const CHART = [0.08, 0.04, 0.92, 0.22] as const;

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createKilaueaEvents([0, 0.26, world.width * 0.6, world.height]);
  let bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (let row = 0; row < events.count; row++) {
    const x = events.positions[row * 2];
    const y = events.positions[row * 2 + 1];
    bounds = [
      Math.min(bounds[0], x),
      Math.min(bounds[1], y),
      Math.max(bounds[2], x),
      Math.max(bounds[3], y)
    ];
  }
  return {
    ...events,
    worldWidth: world.width,
    bounds: bounds as [number, number, number, number],
    table: createCellTable(CLASS_COUNT, CLASS_COUNT, [
      world.width * 0.6,
      0.3,
      world.width * 0.96,
      0.94
    ])
  };
}

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  panel: Panel;
  positions: RunBuffer;
  residual: RunBuffer;
  d: RunBuffer;
  statistics: RunBuffer;
  bins: RunBuffer<'float32'>;
  summary: RunBuffer;
  tablePositions: RunBuffer;
  tableIndices: RunBuffer;
  tableOwners: RunBuffer;
};

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Kīlauea summit earthquakes, 2018: standardized D(u, v) against 99 time permutations; rows distance (near at top), columns time lag; bars are U, the sum of the unstandardized D over all classes, under the permutations, the solid tick the observed U.',
  createInputs,
  build(builder, events) {
    const classCount = CLASS_COUNT * CLASS_COUNT;
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH
    );
    const d = builder.allocate('d', 'float32', classCount);
    const permutationMean = builder.allocate('permutation-mean', 'float32', classCount);
    const permutationDeviation = builder.allocate('permutation-deviation', 'float32', classCount);
    const statistics = builder.allocate('statistics', 'float32', PERMUTATIONS + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUSpaceTimeDiggleTest({
        positions: builder.upload('positions-input', 'float32x2', events.positions).view,
        times: times.view,
        parameters: parameterBuffer.view,
        randomParameters: builder.upload(
          'random-parameters',
          'uint32',
          getGPUSpaceTimeDiggleRandomParameterValues(11)
        ).view,
        gridSize: [16, 16],
        distanceClassCount: CLASS_COUNT,
        timeClassCount: CLASS_COUNT,
        permutationCount: PERMUTATIONS,
        output: {
          d: d.view,
          statistics: statistics.view,
          summary: summary.view,
          permutationMean: permutationMean.view,
          permutationStandardDeviation: permutationDeviation.view
        }
      })
    );
    const residual = builder.allocate('residual', 'float32', classCount);
    addKernelPass(builder, {
      id: 'diggle-residual',
      invocationCount: classCount,
      bindings: [
        {name: 'd', buffer: d, access: 'read'},
        {name: 'mean', buffer: permutationMean, access: 'read'},
        {name: 'deviation', buffer: permutationDeviation, access: 'read'},
        {name: 'residual', buffer: residual, access: 'read_write'}
      ],
      body: `let spread = deviation[deviationOffset + index];
  var notANumberBits = 0x7fc00000u;
  let value = (d[dOffset + index] - mean[meanOffset + index]) / spread;
  residual[residualOffset + index] = select(bitcast<f32>(notANumberBits), value, spread > 0.0);`
    });
    const table = uploadCellTable(builder, events.table);
    return {
      outputs: {
        positions: builder.upload('positions', 'float32x2', events.positions),
        residual,
        d,
        statistics,
        panel: uploadPanel(builder, CHART, {width: events.worldWidth, height: 1}),
        bins: builder.allocate('bins', 'float32', HISTOGRAM_BINS),
        summary,
        tablePositions: table.positions,
        tableIndices: table.indices,
        tableOwners: table.owners
      },
      update: current => {
        times.write(shuffleTimes(events.days, current.observedShare));
        parameterBuffer.write(
          getGPUSpaceTimeKFunctionParameterValues({
            bounds: events.bounds,
            maximumDistance: current.maximumDistance / events.kilometersPerUnit,
            maximumTimeLag: current.maximumTimeLag,
            timeWindow: [0, WINDOW_DAYS]
          })
        );
      }
    };
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a table cell',
    rest: [0.66, 0.2],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const {table} = inputs;
      const cell = getTableCell(table, pointer.x, pointer.y);
      const glyphs: RunGlyph[] = [{shape: 'box', ...getCellBox(table, cell.row, cell.column)}];
      // Pairs of this class: distance in ((a) u / U, (a + 1) u / U], lag likewise.
      const times = shuffleTimes(inputs.days, values.observedShare);
      const distanceStep = values.maximumDistance / CLASS_COUNT / inputs.kilometersPerUnit;
      const lagStep = values.maximumTimeLag / CLASS_COUNT;
      const getClass = (value: number, step: number) => Math.max(0, Math.ceil(value / step) - 1);
      let pairCount = 0;
      for (let row = 0; row < inputs.count; row++) {
        for (let other = row + 1; other < inputs.count; other++) {
          const distance = Math.hypot(
            inputs.positions[row * 2] - inputs.positions[other * 2],
            inputs.positions[row * 2 + 1] - inputs.positions[other * 2 + 1]
          );
          if (getClass(distance, distanceStep) !== cell.row) continue;
          if (getClass(Math.abs(times[row] - times[other]), lagStep) !== cell.column) continue;
          pairCount++;
          if (pairCount > MAXIMUM_DRAWN_PAIRS) continue;
          glyphs.push({
            shape: 'line',
            from: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            to: [inputs.positions[other * 2], inputs.positions[other * 2 + 1]]
          });
        }
      }
      const summary = peek(outputs.summary);
      const chart = drawReferenceHistogram({
        histogram: outputs.bins,
        samples: peek(outputs.statistics),
        start: 1,
        sampleCount: PERMUTATIONS,
        observed: summary?.[GPU_SPACE_TIME_DIGGLE_SUMMARY.observed] ?? Number.NaN,
        rectangle: CHART,
        world
      });
      glyphs.push(...chart.glyphs);
      const residual = peek(outputs.residual)?.[cell.index];
      const distance = ((cell.row + 1) * values.maximumDistance) / CLASS_COUNT;
      const lag = ((cell.column + 1) * values.maximumTimeLag) / CLASS_COUNT;
      return {
        focus: cell.index,
        glyphs,
        readout: `z ${formatValue(residual, 1)} at ${distance.toFixed(1)} km, ${lag.toFixed(1)} d · ${pairCount} pairs · U (all classes) p ${formatValue(
          summary?.[GPU_SPACE_TIME_DIGGLE_SUMMARY.pseudoPGreater],
          2
        )}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    drawPanel(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bins,
      range: [0, 1],
      rectangle: CHART,
      color: [...theme.categories[0], 0.9]
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.5},
    {
      layer: 'polygons',
      positions: outputs.tablePositions,
      indices: outputs.tableIndices,
      owners: outputs.tableOwners,
      values: outputs.residual,
      triangleCount: inputs.table.triangleCount,
      ramp: 'diverging',
      range: [-4, 4]
    }
  ]
});
