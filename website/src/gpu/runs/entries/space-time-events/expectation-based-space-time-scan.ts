// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUExpectationBasedSpaceTimeScan as prospective surveillance of GOES lightning: 40 zones (an 8 x 5
// grid) by 16 half-hour buckets that end at "now". The baseline of every zone and bucket is the
// zone's mean count over the four hours before that bucket, so storms that move into quiet zones
// score high. The raster is the operation's zoneStatistics (best score centered at each zone); boxes
// are the most likely cluster; flashes in its window are bright. The pointer moves "now".

import {
  getGPUExpectationBasedScanParameterValues,
  GPUExpectationBasedSpaceTimeScan,
  GPU_EXPECTATION_BASED_SCAN_CLUSTER,
  GPU_EXPECTATION_BASED_SCAN_NO_ZONE,
  GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH,
  GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes, type RunGlyph} from '../../interaction';
import {createLightningFlashes, formatValue} from './_lib/events';
import {formatClock, OBSERVATION_HOURS} from './_lib/time-window';
import {drawPanel, type Panel, uploadPanel} from './_lib/panel';
import {parameters} from './expectation-based-space-time-scan.parameters';

const COLUMNS = 8;
const ROWS = 5;
const ZONES = COLUMNS * ROWS;
const BUCKET_HOURS = 0.5;
const TIME_BUCKETS = 16;
const HISTORY_BUCKETS = 8;
const MAXIMUM_NEIGHBORS = 9;
const MAXIMUM_WINDOW = 8;
const MAXIMUM_REPLICATES = 199;
const REPLICATES = 99;
/** Smallest baseline, so a zone without history still has an expectation. */
const BASELINE_FLOOR = 0.25;
const STRIP = [0.06, 0.04, 0.94, 0.16] as const;

function createInputs({world}: {world: {width: number; height: number}}) {
  const flashes = createLightningFlashes([0, 0, world.width, world.height], 4096, 0);
  const binCount = Math.ceil(OBSERVATION_HOURS / BUCKET_HOURS);
  const zoneOf = new Uint32Array(flashes.count);
  const counts = new Float32Array(ZONES * binCount);
  for (let row = 0; row < flashes.count; row++) {
    const column = Math.min(
      COLUMNS - 1,
      Math.floor((flashes.positions[row * 2] / world.width) * COLUMNS)
    );
    const zoneRow = Math.min(ROWS - 1, Math.floor((1 - flashes.positions[row * 2 + 1]) * ROWS));
    zoneOf[row] = zoneRow * COLUMNS + column;
    const bin = Math.min(binCount - 1, Math.floor(flashes.hours[row] / BUCKET_HOURS));
    counts[zoneOf[row] * binCount + bin]++;
  }
  // Zone centers, row 0 at the top as the raster layer draws it.
  const centers = Float32Array.from({length: ZONES * 2}, (_, index) => {
    const zone = index >> 1;
    return index % 2 === 0
      ? (((zone % COLUMNS) + 0.5) / COLUMNS) * world.width
      : 1 - (Math.floor(zone / COLUMNS) + 0.5) / ROWS;
  });
  const binTotals = new Float32Array(binCount);
  for (const hour of flashes.hours)
    binTotals[Math.min(binCount - 1, Math.floor(hour / BUCKET_HOURS))]++;
  return {...flashes, world, binCount, zoneOf, counts, centers, binTotals};
}

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  panel: Panel;
  positions: RunBuffer;
  windowFlashes: RunBuffer;
  zoneStatistics: RunBuffer;
  clusterZones: RunBuffer;
  clusterStatistics: RunBuffer;
  summary: RunBuffer;
  bins: RunBuffer;
};

/** First observation bin of the 16 analyzed buckets. */
const getFirstBin = (windowEnd: number) => Math.round(windowEnd / BUCKET_HOURS) - TIME_BUCKETS;

function getCasesAndBaseline(inputs: Inputs, windowEnd: number) {
  const first = getFirstBin(windowEnd);
  const cases = new Uint32Array(ZONES * TIME_BUCKETS);
  const baseline = new Float32Array(ZONES * TIME_BUCKETS);
  for (let zone = 0; zone < ZONES; zone++) {
    for (let bucket = 0; bucket < TIME_BUCKETS; bucket++) {
      const bin = first + bucket;
      cases[zone * TIME_BUCKETS + bucket] = inputs.counts[zone * inputs.binCount + bin];
      let history = 0;
      for (let past = bin - HISTORY_BUCKETS; past < bin; past++) {
        if (past >= 0) history += inputs.counts[zone * inputs.binCount + past];
      }
      baseline[zone * TIME_BUCKETS + bucket] = Math.max(BASELINE_FLOOR, history / HISTORY_BUCKETS);
    }
  }
  return {cases, baseline};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'GOES lightning, May 21–22 2024: expectation-based scan of the latest half hours against each zone’s previous 4 h; boxes and bright flashes are the most likely cluster. 4,096 sampled GLM flashes; p is not adjusted for repeated looks.',
  createInputs,
  build(builder, inputs) {
    const cases = builder.allocate('cases', 'uint32', ZONES * TIME_BUCKETS);
    const baseline = builder.allocate('baseline', 'float32', ZONES * TIME_BUCKETS);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH
    );
    const zoneStatistics = builder.allocate('zone-statistics', 'float32', ZONES);
    const clusterZones = builder.allocate('cluster-zones', 'uint32', MAXIMUM_NEIGHBORS);
    const clusterStatistics = builder.allocate(
      'cluster-statistics',
      'float32',
      GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS
    );
    const summary = builder.allocate(
      'summary',
      'uint32',
      GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH
    );
    builder.graph.add(
      new GPUExpectationBasedSpaceTimeScan({
        positions: builder.upload('centers', 'float32x2', inputs.centers).view,
        cases: cases.view,
        baseline: baseline.view,
        timeBuckets: TIME_BUCKETS,
        maximumNeighbors: MAXIMUM_NEIGHBORS,
        maximumWindowLength: MAXIMUM_WINDOW,
        maximumReplicates: MAXIMUM_REPLICATES,
        parameters: parameterBuffer.view,
        output: {
          clusterZones: clusterZones.view,
          clusterStatistics: clusterStatistics.view,
          statistics: builder.allocate('statistics', 'float32', MAXIMUM_REPLICATES + 1).view,
          summary: summary.view,
          zoneStatistics: zoneStatistics.view
        }
      })
    );
    const windowFlashes = builder.allocate('window-flashes', 'float32', inputs.count);
    return {
      outputs: {
        positions: builder.upload('positions', 'float32x2', inputs.positions),
        windowFlashes,
        zoneStatistics,
        clusterZones,
        clusterStatistics,
        summary,
        panel: uploadPanel(builder, STRIP, {width: inputs.world.width, height: 1}),
        bins: builder.upload('bins', 'float32', inputs.binTotals)
      },
      update: current => {
        const data = getCasesAndBaseline(inputs, current.windowEnd);
        cases.write(data.cases);
        baseline.write(data.baseline);
        parameterBuffer.write(
          getGPUExpectationBasedScanParameterValues({
            seed: 2,
            replicateCount: REPLICATES,
            maximumWindowLength: current.maximumWindowLength
          })
        );
      }
    };
  },
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'time',
    hint: 'move now across',
    x: 'windowEnd',
    y: 'maximumWindowLength',
    decorate: (values, _pointer, {inputs, outputs, peek}) => {
      const glyphs: RunGlyph[] = [];
      const zones = peek(outputs.clusterZones);
      const statistics = peek(outputs.clusterStatistics);
      const summary = peek(outputs.summary);
      const cellWidth = inputs.world.width / COLUMNS;
      for (const zone of zones ?? []) {
        if (zone === GPU_EXPECTATION_BASED_SCAN_NO_ZONE || zone >= ZONES) continue;
        const column = zone % COLUMNS;
        const row = Math.floor(zone / COLUMNS);
        glyphs.push({
          shape: 'box',
          minimum: [column * cellWidth, 1 - (row + 1) / ROWS],
          maximum: [(column + 1) * cellWidth, 1 - row / ROWS]
        });
      }
      const windowLength = summary?.[GPU_EXPECTATION_BASED_SCAN_SUMMARY.windowLength];
      // Brighten only the flashes of the detected cluster: its zones and its window.
      const clusterZones = new Set(zones ?? []);
      const start = values.windowEnd - (windowLength ?? 0) * BUCKET_HOURS;
      outputs.windowFlashes.write(
        Float32Array.from(inputs.hours, (hour, row) =>
          clusterZones.has(inputs.zoneOf[row]) && hour >= start && hour < values.windowEnd ? 1 : 0
        )
      );
      // Timeline strip: the detected window ends at now.
      const toX = (hour: number) =>
        (STRIP[0] + (hour / OBSERVATION_HOURS) * (STRIP[2] - STRIP[0])) * inputs.world.width;
      glyphs.push({
        shape: 'line',
        from: [toX(values.windowEnd), STRIP[1]],
        to: [toX(values.windowEnd), STRIP[3] + 0.02]
      });
      if (windowLength) {
        glyphs.push({
          shape: 'box',
          minimum: [toX(values.windowEnd - windowLength * BUCKET_HOURS), STRIP[1]],
          maximum: [toX(values.windowEnd), STRIP[3]]
        });
      }
      return {
        glyphs,
        readout: `now ${formatClock(values.windowEnd)} · ${
          windowLength === undefined ? '…' : windowLength * BUCKET_HOURS
        } h window · ${formatValue(
          statistics?.[GPU_EXPECTATION_BASED_SCAN_CLUSTER.observedCases],
          0
        )} vs ${formatValue(statistics?.[GPU_EXPECTATION_BASED_SCAN_CLUSTER.expectedCases], 0)} · p ${formatValue(
          statistics?.[GPU_EXPECTATION_BASED_SCAN_CLUSTER.pValue],
          3
        )}`
      };
    }
  }),
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'raster',
      values: outputs.zoneStatistics,
      columns: COLUMNS,
      rows: ROWS,
      range: [0, 9],
      logarithmic: true,
      alpha: 0.8
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.windowFlashes,
      range: [0, 1],
      ramp: 'grey',
      fade: true,
      radius: 2.5
    },
    drawPanel(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bins,
      range: [0, Math.max(...inputs.binTotals)],
      rectangle: STRIP,
      color: [...theme.categories[0], 0.9]
    }
  ]
});
