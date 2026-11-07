// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialScanStatistic (Kulldorff, Poisson model): scans circular windows of zones for excess
// cases. Each zone is colored by the best log-likelihood ratio of any window centered on it, so the
// hot spot we planted (top left) glows; the dim zone in the lower right is a weaker excess.

import {
  getGPUSpatialScanStatisticParameterValues,
  GPUSpatialScanStatistic,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-scan-statistic.parameters';

const COLUMNS = 14;
const ROWS = 8;
const MAXIMUM_PERMUTATIONS = 99;
const MAXIMUM_CLUSTERS = 4;
const RATE_DIRECTIONS = ['high', 'low', 'both'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Zones colored by the strongest scan window centered on them.',
  createInputs: ({random, world}) => {
    const zoneCount = COLUMNS * ROWS;
    const positions = new Float32Array(zoneCount * 2);
    const baseline = new Float32Array(zoneCount);
    const uniform = new Float32Array(zoneCount);
    for (let zone = 0; zone < zoneCount; zone++) {
      positions[zone * 2] = ((zone % COLUMNS) + 0.5) * (world.width / COLUMNS);
      positions[zone * 2 + 1] = (Math.floor(zone / COLUMNS) + 0.5) * (world.height / ROWS);
      baseline[zone] = 100 + Math.floor(random() * 200);
      uniform[zone] = random();
    }
    return {zoneCount, positions, baseline, uniform};
  },
  build(builder, inputs, values) {
    const cases = builder.allocate('cases', 'uint32', inputs.zoneCount);
    const baseline = builder.upload('baseline', 'float32', inputs.baseline);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH
    );
    const clusterIndices = builder.allocate(
      'cluster-indices',
      'uint32',
      MAXIMUM_CLUSTERS * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS
    );
    const clusterStatistics = builder.allocate(
      'cluster-statistics',
      'float32',
      MAXIMUM_CLUSTERS * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS
    );
    const statistics = builder.allocate('statistics', 'float32', MAXIMUM_PERMUTATIONS + 1);
    const summary = builder.allocate(
      'summary',
      'uint32',
      GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
    );
    const zoneStatistics = builder.allocate('zone-statistics', 'float32', inputs.zoneCount);
    builder.graph.add(
      new GPUSpatialScanStatistic({
        positions: builder.upload('positions', 'float32x2', inputs.positions).view,
        model: 'poisson',
        cases: cases.view,
        baseline: baseline.view,
        maximumWindowZones: values.maximumWindowZones,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumClusters: MAXIMUM_CLUSTERS,
        parameters: parameterBuffer.view,
        output: {
          clusterIndices: clusterIndices.view,
          clusterStatistics: clusterStatistics.view,
          statistics: statistics.view,
          summary: summary.view,
          zoneStatistics: zoneStatistics.view
        }
      })
    );
    return {
      outputs: {
        zoneStatistics,
        positions: builder.upload('zone-positions', 'float32x2', inputs.positions)
      },
      update: current => {
        // Case counts: a planted excess (top left, rows 1-3 and columns 2-4) and a weaker one.
        cases.write(
          Uint32Array.from(inputs.baseline, (population, zone) => {
            const column = zone % COLUMNS;
            const row = Math.floor(zone / COLUMNS);
            let rate = 0.05;
            if (column >= 2 && column <= 4 && row >= 1 && row <= 3) rate = current.excessRate;
            if (column >= 10 && column <= 11 && row >= 5 && row <= 6) rate = 0.1;
            return Math.round(population * rate * (0.85 + 0.3 * inputs.uniform[zone]));
          })
        );
        parameterBuffer.write(
          getGPUSpatialScanStatisticParameterValues({
            seed: 7,
            permutationCount: MAXIMUM_PERMUTATIONS,
            maximumPopulationFraction: current.maximumPopulationFraction,
            maximumWindowZones: values.maximumWindowZones,
            rateDirection: RATE_DIRECTIONS[current.rateDirection]
          })
        );
      }
    };
  },
  rebuildOn: ['maximumWindowZones'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 9},
    {layer: 'dots', positions: outputs.positions, values: outputs.zoneStatistics, radius: 13}
  ]
});
