// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialScanStatistic (Kulldorff, Poisson model): scans circular windows of zones for excess
// births given estimated woman-years. Each county shows the best window centered on it; the ring
// and dots mark the most likely cluster. Hovering a county reads its window statistic and rate.

import {
  getGPUSpatialScanStatisticParameterValues,
  GPUSpatialScanStatistic,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import counties from '../../../data/illinois-births.json';
import {projectObservedPositions} from '../../../data/observed-positions';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './spatial-scan-statistic.parameters';

// This map displays observed likelihood ratios only. Use the minimum replicate budget.
const MAXIMUM_PERMUTATIONS = 1;
const MAXIMUM_CLUSTERS = 4;
const RATE_DIRECTIONS = ['both', 'high', 'low'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Illinois county birth rates, colored by the strongest scan window centered on each county.',
  createInputs: ({world}) => {
    const positions = projectObservedPositions(
      counties.map(([, longitude, latitude]) => [longitude, latitude]),
      world
    );
    const baseline = Float32Array.from(counties, county => county[4]);
    const birthCounts = Uint32Array.from(counties, county => county[3]);
    const overallRate =
      birthCounts.reduce((sum, births) => sum + births, 0) /
      baseline.reduce((sum, years) => sum + years, 0);
    return {
      zoneCount: counties.length,
      positions,
      baseline,
      birthCounts,
      overallRate,
      index: createPointIndex(positions),
      focus: {row: -1}
    };
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
        clusterIndices,
        clusterStatistics,
        positions: builder.upload('zone-positions', 'float32x2', inputs.positions)
      },
      update: current => {
        // A contrast of one preserves the published counts. Zero gives every county the statewide rate.
        cases.write(
          Uint32Array.from(inputs.birthCounts, (births, zone) => {
            const expected = inputs.overallRate * inputs.baseline[zone];
            return Math.max(0, Math.round(expected + current.contrast * (births - expected)));
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
  interaction: {
    kind: 'probe',
    hint: 'hover a county',
    rest: [0.5, 0.45],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const glyphs: RunGlyph[] = [];
      // The most likely cluster: its center's nearest zones, ringed to the farthest of them.
      const clusters = peek(outputs.clusterIndices);
      const clusterStatistics = peek(outputs.clusterStatistics);
      let clusterText = '';
      if (clusters && clusterStatistics) {
        const center = clusters[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.center];
        const zoneCount = clusters[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX.zoneCount];
        if (center < inputs.zoneCount && zoneCount > 0) {
          const members = inputs.index.within(
            inputs.positions[center * 2],
            inputs.positions[center * 2 + 1],
            Infinity
          );
          let radius = 0;
          for (const member of members.slice(0, zoneCount)) {
            radius = Math.max(
              radius,
              inputs.index.distanceTo(
                member,
                inputs.positions[center * 2],
                inputs.positions[center * 2 + 1]
              )
            );
            glyphs.push({
              shape: 'dot',
              at: [inputs.positions[member * 2], inputs.positions[member * 2 + 1]]
            });
          }
          glyphs.push({
            shape: 'ring',
            at: [inputs.positions[center * 2], inputs.positions[center * 2 + 1]],
            radius: radius + 0.015,
            dashed: true
          });
          clusterText = `cluster ${zoneCount} counties, LLR ${clusterStatistics[
            GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio
          ].toFixed(1)}`;
        }
      }
      if (row < 0) return {glyphs, readout: clusterText};
      const llr = peek(outputs.zoneStatistics)?.[row];
      const expected = inputs.overallRate * inputs.baseline[row];
      glyphs.push({
        shape: 'crosshair',
        at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
        size: 12
      });
      return {
        glyphs,
        readout: `FIPS ${counties[row][0]} · LLR ${llr === undefined ? '…' : llr.toFixed(1)} · ${(
          inputs.birthCounts[row] / expected
        ).toFixed(2)}x births · max ${values.maximumPopulationFraction * 100}% pop`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 9},
    {layer: 'dots', positions: outputs.positions, values: outputs.zoneStatistics, radius: 13}
  ]
});
