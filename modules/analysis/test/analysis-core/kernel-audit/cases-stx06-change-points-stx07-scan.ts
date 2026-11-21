// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUExpectationBasedScanParameterValues,
  GPUExpectationBasedSpaceTimeScan,
  GPUEmergingHotSpots,
  GPULocationChangePoints,
  GPUPettittTest,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_EXPECTATION_BASED_SCAN_MAXIMUM_NEIGHBORS,
  GPU_EXPECTATION_BASED_SCAN_MAXIMUM_WINDOW_LENGTH,
  GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH,
  GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT,
  GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT,
  GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH,
  GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT,
  GPUSpatialScanStatistic,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPULocationChangePointCost} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const SLICES = GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLICE_COUNT;
const PETTITT_SLICES = GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT;
const CHANGE_POINTS = GPU_LOCATION_CHANGE_POINTS_MAXIMUM_CHANGE_POINTS;

function buildLocationChangePoints(
  {graph, view, maximumRows, construct}: KernelAuditContext,
  cost: GPULocationChangePointCost,
  layout: 'cell-major' | 'period-major',
  sliceCount: number,
  extraCells = 0
): void {
  const cellCount = Math.floor(maximumRows(4) / sliceCount) + extraCells;
  graph.add(
    construct(GPULocationChangePoints, {
      values: view('float32', cellCount * sliceCount),
      layout,
      cellCount,
      sliceCount,
      cost,
      maximumChangePoints: CHANGE_POINTS,
      parameters: view('float32', GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH),
      output: {
        changePointCount: view('uint32', cellCount),
        changePoints: view('uint32', cellCount * CHANGE_POINTS),
        requiredChangePointCounts: view('uint32', cellCount),
        optimalCost: view('float32', cellCount),
        status: {overflow: view('uint32', 1)}
      }
    })
  );
}

function buildPettitt(
  {graph, view, maximumRows, construct}: KernelAuditContext,
  sliceCount: number,
  extraCells = 0
): void {
  const cellCount = Math.floor(maximumRows(4) / sliceCount) + extraCells;
  graph.add(
    construct(GPUPettittTest, {
      values: view('float32', cellCount * sliceCount),
      layout: 'period-major',
      cellCount,
      sliceCount,
      output: {
        statistic: view('uint32', cellCount),
        changeSlice: view('uint32', cellCount),
        pValues: view('float32', cellCount)
      }
    })
  );
}

function buildCorrectedHotSpots(
  {graph, view}: KernelAuditContext,
  correction: 'hamed-rao' | 'yue-wang'
): void {
  const sliceCount = GPU_EMERGING_HOT_SPOTS_MAXIMUM_CORRECTED_TREND_SLICE_COUNT;
  const width = 64;
  const cellCount = width * width;
  graph.add(
    new GPUEmergingHotSpots({
      values: view('float32', cellCount * sliceCount),
      gridWidth: width,
      gridHeight: width,
      sliceCount,
      trendAutocorrelationCorrection: correction,
      trendCorrectionMaximumLag: 3,
      trendCorrectionCriticalZ: 2.5758293035489,
      parameters: view('float32', GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH),
      output: {
        giZScores: view('float32', cellCount * sliceCount),
        trendZ: view('float32', cellCount),
        trendP: view('float32', cellCount),
        trendS: view('sint32', cellCount),
        category: view('uint32', cellCount),
        hotSliceCount: view('uint32', cellCount),
        coldSliceCount: view('uint32', cellCount)
      }
    })
  );
}

function buildExpectationBasedScan(
  {graph, view, maximumRows, construct}: KernelAuditContext,
  extraZones = 0
): void {
  const maximumNeighbors = GPU_EXPECTATION_BASED_SCAN_MAXIMUM_NEIGHBORS;
  const maximumWindowLength = GPU_EXPECTATION_BASED_SCAN_MAXIMUM_WINDOW_LENGTH;
  const zones = Math.floor(maximumRows(4) / maximumNeighbors) + extraZones;
  const timeBuckets = maximumWindowLength;
  const maximumReplicates = 99;
  getGPUExpectationBasedScanParameterValues({seed: 1, replicateCount: 1});
  graph.add(
    construct(GPUExpectationBasedSpaceTimeScan, {
      positions: view('float32x2', zones),
      cases: view('uint32', zones * timeBuckets),
      baseline: view('float32', zones * timeBuckets),
      timeBuckets,
      maximumNeighbors,
      maximumWindowLength,
      maximumReplicates,
      parameters: view('uint32', GPU_EXPECTATION_BASED_SCAN_PARAMETER_LENGTH),
      output: {
        clusterZones: view('uint32', maximumNeighbors),
        clusterStatistics: view('float32', GPU_EXPECTATION_BASED_SCAN_STATISTIC_WORDS),
        statistics: view('float32', maximumReplicates + 1),
        summary: view('uint32', GPU_EXPECTATION_BASED_SCAN_SUMMARY_LENGTH),
        zoneStatistics: view('float32', zones)
      }
    })
  );
}

function buildProspectiveScan({graph, view}: KernelAuditContext): void {
  const zones = 2048;
  const timeBuckets = 32;
  const maximumPermutations = 999;
  const maximumClusters = 8;
  graph.add(
    new GPUSpatialScanStatistic({
      positions: view('float32x2', zones),
      cases: view('uint32', zones * timeBuckets),
      baseline: view('float32', zones * timeBuckets),
      timeBuckets,
      maximumPermutations,
      maximumClusters,
      temporalMode: 'prospective',
      parameters: view('uint32', GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH),
      output: {
        clusterIndices: view('uint32', maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS),
        clusterStatistics: view(
          'float32',
          maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS
        ),
        statistics: view('float32', maximumPermutations + 1),
        summary: view('uint32', GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH)
      }
    })
  );
}

/** Cases of the per-location change points, Pettitt test, corrected trend (STX-06) and scans (STX-07). */
export const STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPULocationChangePoints',
    name: 'mean at the binding limit, cell-major, every output, maximum slices and change points',
    build: context => buildLocationChangePoints(context, 'mean', 'cell-major', SLICES)
  },
  {
    exportName: 'GPULocationChangePoints',
    name: 'slope period-major, binding-limited cube at the maximum slope slices',
    build: context =>
      buildLocationChangePoints(
        context,
        'slope',
        'period-major',
        GPU_LOCATION_CHANGE_POINTS_MAXIMUM_SLOPE_SLICE_COUNT
      )
  },
  {
    exportName: 'GPULocationChangePoints',
    name: 'mean-variance, binding-limited cube at maximum slices',
    build: context => buildLocationChangePoints(context, 'mean-variance', 'cell-major', SLICES)
  },
  {
    exportName: 'GPULocationChangePoints',
    name: 'variance, binding-limited cube at maximum slices',
    build: context => buildLocationChangePoints(context, 'variance', 'cell-major', SLICES)
  },
  {
    exportName: 'GPULocationChangePoints',
    name: 'poisson, binding-limited cube at maximum slices',
    build: context => buildLocationChangePoints(context, 'poisson', 'cell-major', SLICES)
  },
  {
    exportName: 'GPUPettittTest',
    name: 'period-major, binding-limited cube at maximum slices',
    build: context => buildPettitt(context, PETTITT_SLICES)
  },
  {
    exportName: 'GPUEmergingHotSpots',
    name: 'hamed-rao corrected trend at the corrected slice limit',
    build: context => buildCorrectedHotSpots(context, 'hamed-rao')
  },
  {
    exportName: 'GPUEmergingHotSpots',
    name: 'yue-wang corrected trend at the corrected slice limit',
    build: context => buildCorrectedHotSpots(context, 'yue-wang')
  },
  {
    exportName: 'GPUExpectationBasedSpaceTimeScan',
    name: 'binding-limited neighbor table, maximum neighbors and window',
    build: context => buildExpectationBasedScan(context)
  },
  {
    exportName: 'GPUSpatialScanStatistic',
    name: 'prospective temporal mode, 32 buckets',
    build: context => buildProspectiveScan(context)
  },
  {
    exportName: 'GPULocationChangePoints',
    name: 'one cell above the cube binding limit',
    build: context => buildLocationChangePoints(context, 'mean', 'cell-major', SLICES, 1)
  },
  {
    exportName: 'GPUPettittTest',
    name: 'one cell above the cube binding limit',
    build: context => buildPettitt(context, PETTITT_SLICES, 1)
  },
  {
    exportName: 'GPUExpectationBasedSpaceTimeScan',
    name: 'one zone above the count table limit',
    build: context => buildExpectationBasedScan(context, 1)
  }
];
