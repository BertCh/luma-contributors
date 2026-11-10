// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH,
  GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH,
  GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH,
  GPUSpaceTimeDiggleTest,
  GPUSpaceTimeKFunction
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH,
  GPUSpaceTimeKernelDensity,
  GPUSpaceTimeRelativeRisk
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Rows the 16-byte row tables of STX-04 and STX-05 hold in one 128 MiB binding. */
const MAXIMUM_ROWS_16 = 8388608;
/** Rows whose 64 local classes (4 bytes each) fill one 128 MiB binding: `2^25 / 64`. */
const MAXIMUM_LOCAL_ROWS = 524288;

function addDensity(
  {graph, view}: KernelAuditContext,
  options: {
    rows: number;
    width: number;
    height: number;
    sliceCount: number;
    indexGridSize: readonly [number, number];
    spatialKernel?: 'gaussian' | 'quartic' | 'epanechnikov' | 'bisquare';
    temporalKernel?: 'gaussian' | 'quartic' | 'epanechnikov';
    edge?: 'none' | 'uniform';
    normalization?: 'intensity' | 'probability';
  }
): void {
  const {rows, width, height, sliceCount} = options;
  graph.add(
    new GPUSpaceTimeKernelDensity({
      positions: view('float32x2', rows),
      times: view('float32', rows),
      weights: view('float32', rows),
      parameters: view('float32', GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH),
      grid: makeGPURasterMetadata({width, height}),
      sliceCount,
      indexGridSize: options.indexGridSize,
      indexBounds: [0, 0, 100, 100],
      spatialKernel: options.spatialKernel,
      temporalKernel: options.temporalKernel,
      spatialEdgeCorrection: options.edge,
      temporalEdgeCorrection: options.edge,
      normalization: options.normalization,
      output: {
        values: view('float32', width * height * sliceCount),
        weightSum: view('float32', 1),
        status: {invalidCount: view('uint32', 1)}
      }
    })
  );
}

function addKFunction(
  {graph, view}: KernelAuditContext,
  options: {
    rows: number;
    distanceClassCount: number;
    timeClassCount: number;
    marginals?: boolean;
    pairCorrelation?: boolean;
    local?: boolean;
    intensities?: boolean;
  }
): void {
  const bins = options.distanceClassCount * options.timeClassCount;
  graph.add(
    new GPUSpaceTimeKFunction({
      positions: view('float32x2', options.rows),
      times: view('float32', options.rows),
      intensities: options.intensities ? view('float32', options.rows) : undefined,
      mask: view('uint32', options.rows),
      parameters: view('float32', GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH),
      gridSize: [64, 64],
      distanceClassCount: options.distanceClassCount,
      timeClassCount: options.timeClassCount,
      output: {
        k: view('float32', bins),
        ...(options.marginals
          ? {
              spatialK: view('float32', options.distanceClassCount),
              temporalK: view('float32', options.timeClassCount),
              d: view('float32', bins)
            }
          : {}),
        ...(options.pairCorrelation ? {pairCorrelation: view('float32', bins)} : {}),
        pairCounts: view('uint32', bins),
        ...(options.local ? {local: view('float32', options.rows * bins)} : {}),
        status: view('uint32', 1)
      }
    })
  );
}

function addDiggleTest(
  {graph, view}: KernelAuditContext,
  options: {
    rows: number;
    distanceClassCount: number;
    timeClassCount: number;
    permutationCount: number;
  }
): void {
  const bins = options.distanceClassCount * options.timeClassCount;
  graph.add(
    new GPUSpaceTimeDiggleTest({
      positions: view('float32x2', options.rows),
      times: view('float32', options.rows),
      mask: view('uint32', options.rows),
      parameters: view('float32', GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH),
      randomParameters: view('uint32', GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH),
      gridSize: [64, 64],
      distanceClassCount: options.distanceClassCount,
      timeClassCount: options.timeClassCount,
      permutationCount: options.permutationCount,
      output: {
        d: view('float32', bins),
        statistics: view('float32', options.permutationCount + 1),
        summary: view('float32', GPU_SPACE_TIME_DIGGLE_SUMMARY_LENGTH),
        permutationMean: view('float32', bins),
        permutationStandardDeviation: view('float32', bins),
        k: view('float32', bins),
        spatialK: view('float32', options.distanceClassCount),
        temporalK: view('float32', options.timeClassCount),
        status: view('uint32', 1)
      }
    })
  );
}

/** Audit cases of the space-time kernel density and relative risk (STX-04) and of the space-time K family (STX-05). */
export const STX04_STX05_SPACE_TIME_DENSITY_AND_K_AUDIT_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: '256 by 256 cells 64 slices 1 million points every output',
    build: context =>
      addDensity(context, {
        rows: 1_000_000,
        width: 256,
        height: 256,
        sliceCount: 64,
        indexGridSize: [512, 512],
        normalization: 'probability'
      })
  },
  ...(['gaussian', 'quartic', 'epanechnikov', 'bisquare'] as const).map(
    (spatialKernel, index): KernelAuditCase => ({
      exportName: 'GPUSpaceTimeKernelDensity',
      name: `${spatialKernel} space ${(['quartic', 'epanechnikov', 'gaussian'] as const)[index % 3]} time`,
      build: context =>
        addDensity(context, {
          rows: 5000,
          width: 32,
          height: 24,
          sliceCount: 8,
          indexGridSize: [32, 32],
          spatialKernel,
          temporalKernel: (['quartic', 'epanechnikov', 'gaussian'] as const)[index % 3],
          edge: index % 2 === 0 ? 'uniform' : 'none',
          normalization: index % 2 === 0 ? 'intensity' : 'probability'
        })
    })
  ),
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: 'voxels and points at the binding limit',
    build: context =>
      addDensity(context, {
        rows: MAXIMUM_ROWS_16,
        width: 512,
        height: 256,
        sliceCount: 256,
        indexGridSize: [4096, 2048],
        spatialKernel: 'epanechnikov',
        edge: 'none'
      })
  },
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: 'one slice above the voxel limit',
    build: context =>
      addDensity(context, {
        rows: 1000,
        width: 512,
        height: 256,
        sliceCount: 257,
        indexGridSize: [64, 64]
      })
  },
  {
    exportName: 'GPUSpaceTimeKernelDensity',
    name: 'one point above the row limit',
    build: context =>
      addDensity(context, {
        rows: MAXIMUM_ROWS_16 + 1,
        width: 64,
        height: 64,
        sliceCount: 4,
        indexGridSize: [64, 64]
      })
  },
  {
    exportName: 'GPUSpaceTimeRelativeRisk',
    name: 'cases and controls with every output',
    build: ({graph, view}) =>
      graph.add(
        new GPUSpaceTimeRelativeRisk({
          casePositions: view('float32x2', 20000),
          caseTimes: view('float32', 20000),
          caseWeights: view('float32', 20000),
          controlPositions: view('float32x2', 60000),
          controlTimes: view('float32', 60000),
          controlWeights: view('float32', 60000),
          parameters: view('float32', GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH),
          grid: makeGPURasterMetadata({width: 128, height: 96}),
          sliceCount: 24,
          indexGridSize: [128, 128],
          indexBounds: [0, 0, 100, 100],
          output: {
            values: view('float32', 128 * 96 * 24),
            caseDensity: view('float32', 128 * 96 * 24),
            controlDensity: view('float32', 128 * 96 * 24),
            status: {
              caseInvalidCount: view('uint32', 1),
              controlInvalidCount: view('uint32', 1)
            }
          }
        })
      )
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'maximum rows with marginals and D',
    build: context =>
      addKFunction(context, {
        rows: MAXIMUM_ROWS_16,
        distanceClassCount: 8,
        timeClassCount: 8,
        marginals: true
      })
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'one row above the row limit',
    build: context =>
      addKFunction(context, {rows: MAXIMUM_ROWS_16 + 1, distanceClassCount: 4, timeClassCount: 4})
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'local statistics at 64 classes and the binding-limited row count',
    build: context =>
      addKFunction(context, {
        rows: MAXIMUM_LOCAL_ROWS,
        distanceClassCount: 8,
        timeClassCount: 8,
        local: true
      })
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'local statistics one row above the binding limit',
    build: context =>
      addKFunction(context, {
        rows: MAXIMUM_LOCAL_ROWS + 1,
        distanceClassCount: 8,
        timeClassCount: 8,
        local: true
      })
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'pair correlation at 20 by 20 classes',
    build: context =>
      addKFunction(context, {
        rows: 200_000,
        distanceClassCount: 20,
        timeClassCount: 20,
        pairCorrelation: true
      })
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: 'intensities and pair correlation',
    build: context =>
      addKFunction(context, {
        rows: 100_000,
        distanceClassCount: 12,
        timeClassCount: 10,
        pairCorrelation: true,
        intensities: true
      })
  },
  {
    exportName: 'GPUSpaceTimeKFunction',
    name: '64 distance classes by 6 time classes with marginals',
    build: context =>
      addKFunction(context, {
        rows: 100_000,
        distanceClassCount: 64,
        timeClassCount: 6,
        marginals: true
      })
  },
  {
    exportName: 'GPUSpaceTimeDiggleTest',
    name: '999 permutations over 100000 rows at 8 by 8 classes',
    build: context =>
      addDiggleTest(context, {
        rows: 100_000,
        distanceClassCount: 8,
        timeClassCount: 8,
        permutationCount: 999
      })
  },
  {
    exportName: 'GPUSpaceTimeDiggleTest',
    name: '999 permutations over the maximum rows at 4 by 4 classes',
    build: context =>
      addDiggleTest(context, {
        rows: MAXIMUM_ROWS_16,
        distanceClassCount: 4,
        timeClassCount: 4,
        permutationCount: 999
      })
  },
  {
    exportName: 'GPUSpaceTimeDiggleTest',
    name: 'one permutation above the maximum',
    build: context =>
      addDiggleTest(context, {
        rows: 1000,
        distanceClassCount: 4,
        timeClassCount: 4,
        permutationCount: 1000
      })
  }
];
