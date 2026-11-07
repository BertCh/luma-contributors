// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPURasterConservativeRemapping,
  GPURasterTexture,
  GPURasterTemporal,
  GPURasterRegistration,
  GPURasterRegionGraph,
  GPURasterSpectralUnmixing,
  GPURasterSeasonalBreak,
  makeGPURasterMetadata,
  addRasterRegionGraphTiles
} from '../../../src/gpu-raster';
import type {KernelAuditCase} from './kernel-audit-context';

/** Every bounded remote-sensing kernel variant at documented model and binding limits. */
export const COVERAGE_RASTER_KERNEL_CASES: KernelAuditCase[] = [
  ...(['destination-area', 'covered-area', 'extensive'] as const).flatMap(normalization =>
    (['scratch', 'prepare', 'reuse'] as const).map(weightMode => ({
      exportName: 'GPURasterConservativeRemapping',
      name: `maximum-256-overlap-slots-${normalization}-${weightMode}`,
      build: ({graph, view}: Parameters<KernelAuditCase['build']>[0]) => {
        // 131072 destinations * 256 candidate slots * four bytes = one 128 MiB binding.
        const sourceCount = 8192 * 4096;
        const destinationCount = 512 * 256;
        graph.add(
          new GPURasterConservativeRemapping({
            source: {width: 8192, height: 4096, origin: [1e7, 1e7], spacing: [1, 1]},
            destination: {
              width: 512,
              height: 256,
              origin: [1e7 + 0.5, 1e7 + 0.5],
              spacing: [15, 15]
            },
            values: view('float32', sourceCount),
            validity: view('uint32', sourceCount),
            normalization,
            weightMode: weightMode === 'scratch' ? undefined : weightMode,
            preparedWeights:
              weightMode === 'scratch'
                ? undefined
                : {
                    indices: view('uint32', destinationCount * 256),
                    areas: view('float32', destinationCount * 256)
                  },
            output: {
              values: view('float32', destinationCount),
              coveredArea: view('float32', destinationCount),
              status: view('uint32', destinationCount)
            }
          })
        );
      }
    }))
  ),
  {
    exportName: 'GPURasterRegionGraph',
    name: 'maximum-source-binding-explicit-core-assembly',
    build: ({graph, view}) => {
      const width = 4096,
        height = 8192;
      addRasterRegionGraphTiles(graph, {
        grid: makeGPURasterMetadata({width, height}),
        regionCount: 256,
        maximumIterations: 256,
        mergeThreshold: 1,
        tiles: [0, 4096].map(row => ({
          grid: makeGPURasterMetadata({width, height: 4096}),
          origin: [0, row] as const,
          labels: view('uint32', width * 4096),
          values: view('float32', width * 4096),
          sourceRegionIds: view('uint32', 256)
        })),
        output: {
          adjacency: view('uint32', 256 * 256),
          weights: view('float32', 256 * 256),
          mapping: view('uint32', 256),
          status: view('uint32', 1)
        }
      });
    }
  },
  {
    exportName: 'GPURasterTexture',
    name: 'maximum-levels-radius-offset-and-feature-binding',
    build: ({graph, view, maximumRows}) => {
      const count = maximumRows(7 * 4);
      graph.add(
        new GPURasterTexture({
          grid: makeGPURasterMetadata({width: count, height: 1}),
          values: view('float32', count),
          validity: view('uint32', count),
          levels: 32,
          radius: 8,
          offset: [8, -8],
          range: [-100, 100],
          symmetric: true,
          features: view('float32', count * 7),
          pairCount: view('uint32', count)
        })
      );
    }
  },
  ...(['mean', 'minimum', 'maximum', 'median', 'savitzky-golay'] as const).map(operation => ({
    exportName: 'GPURasterTemporal',
    name: `maximum-slices-${operation}`,
    build: ({graph, view, maximumRows}: Parameters<KernelAuditCase['build']>[0]) => {
      const cellCount = Math.floor(maximumRows(4) / 256);
      const length = cellCount * 256;
      const outputLength = operation === 'savitzky-golay' ? length : cellCount;
      graph.add(
        new GPURasterTemporal({
          cellCount,
          sliceCount: 256,
          values: view('float32', length),
          validity: view('uint32', length),
          operation,
          windowLength: 15,
          polynomialOrder: 3,
          edgeMode: 'interp',
          output: {
            values: view('float32', outputLength),
            counts: view('uint32', outputLength),
            chosenSlice: view('uint32', outputLength),
            status: view('uint32', outputLength)
          }
        })
      );
    }
  })),
  {
    exportName: 'GPURasterTemporal',
    name: 'maximum-sg-truncated-support',
    build: ({graph, view, maximumRows}) => {
      const cellCount = Math.floor(maximumRows(4) / 256);
      const length = cellCount * 256;
      graph.add(
        new GPURasterTemporal({
          cellCount,
          sliceCount: 256,
          values: view('float32', length),
          validity: view('uint32', length),
          operation: 'savitzky-golay',
          windowLength: 15,
          polynomialOrder: 3,
          edgeMode: 'truncate',
          output: {
            values: view('float32', length),
            counts: view('uint32', length),
            chosenSlice: view('uint32', length),
            status: view('uint32', length)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURasterRegistration',
    name: 'maximum-periodic-dimensions-and-refinement',
    build: ({graph, view}) => {
      graph.add(
        new GPURasterRegistration({
          grid: makeGPURasterMetadata({width: 2048, height: 2048}),
          reference: view('float32', 2048 ** 2),
          moving: view('float32', 2048 ** 2),
          upsampleFactor: 32,
          output: {shift: view('float32x2', 1), peak: view('float32', 1), status: view('uint32', 1)}
        })
      );
    }
  },
  {
    exportName: 'GPURasterRegistration',
    name: 'maximum-masked-linear-support-and-refinement',
    build: ({graph, view}) => {
      graph.add(
        new GPURasterRegistration({
          mode: 'masked-linear',
          grid: makeGPURasterMetadata({width: 512, height: 512}),
          reference: view('float32', 512 ** 2),
          moving: view('float32', 512 ** 2),
          referenceValidity: view('uint32', 512 ** 2),
          movingValidity: view('uint32', 512 ** 2),
          maximumShift: [511, 511],
          minimumOverlapFraction: 1,
          minimumOverlapCount: 512 ** 2,
          upsampleFactor: 32,
          output: {
            shift: view('float32x2', 1),
            peak: view('float32', 1),
            status: view('uint32', 1),
            overlap: view('uint32', 1)
          }
        })
      );
    }
  },
  ...(['fixed', 'adaptive'] as const).map(mergeMode => ({
    exportName: 'GPURasterRegionGraph',
    name: `maximum-regions-iterations-and-inputs-${mergeMode}`,
    build: ({graph, view, maximumRows}: Parameters<KernelAuditCase['build']>[0]) => {
      const count = maximumRows(4);
      graph.add(
        new GPURasterRegionGraph({
          grid: makeGPURasterMetadata({width: count, height: 1}),
          labels: view('uint32', count),
          values: view('float32', count),
          regionCount: 256,
          mergeMode,
          mergeThreshold: 2,
          maximumIterations: 256,
          output: {
            adjacency: view('uint32', 256 ** 2),
            weights: view('float32', 256 ** 2),
            mapping: view('uint32', 256),
            status: view('uint32', 1)
          }
        })
      );
    }
  })),
  ...(['unconstrained', 'nonnegative', 'simplex'] as const).map(constraint => ({
    exportName: 'GPURasterSpectralUnmixing',
    name: `maximum-bands-members-iterations-${constraint}`,
    build: ({graph, view, maximumRows}: Parameters<KernelAuditCase['build']>[0]) => {
      const bandCount = constraint === 'unconstrained' ? 219 : 195;
      const cellCount = Math.floor(maximumRows(4) / bandCount);
      graph.add(
        new GPURasterSpectralUnmixing({
          cellCount,
          bandCount,
          endmemberCount: 8,
          values: view('float32', cellCount * bandCount),
          validity: view('uint32', cellCount * bandCount),
          endmembers: view('float32', bandCount * 8),
          constraint,
          maximumIterations: 4096,
          output: {
            abundances: view('float32', cellCount * 8),
            residual: view('float32', cellCount),
            status: view('uint32', cellCount),
            iterationCounts: view('uint32', cellCount)
          }
        })
      );
    }
  })),
  {
    exportName: 'GPURasterSeasonalBreak',
    name: 'maximum-dates-component-binding-and-candidate-boundaries',
    build: ({graph, view, maximumRows}) => {
      const cellCount = Math.floor(maximumRows(4) / (256 * 3));
      const length = cellCount * 256;
      graph.add(
        new GPURasterSeasonalBreak({
          cellCount,
          sliceCount: 256,
          period: 12,
          minimumSegmentLength: 3,
          values: view('float32', length),
          validity: view('uint32', length),
          output: {
            coefficients: view('float32', cellCount * 5),
            components: view('float32', length * 3),
            breakIndex: view('uint32', cellCount),
            improvement: view('float32', cellCount),
            status: view('uint32', cellCount)
          }
        })
      );
    }
  },
  ...(['joint', 'iterative'] as const).map(
    fitMethod =>
      ({
        exportName: 'GPURasterSeasonalBreak',
        name: `dummy-season-${fitMethod}-monitoring-missing-interpolation`,
        build: ({graph, view, maximumRows}) => {
          const cellCount = Math.floor(maximumRows(4) / (256 * 3));
          graph.add(
            new GPURasterSeasonalBreak({
              cellCount,
              sliceCount: 256,
              period: 12,
              minimumSegmentLength: 32,
              seasonality: 'dummy',
              fitMethod,
              maximumIterations: 64,
              missingPolicy: 'interpolate',
              monitoring: {historyLength: 128, threshold: 4, minimumConsecutive: 3},
              values: view('float32', cellCount * 256),
              validity: view('uint32', cellCount * 256),
              output: {
                coefficients: view('float32', cellCount * 14),
                components: view('float32', cellCount * 256 * 3),
                breakIndex: view('uint32', cellCount),
                improvement: view('float32', cellCount),
                status: view('uint32', cellCount),
                diagnostics: view('float32', cellCount * 6)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  ),
  ...Array.from({length: 10}, (_, index) => index + 3).map(
    period =>
      ({
        exportName: 'GPURasterSeasonalBreak',
        name: `dummy-period-${period}-largest-accepted-history`,
        build: ({graph, view}) => {
          const columns = period + 2;
          let historyLength = 256;
          while (
            historyLength * columns + columns ** 2 + 2 * columns + 8 > 2048 ||
            historyLength * columns + historyLength + Math.ceil(historyLength / 32) + columns + 64 >
              2048
          )
            historyLength--;
          graph.add(
            new GPURasterSeasonalBreak({
              cellCount: 1,
              sliceCount: 256,
              period,
              seasonality: 'dummy',
              minimumSegmentLength: 3,
              missingPolicy: 'reject',
              monitoring:
                historyLength < 256
                  ? {historyLength, threshold: 4, minimumConsecutive: 1}
                  : undefined,
              values: view('float32', 256),
              validity: view('uint32', 256),
              output: {
                coefficients: view('float32', columns),
                components: view('float32', 768),
                breakIndex: view('uint32', 1),
                improvement: view('float32', 1),
                status: view('uint32', 1),
                diagnostics: view('float32', 6)
              }
            })
          );
        }
      }) satisfies KernelAuditCase
  )
];
