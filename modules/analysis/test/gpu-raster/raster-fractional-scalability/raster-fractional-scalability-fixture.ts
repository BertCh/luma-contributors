// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterFractionalCoverageProps} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import type {GPURasterFractionalCategoricalStatisticsProps} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';

export function createScalabilityProps(
  graph: GPUCommandGraph,
  zoneCount = 2,
  tupleCapacity = 16,
  grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 4,
    height: 4,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  }
) {
  let identifier = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `fractional-${identifier++}`, format, length);
  const coverage: GPURasterFractionalCoverageProps = {
    grid,
    zones: {
      positions: view('float32x2', zoneCount * 4),
      featureOffsets: view('uint32', zoneCount + 1),
      polygonOffsets: view('uint32', zoneCount + 1),
      ringOffsets: view('uint32', zoneCount + 1)
    },
    parameters: view('float32', 8),
    output: {
      zoneIds: view('uint32', tupleCapacity),
      cellIds: view('uint32', tupleCapacity),
      fractions: view('float32', tupleCapacity),
      zoneValidity: view('uint32', zoneCount),
      status: {
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      }
    }
  };
  const categoryCapacity = 2;
  const categorical: GPURasterFractionalCategoricalStatisticsProps = {
    grid,
    zoneCount,
    categoryCapacity,
    coverage: coverage.output,
    parameters: coverage.parameters,
    input: {
      id: 'categories',
      format: 'uint32',
      storage: {kind: 'buffer', values: view('uint32', grid.width * grid.height)}
    },
    output: {
      coverageSums: view('float32', zoneCount * categoryCapacity),
      classAreas: view('float32', zoneCount * categoryCapacity),
      proportions: view('float32', zoneCount * categoryCapacity),
      coveredArea: view('float32', zoneCount),
      validArea: view('float32', zoneCount),
      missingArea: view('float32', zoneCount),
      majority: view('uint32', zoneCount),
      minority: view('uint32', zoneCount),
      validity: view('uint32', zoneCount)
    }
  };
  return {coverage, categorical};
}
