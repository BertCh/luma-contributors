// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUStopValueCubeRecipeProps} from '../../../src/gpu-spatial-analysis/recipes/stop-value-cube-recipe';

type Input = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
  format: Format,
  values: Uint32Array | Float32Array,
  length: number
) => GraphDataView<Format>;

/** Two live stops, one poison tail, signed original-fix times, and reversed dictionary order. */
export function createStopValueCubeFixture(input: Input): GPUStopValueCubeRecipeProps {
  return {
    time: 'start',
    family: 'quadbin',
    resolution: 1,
    // Quadbin z=1: SE path 3, NW path 0. Low 50 unused bits are ones.
    dictionary: input(
      'uint32x2',
      Uint32Array.of(0xffffffff, 0x481fffff, 0xffffffff, 0x4813ffff),
      2
    ),
    weights: {
      offsets: input('uint32', Uint32Array.of(0, 1, 2), 3),
      neighbors: input('uint32', Uint32Array.of(1, 0), 2),
      weights: input('float32', Float32Array.of(1, 1), 2),
      status: {
        overflow: input('uint32', Uint32Array.of(0), 1),
        candidateOverflow: input('uint32', Uint32Array.of(0), 1)
      }
    },
    // Original fix rows are -5, 5 and 15 seconds since Unix epoch, signed Int64 words.
    timestamps: input('uint32x2', Uint32Array.of(0xfffffffb, 0xffffffff, 5, 0, 15, 0), 3),
    temporalContext: {
      unit: 'second',
      epoch: 'unix',
      encoding: 'int64-words',
      axis: {kind: 'irregular'}
    },
    periods: {kind: 'fixed', origin: -10n, width: 10, count: 3},
    stops: {
      centroids: input('float32x2', Float32Array.of(-90, 45, 90, -45, -90, 45), 3),
      startRows: input('uint32', Uint32Array.of(0, 2, 0xffffffff), 3),
      finishRows: input('uint32', Uint32Array.of(1, 2, 0xffffffff), 3),
      status: {
        count: input('uint32', Uint32Array.of(2), 1),
        overflow: input('uint32', Uint32Array.of(0), 1),
        candidateOverflow: input('uint32', Uint32Array.of(0), 1)
      }
    },
    sliceStatuses: [
      {
        overflow: input('uint32', Uint32Array.of(0), 1),
        candidateOverflow: input('uint32', Uint32Array.of(0), 1)
      }
    ]
  };
}
