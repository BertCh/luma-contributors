// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateClassificationScratch} from '../column-classification/column-classification-rows';

/** Preflights the indexed pair front end, histogram, optional moments and nested IQR selection. @internal */
export function validatePairStatisticsBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    views: Readonly<Record<string, GraphDataView | undefined>>;
    rows: number;
    gridSize: readonly [number, number];
    slots: number;
    channels: number;
    moments?: boolean;
    unitVariance?: boolean;
    additionalScratch?: Readonly<Record<string, number>>;
  }
): void {
  validateGraphChunkBindingRanges(graph, id, props.views);
  validateGraphChunkGroupKernelSupport(graph, id, 2);
  const lengths = {
    sortedPoints: props.rows * 3,
    cellOffsets: props.gridSize[0] * props.gridSize[1] + 1,
    moments: props.moments ? props.rows + 6 : 0,
    histogram: props.slots * props.channels * 2 + 1,
    windowParameters: props.views['parameters']?.length ?? 0,
    scalars: 2,
    ...props.additionalScratch
  };
  for (const [name, length] of Object.entries(lengths)) {
    if (length > 0) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        `${name} scratch`,
        length * 4,
        'reduce rows, gridSize or histogram bins'
      );
      validateMaximumCount(
        id,
        `${name} scratch byteLength`,
        length * 4,
        graph.device.limits.maxBufferSize
      );
    }
  }
  // Two linear-interpolated quartiles need four target ranks, 256 bins each and 4*T+2 state words.
  // Call the owned helper so a change to quantile scratch validation is inherited here.
  if (props.unitVariance)
    validateClassificationScratch(graph, id, {histogram: 4 * 256, state: 4 * 4 + 2});
}
