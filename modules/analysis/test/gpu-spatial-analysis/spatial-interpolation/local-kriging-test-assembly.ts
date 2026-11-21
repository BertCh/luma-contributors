// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';

/** Forces small query chunks for sample-index tests without changing public props. @internal */
export function createLocalKrigingForTesting<
  SupportRows extends
    | GraphDataView<'float32x2' | 'float32x4'>
    | GraphVectorView<'float32x2' | 'float32x4'>,
  ValueRows extends GraphDataView<'float32x4'> | GraphVectorView<'float32x4'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'>,
  WeightRows extends GraphDataView<'float32'> | GraphVectorView<'float32'>,
  DriftRows extends
    | GraphDataView<'float32' | 'float32x4'>
    | GraphVectorView<'float32' | 'float32x4'>
>(
  props: GPULocalKrigingProps<SupportRows, ValueRows, StatusRows, WeightRows, DriftRows> & {
    sampleIndexChunkQueryCount?: number;
  }
): GPULocalKriging<SupportRows, ValueRows, StatusRows, WeightRows, DriftRows> {
  const {sampleIndexChunkQueryCount, ...publicProps} = props;
  class TestLocalKriging extends GPULocalKriging<
    SupportRows,
    ValueRows,
    StatusRows,
    WeightRows,
    DriftRows
  > {
    protected override get sampleIndexChunkQueryCount(): number | undefined {
      return sampleIndexChunkQueryCount;
    }
  }
  return new TestLocalKriging(publicProps);
}
