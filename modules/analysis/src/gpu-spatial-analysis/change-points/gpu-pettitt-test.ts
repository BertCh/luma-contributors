// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpaceTimeWeightsLayout} from '../spatial-weights/spatial-weights-space-time-stages';
import {getChangePointSeriesReaderWGSL} from './change-point-series-wgsl';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUPettittTest';

/** Largest `sliceCount`: every invocation compares `sliceCount^2` pairs. */
export const GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT = 512;
/** `changeSlice` value of a cell with fewer than two finite slices. */
export const GPU_PETTITT_TEST_NO_CHANGE_SLICE = 0xffffffff;

/** Caller-owned outputs of {@link GPUPettittTest}, one row per cell. */
export type GPUPettittTestOutput = {
  /** Pettitt's `K = max_t |U_t|` of the finite series, an exact integer (uint32). */
  statistic: GraphDataView<'uint32'>;
  /** First slice after the most likely change: `t + 1` of the first `t` reaching `K` (uint32). */
  changeSlice: GraphDataView<'uint32'>;
  /** Asymptotic two-sided p-value `min(1, 2 exp(-6 K^2 / (n^3 + n^2)))`, 1 for `n < 2`. */
  pValues: GraphDataView<'float32'>;
};

/** Properties for {@link GPUPettittTest}. */
export type GPUPettittTestProps = {
  /** Prefix for generated node IDs. Defaults to `'pettitt-test'`. */
  id?: string;
  /** `cellCount * sliceCount` float32 values in `layout`. NaN is skipped. */
  values: GraphDataView<'float32'>;
  /** Cube layout, see `GPUCubeLayoutTranspose`. */
  layout: GPUSpaceTimeWeightsLayout;
  /** Compile-time cells. */
  cellCount: number;
  /** Compile-time slices, at most `GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT`. */
  sliceCount: number;
  /** Caller-owned outputs. */
  output: GPUPettittTestOutput;
};

/**
 * Pettitt (1979) non-parametric single change point test per location. `U_t = sum_{i<=t} sum_{j>t}
 * sign(x_i - x_j)` runs in exact integers over the finite series (ties contribute 0; NaN is
 * skipped), one invocation and `sliceCount^2` comparisons per cell. The p-value is the usual
 * asymptotic approximation; it is two-sided.
 *
 * Compile-time: `cellCount`, `sliceCount`, `layout`. Per-frame: the contents of `values`.
 */
export class GPUPettittTest implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPettittTestProps;

  constructor(props: GPUPettittTestProps) {
    const id = props.id ?? 'pettitt-test';
    this.id = id;
    this.props = props;
    const {values, cellCount, sliceCount, output} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'sliceCount', sliceCount, 1, GPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT);
    validateSinglePackedView(values, ['float32'], `${id} values`);
    validateViewLength(id, 'values', values, cellCount * sliceCount);
    validateSinglePackedView(output.statistic, ['uint32'], `${id} statistic`);
    validateViewLength(id, 'statistic', output.statistic, cellCount);
    validateSinglePackedView(output.changeSlice, ['uint32'], `${id} changeSlice`);
    validateViewLength(id, 'changeSlice', output.changeSlice, cellCount);
    validateSinglePackedView(output.pValues, ['float32'], `${id} pValues`);
    validateViewLength(id, 'pValues', output.pValues, cellCount);
    validateDisjointWritableViews(id, output);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.statistic, output.changeSlice, output.pValues],
      [values]
    );
  }

  /** Returns the one kernel of this contributor. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateSpatialStatisticsBindings(graph, id, props);
    const {values, layout, cellCount, sliceCount, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      output.statistic,
      output.changeSlice,
      output.pValues
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'values',
      cellCount * sliceCount * 4,
      'reduce cellCount * sliceCount'
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-statistic`,
        operation: OPERATION,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'statistic', view: output.statistic, type: 'u32', access: 'read_write'},
          {name: 'changeSlice', view: output.changeSlice, type: 'u32', access: 'read_write'},
          {name: 'pValues', view: output.pValues, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `${IS_FINITE_VALUE_WGSL}
${GET_NAN_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const NO_CHANGE_SLICE: u32 = ${GPU_PETTITT_TEST_NO_CHANGE_SLICE}u;
${getChangePointSeriesReaderWGSL(layout)}`,
        body: `var series: array<f32, SLICE_COUNT>;
  var position: array<u32, SLICE_COUNT>;
  var seriesLength = 0u;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = readSeriesValue(index, slice);
    if (isFiniteValue(value)) {
      series[seriesLength] = value;
      position[seriesLength] = slice;
      seriesLength++;
    }
  }
  var running = 0;
  var largest = 0;
  var largestRow = 0u;
  if (seriesLength >= 2u) {
    for (var row = 0u; row + 1u < seriesLength; row++) {
      var columnSum = 0;
      for (var other = 0u; other < seriesLength; other++) {
        let difference = series[other] - series[row];
        columnSum += select(select(0, -1, difference < 0.0), 1, difference > 0.0);
      }
      running += columnSum;
      let magnitude = abs(running);
      if (magnitude > largest) {
        largest = magnitude;
        largestRow = row;
      }
    }
  }
  statistic[statisticOffset + index] = u32(largest);
  if (seriesLength >= 2u) {
    let lengthValue = f32(seriesLength);
    let statisticValue = f32(largest);
    changeSlice[changeSliceOffset + index] = position[largestRow + 1u];
    pValues[pValuesOffset + index] = min(1.0, 2.0 * exp(-6.0 * statisticValue * statisticValue / (lengthValue * lengthValue * lengthValue + lengthValue * lengthValue)));
  } else {
    changeSlice[changeSliceOffset + index] = NO_CHANGE_SLICE;
    pValues[pValuesOffset + index] = 1.0;
  }`
      })
    ];
  }
}
