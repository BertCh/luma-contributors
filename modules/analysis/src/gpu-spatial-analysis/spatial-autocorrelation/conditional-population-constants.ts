// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** @internal Exact extrema multiplicities distinguish a constant leave-one-out population. */
export function getConditionalPopulationConstantNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number,
  values: GraphDataView<'float32'>,
  statistics: GraphDataView<'float32'>,
  counts: GraphDataView<'uint32'>,
  parameters: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  const minimumRows = createTransientView(graph, `${id}-minimum-rows`, 'float32', rows),
    maximumRows = createTransientView(graph, `${id}-maximum-rows`, 'float32', rows);
  const minimum = createTransientView(graph, `${id}-minimum`, 'float32', 1),
    maximum = createTransientView(graph, `${id}-maximum`, 'float32', 1);
  const minimumFlags = createTransientView(graph, `${id}-minimum-flags`, 'uint32', rows),
    maximumFlags = createTransientView(graph, `${id}-maximum-flags`, 'uint32', rows);
  const minimumCount = createTransientView(graph, `${id}-minimum-count`, 'uint32', 1),
    maximumCount = createTransientView(graph, `${id}-maximum-count`, 'uint32', 1);
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, type: view.format === 'uint32' ? 'u32' : 'f32', access});
  const node = (name: string, size: number, bindings: WGSLKernelBinding[], body: string) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'spatial-conditional-population',
      variant: name,
      invocationCount: size,
      bindings,
      declarations: IS_FINITE_BITS_WGSL,
      body
    });
  return [
    node(
      'extrema-rows',
      rows,
      [
        binding('statistics', statistics),
        binding('values', values),
        binding('minimumRows', minimumRows, 'read_write'),
        binding('maximumRows', maximumRows, 'read_write')
      ],
      `let value = values[valuesOffset + index]; let finite = isFiniteBits(statistics[statisticsOffset + index]);
minimumRows[minimumRowsOffset + index] = select(3.402823466e38, value, finite); maximumRows[maximumRowsOffset + index] = select(-3.402823466e38, value, finite);`
    ),
    ...new GPUReduction({
      id: `${id}-minimum`,
      input: minimumRows,
      output: minimum,
      operation: 'min'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-maximum`,
      input: maximumRows,
      output: maximum,
      operation: 'max'
    }).getCommandNodes(graph),
    node(
      'extrema-flags',
      rows,
      [
        binding('statistics', statistics),
        binding('values', values),
        binding('minimum', minimum),
        binding('maximum', maximum),
        binding('minimumFlags', minimumFlags, 'read_write'),
        binding('maximumFlags', maximumFlags, 'read_write')
      ],
      `let value = values[valuesOffset + index]; let finite = isFiniteBits(statistics[statisticsOffset + index]);
minimumFlags[minimumFlagsOffset + index] = u32(finite && value == minimum[minimumOffset]); maximumFlags[maximumFlagsOffset + index] = u32(finite && value == maximum[maximumOffset]);`
    ),
    ...new GPUReduction({
      id: `${id}-minimum-count`,
      input: minimumFlags,
      output: minimumCount,
      operation: 'sum'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-maximum-count`,
      input: maximumFlags,
      output: maximumCount,
      operation: 'sum'
    }).getCommandNodes(graph),
    // Raw equality counts must not inherit any loss of donor distinctions during centering.
    node(
      'centered-extrema-rows',
      rows,
      [
        binding('statistics', statistics),
        binding('minimumRows', minimumRows, 'read_write'),
        binding('maximumRows', maximumRows, 'read_write')
      ],
      `let value = statistics[statisticsOffset + index]; let finite = isFiniteBits(value);
minimumRows[minimumRowsOffset + index] = select(3.402823466e38, value, finite); maximumRows[maximumRowsOffset + index] = select(-3.402823466e38, value, finite);`
    ),
    ...new GPUReduction({
      id: `${id}-centered-minimum`,
      input: minimumRows,
      output: minimum,
      operation: 'min'
    }).getCommandNodes(graph),
    ...new GPUReduction({
      id: `${id}-centered-maximum`,
      input: maximumRows,
      output: maximum,
      operation: 'max'
    }).getCommandNodes(graph),
    node(
      'conditional-constants',
      1,
      [
        binding('statistics', statistics, 'read_write'),
        binding('minimum', minimum),
        binding('maximum', maximum),
        binding('minimumCount', minimumCount),
        binding('maximumCount', maximumCount),
        binding('counts', counts),
        binding('parameters', parameters)
      ],
      `let others = max(counts[countsOffset], 1u) - 1u; let computed = parameters[parametersOffset + 1u] == 0.0;
statistics[statisticsOffset + ${rows + 6}u] = minimum[minimumOffset]; statistics[statisticsOffset + ${rows + 7}u] = maximum[maximumOffset];
statistics[statisticsOffset + ${rows + 8}u] = f32(u32(computed && minimumCount[minimumCountOffset] >= others));
statistics[statisticsOffset + ${rows + 9}u] = f32(u32(computed && maximumCount[maximumCountOffset] >= others));`
    )
  ];
}
