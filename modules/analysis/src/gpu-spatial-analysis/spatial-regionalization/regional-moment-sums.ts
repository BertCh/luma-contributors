// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {REGIONAL_MOMENT_WGSL} from './regional-moment-wgsl';

/** Parallel, staged segmented moments; every invocation combines at most two split sums. @internal */
export function createRegionalMomentSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'>;
    sortedRows: GraphDataView<'uint32'>;
    sortedKeys: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    origins: GraphDataView<'float32'>;
    columnCount: number;
    high: GraphDataView<'float32'>;
    low: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, values, sortedRows, sortedKeys, offsets, origins, columnCount} = props;
  const rows = sortedRows.length;
  const labels = offsets.length - 1;
  const length = rows * columnCount;
  let high = createTransientView(graph, `${id}-high`, 'float32', length);
  let low = createTransientView(graph, `${id}-low`, 'float32', length);
  let nextHigh = createTransientView(graph, `${id}-next-high`, 'float32', length);
  let nextLow = createTransientView(graph, `${id}-next-low`, 'float32', length);
  const declarations = `${REGIONAL_MOMENT_WGSL}
const COLUMNS: u32 = ${columnCount}u;
const LABELS: u32 = ${labels}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-prepare`,
      operation,
      invocationCount: length,
      declarations,
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'origins', view: origins, type: 'f32', access: 'read'},
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'high', view: high, type: 'f32', access: 'read_write'},
        {name: 'low', view: low, type: 'f32', access: 'read_write'}
      ],
      body: `let row = index / COLUMNS;
  var moment = vec2<f32>(0.0);
  if (sortedKeys[sortedKeysOffset + row] < LABELS) {
    moment = subtractRegionalValue(values[valuesOffset + sortedRows[sortedRowsOffset + row] * COLUMNS + index % COLUMNS],
      origins[originsOffset + sortedKeys[sortedKeysOffset + row] * COLUMNS + index % COLUMNS], offsets[offsetsOffset]);
  }
  high[highOffset + index] = moment.x;
  low[lowOffset + index] = moment.y;`
    })
  ];
  for (let stride = 1; stride < rows; stride *= 2) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-stage-${stride}`,
        operation,
        invocationCount: length,
        declarations: `${declarations}\noverride MOMENT_STRIDE: u32;`,
        constants: {MOMENT_STRIDE: stride},
        bindings: [
          {name: 'keys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'high', view: high, type: 'f32', access: 'read'},
          {name: 'low', view: low, type: 'f32', access: 'read'},
          {name: 'nextHigh', view: nextHigh, type: 'f32', access: 'read_write'},
          {name: 'nextLow', view: nextLow, type: 'f32', access: 'read_write'}
        ],
        body: `let row = index / COLUMNS;
  var moment = vec2<f32>(high[highOffset + index], low[lowOffset + index]);
  if (row >= MOMENT_STRIDE && keys[keysOffset + row] == keys[keysOffset + row - MOMENT_STRIDE]) {
    let previous = index - MOMENT_STRIDE * COLUMNS;
    moment = addRegionalMoment(vec2<f32>(high[highOffset + previous], low[lowOffset + previous]), moment, offsets[offsetsOffset]);
  }
  nextHigh[nextHighOffset + index] = moment.x;
  nextLow[nextLowOffset + index] = moment.y;`
      })
    );
    [high, nextHigh] = [nextHigh, high];
    [low, nextLow] = [nextLow, low];
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-finish`,
      operation,
      invocationCount: labels * columnCount,
      declarations,
      bindings: [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'high', view: high, type: 'f32', access: 'read'},
        {name: 'low', view: low, type: 'f32', access: 'read'},
        {name: 'sums', view: props.high, type: 'f32', access: 'read_write'},
        {name: 'residuals', view: props.low, type: 'f32', access: 'read_write'}
      ],
      body: `let label = index / COLUMNS;
  let begin = offsets[offsetsOffset + label];
  let end = offsets[offsetsOffset + label + 1u];
  var moment = vec2<f32>(0.0);
  if (end > begin) {
    let source = (end - 1u) * COLUMNS + index % COLUMNS;
    moment = vec2<f32>(high[highOffset + source], low[lowOffset + source]);
  }
  sums[sumsOffset + index] = moment.x;
  residuals[residualsOffset + index] = moment.y;`
    })
  );
  return nodes;
}
