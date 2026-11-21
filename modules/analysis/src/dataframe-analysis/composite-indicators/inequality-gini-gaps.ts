// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkGroupKernelNode,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
} from '../../analysis-core/kernels/graph-chunk-groups';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createInequalityLorenzPrefixNodes} from './inequality-lorenz-prefix';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';

const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read'
});
const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read_write'
});

/**
 * Adjacent income gaps times directly accumulated left/right populations. Both scans use bounded
 * 256-row tiles. The right population is never a rounded total minus a prefix. Mantissa arithmetic
 * scales adjacent incomes before subtraction and weighted-income products before summation,
 * preserving normal coefficients even when an unscaled gap or product would underflow. @internal
 */
export function createInequalityGiniGapNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sortedRows: GraphDataView<'uint32'>;
    sortedZones: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    weights?: GraphDataView<'float32'>;
    zoneCount: number;
    output: GraphDataView<'float32'>;
    outputStride: number;
    outputSlot: number;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    sortedRows,
    sortedZones,
    values,
    weights,
    zoneCount,
    output,
    outputStride,
    outputSlot
  } = props;
  const rowCount = sortedRows.length;
  const reverseRows = createTransientView(graph, `${id}-reverse-rows`, 'uint32', rowCount);
  const reverseZones = createTransientView(graph, `${id}-reverse-zones`, 'uint32', rowCount);
  const offsets = createTransientView(graph, `${id}-offsets`, 'uint32', zoneCount + 1);
  // Population, scaled income mantissa sum, and its common binary exponent per zone.
  const totals = createTransientView(graph, `${id}-totals`, 'float32', 3 * zoneCount);
  const incomeExponents = createTransientView(
    graph,
    `${id}-income-exponents`,
    'float32',
    zoneCount
  );
  const terms = createTransientView(graph, `${id}-terms`, 'float32', rowCount);
  const sums = createTransientView(graph, `${id}-sums`, 'float32', zoneCount);
  const forward = createInequalityLorenzPrefixNodes(graph, {
    id: `${id}-left`,
    sortedRows,
    sortedZones,
    values,
    weights,
    zoneCount
  });
  const reverse = createInequalityLorenzPrefixNodes(graph, {
    id: `${id}-right`,
    sortedRows: reverseRows,
    sortedZones: reverseZones,
    values,
    weights,
    zoneCount
  });
  return [
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-reverse`,
      operation: 'GPUInequality',
      variant: 'gini-reverse',
      invocationCount: rowCount,
      parameterValues: [rowCount],
      bindings: [
        read('rows', sortedRows, 'u32'),
        read('zones', sortedZones, 'u32'),
        write('reverseRows', reverseRows, 'u32'),
        write('reverseZones', reverseZones, 'u32')
      ],
      body: `let source = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u) - 1u - index;
  reverseRows[reverseRowsOffset + index] = rows[rowsOffset + source];
  reverseZones[reverseZonesOffset + index] = zones[zonesOffset + source];`
    }),
    ...forward.nodes,
    ...reverse.nodes,
    ...getSortedSegmentOffsetNodes(graph, {
      id,
      operation: 'GPUInequality',
      groupCount: zoneCount,
      sortedGroupKeys: sortedZones,
      segmentOffsets: offsets
    }),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-income-row-exponents`,
      operation: 'GPUInequality',
      variant: 'gini-income-row-exponents',
      invocationCount: rowCount,
      parameterValues: [zoneCount],
      bindings: [
        read('rows', sortedRows, 'u32'),
        read('zones', sortedZones, 'u32'),
        read('values', values, 'f32'),
        ...(weights ? [read('weights', weights, 'f32')] : []),
        write('terms', terms, 'f32')
      ],
      body: `terms[termsOffset + index] = -512.0;
  if (zones[zonesOffset + index] >= getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u)) { return; }
  let row = rows[rowsOffset + index]; let value = values[valuesOffset + row];
  if (!(value > 0.0)) { return; }
  let weight = ${weights ? 'weights[weightsOffset + row]' : '1.0'};
  terms[termsOffset + index] = f32(frexp(value).exp + frexp(weight).exp);`
    }),
    createSegmentSumNode(graph, {
      id: `${id}-income-exponent-maximum`,
      operation: 'GPUInequality',
      input: terms,
      segmentOffsets: offsets,
      segmentCount: zoneCount,
      output: incomeExponents,
      reduction: 'max'
    }),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-income-terms`,
      operation: 'GPUInequality',
      variant: 'gini-scaled-income-terms',
      invocationCount: rowCount,
      parameterValues: [zoneCount],
      bindings: [
        read('rows', sortedRows, 'u32'),
        read('zones', sortedZones, 'u32'),
        read('values', values, 'f32'),
        ...(weights ? [read('weights', weights, 'f32')] : []),
        read('exponents', incomeExponents, 'f32'),
        write('terms', terms, 'f32')
      ],
      body: `terms[termsOffset + index] = 0.0;
  let zone = zones[zonesOffset + index];
  if (zone >= getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u)) { return; }
  let row = rows[rowsOffset + index]; let value = values[valuesOffset + row];
  if (!(value > 0.0)) { return; }
  let valueParts = frexp(value);
  let weightParts = frexp(${weights ? 'weights[weightsOffset + row]' : '1.0'});
  terms[termsOffset + index] = ldexp(valueParts.fract * weightParts.fract,
    valueParts.exp + weightParts.exp - i32(exponents[exponentsOffset + zone]));`
    }),
    createSegmentSumNode(graph, {
      id: `${id}-income-sum`,
      operation: 'GPUInequality',
      input: terms,
      segmentOffsets: offsets,
      segmentCount: zoneCount,
      output: sums
    }),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-totals`,
      operation: 'GPUInequality',
      variant: 'gini-gap-totals',
      invocationCount: zoneCount,
      bindings: [
        read('offsets', offsets, 'u32'),
        read('population', forward.weights, 'f32'),
        read('income', sums, 'f32'),
        read('exponents', incomeExponents, 'f32'),
        write('totals', totals, 'f32')
      ],
      body: `let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
  var populationTotal = 0.0; var incomeTotal = 0.0;
  if (end > begin) { populationTotal = population[populationOffset + end - 1u]; incomeTotal = income[incomeOffset + index]; }
  totals[totalsOffset + 3u * index] = populationTotal;
  totals[totalsOffset + 3u * index + 1u] = incomeTotal;
  totals[totalsOffset + 3u * index + 2u] = exponents[exponentsOffset + index];`
    }),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-terms`,
      operation: 'GPUInequality',
      variant: 'gini-gap-terms',
      invocationCount: rowCount,
      parameterValues: [rowCount, zoneCount],
      bindings: [
        read('rows', sortedRows, 'u32'),
        read('zones', sortedZones, 'u32'),
        read('values', values, 'f32'),
        read('left', forward.weights, 'f32'),
        read('right', reverse.weights, 'f32'),
        read('totals', totals, 'f32'),
        write('terms', terms, 'f32')
      ],
      body: `terms[termsOffset + index] = 0.0;
  let rowCount = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  let zone = zones[zonesOffset + index];
  if (zone >= getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u) || index + 1u >= rowCount) { return; }
  if (zones[zonesOffset + index + 1u] != zone) { return; }
  let value = values[valuesOffset + rows[rowsOffset + index]];
  let nextValue = values[valuesOffset + rows[rowsOffset + index + 1u]];
  let nextParts = frexp(nextValue);
  var previousScaled = 0.0;
  if (value > 0.0) {
    let previousParts = frexp(value);
    previousScaled = ldexp(previousParts.fract, previousParts.exp - nextParts.exp);
  }
  // Subtract bounded mantissas before restoring the exponent: even a subnormal raw gap survives.
  let gap = nextParts.fract - previousScaled;
  let income = totals[totalsOffset + 3u * zone + 1u];
  if (!(gap > 0.0) || !(income > 0.0)) { return; }
  let gapParts = frexp(gap);
  let leftParts = frexp(left[leftOffset + index]);
  let rightParts = frexp(right[rightOffset + rowCount - 2u - index]);
  let populationParts = frexp(totals[totalsOffset + 3u * zone]);
  let incomeParts = frexp(income);
  let quotient = frexp((gapParts.fract * leftParts.fract * rightParts.fract) / (populationParts.fract * incomeParts.fract));
  terms[termsOffset + index] = ldexp(quotient.fract, gapParts.exp + nextParts.exp + leftParts.exp + rightParts.exp - populationParts.exp - incomeParts.exp - i32(totals[totalsOffset + 3u * zone + 2u]) + quotient.exp);`
    }),
    createSegmentSumNode(graph, {
      id: `${id}-sum`,
      operation: 'GPUInequality',
      input: terms,
      segmentOffsets: offsets,
      segmentCount: zoneCount,
      output: sums
    }),
    createGraphChunkGroupKernelNode(graph, {
      id: `${id}-publish`,
      operation: 'GPUInequality',
      variant: 'gini-gap-publish',
      invocationCount: zoneCount,
      parameterValues: [outputStride, outputSlot],
      bindings: [
        read('sums', sums, 'f32'),
        read('totals', totals, 'f32'),
        write('output', output, 'f32')
      ],
      declarations: GET_NAN_WGSL,
      body: `let stride = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  let slot = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u);
  output[outputOffset + index * stride + slot] = select(getNaN(), sums[sumsOffset + index], totals[totalsOffset + 3u * index + 1u] > 0.0);`
    })
  ];
}
