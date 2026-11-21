// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getDenseLUWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** Reevaluate cancellation-sized Moran variances as a squared norm; ordinary reports are untouched.
 * B=S+cI-c11', S=(W+W')/2, c an observed symmetric link. With M1=0,
 * Var(I)=2(n/S0)^2 ||MBM-tr(MB)M/d||_F^2/[d(d+2)].
 * Work is O(n² p) only when the original trace expression is cancellation-sized, O(np²) otherwise.
 * No n² buffer: one observation per invocation streams the other observations.
 * @internal
 */
export function getDiagnosticsMoranPrecisionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    width: number;
    /** Observation/design constants only; tile overrides belong to the reduction helper. */
    rowConstants: string;
    rowScaleWGSL: string;
    weightBindings: WGSLKernelBinding[];
    rowScaleBindings: WGSLKernelBinding[];
    table: GraphDataView;
    moments: GraphDataView;
    origin: GraphDataView;
    summary: GraphDataView;
    tests: GraphDataView;
  }
): GPUCommandNode<Parameters>[] {
  const {id, rowCount, width: columns} = props;
  const layout = getRegressionTileLayout(rowCount);
  const reducedCount = columns * columns + 2;
  validateRegressionAllocations(graph.device, id, {
    'Moran precision rows': rowCount * (columns + 2),
    'Moran precision partials': layout.tileCount * reducedCount,
    'Moran precision parameters': 2 * columns * columns + 5
  });
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-moran-${name}`, 'float32', length);
  const rows = transient('rows', rowCount * (columns + 2));
  const reduced = transient('reduced', reducedCount);
  const parameters = transient('parameters', 2 * columns * columns + 5);
  const norms = transient('norms', rowCount);
  const norm = transient('norm', 1);
  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, type: 'f32', access});
  const table = binding('table', props.table);
  const summary = binding('summary', props.summary);
  const origin = binding('origin', props.origin);
  const constants = `${props.rowConstants}\nconst PP: u32 = ${columns * columns}u;`;
  const access = `${constants}
${props.rowScaleWGSL}
${ROUNDED_DIFFERENCE_WGSL}
fn fitted(row: u32) -> bool { return table[tableOffset + row * TABLE_WIDTH] > 0.5; }
fn weightAt(row: u32, other: u32) -> f32 {
  var low = offsets[offsetsOffset + row]; var high = offsets[offsetsOffset + row + 1u];
  while (low < high) { let middle = (low + high) / 2u; let neighborRow = neighbors[neighborsOffset + middle];
    if (neighborRow == other) { return weights[weightsOffset + middle] * getRowScale(row); }
    if (neighborRow < other) { low = middle + 1u; } else { high = middle; }
  }
  return 0.0;
}
fn centeredWeight(row: u32, other: u32) -> f32 {
  if (row == other) { return 0.0; }
  return getRoundedDifference(0.5 * weightAt(row, other) + 0.5 * weightAt(other, row), origin[originOffset + 2u], bitcast<u32>(origin[originOffset]));
}`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-moran-centered-products`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-products',
      invocationCount: rowCount,
      bindings: [
        ...props.weightBindings,
        table,
        ...props.rowScaleBindings,
        origin,
        summary,
        binding('rowsOut', rows, 'read_write')
      ],
      declarations: access,
      body: `let base = rowsOutOffset + index * (P + 2u);
  for (var column = 0u; column < P + 2u; column++) { rowsOut[base + column] = 0.0; }
  if (summary[summaryOffset + 7u] != 0.0 || !fitted(index)) { return; }
  var products: array<f32, ${columns}>; var residualLag = 0.0;
  for (var other = 0u; other < ROWS; other++) {
    if (!fitted(other)) { continue; }
    let weight = centeredWeight(index, other); let source = tableOffset + other * TABLE_WIDTH;
    for (var column = 0u; column < P; column++) { products[column] += weight * table[source + column]; }
    residualLag += weight * table[source + P + 1u];
  }
  for (var column = 0u; column < P; column++) { rowsOut[base + column] = products[column]; }
  let residual = table[tableOffset + index * TABLE_WIDTH + P + 1u];
  rowsOut[base + P] = residual * residualLag; rowsOut[base + P + 1u] = residual;`
    }),
    ...getRowTileSumNodes(graph, {
      id: `${id}-moran-centered-moments`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-moments',
      rowCount,
      layout,
      sumCount: reducedCount,
      bindings: [table, binding('rows', rows), summary],
      partial: transient('moment-partials', layout.tileCount * reducedCount),
      output: reduced,
      declarations: `${constants}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (summary[summaryOffset + 7u] != 0.0) { return; }
  let source = rowsOffset + row * (P + 2u);
  for (var first = 0u; first < P; first++) { for (var second = 0u; second < P; second++) {
    (*terms)[first * P + second] += table[tableOffset + row * TABLE_WIDTH + first] * rows[source + second];
  } }
  (*terms)[PP] += rows[source + P]; (*terms)[PP + 1u] += rows[source + P + 1u];
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-moran-centered-projection`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-projection',
      // contract-allow: single-invocation -- dense solve bounded to 16 columns; matrices total <4 KiB; row reductions are separate.
      invocationCount: 1,
      bindings: [
        binding('moments', props.moments),
        binding('reduced', reduced),
        summary,
        origin,
        binding('parametersOut', parameters, 'read_write')
      ],
      declarations: `${constants}\n${IS_FINITE_VALUE_WGSL}\n${getDenseLUWGSL(columns)}`,
      body: `parametersOut[parametersOutOffset + 2u * PP + 1u] = 0.0;
  if (summary[summaryOffset + 7u] != 0.0) { return; }
  var factor: array<f32, ${columns * columns}>;
  for (var slot = 0u; slot < PP; slot++) { factor[slot] = moments[momentsOffset + slot]; }
  var decomposition = factorDenseLU_${columns}(&factor, P, 0u, 0.0, 0.0);
  if (decomposition.status != 0u) { return; }
  var inverse: array<f32, ${columns * columns}>;
  for (var column = 0u; column < P; column++) {
    var right: array<f32, ${columns}>; right[column] = 1.0;
    if (!solveDenseLU_${columns}(&factor, &decomposition, &right)) { return; }
    for (var row = 0u; row < P; row++) { inverse[row * P + column] = right[row]; }
  }
  var trace = 0.0;
  for (var row = 0u; row < P; row++) { for (var column = 0u; column < P; column++) {
    parametersOut[parametersOutOffset + row * P + column] = inverse[row * P + column];
    trace -= inverse[row * P + column] * reduced[reducedOffset + column * P + row];
    var value = 0.0;
    for (var first = 0u; first < P; first++) { for (var second = 0u; second < P; second++) {
      value += inverse[row * P + first] * reduced[reducedOffset + first * P + second] * inverse[second * P + column];
    } }
    parametersOut[parametersOutOffset + PP + row * P + column] = value;
  } }
  parametersOut[parametersOutOffset + 2u * PP] = trace;
  parametersOut[parametersOutOffset + 2u * PP + 1u] = 1.0;
  parametersOut[parametersOutOffset + 2u * PP + 2u] = summary[summaryOffset];
  parametersOut[parametersOutOffset + 2u * PP + 3u] = origin[originOffset + 2u];
  parametersOut[parametersOutOffset + 2u * PP + 4u] = origin[originOffset];`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-moran-centered-norms`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-norms',
      invocationCount: rowCount,
      bindings: [
        ...props.weightBindings,
        table,
        ...props.rowScaleBindings,
        binding('moranModel', parameters),
        binding('rows', rows),
        binding('normsOut', norms, 'read_write')
      ],
      declarations: access
        .replaceAll('origin[originOffset + 2u]', 'moranModel[moranModelOffset + 2u * PP + 3u]')
        .replaceAll('origin[originOffset]', 'moranModel[moranModelOffset + 2u * PP + 4u]'),
      body: `normsOut[normsOutOffset + index] = 0.0;
  if (moranModel[moranModelOffset + 2u * PP + 1u] == 0.0 || !fitted(index)) { return; }
  let meanTrace = moranModel[moranModelOffset + 2u * PP] / (moranModel[moranModelOffset + 2u * PP + 2u] - f32(P));
  var inverseFirst: array<f32, ${columns}>; var adjustedFirst: array<f32, ${columns}>;
  for (var column = 0u; column < P; column++) { for (var inner = 0u; inner < P; inner++) {
    let inverse = moranModel[moranModelOffset + inner * P + column];
    inverseFirst[column] += table[tableOffset + index * TABLE_WIDTH + inner] * inverse;
    adjustedFirst[column] += table[tableOffset + index * TABLE_WIDTH + inner] * moranModel[moranModelOffset + PP + inner * P + column] - rows[rowsOffset + index * (P + 2u) + inner] * inverse;
  } adjustedFirst[column] += meanTrace * inverseFirst[column]; }
  var norm = 0.0;
  for (var other = 0u; other < ROWS; other++) {
    if (!fitted(other)) { continue; }
    var value = centeredWeight(index, other) - select(0.0, meanTrace, index == other);
    for (var column = 0u; column < P; column++) {
      value += adjustedFirst[column] * table[tableOffset + other * TABLE_WIDTH + column] - inverseFirst[column] * rows[rowsOffset + other * (P + 2u) + column];
    }
    norm += value * value;
  }
  normsOut[normsOutOffset + index] = norm;`
    }),
    ...getRowTileSumNodes(graph, {
      id: `${id}-moran-centered-norm`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-norm',
      rowCount,
      layout,
      sumCount: 1,
      bindings: [binding('norms', norms)],
      partial: transient('norm-partials', layout.tileCount),
      output: norm,
      declarations:
        'fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) { (*terms)[0] += norms[normsOffset + row]; }'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-moran-centered-report`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'moran-centered-report',
      invocationCount: 1,
      bindings: [
        binding('summary', props.summary, 'read_write'),
        binding('tests', props.tests, 'read_write'),
        binding('moranModel', parameters),
        binding('reduced', reduced),
        binding('norm', norm),
        origin
      ],
      declarations: `${constants}\n${GET_NAN_WGSL}\n${ERFC_VALUE_WGSL}`,
      body: `if (summary[summaryOffset + 7u] != 0.0) { return; }
  let n = summary[summaryOffset]; let degrees = n - f32(P); let scale = n / summary[summaryOffset + 4u];
  let variance = select(getNaN(), 2.0 * scale * scale * norm[normOffset] / (degrees * (degrees + 2.0)), moranModel[moranModelOffset + 2u * PP + 1u] > 0.5);
  var z = getNaN(); var probability = getNaN();
  if (variance > 0.0) {
    let residualTotal = reduced[reducedOffset + PP + 1u];
    let quadratic = reduced[reducedOffset + PP] + origin[originOffset + 2u] * residualTotal * residualTotal;
    let numerator = scale * (quadratic / (summary[summaryOffset + 1u] * n) - moranModel[moranModelOffset + 2u * PP] / degrees);
    z = numerator / sqrt(variance); probability = erfcValue(abs(z) * 0.70710678);
  }
  summary[summaryOffset + 7u] = variance; summary[summaryOffset + 8u] = z; summary[summaryOffset + 9u] = probability;
  tests[testsOffset + 15u] = z; tests[testsOffset + 17u] = probability;`
    })
  ];
}
