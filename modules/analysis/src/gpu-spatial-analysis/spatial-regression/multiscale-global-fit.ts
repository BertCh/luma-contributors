// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT} from './gpu-generalized-geographically-weighted-regression';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';

/** Tiled centered starting fit of MGWR; unsuccessful systems leave zero coefficient surfaces. @internal */
export function createMultiscaleGlobalFit<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    predictorCount: number;
    bindings: readonly WGSLKernelBinding[];
    /** Defines included(row) and predictor(row, column), with response in column zero below. */
    declarations: string;
    output: GraphDataView<'float32'>;
  }
) {
  const {id, predictorCount} = props;
  const columns = predictorCount + 1;
  const termCount = predictorCount * predictorCount + predictorCount;
  const layout = getRegressionTileLayout(props.rowCount);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'globalFitPartial',
    getMultiscaleGlobalFitPartialLength(props.rowCount, predictorCount) * 4,
    'reduce rowCount or predictorCount'
  );
  const partial = createTransientView(
    graph,
    `${id}-global-partial`,
    'float32',
    getMultiscaleGlobalFitPartialLength(props.rowCount, predictorCount)
  );
  const sample = createTransientView(graph, `${id}-global-sample`, 'float32', 3);
  const deviations = createTransientView(graph, `${id}-global-deviations`, 'float32', columns);
  const moments = createTransientView(graph, `${id}-global-moments`, 'float32', termCount);
  const sampleBinding = {name: 'sample', view: sample, type: 'f32', access: 'read'} as const;
  const deviationsBinding = {
    name: 'deviations',
    view: deviations,
    type: 'f32',
    access: 'read'
  } as const;
  // The row-tile helper declares ROW_COUNT as an override instead of the common literal.
  const common = props.declarations.replace(/^const ROW_COUNT:.*$/m, '');
  const values = `
// Bound response magnitudes below 4 before subtracting, including opposite finite extremes.
// Keep both powers normal and ldexp's exponent in [-126, 126], including a zero sample.
fn getGlobalResponseScale() -> f32 {
  let exponent = bitcast<u32>(sample[sampleOffset + 2u]) & 0x7f800000u;
  return bitcast<f32>(clamp(exponent, 0x00800000u, 0x7e800000u));
}
fn getGlobalValue(row: u32, column: u32) -> f32 {
  if (column == 0u) {
    let exponent = i32(bitcast<u32>(getGlobalResponseScale()) >> 23u) - 127;
    return ldexp(response[responseOffset + row], -exponent);
  }
  return predictor(row, column);
}
fn getGlobalReference(column: u32) -> f32 {
  return getGlobalValue(u32(max(sample[sampleOffset + 1u], 1.0)) - 1u, column);
}`;
  const centered = `${values}
fn getGlobalDeviation(column: u32) -> f32 {
  return deviations[deviationsOffset + column];
}
fn getGlobalMean(column: u32) -> f32 {
  return getGlobalReference(column) + getGlobalDeviation(column);
}`;
  const tileProps = {
    operation: 'GPUMultiscaleGeographicallyWeightedRegression',
    rowCount: props.rowCount,
    layout,
    partial
  };
  return [
    // A valid row anchors deviations even with leading/trailing masked or nonfinite rows.
    ...getRowTileSumNodes(graph, {
      ...tileProps,
      id: `${id}-global-sample`,
      variant: 'global-sample',
      sumCount: 1,
      maximumCount: 2,
      bindings: props.bindings,
      output: sample,
      declarations: `${common}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  (*terms)[0] += 1.0;
  (*terms)[1] = max((*terms)[1], f32(row + 1u));
  (*terms)[2] = max((*terms)[2], abs(response[responseOffset + row]));
}`
    }),
    ...getRowTileSumNodes(graph, {
      ...tileProps,
      id: `${id}-global-deviations`,
      variant: 'global-deviations',
      sumCount: columns,
      bindings: [...props.bindings, sampleBinding],
      output: deviations,
      declarations: `${common}
${values}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  for (var column = 0u; column < ${columns}u; column++) {
    // Normalize before either the tile sum or its pairwise merge: a finite mean can have
    // unnormalized deviation partials larger than float32 (for example 64 * -1e37).
    (*terms)[column] += (getGlobalValue(row, column) - getGlobalReference(column)) / sample[sampleOffset];
  }
}`
    }),
    ...getRowTileSumNodes(graph, {
      ...tileProps,
      id: `${id}-global-moments`,
      variant: 'global-moments',
      sumCount: termCount,
      bindings: [...props.bindings, sampleBinding, deviationsBinding],
      output: moments,
      declarations: `${common}
${centered}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  if (!included(row)) { return; }
  let centeredResponse = (getGlobalValue(row, 0u) - getGlobalReference(0u)) - getGlobalDeviation(0u);
  for (var column = 0u; column < ${predictorCount}u; column++) {
    let first = (predictor(row, column + 1u) - getGlobalReference(column + 1u)) - getGlobalDeviation(column + 1u);
    (*terms)[column] += first * centeredResponse;
    for (var other = 0u; other <= column; other++) {
      (*terms)[${predictorCount}u + column * ${predictorCount}u + other] += first * ((predictor(row, other + 1u) - getGlobalReference(other + 1u)) - getGlobalDeviation(other + 1u));
    }
  }
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-global-fit`,
      operation: 'GPUMultiscaleGeographicallyWeightedRegression',
      variant: 'global-fit',
      // contract-allow: single-invocation; only the at-most-24-column dense solve, no row traversal.
      invocationCount: 1,
      bindings: [
        ...props.bindings,
        sampleBinding,
        deviationsBinding,
        {name: 'moments', view: moments, type: 'f32', access: 'read'},
        {name: 'globalFit', view: props.output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${props.declarations}
${IS_FINITE_VALUE_WGSL}
${centered}
${getCholeskyWGSL(predictorCount, GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT + 1)}`,
      body: `for (var column = 0u; column < ${columns}u; column++) { globalFit[globalFitOffset + column] = 0.0; }
if (sample[sampleOffset] < ${columns}.0) { return; }
var gram: array<f32, ${predictorCount * predictorCount}>;
var cross: array<f32, ${predictorCount}>;
for (var column = 0u; column < ${predictorCount}u; column++) {
  cross[column] = moments[momentsOffset + column];
  for (var other = 0u; other <= column; other++) {
    gram[column * ${predictorCount}u + other] = moments[momentsOffset + ${predictorCount}u + column * ${predictorCount}u + other];
  }
}
var equilibration: array<f32, ${predictorCount}>;
for (var column = 0u; column < ${predictorCount}u; column++) {
  let diagonal = gram[column * ${predictorCount + 1}u];
  if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return; }
  equilibration[column] = inverseSqrt(diagonal);
}
for (var column = 0u; column < ${predictorCount}u; column++) {
  cross[column] *= equilibration[column];
  for (var other = 0u; other <= column; other++) { gram[column * ${predictorCount}u + other] *= equilibration[column] * equilibration[other]; }
}
if (!choleskyFactor_${predictorCount}(&gram)) { return; }
choleskySolve_${predictorCount}(&gram, &cross);
var intercept = getGlobalMean(0u);
for (var column = 0u; column < ${predictorCount}u; column++) {
  let scaledSlope = cross[column] * equilibration[column];
  let slope = scaledSlope * getGlobalResponseScale();
  if (!isFiniteBits(slope)) { return; }
  globalFit[globalFitOffset + column + 1u] = slope;
  intercept -= scaledSlope * getGlobalMean(column + 1u);
}
intercept *= getGlobalResponseScale();
if (isFiniteBits(intercept)) { globalFit[globalFitOffset] = intercept; }`
    })
  ];
}

/** Partials for the three sample terms (count, reference, response scale) or the fit moments. @internal */
export function getMultiscaleGlobalFitPartialLength(
  rowCount: number,
  predictorCount: number
): number {
  return (
    getRegressionTileLayout(rowCount).tileCount *
    Math.max(3, predictorCount * predictorCount + predictorCount)
  );
}
