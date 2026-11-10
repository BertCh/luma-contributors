// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPURasterSeasonalBreakProps} from './gpu-raster-seasonal-break';
import {RASTER_FINITE_WGSL, rasterBinding, getRasterLeastSquaresWGSL} from './raster-kernel-utils';
import {getGPURasterDateAxisBinding} from '../raster-time-series/raster-irregular-date-axis';

/** Bounded dummy/harmonic regression, cyclic decomposition and fixed-history monitoring. */
export function createExtendedSeasonalBreakNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPURasterSeasonalBreakProps,
  axisDeclarations = ''
): GPUCommandNode<Parameters>[] {
  const slices = props.sliceCount;
  const history = props.monitoring?.historyLength ?? slices;
  const columns = props.seasonality === 'dummy' ? props.period + 2 : 5;
  const baselineColumns = columns - 1;
  const seasonalBasis =
    props.seasonality === 'dummy'
      ? `return select(0.0, 1.0, slice % ${props.period}u == column - 1u);`
      : `let angle = 6.283185307179586 * ${props.dateAxis ? 'rasterSliceYears(slice)' : 'f32(slice)'} / ${props.period}; return select(sin(angle), cos(angle), column == 2u);`;
  const sliceTime = props.dateAxis ? 'rasterSliceYears(slice)' : `f32(slice) / ${slices}.0`;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation: 'GPURasterSeasonalBreak',
      variant: 'extended-seasonal-decomposition-monitoring',
      invocationCount: props.cellCount,
      workgroupSize: 2,
      bindings: [
        rasterBinding('values', props.values),
        ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
        ...(props.dateAxis ? [getGPURasterDateAxisBinding(props.dateAxis)] : []),
        ...Object.entries(props.output).map(([name, view]) => rasterBinding(name, view, true))
      ],
      declarations: `${RASTER_FINITE_WGSL}${getRasterLeastSquaresWGSL(history, columns, true)}${axisDeclarations}
fn isObserved(cell: u32, slice: u32) -> bool {
  let source = cell * ${slices}u + slice;
  return finite(values[valuesOffset + source]) ${props.validity ? '&& validity[validityOffset + source] != 0u' : ''};
}
fn seasonalBasis(slice: u32, column: u32) -> f32 { ${seasonalBasis} }
struct ModelFit { coefficients: array<f32, ${columns}>, residual: f32, valid: bool, converged: bool, iterations: u32 }
fn fitModel(design: array<f32, ${history * columns}>, response: array<f32, ${history}>, columnCount: u32) -> ModelFit {
  var result: ModelFit;
  // Rank certification is independent of the bounded iterative trajectory.
  let certified = leastSquares(design, response, columnCount);
  if (!certified.valid) { return result; }
  ${
    props.fitMethod === 'iterative'
      ? `
  var residual = response;
  var norms: array<f32, ${columns}>;
  for (var column = 0u; column < columnCount; column++) {
    for (var row = 0u; row < ${history}u; row++) { let value = design[column * ${history}u + row]; norms[column] += value * value; }
  }
  for (var iteration = 0u; iteration < ${props.maximumIterations ?? 128}u; iteration++) {
    var maximumChange = 0.0; var maximumCoefficient = 0.0;
    for (var column = 0u; column < columnCount; column++) {
      var projection = 0.0;
      for (var row = 0u; row < ${history}u; row++) { projection += design[column * ${history}u + row] * residual[row]; }
      let change = projection / norms[column];
      result.coefficients[column] += change;
      if (!finite(result.coefficients[column])) { return result; }
      maximumChange = max(maximumChange, abs(change)); maximumCoefficient = max(maximumCoefficient, abs(result.coefficients[column]));
      for (var row = 0u; row < ${history}u; row++) { residual[row] -= design[column * ${history}u + row] * change; }
    }
    // Recompute each sweep to avoid accumulating incremental residual roundoff.
    for (var row = 0u; row < ${history}u; row++) {
      var prediction = 0.0;
      for (var column = 0u; column < columnCount; column++) { prediction += design[column * ${history}u + row] * result.coefficients[column]; }
      residual[row] = response[row] - prediction;
    }
    result.iterations = iteration + 1u;
    if (maximumChange <= ${props.tolerance ?? 1e-5} * (1.0 + maximumCoefficient)) { result.converged = true; break; }
  }
  for (var row = 0u; row < ${history}u; row++) { result.residual += residual[row] * residual[row]; }
  result.valid = finite(result.residual);
  `
      : 'result.coefficients = certified.coefficients; result.residual = certified.residual; result.valid = true; result.converged = true; result.iterations = 1u;'
  }
  return result;
}`,
      body: `var training: array<u32, ${Math.ceil(history / 32)}>;
var response: array<f32, ${history}>; var design: array<f32, ${history * columns}>;
var rejected = false;
for (var slice = 0u; slice < ${slices}u; slice++) {
  let source = index * ${slices}u + slice; let value = values[valuesOffset + source];
  let observed = isObserved(index, slice);
  if (slice < ${history}u) {
    if (observed) { training[slice / 32u] |= 1u << (slice % 32u); } rejected = rejected || !observed;
    if (observed) { response[slice] = value; }
  }
}
${
  props.missingPolicy === 'interpolate'
    ? `
for (var slice = 0u; slice < ${history}u; slice++) {
  if (isObserved(index, slice)) { continue; }
  var left = slice; var right = slice;
  while (left > 0u && !isObserved(index, left)) { left--; }
  while (right + 1u < ${history}u && !isObserved(index, right)) { right++; }
  if (isObserved(index, left) && isObserved(index, right) && right > left) {
    response[slice] = mix(values[valuesOffset + index * ${slices}u + left], values[valuesOffset + index * ${slices}u + right], ${props.dateAxis ? 'rasterYearsBetween(slice, left) / rasterYearsBetween(right, left)' : 'f32(slice - left) / f32(right - left)'}); training[slice / 32u] |= 1u << (slice % 32u);
  }
}`
    : ''
}
var count = 0u; var energy = 0.0;
for (var slice = 0u; slice < ${history}u; slice++) {
  if ((training[slice / 32u] & (1u << (slice % 32u))) == 0u) { continue; } count++; energy += response[slice] * response[slice];
  design[slice] = 1.0; design[${history}u + slice] = ${sliceTime};
  for (var column = 2u; column < ${baselineColumns}u; column++) { design[column * ${history}u + slice] = seasonalBasis(slice, column); }
}
var chosen: array<f32, ${columns}>; var boundary = 0xffffffffu; var gain = 0.0;
var code = 1u; var iterations = 0u; var converged = true; var selectedResidual = 0.0; var selectedColumns = ${baselineColumns}u;
if (${props.missingPolicy === 'reject' ? 'rejected' : 'false'}) { code = 4u; }
else if (!finite(energy)) { code = 2u; }
else if (count >= ${Math.max(baselineColumns + 1, props.minimumSegmentLength * 2)}u) {
  let baseline = fitModel(design, response, ${baselineColumns}u); code = 2u;
  iterations = baseline.iterations; converged = baseline.converged;
  if (baseline.valid) {
    code = 0u; chosen = baseline.coefficients; selectedResidual = baseline.residual;
    let residualFloor = max(1e-20, energy * 1e-12);
    let baselineScore = f32(count) * log(max(baseline.residual, residualFloor) / f32(count)) + ${baselineColumns}.0 * log(f32(count));
    for (var split = ${props.minimumSegmentLength}u; split <= ${history - props.minimumSegmentLength}u; split++) {
      var before = 0u; var after = 0u;
      for (var slice = 0u; slice < ${history}u; slice++) {
        design[${baselineColumns * history}u + slice] = 0.0;
        if ((training[slice / 32u] & (1u << (slice % 32u))) != 0u) { if (slice < split) { before++; } else { after++; design[${baselineColumns * history}u + slice] = 1.0; } }
      }
      if (before < ${props.minimumSegmentLength}u || after < ${props.minimumSegmentLength}u) { continue; }
      let candidate = fitModel(design, response, ${columns}u);
      iterations = max(iterations, candidate.iterations);
      if (!candidate.valid) { continue; }
      converged = converged && candidate.converged;
      let candidateGain = baselineScore - (f32(count) * log(max(candidate.residual, residualFloor) / f32(count)) + ${baselineColumns + 2}.0 * log(f32(count)));
      if (candidateGain > max(gain, ${props.minimumBICGain ?? 0})) {
        chosen = candidate.coefficients; gain = candidateGain; boundary = split; selectedResidual = candidate.residual; selectedColumns = ${columns}u;
      }
    }
    if (!converged) { code = 8u; }
  }
}
let fitted = code == 0u || code == 8u;
var residualScale = missing(); var alarm = -1.0; var maximumScore = 0.0; var consecutive = 0u; var previousSign = 0.0;
if (fitted) { residualScale = sqrt(max(selectedResidual, max(1e-20, energy * 1e-12)) / f32(max(1u, count - selectedColumns))); }
for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + index * ${columns}u + column] = select(missing(), chosen[column], fitted); }
for (var slice = 0u; slice < ${slices}u; slice++) {
  let destination = (index * ${slices}u + slice) * 3u;
  var trend = missing(); var seasonal = missing(); var error = missing();
  if (fitted) {
    trend = chosen[0] + chosen[1] * ${sliceTime} + select(0.0, chosen[${baselineColumns}], slice >= boundary);
    seasonal = 0.0; for (var column = 2u; column < ${baselineColumns}u; column++) { seasonal += chosen[column] * seasonalBasis(slice, column); }
    if (isObserved(index, slice)) { error = values[valuesOffset + index * ${slices}u + slice] - trend - seasonal; }
    ${
      props.monitoring
        ? `
    if (slice >= ${history}u) {
      if (!isObserved(index, slice)) { consecutive = 0u; previousSign = 0.0; }
      else {
        let score = error / residualScale;
        if (!finite(score)) { code = 2u; }
        else {
          maximumScore = max(maximumScore, abs(score));
          if (abs(score) >= ${props.monitoring.threshold}) {
            let direction = sign(score); consecutive = select(1u, consecutive + 1u, direction == previousSign); previousSign = direction;
            if (alarm < 0.0 && consecutive >= ${props.monitoring.minimumConsecutive}u) { alarm = f32(slice + 1u - ${props.monitoring.minimumConsecutive}u); }
          } else { consecutive = 0u; previousSign = 0.0; }
        }
      }
    }`
        : ''
    }
  }
  components[componentsOffset + destination] = trend; components[componentsOffset + destination + 1u] = seasonal; components[componentsOffset + destination + 2u] = error;
}
breakIndex[breakIndexOffset + index] = boundary; improvement[improvementOffset + index] = select(missing(), gain, fitted); status[statusOffset + index] = code;
${
  props.output.diagnostics
    ? `
let diagnosticOffset = diagnosticsOffset + index * 6u;
diagnostics[diagnosticOffset] = f32(iterations); diagnostics[diagnosticOffset + 1u] = select(0.0, 1.0, fitted && converged && code != 2u); diagnostics[diagnosticOffset + 2u] = f32(count);
diagnostics[diagnosticOffset + 3u] = residualScale; diagnostics[diagnosticOffset + 4u] = alarm; diagnostics[diagnosticOffset + 5u] = maximumScore;`
    : ''
}`
    })
  ];
}
