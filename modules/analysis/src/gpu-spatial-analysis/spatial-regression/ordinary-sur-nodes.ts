// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {getCholeskyWGSL, getPivotLimitedCholeskyWGSL} from './spatial-regression-solve';
import type {GPUSeeminglyUnrelatedRegression} from './gpu-seemingly-unrelated-regression';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {getRegressionTileLayout} from './regression-row-tiles';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** One source of truth for every scratch offset and the allocation-free preflight. */
function getOrdinarySURLayout(equations: number, columns: number, rows: number) {
  const size = equations * (columns + 1);
  const {tileRowCount: tileRows, tileCount} = getRegressionTileLayout(rows);
  const meansStart = 1;
  const responsesStart = meansStart + equations * columns;
  const initialStart = responsesStart + equations;
  const precisionStart = initialStart + equations * columns;
  const errorScaleStart = precisionStart + equations * equations;
  const centeredSolutionStart = errorScaleStart + 1;
  const runtimeZeroSlot = centeredSolutionStart + size;
  const partialStart = runtimeZeroSlot + 1;
  const partialStride = size * (size + 1);
  return {
    tileRows,
    tileCount,
    meansStart,
    responsesStart,
    initialStart,
    precisionStart,
    errorScaleStart,
    centeredSolutionStart,
    runtimeZeroSlot,
    partialStart,
    partialStride,
    length: partialStart + tileCount * partialStride
  };
}

/** Exact scratch size shared with the contributor's allocation-free binding preflight. */
export function getOrdinarySURScratchLength(
  equationCount: number,
  predictorCount: number,
  rowCount: number
): number {
  return getOrdinarySURLayout(equationCount, predictorCount, rowCount).length;
}

/** Bounded row-tile statistics followed by the existing small scaled FGLS solves. */
export function getOrdinarySURNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUSeeminglyUnrelatedRegression
): GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const equations = props.equationCount;
  const columns = props.predictorCount;
  const rows = props.rowCount;
  const width = columns + 1;
  const size = equations * width;
  const {
    tileRows,
    tileCount,
    meansStart,
    responsesStart,
    initialStart,
    precisionStart,
    errorScaleStart,
    centeredSolutionStart,
    runtimeZeroSlot,
    partialStart,
    partialStride,
    length
  } = getOrdinarySURLayout(equations, columns, rows);
  // The contributor preflights callers and this entire layout before entering the builder.
  // Each completed stage reuses the partial region; dependencies serialize those lifetimes.
  const scratch = createTransientView(graph, `${id}-statistics`, 'float32', length);
  const scratchBinding = {
    name: 'scratch',
    view: scratch,
    type: 'f32',
    access: 'read_write'
  } as const;
  const statusBinding = {
    name: 'status',
    view: props.output.status,
    type: 'u32',
    access: 'read_write'
  } as const;
  const coefficientsBinding = {
    name: 'coefficients',
    view: props.output.coefficients,
    type: 'f32',
    access: 'read_write'
  } as const;
  const covarianceBinding = {
    name: 'covariance',
    view: props.output.covariance,
    type: 'f32',
    access: 'read_write'
  } as const;
  const residualCovarianceBinding = {
    name: 'residualCovariance',
    view: props.output.residualCovariance,
    type: 'f32',
    access: 'read_write'
  } as const;
  const inputs = [
    {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
    {name: 'responses', view: props.responses, type: 'f32', access: 'read'},
    ...(props.mask ? [{name: 'mask', view: props.mask, type: 'u32', access: 'read'} as const] : []),
    ...(props.initialResiduals
      ? [
          {
            name: 'initialResiduals',
            view: props.initialResiduals,
            type: 'f32',
            access: 'read'
          } as const
        ]
      : [])
  ] as const;
  const constants = `${GET_NAN_WGSL}
${IS_FINITE_BITS_WGSL}
${ROUNDED_DIFFERENCE_WGSL}
const MEANS_START: u32 = ${meansStart}u;
const RESPONSES_START: u32 = ${responsesStart}u;
const INITIAL_START: u32 = ${initialStart}u;
const PRECISION_START: u32 = ${precisionStart}u;
const ERROR_SCALE_START: u32 = ${errorScaleStart}u;
const CENTERED_SOLUTION_START: u32 = ${centeredSolutionStart}u;
const RUNTIME_ZERO_SLOT: u32 = ${runtimeZeroSlot}u;
const PARTIAL_START: u32 = ${partialStart}u;
const PARTIAL_STRIDE: u32 = ${partialStride}u;
// Fixed pairwise sum over the tiles (a float compensation is folded away by the shader compiler).
fn sumPartials(slot: u32) -> f32 {
  ${getPairwiseSumWGSL({
    count: `${tileCount}u`,
    name: 'total',
    term: 'scratch[scratchOffset + PARTIAL_START + totalItem * PARTIAL_STRIDE + slot]'
  })}
  return total;
}`;
  const inputDeclarations = `${constants}
fn included(row: u32) -> bool {
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  for (var equation = 0u; equation < ${equations}u; equation++) {
    ${props.initialResiduals ? `if (!isFiniteBits(initialResiduals[initialResidualsOffset + equation * ${rows}u + row])) { return false; }` : ''}
    if (!isFiniteBits(responses[responsesOffset + equation * ${rows}u + row])) { return false; }
    for (var column = 0u; column < ${columns}u; column++) {
      if (!isFiniteBits(predictors[predictorsOffset + (equation * ${rows}u + row) * ${columns}u + column])) { return false; }
    }
  }
  return true;
}
fn predictor(row: u32, equation: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return getRoundedDifference(
    predictors[predictorsOffset + (equation * ${rows}u + row) * ${columns}u + column - 1u],
    scratch[scratchOffset + MEANS_START + equation * ${columns}u + column - 1u],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
}
fn residual(row: u32, equation: u32) -> f32 {
  ${
    props.initialResiduals
      ? `return initialResiduals[initialResidualsOffset + equation * ${rows}u + row];`
      : `
  var value = getRoundedDifference(responses[responsesOffset + equation * ${rows}u + row],
    scratch[scratchOffset + RESPONSES_START + equation],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
  for (var column = 0u; column < ${columns}u; column++) {
    value -= predictor(row, equation, column + 1u) * scratch[scratchOffset + INITIAL_START + equation * ${columns}u + column];
  }
  return value;`
  }
}`;
  const solveDeclarations = `${constants}\n${IS_FINITE_VALUE_WGSL}\n${[...new Set([columns, equations, size])].map(matrixSize => `${getCholeskyWGSL(matrixSize)}\n${getPivotLimitedCholeskyWGSL(matrixSize)}`).join('\n')}`;
  const makeNode = (
    variant: string,
    invocationCount: number,
    bindings: ReadonlyArray<WGSLKernelBinding>,
    declarations: string,
    body: string
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${variant}`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant,
      invocationCount,
      workgroupSize: 64,
      bindings,
      declarations,
      body
    });
  return [
    makeNode(
      'tile-means',
      tileCount,
      [...inputs, scratchBinding],
      inputDeclarations,
      `
var count = 0u;
var means: array<f32, ${size}>;
for (var row = index * ${tileRows}u; row < min((index + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (!included(row)) { continue; }
  count++;
  for (var equation = 0u; equation < ${equations}u; equation++) {
    let responseSlot = ${equations * columns}u + equation;
    means[responseSlot] += (responses[responsesOffset + equation * ${rows}u + row] - means[responseSlot]) / f32(count);
    for (var column = 0u; column < ${columns}u; column++) {
      let slot = equation * ${columns}u + column;
      means[slot] += (predictors[predictorsOffset + (equation * ${rows}u + row) * ${columns}u + column] - means[slot]) / f32(count);
    }
  }
}
let start = scratchOffset + PARTIAL_START + index * PARTIAL_STRIDE;
scratch[start] = f32(count);
for (var slot = 0u; slot < ${size}u; slot++) { scratch[start + 1u + slot] = means[slot]; }
`
    ),
    makeNode(
      'merge-means',
      1,
      [
        scratchBinding,
        statusBinding,
        coefficientsBinding,
        covarianceBinding,
        residualCovarianceBinding
      ],
      constants,
      `
status[statusOffset] = 1u;
scratch[scratchOffset + RUNTIME_ZERO_SLOT] = 0.0;
for (var slot = 0u; slot < ${size}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${size * size}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${equations * equations}u; slot++) { residualCovariance[residualCovarianceOffset + slot] = getNaN(); }
var count = 0u;
var means: array<f32, ${size}>;
for (var tile = 0u; tile < ${tileCount}u; tile++) {
  let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE;
  let tileCount = u32(scratch[start]);
  if (tileCount == 0u) { continue; }
  count += tileCount;
  for (var slot = 0u; slot < ${size}u; slot++) {
    means[slot] += (scratch[start + 1u + slot] - means[slot]) * (f32(tileCount) / f32(count));
  }
}
scratch[scratchOffset] = f32(count);
for (var slot = 0u; slot < ${size}u; slot++) { scratch[scratchOffset + MEANS_START + slot] = means[slot]; }
if (count > ${Math.max(width, equations)}u) { status[statusOffset] = 0u; }
`
    ),
    makeNode(
      'tile-ols',
      tileCount * equations,
      [...inputs, scratchBinding, statusBinding],
      inputDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
let tile = index / ${equations}u;
let equation = index % ${equations}u;
var normal: array<f32, ${columns * columns}>;
var right: array<f32, ${columns}>;
for (var row = tile * ${tileRows}u; row < min((tile + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (!included(row)) { continue; }
  let response = responses[responsesOffset + equation * ${rows}u + row] - scratch[scratchOffset + RESPONSES_START + equation];
  for (var first = 0u; first < ${columns}u; first++) {
    let value = predictor(row, equation, first + 1u);
    right[first] += value * response;
    for (var second = 0u; second < ${columns}u; second++) { normal[first * ${columns}u + second] += value * predictor(row, equation, second + 1u); }
  }
}
let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + equation * ${columns * (columns + 1)}u;
for (var slot = 0u; slot < ${columns * columns}u; slot++) { scratch[start + slot] = normal[slot]; }
for (var slot = 0u; slot < ${columns}u; slot++) { scratch[start + ${columns * columns}u + slot] = right[slot]; }
`
    ),
    makeNode(
      'solve-ols',
      1,
      [scratchBinding, statusBinding],
      solveDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
for (var equation = 0u; equation < ${equations}u; equation++) {
var normal: array<f32, ${columns * columns}>;
var right: array<f32, ${columns}>;
var scales: array<f32, ${columns}>;
let start = equation * ${columns * (columns + 1)}u;
for (var slot = 0u; slot < ${columns * columns}u; slot++) { normal[slot] = sumPartials(start + slot); }
for (var slot = 0u; slot < ${columns}u; slot++) { right[slot] = sumPartials(start + ${columns * columns}u + slot); }
for(var column=0u;column<${columns}u;column++){let diagonal=normal[column*${columns + 1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${columns}u;first++){for(var second=0u;second<${columns}u;second++){normal[first*${columns}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${columns}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${columns}(&normal,&right);for(var column=0u;column<${columns}u;column++){scratch[scratchOffset+INITIAL_START+equation*${columns}u+column]=right[column]*scales[column];}}

`
    ),
    makeNode(
      'tile-error-covariance',
      tileCount * equations * equations,
      [...inputs, scratchBinding, statusBinding],
      inputDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
let tile = index / ${equations * equations}u;
let slot = index % ${equations * equations}u;
var value = 0.0;
// Normalize during accumulation: the unnormalized sum can overflow for a finite covariance.
for (var row = tile * ${tileRows}u; row < min((tile + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (included(row)) { value += residual(row, slot / ${equations}u) * residual(row, slot % ${equations}u) / scratch[scratchOffset]; }
}
scratch[scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + slot] = value;
`
    ),
    makeNode(
      'solve-error-covariance',
      1,
      [scratchBinding, statusBinding, residualCovarianceBinding],
      solveDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
var sigma: array<f32, ${equations * equations}>;
for (var slot = 0u; slot < ${equations * equations}u; slot++) { sigma[slot] = sumPartials(slot); }
for(var slot=0u;slot<${equations * equations}u;slot++){residualCovariance[residualCovarianceOffset+slot]=sigma[slot];}
// A common covariance scale cancels from coefficients and is restored to their covariance.
// Keep the precision and GLS products away from subnormal values for large error magnitudes.
var errorScale = 0.0;
for (var equation = 0u; equation < ${equations}u; equation++) { errorScale = max(errorScale, sigma[equation * ${equations + 1}u]); }
if (!(errorScale > 0.0) || !isFiniteBits(errorScale)) { status[statusOffset] = 3u; return; }
scratch[scratchOffset + ERROR_SCALE_START] = errorScale;
for (var slot = 0u; slot < ${equations * equations}u; slot++) { sigma[slot] /= errorScale; }
if(!choleskyFactorLimited_${equations}(&sigma)){status[statusOffset]=3u;return;}

for(var column=0u;column<${equations}u;column++){var unit:array<f32,${equations}>;unit[column]=1.0;choleskySolve_${equations}(&sigma,&unit);for(var row=0u;row<${equations}u;row++){scratch[scratchOffset+PRECISION_START+row*${equations}u+column]=unit[row];}}

`
    ),
    makeNode(
      'tile-gls',
      tileCount * size,
      [...inputs, scratchBinding, statusBinding],
      inputDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
let tile = index / ${size}u;
let first = index % ${size}u;
let equation = first / ${width}u;
var normal: array<f32, ${size}>;
var right = 0.0;
for (var row = tile * ${tileRows}u; row < min((tile + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (!included(row)) { continue; }
  let value = predictor(row, equation, first % ${width}u);
  for (var other = 0u; other < ${equations}u; other++) {
    right += scratch[scratchOffset + PRECISION_START + equation * ${equations}u + other] * value
      * (responses[responsesOffset + other * ${rows}u + row] - scratch[scratchOffset + RESPONSES_START + other]);
  }
  for (var second = 0u; second < ${size}u; second++) {
    normal[second] += scratch[scratchOffset + PRECISION_START + equation * ${equations}u + second / ${width}u]
      * value * predictor(row, second / ${width}u, second % ${width}u);
  }
}
let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + first * ${size + 1}u;
for (var second = 0u; second < ${size}u; second++) { scratch[start + second] = normal[second]; }
scratch[start + ${size}u] = right;
`
    ),
    makeNode(
      'solve-gls',
      1,
      [scratchBinding, statusBinding, coefficientsBinding, covarianceBinding],
      solveDeclarations,
      `
if (status[statusOffset] != 0u) { return; }
var normal: array<f32, ${size * size}>;
var right: array<f32, ${size}>;
for (var first = 0u; first < ${size}u; first++) {
  right[first] = sumPartials(first * ${size + 1}u + ${size}u);
  for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] = sumPartials(first * ${size + 1}u + second); }
}
var scales:array<f32,${size}>;for(var column=0u;column<${size}u;column++){let diagonal=normal[column*${size + 1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${size}u;first++){for(var second=0u;second<${size}u;second++){normal[first*${size}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${size}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${size}(&normal,&right);
for (var slot = 0u; slot < ${size}u; slot++) {
  scratch[scratchOffset + CENTERED_SOLUTION_START + slot] = right[slot] * scales[slot];
}
for(var equation=0u;equation<${equations}u;equation++){var intercept=scratch[scratchOffset+RESPONSES_START+equation]+right[equation*${width}u]*scales[equation*${width}u];for(var column=0u;column<${columns}u;column++){let slot=equation*${width}u+column+1u;let coefficient=right[slot]*scales[slot];coefficients[coefficientsOffset+slot]=coefficient;intercept-=scratch[scratchOffset+MEANS_START+equation*${columns}u+column]*coefficient;}coefficients[coefficientsOffset+equation*${width}u]=intercept;}
for(var targetColumn=0u;targetColumn<${size}u;targetColumn++){var transform:array<f32,${size}>;transform[targetColumn]=1.0;if(targetColumn%${width}u==0u){for(var column=0u;column<${columns}u;column++){transform[targetColumn+column+1u]=-scratch[scratchOffset+MEANS_START+(targetColumn/${width}u)*${columns}u+column];}}for(var column=0u;column<${size}u;column++){transform[column]*=scales[column];}choleskySolve_${size}(&normal,&transform);for(var column=0u;column<${size}u;column++){transform[column]*=scales[column];}
for(var destination=0u;destination<${size}u;destination++){var value=transform[destination];if(destination%${width}u==0u){for(var column=0u;column<${columns}u;column++){value-=scratch[scratchOffset+MEANS_START+(destination/${width}u)*${columns}u+column]*transform[destination+column+1u];}}covariance[covarianceOffset+destination*${size}u+targetColumn]=value*scratch[scratchOffset+ERROR_SCALE_START];}}

`
    ),
    makeNode(
      'residuals',
      rows * equations,
      [
        ...inputs,
        scratchBinding,
        statusBinding,
        {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
      ],
      inputDeclarations,
      `
let equation = index / ${rows}u;
let row = index % ${rows}u;
var value = getNaN();
if (status[statusOffset] == 0u && included(row)) {
  // The raw intercept can lose the small centered intercept when predictor means are large.
  // Predict from the centered fit, rounding the first difference before subtracting small terms.
  value = getRoundedDifference(responses[responsesOffset + index],
    scratch[scratchOffset + RESPONSES_START + equation],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
  value -= scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${width}u];
  for (var column = 0u; column < ${columns}u; column++) {
    value -= predictor(row, equation, column + 1u)
      * scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${width}u + column + 1u];
  }
}
residuals[residualsOffset + index] = value;
`
    )
  ];
}
