// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64arithmetic} from '@luma.gl/shaders';
import {getSURGLSRefinementWGSL, SUR_GLS_REFINEMENT_COUNT} from './sur-gls-refinement-wgsl';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  createWGSLDispatchGate,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {getCholeskyWGSL, getPivotLimitedCholeskyWGSL} from './spatial-regression-solve';
import {
  GPU_SUR_MAXIMUM_EQUATION_COUNT,
  GPU_SUR_MAXIMUM_PREDICTOR_COUNT,
  GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT,
  type GPUSeeminglyUnrelatedRegression
} from './gpu-seemingly-unrelated-regression';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {validateMaximumCount} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getRegressionTileLayout, REGRESSION_TILE_MERGE_GROUP_SIZE} from './regression-row-tiles';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {LOCAL_REGRESSION_PAIRS_WGSL} from './local-regression-pairs-wgsl';
import {createSURParameterNodes} from './seemingly-unrelated-regression-parameters';

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
  const iterationStateStart = partialStart + tileCount * partialStride;
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
    iterationStateStart,
    length: iterationStateStart + 2
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
  contributor: GPUSeeminglyUnrelatedRegression,
  convergenceCoordinates?: {
    predictors: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const iterationCount = props.output.iteration?.status.iterationCount;
  const equations = props.equationCount;
  const columns = props.predictorCount;
  const rows = props.rowCount;
  validateMaximumCount(id, 'equationCount', equations, GPU_SUR_MAXIMUM_EQUATION_COUNT);
  validateMaximumCount(id, 'predictorCount', columns, GPU_SUR_MAXIMUM_PREDICTOR_COUNT);
  validateMaximumCount(
    id,
    'system coefficients',
    equations * (columns + 1),
    GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT
  );
  validateMaximumCount(
    id,
    'SUR tileCount',
    getRegressionTileLayout(rows).tileCount,
    REGRESSION_TILE_MERGE_GROUP_SIZE
  );
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
    iterationStateStart,
    length
  } = getOrdinarySURLayout(equations, columns, rows);
  // The contributor preflights callers and this entire layout before entering the builder.
  // Each completed stage reuses the partial region; dependencies serialize those lifetimes.
  const scratch = createTransientView(graph, `${id}-statistics`, 'float32', length);
  const parameterPlan =
    props.iteration && !props.parameters
      ? createSURParameterNodes(graph, id, Float32Array.of(props.iteration.tolerance ?? 1e-5))
      : undefined;
  const parameters = props.parameters ?? parameterPlan?.parameters;
  const parameterBindings = parameters
    ? [{name: 'parameters', view: parameters, type: 'f32', access: 'read'} as const]
    : [];
  // Spatial centering is numerical only: stopping still uses the original filtered-X
  // intercept, with the response anchored at its first observation, as in the CPU twin.
  const comparison =
    props.iteration && convergenceCoordinates
      ? {
          previous: createTransientView(graph, `${id}-convergence-previous`, 'float32', size),
          disabled: createSURParameterNodes(graph, `${id}-convergence-disabled`, Float32Array.of(0))
        }
      : undefined;
  const iterationGates = props.iteration
    ? Array.from({length: props.iteration.maximumIterations}, (_, round) =>
        createWGSLDispatchGate<Parameters>(graph, {
          id: `${id}-iteration-dispatch-${round + 1}`,
          operation: 'GPUSeeminglyUnrelatedRegression',
          shapes: [
            [tileCount * equations * equations, 64],
            [1, 64],
            [tileCount * size, 64],
            [1, 64]
          ]
        })
      )
    : [];
  const iterationGate = iterationGates[0];
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
const ITERATION_COUNT: u32 = ${iterationStateStart}u;
const ITERATION_ACTIVE: u32 = ${iterationStateStart + 1}u;
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
    props.iteration
      ? `if (scratch[scratchOffset + ITERATION_COUNT] > 0.0) {
    var value = getRoundedDifference(responses[responsesOffset + equation * ${rows}u + row],
      scratch[scratchOffset + RESPONSES_START + equation],
      bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
    var magnitude = abs(value) + abs(scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${width}u]);
    value -= scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${width}u];
    for (var column = 0u; column < ${columns}u; column++) {
      let term = predictor(row, equation, column + 1u) * scratch[scratchOffset + CENTERED_SOLUTION_START + equation * ${width}u + column + 1u];
      value -= term;
      magnitude += abs(term);
    }
    return select(value, 0.0, abs(value) <= f32(${width}u) * 9.5367431640625e-7 * magnitude);
  }`
      : ''
  }
  ${
    props.initialResiduals
      ? `return initialResiduals[initialResidualsOffset + equation * ${rows}u + row];`
      : `
  var value = getRoundedDifference(responses[responsesOffset + equation * ${rows}u + row],
    scratch[scratchOffset + RESPONSES_START + equation],
    bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
  var magnitude = abs(value);
  for (var column = 0u; column < ${columns}u; column++) {
    let term = predictor(row, equation, column + 1u) * scratch[scratchOffset + INITIAL_START + equation * ${columns}u + column];
    value -= term;
    magnitude += abs(term);
  }
  return select(value, 0.0, abs(value) <= f32(${width}u) * 9.5367431640625e-7 * magnitude);`
  }
}`;
  const solveDeclarations = `${constants}\n${IS_FINITE_VALUE_WGSL}\n${[...new Set([columns, equations, size])].map(matrixSize => `${getCholeskyWGSL(matrixSize)}\n${getPivotLimitedCholeskyWGSL(matrixSize)}`).join('\n')}`;
  const makeNode = (
    variant: string,
    invocationCount: number,
    bindings: ReadonlyArray<WGSLKernelBinding>,
    declarations: string,
    body: string,
    gate?: Pick<WGSLKernelProps<Parameters>, 'condition' | 'extraResources'>
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${variant}`,
      operation: 'GPUSeeminglyUnrelatedRegression',
      variant,
      invocationCount,
      workgroupSize: 64,
      modules: variant.endsWith('solve-gls') ? [fp64arithmetic] : undefined,
      bindings,
      declarations,
      body,
      ...gate
    });
  return [
    ...(parameterPlan?.nodes ?? []),
    ...(comparison?.disabled.nodes ?? []),
    // Initialize future round arguments every frame. Each solve reads its own gate and
    // writes only the next round's separate buffer, so no pass storage-writes its indirect args.
    ...Array.from({length: Math.ceil(iterationGates.length / 8)}, (_, batch) => {
      const gates = iterationGates.slice(batch * 8, batch * 8 + 8);
      return makeNode(
        `initialize-gates-${batch}`,
        1,
        gates.map((gate, slot) => ({...gate.binding, name: `dispatch${slot}`})),
        '',
        gates
          .map(
            (_, slot) =>
              `for (var entry = 0u; entry < 12u; entry++) { dispatch${slot}[dispatch${slot}Offset + entry] = 0u; }`
          )
          .join('\n')
      );
    }),
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
    // contract-allow: single-invocation -- REGRESSION_TILE_MERGE_GROUP_SIZE = 4096 bounds the default-route tile merge; GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT = 16 bounds 256 covariance entries; GPU_SUR_MAXIMUM_EQUATION_COUNT = 4 bounds error covariance.
    makeNode(
      'merge-means',
      1,
      [
        scratchBinding,
        statusBinding,
        coefficientsBinding,
        covarianceBinding,
        residualCovarianceBinding,
        ...(iterationGate ? [iterationGate.binding] : []),
        ...parameterBindings
      ],
      constants,
      `
status[statusOffset] = 1u;
scratch[scratchOffset + ITERATION_COUNT] = 0.0;
scratch[scratchOffset + ITERATION_ACTIVE] = 1.0;
${iterationGate?.disarmWGSL ?? ''}
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
${parameters ? 'if (!isFiniteBits(parameters[parametersOffset]) || parameters[parametersOffset] < 1.1754943508222875e-38) { status[statusOffset] = 5u; }' : ''}
${iterationGate ? `if (status[statusOffset] == 0u) { ${iterationGate.armWGSL} }` : ''}
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
    // contract-allow: single-invocation -- GPU_SUR_MAXIMUM_EQUATION_COUNT = 4 systems, GPU_SUR_MAXIMUM_PREDICTOR_COUNT = 7 slopes, REGRESSION_TILE_MERGE_GROUP_SIZE = 4096 tiles per merge; SUR uses the default tile layout without hierarchical totals.
    makeNode(
      'solve-ols',
      1,
      [scratchBinding, statusBinding],
      `${solveDeclarations}${convergenceCoordinates && columns === 1 ? `\n${LOCAL_REGRESSION_PAIRS_WGSL}` : ''}`,
      `
if (status[statusOffset] != 0u) { return; }
for (var equation = 0u; equation < ${equations}u; equation++) {
var normal: array<f32, ${columns * columns}>;
var right: array<f32, ${columns}>;
var scales: array<f32, ${columns}>;
let start = equation * ${columns * (columns + 1)}u;
for (var slot = 0u; slot < ${columns * columns}u; slot++) { normal[slot] = sumPartials(start + slot); }
for (var slot = 0u; slot < ${columns}u; slot++) { right[slot] = sumPartials(start + ${columns * columns}u + slot); }
${
  convergenceCoordinates && columns === 1
    ? `// Correct the scalar quotient by its remainder. An inaccurate reciprocal can
// move a representable slope by an ulp before GLS restores its intercept.
let variance = normal[0];
if (variance >= 1.1754943508222875e-38 && variance <= 8.507059173023462e37) {
  let slope = divideRegressionPair(vec2f(right[0], 0.0), vec2f(variance, 0.0)).x;
  if (isFiniteBits(slope)) { scratch[scratchOffset + INITIAL_START + equation] = slope; continue; }
}`
    : ''
}
for(var column=0u;column<${columns}u;column++){let diagonal=normal[column*${columns + 1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${columns}u;first++){for(var second=0u;second<${columns}u;second++){normal[first*${columns}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${columns}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${columns}(&normal,&right);for(var column=0u;column<${columns}u;column++){scratch[scratchOffset+INITIAL_START+equation*${columns}u+column]=right[column]*scales[column];}}

`
    ),
    ...Array.from({length: props.iteration?.maximumIterations ?? 1}, (_, round) => {
      const makeRoundNode = (
        variant: string,
        invocationCount: number,
        bindings: ReadonlyArray<WGSLKernelBinding>,
        declarations: string,
        body: string
      ) => {
        const gatedSlot =
          variant === 'tile-error-covariance'
            ? 0
            : variant === 'solve-error-covariance'
              ? 1
              : variant === 'tile-gls'
                ? 2
                : 3;
        const nextGate = variant === 'solve-gls' ? iterationGates[round + 1] : undefined;
        const node = makeNode(
          props.iteration ? `iteration-${round + 1}-${variant}` : variant,
          invocationCount,
          [
            ...bindings.map(binding =>
              comparison && variant === 'solve-gls' && binding.name === 'parameters'
                ? {...binding, view: comparison.disabled.parameters}
                : binding
            ),
            ...(nextGate ? [nextGate.binding] : [])
          ],
          declarations,
          `${props.iteration ? 'if (scratch[scratchOffset + ITERATION_ACTIVE] == 0.0 || status[statusOffset] != 0u) { return; }' : ''}\n${body}`,
          iterationGates[round]?.getGate(gatedSlot)
        );
        if (!comparison || !convergenceCoordinates || variant !== 'solve-gls') return node;
        const coordinateBindings: WGSLKernelBinding[] = [
          coefficientsBinding,
          scratchBinding,
          {name: 'origins', view: convergenceCoordinates.predictors, type: 'f32', access: 'read'},
          {
            name: 'spatialParameters',
            view: convergenceCoordinates.parameters,
            type: 'f32',
            access: 'read'
          },
          {name: 'previous', view: comparison.previous, type: 'f32', access: 'read_write'}
        ];
        const coordinateWGSL = `${constants}
fn comparisonCoefficient(slot: u32) -> f32 {
 var value = coefficients[coefficientsOffset + slot];
 if (slot % ${width}u == 0u) {
  let equation = slot / ${width}u;
  ${Array.from({length: columns}, (_, column) => `value = getRoundedDifference(value, (1.0 - spatialParameters[spatialParametersOffset + equation]) * origins[originsOffset + equation * ${rows * columns}u + ${column}u] * coefficients[coefficientsOffset + slot + ${column + 1}u], bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));`).join('\n')}
 }
 return value;
}`;
        return [
          makeNode(
            `iteration-${round + 1}-capture-convergence`,
            size,
            coordinateBindings,
            coordinateWGSL,
            'previous[previousOffset + index] = comparisonCoefficient(index);',
            iterationGates[round]?.getGate(3)
          ),
          node,
          makeNode(
            `iteration-${round + 1}-finish-convergence`,
            1,
            [
              ...coordinateBindings,
              statusBinding,
              ...parameterBindings,
              ...(nextGate ? [nextGate.binding] : [])
            ],
            coordinateWGSL,
            `if (scratch[scratchOffset + ITERATION_COUNT] <= 1.0 || (status[statusOffset] != 0u && status[statusOffset] != 4u)) { return; }
var largestChange = 0.0;
${Array.from({length: size}, (_, slot) => `largestChange = max(largestChange, abs(comparisonCoefficient(${slot}u) - previous[previousOffset + ${slot}u]) / max(1.0, abs(previous[previousOffset + ${slot}u])));`).join('\n')}
if (largestChange < parameters[parametersOffset]) {
 status[statusOffset] = 0u; scratch[scratchOffset + ITERATION_ACTIVE] = 0.0;
 ${nextGate?.disarmWGSL ?? ''}
}`,
            iterationGates[round]?.getGate(3)
          )
        ];
      };
      return [
        makeRoundNode(
          'tile-error-covariance',
          tileCount * equations * equations,
          [...inputs, scratchBinding, statusBinding],
          inputDeclarations,
          `
if (status[statusOffset] != 0u) { return; }
let tile = index / ${equations * equations}u;
let slot = index % ${equations * equations}u;
var value = 0.0;
${convergenceCoordinates ? 'var unnormalized = 0.0; var rawFinite = true;' : ''}
// Normalize during accumulation: the unnormalized sum can overflow for a finite covariance.
for (var row = tile * ${tileRows}u; row < min((tile + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (included(row)) {
    let first = residual(row, slot / ${equations}u);
    let second = residual(row, slot % ${equations}u);
    let product = first * second;
    ${
      convergenceCoordinates
        ? `// The common divisor comes after the dot product on the spatial path. This
    // preserves cancelling products before dividing by a non-dyadic row count.
    unnormalized += getRoundedDifference(product, 0.0, bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
    rawFinite = rawFinite && isFiniteBits(product) && isFiniteBits(unnormalized);`
        : ''
    }
    if (isFiniteBits(product)) {
      // Preserve the established rounding order whenever the product is representable.
      value += product / scratch[scratchOffset];
    } else {
      // A finite covariance can have an overflowing product before division by n.
      // Fix the quotient's bits before multiplying so fast math cannot regroup it.
      let normalizedFirst = getRoundedDifference(first / scratch[scratchOffset], 0.0,
        bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]));
      value += normalizedFirst * second;
    }
  }
}
${convergenceCoordinates ? 'if (rawFinite) { value = unnormalized / scratch[scratchOffset]; }' : ''}
scratch[scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + slot] = value;
`
        ),
        // contract-allow: single-invocation -- GPU_SUR_MAXIMUM_EQUATION_COUNT = 4 bounds 16 matrix entries; REGRESSION_TILE_MERGE_GROUP_SIZE = 4096 bounds each default-route merge.
        makeRoundNode(
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
        makeRoundNode(
          'tile-gls',
          tileCount * size,
          [...inputs, scratchBinding, statusBinding],
          inputDeclarations,
          `
if (status[statusOffset] != 0u) { return; }
let tile = index / ${size}u;
let first = index % ${size}u;
let equation = first / ${width}u;
${
  convergenceCoordinates
    ? `var normal: array<f32, ${size}>;
var equationRight: array<f32, ${equations}>;
let runtimeZero = bitcast<u32>(scratch[scratchOffset + RUNTIME_ZERO_SLOT]);
for (var row = tile * ${tileRows}u; row < min((tile + 1u) * ${tileRows}u, ${rows}u); row++) {
  if (!included(row)) { continue; }
  let value = predictor(row, equation, first % ${width}u);
  for (var other = 0u; other < ${equations}u; other++) {
    // Solve for the change from equation-wise OLS. Bringing the precision matrix into
    // each row product can bury a small cross-equation correction in rounded large terms.
    var residual = getRoundedDifference(responses[responsesOffset + other * ${rows}u + row],
      scratch[scratchOffset + RESPONSES_START + other], runtimeZero);
    for (var column = 0u; column < ${columns}u; column++) {
      residual = fma(-predictor(row, other, column + 1u),
        scratch[scratchOffset + INITIAL_START + other * ${columns}u + column], residual);
    }
    equationRight[other] += value * residual;
  }
  for (var second = 0u; second < ${size}u; second++) {
    normal[second] += value * predictor(row, second / ${width}u, second % ${width}u);
  }
}
let start = scratchOffset + PARTIAL_START + tile * PARTIAL_STRIDE + first * ${size + 1}u;
for (var second = 0u; second < ${size}u; second++) {
  scratch[start + second] = normal[second]
    * scratch[scratchOffset + PRECISION_START + equation * ${equations}u + second / ${width}u];
}
var right = 0.0;
for (var other = 0u; other < ${equations}u; other++) {
  right += scratch[scratchOffset + PRECISION_START + equation * ${equations}u + other] * equationRight[other];
}
scratch[start + ${size}u] = right;`
    : `var normal: array<f32, ${size}>;
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
scratch[start + ${size}u] = right;`
}
`
        ),
        // contract-allow: single-invocation -- GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT = 16 bounds 256 covariance entries; REGRESSION_TILE_MERGE_GROUP_SIZE = 4096 bounds each merge; SUR_GLS_REFINEMENT_COUNT = 2 bounds coefficient and inverse-solve correction passes.
        makeRoundNode(
          'solve-gls',
          1,
          [
            scratchBinding,
            statusBinding,
            coefficientsBinding,
            covarianceBinding,
            ...parameterBindings
          ],
          `${solveDeclarations}\n${getSURGLSRefinementWGSL(size, SUR_GLS_REFINEMENT_COUNT)}`,
          `
if (status[statusOffset] != 0u) { return; }
${
  props.iteration
    ? `var previous: array<f32, ${size}>;
for (var slot = 0u; slot < ${size}u; slot++) {
  if (scratch[scratchOffset + ITERATION_COUNT] > 0.0) { previous[slot] = coefficients[coefficientsOffset + slot]; }
  else {
    let equation = slot / ${width}u;
    let column = slot % ${width}u;
    if (column > 0u) { previous[slot] = scratch[scratchOffset + INITIAL_START + equation * ${columns}u + column - 1u]; }
    else {
      previous[slot] = scratch[scratchOffset + RESPONSES_START + equation];
      for (var predictorColumn = 0u; predictorColumn < ${columns}u; predictorColumn++) {
        previous[slot] -= scratch[scratchOffset + MEANS_START + equation * ${columns}u + predictorColumn] * scratch[scratchOffset + INITIAL_START + equation * ${columns}u + predictorColumn];
      }
    }
  }
}`
    : ''
}
var normal: array<f32, ${size * size}>;
var right: array<f32, ${size}>;
for (var first = 0u; first < ${size}u; first++) {
  right[first] = sumPartials(first * ${size + 1}u + ${size}u);
  for (var second = 0u; second < ${size}u; second++) { normal[first * ${size}u + second] = sumPartials(first * ${size + 1}u + second); }
}
// Keep the unscaled system: its residual can be evaluated in double-single precision.
var originalNormal: array<f32, ${size * size}>;
var originalRight: array<f32, ${size}>;
for (var slot = 0u; slot < ${size * size}u; slot++) { originalNormal[slot] = normal[slot]; }
for (var slot = 0u; slot < ${size}u; slot++) { originalRight[slot] = right[slot]; }
var scales:array<f32,${size}>;for(var column=0u;column<${size}u;column++){let diagonal=normal[column*${size + 1}u];if(!(diagonal>0.0)){status[statusOffset]=2u;return;}scales[column]=inverseSqrt(diagonal);right[column]*=scales[column];}
for(var first=0u;first<${size}u;first++){for(var second=0u;second<${size}u;second++){normal[first*${size}u+second]*=scales[first]*scales[second];}}
if(!choleskyFactorLimited_${size}(&normal)){status[statusOffset]=2u;return;}choleskySolve_${size}(&normal,&right);
var solution: array<f32, ${size}>;
for (var slot = 0u; slot < ${size}u; slot++) { solution[slot] = right[slot] * scales[slot]; }
refineSURSolution_${size}(&originalNormal, &originalRight, &normal, &scales, &solution);
${
  convergenceCoordinates
    ? `for (var equation = 0u; equation < ${equations}u; equation++) {
  for (var column = 0u; column < ${columns}u; column++) {
    solution[equation * ${width}u + column + 1u] +=
      scratch[scratchOffset + INITIAL_START + equation * ${columns}u + column];
  }
}`
    : ''
}

for (var slot = 0u; slot < ${size}u; slot++) {
  scratch[scratchOffset + CENTERED_SOLUTION_START + slot] = solution[slot];
}
for(var equation=0u;equation<${equations}u;equation++){var intercept=scratch[scratchOffset+RESPONSES_START+equation]+solution[equation*${width}u];for(var column=0u;column<${columns}u;column++){let slot=equation*${width}u+column+1u;let coefficient=solution[slot];coefficients[coefficientsOffset+slot]=coefficient;intercept-=scratch[scratchOffset+MEANS_START+equation*${columns}u+column]*coefficient;}coefficients[coefficientsOffset+equation*${width}u]=intercept;}
for(var targetColumn=0u;targetColumn<${size}u;targetColumn++){var transform:array<f32,${size}>;transform[targetColumn]=1.0;if(targetColumn%${width}u==0u){for(var column=0u;column<${columns}u;column++){transform[targetColumn+column+1u]=-scratch[scratchOffset+MEANS_START+(targetColumn/${width}u)*${columns}u+column];}}var covarianceRight: array<f32, ${size}>;for(var column=0u;column<${size}u;column++){covarianceRight[column]=transform[column];transform[column]*=scales[column];}choleskySolve_${size}(&normal,&transform);for(var column=0u;column<${size}u;column++){transform[column]*=scales[column];}
refineSURSolution_${size}(&originalNormal, &covarianceRight, &normal, &scales, &transform);
for(var destination=0u;destination<${size}u;destination++){var value=transform[destination];if(destination%${width}u==0u){for(var column=0u;column<${columns}u;column++){value-=scratch[scratchOffset+MEANS_START+(destination/${width}u)*${columns}u+column]*transform[destination+column+1u];}}covariance[covarianceOffset+destination*${size}u+targetColumn]=value*scratch[scratchOffset+ERROR_SCALE_START];}}
scratch[scratchOffset + ITERATION_COUNT] += 1.0;
${
  props.iteration
    ? `var largestChange = 0.0;
for (var slot = 0u; slot < ${size}u; slot++) {
  largestChange = max(largestChange, abs(coefficients[coefficientsOffset + slot] - previous[slot]) / max(1.0, abs(previous[slot])));
}
if (${props.initialResiduals ? 'scratch[scratchOffset + ITERATION_COUNT] > 1.0 && ' : ''}largestChange < parameters[parametersOffset]) {
  scratch[scratchOffset + ITERATION_ACTIVE] = 0.0;
} ${round + 1 === props.iteration.maximumIterations ? 'else { status[statusOffset] = 4u; scratch[scratchOffset + ITERATION_ACTIVE] = 0.0; }' : `else { ${iterationGates[round + 1]?.armWGSL ?? ''} }`}`
    : ''
}

`
        )
      ];
    }).flat(2),
    ...(props.iteration || iterationCount
      ? [
          // contract-allow: single-invocation -- GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT = 16 bounds coefficients and 256 covariance entries; GPU_SUR_MAXIMUM_EQUATION_COUNT = 4 bounds 16 error covariance entries.
          makeNode(
            'publish-iteration',
            1,
            [
              scratchBinding,
              ...(props.iteration
                ? [statusBinding, coefficientsBinding, covarianceBinding, residualCovarianceBinding]
                : []),
              ...(iterationCount
                ? [
                    {
                      name: 'iterationCount',
                      view: iterationCount,
                      type: 'u32',
                      access: 'read_write'
                    } as const
                  ]
                : [])
            ],
            constants,
            `${iterationCount ? 'iterationCount[iterationCountOffset] = u32(scratch[scratchOffset + ITERATION_COUNT]);' : ''}
${
  props.iteration
    ? `if (status[statusOffset] != 0u && status[statusOffset] != 4u) {
  for (var slot = 0u; slot < ${size}u; slot++) { coefficients[coefficientsOffset + slot] = getNaN(); }
  for (var slot = 0u; slot < ${size * size}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
  for (var slot = 0u; slot < ${equations * equations}u; slot++) { residualCovariance[residualCovarianceOffset + slot] = getNaN(); }
}`
    : ''
}`
          )
        ]
      : []),
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
if ((status[statusOffset] == 0u ${props.iteration ? '|| status[statusOffset] == 4u' : ''}) && included(row)) {
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
