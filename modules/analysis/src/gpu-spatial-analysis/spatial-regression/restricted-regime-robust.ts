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
import {
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../analysis-core/kernels/incomplete-gamma-wgsl';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {createWeightsLossNodes} from '../spatial-weights/spatial-weights-loss';
import {getRowTileSumNodes, type RegressionTileLayout} from './regression-row-tiles';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';

/** Optional covariance estimator; HAC consumes a symmetric, unstandardized kernel with unit diagonal. */
export type RestrictedRegimeRobustOptions<Kernel = GPUSpatialWeights> =
  | {type: 'white'; smallSample?: boolean}
  | {type: 'hac'; kernelWeights: Kernel};

/** Robust inference succeeded (including the no-varying-coefficients case). */
export const RESTRICTED_REGIME_ROBUST_OK = 0;
/** The restricted fit failed or has no residual degrees of freedom; robust outputs are suppressed. */
export const RESTRICTED_REGIME_ROBUST_FIT_UNAVAILABLE = 1;
/** Incomplete or invalid kernel: missing unit diagonal, asymmetry, or invalid CSR/value. */
export const RESTRICTED_REGIME_ROBUST_INVALID_KERNEL = 2;
/** Nonfinite covariance or a tested contrast covariance that is not numerically positive definite. */
export const RESTRICTED_REGIME_ROBUST_INVALID_COVARIANCE = 3;

/** Optional outputs, all required when robust inference is requested. */
export type RestrictedRegimeRobustOutput<
  View = GraphDataView<'float32'>,
  Status = GraphDataView<'uint32'>
> = {
  covariance: View;
  chow: View;
  stability: View;
  status: Status;
};

/** Internal workspace locations of the restricted shifted fit. */
export type RestrictedRegimeWorkspace = {
  count: number;
  status: number;
  levels: number;
  estimate: number;
  inverse: number;
  transform: number;
  origin: number;
  forms: number;
  runtimeZero: number;
};

/** Appends robust inference without changing the classical command graph. @internal */
export function getRestrictedRegimeRobustNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    predictorCount: number;
    regimeCount: number;
    coefficientCount: number;
    common: readonly boolean[];
    offsets: readonly number[];
    inputs: readonly WGSLKernelBinding[];
    declarations: string;
    workspace: GraphDataView<'float32'>;
    locations: RestrictedRegimeWorkspace;
    layout: RegressionTileLayout;
    robust: RestrictedRegimeRobustOptions;
    output: RestrictedRegimeRobustOutput;
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    rowCount,
    predictorCount: columns,
    coefficientCount: size,
    locations,
    robust,
    output
  } = props;
  const kernel = robust.type === 'hac' ? robust.kernelWeights : undefined;
  // Scores retain signed coefficient influences and residual formation error for later contrasts.
  const scoreStride = 2 * size + 3;
  const scores = createTransientView(
    graph,
    `${id}-robust-scores`,
    'float32',
    rowCount * scoreStride
  );
  const matrixEntries = size * size;
  const invalidKernelSlot = matrixEntries;
  const maximumNeighborsSlot = invalidKernelSlot + 1;
  const totals = createTransientView(
    graph,
    `${id}-robust-totals`,
    'float32',
    maximumNeighborsSlot + 1
  );
  const shiftedCovariance = createTransientView(
    graph,
    `${id}-robust-shifted-covariance`,
    'float32',
    matrixEntries
  );
  const workspaceBinding = {
    name: 'workspace',
    view: props.workspace,
    type: 'f32' as const,
    access: 'read' as const
  };
  const kernelStatus = kernel?.status
    ? Object.entries(kernel.status).filter(
        ([name, view]) =>
          [
            'overflow',
            'candidateOverflow',
            'iterationLimitReached',
            'iterationLimit',
            'invalidInput',
            'invalidCount',
            'unresolvedCount',
            'approximation',
            'converged',
            'complete'
          ].includes(name) && Boolean(view)
      )
    : [];
  const kernelIncomplete = kernelStatus.length
    ? createTransientView(graph, `${id}-robust-kernel-incomplete`, 'uint32', 1)
    : undefined;
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-robust-scores`,
      operation: 'GPURestrictedRegimeRegression',
      variant: 'robust-scores',
      invocationCount: rowCount,
      bindings: [
        ...props.inputs,
        workspaceBinding,
        {name: 'scores', view: scores, type: 'f32', access: 'read_write'}
      ],
      declarations: `${props.declarations}\n${ROUNDED_DIFFERENCE_WGSL}`,
      body: `let base = scoresOffset + index * ${scoreStride}u;
for (var slot = 0u; slot < ${scoreStride}u; slot++) { scores[base + slot] = 0.0; }
if (!isFiniteBits(workspace[workspaceOffset + ${locations.estimate}u]) || workspace[workspaceOffset + ${locations.count}u] <= ${size}.0 || !included(index)) { return; }
let regime = regimes[regimesOffset + index];
let level = workspaceOffset + ${locations.levels}u + regime * ${columns + 2}u;
let formBase = workspaceOffset + ${locations.forms}u;
let runtimeZero = bitcast<u32>(workspace[workspaceOffset + ${locations.runtimeZero}u]);
let pivot = u32(workspace[formBase]);
let pivotDeviation = getRoundedDifference(rawValue(index, pivot), workspace[level + 1u + pivot], runtimeZero);
var design: array<f32, ${size}>;
var residual = getRoundedDifference(response[responseOffset + index], workspace[level + 1u], runtimeZero);
var residualScale = abs(residual);
for (var column = 0u; column < ${columns + 1}u; column++) {
  var deviation = 0.0;
  if (column > 0u) { deviation = getRoundedDifference(rawValue(index, column), workspace[level + 1u + column], runtimeZero); }
  let value = workspace[formBase + 1u + column] + deviation + workspace[formBase + ${columns + 2}u + column] * pivotDeviation;
  let slot = OFFSETS[column] + select(regime, 0u, COMMON[column]);
  design[slot] = value;
  let term = value * workspace[workspaceOffset + ${locations.estimate}u + slot];
  residual = getRoundedDifference(residual, term, runtimeZero);
  residualScale += abs(term);
}
for (var first = 0u; first < ${size}u; first++) { var value = 0.0;
  for (var second = 0u; second < ${size}u; second++) { let term = workspace[workspaceOffset + ${locations.inverse}u + first * ${size}u + second] * design[second]; value += term; }
  scores[base + first] = value * residual;
  scores[base + ${size + 1}u + first] = value;
}
scores[base + ${size}u] = residual;
scores[base + ${2 * size + 1}u] = ${8 * 2 ** -23 * size} * residualScale;`
    }),
    ...getRowTileSumNodes<Parameters>(graph, {
      id: `${id}-robust-meat`,
      operation: 'GPURestrictedRegimeRegression',
      variant: 'robust-meat',
      rowCount,
      layout: props.layout,
      sumCount: invalidKernelSlot + 1,
      maximumCount: 1,
      bindings: [
        {name: 'scores', view: scores, type: 'f32', access: 'read'},
        ...(kernel
          ? [
              {
                name: 'offsets',
                view: kernel.offsets,
                type: 'u32' as const,
                access: 'read' as const
              },
              {
                name: 'neighbors',
                view: kernel.neighbors,
                type: 'u32' as const,
                access: 'read' as const
              },
              {
                name: 'weights',
                view: kernel.weights,
                type: 'f32' as const,
                access: 'read' as const
              },
              ...(['requiredCount', 'count'] as const).flatMap(name =>
                kernel.status?.[name]
                  ? [
                      {
                        name: `kernel_${name}`,
                        view: kernel.status[name]!,
                        type: 'u32' as const,
                        access: 'read' as const
                      }
                    ]
                  : []
              )
            ]
          : [])
      ],
      declarations: `${IS_FINITE_VALUE_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${kernel ? getBinarySearchWGSL('neighbors') : ''}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let base = scoresOffset + row * ${scoreStride}u;
  var smoothed: array<f32, ${size}>;
  var compensation: array<f32, ${size}>;
  let runtimeZero = bitcast<u32>(scores[base + ${2 * size + 2}u]);
  ${
    kernel
      ? `let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u];
  var diagonal = false; var previous = 0u;
  let retained = offsets[offsetsOffset + ${rowCount}u];
  if (row == 0u) {
    if (retained > ${kernel.neighbors.length}u ${kernel.status?.count ? '|| retained != kernel_count[kernel_countOffset]' : ''} ${kernel.status?.requiredCount ? `|| kernel_requiredCount[kernel_requiredCountOffset] > retained ${kernel.status?.count ? '|| kernel_requiredCount[kernel_requiredCountOffset] > kernel_count[kernel_countOffset]' : ''}` : ''}) { (*terms)[${invalidKernelSlot}] += 1.0; }
  }
  if (begin > end || end > ${kernel.neighbors.length}u || (row == 0u && begin != 0u)) { (*terms)[${invalidKernelSlot}] += 1.0; return; }
  (*terms)[${maximumNeighborsSlot}] = max((*terms)[${maximumNeighborsSlot}], f32(end - begin));
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor >= ${rowCount}u || (slot > begin && neighbor <= previous) || !isFiniteValue(weight) || weight < 0.0) { (*terms)[${invalidKernelSlot}] += 1.0; continue; }
    previous = neighbor;
    if (neighbor == row && weight == 1.0) { diagonal = true; }
    // Check symmetry without a dense row scan. CSR rows must be sorted and unique.
    let otherBegin = min(offsets[offsetsOffset + neighbor], ${kernel.neighbors.length}u);
    let otherEnd = min(offsets[offsetsOffset + neighbor + 1u], ${kernel.neighbors.length}u);
    let low = lowerBoundNeighbors(otherBegin, otherEnd, row);
    if (low >= otherEnd) { (*terms)[${invalidKernelSlot}] += 1.0; }
    else if (neighbors[neighborsOffset + low] != row || weights[weightsOffset + low] != weight) { (*terms)[${invalidKernelSlot}] += 1.0; }
    for (var column = 0u; column < ${size}u; column++) {
      let value = scores[scoresOffset + neighbor * ${scoreStride}u + column];
      let term = weight * value;
      // Kahan with rounded differences prevents Metal fast math from deleting compensation.
      let adjusted = getRoundedDifference(term, compensation[column], runtimeZero);
      let next = getRoundedDifference(smoothed[column], -adjusted, runtimeZero);
      compensation[column] = getRoundedDifference(getRoundedDifference(next, smoothed[column], runtimeZero), adjusted, runtimeZero);
      smoothed[column] = next;
    }
  }
  if (!diagonal) { (*terms)[${invalidKernelSlot}] += 1.0; }`
      : `(*terms)[${maximumNeighborsSlot}] = 1.0; for (var column = 0u; column < ${size}u; column++) { smoothed[column] = scores[base + column]; }`
  }
  for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) {
    (*terms)[first * ${size}u + second] += scores[base + first] * smoothed[second];
  } }
}`,
      partial: createTransientView(
        graph,
        `${id}-robust-partials`,
        'float32',
        props.layout.tileCount * (maximumNeighborsSlot + 1)
      ),
      output: totals
    }),
    ...(kernelIncomplete
      ? createWeightsLossNodes(
          graph,
          `${id}-robust-kernel-status`,
          kernelStatus
            .filter(([name]) => name !== 'complete' && name !== 'converged')
            .map(([, view]) => view!),
          kernelIncomplete,
          false,
          kernelStatus
            .filter(([name]) => name === 'complete' || name === 'converged')
            .map(([, view]) => view!)
        )
      : []),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-robust-covariance`,
      operation: 'GPURestrictedRegimeRegression',
      variant: 'robust-covariance',
      // contract-allow: single-invocation -- <=16 coefficients, no row scan.
      invocationCount: 1,
      bindings: [
        workspaceBinding,
        {name: 'totals', view: totals, type: 'f32', access: 'read'},
        ...(kernel?.status?.requiredCount
          ? [
              {
                name: 'kernelRequiredCount',
                view: kernel.status.requiredCount,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(kernelIncomplete
          ? [
              {
                name: 'incomplete',
                view: kernelIncomplete,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'shifted', view: shiftedCovariance, type: 'f32', access: 'read_write'},
        {name: 'covariance', view: output.covariance, type: 'f32', access: 'read_write'},
        {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}`,
      body: `status[statusOffset] = 1u;
for (var slot = 0u; slot < ${matrixEntries}u; slot++) { shifted[shiftedOffset + slot] = getNaN(); }
for (var slot = 0u; slot < ${matrixEntries}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); }
if (!isFiniteValue(workspace[workspaceOffset + ${locations.estimate}u]) || workspace[workspaceOffset + ${locations.count}u] <= ${size}.0) { return; }
status[statusOffset] = 2u;
if (totals[totalsOffset + ${invalidKernelSlot}u] != 0.0 ${kernelIncomplete ? '|| incomplete[incompleteOffset] != 0u' : ''} ${kernel?.status?.requiredCount ? `|| kernelRequiredCount[kernelRequiredCountOffset] > ${kernel.neighbors.length}u` : ''}) { return; }
let count = workspace[workspaceOffset + ${locations.count}u];
let correction = ${robust.type === 'white' && robust.smallSample ? `count / (count - ${size}.0)` : '1.0'};
var matrix: array<f32, ${size * size}>; var product: array<f32, ${size * size}>;
status[statusOffset] = 3u;
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) {
  let value = correction * (0.5 * totals[totalsOffset + first * ${size}u + second] + 0.5 * totals[totalsOffset + second * ${size}u + first]);
  if (!isFiniteValue(value)) { return; } matrix[first * ${size}u + second] = value;
} }
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) { var value = 0.0;
  for (var inner = 0u; inner < ${size}u; inner++) { value += workspace[workspaceOffset + ${locations.transform}u + first * ${size}u + inner] * matrix[inner * ${size}u + second]; }
  product[first * ${size}u + second] = value;
} }
var raw: array<f32, ${size * size}>;
for (var first = 0u; first < ${size}u; first++) { for (var second = 0u; second < ${size}u; second++) { var value = 0.0;
  for (var inner = 0u; inner < ${size}u; inner++) { value += product[first * ${size}u + inner] * workspace[workspaceOffset + ${locations.transform}u + second * ${size}u + inner]; }
  if (!isFiniteValue(value)) { return; } raw[first * ${size}u + second] = value;
} }
for (var slot = 0u; slot < ${size * size}u; slot++) { shifted[shiftedOffset + slot] = matrix[slot]; covariance[covarianceOffset + slot] = raw[slot]; }
status[statusOffset] = 0u;`
    })
  ];
  const contrastColumns = props.common.flatMap((shared, column) =>
    shared ? [] : Array.from({length: props.regimeCount - 1}, () => column)
  );
  const positives = props.common.flatMap((shared, column) =>
    shared
      ? []
      : Array.from(
          {length: props.regimeCount - 1},
          (_, regime) => props.offsets[column] + regime + 1
        )
  );
  const negatives = props.common.flatMap((shared, column) =>
    shared ? [] : Array.from({length: props.regimeCount - 1}, () => props.offsets[column])
  );
  const contrasts = Math.max(1, positives.length);
  const literal = (values: number[]) =>
    (values.length ? values : [0]).map(value => `${value}u`).join(',');
  const basisStride = 2 + contrasts + contrasts * size;
  const contrastEntries = contrasts * contrasts;
  const projectedStride = 3 * contrastEntries;
  const basis = createTransientView(
    graph,
    `${id}-robust-basis`,
    'float32',
    (columns + 2) * basisStride
  );
  const projected = createTransientView(
    graph,
    `${id}-robust-projected`,
    'float32',
    (columns + 2) * projectedStride
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-robust-basis`,
      operation: 'GPURestrictedRegimeRegression',
      variant: 'robust-basis',
      invocationCount: columns + 2,
      bindings: [
        workspaceBinding,
        {name: 'shifted', view: shiftedCovariance, type: 'f32', access: 'read'},
        {name: 'basis', view: basis, type: 'f32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_VALUE_WGSL}
const COLUMNS = array<u32, ${contrasts}>(${literal(contrastColumns)});
const POSITIVES = array<u32, ${contrasts}>(${literal(positives)});
const NEGATIVES = array<u32, ${contrasts}>(${literal(negatives)});`,
      body: `let test = index;
  var selected: array<u32, ${contrasts}>; var count = 0u;
  for (var contrast = 0u; contrast < ${positives.length}u; contrast++) { if (test == ${columns + 1}u || COLUMNS[contrast] == test) { selected[count] = contrast; count++; } }

  var rows: array<f32, ${contrasts * size}>; var difference: array<f32, ${contrasts}>;
  for (var first = 0u; first < count; first++) { let positive = POSITIVES[selected[first]]; let negative = NEGATIVES[selected[first]];
    difference[first] = workspace[workspaceOffset + ${locations.origin}u + positive] - workspace[workspaceOffset + ${locations.origin}u + negative];
    for (var slot = 0u; slot < ${size}u; slot++) { rows[first * ${size}u + slot] = workspace[workspaceOffset + ${locations.transform}u + positive * ${size}u + slot] - workspace[workspaceOffset + ${locations.transform}u + negative * ${size}u + slot]; }
  }
  // Orthogonalize the joint intercept/slope contrasts in the covariance metric.
  // A regime mean can be arbitrarily far from the informative rows after a leverage outlier.
  for (var first = 0u; first < count; first++) { if (COLUMNS[selected[first]] != 0u) { continue; }
    let regime = POSITIVES[selected[first]] - NEGATIVES[selected[first]];
    for (var second = 0u; second < count; second++) { let other = selected[second];
      if (COLUMNS[other] == 0u || POSITIVES[other] - NEGATIVES[other] != regime) { continue; }
      var cross = 0.0; var slopeVariance = 0.0;
      for (var left = 0u; left < ${size}u; left++) { for (var right = 0u; right < ${size}u; right++) {
        let entry = shifted[shiftedOffset + left * ${size}u + right];
        cross += rows[first * ${size}u + left] * entry * rows[second * ${size}u + right];
        slopeVariance += rows[second * ${size}u + left] * entry * rows[second * ${size}u + right];
      } }
      if (!(slopeVariance > 0.0) || !isFiniteValue(cross) || !isFiniteValue(slopeVariance)) { continue; }
      let weight = -cross / slopeVariance;
      for (var slot = 0u; slot < ${size}u; slot++) { rows[first * ${size}u + slot] += weight * rows[second * ${size}u + slot]; }
    }
  }

let destination = basisOffset + index * ${basisStride}u;
basis[destination] = f32(count);
basis[destination + 1u] = ${robust.type === 'white' && robust.smallSample ? `workspace[workspaceOffset + ${locations.count}u] / (workspace[workspaceOffset + ${locations.count}u] - ${size}.0)` : '1.0'};
for (var first = 0u; first < count; first++) {
  for (var slot = 0u; slot < ${size}u; slot++) { difference[first] += rows[first * ${size}u + slot] * workspace[workspaceOffset + ${locations.estimate}u + slot]; }
  basis[destination + 2u + first] = difference[first];
  for (var slot = 0u; slot < ${size}u; slot++) { basis[destination + ${2 + contrasts}u + first * ${size}u + slot] = rows[first * ${size}u + slot]; }
}
`
    })
  );
  for (let test = 0; test < columns + 2; test++) {
    nodes.push(
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-robust-projected-${test}`,
        operation: 'GPURestrictedRegimeRegression',
        variant: 'robust-projected',
        rowCount,
        layout: props.layout,
        sumCount: projectedStride,
        bindings: [
          {name: 'scores', view: scores, type: 'f32', access: 'read'},
          {name: 'basis', view: basis, type: 'f32', access: 'read'},
          ...(kernel
            ? [
                {
                  name: 'offsets',
                  view: kernel.offsets,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'neighbors',
                  view: kernel.neighbors,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'weights',
                  view: kernel.weights,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `${ROUNDED_DIFFERENCE_WGSL}
fn projectRow(row: u32, values: ptr<function, array<f32, ${contrasts}>>, errors: ptr<function, array<f32, ${contrasts}>>) {
  let base = scoresOffset + row * ${scoreStride}u;
  let residual = scores[base + ${size}u];
  let residualError = scores[base + ${2 * size + 1}u];
  let count = u32(basis[basisOffset + ${test * basisStride}u]);
  for (var contrast = 0u; contrast < count; contrast++) {
    var influence = 0.0; var absoluteInfluence = 0.0;
    for (var slot = 0u; slot < ${size}u; slot++) {
      let term = basis[basisOffset + ${test * basisStride + 2 + contrasts}u + contrast * ${size}u + slot] * scores[base + ${size + 1}u + slot];
      influence += term; absoluteInfluence += abs(term);
    }
    (*values)[contrast] = influence * residual;
    // Take absolute values AFTER the signed influence meets the chosen contrast basis.
    (*errors)[contrast] = abs(influence) * residualError + ${2 * 2 ** -23 * size} * absoluteInfluence * abs(residual);
  }
}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let count = u32(basis[basisOffset + ${test * basisStride}u]);
  if (count == 0u) { return; }
  var values: array<f32, ${contrasts}>; var errors: array<f32, ${contrasts}>;
  projectRow(row, &values, &errors);
  var smoothed: array<f32, ${contrasts}>; var smoothedErrors: array<f32, ${contrasts}>;
  ${
    kernel
      ? `var absoluteNeighbors: array<f32, ${contrasts}>; var compensation: array<f32, ${contrasts}>;
  let runtimeZero = bitcast<u32>(scores[scoresOffset + row * ${scoreStride}u + ${2 * size + 2}u]);
  let begin = min(offsets[offsetsOffset + row], ${kernel.neighbors.length}u);
  let end = min(offsets[offsetsOffset + row + 1u], ${kernel.neighbors.length}u);
  for (var slot = begin; slot < end; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ${rowCount}u) { continue; }
    var other: array<f32, ${contrasts}>; var otherErrors: array<f32, ${contrasts}>;
    projectRow(neighbor, &other, &otherErrors);
    let weight = weights[weightsOffset + slot];
    for (var column = 0u; column < count; column++) {
      let adjusted = getRoundedDifference(weight * other[column], compensation[column], runtimeZero);
      let next = getRoundedDifference(smoothed[column], -adjusted, runtimeZero);
      compensation[column] = getRoundedDifference(getRoundedDifference(next, smoothed[column], runtimeZero), adjusted, runtimeZero);
      smoothed[column] = next;
      absoluteNeighbors[column] += weight * abs(other[column]);
      smoothedErrors[column] += weight * otherErrors[column];
    }
  }
  for (var column = 0u; column < count; column++) {
    smoothedErrors[column] += (${4 * 2 ** -23} + ${8 * 2 ** -46} * f32(end - begin)) * absoluteNeighbors[column];
  }`
      : `smoothed = values; smoothedErrors = errors;`
  }
  let correction = basis[basisOffset + ${test * basisStride + 1}u];
  for (var first = 0u; first < count; first++) { for (var second = 0u; second < count; second++) {
    let entry = first * ${contrasts}u + second;
    let product = values[first] * smoothed[second];
    (*terms)[entry] += correction * product;
    // Tile rounding applies to the smoothed product, not to uncancelled CSR terms.
    (*terms)[${contrastEntries}u + entry] += correction * abs(product);
    (*terms)[${2 * contrastEntries}u + entry] += correction * (errors[first] * (abs(smoothed[second]) + smoothedErrors[second]) + abs(values[first]) * smoothedErrors[second]);
  } }
}`,
        partial: createTransientView(
          graph,
          `${id}-robust-projected-partials-${test}`,
          'float32',
          props.layout.tileCount * projectedStride
        ),
        output: projected,
        outputSlot: test * projectedStride
      })
    );
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-robust-tests`,
      operation: 'GPURestrictedRegimeRegression',
      variant: 'robust-wald',
      // contract-allow: single-invocation -- <=14 contrasts, <=9 tests; no row scan. One writer owns status.
      invocationCount: 1,
      bindings: [
        {name: 'basis', view: basis, type: 'f32', access: 'read'},
        {name: 'projected', view: projected, type: 'f32', access: 'read'},
        {name: 'covariance', view: output.covariance, type: 'f32', access: 'read_write'},
        {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
        {name: 'stability', view: output.stability, type: 'f32', access: 'read_write'},
        {name: 'chow', view: output.chow, type: 'f32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${getCholeskyWGSL(contrasts)}
const COLUMNS = array<u32, ${contrasts}>(${literal(contrastColumns)});
const POSITIVES = array<u32, ${contrasts}>(${literal(positives)});
const NEGATIVES = array<u32, ${contrasts}>(${literal(negatives)});`,
      body: `for (var test = 0u; test < ${columns + 2}u; test++) { stability[stabilityOffset + test * 3u] = getNaN(); stability[stabilityOffset + test * 3u + 1u] = 0.0; stability[stabilityOffset + test * 3u + 2u] = getNaN(); }
chow[chowOffset] = getNaN(); chow[chowOffset + 1u] = 0.0; chow[chowOffset + 2u] = getNaN();
if (status[statusOffset] != 0u) { return; }
var results: array<f32, ${(columns + 2) * 3}>;
for (var test = 0u; test < ${columns + 2}u; test++) {
  results[test * 3u] = getNaN(); results[test * 3u + 2u] = getNaN();
  let source = basisOffset + test * ${basisStride}u;
  let count = u32(basis[source]);
  if (count == 0u) { continue; }
  var difference: array<f32, ${contrasts}>;
  var normal: array<f32, ${contrasts * contrasts}>; var right: array<f32, ${contrasts}>;
  var scales: array<f32, ${contrasts}>; var uncertainty: array<f32, ${contrasts * contrasts}>;
  for (var first = 0u; first < ${contrasts}u; first++) { normal[first * ${contrasts + 1}u] = 1.0; }
  var valid = true;
  for (var first = 0u; first < count; first++) {
    difference[first] = basis[source + 2u + first];
    for (var second = 0u; second < count; second++) {
      let entry = first * ${contrasts}u + second;
      let reverse = second * ${contrasts}u + first;
      let totalsBase = projectedOffset + test * ${projectedStride}u;
      normal[entry] = 0.5 * (projected[totalsBase + entry] + projected[totalsBase + reverse]);
      uncertainty[entry] = 0.5 * (
        ${8 * 2 ** -23 * (props.layout.tileRowCount + Math.ceil(Math.log2(props.layout.tileCount)) + size)} * (projected[totalsBase + ${contrastEntries}u + entry] + projected[totalsBase + ${contrastEntries}u + reverse]) +
        projected[totalsBase + ${2 * contrastEntries}u + entry] + projected[totalsBase + ${2 * contrastEntries}u + reverse]);
    }
    let diagonal = normal[first * ${contrasts + 1}u];
    if (!(diagonal > uncertainty[first * ${contrasts + 1}u]) || !isFiniteValue(diagonal)) { valid = false; }
    scales[first] = inverseSqrt(max(diagonal, 1e-30));
  }
  for (var first = 0u; first < count; first++) { right[first] = difference[first] * scales[first];
    for (var second = 0u; second < count; second++) { normal[first * ${contrasts}u + second] *= scales[first] * scales[second]; uncertainty[first * ${contrasts}u + second] *= scales[first] * scales[second]; }
  }
  if (valid) { valid = choleskyFactor_${contrasts}(&normal); }
  if (valid) {
    var elimination: array<f32, ${contrasts * contrasts}>;
    for (var row = 0u; row < count; row++) {
      elimination[row * ${contrasts + 1}u] = 1.0;
      for (var previous = 0u; previous < row; previous++) {
        let weight = normal[row * ${contrasts}u + previous] / normal[previous * ${contrasts + 1}u];
        for (var column = 0u; column < count; column++) { elimination[row * ${contrasts}u + column] -= weight * elimination[previous * ${contrasts}u + column]; }
      }
      var error = 0.0;
      for (var first = 0u; first < count; first++) { for (var second = 0u; second < count; second++) {
        error += abs(elimination[row * ${contrasts}u + first]) * uncertainty[first * ${contrasts}u + second] * abs(elimination[row * ${contrasts}u + second]);
      } }
      let pivot = normal[row * ${contrasts + 1}u];
      if (!(pivot * pivot > error)) { valid = false; }
    }
  }
  if (!valid) { status[statusOffset] = 3u; break; }
  choleskySolve_${contrasts}(&normal, &right);
  var statistic = 0.0; for (var first = 0u; first < count; first++) { statistic += difference[first] * scales[first] * right[first]; }
  if (!isFiniteValue(statistic) || statistic < 0.0) { status[statusOffset] = 3u; break; }
  results[test * 3u] = statistic; results[test * 3u + 1u] = f32(count); results[test * 3u + 2u] = getChiSquareSurvivalValue(statistic, f32(count));
}
if (status[statusOffset] != 0u) { for (var slot = 0u; slot < ${size * size}u; slot++) { covariance[covarianceOffset + slot] = getNaN(); } return; }
for (var slot = 0u; slot < ${(columns + 2) * 3}u; slot++) { stability[stabilityOffset + slot] = results[slot]; }
for (var slot = 0u; slot < 3u; slot++) { chow[chowOffset + slot] = results[${(columns + 1) * 3}u + slot]; }`
    })
  );
  return nodes;
}
