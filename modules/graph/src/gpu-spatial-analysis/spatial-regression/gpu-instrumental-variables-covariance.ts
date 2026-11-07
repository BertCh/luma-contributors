// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

export type GPUInstrumentalVariablesCovarianceProps = {
  id?: string;
  /** Row-major structural design Z, excluding the implicit intercept. */
  design: GraphDataView<'float32'>;
  coefficientCount: number;
  /** Row-major instruments H, excluding the implicit intercept. Include the exogenous design. */
  instruments: GraphDataView<'float32'>;
  instrumentCount: number;
  /** Structural residuals from the SAME rows and fitted IV model. All rows must be finite. */
  residuals: GraphDataView<'float32'>;
  method?: 'HC0' | 'HC1' | 'HAC';
  /** HAC kernel, distinct from the model's spatial-lag weights. Symmetric, unit diagonal, PSD. */
  kernel?: GPUSpatialWeights;
  output: {
    /** Full row-major (coefficientCount + 1)² covariance, intercept first. */
    covariance: GraphDataView<'float32'>;
    /** 0 success; 1 singular projection; 2 insufficient rows; 3 nonfinite input; 4 invalid kernel. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Centered and scaled IV sandwich covariance: A H' diag(u) K diag(u) H A', where
 * A = (Z'P_H Z)^-1 Z'H(H'H)^-1. HC0 uses identity K; HC1 multiplies by n/(n-p),
 * p including the intercept. HAC has no finite-sample correction. The kernel's diagonal,
 * symmetry, bounds and nonnegative weights are checked on GPU; PSD remains a caller precondition.
 * Inputs contain only estimation rows, with no automatic deletion/imputation. This is a covariance
 * stage, never a new estimator or an OLS sandwich applied incorrectly to endogenous regressors.
 */
export class GPUInstrumentalVariablesCovariance implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUInstrumentalVariablesCovarianceProps) {
    this.id = props.id ?? 'instrumental-variables-covariance';
    const rows = props.residuals.length;
    validateCapacity(this.id, 'rows', rows, 1, 65536);
    validateCapacity(this.id, 'coefficientCount', props.coefficientCount, 1, 15);
    validateCapacity(this.id, 'instrumentCount', props.instrumentCount, props.coefficientCount, 16);
    if (
      !['HC0', 'HC1', 'HAC'].includes(props.method ?? 'HC0') ||
      (props.method === 'HAC') !== Boolean(props.kernel)
    )
      throw new Error(`${this.id}: HAC requires a kernel; HC methods do not accept one`);
    if (props.kernel && validateGPUSpatialWeights(this.id, props.kernel) !== rows)
      throw new Error(`${this.id}: kernel row mismatch`);
    for (const [view, format, length] of [
      [props.design, 'float32', rows * props.coefficientCount],
      [props.instruments, 'float32', rows * props.instrumentCount],
      [props.residuals, 'float32', rows],
      [props.output.covariance, 'float32', (props.coefficientCount + 1) ** 2],
      [props.output.status, 'uint32', 1]
    ] as const) {
      validatePackedView(view, [format], this.id);
      if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.design,
      props.instruments,
      props.residuals,
      props.kernel?.offsets,
      props.kernel?.neighbors,
      props.kernel?.weights,
      props.kernel?.status?.overflow
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.residuals.length;
    const columns = props.coefficientCount;
    const instruments = props.instrumentCount;
    const coefficients = columns + 1;
    const means = createTransientView(graph, `${id}-means`, 'float32', columns + instruments);
    const projection = createTransientView(
      graph,
      `${id}-projection`,
      'float32',
      columns * instruments
    );
    const scores = createTransientView(graph, `${id}-scores`, 'float32', rows * coefficients);
    const kernelStatus = createTransientView(graph, `${id}-kernel-status`, 'uint32', rows);
    const covariance = createTransientView(
      graph,
      `${id}-candidate-covariance`,
      'float32',
      coefficients ** 2
    );
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const node = (
      name: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUInstrumentalVariablesCovariance',
        variant: name,
        invocationCount,
        bindings,
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${declarations}`,
        body
      });
    validateGraphViewsBelongToGraph(id, graph, [
      props.design,
      props.instruments,
      props.residuals,
      props.kernel?.offsets,
      props.kernel?.neighbors,
      props.kernel?.weights,
      props.kernel?.status?.overflow,
      ...Object.values(props.output)
    ]);
    const nodes = [
      node(
        'means',
        columns + instruments,
        [
          binding('design', props.design),
          binding('instruments', props.instruments),
          binding('means', means, 'read_write')
        ],
        `var sum = 0.0; var correction = 0.0;
for (var row = 0u; row < ${rows}u; row++) { var value = 0.0;
if (index < ${columns}u) { value = design[designOffset + row * ${columns}u + index]; } else { value = instruments[instrumentsOffset + row * ${instruments}u + index - ${columns}u]; }
let term = value - correction; let next = sum + term; correction = (next - sum) - term; sum = next; }
means[meansOffset + index] = sum / ${rows}.0;`
      ),
      node(
        'projection',
        1,
        [
          binding('design', props.design),
          binding('instruments', props.instruments),
          binding('means', means),
          binding('projection', projection, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        `status[statusOffset] = 0u;
for (var slot = 0u; slot < ${columns * instruments}u; slot++) { projection[projectionOffset + slot] = getNaN(); }
if (${rows}u <= ${coefficients}u) { status[statusOffset] = 2u; return; }
var instrumentGram: array<f32, ${instruments * instruments}>;
var crossGram: array<f32, ${instruments * columns}>;
var designScale: array<f32, ${columns}>;
for (var row = 0u; row < ${rows}u; row++) {
 var instrumentRow: array<f32, ${instruments}>; var designRow: array<f32, ${columns}>;
 for (var column = 0u; column < ${columns}u; column++) { let value = design[designOffset + row * ${columns}u + column] - means[meansOffset + column]; if (!isFiniteBits(value)) { status[statusOffset] = 3u; return; } designRow[column] = value; designScale[column] += value * value; }
 for (var instrument = 0u; instrument < ${instruments}u; instrument++) { let value = instruments[instrumentsOffset + row * ${instruments}u + instrument] - means[meansOffset + ${columns}u + instrument]; if (!isFiniteBits(value)) { status[statusOffset] = 3u; return; } instrumentRow[instrument] = value; }
 for (var first = 0u; first < ${instruments}u; first++) {
  for (var second = 0u; second < ${instruments}u; second++) { instrumentGram[first * ${instruments}u + second] += instrumentRow[first] * instrumentRow[second]; }
  for (var column = 0u; column < ${columns}u; column++) { crossGram[first * ${columns}u + column] += instrumentRow[first] * designRow[column]; }
 }
}
var instrumentScale: array<f32, ${instruments}>;
for (var instrument = 0u; instrument < ${instruments}u; instrument++) { let diagonal = instrumentGram[instrument * ${instruments + 1}u]; if (!(diagonal > 0.0)) { status[statusOffset] = 1u; return; } instrumentScale[instrument] = inverseSqrt(diagonal); }
for (var column = 0u; column < ${columns}u; column++) { if (!(designScale[column] > 0.0)) { status[statusOffset] = 1u; return; } designScale[column] = inverseSqrt(designScale[column]); }
for (var first = 0u; first < ${instruments}u; first++) { for (var second = 0u; second < ${instruments}u; second++) { instrumentGram[first * ${instruments}u + second] *= instrumentScale[first] * instrumentScale[second]; } }
if (!choleskyFactor_${instruments}(&instrumentGram)) { status[statusOffset] = 1u; return; }
var projected: array<f32, ${instruments * columns}>;
for (var column = 0u; column < ${columns}u; column++) {
 var right: array<f32, ${instruments}>;
 for (var instrument = 0u; instrument < ${instruments}u; instrument++) { right[instrument] = crossGram[instrument * ${columns}u + column] * instrumentScale[instrument] * designScale[column]; }
 choleskySolve_${instruments}(&instrumentGram, &right);
 for (var instrument = 0u; instrument < ${instruments}u; instrument++) { projected[instrument * ${columns}u + column] = right[instrument]; }
}
var normal: array<f32, ${columns * columns}>;
for (var first = 0u; first < ${columns}u; first++) { for (var second = 0u; second < ${columns}u; second++) {
 var sum = 0.0; for (var instrument = 0u; instrument < ${instruments}u; instrument++) { sum += crossGram[instrument * ${columns}u + first] * instrumentScale[instrument] * designScale[first] * projected[instrument * ${columns}u + second]; }
 normal[first * ${columns}u + second] = sum;
} }
if (!choleskyFactor_${columns}(&normal)) { status[statusOffset] = 1u; return; }
for (var instrument = 0u; instrument < ${instruments}u; instrument++) { var right: array<f32, ${columns}>;
 for (var column = 0u; column < ${columns}u; column++) { right[column] = projected[instrument * ${columns}u + column]; }
 choleskySolve_${columns}(&normal, &right);
 for (var column = 0u; column < ${columns}u; column++) { projection[projectionOffset + column * ${instruments}u + instrument] = right[column] * designScale[column] * instrumentScale[instrument]; }
}`,
        `${getCholeskyWGSL(instruments)}\n${columns === instruments ? '' : getCholeskyWGSL(columns)}`
      ),
      node(
        'scores',
        rows,
        [
          binding('instruments', props.instruments),
          binding('means', means),
          binding('projection', projection),
          binding('residuals', props.residuals),
          binding('scores', scores, 'read_write')
        ],
        `let residual = residuals[residualsOffset + index]; var intercept = 1.0 / ${rows}.0;
for (var column = 0u; column < ${columns}u; column++) { var influence = 0.0;
for (var instrument = 0u; instrument < ${instruments}u; instrument++) { influence += projection[projectionOffset + column * ${instruments}u + instrument] * (instruments[instrumentsOffset + index * ${instruments}u + instrument] - means[meansOffset + ${columns}u + instrument]); }
intercept -= means[meansOffset + column] * influence;
scores[scoresOffset + index * ${coefficients}u + column + 1u] = residual * influence;
}
scores[scoresOffset + index * ${coefficients}u] = residual * intercept;`
      )
    ];
    if (props.kernel) {
      const kernel = props.kernel;
      nodes.push(
        node(
          'validate-kernel',
          rows,
          [
            binding('offsets', kernel.offsets, 'read', 'u32'),
            binding('neighbors', kernel.neighbors, 'read', 'u32'),
            binding('weights', kernel.weights),
            binding('kernelStatus', kernelStatus, 'read_write', 'u32'),
            ...(kernel.status?.overflow
              ? [binding('overflow', kernel.status.overflow, 'read', 'u32')]
              : [])
          ],
          `let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u]; var invalid = begin > end || end > ${kernel.neighbors.length}u ${kernel.status?.overflow ? '|| overflow[overflowOffset] != 0u' : ''}; var diagonal = 0.0;
if (!invalid) { for (var slot = begin; slot < end; slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
 if (neighbor >= ${rows}u || !isFiniteBits(weight) || weight < 0.0) { invalid = true; continue; }
 if (neighbor == index) { diagonal += weight; }
 let reverseBegin = offsets[offsetsOffset + neighbor]; let reverseEnd = offsets[offsetsOffset + neighbor + 1u];
 if (reverseBegin > reverseEnd || reverseEnd > ${kernel.neighbors.length}u) { invalid = true; continue; }
 var reverse = 0.0; var forward = 0.0;
 for (var other = reverseBegin; other < reverseEnd; other++) { if (neighbors[neighborsOffset + other] == index) { reverse += weights[weightsOffset + other]; } }
 for (var other = begin; other < end; other++) { if (neighbors[neighborsOffset + other] == neighbor) { forward += weights[weightsOffset + other]; } }
 invalid = invalid || abs(reverse - forward) > 1e-5 * max(1.0, abs(forward));
} }
invalid = invalid || abs(diagonal - 1.0) > 1e-5;
kernelStatus[kernelStatusOffset + index] = select(0u, 4u, invalid);`
        )
      );
    }
    nodes.push(
      node(
        'covariance',
        coefficients ** 2,
        [
          binding('scores', scores),
          ...(props.kernel
            ? [
                binding('offsets', props.kernel.offsets, 'read', 'u32'),
                binding('neighbors', props.kernel.neighbors, 'read', 'u32'),
                binding('weights', props.kernel.weights)
              ]
            : []),
          binding('covariance', covariance, 'read_write')
        ],
        `let first = index / ${coefficients}u; let second = index % ${coefficients}u; var sum = 0.0; var correction = 0.0;
for (var row = 0u; row < ${rows}u; row++) { var right = scores[scoresOffset + row * ${coefficients}u + second];
${
  props.kernel
    ? `right = 0.0;
let begin = offsets[offsetsOffset + row]; let end = offsets[offsetsOffset + row + 1u];
if (begin <= end && end <= ${props.kernel.neighbors.length}u) { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${rows}u) { right += weights[weightsOffset + slot] * scores[scoresOffset + neighbor * ${coefficients}u + second]; } } }`
    : ''
}
let term = scores[scoresOffset + row * ${coefficients}u + first] * right - correction;
let next = sum + term; correction = (next - sum) - term; sum = next; }
covariance[covarianceOffset + index] = sum ${props.method === 'HC1' ? `* ${rows}.0 / ${Math.max(rows - coefficients, 1)}.0` : ''};`
      )
    );
    nodes.push(
      node(
        'finish',
        1,
        [
          binding('candidate', covariance),
          binding('covariance', props.output.covariance, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32'),
          ...(props.kernel ? [binding('kernelStatus', kernelStatus, 'read', 'u32')] : [])
        ],
        `var code = status[statusOffset];
${props.kernel ? `for (var row = 0u; row < ${rows}u; row++) { if (kernelStatus[kernelStatusOffset + row] != 0u) { code = 4u; } }` : ''}
for (var slot = 0u; slot < ${coefficients ** 2}u; slot++) { if (code == 0u && !isFiniteBits(candidate[candidateOffset + slot])) { code = 3u; } }
for (var slot = 0u; slot < ${coefficients ** 2}u; slot++) { covariance[covarianceOffset + slot] = select(getNaN(), candidate[candidateOffset + slot], code == 0u); }
status[statusOffset] = code;`
      )
    );
    return nodes;
  }
}
