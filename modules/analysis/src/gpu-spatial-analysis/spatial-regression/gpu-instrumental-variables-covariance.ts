// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  getRegressionTileLayout,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from './regression-row-tiles';
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
 *
 * Every row sum is a row-tile reduction with a compensated, fixed-order tile merge (the pattern of
 * `GPUOrdinaryLeastSquares`): the column means, the centered moments H'H, H'Z and diag(Z'Z), and
 * the meat matrix. One invocation solves the small projection system from the moments. The HAC
 * kernel is applied row-parallel over its CSR neighbors (one invocation per row smooths the
 * scores, `O(nnz * p)`), never over all pairs. Rows are limited by float32 counts (`2^24`) and
 * by the design, instrument and score bindings of the device.
 */
export class GPUInstrumentalVariablesCovariance implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUInstrumentalVariablesCovarianceProps) {
    this.id = props.id ?? 'instrumental-variables-covariance';
    const rows = props.residuals.length;
    validateCapacity(this.id, 'rows', rows, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
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
    // The design or instrument view is the largest caller binding, the score rows the largest
    // transient one; both are float32 rows of at most 16 values.
    validateCapacity(
      id,
      'rows',
      rows,
      1,
      Math.min(
        REGRESSION_TILE_MAXIMUM_ROW_COUNT,
        getStorageBindingElementLimit(graph.device, 4 * Math.max(instruments, coefficients))
      )
    );
    const layout = getRegressionTileLayout(rows);
    const instrumentMomentCount = (instruments * (instruments + 1)) / 2;
    const momentCount = instrumentMomentCount + instruments * columns + columns;
    const meatCount = coefficients ** 2;
    // Float32 workspace: column means of [design, instruments] and a nonfinite flag, the centered
    // moments H'H (packed upper triangle), H'Z and diag(Z'Z), then the meat matrix and, with a
    // kernel, the invalid-kernel flag.
    const workspaceFlag = columns + instruments;
    const workspaceMoments = workspaceFlag + 1;
    const workspaceMeat = workspaceMoments + momentCount;
    const workspace = createTransientView(
      graph,
      `${id}-workspace`,
      'float32',
      workspaceMeat + meatCount + 1
    );
    const projection = createTransientView(
      graph,
      `${id}-projection`,
      'float32',
      columns * instruments
    );
    const scores = createTransientView(graph, `${id}-scores`, 'float32', rows * coefficients);
    const kernelStatus = createTransientView(graph, `${id}-kernel-status`, 'uint32', rows);
    const smoothedScores = props.kernel
      ? createTransientView(graph, `${id}-smoothed-scores`, 'float32', rows * coefficients)
      : undefined;
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
    const rowWGSL = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn getColumn(row: u32, column: u32) -> f32 {
  if (column < ${columns}u) { return design[designOffset + row * ${columns}u + column]; }
  return instruments[instrumentsOffset + row * ${instruments}u + column - ${columns}u];
}`;
    const nodes = [
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-sums`,
        operation: 'GPUInstrumentalVariablesCovariance',
        variant: 'sums',
        rowCount: rows,
        layout,
        sumCount: columns + instruments,
        maximumCount: 1,
        bindings: [binding('design', props.design), binding('instruments', props.instruments)],
        declarations: `${rowWGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var column = 0u; column < SUM_COUNT; column++) {
    let value = getColumn(row, column);
    (*terms)[column] += value;
    if (!isFiniteBits(value)) { (*terms)[SUM_COUNT] = 1.0; }
  }
}`,
        partial: createTransientView(
          graph,
          `${id}-sum-partials`,
          'float32',
          layout.tileCount * (workspaceFlag + 1)
        ),
        output: workspace
      }),
      // Column sums become means in place, with the row count as a literal (as before the tiles).
      node(
        'means',
        columns + instruments,
        [binding('workspace', workspace, 'read_write')],
        `workspace[workspaceOffset + index] = workspace[workspaceOffset + index] / ${rows}.0;`
      ),
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-moments`,
        operation: 'GPUInstrumentalVariablesCovariance',
        variant: 'moments',
        rowCount: rows,
        layout,
        sumCount: momentCount,
        bindings: [
          binding('design', props.design),
          binding('instruments', props.instruments),
          binding('workspace', workspace)
        ],
        declarations: `${rowWGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  var centered: array<f32, ${columns + instruments}>;
  for (var column = 0u; column < ${columns + instruments}u; column++) {
    centered[column] = getColumn(row, column) - workspace[workspaceOffset + column];
  }
  var slot = 0u;
  for (var first = 0u; first < ${instruments}u; first++) {
    for (var second = first; second < ${instruments}u; second++) {
      (*terms)[slot] += centered[${columns}u + first] * centered[${columns}u + second];
      slot++;
    }
  }
  for (var first = 0u; first < ${instruments}u; first++) {
    for (var column = 0u; column < ${columns}u; column++) {
      (*terms)[slot] += centered[${columns}u + first] * centered[column];
      slot++;
    }
  }
  for (var column = 0u; column < ${columns}u; column++) {
    (*terms)[slot] += centered[column] * centered[column];
    slot++;
  }
}`,
        partial: createTransientView(
          graph,
          `${id}-moment-partials`,
          'float32',
          layout.tileCount * momentCount
        ),
        output: workspace,
        outputSlot: workspaceMoments
      }),
      node(
        'projection',
        1,
        [
          binding('workspace', workspace),
          binding('projection', projection, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        // A small dense solve over at most 16 instruments: the row sums come from the tiles above.
        `status[statusOffset] = 0u;
for (var slot = 0u; slot < ${columns * instruments}u; slot++) { projection[projectionOffset + slot] = getNaN(); }
if (${rows}u <= ${coefficients}u) { status[statusOffset] = 2u; return; }
var nonfinite = workspace[workspaceOffset + ${workspaceFlag}u] != 0.0;
for (var column = 0u; column < ${columns + instruments}u; column++) { nonfinite = nonfinite || !isFiniteBits(workspace[workspaceOffset + column]); }
if (nonfinite) { status[statusOffset] = 3u; return; }
var instrumentGram: array<f32, ${instruments * instruments}>;
var crossGram: array<f32, ${instruments * columns}>;
var designScale: array<f32, ${columns}>;
var slot = ${workspaceMoments}u;
for (var first = 0u; first < ${instruments}u; first++) { for (var second = first; second < ${instruments}u; second++) {
 let value = workspace[workspaceOffset + slot]; slot++;
 instrumentGram[first * ${instruments}u + second] = value; instrumentGram[second * ${instruments}u + first] = value;
} }
for (var first = 0u; first < ${instruments}u; first++) { for (var column = 0u; column < ${columns}u; column++) { crossGram[first * ${columns}u + column] = workspace[workspaceOffset + slot]; slot++; } }
for (var column = 0u; column < ${columns}u; column++) { designScale[column] = workspace[workspaceOffset + slot]; slot++; }
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
          binding('workspace', workspace),
          binding('projection', projection),
          binding('residuals', props.residuals),
          binding('scores', scores, 'read_write')
        ],
        `let residual = residuals[residualsOffset + index]; var intercept = 1.0 / ${rows}.0;
for (var column = 0u; column < ${columns}u; column++) { var influence = 0.0;
for (var instrument = 0u; instrument < ${instruments}u; instrument++) { influence += projection[projectionOffset + column * ${instruments}u + instrument] * (instruments[instrumentsOffset + index * ${instruments}u + instrument] - workspace[workspaceOffset + ${columns}u + instrument]); }
intercept -= workspace[workspaceOffset + column] * influence;
scores[scoresOffset + index * ${coefficients}u + column + 1u] = residual * influence;
}
scores[scoresOffset + index * ${coefficients}u] = residual * intercept;`
      )
    ];
    if (props.kernel && smoothedScores) {
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
        ),
        // One invocation per row smooths the scores over its kernel neighbors: O(nnz * p) work.
        node(
          'smooth-scores',
          rows,
          [
            binding('scores', scores),
            binding('offsets', kernel.offsets, 'read', 'u32'),
            binding('neighbors', kernel.neighbors, 'read', 'u32'),
            binding('weights', kernel.weights),
            binding('smoothed', smoothedScores, 'read_write')
          ],
          `var right: array<f32, ${coefficients}>;
let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (begin <= end && end <= ${kernel.neighbors.length}u) { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; if (neighbor < ${rows}u) {
 let weight = weights[weightsOffset + slot];
 for (var second = 0u; second < ${coefficients}u; second++) { right[second] += weight * scores[scoresOffset + neighbor * ${coefficients}u + second]; } } } }
for (var second = 0u; second < ${coefficients}u; second++) { smoothed[smoothedOffset + index * ${coefficients}u + second] = right[second]; }`
        )
      );
    }
    nodes.push(
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-meat`,
        operation: 'GPUInstrumentalVariablesCovariance',
        variant: 'meat',
        rowCount: rows,
        layout,
        sumCount: meatCount,
        maximumCount: props.kernel ? 1 : 0,
        bindings: [
          binding('scores', scores),
          ...(smoothedScores
            ? [
                binding('smoothed', smoothedScores),
                binding('kernelStatus', kernelStatus, 'read', 'u32')
              ]
            : [])
        ],
        // HC: sum of s_i s_i'. HAC: sum of s_i (K s)_i', with the smoothed scores above.
        declarations: `${IS_FINITE_BITS_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  for (var first = 0u; first < ${coefficients}u; first++) {
    let left = scores[scoresOffset + row * ${coefficients}u + first];
    for (var second = 0u; second < ${coefficients}u; second++) {
      (*terms)[first * ${coefficients}u + second] += left * ${smoothedScores ? 'smoothed[smoothedOffset' : 'scores[scoresOffset'} + row * ${coefficients}u + second];
    }
  }
  ${smoothedScores ? `if (kernelStatus[kernelStatusOffset + row] != 0u) { (*terms)[SUM_COUNT] = 1.0; }` : ''}
}`,
        partial: createTransientView(
          graph,
          `${id}-meat-partials`,
          'float32',
          layout.tileCount * (meatCount + (props.kernel ? 1 : 0))
        ),
        output: workspace,
        outputSlot: workspaceMeat
      }),
      node(
        'finish',
        1,
        [
          binding('workspace', workspace),
          binding('covariance', props.output.covariance, 'read_write'),
          binding('status', props.output.status, 'read_write', 'u32')
        ],
        `var code = status[statusOffset];
${props.kernel ? `if (workspace[workspaceOffset + ${workspaceMeat + meatCount}u] != 0.0) { code = 4u; }` : ''}
let scale = ${props.method === 'HC1' ? `${rows}.0 / ${Math.max(rows - coefficients, 1)}.0` : '1.0'};
for (var slot = 0u; slot < ${meatCount}u; slot++) { if (code == 0u && !isFiniteBits(workspace[workspaceOffset + ${workspaceMeat}u + slot] * scale)) { code = 3u; } }
for (var slot = 0u; slot < ${meatCount}u; slot++) { covariance[covarianceOffset + slot] = select(getNaN(), workspace[workspaceOffset + ${workspaceMeat}u + slot] * scale, code == 0u); }
status[statusOffset] = code;`
      )
    );
    return nodes;
  }
}
