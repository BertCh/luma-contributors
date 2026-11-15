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
  validateRequirement,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/inference';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  getInstrumentalVariablesCovarianceStatusView,
  getInstrumentalVariablesCovarianceCommandNodeEstimate,
  validateInstrumentalVariablesCovarianceOptions,
  INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_COEFFICIENT_COUNT,
  INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_INSTRUMENT_COUNT,
  INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE,
  INSTRUMENTAL_VARIABLES_COVARIANCE_PSD_TOLERANCE,
  type GPUInstrumentalVariablesCovarianceMethod
} from './instrumental-variables-covariance-parameters';
export {GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS} from './instrumental-variables-covariance-parameters';
export type {GPUInstrumentalVariablesCovarianceMethod} from './instrumental-variables-covariance-parameters';
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
  method?: GPUInstrumentalVariablesCovarianceMethod;
  /** One key per estimation row in [0, clusterCount); no missing keys or automatic deletion. */
  clusters?: GraphDataView<'uint32'>;
  /** Capacity for dense cluster keys, at most the row count. Empty slots do not count in CR1. */
  clusterCount?: number;
  /** HAC kernel, distinct from the model's spatial-lag weights. Symmetric, unit diagonal, PSD. */
  kernel?: GPUSpatialWeights;
  output: {
    /** Full row-major (coefficientCount + 1)² covariance, intercept first. */
    covariance: GraphDataView<'float32'>;
    /** Named solver port; legacy scalar accepted. See GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS.
     * Failures publish NaN covariance: 5 invalid leverage, 6 invalid clusters, 7 non-PSD covariance. */
    status: Pick<GPUSolverStatusPort, 'code'> | GraphDataView<'uint32'>;
  };
};

/**
 * Centered and scaled IV sandwich covariance: A H' diag(u) K diag(u) H A', where
 * A = (Z'P_H Z)^-1 Z'H(H'H)^-1. HC0 uses identity K; HC1 multiplies by n/(n-p),
 * p including the intercept. HAC has no finite-sample correction. The kernel's diagonal,
 * symmetry, bounds and nonnegative weights are checked on GPU. The returned covariance (not the
 * entire HAC kernel) is certified PSD to relative tolerance 1e-5 using a shifted Cholesky test.
 * A zero or rank-deficient covariance is accepted; a failed certificate returns code 7 and NaNs.
 * HC2/HC3 divide each structural residual by sqrt(1-h) / (1-h), respectively, where h is the
 * diagonal of the orthogonal hat matrix of the projected design P_H Z, including the intercept.
 * A denominator <= 1e-6, or h < -1e-6, gives code 5; leverage is never silently clipped.
 * CLUSTER sums influence scores by cluster and uses CR1: G/(G-1) * (n-1)/(n-p), with occupied
 * groups G (at least two) and p including the intercept. Invalid/missing keys give code 6.
 * Compile-time: method, shapes, optional views and clusterCount. Per-frame: all buffer contents.
 * Inputs contain only estimation rows, with no automatic deletion/imputation. This is a covariance
 * stage, never a new estimator or an OLS sandwich applied incorrectly to endogenous regressors.
 *
 * Every row sum is a row-tile reduction with a fixed-order pairwise tile merge (the pattern of
 * `GPUOrdinaryLeastSquares`): the column means, the centered moments H'H, H'Z and diag(Z'Z), and
 * the meat matrix. One invocation solves the small projection system from the moments. The HAC
 * kernel is applied row-parallel over its CSR neighbors (one invocation per row smooths the
 * scores, `O(nnz * p)`), never over all pairs. Rows are limited by float32 counts (`2^24`) and
 * by the design, instrument and score bindings of the device.
 */
export class GPUInstrumentalVariablesCovariance implements GPUCommandNodeProducer {
  readonly id: string;
  readonly status: GraphDataView<'uint32'>;
  /** Includes a conservative allowance for the published radix sort and per-column gathers. */
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUInstrumentalVariablesCovarianceProps) {
    this.id = props.id ?? 'instrumental-variables-covariance';
    this.status = getInstrumentalVariablesCovarianceStatusView(props.output.status);
    const rows = props.residuals.length;
    validateCapacity(this.id, 'rows', rows, 1, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    validateCapacity(
      this.id,
      'coefficientCount',
      props.coefficientCount,
      1,
      INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_COEFFICIENT_COUNT
    );
    validateCapacity(
      this.id,
      'instrumentCount',
      props.instrumentCount,
      props.coefficientCount,
      INSTRUMENTAL_VARIABLES_COVARIANCE_MAXIMUM_INSTRUMENT_COUNT
    );
    validateInstrumentalVariablesCovarianceOptions(this.id, rows, props);
    if (props.kernel)
      validateRequirement(
        this.id,
        validateGPUSpatialWeights(this.id, props.kernel) === rows,
        'kernel row count must equal residuals.length'
      );
    // 256 nodes cover the published radix-sort passes for <=24-bit dense keys, the sort
    // preparation and offset search, and the fixed covariance stages. Each score/count column
    // adds an extraction, gather, segment reduction and pack (the count replaces its pack by
    // two occupied-group reduction nodes). The node spec checks this bound at maximum shape.
    this.estimatedCommandNodeCount = getInstrumentalVariablesCovarianceCommandNodeEstimate(
      props.coefficientCount,
      Boolean(props.clusters)
    );
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    for (const [name, view, format, length] of [
      ['design', props.design, 'float32', rows * props.coefficientCount],
      ['instruments', props.instruments, 'float32', rows * props.instrumentCount],
      ['residuals', props.residuals, 'float32', rows],
      ['output.covariance', props.output.covariance, 'float32', (props.coefficientCount + 1) ** 2],
      ['output.status.code', this.status, 'uint32', 1],
      ...(props.clusters ? [['clusters', props.clusters, 'uint32', rows] as const] : [])
    ] as const) {
      validateSinglePackedView(view, [format], this.id);
      validateRequirement(this.id, view.length === length, `${name} length must equal ${length}`);
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.output.covariance, this.status],
      [
        props.design,
        props.instruments,
        props.residuals,
        props.clusters,
        props.kernel?.offsets,
        props.kernel?.neighbors,
        props.kernel?.weights,
        props.kernel?.status?.overflow
      ]
    );
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
      workspaceMeat + meatCount + 3
    );
    const projection = createTransientView(
      graph,
      `${id}-projection`,
      'float32',
      2 * columns * instruments
    );
    const scores = createTransientView(graph, `${id}-scores`, 'float32', rows * coefficients);
    const rowStatus = createTransientView(graph, `${id}-row-status`, 'float32', rows);
    const needsLeverage = props.method === 'HC2' || props.method === 'HC3';
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
      props.clusters,
      props.kernel?.offsets,
      props.kernel?.neighbors,
      props.kernel?.weights,
      props.kernel?.status?.overflow,
      props.output.covariance,
      this.status
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
          binding('status', this.status, 'read_write', 'u32')
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
 for (var instrument = 0u; instrument < ${instruments}u; instrument++) { projected[instrument * ${columns}u + column] = right[instrument];
 projection[projectionOffset + ${columns * instruments}u + column * ${instruments}u + instrument] = right[instrument] * instrumentScale[instrument] / designScale[column]; }
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
        `${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(instruments)}\n${columns === instruments ? '' : getCholeskyWGSL(columns)}`
      ),
      node(
        'scores',
        rows,
        [
          binding('instruments', props.instruments),
          binding('workspace', workspace),
          binding('projection', projection),
          binding('residuals', props.residuals),
          binding('scores', scores, 'read_write'),
          binding('rowStatus', rowStatus, 'read_write'),
          ...(props.clusters ? [binding('clusters', props.clusters, 'read', 'u32')] : [])
        ],
        `var residual = residuals[residualsOffset + index]; var intercept = 1.0 / ${rows}.0;
var influences: array<f32, ${columns}>;
var leverage = 1.0 / ${rows}.0;
for (var column = 0u; column < ${columns}u; column++) { var influence = 0.0;
 var fitted = 0.0;
 for (var instrument = 0u; instrument < ${instruments}u; instrument++) {
  let centered = instruments[instrumentsOffset + index * ${instruments}u + instrument] - workspace[workspaceOffset + ${columns}u + instrument];
  influence += projection[projectionOffset + column * ${instruments}u + instrument] * centered;
  ${needsLeverage ? `fitted += projection[projectionOffset + ${columns * instruments}u + column * ${instruments}u + instrument] * centered;` : ''}
 }
 influences[column] = influence;
 leverage += fitted * influence;
 intercept -= workspace[workspaceOffset + column] * influence;
}
var code = 0.0;
${
  needsLeverage
    ? `let denominator = 1.0 - leverage;
if (!isFiniteBits(leverage) || leverage < -${INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE} || denominator <= ${INSTRUMENTAL_VARIABLES_COVARIANCE_LEVERAGE_TOLERANCE}) { code = 5.0; }
residual /= ${props.method === 'HC2' ? 'sqrt(denominator)' : 'denominator'};`
    : ''
}
${props.clusters ? `if (clusters[clustersOffset + index] >= ${props.clusterCount}u) { code = 6.0; }` : ''}
rowStatus[rowStatusOffset + index] = code;
for (var column = 0u; column < ${columns}u; column++) { scores[scoresOffset + index * ${coefficients}u + column + 1u] = residual * influences[column]; }
scores[scoresOffset + index * ${coefficients}u] = residual * intercept;`
      )
    ];
    nodes.push(
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-row-status`,
        operation: 'GPUInstrumentalVariablesCovariance',
        variant: 'row-status',
        rowCount: rows,
        layout,
        sumCount: 0,
        maximumCount: 1,
        bindings: [binding('rowStatus', rowStatus)],
        declarations: `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
        (*terms)[0] = max((*terms)[0], rowStatus[rowStatusOffset + row]);
      }`,
        partial: createTransientView(
          graph,
          `${id}-row-status-partials`,
          'float32',
          layout.tileCount
        ),
        output: workspace,
        outputSlot: workspaceMeat + meatCount + 1
      })
    );
    let meatScores = scores;
    let meatRows = rows;
    if (props.clusters) {
      meatRows = props.clusterCount!;
      meatScores = createTransientView(
        graph,
        `${id}-cluster-scores`,
        'float32',
        meatRows * coefficients
      );
      const contributions = Array.from({length: coefficients + 1}, (_, column) =>
        createTransientView(graph, `${id}-contribution-${column}`, 'float32', rows)
      );
      const totals = contributions.map((_, column) =>
        createTransientView(graph, `${id}-cluster-total-${column}`, 'float32', meatRows)
      );
      for (let column = 0; column <= coefficients; column++) {
        nodes.push(
          node(
            `cluster-contribution-${column}`,
            rows,
            [
              binding('scores', scores),
              binding('contribution', contributions[column], 'read_write')
            ],
            `contribution[contributionOffset + index] = ${column === coefficients ? '1.0' : `scores[scoresOffset + index * ${coefficients}u + ${column}u]`};`
          )
        );
      }
      nodes.push(
        ...getSortedSegmentSumNodes(graph, {
          id: `${id}-clusters`,
          operation: 'GPUInstrumentalVariablesCovariance',
          segmentCount: meatRows,
          segmentKeys: props.clusters,
          reductions: contributions.map((contribution, column) => ({
            name: `score-${column}`,
            contributions: contribution,
            output: totals[column]
          }))
        })
      );
      for (let column = 0; column < coefficients; column++) {
        nodes.push(
          node(
            `cluster-pack-${column}`,
            meatRows,
            [binding('total', totals[column]), binding('scores', meatScores, 'read_write')],
            `scores[scoresOffset + index * ${coefficients}u + ${column}u] = total[totalOffset + index];`
          )
        );
      }
      const clusterLayout = getRegressionTileLayout(meatRows);
      nodes.push(
        ...getRowTileSumNodes<Parameters>(graph, {
          id: `${id}-occupied-clusters`,
          operation: 'GPUInstrumentalVariablesCovariance',
          variant: 'occupied-clusters',
          rowCount: meatRows,
          layout: clusterLayout,
          sumCount: 1,
          bindings: [binding('counts', totals[coefficients])],
          declarations: `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
          (*terms)[0] += select(0.0, 1.0, counts[countsOffset + row] > 0.0);
        }`,
          partial: createTransientView(
            graph,
            `${id}-occupied-partials`,
            'float32',
            clusterLayout.tileCount
          ),
          output: workspace,
          outputSlot: workspaceMeat + meatCount + 2
        })
      );
    }
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
        rowCount: meatRows,
        layout: getRegressionTileLayout(meatRows),
        sumCount: meatCount,
        maximumCount: props.kernel ? 1 : 0,
        bindings: [
          binding('scores', meatScores),
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
          getRegressionTileLayout(meatRows).tileCount * (meatCount + (props.kernel ? 1 : 0))
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
          binding('status', this.status, 'read_write', 'u32')
        ],
        `var code = status[statusOffset];
${props.kernel ? `if (workspace[workspaceOffset + ${workspaceMeat + meatCount}u] != 0.0) { code = 4u; }` : ''}
if (code == 0u) { code = u32(workspace[workspaceOffset + ${workspaceMeat + meatCount + 1}u]); }
${
  props.clusters
    ? `let groups = workspace[workspaceOffset + ${workspaceMeat + meatCount + 2}u];
if (code == 0u && groups < 2.0) { code = 6u; }`
    : ''
}
let scale = ${props.method === 'HC1' ? `${rows}.0 / ${Math.max(rows - coefficients, 1)}.0` : props.clusters ? `(groups / max(groups - 1.0, 1.0)) * ${rows - 1}.0 / ${Math.max(rows - coefficients, 1)}.0` : '1.0'};
var matrix: array<f32, ${meatCount}>;
var magnitude = 0.0;
for (var first = 0u; first < ${coefficients}u; first++) { for (var second = 0u; second < ${coefficients}u; second++) {
 let value = (0.5 * workspace[workspaceOffset + ${workspaceMeat}u + first * ${coefficients}u + second] + 0.5 * workspace[workspaceOffset + ${workspaceMeat}u + second * ${coefficients}u + first]) * scale;
 matrix[first * ${coefficients}u + second] = value;
 if (code == 0u && !isFiniteBits(value)) { code = 3u; }
 magnitude = max(magnitude, abs(value));
} }
if (code == 0u && magnitude > 0.0) {
 var shifted = matrix;
 for (var first = 0u; first < ${coefficients}u; first++) { for (var second = 0u; second < ${coefficients}u; second++) {
   shifted[first * ${coefficients}u + second] /= magnitude;
 } shifted[first * ${coefficients + 1}u] += ${INSTRUMENTAL_VARIABLES_COVARIANCE_PSD_TOLERANCE}; }
 if (!choleskyFactor_${coefficients}(&shifted)) { code = 7u; }
}
for (var slot = 0u; slot < ${meatCount}u; slot++) { covariance[covarianceOffset + slot] = select(getNaN(), matrix[slot], code == 0u); }
status[statusOffset] = code;`,
        `${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(coefficients)}`
      )
    );
    return nodes;
  }
}
