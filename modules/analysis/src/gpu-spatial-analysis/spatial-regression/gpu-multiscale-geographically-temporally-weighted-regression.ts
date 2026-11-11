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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validatePredictionViews} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getLocalKernelWeightWGSL} from './local-weighted-least-squares-wgsl';
import {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS} from './geographically-weighted-regression-parameters';
import {
  createSpaceTimePointNodes,
  GPUGeographicallyTemporallyWeightedRegression,
  GPU_GTWR_MAXIMUM_PREDICTOR_COUNT,
  GPU_GTWR_MAXIMUM_ROW_COUNT,
  type GPUGeographicallyTemporallyWeightedRegressionProps,
  type GPUGTWRSettings
} from './gpu-geographically-temporally-weighted-regression';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';

/** Most backfitting iterations; the loop is unrolled into command nodes. */
export const GPU_MGTWR_MAXIMUM_ITERATION_COUNT = 32;

/** Per-frame settings packed by {@link getGPUMGTWRParameterValues}. */
export type GPUMGTWRSettings = {
  /** Bandwidth and time scale of the single GTWR fit that starts the backfitting. */
  initial: GPUGTWRSettings;
  /** Bandwidth and time scale of each component: the intercept first, then each predictor. */
  components: readonly GPUGTWRSettings[];
};

/**
 * Packs settings into the float32 layout of {@link GPUMultiscaleGeographicallyTemporallyWeightedRegression}:
 * `[initialBandwidth, initialTimeScale, bandwidth_0, timeScale_0, ..., bandwidth_k, timeScale_k]`,
 * `2 + 2 (predictorCount + 1)` values.
 *
 * @throws If a bandwidth is not a positive finite number or a time scale is negative or not finite.
 */
export function getGPUMGTWRParameterValues(settings: GPUMGTWRSettings): Float32Array {
  const values = new Float32Array(2 + 2 * settings.components.length);
  [settings.initial, ...settings.components].forEach((entry, index) => {
    const bandwidth = Math.fround(entry.bandwidth);
    const timeScale = Math.fround(entry.timeScale);
    if (!(Number.isFinite(bandwidth) && bandwidth > 0)) {
      throw new Error('getGPUMGTWRParameterValues bandwidth must be a positive finite number');
    }
    if (!(Number.isFinite(timeScale) && timeScale >= 0)) {
      throw new Error('getGPUMGTWRParameterValues timeScale must be a non-negative finite number');
    }
    values[2 * index] = bandwidth;
    values[2 * index + 1] = timeScale;
  });
  return values;
}

/** Caller-owned outputs of {@link GPUMultiscaleGeographicallyTemporallyWeightedRegression}. */
export type GPUMultiscaleGeographicallyTemporallyWeightedRegressionOutput = {
  /**
   * Coefficients of the last iteration, row-major `row * (predictorCount + 1) + column`, the
   * intercept first. NaN where a component smoother was singular, and for excluded rows.
   */
  coefficients: GraphDataView<'float32'>;
  /** Fitted value `sum_j beta_j x_j` per row; NaN for excluded rows. */
  fitted: GraphDataView<'float32'>;
  /** Residual `y - fitted` per row; NaN for excluded rows. */
  residuals: GraphDataView<'float32'>;
  /** Per-row status: `OK`, `SINGULAR` (a coefficient is NaN) or `EXCLUDED`. */
  localStatus: GraphDataView<'uint32'>;
  /**
   * Convergence score of each iteration, `sqrt(sum (XB_new - XB_old)^2 / n / sum (sum_j XB_new)^2)`
   * (mgtwr `multi_bws` with `rss_score` false): a series falling toward zero means the
   * backfitting converged.
   */
  scores: GraphDataView<'float32'>;
};

/** Properties for {@link GPUMultiscaleGeographicallyTemporallyWeightedRegression}. */
export type GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps = Omit<
  GPUGeographicallyTemporallyWeightedRegressionProps,
  'output' | 'parameters'
> & {
  /** Backfitting iterations, 1 to {@link GPU_MGTWR_MAXIMUM_ITERATION_COUNT}; fixed at compile time. */
  iterationCount: number;
  /**
   * `2 + 2 (predictorCount + 1)` float32 values from {@link getGPUMGTWRParameterValues}.
   * Per-frame.
   */
  parameters: GraphDataView<'float32'>;
  output: GPUMultiscaleGeographicallyTemporallyWeightedRegressionOutput;
};

const OPERATION = 'GPUMultiscaleGeographicallyTemporallyWeightedRegression';
const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;

/**
 * Multiscale GTWR by backfitting (the algorithm of mgtwr `multi_bws`, Wu et al. 2019 on Fotheringham
 * et al. 2017). A GTWR fit at the initial `(bandwidth, timeScale)` starts the components
 * `XB_j = beta_j x_j`. Each iteration then visits the components in order: component `j` is
 * smoothed against the partial residual `XB_j + err` by a one-covariate local regression with its
 * own bandwidth and time scale, `beta_ij = sum_l w_il x_lj r_l / sum_l w_il x_lj^2` (no local
 * intercept: the intercept is component 0), and the residual `err` is updated before the next
 * component (Gauss-Seidel order, as in mgtwr).
 *
 * The bandwidths are the caller's; mgtwr's search for them (a golden-section search per
 * component) is not ported, so profile them by rewriting `parameters` and comparing `scores` or
 * the residuals. The iteration count is compile-time and every iteration runs: mgtwr stops at its
 * tolerance, here the `scores` show where that would have happened. A row whose initial local fit
 * is singular starts from zero components. A component smoother whose weighted `x^2` sum is not
 * positive gives a NaN coefficient and contributes nothing, which keeps the rest of the fit finite.
 * Exhaustive over rows, float32, `O(iterations x predictors x rows^2)`.
 */
export class GPUMultiscaleGeographicallyTemporallyWeightedRegression
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly rows: number;
  constructor(readonly props: GPUMultiscaleGeographicallyTemporallyWeightedRegressionProps) {
    this.id = props.id ?? 'multiscale-geographically-temporally-weighted-regression';
    const {id} = this;
    const columns = props.predictorCount;
    validateCapacity(id, 'predictorCount', columns, 1, GPU_GTWR_MAXIMUM_PREDICTOR_COUNT);
    validateCapacity(
      id,
      'iterationCount',
      props.iterationCount,
      1,
      GPU_MGTWR_MAXIMUM_ITERATION_COUNT
    );
    resolveGPUTemporalContext(id, props.temporalContext, {encodings: ['float32-relative']});
    validateSinglePackedView(props.response, ['float32'], `${id} response`);
    this.rows = props.response.length;
    validateCapacity(id, 'rows', this.rows, 2, GPU_GTWR_MAXIMUM_ROW_COUNT);
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateViewLength(id, 'positions', props.positions, this.rows);
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateViewLength(id, 'times', props.times, this.rows);
    validateSinglePackedView(props.predictors, ['float32'], `${id} predictors`);
    validateViewLength(id, 'predictors', props.predictors, this.rows * columns);
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', props.mask, this.rows);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < 2 + 2 * (columns + 1)) {
      throw new Error(`${id} parameters must have ${2 + 2 * (columns + 1)} values`);
    }
    const {output} = props;
    for (const [name, view, format, length] of [
      ['coefficients', output.coefficients, 'float32', this.rows * (columns + 1)],
      ['fitted', output.fitted, 'float32', this.rows],
      ['residuals', output.residuals, 'float32', this.rows],
      ['localStatus', output.localStatus, 'uint32', this.rows],
      ['scores', output.scores, 'float32', props.iterationCount]
    ] as const) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
    validatePredictionViews(id, this.getInputViews(), Object.values(output));
  }

  private getInputViews(): GraphDataView[] {
    const {props} = this;
    const views: (GraphDataView | undefined)[] = [
      props.positions,
      props.times,
      props.predictors,
      props.response,
      props.mask,
      props.parameters
    ];
    return views.filter((view): view is GraphDataView => view !== undefined);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows} = this;
    const {output} = props;
    const columns = props.predictorCount;
    const size = columns + 1;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...Object.values(output)]);
    const transient = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView<Format, Parameters>(graph, `${id}-${name}`, format, length);
    const initial = {
      coefficients: transient('initial-coefficients', 'float32', rows * size),
      localStatus: transient('initial-status', 'uint32', rows),
      fitted: transient('initial-fitted', 'float32', rows),
      hatDiagonal: transient('initial-hat', 'float32', rows),
      residuals: transient('initial-residuals', 'float32', rows),
      summary: transient('initial-summary', 'float32', 12)
    };
    const spaceTime = createSpaceTimePointNodes(graph, id, props, rows);
    const components = [
      transient('components-a', 'float32', rows * size),
      transient('components-b', 'float32', rows * size)
    ];
    const errors = [transient('errors-a', 'float32', rows), transient('errors-b', 'float32', rows)];
    // Design column 0 is the intercept; column c is predictor c - 1.
    const designWGSL = `fn getDesign(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return predictors[predictorsOffset + row * ${columns}u + column - 1u];
}`;
    const weightWGSL = getLocalKernelWeightWGSL();
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUGeographicallyTemporallyWeightedRegression({
        id: `${id}-initial`,
        positions: props.positions,
        times: props.times,
        temporalContext: props.temporalContext,
        predictors: props.predictors,
        predictorCount: columns,
        response: props.response,
        mask: props.mask,
        kernel: props.kernel,
        parameters: props.parameters,
        output: initial
      }).getCommandNodes(graph),
      ...spaceTime.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-start`,
        operation: OPERATION,
        variant: 'initial-components',
        invocationCount: rows,
        bindings: [
          {name: 'points', view: spaceTime.points, type: 'f32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'initialCoefficients', view: initial.coefficients, type: 'f32', access: 'read'},
          {name: 'components', view: components[0], type: 'f32', access: 'read_write'},
          {name: 'errors', view: errors[0], type: 'f32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}
${designWGSL}`,
        // Components are beta_j x_j of the initial fit. A singular row starts from zero components
        // and the whole response as its error; an excluded row holds zeros.
        body: `let included = isFiniteBits(points[pointsOffset + 3u * index]);
  var total = 0.0;
  var finite = included;
  for (var column = 0u; column < ${size}u; column++) {
    let design = getDesign(index, column);
    let value = initialCoefficients[initialCoefficientsOffset + index * ${size}u + column] * design;
    finite = finite && isFiniteBits(value);
  }
  for (var column = 0u; column < ${size}u; column++) {
    let design = getDesign(index, column);
    let value = initialCoefficients[initialCoefficientsOffset + index * ${size}u + column] * design;
    let component = select(0.0, value, finite);
    components[componentsOffset + index * ${size}u + column] = component;
    total += component;
  }
  errors[errorsOffset + index] = select(0.0, response[responseOffset + index] - total, included);`
      })
    ];
    for (let iteration = 0; iteration < props.iterationCount; iteration++) {
      const previous = components[iteration % 2];
      const next = components[(iteration + 1) % 2];
      for (let component = 0; component < size; component++) {
        const step = iteration * size + component;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-component-${component}`,
            operation: OPERATION,
            variant: 'backfit-component',
            invocationCount: rows,
            bindings: [
              {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
              {name: 'points', view: spaceTime.points, type: 'f32', access: 'read'},
              {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
              {name: 'previous', view: previous, type: 'f32', access: 'read'},
              {name: 'next', view: next, type: 'f32', access: 'read_write'},
              {name: 'errorsIn', view: errors[step % 2], type: 'f32', access: 'read'},
              {name: 'errorsOut', view: errors[(step + 1) % 2], type: 'f32', access: 'read_write'},
              {
                name: 'coefficients',
                view: output.coefficients,
                type: 'f32',
                access: 'read_write'
              }
            ],
            declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}\n${weightWGSL}`,
            body: `let bandwidth = parameters[parametersOffset + ${2 + 2 * component}u];
  let timeScale = parameters[parametersOffset + ${3 + 2 * component}u];
  let originX = points[pointsOffset + 3u * index];
  if (!isFiniteBits(originX)) {
    next[nextOffset + index * ${size}u + ${component}u] = 0.0;
    errorsOut[errorsOutOffset + index] = 0.0;
    coefficients[coefficientsOffset + index * ${size}u + ${component}u] = getNaN();
  } else {
    let originY = points[pointsOffset + 3u * index + 1u];
    let originTime = points[pointsOffset + 3u * index + 2u];
    var numerator = 0.0;
    var denominator = 0.0;
    for (var row = 0u; row < ${rows}u; row++) {
      let rowX = points[pointsOffset + 3u * row];
      if (!isFiniteBits(rowX)) { continue; }
      let deltaX = rowX - originX;
      let deltaY = points[pointsOffset + 3u * row + 1u] - originY;
      let deltaTime = points[pointsOffset + 3u * row + 2u] - originTime;
      let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
      let weight = getLocalKernelWeight(distance, bandwidth, ${props.kernel === 'bisquare' ? 'true' : 'false'});
      if (!(weight > 0.0)) { continue; }
      ${
        component === 0
          ? 'let design = 1.0;'
          : `let design = predictors[predictorsOffset + row * ${columns}u + ${component - 1}u];`
      }
      let partial = previous[previousOffset + row * ${size}u + ${component}u] + errorsIn[errorsInOffset + row];
      numerator += weight * design * partial;
      denominator += weight * design * design;
    }
    ${
      component === 0
        ? 'let ownDesign = 1.0;'
        : `let ownDesign = predictors[predictorsOffset + index * ${columns}u + ${component - 1}u];`
    }
    let ownPartial = previous[previousOffset + index * ${size}u + ${component}u] + errorsIn[errorsInOffset + index];
    let ok = denominator > 0.0 && isFiniteBits(numerator) && isFiniteBits(denominator);
    let beta = select(getNaN(), numerator / denominator, ok);
    let smoothed = select(0.0, ownDesign * beta, ok);
    next[nextOffset + index * ${size}u + ${component}u] = smoothed;
    errorsOut[errorsOutOffset + index] = ownPartial - smoothed;
    coefficients[coefficientsOffset + index * ${size}u + ${component}u] = beta;
  }`
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-score`,
          operation: OPERATION,
          variant: 'convergence-score',
          invocationCount: 1,
          bindings: [
            {name: 'points', view: spaceTime.points, type: 'f32', access: 'read'},
            {name: 'previous', view: previous, type: 'f32', access: 'read'},
            {name: 'next', view: next, type: 'f32', access: 'read'},
            {name: 'scores', view: output.scores, type: 'f32', access: 'read_write'}
          ],
          declarations: `${IS_FINITE_BITS_WGSL}`,
          body: `var count = 0.0;
  var change = 0.0;
  var total = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    if (!isFiniteBits(points[pointsOffset + 3u * row])) { continue; }
    count += 1.0;
    var sum = 0.0;
    for (var column = 0u; column < ${size}u; column++) {
      let value = next[nextOffset + row * ${size}u + column];
      let difference = value - previous[previousOffset + row * ${size}u + column];
      change += difference * difference;
      sum += value;
    }
    total += sum * sum;
  }
  scores[scoresOffset + ${iteration}u] = sqrt(change / count / total);`
        })
      );
    }
    const final = components[props.iterationCount % 2];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-results`,
        operation: OPERATION,
        variant: 'results',
        invocationCount: rows,
        bindings: [
          {name: 'points', view: spaceTime.points, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'finalComponents', view: final, type: 'f32', access: 'read'},
          {name: 'coefficients', view: output.coefficients, type: 'f32', access: 'read'},
          {name: 'fitted', view: output.fitted, type: 'f32', access: 'read_write'},
          {name: 'residuals', view: output.residuals, type: 'f32', access: 'read_write'},
          {name: 'localStatus', view: output.localStatus, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_BITS_WGSL}\n${GET_NAN_WGSL}`,
        body: `let included = isFiniteBits(points[pointsOffset + 3u * index]);
  var total = 0.0;
  var finite = true;
  for (var column = 0u; column < ${size}u; column++) {
    total += finalComponents[finalComponentsOffset + index * ${size}u + column];
    finite = finite && isFiniteBits(coefficients[coefficientsOffset + index * ${size}u + column]);
  }
  fitted[fittedOffset + index] = select(getNaN(), total, included);
  residuals[residualsOffset + index] = select(getNaN(), response[responseOffset + index] - total, included);
  localStatus[localStatusOffset + index] = select(${STATUS.EXCLUDED}u, select(${STATUS.SINGULAR}u, ${STATUS.OK}u, finite), included);`
      })
    );
    return nodes;
  }
}
