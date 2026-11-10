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
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUApproximationStatusPort,
  type GPUInvalidStatusPort,
  type GPUIterationStatusPort,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  getRegressionTileLayout,
  getRowTileSumDispatchShapes,
  getRowTileSumNodes,
  REGRESSION_TILE_MAXIMUM_ROW_COUNT
} from '../spatial-regression/regression-row-tiles';
import {GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} from './sinkhorn-divergence-parameters';
import {
  SINKHORN_DIVERGENCE_STATE_LENGTH,
  SINKHORN_DIVERGENCE_STATE_SLOTS,
  SINKHORN_LOG_SUM_WGSL
} from './sinkhorn-divergence-wgsl';

const OPERATION = 'GPUSinkhornDivergence';

/** Default number of Sinkhorn iterations unrolled into the graph. */
export const GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS = 100;

/**
 * Largest unrolled iteration count. An iteration is seven command nodes, so this stays far below
 * the shared command node budget.
 */
export const GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS = 1000;

/**
 * Largest grid side. A blur invocation walks one row or one column of at most this many cells (the
 * bounded neighbourhood of the contributor contract), so the per-invocation work stays bounded.
 */
export const GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE = 2048;

/** Command nodes outside the iterations: mass sums (2), logs, initialize, final sums (2), publications (2). */
const FIXED_COMMAND_NODE_COUNT = 8;
/** Command nodes of one iteration: two blurs, error tiles and merge, update, two blurs. */
const ITERATION_COMMAND_NODE_COUNT = 7;

/** Caller-owned outputs of {@link GPUSinkhornDivergence}. */
export type GPUSinkhornDivergenceOutput = {
  /**
   * One float32 row: the debiased Sinkhorn divergence
   * `S = OT(source, target) - OT(source, source) / 2 - OT(target, target) / 2` in squared position
   * units, with `OT` the entropic transport cost `min <P, C> + epsilon KL(P | a x b)`. NaN when
   * the masses or parameters could not be used (see `status.invalidCount`).
   */
  divergence: GraphDataView<'float32'>;
  /**
   * Optional three float32 rows `[OT(source, target), OT(source, source), OT(target, target)]`.
   * The divergence is a difference of these, so it carries their rounding; report them with it.
   */
  components?: GraphDataView<'float32'>;
  /**
   * Optional one-row L1 distance between the target marginal of the plan and the normalized target
   * masses, measured in the last executed iteration.
   */
  marginalError?: GraphDataView<'float32'>;
  /** Optional diagnostic scalars. */
  status?: GPUIterationStatusPort & GPUInvalidStatusPort & GPUApproximationStatusPort;
};

/**
 * Properties for {@link GPUSinkhornDivergence}.
 *
 * Per-frame (no recompile): the contents of `parameters` (cell size, regularization, tolerance,
 * schedule) and of `source` and `target`. Compile-time (needs a new graph): the grid size,
 * `maximumIterations`, `blurRadius` and which optional outputs are present.
 */
export type GPUSinkhornDivergenceProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'sinkhorn-divergence'`. */
  id?: string;
  /** Coordinate and metric contract of the grid: planar/native only. */
  spatialContext?: GPUSpatialContext;
  /**
   * The grid both masses live on. Only its width and height are used; the cell size comes from
   * `parameters` so it can change per frame.
   */
  grid: GPURasterMetadata;
  /**
   * Source masses, `grid.width * grid.height` float32 rows in raster row order. Non-negative and
   * finite; a negative or non-finite value is counted in `status.invalidCount` and treated as
   * zero. The masses are normalized to sum to one.
   */
  source: GraphDataView<'float32'>;
  /** Target masses, as `source`. */
  target: GraphDataView<'float32'>;
  /**
   * Per-frame float32 view of at least {@link GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} values
   * written with `getGPUSinkhornDivergenceParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Number of Sinkhorn iterations unrolled into the graph, `1` to
   * {@link GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS}. Defaults to
   * {@link GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS}.
   */
  maximumIterations?: number;
  /**
   * Radius in cells of each 1D blur window. Defaults to the full row and column (the longer grid
   * side minus one), which is exact. A smaller radius drops Gibbs kernel terms beyond it, which is
   * exact when `radius * cell size` covers the distance where `exp(-distance^2 / epsilon)` is
   * negligible against the nearest mass, and sets `status.approximation`; a cell with no mass
   * inside its window gets a huge potential.
   */
  blurRadius?: number;
  /** Caller-owned outputs. */
  output: GPUSinkhornDivergenceOutput;
};

/**
 * Sinkhorn divergence between two spatial distributions on the same grid (Genevay et al. 2018;
 * Feydy et al. 2019), the debiased entropic optimal transport cost
 * `S(a, b) = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2` with squared Euclidean cost between cell
 * centers and `OT(a, b) = min_P <P, C> + epsilon KL(P | a x b)` over couplings with marginals `a`
 * and `b`. It is zero for `a = b`, non-negative, and metrizes weak convergence, so unlike the
 * plain entropic cost it vanishes on identical inputs. Balanced only: both masses are normalized
 * to one, so a difference in total mass is not seen; unbalanced (KL-marginal) transport is not
 * implemented. POT 0.9.7's `empirical_sinkhorn_divergence` omits the entropic term and is a
 * different quantity (see the docs).
 *
 * Implementation, in the log domain: with potentials `f`, `g` the dual iteration is
 * `g_j = -epsilon LSE_i (f_i / epsilon + log a_i - C_ij / epsilon)` and the same for `f`. The
 * Gibbs kernel `exp(-C / epsilon)` is separable in the two grid axes, so each `LSE` over all cells
 * is two 1D log-sum-exp passes, one along rows and one along columns (the raster convolution
 * pattern in the log semiring), `O(cells * (width + height))` per update instead of
 * `O(cells^2)`. The three problems `(a, b)`, `(a, a)`, `(b, b)` run in one batch; the symmetric
 * problems use the averaged update `f <- (f + T f) / 2`. After each source-side-to-target-side
 * update a reduction measures the L1 error of the target marginal; a one-thread node tests it
 * against `tolerance`, applies the epsilon schedule (`epsilonStart * scalingFactor^k` down to
 * `epsilon`), and disarms a GPU dispatch gate, so finished iterations launch nothing. When
 * `maximumIterations` is reached first, the latest potentials are used and
 * `status.iterationLimitReached` is set. The divergence is evaluated from the potentials as
 * `sum a (f_ab - f_aa) + sum b (g_ab - g_bb)`, combining the differences before the tiled,
 * compensated reduction, which keeps the cancellation small.
 *
 * Float32: the exponents are the potentials over epsilon plus log masses minus squared distance
 * over epsilon, up to `maximum cost / epsilon` in size, each rounded to a relative `6e-8`. Keep
 * `epsilon` at least about `1e-3` times the squared grid diagonal. The divergence of identical
 * inputs is exactly the difference of two paths that agree only to the stopping tolerance and the
 * rounding; see the documentation for the measured size.
 */
export class GPUSinkhornDivergence implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSinkhornDivergenceProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Unrolled iteration count. */
  readonly maximumIterations: number;
  /** Blur window radius in cells. */
  readonly blurRadius: number;
  /** Whether the blur window covers the whole row and column (the result is exact). */
  readonly isExactBlur: boolean;
  /** Upper bound of the command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUSinkhornDivergenceProps) {
    this.id = props.id ?? 'sinkhorn-divergence';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    });
    const cellCount = validateRasterGrid(id, props.grid);
    const {width, height} = props.grid;
    validateMaximumCount(
      id,
      'grid.width and grid.height',
      Math.max(width, height),
      GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE
    );
    // Masses and counts are summed in float32, exact for integers up to 2^24.
    validateMaximumCount(id, 'cells', cellCount, REGRESSION_TILE_MAXIMUM_ROW_COUNT);
    this.maximumIterations =
      props.maximumIterations ?? GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS;
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      1,
      GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS
    );
    const fullRadius = Math.max(width, height) - 1;
    this.blurRadius = props.blurRadius ?? fullRadius;
    validateCapacity(
      id,
      'blurRadius',
      this.blurRadius,
      1,
      GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE
    );
    this.isExactBlur = this.blurRadius >= fullRadius;
    validatePackedView(props.source, ['float32'], `${id} source`);
    validatePackedView(props.target, ['float32'], `${id} target`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.source.length === cellCount && props.target.length === cellCount,
      'source and target must hold grid.width * grid.height rows'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH} float32 values`
    );
    const {output} = props;
    validatePackedView(output.divergence, ['float32'], `${id} output.divergence`);
    validateRequirement(
      id,
      output.divergence.length === 1,
      'output.divergence must contain exactly one row'
    );
    if (output.components) {
      validatePackedView(output.components, ['float32'], `${id} output.components`);
      validateRequirement(
        id,
        output.components.length === 3,
        'output.components must contain exactly three rows'
      );
    }
    if (output.marginalError) {
      validatePackedView(output.marginalError, ['float32'], `${id} output.marginalError`);
      validateRequirement(
        id,
        output.marginalError.length === 1,
        'output.marginalError must contain exactly one row'
      );
    }
    for (const [name, view] of Object.entries(output.status ?? {})) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.status.${name}`);
        validateRequirement(
          id,
          view.length === 1,
          `output.status.${name} must contain exactly one row`
        );
      }
    }
    const outputViews = this.getOutputViews();
    validateRequirement(
      id,
      new Set(outputViews.map(view => view.buffer)).size === outputViews.length,
      'outputs must use separate buffers'
    );
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.source,
      props.target,
      props.parameters
    ]);
    this.estimatedCommandNodeCount =
      FIXED_COMMAND_NODE_COUNT + ITERATION_COMMAND_NODE_COUNT * this.maximumIterations;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.divergence,
      output.components,
      output.marginalError,
      ...Object.values(output.status ?? {})
    ].filter(view => view !== undefined);
  }

  /**
   * Returns, in order: the mass sums, the log masses with zeroed potentials, the initialization,
   * the unrolled iterations (cross-problem and self-problem blurs, marginal error, update, and
   * the second half-update), the final sums, and the publication of values and status.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations, blurRadius} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const cellCount = width * height;
    validateGraphViewsBelongToGraph(id, graph, [
      props.source,
      props.target,
      props.parameters,
      ...this.getOutputViews()
    ]);
    // Four potential blocks and three blur blocks per cell, in single bindings.
    validateMaximumCount(id, 'cells', cellCount, getStorageBindingElementLimit(graph.device, 16));
    const slots = SINKHORN_DIVERGENCE_STATE_SLOTS;
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const state = f32('state', SINKHORN_DIVERGENCE_STATE_LENGTH);
    const masses = f32('masses', 4);
    const logSource = f32('log-source', cellCount);
    const logTarget = f32('log-target', cellCount);
    const potentials = f32('potentials', 4 * cellCount);
    const blurred = f32('blurred', 3 * cellCount);
    const errors = f32('errors', cellCount);
    const finalSums = f32('final-sums', 5);
    const layout = getRegressionTileLayout(cellCount);
    const [tileShape, singleMergeShape] = getRowTileSumDispatchShapes(layout, 1);
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: OPERATION,
      shapes: [
        [3 * cellCount],
        [3 * cellCount],
        tileShape,
        singleMergeShape,
        [cellCount],
        [cellCount]
      ]
    });
    const stateConstants = Object.entries(slots)
      .map(
        ([name, slot]) =>
          `const S_${name.replace(/[A-Z]/g, letter => `_${letter}`).toUpperCase()}: u32 = ${slot}u;`
      )
      .join('\n');
    const gridConstants = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const N: u32 = ${cellCount}u;
const RADIUS: u32 = ${blurRadius}u;`;
    const nanDeclaration =
      'fn getNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }';
    const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'f32',
      access: 'read'
    });
    const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'f32',
      access: 'read_write'
    });
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Mass sums and invalid counts of the two inputs.
    nodes.push(
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-masses`,
        operation: OPERATION,
        variant: 'masses',
        rowCount: cellCount,
        layout,
        sumCount: 4,
        bindings: [read('sourceMasses', props.source), read('targetMasses', props.target)],
        declarations: `${IS_FINITE_VALUE_WGSL}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let sourceMass = sourceMasses[sourceMassesOffset + row];
  if (isFiniteValue(sourceMass) && sourceMass >= 0.0) { (*terms)[0] += sourceMass; } else { (*terms)[2] += 1.0; }
  let targetMass = targetMasses[targetMassesOffset + row];
  if (isFiniteValue(targetMass) && targetMass >= 0.0) { (*terms)[1] += targetMass; } else { (*terms)[3] += 1.0; }
}`,
        partial: f32('mass-partials', layout.tileCount * 4),
        output: masses
      })
    );

    // 2. Log of the normalized masses (masked where zero or invalid), potentials set to zero.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-log-masses`,
        operation: OPERATION,
        variant: 'log-masses',
        bindings: [
          read('sourceMasses', props.source),
          read('targetMasses', props.target),
          read('masses', masses),
          write('logSource', logSource),
          write('logTarget', logTarget),
          write('potentials', potentials)
        ],
        invocationCount: cellCount,
        declarations: `${IS_FINITE_VALUE_WGSL}
${SINKHORN_LOG_SUM_WGSL}
const N: u32 = ${cellCount}u;`,
        body: `let sourceMass = sourceMasses[sourceMassesOffset + index];
  let targetMass = targetMasses[targetMassesOffset + index];
  let sourceTotal = masses[massesOffset];
  let targetTotal = masses[massesOffset + 1u];
  let sourceValid = isFiniteValue(sourceMass) && sourceMass > 0.0 && sourceTotal > 0.0;
  let targetValid = isFiniteValue(targetMass) && targetMass > 0.0 && targetTotal > 0.0;
  logSource[logSourceOffset + index] = select(MASKED, log(sourceMass / sourceTotal), sourceValid);
  logTarget[logTargetOffset + index] = select(MASKED, log(targetMass / targetTotal), targetValid);
  potentials[potentialsOffset + index] = 0.0;
  potentials[potentialsOffset + N + index] = 0.0;
  potentials[potentialsOffset + 2u * N + index] = 0.0;
  potentials[potentialsOffset + 3u * N + index] = 0.0;`
      })
    );

    // 3. Initialization and arming of the iteration gate.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        invocationCount: 1,
        bindings: [
          read('params', props.parameters),
          read('masses', masses),
          write('state', state),
          gate.binding
        ],
        declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${nanDeclaration}`,
        body: `let cellWidth = params[paramsOffset];
  let cellHeight = params[paramsOffset + 1u];
  let epsilon = params[paramsOffset + 2u];
  let tolerance = params[paramsOffset + 3u];
  let epsilonStart = params[paramsOffset + 4u];
  let scalingFactor = params[paramsOffset + 5u];
  let valid = masses[massesOffset] > 0.0 && masses[massesOffset + 1u] > 0.0 &&
    cellWidth > 0.0 && cellHeight > 0.0 && epsilon > 0.0 && tolerance > 0.0 &&
    epsilonStart >= epsilon && scalingFactor > 0.0 && scalingFactor <= 1.0 &&
    isFiniteValue(cellWidth) && isFiniteValue(cellHeight) && isFiniteValue(epsilon) &&
    isFiniteValue(tolerance) && isFiniteValue(epsilonStart);
  state[stateOffset + S_EPSILON] = select(1.0, max(epsilon, epsilonStart), valid);
  state[stateOffset + S_DONE] = select(1.0, 0.0, valid);
  state[stateOffset + S_VALID] = select(0.0, 1.0, valid);
  state[stateOffset + S_ITERATION_COUNT] = 0.0;
  state[stateOffset + S_CONVERGED] = 0.0;
  state[stateOffset + S_ITERATION_LIMIT_REACHED] = 0.0;
  state[stateOffset + S_MARGINAL_ERROR] = select(getNaN(), 0.0, valid);
  ${gate.disarmWGSL}
  if (valid) {
    ${gate.armWGSL}
  }`
      })
    );

    // 4. The iterations. Blur passes read the regularization of the state.
    const blurDeclarations = `${gridConstants}\n${stateConstants}\n${SINKHORN_LOG_SUM_WGSL}`;
    const getBlurX = (stage: 1 | 2) => {
      const jobCount = stage === 1 ? 3 : 1;
      return {
        bindings: [
          read('potentials', potentials),
          read('logSource', logSource),
          read('logTarget', logTarget),
          read('state', state),
          read('params', props.parameters),
          write('blurred', blurred)
        ],
        invocationCount: jobCount * cellCount,
        // Stage 1 jobs: (source of cross, log source), (self source, log source), (self target, log target).
        // Stage 2 has the one job: target of cross with log target.
        body: `let job = index / N;
  let cell = index % N;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  let epsilon = state[stateOffset + S_EPSILON];
  let cellWidth = params[paramsOffset];
  ${
    stage === 1
      ? 'let inputBlock = select(select(0u, 2u, job == 1u), 3u, job == 2u);\n  let useTarget = job == 2u;'
      : 'let inputBlock = 1u;\n  let useTarget = true;'
  }
  var maximum = MASKED;
  var sum = 0.0;
  let first = column - min(column, RADIUS);
  let last = min(column + RADIUS, WIDTH - 1u);
  for (var other = first; other <= last; other++) {
    let at = row * WIDTH + other;
    let weight = select(logSource[logSourceOffset + at], logTarget[logTargetOffset + at], useTarget);
    if (weight > MASKED_LIMIT) {
      let distance = (f32(other) - f32(column)) * cellWidth;
      accumulateLogSum(potentials[potentialsOffset + inputBlock * N + at] / epsilon + weight - distance * distance / epsilon, &maximum, &sum);
    }
  }
  blurred[blurredOffset + index] = select(MASKED, maximum + log(sum), sum > 0.0);`
      };
    };
    const getBlurY = (stage: 1 | 2) => {
      const jobCount = stage === 1 ? 3 : 1;
      return {
        bindings: [
          read('blurred', blurred),
          write('potentials', potentials),
          read('logTarget', logTarget),
          read('state', state),
          read('params', props.parameters),
          ...(stage === 1 ? [write('errors', errors)] : [])
        ],
        invocationCount: jobCount * cellCount,
        body: `let job = index / N;
  let cell = index % N;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  let epsilon = state[stateOffset + S_EPSILON];
  let cellHeight = params[paramsOffset + 1u];
  var maximum = MASKED;
  var sum = 0.0;
  let first = row - min(row, RADIUS);
  let last = min(row + RADIUS, HEIGHT - 1u);
  for (var other = first; other <= last; other++) {
    let distance = (f32(other) - f32(row)) * cellHeight;
    accumulateLogSum(blurred[blurredOffset + job * N + other * WIDTH + column] - distance * distance / epsilon, &maximum, &sum);
  }
  let updated = select(UNREACHABLE, -epsilon * (maximum + log(sum)), sum > 0.0);
  ${
    stage === 1
      ? `let outputBlock = select(select(1u, 2u, job == 1u), 3u, job == 2u);
  let slot = potentialsOffset + outputBlock * N + cell;
  let previous = potentials[slot];
  if (job == 0u) {
    // L1 error of the target marginal the previous potentials implied: b_j |exp((g_old - g_new) / epsilon) - 1|.
    let targetLog = logTarget[logTargetOffset + cell];
    let targetMass = select(0.0, exp(targetLog), targetLog > MASKED_LIMIT);
    errors[errorsOffset + cell] = targetMass * abs(exp(min((previous - updated) / epsilon, 80.0)) - 1.0);
    potentials[slot] = updated;
  } else {
    potentials[slot] = 0.5 * (previous + updated);
  }`
      : 'potentials[potentialsOffset + cell] = updated;'
  }`
      };
    };
    const kernel = (
      name: string,
      variant: string,
      pass: {bindings: WGSLKernelBinding[]; invocationCount: number; body: string},
      gateSlot: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant,
        bindings: pass.bindings,
        invocationCount: pass.invocationCount,
        declarations: `${blurDeclarations}\nconst UNREACHABLE: f32 = 1.0e30;`,
        body: pass.body,
        ...gate.getGate(gateSlot)
      });
    for (let iteration = 0; iteration < maximumIterations; iteration++) {
      const prefix = `iteration-${iteration}`;
      nodes.push(
        kernel(`${prefix}-blur-columns-first`, 'blur-x-batch', getBlurX(1), 0),
        kernel(`${prefix}-blur-rows-first`, 'blur-y-batch', getBlurY(1), 1),
        ...getRowTileSumNodes<Parameters>(graph, {
          id: `${id}-${prefix}-marginal-error`,
          operation: OPERATION,
          variant: 'marginal-error',
          rowCount: cellCount,
          layout,
          sumCount: 1,
          bindings: [read('errors', errors)],
          declarations: `fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  (*terms)[0] += errors[errorsOffset + row];
}`,
          partial: f32(`${prefix}-error-partials`, layout.tileCount),
          output: state,
          outputSlot: slots.marginalError,
          tileGate: gate.getGate(2),
          mergeGate: gate.getGate(3)
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${prefix}-update`,
          operation: OPERATION,
          variant: 'update',
          invocationCount: 1,
          bindings: [read('params', props.parameters), write('state', state), gate.binding],
          declarations: `${stateConstants}
const MAXIMUM_ITERATIONS: f32 = ${maximumIterations}.0;`,
          body: `if (state[stateOffset + S_DONE] != 0.0) {
    return;
  }
  let iterations = state[stateOffset + S_ITERATION_COUNT] + 1.0;
  state[stateOffset + S_ITERATION_COUNT] = iterations;
  let finalEpsilon = params[paramsOffset + 2u];
  let tolerance = params[paramsOffset + 3u];
  // The error only counts once the regularization has reached its final value.
  let atFinalEpsilon = state[stateOffset + S_EPSILON] <= finalEpsilon * 1.000001;
  let converged = atFinalEpsilon && state[stateOffset + S_MARGINAL_ERROR] <= tolerance;
  state[stateOffset + S_EPSILON] = max(finalEpsilon, params[paramsOffset + 4u] * pow(params[paramsOffset + 5u], iterations));
  if (converged) {
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_CONVERGED] = 1.0;
    ${gate.disarmWGSL}
  } else if (iterations >= MAXIMUM_ITERATIONS) {
    // The second half of this last iteration still runs: the gate stays armed.
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_ITERATION_LIMIT_REACHED] = 1.0;
  }`
        }),
        kernel(`${prefix}-blur-columns-second`, 'blur-x-single', getBlurX(2), 4),
        kernel(`${prefix}-blur-rows-second`, 'blur-y-single', getBlurY(2), 5)
      );
    }

    // 5. Dual objectives from the potentials.
    nodes.push(
      ...getRowTileSumNodes<Parameters>(graph, {
        id: `${id}-final-sums`,
        operation: OPERATION,
        variant: 'final-sums',
        rowCount: cellCount,
        layout,
        sumCount: 5,
        bindings: [
          read('potentials', potentials),
          read('logSource', logSource),
          read('logTarget', logTarget)
        ],
        declarations: `${SINKHORN_LOG_SUM_WGSL}
const N: u32 = ${cellCount}u;
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let sourceLog = logSource[logSourceOffset + row];
  if (sourceLog > MASKED_LIMIT) {
    let mass = exp(sourceLog);
    let cross = potentials[potentialsOffset + row];
    let own = potentials[potentialsOffset + 2u * N + row];
    (*terms)[0] += mass * cross;
    (*terms)[2] += mass * own;
    (*terms)[4] += mass * (cross - own);
  }
  let targetLog = logTarget[logTargetOffset + row];
  if (targetLog > MASKED_LIMIT) {
    let mass = exp(targetLog);
    let cross = potentials[potentialsOffset + N + row];
    let own = potentials[potentialsOffset + 3u * N + row];
    (*terms)[1] += mass * cross;
    (*terms)[3] += mass * own;
    (*terms)[4] += mass * (cross - own);
  }
}`,
        partial: f32('final-partials', layout.tileCount * 5),
        output: finalSums
      })
    );

    // 6. Publish the values, then the status.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-values`,
        operation: OPERATION,
        variant: 'publish-values',
        invocationCount: 1,
        bindings: [
          read('state', state),
          read('finalSums', finalSums),
          write('divergence', output.divergence),
          ...(output.components ? [write('components', output.components)] : []),
          ...(output.marginalError ? [write('marginalError', output.marginalError)] : [])
        ],
        declarations: `${stateConstants}\n${nanDeclaration}`,
        body: `let valid = state[stateOffset + S_VALID] != 0.0;
  let nan = getNaN();
  divergence[divergenceOffset] = select(nan, finalSums[finalSumsOffset + 4u], valid);
  ${
    output.components
      ? `components[componentsOffset] = select(nan, finalSums[finalSumsOffset] + finalSums[finalSumsOffset + 1u], valid);
  components[componentsOffset + 1u] = select(nan, 2.0 * finalSums[finalSumsOffset + 2u], valid);
  components[componentsOffset + 2u] = select(nan, 2.0 * finalSums[finalSumsOffset + 3u], valid);`
      : ''
  }
  ${output.marginalError ? 'marginalError[marginalErrorOffset] = select(nan, state[stateOffset + S_MARGINAL_ERROR], valid);' : ''}`
      })
    );
    const status = output.status ?? {};
    const statusBindings = (
      [
        ['converged', status.converged],
        ['iterationLimitReached', status.iterationLimitReached],
        ['iterationCount', status.iterationCount],
        ['invalidCount', status.invalidCount],
        ['approximation', status.approximation]
      ] as const
    ).filter(([, view]) => view !== undefined);
    if (statusBindings.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-status`,
          operation: OPERATION,
          variant: 'publish-status',
          invocationCount: 1,
          bindings: [
            read('state', state),
            read('masses', masses),
            ...statusBindings.map(
              ([name, view]) => ({name, view: view!, type: 'u32', access: 'read_write'}) as const
            )
          ],
          declarations: stateConstants,
          body: `${status.converged ? 'converged[convergedOffset] = u32(state[stateOffset + S_CONVERGED]);' : ''}
  ${status.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = u32(state[stateOffset + S_ITERATION_LIMIT_REACHED]);' : ''}
  ${status.iterationCount ? 'iterationCount[iterationCountOffset] = u32(state[stateOffset + S_ITERATION_COUNT]);' : ''}
  ${status.invalidCount ? 'invalidCount[invalidCountOffset] = u32(masses[massesOffset + 2u] + masses[massesOffset + 3u]);' : ''}
  ${status.approximation ? `approximation[approximationOffset] = ${this.isExactBlur ? '0u' : '1u'};` : ''}`
        })
      );
    }
    return nodes;
  }
}
