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
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {getEquilibratedCholeskyWGSL} from './local-weighted-least-squares-wgsl';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../analysis-core/kernels/incomplete-gamma-wgsl';
import {getDenseLUWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {getPanelLagTiledNodes} from './panel-lag-tiled-nodes';
import {PANEL_LAG_PROFILE_WGSL} from './panel-lag-profile-wgsl';
import {PANEL_LAG_PROFILE_SCORE_WGSL} from './panel-lag-profile-score-wgsl';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {getPanelWithinNodes, getPanelWithinScratchLengths} from './panel-within-nodes';
import {getPanelLagPoleAllocationLengths, getPanelLagPoleNodes} from './panel-lag-pole-nodes';
import {GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT} from './spatial-exact-log-determinant';
import {getRegressionTileLayout, REGRESSION_TILE_MAXIMUM_ROW_COUNT} from './regression-row-tiles';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

/** @internal Capture weights metadata and declared status fields without copying graph views. */
export function snapshotPanelLagWeights<
  Weights extends {
    offsets: unknown;
    neighbors: unknown;
    weights: unknown;
    distances?: unknown;
    status?: Partial<Record<keyof NonNullable<GPUSpatialWeights['status']>, unknown>>;
    information?: GPUSpatialWeights['information'];
    spatialContext?: GPUSpatialWeights['spatialContext'];
  }
>(weights: Weights): Weights {
  const snapshot = snapshotConstructionFields(weights, [
    'offsets',
    'neighbors',
    'weights',
    'distances',
    'status',
    'information',
    'spatialContext'
  ]);
  const information = snapshot.information
    ? snapshotConstructionFields(snapshot.information, [
        'rowCount',
        'transform',
        'symmetric',
        'hasSelfWeights',
        'provenance'
      ])
    : undefined;
  return Object.freeze({
    ...snapshot,
    status: snapshot.status
      ? snapshotConstructionFields(snapshot.status, [
          'complete',
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow',
          'uncertainCount',
          'invalidCount',
          'iterationCount',
          'converged',
          'iterationLimitReached',
          'approximation'
        ])
      : undefined,
    information: information
      ? Object.freeze({
          ...information,
          provenance: information.provenance
            ? snapshotConstructionFields(information.provenance, [
                'source',
                'criterion',
                'k',
                'radius',
                'order'
              ])
            : undefined
        })
      : undefined,
    spatialContext: snapshot.spatialContext
      ? snapshotConstructionSpatialContext(snapshot.spatialContext)
      : undefined
  }) as Weights;
}

/** Supplied entity spectra share the existing exact-logdet decomposition envelope. */
export const GPU_PANEL_LAG_TILED_MAXIMUM_ENTITY_COUNT =
  GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT;
/** Tiled counts and row divisors stay exact in float32; scratch is checked on the device. */
export const GPU_PANEL_LAG_TILED_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;

/**
 * Lowest random-effects `phi` the fit follows. `phi = sigma_e / sqrt(T sigma_u^2 + sigma_e^2)`
 * falls with the size of the entity effects against the residuals; float32 data cannot carry a
 * residual much below 1e-6 of an entity effect, so with at most 32 periods `phi` stays above
 * about 2e-8 on data the kernel can resolve. An estimate below this limit is held at it and the
 * fit is status 3 with boundary bit 4. (The limit was 1e-5 and was reported as status 0: entity
 * effects of 3000 with residuals of 0.004, `phi` 3e-7, came out with `sigma2` 64 times too
 * large.)
 */
const RANDOM_EFFECTS_MINIMUM_PHI = 1e-8;
/**
 * The alternation stops when `phi` changes by no more than `tolerance`, as an absolute change
 * for `phi` of this value or more and scaled by `phi` over this value below it: an absolute
 * tolerance of 1e-5 is no test at all for a `phi` of 3e-7.
 */
const RANDOM_EFFECTS_RELATIVE_PHI = 1e-2;

/** The exact determinant and inverse hold a dense entity matrix in private memory. */
export const GPU_PANEL_LAG_MAXIMUM_ENTITY_COUNT = 32;
/** The reference fit walks periods and rows serially during each likelihood profile. */
export const GPU_PANEL_LAG_MAXIMUM_PERIOD_COUNT = 32;
/** Dense normal and information matrices include an intercept and the spatial parameter. */
const MAXIMUM_PREDICTOR_COUNT = 7;
/** Alternating random-effects profiling remains a bounded serial reference fit. */
const MAXIMUM_ITERATION_COUNT = 128;
/** Golden-section searches remain bounded inside one invocation. */
const MAXIMUM_SEARCH_ITERATION_COUNT = 64;

export type GPUSpatialPanelLagMaximumLikelihoodProps = {
  id?: string;
  /** Auto uses tiled unit effects when eigenvalues are supplied; reference preserves the dense route. */
  fitMode?: 'auto' | 'reference' | 'tiled';
  /** Entity-graph eigenvalues from getSpatialWeightsEigenvalues, required for tiled unit effects.
   * The caller supplies the spectrum of W (which must be similar to a symmetric matrix).
   * Complete panel rows are tiled; the determinant is T times the entity log determinant.
   */
  eigenvalues?: GraphDataView<'float32'>;
  weights: GPUSpatialWeights;
  periodCount: number;
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  response: GraphDataView<'float32'>;
  effects?: 'random' | 'unit';
  maximumIterations?: number;
  maximumSearchIterations?: number;
  tolerance?: number;
  output: {
    /** Intercept, slopes, rho, phi. Unit effects absorb the intercept (NaN) and set phi=0. */
    coefficients: GraphDataView<'float32'>;
    /** Conditional expected-information covariance of intercept/slopes/rho, excluding phi and variance. */
    covariance: GraphDataView<'float32'>;
    /**
     * sigma2, spreg-reported log likelihood, outer sweeps, phi change, rho bracket width,
     * boundary bits: 1 rho near an end of its interval; 2 random-effects phi at an end (1, or
     * below its lowest value); 4 phi below its lowest value (1e-8), always with bit 2 and
     * status 3.
     */
    summary: GraphDataView<'float32'>;
    /**
     * 0 converged; 1 invalid/incomplete inputs; 2 singular; 3 iteration/search exhaustion, or a
     * random-effects phi below the lowest value the fit follows (boundary bit 4: the published
     * numbers are those of phi held at that value, not the maximum; no covariance).
     */
    status: GraphDataView<'uint32'>;
    /**
     * Quasi-demeaned fitted response and innovations, matching spreg. `fitted` is the
     * quasi-demeaned response `y - (1 - phi) mean_i(y)` at the published phi less the residual,
     * so it moves with phi by the entity mean of the response per unit of phi; `residuals` do
     * not (see the class documentation).
     */
    fitted: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
  };
  /** Random-versus-unit comparison. Tests are [statistic, df, p]; df=predictorCount+2 follows spreg.k, rather than the smaller contrast rank. Status: 0 valid, 1 unsuccessful fit, 2 invalid covariance difference. */
  hausman?: {tests: GraphDataView<'float32'>; status: GraphDataView<'uint32'>};
};

/**
 * Balanced spatial lag ML, random or unit effects. Random mode alternates analytic phi
 * minimization with bounded exact-logdet rho profiling, matching spreg Panel_RE_Lag.
 * The reference route uses a dense entity determinant and is limited to 32 entities, 32 periods,
 * 7 predictors. With unit effects and caller-supplied entity eigenvalues, auto selects row tiles
 * and a storage-backed pivoted inverse: at most 1024 entities and 2^24 total panel rows. Random
 * effects and Hausman comparisons retain the reference route. Complete panels and nonnegative row-standardized zero-diagonal weights are required.
 * No hidden regularization or CPU solve. Hausman uses spreg's RE-minus-FE conditional variance
 * difference; indefinite/singular or negative-statistic comparisons are explicitly invalid.
 *
 * An admissible zero-RSS profile minimum has no finite Gaussian maximum (status 2). A
 * sufficient original-input integer certificate also checks nearby rounded fits: if the
 * response differences within each entity equal X beta + rho W y differences exactly, all
 * floating outputs are withheld as NaN and Hausman is status 1/all NaN. The certificate tests
 * 24 rounded candidates; it never uses a variance cutoff. For a single predictor, temporal
 * differences of y-rho*Wy are also checked for exact proportionality at those rho candidates
 * and at zero, so a rational slope such as 1/3 need not be representable in f32.
 * It supports normal/zero inputs with triple-product exponent span at most 168, or four-factor
 * cross-product span at most 140; unsupported input gives no certificate. This
 * is a sufficient degeneracy check, not a complete symbolic rank decision.
 * Squared moments that underflow to zero also fail with status 2 and can leave every floating
 * diagnostic NaN. The CPU twin can retain a finite float64 likelihood before its variance
 * rounds to zero at float32 publication; failure diagnostics need not match across precisions.
 *
 * The fit runs on centered data: the pooled mean of each predictor and of the response is
 * subtracted before quasi-demeaning (the spatial lag of the response is shifted by the same
 * response mean), so a predictor with a large mean (a year, an elevation, an income) costs no
 * precision. The model is unchanged by the shift except for its intercept, which is mapped back
 * as `intercept + (1 - rho) mean(y) - slopes' mean(x)`, with the covariance row and column of the
 * intercept transformed to match. Every normal matrix and the information matrix are factored
 * after diagonal equilibration; a system is singular (status 2) when a Cholesky pivot of the
 * unit-diagonal matrix is below `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (`1e-3`), the limit
 * of `GPUOrdinaryLeastSquares`.
 *
 * The concentrated variance is formed directly. With `e0` and `e1` the residuals of the
 * quasi-demeaned response and of its lag on the predictors, `rho0 = e0'e1 / e1'e1`,
 * `r = e0 - rho0 e1` summed per row, `rho* = rho0 + r'e1 / e1'e1` and
 * `S = r'r - (r'e1)^2 / e1'e1`: `u'u = S + e1'e1 (rho - rho*)^2`, two non-negative terms. The
 * bounded brackets over `rho` and the published `sigma2`, log-likelihood and covariance all use
 * it. The form `e0'e0 - 2 rho e0'e1 + rho^2 e1'e1` of three float32 sums is rounding once the
 * model fits well: with `u'u` at 3e-5 of `e0'e0` the published `sigma2` was 1% to 2% off, at
 * 3e-7 it was 36% low and 96% high, and at 3e-9 it was 10 to 241 times too large with status 0,
 * or negative (`math-review-panel-lag-conditioning.spec.ts`, against spreg 1.9.1
 * `Panel_FE_Lag` and `Panel_RE_Lag`: now within 1e-5, 6.3e-5 and 1.7e-3, the last being the
 * float32 rounding of the fitted values). The pack node sums the lag of the response from
 * differences rounded as written and carries the row sum of the float32 weights exactly.
 * After eight coarse golden steps, bracket decisions use the analytic likelihood derivative,
 * `n (rho-rho*) / (S/(e1'e1)+(rho-rho*)^2) + T tr(W (I-rho W)^-1)`, rather than comparing
 * nearly equal rounded logarithms. The same search-step budget and golden bracket geometry
 * apply. This remains a local bounded profile method, not a proof of a global maximum.
 *
 * Quasi-demeaning. Entity effects survive the pooled centering. Each column is packed as its
 * within deviation (centered value less its entity mean, rounded as written) and its entity
 * mean, and the quasi-demeaned value `value - (1 - phi) mean` is formed as
 * `deviation + phi mean`, whose terms do not cancel. Formed as written, the shader compiler of
 * the test GPU regrouped `(value - mean) - fitted` as `value - (mean + fitted)`: with entity
 * effects of +3000 and -3000 and residuals of 0.004 every published residual was a multiple of
 * the float32 step of 3000, and `rho` was 0.026 standard errors off (an earlier fix reached 0.002;
 * `math-review-panel-lag-conditioning.spec.ts`, against spreg `Panel_FE_Lag`). The normal
 * equations of each step of the alternation are the polynomial in `phi` of three moment sums
 * (deviations, deviations with means, means) taken once, so they vary smoothly with `phi`.
 * Normal-equation moments use fixed pairwise row sums, including inside each tile on the tiled
 * route. Serial sums lost small predictor squares after large ones and biased the slope even
 * after rho profiling was accurate. CPU moments retain float64 accumulation.
 * The reference pack now lags each neighbor's within response before reintroducing lagged
 * entity means. Within values use the shared split-center transform, preserving constant
 * columns and isolated extremes without subtracting a first-observation reference. The helper
 * also exposes its two-part means internally so random effects retain the same mean/deviation
 * decomposition.
 * Temporal demeaning commutes with time-invariant W. Forming W y first rounded
 * within variation to the step of an effect near 3000, biasing rho by 0.012 standard errors
 * on a later Metal run. Entity means remain float32 sums; their error can still leave a
 * constant entity offset. The new lag order preserves the small within variation itself.
 *
 * Random-effects `phi`. `phi` is followed down to 1e-8 (`RANDOM_EFFECTS_MINIMUM_PHI`; it was
 * held at 1e-5 and the constrained fit was returned with status 0: with entity effects of
 * +3000 and -3000 and residuals of 0.004, spreg `phi` 2.5e-7, `sigma2` was 105 times spreg's
 * and `rho` 3.5 standard errors off). On that scene the fit now has `phi` within 0.2% of spreg
 * `Panel_RE_Lag`, `rho` within 0.009 standard errors, the intercept within 0.2, `sigma2` within
 * 1.1% and the log-likelihood within 1.5: the float32 entity means again. An estimate below
 * 1e-8 is held there and reported: status 3, boundary bits 2 and 4, no covariance. The
 * alternation stops on the change of `phi` against `tolerance`, absolute for a `phi` of 0.01 or
 * more and in proportion to `phi / 0.01` below (`RANDOM_EFFECTS_RELATIVE_PHI`).
 * `fitted`. The output is the quasi-demeaned response at the published `phi` less the
 * residual: `fitted + residuals = y - (1 - phi) mean_i(y)`. A `phi` that differs from spreg's
 * by `d`, inside the tolerance of the alternation, moves `fitted` by `d` times the entity mean
 * of the response while leaving the residuals alone: with a response mean of 10,000 and
 * residuals of 0.004 a `phi` within 4.5e-7 of spreg's gives fitted values 1.2 residual
 * deviations from spreg's (and a float32 `phi` near 0.3 cannot be closer than 3e-8).
 */
export class GPUSpatialPanelLagMaximumLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  readonly fitMode: 'reference' | 'tiled';
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUSpatialPanelLagMaximumLikelihoodProps;
  constructor(props: GPUSpatialPanelLagMaximumLikelihoodProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'fitMode',
      'eigenvalues',
      'weights',
      'periodCount',
      'predictors',
      'predictorCount',
      'response',
      'effects',
      'maximumIterations',
      'maximumSearchIterations',
      'tolerance',
      'output',
      'hausman'
    ]);
    props = Object.freeze({
      ...snapshot,
      weights: snapshotPanelLagWeights(snapshot.weights),
      output: snapshotConstructionFields(snapshot.output, [
        'coefficients',
        'covariance',
        'summary',
        'status',
        'fitted',
        'residuals'
      ]),
      hausman: snapshot.hausman
        ? snapshotConstructionFields(snapshot.hausman, ['tests', 'status'])
        : undefined
    });
    this.props = props;
    this.id = props.id ?? 'spatial-panel-lag-maximum-likelihood';
    const entities = validateGPUSpatialWeights(this.id, props.weights);
    const rows = entities * props.periodCount;
    const columns = props.predictorCount + 1;
    validateRequirement(
      this.id,
      props.fitMode === undefined || ['auto', 'reference', 'tiled'].includes(props.fitMode),
      "fitMode must be 'auto', 'reference' or 'tiled'"
    );
    this.fitMode =
      props.fitMode === 'reference'
        ? 'reference'
        : props.fitMode === 'tiled' || props.eigenvalues
          ? 'tiled'
          : 'reference';
    validateRequirement(
      this.id,
      this.fitMode !== 'tiled' || (props.effects === 'unit' && Boolean(props.eigenvalues)),
      "tiled fitMode requires effects 'unit' and entity eigenvalues"
    );
    validateRequirement(
      this.id,
      this.fitMode !== 'reference' || !props.eigenvalues,
      'eigenvalues require tiled fitMode'
    );
    validateCapacity(
      this.id,
      'entities',
      entities,
      2,
      this.fitMode === 'tiled'
        ? GPU_PANEL_LAG_TILED_MAXIMUM_ENTITY_COUNT
        : GPU_PANEL_LAG_MAXIMUM_ENTITY_COUNT
    );
    validateCapacity(
      this.id,
      'periodCount',
      props.periodCount,
      2,
      this.fitMode === 'tiled'
        ? GPU_PANEL_LAG_TILED_MAXIMUM_ROW_COUNT
        : GPU_PANEL_LAG_MAXIMUM_PERIOD_COUNT
    );
    if (this.fitMode === 'tiled')
      validateCapacity(
        this.id,
        'entities * periodCount',
        rows,
        1,
        GPU_PANEL_LAG_TILED_MAXIMUM_ROW_COUNT
      );
    if (props.eigenvalues) {
      validateSinglePackedView(props.eigenvalues, ['float32'], `${this.id} eigenvalues`);
      validateRequirement(
        this.id,
        props.eigenvalues.length === entities,
        'eigenvalues must hold one value per entity'
      );
    }
    const pivotPasses =
      props.predictorCount === 1
        ? Math.max(1, Math.ceil(Math.log2(getRegressionTileLayout(rows).tileCount)))
        : 1;
    this.estimatedCommandNodeCount =
      this.fitMode === 'tiled'
        ? 5 * entities +
          78 +
          Number(
            Boolean(
              props.weights.status?.count ||
                props.weights.status?.requiredCount ||
                props.weights.status?.complete
            )
          )
        : 34 + Number(Boolean(props.weights.status?.complete));
    this.estimatedCommandNodeCount += pivotPasses - 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateCapacity(this.id, 'predictorCount', props.predictorCount, 1, MAXIMUM_PREDICTOR_COUNT);
    validateCapacity(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? 64,
      1,
      MAXIMUM_ITERATION_COUNT
    );
    validateCapacity(
      this.id,
      'maximumSearchIterations',
      props.maximumSearchIterations ?? 32,
      1,
      MAXIMUM_SEARCH_ITERATION_COUNT
    );
    validateRequirement(
      this.id,
      Math.fround(props.tolerance ?? 1e-5) > 0 &&
        Number.isFinite(Math.fround(props.tolerance ?? 1e-5)),
      'tolerance must be positive and finite when represented as float32'
    );
    if (
      !['random', 'unit'].includes(props.effects ?? 'random') ||
      !(Number.isFinite(props.tolerance ?? 1e-5) && (props.tolerance ?? 1e-5) > 0) ||
      (props.hausman && props.effects === 'unit')
    )
      throw new Error(`${this.id}: invalid estimator settings`);
    for (const [view, format, length] of [
      [props.predictors, 'float32', rows * props.predictorCount],
      [props.response, 'float32', rows],
      [props.output.coefficients, 'float32', columns + 2],
      [props.output.covariance, 'float32', (columns + 1) ** 2],
      [props.output.summary, 'float32', 6],
      [props.output.status, 'uint32', 1],
      [props.output.fitted, 'float32', rows],
      [props.output.residuals, 'float32', rows],
      [props.hausman?.tests, 'float32', 3],
      [props.hausman?.status, 'uint32', 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateRegressionOutputDispatches(
      this.id,
      {
        ...props.output,
        hausmanTests: props.hausman?.tests,
        hausmanStatus: props.hausman?.status
      },
      [
        ['coefficients', 'covariance', 'summary', 'status'],
        ['fitted', 'residuals'],
        ['hausmanTests', 'hausmanStatus']
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [...Object.values(props.output), ...Object.values(props.hausman ?? {})],
      [
        props.predictors,
        props.response,
        props.eigenvalues,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.status?.complete,
        props.weights.status?.count,
        props.weights.status?.requiredCount,
        props.weights.status?.overflow,
        props.weights.status?.candidateOverflow
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    if (this.fitMode === 'tiled') return getPanelLagTiledNodes(graph, id, props);
    const entities = props.weights.offsets.length - 1,
      rows = props.response.length,
      columns = props.predictorCount + 1;
    const width = columns + 2;
    const covarianceOffset = width,
      summaryOffset = width + (columns + 1) ** 2,
      statusOffset = summaryOffset + 6,
      // Intercept of the centered model, read by the fitted and residual node.
      centeredInterceptOffset = statusOffset + 1,
      stride = statusOffset + 2;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      response: props.response,
      offsets: props.weights.offsets,
      neighbors: props.weights.neighbors,
      weights: props.weights.weights,
      count: props.weights.status?.count,
      complete: props.weights.status?.complete,
      requiredCount: props.weights.status?.requiredCount,
      overflow: props.weights.status?.overflow,
      candidateOverflow: props.weights.status?.candidateOverflow,
      ...props.output,
      hausmanTests: props.hausman?.tests,
      hausmanStatus: props.hausman?.status
    });
    for (const [name, length] of [
      ['packed panel', rows * width * 2],
      ['dense weights', entities * entities],
      ['model results', stride * (props.hausman ? 2 : 1)],
      ['spatial product', entities * entities * (props.hausman ? 2 : 1)],
      ['pooled means', columns + 1],
      ['within predictors', props.predictors.length],
      ['within response', rows],
      [
        'split entity means',
        getPanelWithinScratchLengths(entities, props.periodCount, props.predictorCount).meanLength
      ],
      ...Object.entries(getPanelLagPoleAllocationLengths(rows, props.predictorCount))
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce panel dimensions'
      );
    }
    const data = createTransientView(graph, `${id}-packed-panel`, 'float32', rows * width * 2),
      dense = createTransientView(graph, `${id}-dense-weights`, 'float32', entities * entities),
      results = createTransientView(
        graph,
        `${id}-model-results`,
        'float32',
        stride * (props.hausman ? 2 : 1)
      ),
      spatialProduct = createTransientView(
        graph,
        `${id}-spatial-product`,
        'float32',
        entities * entities * (props.hausman ? 2 : 1)
      );
    // Pooled means: slot 0 the response, slots 1..K the predictors. Any constant shift gives the
    // same model, so these need not be exact; the same float32 values are used to map back.
    // The last slot is a zero written at run time (the runtime zero of getRoundedDifference).
    const centers = createTransientView(graph, `${id}-pooled-means`, 'float32', columns + 1);
    const withinPredictors = createTransientView(
      graph,
      `${id}-within-predictors`,
      'float32',
      props.predictors.length
    );
    const withinResponse = createTransientView(graph, `${id}-within-response`, 'float32', rows);
    const entityCenters = createTransientView(
      graph,
      `${id}-entity-centers`,
      'float32',
      getPanelWithinScratchLengths(entities, props.periodCount, props.predictorCount).meanLength
    );
    const runtimeZeroWGSL = `bitcast<u32>(centers[centersOffset + ${columns}u])`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-pooled-means`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'pooled-means',
        invocationCount: columns,
        bindings: [
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        // The first value is removed before summing, so the sum holds only the spread.
        body: `var reference = response[responseOffset]; if (index > 0u) { reference = predictors[predictorsOffset + index - 1u]; }
var total = 0.0; for (var row = 0u; row < ${rows}u; row++) { var value = response[responseOffset + row]; if (index > 0u) { value = predictors[predictorsOffset + row * ${columns - 1}u + index - 1u]; } total += value - reference; }
let center = reference + total / ${rows}.0; centers[centersOffset + index] = select(0.0, center, isFiniteBits(center));
// A zero the compiler cannot see: the runtime zero of getRoundedDifference.
if (index == 0u) { centers[centersOffset + ${columns}u] = 0.0; }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-dense-weights`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'validate-weights',
        invocationCount: entities,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: props.weights.weights, type: 'f32', access: 'read'},
          {name: 'dense', view: dense, type: 'f32', access: 'read_write'},
          ...(['count', 'requiredCount'] as const).flatMap(name =>
            props.weights.status?.[name]
              ? [
                  {
                    name,
                    view: props.weights.status[name]!,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []
          ),
          ...(props.weights.status?.overflow
            ? [
                {
                  name: 'overflow',
                  view: props.weights.status.overflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...(props.weights.status?.candidateOverflow
            ? [
                {
                  name: 'candidateOverflow',
                  view: props.weights.status.candidateOverflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `for (var column = 0u; column < ${entities}u; column++) { dense[denseOffset + index * ${entities}u + column] = 0.0; } var valid = ${props.weights.status?.overflow ? 'overflow[overflowOffset] == 0u' : 'true'} && ${props.weights.status?.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] == 0u' : 'true'};
${props.weights.status?.count ? `valid=valid && count[countOffset]==offsets[offsetsOffset+${entities}u]-offsets[offsetsOffset] && count[countOffset]<=${props.weights.neighbors.length}u;` : ''}
${props.weights.status?.requiredCount ? `valid=valid && requiredCount[requiredCountOffset]<=min(offsets[offsetsOffset+${entities}u]-offsets[offsetsOffset],${props.weights.neighbors.length}u);` : ''}
var total = 0.0; let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u]; if (end < begin || end > ${props.weights.neighbors.length}u) { valid = false; } else { for (var slot = begin; slot < end; slot++) { let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot]; if (neighbor >= ${entities}u || neighbor == index || !(weight >= 0.0) || !isFiniteBits(weight) || (slot > begin && neighbors[neighborsOffset + slot - 1u] >= neighbor)) { valid = false; } else { dense[denseOffset + index * ${entities}u + neighbor] = weight; total += weight; } } } if (!valid || abs(total-1.0) > 1.0e-5) { dense[denseOffset + index * ${entities}u] = getNaN(); }`
      }),
      ...(props.weights.status?.complete
        ? [
            createWGSLKernelNode(graph, {
              id: `${id}-validate-complete`,
              operation: 'GPUSpatialPanelLagMaximumLikelihood',
              invocationCount: entities,
              bindings: [
                {
                  name: 'complete',
                  view: props.weights.status.complete,
                  type: 'u32',
                  access: 'read'
                },
                {name: 'dense', view: dense, type: 'f32', access: 'read_write'}
              ],
              declarations: GET_NAN_WGSL,
              body: `if(complete[completeOffset]==0u){dense[denseOffset+index*${entities}u]=getNaN();}`
            })
          ]
        : []),
      ...getPanelWithinNodes(graph, {
        id: `${id}-within-pack`,
        entityCount: entities,
        periodCount: props.periodCount,
        predictorCount: props.predictorCount,
        predictors: props.predictors,
        response: props.response,
        output: {predictors: withinPredictors, response: withinResponse, means: entityCenters}
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pack`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'pack-panel-means',
        invocationCount: rows,
        bindings: [
          {name: 'withinPredictors', view: withinPredictors, type: 'f32', access: 'read'},
          {name: 'withinResponse', view: withinResponse, type: 'f32', access: 'read'},
          {name: 'entityCenters', view: entityCenters, type: 'f32', access: 'read'},
          {name: 'dense', view: dense, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'data', view: data, type: 'f32', access: 'read_write'}
        ],
        // Every predictor and the response are packed minus their pooled mean. The lag of the
        // response is packed as W y - mean(y): the lag of the centered response plus
        // mean(y) (rowSum - 1), with the row sum of the float32 weights carried exactly
        // (getAdditionRoundingError returns what each float32 addition drops: three weights of
        // float32(1/3) sum to 1.0000000298, and 3e-8 of a response near 10,000 is a tenth of a
        // residual of 0.004). Every difference from a pooled mean is rounded as written
        // (getRoundedDifference), so the sums over periods and over neighbours cannot be
        // regrouped into sums of the raw values less a multiple of the mean.
        // Each column is stored as its within deviation (centered value less its entity mean,
        // rounded as written) and its entity mean: the quasi-demeaned value is then
        // deviation + phi * mean, a sum whose terms do not cancel (see datum).
        declarations: `${ROUNDED_DIFFERENCE_WGSL}\n${ADDITION_ROUNDING_ERROR_WGSL}
fn getPanelCenteredEntityMean(entity:u32,component:u32,center:f32,runtimeZero:u32)->f32 {
 let start=entityCentersOffset+2u*(entity*${columns}u+component);
 return getRoundedDifference(entityCenters[start],center,runtimeZero)+entityCenters[start+1u];
}`,
        body: `let entity = index % ${entities}u; let period = index / ${entities}u; let base = dataOffset + index * ${width * 2}u;
let runtimeZero = ${runtimeZeroWGSL};
data[base]=0.0;data[base+${width}u]=1.0;
for (var column = 1u; column < ${columns}u; column++) {
 data[base+column]=withinPredictors[withinPredictorsOffset+index*${columns - 1}u+column-1u];
 data[base+${width}u+column]=getPanelCenteredEntityMean(entity,column,centers[centersOffset+column],runtimeZero);
}
let responseCenter = centers[centersOffset]; var rowSum = 0.0; var rowSumError = 0.0;
for (var neighbor = 0u; neighbor < ${entities}u; neighbor++) { let weight = dense[denseOffset + entity * ${entities}u + neighbor]; let nextRowSum = rowSum + weight; rowSumError += getAdditionRoundingError(rowSum, weight, nextRowSum); rowSum = nextRowSum; }
let lagShift = responseCenter * (getRoundedDifference(rowSum, 1.0, runtimeZero) + rowSumError);
let meanResponse=getPanelCenteredEntityMean(entity,0u,responseCenter,runtimeZero);
var meanLag=lagShift;var lagDeviation=0.0;
for(var neighbor=0u;neighbor<${entities}u;neighbor++){
 let weight=dense[denseOffset+entity*${entities}u+neighbor];
 let neighborMean=getPanelCenteredEntityMean(neighbor,0u,responseCenter,runtimeZero);
 let neighborDeviation=withinResponse[withinResponseOffset+period*${entities}u+neighbor];
 lagDeviation+=weight*neighborDeviation;meanLag+=weight*neighborMean;
}
data[base + ${columns}u] = withinResponse[withinResponseOffset+index]; data[base + ${columns + 1}u] = lagDeviation; data[base + ${width + columns}u] = meanResponse; data[base + ${width + columns + 1}u] = meanLag;`
      })
    ];
    const addFit = (unit: boolean, model: number) => {
      const predictors = unit ? columns - 1 : columns;
      const parameterCount = predictors + 1;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${unit ? 'unit' : 'random'}-fit`,
          operation: 'GPUSpatialPanelLagMaximumLikelihood',
          variant: unit ? 'unit-ml' : 'random-ml',
          // contract-allow: single-invocation -- reference fit bounded by 32 entities, 32 periods, 7 predictors, 128 sweeps and 64 search steps.
          invocationCount: 1,
          bindings: [
            {name: 'data', view: data, type: 'f32', access: 'read'},
            {name: 'dense', view: dense, type: 'f32', access: 'read'},
            {name: 'centers', view: centers, type: 'f32', access: 'read'},
            {name: 'results', view: results, type: 'f32', access: 'read_write'},
            {name: 'spatialProduct', view: spatialProduct, type: 'f32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${PANEL_LAG_PROFILE_WGSL}\n${PANEL_LAG_PROFILE_SCORE_WGSL}\n${getCholeskyWGSL(predictors)}\n${getEquilibratedCholeskyWGSL(predictors)}\n${getCholeskyWGSL(parameterCount)}\n${getEquilibratedCholeskyWGSL(parameterCount)}\n${getInverseWGSL(entities)}
// The quasi-demeaned value, value - (1 - phi) mean, as the within deviation plus phi times the
// entity mean. The pack node stores the deviation, value - mean, rounded as written
// (getRoundedDifference). Entity effects survive the pooled centering, so for large effects of
// opposing sign value and mean are both large and their difference small: formed here as
// value - (1 - phi) mean, the compiler regrouped (value - mean) - fitted as
// value - (mean + fitted), and every residual was a multiple of the float32 step of the entity
// effect. The two terms of this sum do not cancel.
fn datum(row:u32,column:u32,phi:f32)->f32 {let base=dataOffset+row*${width * 2}u;return data[base+column]+phi*data[base+${width}u+column];}
// The moment of two quasi-demeaned columns from the three sums of their deviations and means.
fn getQuasiMoment(deviationMoment:f32,crossMoment:f32,meanMoment:f32,phi:f32)->f32 {return deviationMoment+phi*(crossMoment+phi*meanMoment);}
// Residuals of the quasi-demeaned response (x) and of its lag (y) on the predictors, one row.
fn getRowResiduals(row:u32,phi:f32,firstBeta:ptr<function,array<f32,${predictors}>>,lagBeta:ptr<function,array<f32,${predictors}>>)->vec2<f32> {
 var residual=datum(row,${columns}u,phi);var lagResidual=datum(row,${columns + 1}u,phi);
 for(var column=0u;column<${predictors}u;column++){let value=datum(row,column+${unit ? 1 : 0}u,phi);residual-=value*(*firstBeta)[column];lagResidual-=value*(*lagBeta)[column];}
 return vec2<f32>(residual,lagResidual);
}
// The concentrated variance u(rho)'u(rho) / rows in its direct form. direct = [S, e1'e1, rho0,
// correction]: S is the residual sum of squares at its minimizer rho* = rho0 + correction, so
// u'u = S + e1'e1 (rho - rho*)^2, a sum of two non-negative terms. The former
// e0'e0 - 2 rho e0'e1 + rho^2 e1'e1 of three float32 sums is rounding once the model fits
// well: at u'u = 3e-7 e0'e0 the published sigma2 was 36% low or 96% high, at 3e-9 it was
// tens to hundreds of times too large, or negative.
fn getVariance(rho:f32,direct:vec4<f32>)->f32 {
 let offset=getRoundedDifference(rho,direct.z,${runtimeZeroWGSL})-direct.w;
 return (direct.x+direct.y*offset*offset)/${rows}.0;
}
// Trace from LU solves, one inverse column at a time: no second dense local matrix.
fn getPanelLagTrace(rho:f32)->vec2f {
 var matrix:array<f32,${entities * entities}>;
 for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){matrix[row*${entities}u+column]=select(0.0,1.0,row==column)-rho*dense[denseOffset+row*${entities}u+column];}}
 var factorization=factorDenseLU_${entities}(&matrix,${entities}u,0u,1.0e-10,0.0);
 if(factorization.status!=0u){return vec2f(0.0);}
 var trace=0.0;
 for(var column=0u;column<${entities}u;column++){
  var inverseColumn:array<f32,${entities}>;inverseColumn[column]=1.0;
  if(!solveDenseLU_${entities}(&matrix,&factorization,&inverseColumn)){return vec2f(0.0);}
  for(var other=0u;other<${entities}u;other++){trace+=dense[denseOffset+column*${entities}u+other]*inverseColumn[other];}
 }
 return vec2f(trace,1.0);
}
fn evaluatePanelObjective(rho:f32,direct:vec4<f32>,centered:bool)->f32 {
 let variance=getVariance(rho,direct); if (!(variance > 0.0)) {return 3.402823e38;}
 var matrix:array<f32,${entities * entities}>;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){matrix[row*${entities}u+column]=select(0.0,1.0,row==column)-rho*dense[denseOffset+row*${entities}u+column];}}
 let factorization=factorDenseLU_${entities}(&matrix,${entities}u,0u,1.0e-10,0.0);
 if(factorization.status!=0u){return 3.402823e38;}
 let determinant=factorization.logAbsDeterminant;
 let logarithm=select(log(variance),getPanelLagProfileLogVariance(rho,direct,${rows}.0,${runtimeZeroWGSL}),centered);
 return ${getWGSLFloatLiteral(rows / 2)}*logarithm-${props.periodCount}.0*determinant;
}
fn objective(rho:f32,direct:vec4<f32>)->f32 {return evaluatePanelObjective(rho,direct,false);}
fn profileObjective(rho:f32,direct:vec4<f32>)->f32 {return evaluatePanelObjective(rho,direct,true);
}`,
          body: `let base=resultsOffset+${model * stride}u;for(var slot=0u;slot<${stride}u;slot++){results[base+slot]=getNaN();}results[base+${statusOffset}u]=1.0;
for(var slot=0u;slot<${rows * width * 2}u;slot++){if(!isFiniteBits(data[dataOffset+slot])){return;}}for(var slot=0u;slot<${entities * entities}u;slot++){if(!isFiniteBits(dense[denseOffset+slot])){return;}}
var beta:array<f32,${predictors}>;var pooled:array<f32,${predictors * predictors}>;
// The moments of the normal equations, summed once from the within deviations d and the entity means m of every column: sum d d', sum (d m' + m d') and sum m m' for each predictor against each predictor, the response and its lag. At any phi the moment of two quasi-demeaned columns is their polynomial in phi (getQuasiMoment), so the normal equations are a smooth function of phi and the alternation settles; per-row values rounded afresh at every phi made the solve of nearly collinear predictors jitter (it did not settle below 1e-4 in phi for a squared cosine of 1 - 4e-6).
var deviationMoments:array<f32,${predictors * (predictors + 2)}>;var crossMoments:array<f32,${predictors * (predictors + 2)}>;var meanMoments:array<f32,${predictors * (predictors + 2)}>;
// Serial row sums lose small squares after large ones, biasing the conditional slope even when
// rho is accurate. Fixed pairwise reductions bound this error by the tree depth. Float Kahan
// compensation is not safe under Metal reassociation; use the shared explicit reduction order.
for(var column=0u;column<${predictors}u;column++){
 for(var other=0u;other<${predictors + 2}u;other++){
  var otherColumn=other+${unit ? 1 : 0}u;if(other>=${predictors}u){otherColumn=other+${columns - predictors}u;}
  let slot=column*${predictors + 2}u+other;
  ${[
    ['deviation', `data[rowBase+column+${unit ? 1 : 0}u]*data[rowBase+otherColumn]`],
    [
      'cross',
      `data[rowBase+column+${unit ? 1 : 0}u]*data[rowBase+${width}u+otherColumn]+data[rowBase+${width}u+column+${unit ? 1 : 0}u]*data[rowBase+otherColumn]`
    ],
    ['mean', `data[rowBase+${width}u+column+${unit ? 1 : 0}u]*data[rowBase+${width}u+otherColumn]`]
  ]
    .map(
      ([name, term]) =>
        `{${getPairwiseSumWGSL({count: `${rows}u`, name, setup: `let rowBase=dataOffset+${name}Item*${width * 2}u;`, term})}\n${name}Moments[slot]=${name};}`
    )
    .join('\n')}
 }}
var scale:array<f32,${predictors}>;results[base+${statusOffset}u]=2.0;
${
  unit
    ? ''
    : `// Only random effects need a pooled beta to initialize their first phi update.
for(var column=0u;column<${predictors}u;column++){beta[column]=getQuasiMoment(deviationMoments[column*${predictors + 2}u+${predictors}u],crossMoments[column*${predictors + 2}u+${predictors}u],meanMoments[column*${predictors + 2}u+${predictors}u],1.0);for(var other=0u;other<=column;other++){let slot=column*${predictors + 2}u+other;pooled[column*${predictors}u+other]=getQuasiMoment(deviationMoments[slot],crossMoments[slot],meanMoments[slot],1.0);}}
if(!factorEquilibrated_${predictors}(&pooled,&scale)){return;}solveEquilibrated_${predictors}(&pooled,&scale,&beta);`
}
var rho=0.1;var phi=${unit ? '0.0' : '0.1'};var belowLimit=false;var state=3u;var completed=0u;var change=0.0;var bracketWidth=2.0;var first=0.0;var cross=0.0;var last=0.0;var direct=vec4<f32>(0.0,0.0,0.0,0.0);var gram:array<f32,${predictors * predictors}>;
let runtimeZero=${runtimeZeroWGSL};
for(var iteration=0u;iteration<${unit ? 1 : (props.maximumIterations ?? 64)}u;iteration++){
 ${unit ? '' : `let oldPhi=phi;var within=0.0;var between=0.0;for(var row=0u;row<${rows}u;row++){let rowBase=dataOffset+row*${width * 2}u;var deviation=data[rowBase+${columns}u]-rho*data[rowBase+${columns + 1}u];var mean=data[rowBase+${width + columns}u]-rho*data[rowBase+${width + columns + 1}u];for(var column=0u;column<${predictors}u;column++){deviation-=beta[column]*data[rowBase+column];mean-=beta[column]*data[rowBase+${width}u+column];}within+=deviation*deviation;between+=mean*mean;}phi=1.0;belowLimit=false;if(between>0.0){let estimate=sqrt(within/(${props.periodCount - 1}.0*between));belowLimit=estimate<${getWGSLFloatLiteral(RANDOM_EFFECTS_MINIMUM_PHI)};phi=clamp(estimate,${getWGSLFloatLiteral(RANDOM_EFFECTS_MINIMUM_PHI)},1.0);}change=abs(phi-oldPhi);`}
 var firstBeta:array<f32,${predictors}>;var lagBeta:array<f32,${predictors}>;for(var slot=0u;slot<${predictors * predictors}u;slot++){gram[slot]=0.0;}
 for(var column=0u;column<${predictors}u;column++){let responseSlot=column*${predictors + 2}u+${predictors}u;firstBeta[column]=getQuasiMoment(deviationMoments[responseSlot],crossMoments[responseSlot],meanMoments[responseSlot],phi);lagBeta[column]=getQuasiMoment(deviationMoments[responseSlot+1u],crossMoments[responseSlot+1u],meanMoments[responseSlot+1u],phi);for(var other=0u;other<=column;other++){let slot=column*${predictors + 2}u+other;gram[column*${predictors}u+other]=getQuasiMoment(deviationMoments[slot],crossMoments[slot],meanMoments[slot],phi);}}
 var factor=gram;if(!factorEquilibrated_${predictors}(&factor,&scale)){state=2u;break;}solveEquilibrated_${predictors}(&factor,&scale,&firstBeta);solveEquilibrated_${predictors}(&factor,&scale,&lagBeta);first=0.0;cross=0.0;last=0.0;
 for(var row=0u;row<${rows}u;row++){let pair=getRowResiduals(row,phi,&firstBeta,&lagBeta);first+=pair.x*pair.x;cross+=pair.x*pair.y;last+=pair.y*pair.y;}
 // The direct form: r = e0 - rho0 e1 per row at rho0 = e0'e1 / e1'e1 (rounded as written, so the
 // sums stay sums of the small differences), then S = r'r - (r'e1)^2 / e1'e1 and the
 // correction r'e1 / e1'e1 of rho0. Without a lag residual the variance does not depend on rho.
 direct=vec4<f32>(first,0.0,0.0,0.0);
 if(last>0.0){let parameterZero=cross/last;var remainderSquares=0.0;var remainderCross=0.0;
  for(var row=0u;row<${rows}u;row++){let pair=getRowResiduals(row,phi,&firstBeta,&lagBeta);let remainder=getRoundedDifference(pair.x,parameterZero*pair.y,runtimeZero);remainderSquares+=remainder*remainder;remainderCross+=remainder*pair.y;}
  let correction=remainderCross/last;direct=vec4<f32>(remainderSquares-remainderCross*correction,last,parameterZero,correction);}
 // A nonpositive numerical minimum includes exact poles and squared-moment underflow.
 if(direct.x<=0.0&&abs(direct.z+direct.w)<=0.999){results[base+${statusOffset}u]=2.0;return;}
 var lower=-0.999;var upper=0.999;var left=lower+0.38196601125*(upper-lower);var right=lower+0.61803398875*(upper-lower);var leftScore=profileObjective(left,direct);var rightScore=profileObjective(right,direct);
 for(var step=0u;step<${props.maximumSearchIterations ?? 32}u;step++){
  var takeLeft=leftScore<=rightScore;
  if(step>=8u){let midpoint=lower+0.5*(upper-lower);let score=getPanelLagProfileScore(midpoint,direct,getPanelLagTrace(midpoint),${rows}.0,${props.periodCount}.0,runtimeZero);if(score.y!=0.0){takeLeft=score.x>=0.0;}}
  if(takeLeft){upper=right;right=left;rightScore=leftScore;left=lower+0.38196601125*(upper-lower);leftScore=profileObjective(left,direct);}else{lower=left;left=right;leftScore=rightScore;right=lower+0.61803398875*(upper-lower);rightScore=profileObjective(right,direct);}
 }
 rho=select(right,left,leftScore<=rightScore);bracketWidth=upper-lower;for(var column=0u;column<${predictors}u;column++){beta[column]=firstBeta[column]-rho*lagBeta[column];}completed=iteration+1u;
 if(${unit ? 'true' : `change <= ${getWGSLFloatLiteral(props.tolerance ?? 1e-5)} * min(1.0, phi / ${getWGSLFloatLiteral(RANDOM_EFFECTS_RELATIVE_PHI)})`}){state=0u;break;}
}
if((state==0u||state==3u)&&bracketWidth>${getWGSLFloatLiteral(props.tolerance ?? 1e-5)}){state=3u;}
// A phi below the lowest supported value is held at it: the fit is that of the constrained problem, not the maximum. Not converged (status 3), with its own boundary bit.
if(state==0u&&belowLimit){state=3u;}
for(var column=0u;column<${predictors}u;column++){results[base+column+${unit ? 1 : 0}u]=beta[column];}results[base+${columns}u]=rho;results[base+${columns + 1}u]=phi;
${
  unit
    ? ''
    : `// Back to the uncentered data: intercept = centered intercept + (1 - rho) mean(y) - slopes' mean(x).
results[base+${centeredInterceptOffset}u]=beta[0];var intercept=beta[0]+(1.0-rho)*centers[centersOffset];for(var column=1u;column<${predictors}u;column++){intercept-=centers[centersOffset+column]*beta[column];}results[base]=intercept;`
}
let variance=getVariance(rho,direct);if(!(variance>0.0)||!isFiniteBits(variance)){state=2u;}results[base+${summaryOffset}u]=variance;results[base+${summaryOffset + 1}u]=-objective(rho,direct)-${getWGSLFloatLiteral(rows / 2)}*(1.0+1.8378770664093453)${unit ? `-${getWGSLFloatLiteral(rows / 2)}*log(${getWGSLFloatLiteral(rows)})` : ''};results[base+${summaryOffset + 2}u]=f32(completed);results[base+${summaryOffset + 3}u]=change;results[base+${summaryOffset + 4}u]=bracketWidth;results[base+${summaryOffset + 5}u]=select(0.0,1.0,abs(rho)>.9989)+select(0.0,2.0,(phi>=.99999||belowLimit) && ${unit ? 'false' : 'true'})+select(0.0,4.0,belowLimit);results[base+${statusOffset}u]=f32(state);if(state!=0u||!(variance>0.0)){return;}
var spatialInverse:array<f32,${entities * entities}>;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){spatialInverse[row*${entities}u+column]=select(0.0,1.0,row==column)-rho*dense[denseOffset+row*${entities}u+column];}}
if(!inverse_${entities}(&spatialInverse)){results[base+${statusOffset}u]=2.0;return;}
for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column]=0.0;for(var other=0u;other<${entities}u;other++){spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column]+=dense[denseOffset+row*${entities}u+other]*spatialInverse[other*${entities}u+column];}}}
var traceSquare=0.0;var traceProduct=0.0;for(var row=0u;row<${entities}u;row++){for(var column=0u;column<${entities}u;column++){let value=spatialProduct[spatialProductOffset+${model * entities * entities}u+row*${entities}u+column];traceSquare+=value*spatialProduct[spatialProductOffset+${model * entities * entities}u+column*${entities}u+row];traceProduct+=value*value;}}
var information:array<f32,${parameterCount * parameterCount}>;for(var column=0u;column<${predictors}u;column++){for(var other=0u;other<=column;other++){information[column*${parameterCount}u+other]=gram[column*${predictors}u+other]/variance;}}
var predictionSquares=0.0;for(var row=0u;row<${rows}u;row++){let entity=row%${entities}u;let periodStart=row-entity;var lagPrediction=0.0;for(var neighbor=0u;neighbor<${entities}u;neighbor++){var prediction=0.0;for(var column=0u;column<${predictors}u;column++){prediction+=datum(periodStart+neighbor,column+${unit ? 1 : 0}u,phi)*beta[column];}lagPrediction+=spatialProduct[spatialProductOffset+${model * entities * entities}u+entity*${entities}u+neighbor]*prediction;}predictionSquares+=lagPrediction*lagPrediction;for(var column=0u;column<${predictors}u;column++){information[${predictors * parameterCount}u+column]+=datum(row,column+${unit ? 1 : 0}u,phi)*lagPrediction/variance;}}
information[${parameterCount * parameterCount - 1}u]=${props.periodCount}.0*(traceSquare+traceProduct)+predictionSquares/variance;var informationScale:array<f32,${parameterCount}>;if(!factorEquilibrated_${parameterCount}(&information,&informationScale)){results[base+${statusOffset}u]=2.0;return;}
var centered:array<f32,${parameterCount * parameterCount}>;for(var column=0u;column<${parameterCount}u;column++){var inverseColumn:array<f32,${parameterCount}>;inverseColumn[column]=1.0;solveEquilibrated_${parameterCount}(&information,&informationScale,&inverseColumn);for(var row=0u;row<${parameterCount}u;row++){centered[row*${parameterCount}u+column]=inverseColumn[row];}}
${
  unit
    ? ''
    : `// Covariance of the uncentered parameters: the intercept row of the map from centered to raw
// parameters is [1, -mean(x), -mean(y)] (see the intercept above); the other rows are unchanged.
var transform:array<f32,${parameterCount}>;transform[0]=1.0;transform[${parameterCount - 1}u]=-centers[centersOffset];
for(var column=1u;column<${predictors}u;column++){transform[column]=-centers[centersOffset+column];}
var mixed:array<f32,${parameterCount}>;var corner=0.0;for(var column=0u;column<${parameterCount}u;column++){for(var other=0u;other<${parameterCount}u;other++){mixed[column]+=transform[other]*centered[other*${parameterCount}u+column];}}
for(var column=0u;column<${parameterCount}u;column++){corner+=mixed[column]*transform[column];}
for(var column=1u;column<${parameterCount}u;column++){centered[column]=mixed[column];centered[column*${parameterCount}u]=mixed[column];}centered[0]=corner;`
}
for(var column=0u;column<${parameterCount}u;column++){for(var row=0u;row<${parameterCount}u;row++){results[base+${covarianceOffset}u+(row+${unit ? 1 : 0}u)*${columns + 1}u+column+${unit ? 1 : 0}u]=centered[row*${parameterCount}u+column];}}`
        })
      );
    };
    addFit(props.effects === 'unit', 0);
    if (props.hausman) addFit(true, 1);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'publish',
        invocationCount: 1,
        bindings: [
          {name: 'results', view: results, type: 'f32', access: 'read'},
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'covariance', view: props.output.covariance, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `for(var slot=0u;slot<${width}u;slot++){coefficients[coefficientsOffset+slot]=results[resultsOffset+slot];}for(var slot=0u;slot<${(columns + 1) ** 2}u;slot++){covariance[covarianceOffset+slot]=results[resultsOffset+${covarianceOffset}u+slot];}for(var slot=0u;slot<6u;slot++){summary[summaryOffset+slot]=results[resultsOffset+${summaryOffset}u+slot];}status[statusOffset]=u32(results[resultsOffset+${statusOffset}u]);`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-residuals`,
        operation: 'GPUSpatialPanelLagMaximumLikelihood',
        variant: 'quasi-demeaned-predictions',
        invocationCount: rows,
        bindings: [
          {name: 'data', view: data, type: 'f32', access: 'read'},
          {name: 'results', view: results, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'},
          {name: 'residuals', view: props.output.residuals, type: 'f32', access: 'read_write'}
        ],
        // Every quasi-demeaned difference is rounded as written, as in datum of the fit kernel:
        // the published residual is (y - (1 - phi) mean) - predicted, not y - ((1 - phi) mean +
        // predicted).
        declarations: GET_NAN_WGSL,
        body: `fitted[fittedOffset+index]=getNaN();residuals[residualsOffset+index]=getNaN();let state=u32(results[resultsOffset+${statusOffset}u]);if(state!=0u&&state!=3u){return;}let base=dataOffset+index*${width * 2}u;let phi=results[resultsOffset+${columns + 1}u];var predicted=results[resultsOffset+${columns}u]*(data[base+${columns + 1}u]+phi*data[base+${width + columns + 1}u]);for(var column=${props.effects === 'unit' ? 1 : 0}u;column<${columns}u;column++){var coefficient=results[resultsOffset+column];if(column==0u){coefficient=results[resultsOffset+${centeredInterceptOffset}u];}predicted+=coefficient*(data[base+column]+phi*data[base+${width}u+column]);}fitted[fittedOffset+index]=predicted+phi*centers[centersOffset];residuals[residualsOffset+index]=(data[base+${columns}u]+phi*data[base+${width + columns}u])-predicted;`
      })
    );
    if (props.hausman)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-hausman`,
          operation: 'GPUSpatialPanelLagMaximumLikelihood',
          variant: 'hausman',
          invocationCount: 1,
          bindings: [
            {name: 'results', view: results, type: 'f32', access: 'read'},
            {name: 'tests', view: props.hausman.tests, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.hausman.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${getInverseWGSL(columns)}\n${IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(columns)}`,
          body: `status[statusOffset]=1u;for(var slot=0u;slot<3u;slot++){tests[testsOffset+slot]=getNaN();}if(results[resultsOffset+${statusOffset}u]!=0.0||results[resultsOffset+${stride + statusOffset}u]!=0.0){return;}var difference:array<f32,${columns * columns}>;var contrast:array<f32,${columns}>;for(var row=0u;row<${columns}u;row++){contrast[row]=results[resultsOffset+${stride + 1}u+row]-results[resultsOffset+1u+row];for(var column=0u;column<${columns}u;column++){let offset=${covarianceOffset}u+(row+1u)*${columns + 1}u+column+1u;difference[row*${columns}u+column]=results[resultsOffset+offset]-results[resultsOffset+${stride}u+offset];}}status[statusOffset]=2u;var factor=difference;if(!choleskyFactor_${columns}(&factor)){return;}if(!inverse_${columns}(&difference)){return;}var statistic=0.0;for(var row=0u;row<${columns}u;row++){for(var column=0u;column<${columns}u;column++){statistic+=contrast[row]*difference[row*${columns}u+column]*contrast[column];}}tests[testsOffset]=statistic;tests[testsOffset+1u]=${columns + 1}.0;if(!(statistic>=0.0)||!isFiniteBits(statistic)){return;}tests[testsOffset+2u]=getChiSquareSurvivalValue(statistic,${columns + 1}.0);status[statusOffset]=0u;`
        })
      );
    return [...nodes, ...getPanelLagPoleNodes(graph, id, props)];
  }
}
/** Panel-specific in-place adapter over the shared scale-aware LU inverse. */
function getInverseWGSL(size: number): string {
  return `${getDenseLUWGSL(size)}
fn inverse_${size}(matrix:ptr<function,array<f32,${size * size}>>)->bool {
  var factorization=factorDenseLU_${size}(matrix,${size}u,0u,0.0,0.0);
  if(factorization.status!=0u){return false;}
  var inverse:array<f32,${size * size}>;
  if(!invertDenseLU_${size}(matrix,&factorization,&inverse)){return false;}
  (*matrix)=inverse;
  return true;
}`;
}
