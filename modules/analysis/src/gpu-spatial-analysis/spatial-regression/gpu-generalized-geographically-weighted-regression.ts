// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  createWGSLDispatchGate,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateCapacity,
  validateMaximumCount,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createGridIndexNodes,
  getRingNeighborDistancesWGSL,
  getDefaultGWRGridSize,
  type GridIndexViews
} from './gwr-grid-index';
import {getGWRGridScratchSizes} from './gwr-scalability-utils';
import {getGeneralizedSelectionNodes} from './generalized-gwr-selection';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  GENERALIZED_GWR_MAXIMUM_CANDIDATE_COUNT,
  GENERALIZED_GWR_MAXIMUM_SEARCH_ITERATIONS,
  GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS,
  GENERALIZED_GWR_GRID_NODE_ALLOWANCE,
  GENERALIZED_GWR_SEPARATION_ERROR_UNIT,
  GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT,
  LOCAL_REGRESSION_MAXIMUM_ITERATIONS,
  LOCAL_REGRESSION_MAXIMUM_GRID_CELL_COUNT,
  GENERALIZED_GWR_FOCUS_BATCH_ROWS,
  GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT,
  GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE,
  GENERALIZED_GWR_MAXIMUM_SHADER_FLOAT,
  GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES,
  getGeneralizedGWRScoreMergeLayout
} from './generalized-gwr-limits';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  getCandidateLoopWGSL,
  getEquilibratedCholeskyWGSL,
  getLocalDistanceWGSL,
  getLocalGridSupportWGSL,
  getLocalKernelWeightWGSL
} from './local-weighted-least-squares-wgsl';

/**
 * Largest `predictorCount` of the iterative local regressions (generalized GWR and multiscale
 * GWR); the design has one more column (the intercept). The generalized fit keeps three
 * `(predictorCount + 1)^2` float matrices per thread, which stays in private memory at this size.
 */
export const GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT = 24;
/** Largest row count of an iterative local regression without a grid index (dense neighborhood scan). */
export const GPU_LOCAL_REGRESSION_MAXIMUM_ROW_COUNT = 65536;
/** Shared reference limit for multiscale fits and indexed, untruncated Gaussian GGWR. */
export const GPU_LOCAL_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT = 1048576;

/** Iterative regression status shared by the bounded generalized and multiscale fits. */
export const GPU_ITERATIVE_REGRESSION_STATUS = {
  CONVERGED: 0,
  EXCLUDED: 1,
  SINGULAR: 2,
  ITERATION_LIMIT: 3,
  NUMERICAL_FAILURE: 4,
  /** Constant boundary response: logistic all zero/one, or Poisson all zero. */
  DEGENERATE_RESPONSE: 5,
  /** Complete/quasi separation witnessed by a local likelihood recession direction. */
  SEPARATION: 6
} as const;
export type GPUGeneralizedGeographicallyWeightedRegressionProps = {
  id?: string;
  family: 'poisson' | 'logistic';
  /**
   * Planar (default), or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` for
   * `(longitude, latitude)` degrees positions and great-circle distances in the context units
   * (bandwidths in meters on a sphere of `sphereRadius`). Great-circle contexts exclude rows with a
   * latitude outside [-90, 90] and cannot use `indexGridSize`.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Row-major predictors without intercept. */
  predictors: GraphDataView<'float32'>;
  predictorCount: number;
  /** Integer counts >=0 for Poisson; 0 or 1 for Bernoulli logistic. */
  response: GraphDataView<'float32'>;
  /** Additive linear-predictor offset: log exposure for Poisson. Defaults to zero. */
  offset?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** `[bandwidth, relativeCoefficientTolerance]`. Adaptive bandwidth is an integer neighbor count. */
  parameters: GraphDataView<'float32'>;
  kernel?: 'gaussian' | 'bisquare';
  /** Defaults to fixed distance. Adaptive counts complete-case rows, including the focus. */
  bandwidthMode?: 'fixed' | 'adaptive';
  /** Compile-time adaptive selection capacity, 1–128; defaults to 128. */
  maximumNeighborCount?: number;
  maximumIterations?: number;
  /** Exclude each focus from its own training neighborhood. */
  leaveOneOut?: boolean;
  /**
   * Dimensions `[columns, rows]` of an internal `GPUGridIndex` over the training rows, or omitted
   * for an automatic grid above 65,536 rows with planar finite-support kernels.
   * `false` keeps the bounded dense reference route. Compile-time, planar only.
   * Bisquare kernels (fixed or adaptive bandwidth), and Gaussian kernels with a `gaussianCutoff`,
   * then visit only the grid cells covering the bandwidth; the exact Gaussian kernel is unbounded
   * and still visits every row. Adaptive neighbor selection walks grid cells in rings around the
   * focus instead of scanning every row. Finite-support indexed fits accept up to 16,777,216 rows,
   * subject to device binding limits; untruncated Gaussian retains its 1,048,576-row reference cap.
   * The grid sums rows in cell order, so results differ from the dense scan in float32 rounding only.
   */
  indexGridSize?: readonly [number, number] | false;
  /**
   * Opt-in truncation radius, in bandwidths, of the Gaussian kernel: weights of rows farther than
   * `gaussianCutoff * bandwidth` are exactly 0, which lets the Gaussian kernel use `indexGridSize`
   * (fixed or adaptive bandwidth). The omitted weight of a row is at most `exp(-gaussianCutoff^2 / 2)`
   * (`exp(-12.5) = 3.7e-6` at 5, below float32 resolution of a unit weight at about 6). Omitted or
   * undefined is the exact unbounded Gaussian, which still visits every row. Ignored by the
   * bisquare kernel, whose support is already bounded.
   */
  gaussianCutoff?: number;
  /** Predict at finite query rows excluded by the training mask. Their responses are never used. */
  predictExcluded?: boolean;
  /** Training-only grid LOO/AICc or bounded AICc golden-section selection. */
  bandwidthSelection?: {
    output: {
      /** Grid: one per candidate. Golden section: two per search iteration; unused slots are NaN. */
      scores: GraphDataView<'float32'>;
      bandwidth: GraphDataView<'float32'>;
      /** 0 selected; 1 no valid candidate (including invalid runtime bounds). */
      status: GraphDataView<'uint32'>;
      /** Optional candidate history, same length/layout as scores. */
      evaluatedBandwidths?: GraphDataView<'float32'>;
    };
    /** Optional scalar search status; grid searches execute one pass. */
    status?: GPUIterationStatusPort;
  } & (
    | {
        method?: 'grid';
        bandwidths: readonly number[];
        criterion?: 'loo-deviance' | 'aicc';
      }
    | {
        method: 'golden-section';
        criterion?: 'aicc';
        /** Per-frame [lower bound, upper bound, absolute score tolerance]. Positive finite bounds, lower < upper. */
        parameters: GraphDataView<'float32'>;
        /** Compile-time search budget, 1–64 (default 24). */
        maximumIterations?: number;
      }
  );
  /** Model-specific scale-one GLM covariance and diagonal influence. */
  inference?: {
    standardErrors: GraphDataView<'float32'>;
    influence: GraphDataView<'float32'>;
  };
  output: {
    coefficients: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    status: {
      /** Per observation: 0 converged, 1 excluded, 2 singular/insufficient neighborhood,
       * 3 iteration limit (finite iterate published), 4 numerical failure,
       * 5 constant boundary response, 6 witnessed complete/quasi separation. */
      codes: GraphDataView<'uint32'>;
      /** Number of IRLS steps per observation. */
      iterationCount: GraphDataView<'uint32'>;
    };
  };
};

/**
 * Local Poisson log-link or Bernoulli logit regression with a fixed distance or adaptive neighbor bandwidth. Newton/IRLS
 * fits use positive spatial weights, Cholesky solves and bounded likelihood step-halving. Every
 * location is fitted independently on GPU, with complete-case rows and explicit local statuses.
 * Reference-tier dense neighborhood traversal is bounded by the row limits below. Optional grid
 * indexing narrows neighborhoods. No hidden ridge, CPU fallback or separation correction.
 * Inference uses scale-one GLM sandwich covariance and diagonal influence. Constant boundary
 * responses receive status 5; an exact column threshold or a decided recession direction
 * receives status 6. Opposing or numerically undecided moves cannot certify separation.
 * This is a sufficient certificate, not an exhaustive LP
 * test: uncertified separation can still reach another failure status. Budget exhaustion publishes
 * the current finite iterate with status 3; other failed/excluded outputs remain NaN.
 * Per-frame: parameters, data/mask/offset values and golden-section selection parameters.
 * Compile-time: family, kernel, bandwidthMode, capacities, iteration budgets, selection method,
 * grid dimensions, gaussianCutoff and optional views. Dense/untruncated-Gaussian AICc selection
 * is capped at 4,096 rows; finite-support indexed selection scores all training rows together.
 * Scores sum increasing rows within 256-row tiles, adjacent tiles within 256-tile groups when
 * needed, then groups in order. No invocation sums more than 256 partials.
 * Poisson linear predictors outside [-30,30] are numerical failure
 * rather than silently changing the likelihood. Limits: 65,536 dense rows, 16,777,216 indexed
 * finite-support rows (subject to binding limits), 1,048,576 indexed untruncated-Gaussian rows,
 * 1–24 predictors and 128 iterations. Dense traversal has quadratic work.
 * At most 65,536 focal states are reused across tiles and bandwidth candidates. Public inputs
 * and outputs remain single packed views; large coefficient surfaces can require a larger device binding limit.
 */
export class GPUGeneralizedGeographicallyWeightedRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  /** Upper bound including the published grid/sort node allowance. */
  readonly estimatedCommandNodeCount: number;
  /** Sphere radius of great-circle distances, or null for planar distances. */
  readonly sphereRadius: number | null;
  readonly props: Omit<GPUGeneralizedGeographicallyWeightedRegressionProps, 'indexGridSize'> & {
    indexGridSize?: readonly [number, number];
  };
  private readonly fittedScale: 'response' | 'link';

  constructor(inputProps: GPUGeneralizedGeographicallyWeightedRegressionProps);
  constructor(
    inputProps: GPUGeneralizedGeographicallyWeightedRegressionProps,
    fittedScale: 'response' | 'link' = 'response'
  ) {
    this.fittedScale = fittedScale;
    const props = {...inputProps, indexGridSize: inputProps.indexGridSize || undefined};
    this.id = props.id ?? 'generalized-geographically-weighted-regression';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    this.sphereRadius =
      this.spatialContext.metric === 'great-circle'
        ? getGPUSphereRadius(this.spatialContext)
        : null;
    if (this.sphereRadius !== null) {
      validateFiniteNumber(
        this.id,
        'spatialContext.sphereRadius',
        this.sphereRadius,
        Number.MIN_VALUE,
        GENERALIZED_GWR_MAXIMUM_SHADER_FLOAT
      );
    }
    if (
      inputProps.indexGridSize === undefined &&
      props.response.length > GPU_LOCAL_REGRESSION_MAXIMUM_ROW_COUNT &&
      this.sphereRadius === null &&
      (props.kernel === 'bisquare' || props.gaussianCutoff !== undefined)
    ) {
      props.indexGridSize = getDefaultGWRGridSize(props.response.length);
    }
    this.props = props;
    const hasScalableIndex =
      Boolean(props.indexGridSize) &&
      (props.kernel === 'bisquare' || props.gaussianCutoff !== undefined);
    validateRegressionInputs(
      this.id,
      props,
      this.sphereRadius !== null,
      hasScalableIndex
        ? GENERALIZED_GWR_MAXIMUM_INDEXED_ROW_COUNT
        : GPU_LOCAL_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT
    );
    validateRequirement(
      this.id,
      !props.bandwidthMode || ['fixed', 'adaptive'].includes(props.bandwidthMode),
      'bandwidthMode must be fixed or adaptive'
    );
    const capacity = props.maximumNeighborCount ?? GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT;
    validateCapacity(
      this.id,
      'maximumNeighborCount',
      capacity,
      1,
      GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT
    );
    validateRequirement(
      this.id,
      ['poisson', 'logistic'].includes(props.family),
      'family must be poisson or logistic'
    );
    if (props.offset) {
      validateSinglePackedView(props.offset, ['float32'], this.id);
      validateViewLength(this.id, 'offset', props.offset, props.response.length);
    }
    const rows = props.response.length;
    const selection = props.bandwidthSelection;
    let candidateCount = 0;
    if (selection) {
      validateRequirement(
        this.id,
        !props.leaveOneOut,
        'leaveOneOut must be false with bandwidthSelection'
      );
      validateRequirement(
        this.id,
        !selection.method || ['grid', 'golden-section'].includes(selection.method),
        'bandwidthSelection.method must be grid or golden-section'
      );
      if (selection.method === 'golden-section') {
        validateCapacity(
          this.id,
          'bandwidthSelection.maximumIterations',
          selection.maximumIterations ?? 24,
          1,
          GENERALIZED_GWR_MAXIMUM_SEARCH_ITERATIONS
        );
        validateSinglePackedView(selection.parameters, ['float32'], this.id);
        validateViewLength(this.id, 'bandwidthSelection.parameters', selection.parameters, 3);
        validateRequirement(
          this.id,
          !selection.criterion || selection.criterion === 'aicc',
          'golden-section bandwidthSelection.criterion must be aicc'
        );
        candidateCount = 2 * (selection.maximumIterations ?? 24);
      } else {
        validateCapacity(
          this.id,
          'bandwidthSelection.bandwidths.length',
          selection.bandwidths.length,
          2,
          GENERALIZED_GWR_MAXIMUM_CANDIDATE_COUNT
        );
        validateRequirement(
          this.id,
          !selection.criterion || ['loo-deviance', 'aicc'].includes(selection.criterion),
          'bandwidthSelection.criterion must be loo-deviance or aicc'
        );
        selection.bandwidths.forEach((value, index) => {
          validateFiniteNumber(
            this.id,
            `bandwidthSelection.bandwidths[${index}]`,
            value,
            Number.MIN_VALUE,
            GENERALIZED_GWR_MAXIMUM_SHADER_FLOAT
          );
          if (props.bandwidthMode === 'adaptive') {
            validateCapacity(
              this.id,
              `bandwidthSelection.bandwidths[${index}]`,
              value,
              1,
              capacity
            );
          }
        });
        candidateCount = selection.bandwidths.length;
      }
      if (
        !hasScalableIndex &&
        (selection.method === 'golden-section' || selection.criterion === 'aicc')
      ) {
        validateCapacity(
          this.id,
          'bandwidthSelection AICc rows',
          rows,
          1,
          GENERALIZED_GWR_MAXIMUM_SEARCH_ROWS
        );
      }
      for (const [name, view] of Object.entries(selection.output)) {
        if (name === 'evaluatedBandwidths' && !view) continue;
        validateSinglePackedView(view, [name === 'status' ? 'uint32' : 'float32'], this.id);
        validateViewLength(
          this.id,
          `bandwidthSelection.output.${name}`,
          view,
          name === 'scores' || name === 'evaluatedBandwidths' ? candidateCount : 1
        );
      }
      for (const [name, view] of Object.entries(selection.status ?? {})) {
        if (!view) continue;
        validateSinglePackedView(view, ['uint32'], this.id);
        validateViewLength(this.id, `bandwidthSelection.status.${name}`, view, 1);
      }
    }
    if (selection) {
      validateMaximumCount(
        this.id,
        'bandwidthSelection final score partials',
        getGeneralizedGWRScoreMergeLayout(rows).finalTileCount,
        GENERALIZED_GWR_MAXIMUM_FINAL_SCORE_TILES
      );
    }
    const tileCount = Math.ceil(rows / GENERALIZED_GWR_FOCUS_BATCH_ROWS);
    const gridAllowance = props.indexGridSize ? GENERALIZED_GWR_GRID_NODE_ALLOWANCE : 0;
    // Every candidate has an init, three nodes per focus tile, parameter/deviance/score,
    // at most two score merges, and one search update. Final fit has its own init/tiles.
    this.estimatedCommandNodeCount =
      gridAllowance +
      1 +
      3 * tileCount +
      (selection ? candidateCount * (3 * tileCount + 8) + 8 : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (props.inference) {
      validateSinglePackedView(props.inference.standardErrors, ['float32'], this.id);
      validateSinglePackedView(props.inference.influence, ['float32'], this.id);
      validateViewLength(
        this.id,
        'inference.standardErrors',
        props.inference.standardErrors,
        rows * (props.predictorCount + 1)
      );
      validateViewLength(this.id, 'inference.influence', props.inference.influence, rows);
    }
    for (const [name, view] of Object.entries(getOutputColumns(props.output))) {
      validateSinglePackedView(
        view,
        [name === 'coefficients' || name === 'fitted' ? 'float32' : 'uint32'],
        this.id
      );
      validateViewLength(
        this.id,
        `output.${name}`,
        view,
        rows * (name === 'coefficients' ? props.predictorCount + 1 : 1)
      );
    }
    const allOutputEntries: (readonly [string, GraphDataView | undefined])[] = [
      ...Object.entries(getOutputColumns(props.output)).map(
        ([name, view]) => [`output.${name}`, view] as const
      ),
      ...Object.entries(props.inference ?? {}).map(
        ([name, view]) => [`inference.${name}`, view] as const
      ),
      ...Object.entries(selection?.output ?? {}).map(
        ([name, view]) => [`bandwidthSelection.output.${name}`, view] as const
      ),
      ...Object.entries(selection?.status ?? {}).map(
        ([name, view]) => [`bandwidthSelection.status.${name}`, view] as const
      )
    ];
    const outputEntries = allOutputEntries.filter(
      (entry): entry is readonly [string, GraphDataView] => entry[1] !== undefined
    );
    const coBoundGroups = [
      [
        ...Object.keys(getOutputColumns(props.output)).map(name => `output.${name}`),
        ...Object.keys(props.inference ?? {}).map(name => `inference.${name}`)
      ],
      ['bandwidthSelection.output.scores', 'bandwidthSelection.output.evaluatedBandwidths'],
      [
        'bandwidthSelection.output.scores',
        'bandwidthSelection.output.bandwidth',
        'bandwidthSelection.output.status',
        ...Object.keys(selection?.status ?? {}).map(name => `bandwidthSelection.status.${name}`)
      ]
    ];
    for (let first = 0; first < outputEntries.length; first++) {
      const [firstName, firstView] = outputEntries[first];
      for (const [secondName, secondView] of outputEntries.slice(first + 1)) {
        const coBound = coBoundGroups.some(
          group => group.includes(firstName) && group.includes(secondName)
        );
        const firstRange = getViewBindingRange(firstView);
        const secondRange = getViewBindingRange(secondView);
        const overlaps =
          firstView.buffer === secondView.buffer &&
          (coBound
            ? firstRange.offset < secondRange.offset + secondRange.size &&
              secondRange.offset < firstRange.offset + firstRange.size
            : firstView.byteOffset < secondView.byteOffset + secondView.length * 4 &&
              secondView.byteOffset < firstView.byteOffset + firstView.length * 4);
        validateRequirement(
          this.id,
          !overlaps,
          `${firstName} and ${secondName} output ranges must not overlap`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      outputEntries.map(([, view]) => view),
      [
        ...getRegressionBindings(props).map(binding => binding.view),
        ...(props.bandwidthSelection?.method === 'golden-section'
          ? [props.bandwidthSelection.parameters]
          : [])
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[];
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    /** Shared training index and search gate; used only by the internal selection composition. @internal */
    options: {
      gridSetup?: ReturnType<typeof createLocalRegressionGrid<Parameters>>;
      gate?: WGSLDispatchGate<Parameters>;
      fits?: GraphDataView<'float32'>;
      focusGate?: WGSLDispatchGate<Parameters>;
    } = {}
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.response.length;
    const columns = props.predictorCount + 1;
    const iterations = props.maximumIterations ?? 32;
    const stride = columns + 3 + (props.inference ? columns + 1 : 0);
    const tileRows = Math.min(rows, GENERALIZED_GWR_FOCUS_BATCH_ROWS);
    const tileCount = Math.ceil(rows / tileRows);
    const selection = props.bandwidthSelection;
    const views = [
      ...getRegressionBindings(props).map(binding => binding.view),
      ...Object.values(getOutputColumns(props.output)),
      ...Object.values(props.inference ?? {}),
      ...Object.values(selection?.output ?? {}),
      ...Object.values(selection?.status ?? {}),
      ...(selection?.method === 'golden-section' ? [selection.parameters] : []),
      options.fits,
      options.focusGate?.view,
      options.gridSetup?.grid?.cellOffsets,
      options.gridSetup?.grid?.sortedIds
    ].filter((view): view is GraphDataView => view !== undefined);
    validateGraphViewsBelongToGraph(id, graph, views);
    // Binding ranges include the alignment prefix, even for caller subviews.
    for (const view of views) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'view binding',
        getViewBindingRange(view).size,
        'reduce rows or predictorCount'
      );
    }
    // These cover every derived buffer, including published grid/radix/scan scratch. Scan
    // levels shrink from the cell-offset or radix histogram range checked here.
    const scratch: (readonly [string, number])[] = [
      ['fit state', (tileRows * (selection ? 2 * columns + 4 : stride) + 1) * 4],
      ['focus dispatch', 12]
    ];
    if (selection)
      scratch.push(
        ['selection coefficients/errors', rows * columns * 4],
        ['selection fitted/codes/iterations/eligible/influence', rows * 4],
        ['selection tile scores', getGeneralizedGWRScoreMergeLayout(rows).tileCount * 16],
        ['selection merged scores', getGeneralizedGWRScoreMergeLayout(rows).mergedTileCount * 16],
        ['selection parameters/search/gate', 128]
      );
    // Use the shared grid's exact aligned extent layout and published primitive scratch bounds.
    scratch.push(...getGWRGridScratchSizes(rows, props.indexGridSize ?? null));
    for (const [name, bytes] of scratch)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce rows, predictorCount or indexGridSize'
      );
    if (selection) return getGeneralizedSelectionNodes(graph, this);
    const fits =
      options.fits ?? createTransientView(graph, `${id}-fits`, 'float32', tileRows * stride + 1);
    const gridSetup =
      options.gridSetup ??
      createLocalRegressionGrid(
        graph,
        id,
        props,
        this.sphereRadius,
        props.family === 'poisson'
          ? 'responseValue >= 0.0 && floor(responseValue) == responseValue'
          : 'responseValue == 0.0 || responseValue == 1.0'
      );
    const effective = {...props, response: gridSetup.props.response, mask: gridSetup.props.mask};
    const bindings = [
      ...getRegressionBindings(effective),
      ...(gridSetup.grid
        ? [
            {
              name: 'cellOffsets',
              view: gridSetup.grid.cellOffsets,
              type: 'u32' as const,
              access: 'read' as const
            },
            {
              name: 'sortedIds',
              view: gridSetup.grid.sortedIds,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ];
    const neighborhood = {sphereRadius: this.sphereRadius, grid: props.indexGridSize};
    // Adaptive neighbor selection walks grid rings around the focus instead of scanning every row.
    const hasRingSearch = props.bandwidthMode === 'adaptive' && Boolean(props.indexGridSize);
    // One extra slot holds the focus itself (distance 0) under leave-one-out.
    const maximumNeighborSlots =
      (props.maximumNeighborCount ?? GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT) +
      (props.leaveOneOut ? 1 : 0);
    const loop = (body: string) =>
      `{\n${getCandidateLoopWGSL(body, Boolean(props.indexGridSize), 'included(row) && responseValid(row)')}\n}`;
    const common = `${getRegressionWGSL(effective, neighborhood)}\n${neighborhood.grid ? '' : IS_FINITE_VALUE_WGSL}\n${getCholeskyWGSL(columns, GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT + 1)}\n${getEquilibratedCholeskyWGSL(columns)}
${hasRingSearch ? `const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighborSlots}u;\nconst SENTINEL: f32 = 3.0e38;\n${getRingNeighborDistancesWGSL()}` : ''}
fn responseValid(row: u32) -> bool {
  let responseValue = response[responseOffset + row];
  return ${props.family === 'poisson' ? 'responseValue >= 0.0 && floor(responseValue) == responseValue' : 'responseValue == 0.0 || responseValue == 1.0'};
}
// IRLS runs in a design basis centered on the focal row and scaled by the largest in-bandwidth
// deviation, so large mean-to-spread predictors (years, dollars, projected coordinates) keep the
// information matrix well conditioned in float32. Coefficients are mapped back at the end.
var<private> centerValues: array<f32, ${columns}>;
var<private> scaleValues: array<f32, ${columns}>;
fn designValue(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return (predictor(row, column) - centerValues[column]) / scaleValues[column];
}
fn linearPredictor(row: u32, coefficients: ptr<function, array<f32, ${columns}>>) -> f32 {
  var value = ${props.offset ? 'offsets[offsetsOffset + row]' : '0.0'};
  for (var column = 0u; column < ${columns}u; column++) { value += (*coefficients)[column] * designValue(row, column); }
  return value;
}
// Symmetric diagonal equilibration of the lower-triangular information, then Cholesky (shared with GWR).
fn factorInformation(information: ptr<function, array<f32, ${columns * columns}>>, equilibration: ptr<function, array<f32, ${columns}>>) -> bool {
  return factorEquilibrated_${columns}(information, equilibration);
}
fn solveInformation(information: ptr<function, array<f32, ${columns * columns}>>, equilibration: ptr<function, array<f32, ${columns}>>, vector: ptr<function, array<f32, ${columns}>>) {
  solveEquilibrated_${columns}(information, equilibration, vector);
}
fn meanResponse(linear: f32) -> f32 {
  return ${props.family === 'poisson' ? 'exp(linear)' : 'select(exp(linear) / (1.0 + exp(linear)), 1.0 / (1.0 + exp(-linear)), linear >= 0.0)'};
}
fn varianceResponse(linear: f32) -> f32 {
  ${props.family === 'poisson' ? 'return exp(linear);' : 'let tail = exp(-abs(linear)); return tail / ((1.0 + tail) * (1.0 + tail));'}
}
fn logLikelihood(focus: u32, bandwidth: f32, coefficients: ptr<function, array<f32, ${columns}>>) -> f32 {
  var likelihood = 0.0;
  let origin = getPosition(focus);
  ${loop(`let weight = spatialWeight(focus, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let linear = linearPredictor(row, coefficients);
    if (!isFiniteBits(linear) || abs(linear) > ${props.family === 'poisson' ? '30.0' : '80.0'}) { return getNaN(); }
    let responseValue = response[responseOffset + row];
    likelihood += weight * (${props.family === 'poisson' ? 'responseValue * linear - exp(linear)' : 'responseValue * linear - max(linear, 0.0) - log(1.0 + exp(-abs(linear)))'});`)}
  return likelihood;
}`;
    const nodes: GPUCommandNode<Parameters>[] = options.gridSetup ? [] : [...gridSetup.nodes];
    // Only the dispatch record changes between focal tiles, including the partial last tile.
    // Each candidate finishes with this record disarmed; skipped candidates cannot reuse stale work.
    const focusGate =
      options.focusGate ??
      createWGSLDispatchGate(graph, {
        id: `${id}-focus-dispatch`,
        operation: 'GPUGeneralizedGeographicallyWeightedRegression',
        shapes: [[tileRows, GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE]]
      });
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-focus-init`,
        operation: 'GPUGeneralizedGeographicallyWeightedRegression',
        invocationCount: 1,
        ...(options.gate?.getGate(1) ?? {}),
        bindings: [
          {name: 'fits', view: fits, type: 'f32', access: 'read_write'},
          focusGate.binding
        ],
        body: `fits[fitsOffset] = 0.0; ${focusGate.armWGSL}`
      })
    );
    for (let tile = 0; tile < tileCount; tile++) {
      const suffix = tileCount > 1 ? `-tile${tile}` : '';
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-fit${suffix}`,
          operation: 'GPUGeneralizedGeographicallyWeightedRegression',
          variant: props.family,
          invocationCount: tileRows,
          workgroupSize: GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE,
          ...focusGate.getGate(0),
          bindings: [...bindings, {name: 'fits', view: fits, type: 'f32', access: 'read_write'}],
          declarations: common,
          body: `let focus = index + u32(fits[fitsOffset]);
if (focus >= ROW_COUNT) { return; }
let base = fitsOffset + 1u + index * ${stride}u;
for (var column = 0u; column < ${stride}u; column++) { fits[base + column] = getNaN(); }
fits[base + ${columns + 1}u] = 1.0; fits[base + ${columns + 2}u] = 0.0;
if (${props.predictExcluded ? '!queryValid(focus)' : '!included(focus) || !responseValid(focus)'}) { return; }
var bandwidth = parameters[parametersOffset]; let tolerance = parameters[parametersOffset + 1u];
if (!(bandwidth > 0.0) || !isFiniteBits(bandwidth) || !(tolerance > 0.0) || !isFiniteBits(tolerance)) { fits[base + ${columns + 1}u] = 4.0; return; }
${
  props.bandwidthMode === 'adaptive'
    ? `
if (floor(bandwidth) != bandwidth || bandwidth > ${props.maximumNeighborCount ?? GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT}.0) { fits[base + ${columns + 1}u] = 4.0; return; }
let neighborCount = u32(bandwidth);
${
  hasRingSearch
    ? `// The grid holds the eligible rows; with leave-one-out the focus is one of them (distance 0), so
// the kth neighbor excluding it is entry k of the ring list.
let ringCount = neighborCount + select(0u, 1u, ${props.leaveOneOut ? 'included(focus) && responseValid(focus)' : 'false'});
var ringList: array<f32, ${maximumNeighborSlots}>;
selectNeighborDistances(focus, ringCount, &ringList);
if (ringList[ringCount - 1u] >= SENTINEL) { fits[base + ${columns + 1}u] = 2.0; return; }
bandwidth = sqrt(ringList[ringCount - 1u]) * 1.00001;`
    : `var distances: array<f32, ${props.maximumNeighborCount ?? GENERALIZED_GWR_MAXIMUM_NEIGHBOR_COUNT}>;
for (var neighbor = 0u; neighbor < neighborCount; neighbor++) { distances[neighbor] = 3.402823e38; }
var eligibleCount = 0u;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row) || !responseValid(row) ${props.leaveOneOut ? '|| row == focus' : ''}) { continue; }
  let distance = getDistance(getPosition(focus), getPosition(row));
  if (!isFiniteBits(distance)) { fits[base + ${columns + 1}u] = 4.0; return; }
  eligibleCount++;
  var candidateDistance = distance;
  for (var neighbor = 0u; neighbor < neighborCount; neighbor++) {
    if (candidateDistance < distances[neighbor]) { let previous = distances[neighbor]; distances[neighbor] = candidateDistance; candidateDistance = previous; }
  }
}
if (eligibleCount < neighborCount) { fits[base + ${columns + 1}u] = 2.0; return; }
bandwidth = distances[neighborCount - 1u] * 1.00001;`
}
// A zero kth distance has no positive distance scale; never invent a coordinate-unit floor.
if (!(bandwidth > 0.0)) { fits[base + ${columns + 1}u] = 2.0; return; }
if (!isFiniteBits(bandwidth)) { fits[base + ${columns + 1}u] = 4.0; return; }
`
    : ''
}
var positiveCount = 0u; var responseSum = 0.0; var zeroResponseCount = 0u;
var minimumZeroValues: array<f32, ${columns}>;
var maximumZeroValues: array<f32, ${columns}>;
var minimumPositiveValues: array<f32, ${columns}>;
var maximumPositiveValues: array<f32, ${columns}>;
let origin = getPosition(focus);
for (var column = 1u; column < ${columns}u; column++) {
  centerValues[column] = predictor(focus, column); scaleValues[column] = 0.0;
  minimumZeroValues[column] = bitcast<f32>(0x7f7fffffu);
  maximumZeroValues[column] = -bitcast<f32>(0x7f7fffffu);
  minimumPositiveValues[column] = bitcast<f32>(0x7f7fffffu);
  maximumPositiveValues[column] = -bitcast<f32>(0x7f7fffffu);
}
${loop(`if (!(spatialWeight(focus, row, bandwidth) > 0.0)) { continue; }
  let observed = response[responseOffset + row];
  positiveCount++; responseSum += observed;
  zeroResponseCount += select(0u, 1u, observed == 0.0);
  for (var column = 1u; column < ${columns}u; column++) {
    let value = predictor(row, column);
    scaleValues[column] = max(scaleValues[column], abs(value - centerValues[column]));
    if (observed == 0.0) {
      minimumZeroValues[column] = min(minimumZeroValues[column], value);
      maximumZeroValues[column] = max(maximumZeroValues[column], value);
    } else {
      minimumPositiveValues[column] = min(minimumPositiveValues[column], value);
      maximumPositiveValues[column] = max(maximumPositiveValues[column], value);
    }
  }`)}
for (var column = 1u; column < ${columns}u; column++) { if (!(scaleValues[column] > 0.0) || !isFiniteBits(scaleValues[column])) { scaleValues[column] = 1.0; } }
if (positiveCount < ${columns}u) { fits[base + ${columns + 1}u] = 2.0; return; }
if (responseSum == 0.0 ${props.family === 'logistic' ? '|| responseSum == f32(positiveCount)' : ''}) { fits[base + ${columns + 1}u] = 5.0; return; }
// Raw float32 comparisons certify a single-column threshold without rounded dot products.
// Logistic: the two class intervals touch or are disjoint, with at least one strict move.
// Poisson: every positive-count row is on the threshold; zero-count rows move to one side.
if (zeroResponseCount > 0u) {
  for (var column = 1u; column < ${columns}u; column++) {
    let minimumZero = minimumZeroValues[column]; let maximumZero = maximumZeroValues[column];
    let minimumPositive = minimumPositiveValues[column]; let maximumPositive = maximumPositiveValues[column];
    let separatesColumn = ${
      props.family === 'logistic'
        ? `((maximumZero <= minimumPositive) && (minimumZero < minimumPositive || maximumPositive > maximumZero)) ||
      ((maximumPositive <= minimumZero) && (minimumPositive < minimumZero || maximumZero > maximumPositive))`
        : `(minimumPositive == maximumPositive) &&
      ((minimumZero >= minimumPositive && maximumZero > minimumPositive) ||
       (maximumZero <= minimumPositive && minimumZero < minimumPositive))`
    };
    if (separatesColumn) { fits[base + ${columns + 1}u] = 6.0; return; }
  }
}
var coefficients: array<f32, ${columns}>;
var state = 3u; var completed = 0u;
for (var iteration = 0u; iteration < ${iterations}u; iteration++) {
  var information: array<f32, ${columns * columns}>; var gradient: array<f32, ${columns}>;
  var equilibration: array<f32, ${columns}>;
  var observations = 0u; var finite = true;
  ${loop(`if (!finite) { continue; }
    let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
    let linear = linearPredictor(row, &coefficients);
    if (!isFiniteBits(linear) || abs(linear) > ${props.family === 'poisson' ? '30.0' : '80.0'}) { finite = false; continue; }
    let mean = meanResponse(linear); let variance = varianceResponse(linear);
    observations++;
    var design: array<f32, ${columns}>;
    for (var column = 0u; column < ${columns}u; column++) { design[column] = designValue(row, column); }
    for (var column = 0u; column < ${columns}u; column++) {
      gradient[column] += spatial * design[column] * (response[responseOffset + row] - mean);
      for (var other = 0u; other <= column; other++) { information[column * ${columns}u + other] += spatial * variance * design[column] * design[other]; }
    }`)}
  completed = iteration + 1u;
  if (!finite) { state = 4u; break; }
  if (observations < ${columns}u || !factorInformation(&information, &equilibration)) { state = 2u; break; }
  solveInformation(&information, &equilibration, &gradient);
  // Certify only decided moves in the recession cone. A rounding bound rejects ambiguous
  // signs; it never permits an opposing move. Only structurally zero terms establish zero.
  var directionScale = 0.0; var finiteDirection = true;
  for (var column = 0u; column < ${columns}u; column++) {
    finiteDirection = finiteDirection && isFiniteBits(gradient[column]);
    directionScale = max(directionScale, abs(gradient[column]));
  }
  var separates = finiteDirection && directionScale > 0.0 && isFiniteBits(directionScale);
  var strictSeparation = false;
  ${loop(`if (!(spatialWeight(focus, row, bandwidth) > 0.0) || !separates) { continue; }
    var direction = 0.0; var directionMagnitude = 0.0; var standsStill = true;
    for (var column = 0u; column < ${columns}u; column++) {
      let coefficient = gradient[column] / directionScale;
      let term = coefficient * designValue(row, column);
      direction += term; directionMagnitude += abs(term);
      if (coefficient != 0.0) {
        if (column == 0u) { standsStill = false; }
        else if (predictor(row, column) != centerValues[column]) { standsStill = false; }
      }
    }
    let errorBound = ${columns + 8}.0 * ${GENERALIZED_GWR_SEPARATION_ERROR_UNIT} * directionMagnitude;
    let decided = isFiniteBits(direction) && isFiniteBits(directionMagnitude) && directionMagnitude > 0.0;
    let observed = response[responseOffset + row];
    ${
      props.family === 'logistic'
        ? `let margin = select(-direction, direction, observed == 1.0);
    let movesRight = decided && margin > errorBound;
    separates = standsStill || movesRight;
    strictSeparation = strictSeparation || movesRight;`
        : `let decreasesZero = decided && direction < -errorBound;
    separates = standsStill || (observed == 0.0 && decreasesZero);
    strictSeparation = strictSeparation || (observed == 0.0 && decreasesZero);`
    }`)}
  if (separates && strictSeparation) { state = 6u; break; }
  let previousLikelihood = logLikelihood(focus, bandwidth, &coefficients);
  var step = 1.0; var accepted = false; var candidate: array<f32, ${columns}>;
  for (var attempt = 0u; attempt < 16u; attempt++) {
    for (var column = 0u; column < ${columns}u; column++) { candidate[column] = coefficients[column] + step * gradient[column]; }
    let likelihood = logLikelihood(focus, bandwidth, &candidate);
    if (isFiniteBits(likelihood) && likelihood >= previousLikelihood - 1.0e-6 * max(1.0, abs(previousLikelihood))) { accepted = true; break; }
    step *= 0.5;
  }
  if (!accepted) { state = 4u; break; }
  var change = 0.0;
  for (var column = 0u; column < ${columns}u; column++) { change = max(change, abs(candidate[column] - coefficients[column]) / max(1.0, abs(coefficients[column]))); coefficients[column] = candidate[column]; }
  if (change <= tolerance) { state = 0u; break; }
}
if (state == 0u || state == 3u) {
  // Publish the current finite iterate even when the iteration budget is exhausted.
  // Map the centered, scaled coefficients back to the raw predictor basis.
  fits[base + ${columns}u] = ${this.fittedScale === 'link' ? 'linearPredictor(focus, &coefficients)' : 'meanResponse(linearPredictor(focus, &coefficients))'};
  var intercept = coefficients[0];
  for (var column = 1u; column < ${columns}u; column++) {
    fits[base + column] = coefficients[column] / scaleValues[column];
    intercept -= fits[base + column] * centerValues[column];
  }
  fits[base] = intercept;
  ${
    props.inference
      ? `
  var finalInformation: array<f32, ${columns * columns}>;
  var finalEquilibration: array<f32, ${columns}>;
  ${loop(`let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
    let variance = varianceResponse(linearPredictor(row, &coefficients));
    for (var column = 0u; column < ${columns}u; column++) {
      for (var other = 0u; other <= column; other++) { finalInformation[column * ${columns}u + other] += spatial * variance * designValue(row, column) * designValue(row, other); }
    }`)}
  if (factorInformation(&finalInformation, &finalEquilibration)) {
    // Raw coefficient c is row c of the affine map A from the centered basis, so its variance
    // solves with A^T e_c and projects each row on its centered design vector.
    for (var column = 0u; column < ${columns}u; column++) {
      var inverseColumn: array<f32, ${columns}>;
      if (column == 0u) {
        inverseColumn[0] = 1.0;
        for (var other = 1u; other < ${columns}u; other++) { inverseColumn[other] = -centerValues[other] / scaleValues[other]; }
      } else {
        inverseColumn[column] = 1.0 / scaleValues[column];
      }
      solveInformation(&finalInformation, &finalEquilibration, &inverseColumn);
      var coefficientVariance = 0.0;
      ${loop(`let spatial = spatialWeight(focus, row, bandwidth); if (!(spatial > 0.0)) { continue; }
        let variance = varianceResponse(linearPredictor(row, &coefficients));
        var projection = 0.0;
        for (var other = 0u; other < ${columns}u; other++) { projection += inverseColumn[other] * designValue(row, other); }
        coefficientVariance += spatial * spatial * variance * projection * projection;`)}
      fits[base + ${columns + 3}u + column] = sqrt(max(0.0, coefficientVariance));
    }
    // The focal design vector is (1, 0, ..., 0), so the leverage is the first inverse entry.
    var focalColumn: array<f32, ${columns}>;
    focalColumn[0] = 1.0;
    solveInformation(&finalInformation, &finalEquilibration, &focalColumn);
    fits[base + ${2 * columns + 3}u] = focalColumn[0] * spatialWeight(focus, focus, bandwidth) * varianceResponse(linearPredictor(focus, &coefficients));
  }
  `
      : ''
  }
}
fits[base + ${columns + 1}u] = f32(state); fits[base + ${columns + 2}u] = f32(completed);`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-publish${suffix}`,
          operation: 'GPUGeneralizedGeographicallyWeightedRegression',
          variant: 'publish',
          invocationCount: tileRows,
          workgroupSize: GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE,
          ...focusGate.getGate(0),
          bindings: [
            {name: 'fits', view: fits, type: 'f32', access: 'read'},
            ...Object.entries(getOutputColumns(props.output)).map(([name, view]) => ({
              name,
              view,
              type:
                name === 'coefficients' || name === 'fitted' ? ('f32' as const) : ('u32' as const),
              access: 'read_write' as const
            })),
            ...Object.entries(props.inference ?? {}).map(([name, view]) => ({
              name,
              view,
              type: 'f32' as const,
              access: 'read_write' as const
            }))
          ],
          body: `let row = index + u32(fits[fitsOffset]);
if (row >= ${rows}u) { return; }
let base = fitsOffset + 1u + index * ${stride}u;
for (var column = 0u; column < ${columns}u; column++) { coefficients[coefficientsOffset + row * ${columns}u + column] = fits[base + column]; }
fitted[fittedOffset + row] = fits[base + ${columns}u]; status[statusOffset + row] = u32(fits[base + ${columns + 1}u]); iterations[iterationsOffset + row] = u32(fits[base + ${columns + 2}u]);
${props.inference ? `for (var column = 0u; column < ${columns}u; column++) { standardErrors[standardErrorsOffset + row * ${columns}u + column] = fits[base + ${columns + 3}u + column]; } influence[influenceOffset + row] = fits[base + ${2 * columns + 3}u];` : ''}`
        })
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-focus-advance${suffix}`,
          operation: 'GPUGeneralizedGeographicallyWeightedRegression',
          invocationCount: 1,
          ...(options.gate?.getGate(1) ?? {}),
          bindings: [
            {name: 'fits', view: fits, type: 'f32', access: 'read_write'},
            focusGate.binding
          ],
          body: `fits[fitsOffset] += ${tileRows}.0;
          let remainingRows = ${rows}u - min(${rows}u, u32(fits[fitsOffset]));
          dispatch[dispatchOffset] = (min(remainingRows, ${tileRows}u) + ${GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE - 1}u) / ${GENERALIZED_GWR_FOCUS_WORKGROUP_SIZE}u;`
        })
      );
    }
    return nodes;
  }
}

/** Fits a bandwidth-selection candidate with its shared grid and dispatch gates. @internal */
export function getGeneralizedRegressionCompositionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUGeneralizedGeographicallyWeightedRegressionProps,
  options: {
    fittedScale?: 'response' | 'link';
    gridSetup?: ReturnType<typeof createLocalRegressionGrid<Parameters>>;
    gate?: WGSLDispatchGate<Parameters>;
    fits?: GraphDataView<'float32'>;
    focusGate?: WGSLDispatchGate<Parameters>;
  }
): readonly GPUCommandNode<Parameters>[] {
  const contributor = Reflect.construct(GPUGeneralizedGeographicallyWeightedRegression, [
    props,
    options.fittedScale
  ]) as GPUGeneralizedGeographicallyWeightedRegression;
  return Reflect.apply(contributor.getCommandNodes, contributor, [
    graph,
    options
  ]) as readonly GPUCommandNode<Parameters>[];
}

/** Shared bounded input contract for local iterative regression. @internal */
export type LocalRegressionInputs = Omit<
  GPUGeneralizedGeographicallyWeightedRegressionProps,
  'family' | 'output' | 'inference' | 'bandwidthSelection' | 'predictExcluded' | 'indexGridSize'
> & {indexGridSize?: readonly [number, number]};

/** Neighborhood options of {@link getRegressionWGSL}. @internal */
export type LocalRegressionNeighborhood = {
  /** Great-circle sphere radius, or null/omitted for planar distances. */
  sphereRadius?: number | null;
  /** Grid dimensions when the kernel binds `cellOffsets` and `sortedIds`. */
  grid?: readonly [number, number];
  /**
   * Emit only the distance, kernel and (optional) grid support: `getPosition`, `getDistance`,
   * `getLocalKernelWeight`, `spatialWeight`, `getIsBisquare` and `ROW_COUNT`, for kernels that
   * bind `positions` but not the predictors or response.
   */
  neighborhoodOnly?: boolean;
};

/**
 * Builds the grid index of a local regression. With a grid, the mask is folded into a masked
 * response (NaN for masked rows) so that the grid-binding kernels stay within eight storage
 * buffers; the returned `props` are the ones kernels must bind. Without `indexGridSize` nothing
 * is built and `props` are unchanged. @internal
 */
export function createLocalRegressionGrid<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: LocalRegressionInputs,
  sphereRadius: number | null,
  /** WGSL boolean over `responseValue` that is true for a usable response; defaults to true. */
  responseCheck = 'true'
): {
  props: LocalRegressionInputs;
  grid: GridIndexViews | undefined;
  nodes: GPUCommandNode<Parameters>[];
} {
  if (!props.indexGridSize) return {props, grid: undefined, nodes: []};
  const rows = props.response.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  let response = props.response;
  if (props.mask) {
    response = createTransientView(graph, `${id}-masked-response`, 'float32', rows);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-masked-response`,
        operation: 'GPULocalRegressionGrid',
        variant: 'masked-response',
        invocationCount: rows,
        bindings: [
          {name: 'source', view: props.response, type: 'f32', access: 'read'},
          {name: 'mask', view: props.mask, type: 'u32', access: 'read'},
          {name: 'masked', view: response, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: 'masked[maskedOffset + index] = select(getNaN(), source[sourceOffset + index], mask[maskOffset + index] != 0u);'
      })
    );
  }
  const effective: LocalRegressionInputs = {...props, response, mask: undefined};
  const gridPositions = createTransientView(graph, `${id}-grid-positions`, 'float32x2', rows);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-grid-positions`,
      operation: 'GPULocalRegressionGrid',
      variant: 'grid-positions',
      invocationCount: rows,
      bindings: [
        ...getRegressionBindings(effective),
        {name: 'gridPositions', view: gridPositions, type: 'f32', access: 'read_write'}
      ],
      declarations: `${getRegressionWGSL(effective, {sphereRadius})}
fn isTrainingRow(row: u32) -> bool {
  let responseValue = response[responseOffset + row];
  return included(row) && (${responseCheck});
}`,
      // The grid index ignores NaN points, so rows that are not training rows drop out of it.
      body: `let keep = isTrainingRow(index);
  let invalid = getNaN();
  gridPositions[gridPositionsOffset + 2u * index] = select(invalid, positions[positionsOffset + 2u * index], keep);
  gridPositions[gridPositionsOffset + 2u * index + 1u] = select(invalid, positions[positionsOffset + 2u * index + 1u], keep);`
    })
  );
  const grid = createGridIndexNodes(graph, {
    id: `${id}-grid`,
    operation: 'GPULocalRegressionGrid',
    positions: gridPositions,
    rowCount: rows,
    gridSize: props.indexGridSize
  });
  nodes.push(...grid.nodes);
  return {props: effective, grid, nodes};
}
export function validateRegressionInputs(
  id: string,
  props: LocalRegressionInputs,
  isGreatCircle = false,
  maximumIndexedRows = GPU_LOCAL_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT
): void {
  const rows = props.response.length;
  const maximumRows = props.indexGridSize
    ? maximumIndexedRows
    : GPU_LOCAL_REGRESSION_MAXIMUM_ROW_COUNT;
  validateCapacity(id, 'response rows', rows, 1, maximumRows);
  validateCapacity(
    id,
    'predictorCount',
    props.predictorCount,
    1,
    GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
  );
  if (props.indexGridSize) {
    const [gridColumns, gridRows] = props.indexGridSize;
    validateRequirement(id, !isGreatCircle, 'indexGridSize requires a planar spatialContext');
    validateCapacity(
      id,
      'indexGridSize columns',
      gridColumns,
      1,
      LOCAL_REGRESSION_MAXIMUM_GRID_CELL_COUNT
    );
    validateCapacity(
      id,
      'indexGridSize rows',
      gridRows,
      1,
      LOCAL_REGRESSION_MAXIMUM_GRID_CELL_COUNT
    );
    validateCapacity(
      id,
      'indexGridSize cells',
      gridColumns * gridRows,
      1,
      LOCAL_REGRESSION_MAXIMUM_GRID_CELL_COUNT
    );
  }
  const iterations = props.maximumIterations ?? 32;
  validateCapacity(id, 'maximumIterations', iterations, 1, LOCAL_REGRESSION_MAXIMUM_ITERATIONS);
  validateRequirement(
    id,
    !props.kernel || ['gaussian', 'bisquare'].includes(props.kernel),
    'kernel must be gaussian or bisquare'
  );
  if (props.gaussianCutoff !== undefined) {
    validateFiniteNumber(
      id,
      'gaussianCutoff',
      props.gaussianCutoff,
      Number.MIN_VALUE,
      GENERALIZED_GWR_MAXIMUM_SHADER_FLOAT
    );
  }
  validateSinglePackedView(props.positions, ['float32x2'], id);
  validateSinglePackedView(props.predictors, ['float32'], id);
  validateSinglePackedView(props.response, ['float32'], id);
  validateSinglePackedView(props.parameters, ['float32'], id);
  validateViewLength(id, 'positions', props.positions, rows);
  validateViewLength(id, 'predictors', props.predictors, rows * props.predictorCount);
  validateRequirement(id, props.parameters.length >= 2, 'parameters must contain at least 2 rows');
  if (props.mask) {
    validateSinglePackedView(props.mask, ['uint32'], id);
    validateViewLength(id, 'mask', props.mask, rows);
  }
}
export function getRegressionBindings(props: LocalRegressionInputs): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
    {name: 'response', view: props.response, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    ...(props.offset
      ? [{name: 'offsets', view: props.offset, type: 'f32' as const, access: 'read' as const}]
      : []),
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : [])
  ];
}
export function getRegressionWGSL(
  props: LocalRegressionInputs,
  neighborhood: LocalRegressionNeighborhood = {}
): string {
  const sphereRadius = neighborhood.sphereRadius ?? null;
  const positionValid = (row: string) =>
    `isFiniteBits(positions[positionsOffset + 2u * ${row}]) && isFiniteBits(positions[positionsOffset + 2u * ${row} + 1u])${
      sphereRadius === null ? '' : ` && abs(positions[positionsOffset + 2u * ${row} + 1u]) <= 90.0`
    }`;
  const neighborhoodWGSL = `fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
${getLocalDistanceWGSL(sphereRadius)}
${getLocalKernelWeightWGSL(props.gaussianCutoff ?? null)}
${neighborhood.grid ? getLocalGridSupportWGSL(neighborhood.grid, props.gaussianCutoff ?? null) : ''}
fn spatialWeight(focus: u32, row: u32, bandwidth: f32) -> f32 {
  ${props.leaveOneOut ? 'if (focus == row) { return 0.0; }' : ''}
  return getLocalKernelWeight(getDistance(getPosition(focus), getPosition(row)), bandwidth, getIsBisquare());
}`;
  const header = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const ROW_COUNT: u32 = ${props.response.length}u;
fn getIsBisquare() -> bool { return ${props.kernel === 'bisquare' ? 'true' : 'false'}; }`;
  if (neighborhood.neighborhoodOnly) return `${header}\n${neighborhoodWGSL}`;
  return `${header}
fn queryValid(row: u32) -> bool {
  ${props.offset ? 'if (!isFiniteBits(offsets[offsetsOffset + row])) { return false; }' : ''}
  if (!(${positionValid('row')})) { return false; }
  for (var column = 0u; column < ${props.predictorCount}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${props.predictorCount}u + column])) { return false; } }
  return true;
}
fn included(row: u32) -> bool {
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  ${props.offset ? 'if (!isFiniteBits(offsets[offsetsOffset + row])) { return false; }' : ''}
  if (!isFiniteBits(response[responseOffset + row]) || !(${positionValid('row')})) { return false; }
  for (var column = 0u; column < ${props.predictorCount}u; column++) { if (!isFiniteBits(predictors[predictorsOffset + row * ${props.predictorCount}u + column])) { return false; } }
  return true;
}
fn predictor(row: u32, column: u32) -> f32 {
  if (column == 0u) { return 1.0; }
  return predictors[predictorsOffset + row * ${props.predictorCount}u + column - 1u];
}
${neighborhoodWGSL}`;
}

function getOutputColumns(output: GPUGeneralizedGeographicallyWeightedRegressionProps['output']) {
  const {status, ...columns} = output;
  return {...columns, status: status.codes, iterations: status.iterationCount};
}
