// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateNetworkViewBindings, validateNetworkWritableViews} from '../network-view-utils';

import {
  createTransientView,
  GPUTranspose,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateRequirement,
  validateCapacity,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUNetworkCostMatrixResult} from './gpu-network-cost-matrix';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUCatchmentAccessibility,
  type GPUCatchmentMethod
} from '../../gpu-spatial-analysis/catchment-accessibility/index';
import {GPUNetworkCostWeights} from '../network-cost-weights/index';
import {
  createAccessibilityColumnGatherNode,
  createAccessibilityRowReduceNode
} from './network-accessibility-passes';

const OPERATION = 'GPUNetworkAccessibility';

/** Inputs to the directed-trip-preserving matrix catchment adapter. */
export type GPUNetworkCatchmentAdapterProps = {
  id?: string;
  matrix: GPUNetworkCostMatrixResult;
  /** Which storage axis indexes facilities; this never changes trip direction. */
  facilityAxis: 'rows' | 'columns';
  /** Intended trips, checked against the matrix direction before rearranging storage. */
  trips: 'demand-to-facility' | 'facility-to-demand';
  supply: GraphDataView<'float32'>;
  demand: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  output: GraphDataView<'float32'>;
  ratios?: GraphDataView<'float32'>;
  complete: GraphDataView<'uint32'>;
  method?: GPUCatchmentMethod;
};

/**
 * Adds catchment scoring for a labelled matrix. A demand-major matrix is transposed with public
 * GPUTranspose, preserving its directed trips. Axis maps remain borrowed in the returned matrix.
 * Requires convergence evidence; noded source evidence is combined with it on-device.
 */
export function addNetworkCatchmentAdapter<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkCatchmentAdapterProps
): {matrix: GPUNetworkCostMatrixResult; contributor: GPUNetworkAccessibility} {
  const captured = snapshotConstructionFields(props, [
    'id',
    'matrix',
    'facilityAxis',
    'trips',
    'supply',
    'demand',
    'parameters',
    'output',
    'ratios',
    'complete',
    'method'
  ]);
  props = Object.freeze({
    ...captured,
    matrix: snapshotConstructionFields(captured.matrix, [
      'values',
      'rowCount',
      'columnCount',
      'startNodes',
      'startRows',
      'startsPerRow',
      'targetNodes',
      'direction',
      'converged',
      'sourceComplete',
      'costLimit'
    ])
  });
  const id = props.id ?? 'network-catchment-adapter';
  const {matrix} = props;
  validateCapacity(id, 'matrix.rowCount', matrix.rowCount);
  validateCapacity(id, 'matrix.columnCount', matrix.columnCount);
  validateSinglePackedView(matrix.values, ['float32'], `${id} matrix.values`);
  validateRequirement(
    id,
    matrix.values.length === matrix.rowCount * matrix.columnCount,
    `matrix.values length must equal rowCount * columnCount (${matrix.rowCount * matrix.columnCount}); received ${matrix.values.length}`
  );
  validateRequirement(
    id,
    matrix.direction === 'start-to-target' || matrix.direction === 'target-to-start',
    "matrix.direction must be 'start-to-target' or 'target-to-start'"
  );
  validateRequirement(
    id,
    props.facilityAxis === 'rows' || props.facilityAxis === 'columns',
    "facilityAxis must be 'rows' or 'columns'"
  );
  validateRequirement(
    id,
    props.trips === 'demand-to-facility' || props.trips === 'facility-to-demand',
    "trips must be 'demand-to-facility' or 'facility-to-demand'"
  );
  const demandToFacility =
    props.facilityAxis === 'columns'
      ? matrix.direction === 'start-to-target'
      : matrix.direction === 'target-to-start';
  validateRequirement(
    id,
    demandToFacility === (props.trips === 'demand-to-facility'),
    'trips must match matrix.direction and facilityAxis; transposing storage does not reverse trips'
  );
  validateRequirement(id, Boolean(matrix.converged), 'matrix.converged is required');
  for (const [name, view] of [
    ['matrix.converged', matrix.converged],
    ['matrix.sourceComplete', matrix.sourceComplete]
  ] as const) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      validateRequirement(
        id,
        view.length === 1,
        `${name} length must equal 1; received ${view.length}`
      );
    }
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [props.output, props.ratios, props.complete],
    [
      matrix.values,
      matrix.converged,
      matrix.sourceComplete,
      matrix.costLimit,
      props.supply,
      props.demand,
      props.parameters
    ]
  );
  validateRequirement(
    id,
    props.supply.length === (props.facilityAxis === 'rows' ? matrix.rowCount : matrix.columnCount),
    'supply length must equal the facility axis count'
  );
  validateRequirement(
    id,
    props.demand.length === (props.facilityAxis === 'rows' ? matrix.columnCount : matrix.rowCount),
    'demand length must equal the demand axis count'
  );
  // Validate the scorer against the same logical facility-major shape before allocating
  // transpose/certificate resources. Its constructor reads shapes, not matrix element order.
  const scorerProps: GPUNetworkAccessibilityProps = {
    id,
    costs: matrix.values,
    costsComplete: matrix.converged!,
    costLimit: matrix.costLimit,
    status: {complete: props.complete},
    orientation: 'opportunity-rows',
    opportunityWeights: props.supply,
    parameters: props.parameters,
    catchment: {
      method: props.method,
      demand: props.demand,
      output: props.output,
      ratios: props.ratios
    }
  };
  new GPUNetworkAccessibility(scorerProps);
  validateNetworkViewBindings(
    id,
    graph,
    [
      matrix.values,
      matrix.converged,
      matrix.sourceComplete,
      matrix.costLimit,
      props.supply,
      props.demand,
      props.parameters,
      props.output,
      props.ratios,
      props.complete
    ],
    {
      facilityMajor: props.facilityAxis === 'columns' ? matrix.values.length * 4 : 0,
      catchmentPairs: matrix.values.length * 4,
      catchmentOffsets: (Math.max(matrix.rowCount, matrix.columnCount) + 1) * 4
    }
  );
  let costs = matrix.values;
  if (props.facilityAxis === 'columns') {
    costs = createTransientView(graph, `${id}-facility-major`, 'float32', matrix.values.length);
    graph.add(
      new GPUTranspose({
        id: `${id}-transpose`,
        input: matrix.values,
        output: costs,
        rows: matrix.rowCount,
        columns: matrix.columnCount
      })
    );
  }
  let costsComplete = matrix.converged!;
  if (matrix.sourceComplete) {
    costsComplete = createTransientView(graph, `${id}-costs-complete`, 'uint32', 1);
    graph.add(
      createWGSLKernelNode(graph, {
        id: `${id}-costs-complete`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'search', view: matrix.converged!, type: 'u32', access: 'read'},
          {name: 'source', view: matrix.sourceComplete, type: 'u32', access: 'read'},
          {name: 'complete', view: costsComplete, type: 'u32', access: 'read_write'}
        ],
        body: 'complete[completeOffset] = u32(search[searchOffset] != 0u && source[sourceOffset] != 0u);'
      })
    );
  }
  const contributor = new GPUNetworkAccessibility({
    ...scorerProps,
    costs,
    costsComplete
  });
  graph.add(contributor);
  return {matrix, contributor};
}

/** Number of `float32` words in the per-frame parameter view of {@link GPUNetworkAccessibility}. */
export const GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH = 4;

/**
 * Distance-decay function of {@link GPUNetworkAccessibility}:
 * - `'none'`: every opportunity within the threshold counts fully (classic 2SFCA catchment).
 * - `'exponential'`: negative exponential `exp(-beta * cost)`.
 * - `'power'`: power decay `max(cost, minimumCost) ^ -beta`.
 */
export type GPUNetworkAccessibilityDecay = 'none' | 'exponential' | 'power';

/** Per-frame scoring settings of {@link GPUNetworkAccessibility}. */
export type GPUNetworkAccessibilitySettings = {
  /**
   * Cost cutoff: only matrix entries with a finite cost `<= threshold` contribute. `Infinity`
   * keeps every reached entry. Must not exceed the `costLimit` the matrix was built with.
   */
  threshold: number;
  /** Power-decay floor so zero costs stay finite. Must be positive and finite. Defaults to 1. */
  minimumCost?: number;
} & (
  | {decay?: 'none'; beta?: number}
  | {
      /** Decay applied to gravity and catchment scoring. Exponential and power require beta. */
      decay: GPUNetworkAccessibilityDecay;
      /** Finite nonnegative rate, also finite after float32 conversion, in inverse cost units (exponential) or exponent (power). Zero explicitly selects constant weights. */
      beta: number;
    }
);

const DECAY_CODES: Record<GPUNetworkAccessibilityDecay, number> = {
  none: 0,
  exponential: 1,
  power: 2
};

/**
 * Packs per-frame scoring settings into the four `float32` words read by
 * {@link GPUNetworkAccessibility}: threshold, decay code, beta, minimum cost.
 *
 * @throws If `threshold` is NaN, a rate-dependent decay omits `beta`, `beta` is negative or not finite, or `minimumCost` is not a
 * positive finite number (`max(0, minimumCost) ^ -beta` would be infinite).
 */
export function getGPUNetworkAccessibilityParameterValues(
  settings: GPUNetworkAccessibilitySettings
): Float32Array {
  const suppliedBeta = settings.beta;
  const minimumCost = settings.minimumCost ?? 1;
  const threshold = settings.threshold;
  const decay = settings.decay ?? 'none';
  const beta = suppliedBeta === undefined && decay === 'none' ? 0 : suppliedBeta;
  validateRequirement(
    OPERATION,
    Object.hasOwn(DECAY_CODES, decay),
    `decay must be 'none', 'exponential' or 'power'; received ${String(decay)}`
  );
  validateRequirement(
    OPERATION,
    !Number.isNaN(threshold),
    `threshold must not be NaN; received ${threshold}`
  );
  validateRequirement(
    OPERATION,
    beta !== undefined && Number.isFinite(beta) && Number.isFinite(Math.fround(beta)) && beta >= 0,
    `beta must be a non-negative finite number after float32 conversion for decay '${decay}'; received ${String(suppliedBeta)}; write beta: 0 for constant weights`
  );
  validateRequirement(
    OPERATION,
    Number.isFinite(minimumCost) && minimumCost > 0,
    `minimumCost must be a positive finite number; received ${minimumCost}`
  );
  return Float32Array.from([threshold, DECAY_CODES[decay], beta!, minimumCost]);
}

/**
 * Floating catchment area (2SFCA or 3SFCA) inputs and outputs of {@link GPUNetworkAccessibility}.
 * Only for `orientation: 'opportunity-rows'`, where rows are facilities and `opportunityWeights`
 * holds their supply.
 */
export type GPUNetworkAccessibilityCatchment = {
  /**
   * `'2sfca'` (default) or `'3sfca'`, the method of `GPUCatchmentAccessibility`, which runs on the
   * decayed matrix entries within the threshold.
   */
  method?: GPUCatchmentMethod;
  /** Demand (for example population) per column node. */
  demand: GraphDataView<'float32'>;
  /**
   * Per-node accessibility. 2SFCA: `A_i = sum_j R_j f(c_ij)` over facilities within the threshold.
   */
  output: GraphDataView<'float32'>;
  /**
   * Optional per-facility supply-to-demand ratio, 2SFCA `R_j = S_j / sum_i P_i f(c_ij)`, 0 when no
   * demand is in its catchment.
   */
  ratios?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUNetworkAccessibility}.
 *
 * Compile-time: matrix shape, orientation, and which outputs exist. Per-frame: the contents of
 * `costs`, weights, demand, and `parameters` (threshold, decay, beta, minimum cost).
 */
export type GPUNetworkAccessibilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-accessibility'`. */
  id?: string;
  /**
   * Row-major `[rowCount x nodeCount]` cost matrix, for example the `costs` of
   * `GPUNetworkCostMatrix`. Unreached entries are `+Infinity`. A matrix whose search stopped at its
   * round budget also writes `+Infinity` (or an overestimate) for entries it did not finish, so
   * pass its completeness word as `costsComplete` and request `status.complete`.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Optional one-row completeness word of `costs`, for example the `output.status.converged` of
   * `GPUNetworkCostMatrix`: nonzero means every entry is final. Required by `status.complete`.
   * Without a public `status.complete`, zero suppresses every requested score and catchment ratio.
   */
  costsComplete?: GraphDataView<'uint32'>;
  /**
   * Optional one-row cost limit the matrix was built with (the `costLimit` view of
   * `GPUNetworkCostMatrix`). `status.complete` is 0 when the per-frame threshold exceeds it,
   * because entries above the limit are `+Infinity` and would score as unreached. Supplied route
   * loss suppresses all requested scores and catchment ratios with NaN when status.complete is
   * omitted. With a public certificate, outputs retain their incumbent values. A NaN limit
   * conservatively fails the certificate, even though the matrix producer treats NaN as no cutoff.
   */
  costLimit?: GraphDataView<'float32'>;
  /** Optional GPU-written status words. */
  status?: {
    /**
     * Optional one-row flag: 1 when `costsComplete` is nonzero and the threshold does not exceed
     * `costLimit` (when given), so every requested score is complete; otherwise 0. Requires
     * `costsComplete`.
     */
    complete?: GraphDataView<'uint32'>;
  };
  /**
   * Which side the matrix rows are on. Defaults to `'opportunity-rows'`.
   * - `'opportunity-rows'`: rows are opportunity (or facility) searches over the reverse network,
   *   columns are origin nodes. `opportunityWeights` has one row per matrix row and outputs have one
   *   row per node.
   * - `'origin-rows'`: rows are origin searches over the forward network, columns are nodes that
   *   hold opportunities. `opportunityWeights` has one row per node and outputs one row per matrix
   *   row.
   */
  orientation?: 'opportunity-rows' | 'origin-rows';
  /** Opportunity weight (jobs, beds, supply) per opportunity row or node. */
  opportunityWeights: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: `GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH` `float32` words written with
   * {@link getGPUNetworkAccessibilityParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /** Optional cumulative opportunities within the threshold, per origin. */
  cumulative?: GraphDataView<'float32'>;
  /** Optional gravity accessibility `sum_j O_j f(c_ij)` within the threshold, per origin. */
  gravity?: GraphDataView<'float32'>;
  /**
   * Optional floating catchment area. It is a composition: two `GPUNetworkCostWeights` turn the
   * matrix into facility weights and their transpose, and `GPUCatchmentAccessibility` scores them.
   * Each weights holds `rowCount * nodeCount` transient slots (the worst case, so it never
   * overflows); for large matrices build sparser weights yourself and run those two contributors.
   */
  catchment?: GPUNetworkAccessibilityCatchment;
} & (
  | {cumulative: GraphDataView<'float32'>}
  | {gravity: GraphDataView<'float32'>}
  | {catchment: GPUNetworkAccessibilityCatchment}
);

/**
 * Scores per-origin accessibility from a retained network cost matrix: cumulative opportunities
 * within a per-frame cost threshold, gravity accessibility with negative-exponential or power decay,
 * and the two- or three-step floating catchment area (2SFCA or 3SFCA, with the decay applied inside
 * each catchment). The catchment is not a separate kernel: it composes `GPUNetworkCostWeights` and
 * `GPUCatchmentAccessibility`, the one floating catchment implementation.
 *
 * The contributor only reads the matrix, so changing the threshold, decay, or beta re-encodes a few
 * linear passes and never re-runs a shortest-path search: build the matrix with
 * `GPUNetworkCostMatrix` in one graph, encode it when the network or the opportunity set changes,
 * and encode this contributor's graph every frame.
 *
 * Sums are deterministic without float atomics: per-node outputs gather the rows in ascending order
 * in one invocation, and per-row outputs reduce each row in one workgroup with a fixed tree.
 */
export class GPUNetworkAccessibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkAccessibilityProps;
  /** Resolved matrix orientation. */
  readonly orientation: 'opportunity-rows' | 'origin-rows';
  /** Matrix rows. */
  readonly rowCount: number;
  /** Matrix columns (network nodes). */
  readonly nodeCount: number;

  constructor(props: GPUNetworkAccessibilityProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'costs',
      'costsComplete',
      'costLimit',
      'status',
      'orientation',
      'opportunityWeights',
      'parameters',
      'cumulative',
      'gravity',
      'catchment'
    ]);
    props = Object.freeze({
      ...captured,
      status: captured.status && snapshotConstructionFields(captured.status, ['complete']),
      catchment:
        captured.catchment &&
        snapshotConstructionFields(captured.catchment, ['method', 'demand', 'output', 'ratios'])
    }) as GPUNetworkAccessibilityProps;
    this.id = props.id ?? 'network-accessibility';
    this.props = props;
    const outputs = {
      cumulative: props.cumulative,
      gravity: props.gravity,
      catchment: props.catchment?.output,
      ratios: props.catchment?.ratios,
      complete: props.status?.complete
    };
    validateNetworkWritableViews(this.id, outputs, [
      {outputs: {cumulative: outputs.cumulative, gravity: outputs.gravity}},
      {outputs: {complete: outputs.complete}},
      {outputs: {ratios: outputs.ratios}},
      {outputs: {catchment: outputs.catchment}, inputs: {ratios: outputs.ratios}}
    ]);
    this.orientation = props.orientation ?? 'opportunity-rows';
    const {id} = this;
    validateRequirement(
      id,
      props.orientation === undefined ||
        props.orientation === 'opportunity-rows' ||
        props.orientation === 'origin-rows',
      "orientation must be 'opportunity-rows' or 'origin-rows'"
    );
    const {catchment} = props;
    for (const [name, view] of [
      ['costs', props.costs],
      ['opportunityWeights', props.opportunityWeights],
      ['parameters', props.parameters],
      ['cumulative', props.cumulative],
      ['gravity', props.gravity],
      ['catchment.demand', catchment?.demand],
      ['catchment.output', catchment?.output],
      ['catchment.ratios', catchment?.ratios]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    for (const [name, view, format] of [
      ['costsComplete', props.costsComplete, 'uint32'],
      ['costLimit', props.costLimit, 'float32'],
      ['status.complete', props.status?.complete, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} ${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} ${name} must contain exactly one row`);
        }
      }
    }
    if (props.status?.complete && !props.costsComplete) {
      throw new Error(`${id} status.complete requires costsComplete, the matrix completeness word`);
    }
    if (props.parameters.length !== GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH} rows`
      );
    }
    const weightCount = props.opportunityWeights.length;
    if (weightCount < 1 || props.costs.length % weightCount !== 0) {
      throw new Error(`${id} costs length must be a multiple of opportunityWeights length`);
    }
    const otherCount = props.costs.length / weightCount;
    if (otherCount < 1) {
      throw new Error(`${id} costs must not be empty`);
    }
    const isOpportunityRows = this.orientation === 'opportunity-rows';
    this.rowCount = isOpportunityRows ? weightCount : otherCount;
    this.nodeCount = isOpportunityRows ? otherCount : weightCount;
    const originCount = isOpportunityRows ? this.nodeCount : this.rowCount;
    for (const [name, view] of [
      ['cumulative', props.cumulative],
      ['gravity', props.gravity]
    ] as const) {
      if (view && view.length !== originCount) {
        throw new Error(`${id} ${name} length must equal the origin count`);
      }
    }
    if (catchment) {
      const method = catchment.method ?? '2sfca';
      if (method !== '2sfca' && method !== '3sfca') {
        throw new Error(`${id} catchment.method must be '2sfca' or '3sfca'`);
      }
      if (!isOpportunityRows) {
        throw new Error(`${id} catchment requires orientation 'opportunity-rows'`);
      }
      if (catchment.demand.length !== this.nodeCount) {
        throw new Error(`${id} catchment.demand length must equal the node count`);
      }
      if (catchment.output.length !== this.nodeCount) {
        throw new Error(`${id} catchment.output length must equal the node count`);
      }
      if (catchment.ratios && catchment.ratios.length !== this.rowCount) {
        throw new Error(`${id} catchment.ratios length must equal the row count`);
      }
    }
    if (!props.cumulative && !props.gravity && !catchment) {
      throw new Error(`${id} needs at least one of cumulative, gravity, or catchment`);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(outputs), getInputs(props));
  }

  /**
   * Returns one `scores` node when `cumulative` or `gravity` is requested, then for a catchment the
   * nodes of the facility weights (`catchment-facility-weights-*`), the demand weights
   * (`catchment-demand-weights-*`) and `GPUCatchmentAccessibility` (`catchment-*`), then a
   * one-invocation `complete` node when `status.complete` is requested. Without that public
   * certificate, supplied evidence adds a private `input-complete` node and dependent-output
   * suppression nodes; certificate routes retain all incumbent numerical outputs.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, nodeCount} = this;
    const {catchment} = props;
    validateNetworkViewBindings(id, graph, [
      ...getInputs(props),
      props.cumulative,
      props.gravity,
      catchment?.output,
      catchment?.ratios,
      props.status?.complete
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (catchment)
      validateNetworkViewBindings(id, graph, [], {
        catchmentPairs: rowCount * nodeCount * 4,
        catchmentOffsets: (Math.max(rowCount, nodeCount) + 1) * 4
      });
    let inputComplete: GraphDataView<'uint32'> | undefined;
    if (!props.status?.complete && (props.costsComplete || props.costLimit)) {
      inputComplete = createTransientView(graph, `${id}-input-complete`, 'uint32', 1);
      const bindings: WGSLKernelBinding[] = [
        {name: 'complete', view: inputComplete, type: 'u32', access: 'read_write'}
      ];
      const conditions: string[] = [];
      if (props.costsComplete) {
        bindings.push({
          name: 'costsComplete',
          view: props.costsComplete,
          type: 'u32',
          access: 'read'
        });
        conditions.push('costsComplete[costsCompleteOffset] != 0u');
      }
      if (props.costLimit) {
        bindings.push(
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'costLimit', view: props.costLimit, type: 'f32', access: 'read'}
        );
        conditions.push('parameters[parametersOffset] <= costLimit[costLimitOffset]');
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-input-complete`,
          operation: OPERATION,
          variant: 'input-complete',
          invocationCount: 1,
          bindings,
          body: `complete[completeOffset] = select(0u, 1u, ${conditions.join(' && ')});`
        })
      );
    }
    if (props.cumulative || props.gravity) {
      nodes.push(
        this.orientation === 'opportunity-rows'
          ? createAccessibilityColumnGatherNode<Parameters>(graph, {
              id: `${id}-scores`,
              operation: OPERATION,
              rowCount,
              columnCount: nodeCount,
              costs: props.costs,
              rowWeights: props.opportunityWeights,
              parameters: props.parameters,
              cumulative: props.cumulative,
              gravity: props.gravity
            })
          : createAccessibilityRowReduceNode<Parameters>(graph, {
              id: `${id}-scores`,
              operation: OPERATION,
              variant: 'row-reduce',
              rowCount,
              columnCount: nodeCount,
              costs: props.costs,
              columnWeights: props.opportunityWeights,
              parameters: props.parameters,
              cumulative: props.cumulative,
              gravity: props.gravity
            })
      );
    }
    if (catchment) {
      // The parameter words [threshold, decay, beta, minimumCost] are exactly the
      // GPUNetworkCostWeights layout [costLimit, decay, beta, minimumCost] with the same decay codes,
      // so one view drives both weights.
      const [facilityWeights, demandWeights] = (['rows', 'columns'] as const).map(orientation => {
        const weightsId = `${id}-catchment-${orientation === 'rows' ? 'facility' : 'demand'}-weights`;
        const weightsRowCount = orientation === 'rows' ? rowCount : nodeCount;
        const capacity = rowCount * nodeCount;
        const costWeights = new GPUNetworkCostWeights({
          id: weightsId,
          costs: props.costs,
          rowCount,
          orientation,
          parameters: props.parameters,
          output: {
            offsets: createTransientView(
              graph,
              `${weightsId}-csr-offsets`,
              'uint32',
              weightsRowCount + 1
            ),
            neighbors: createTransientView(graph, `${weightsId}-csr-neighbors`, 'uint32', capacity),
            weights: createTransientView(graph, `${weightsId}-csr-weights`, 'float32', capacity)
          }
        });
        nodes.push(...costWeights.getCommandNodes(graph));
        return costWeights.outputWeights;
      });
      nodes.push(
        ...new GPUCatchmentAccessibility({
          id: `${id}-catchment`,
          method: catchment.method ?? '2sfca',
          supply: props.opportunityWeights,
          demand: catchment.demand,
          facilityWeights,
          demandWeights,
          costsComplete: inputComplete,
          output: {accessibility: catchment.output, ratios: catchment.ratios}
        }).getCommandNodes(graph)
      );
    }
    if (inputComplete) {
      for (const [name, output] of [
        ['cumulative', props.cumulative],
        ['gravity', props.gravity]
      ] as const) {
        if (!output) continue;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-suppress-${name}`,
            operation: OPERATION,
            variant: `suppress-${name}`,
            invocationCount: output.length,
            bindings: [
              {name: 'complete', view: inputComplete, type: 'u32', access: 'read'},
              {name: 'scores', view: output, type: 'f32', access: 'read_write'}
            ],
            body: `if (complete[completeOffset] == 0u) {
  scores[scoresOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
}`
          })
        );
      }
    }
    const complete = props.status?.complete;
    if (complete && props.costsComplete) {
      // The catchment weights hold rowCount * nodeCount slots, so they never overflow; the scores
      // are complete exactly when the matrix is and the threshold is within its cost limit.
      // `threshold <= costLimit` is false for a NaN limit, and +Infinity <= +Infinity is true.
      const bindings: WGSLKernelBinding[] = [
        {name: 'costsComplete', view: props.costsComplete, type: 'u32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
      ];
      if (props.costLimit) {
        bindings.push(
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'costLimit', view: props.costLimit, type: 'f32', access: 'read'}
        );
      }
      const withinLimit = props.costLimit
        ? ' && parameters[parametersOffset] <= costLimit[costLimitOffset]'
        : '';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-complete`,
          operation: OPERATION,
          variant: props.costLimit ? 'complete-limit' : 'complete',
          bindings,
          invocationCount: 1,
          body: `let isComplete = costsComplete[costsCompleteOffset] != 0u${withinLimit};
  complete[completeOffset] = select(0u, 1u, isComplete);`
        })
      );
    }
    return nodes;
  }
}

/** Returns every read-only view of an accessibility contributor. */
function getInputs(props: GPUNetworkAccessibilityProps): (GraphDataView | undefined)[] {
  return [
    props.costs,
    props.opportunityWeights,
    props.parameters,
    props.catchment?.demand,
    props.costsComplete,
    props.costLimit
  ];
}
