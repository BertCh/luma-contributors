// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addPreparedRecipe} from './prepared-recipe';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPUGlobalSpatialStatistics
} from '../global-spatial-statistics/index';
import {
  GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER,
  GPUSpatialWeightsAlgebra
} from '../spatial-weights/gpu-spatial-weights-algebra';
import {GPUSpatialWeightsTransform} from '../spatial-weights/gpu-spatial-weights-transform';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {getRecipeView as getOrCreateView, validateRecipeBoundary} from './recipe-utils';
import {createRecipeKernelNode} from './recipe-utils';
import {RecipeBuilder, createTransientSpatialWeights, type GPURecipeResult} from './recipe-utils';

/** Columns of one order row of `outputs.correlogram`. */
export const GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS = {
  /** Moran's I of the order. */
  statistic: 0,
  /** Its expectation `-1 / (n - 1)`. */
  expected: 1,
  /** Variance under normality. */
  varianceNormality: 2,
  /** z-score under normality. */
  zNormality: 3,
  /** Two-sided normal p-value under normality. */
  pNormality: 4,
  /** Variance under randomization. */
  varianceRandomization: 5,
  /** z-score under randomization. */
  zRandomization: 6,
  /** Two-sided normal p-value under randomization. */
  pRandomization: 7,
  /** Rows that took part (`n`, islands excluded). */
  count: 8,
  /** Rows with no neighbor at exactly this order. */
  islandCount: 9,
  /** Columns per order row. */
  length: 10
} as const;

/** Caller-owned outputs of {@link addContiguityOrderCorrelogramRecipe}. */
export type GPUContiguityOrderCorrelogramRecipeOutputs = {
  /** `maximumOrder * GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS.length` floats, order-major. */
  correlogram: GraphDataView<'float32'>;
  /** One uint32 per order: 1 when the slot capacity overflowed, so that order is incomplete. */
  incompleteOrders: GraphDataView<'uint32'>;
};

/** Properties for {@link addContiguityOrderCorrelogramRecipe}. */
export type GPUContiguityOrderCorrelogramRecipeProps = {
  /** Prefix for node and view IDs. Defaults to `'contiguity-order-correlogram'`. */
  id?: string;
  /** Contiguity (or k-nearest-neighbor) weights; only the sparsity pattern is used. */
  weights: GPUSpatialWeights;
  /** Values, one per row. */
  values: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Largest contiguity order, in `[1, GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER]`; orders 1 to this are evaluated. */
  maximumOrder: number;
  /**
   * Slot capacity of each order's weights. Defaults to `weights.neighbors.length * maximumOrder`,
   * which holds a square lattice (the ring of order `k` has `k` times the links of order 1).
   * Memory grows with `maximumOrder` times this capacity; check `outputs.incompleteOrders`.
   */
  slotCapacity?: number;
  /**
   * Slot capacity of the intermediate sets of the ring construction, which hold every neighbor
   * within the order (about `k (k + 1) / 2` times the order 1 links on a square lattice).
   * Defaults to `weights.neighbors.length * maximumOrder * (maximumOrder + 1) / 2`, at most
   * `rows * (rows - 1)`.
   */
  workCapacity?: number;
  /** Caller-owned outputs; graph-owned when omitted. */
  outputs?: Partial<GPUContiguityOrderCorrelogramRecipeOutputs>;
};

/** Result of {@link addContiguityOrderCorrelogramRecipe}. */
export type GPUContiguityOrderCorrelogramRecipeResult = GPURecipeResult<
  GPUContiguityOrderCorrelogramRecipeOutputs,
  {orderWeights: readonly GPUSpatialWeights[]}
>;

/**
 * Spatial correlogram over contiguity orders: global Moran's I with analytic inference for the
 * neighbors at exactly 1, 2, ..., `maximumOrder` steps, each with row-standardised weights
 * (`libpysal` `higher_order` rings and `esda.Moran`). It composes `GPUSpatialWeightsAlgebra`
 * (`higherOrder`), `GPUSpatialWeightsTransform` (row standardisation) and
 * `GPUGlobalSpatialStatistics` once per order, so every limit of those applies: the algebra's
 * frontier work grows with the squared degree, and an order whose ring needs more than `slotCapacity`
 * slots is reported in `outputs.incompleteOrders` and its row is not valid.
 *
 * Rows with no neighbor at an order are islands and are excluded from that order's `n`.
 * Reference: a binary64 breadth-first ring construction and Moran's I with the same island rule
 * (libpysal and esda were not installed for this item).
 */
export function addContiguityOrderCorrelogramRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUContiguityOrderCorrelogramRecipeProps
): GPUContiguityOrderCorrelogramRecipeResult {
  return addPreparedRecipe(graph, props, declareContiguityOrderCorrelogramRecipe);
}

function declareContiguityOrderCorrelogramRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUContiguityOrderCorrelogramRecipeProps
): GPUContiguityOrderCorrelogramRecipeResult {
  const captured = snapshotWeightsFields(props, [
    'id',
    'weights',
    'values',
    'mask',
    'maximumOrder',
    'slotCapacity',
    'workCapacity',
    'outputs'
  ]);
  props = Object.freeze({
    ...captured,
    weights: snapshotWeightsPort(captured.weights),
    ...(captured.outputs
      ? {outputs: snapshotWeightsFields(captured.outputs, ['correlogram', 'incompleteOrders'])}
      : {})
  });
  const id = props.id ?? 'contiguity-order-correlogram';
  const {maximumOrder} = props;
  validateCapacity(id, 'maximumOrder', maximumOrder, 1, GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER);
  const rows = props.values.length;
  validateRequirement(
    id,
    props.weights.offsets.length - 1 === rows,
    'values length must equal the weights row count'
  );
  const slotCapacity = props.slotCapacity ?? props.weights.neighbors.length * maximumOrder;
  validateCapacity(id, 'slotCapacity', slotCapacity);
  const workCapacity =
    props.workCapacity ??
    Math.max(
      slotCapacity,
      Math.min(
        rows * Math.max(rows - 1, 1),
        (props.weights.neighbors.length * maximumOrder * (maximumOrder + 1)) / 2
      )
    );
  validateCapacity(id, 'workCapacity', workCapacity);
  const columns = GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS;
  const outputs: GPUContiguityOrderCorrelogramRecipeOutputs = {
    correlogram: getOrCreateView(
      graph,
      `${id}-correlogram`,
      'float32',
      maximumOrder * columns.length,
      props.outputs?.correlogram
    ),
    incompleteOrders: getOrCreateView(
      graph,
      `${id}-incomplete-orders`,
      'uint32',
      maximumOrder,
      props.outputs?.incompleteOrders
    )
  };
  validateRecipeBoundary(
    graph,
    id,
    [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.status?.overflow,
      props.weights.status?.candidateOverflow,
      props.values,
      props.mask
    ],
    [outputs.correlogram, outputs.incompleteOrders]
  );
  const builder = new RecipeBuilder(graph);
  const orderWeights: GPUSpatialWeights[] = [];
  const stages: {
    stage: string;
    status: {overflow: GraphDataView<'uint32'>; candidateOverflow: GraphDataView<'uint32'>};
  }[] = [];
  for (let order = 1; order <= maximumOrder; order++) {
    const orderId = `${id}-order-${order}`;
    const ring = createTransientSpatialWeights(graph, `${orderId}-ring`, rows, slotCapacity);
    ring.status.candidateOverflow = createTransientView(
      graph,
      `${orderId}-candidate-overflow`,
      'uint32',
      1
    );
    const standardized = createTransientView(
      graph,
      `${orderId}-row-weights`,
      'float32',
      ring.neighbors.length
    );
    const summary = createTransientView(
      graph,
      `${orderId}-summary`,
      'float32',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
    );
    builder.add(
      new GPUSpatialWeightsAlgebra({
        id: `${orderId}-ring`,
        operation: 'higherOrder',
        weights: props.weights,
        order,
        workCapacity,
        output: ring
      })
    );
    builder.add(
      new GPUSpatialWeightsTransform({
        id: `${orderId}-standardize`,
        operation: 'row',
        weights: ring,
        output: standardized
      })
    );
    const orderRing = {...ring, weights: standardized};
    orderWeights.push(orderRing);
    builder.add(
      new GPUGlobalSpatialStatistics({
        id: `${orderId}-moran`,
        weights: orderRing,
        values: props.values,
        mask: props.mask,
        statistics: ['moran'],
        output: {summary}
      })
    );
    const base = (order - 1) * columns.length;
    const moranOffset = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran;
    builder.addGlue(
      `${orderId}-copy`,
      createRecipeKernelNode(graph, {
        id: `${orderId}-copy`,
        operation: 'GPUContiguityOrderCorrelogramRecipe',
        variant: 'copy-order',
        bindings: [
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'ringOverflow', view: ring.status.overflow, type: 'u32', access: 'read'},
          {
            name: 'ringCandidateOverflow',
            view: ring.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'correlogram', view: outputs.correlogram, type: 'f32', access: 'read_write'},
          {
            name: 'incompleteOrders',
            view: outputs.incompleteOrders,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `for (var field = 0u; field < 8u; field++) {
    correlogram[correlogramOffset + ${base}u + field] = summary[summaryOffset + ${moranOffset}u + field];
  }
  correlogram[correlogramOffset + ${base + columns.count}u] = summary[summaryOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.count}u];
  correlogram[correlogramOffset + ${base + columns.islandCount}u] = summary[summaryOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.islandCount}u];
  incompleteOrders[incompleteOrdersOffset + ${order - 1}u] = u32(ringOverflow[ringOverflowOffset] != 0u || ringCandidateOverflow[ringCandidateOverflowOffset] != 0u);`
      })
    );
    stages.push({
      stage: `order-${order}`,
      status: {overflow: ring.status.overflow, candidateOverflow: ring.status.candidateOverflow}
    });
  }
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {orderWeights},
    status: {stages}
  };
}
