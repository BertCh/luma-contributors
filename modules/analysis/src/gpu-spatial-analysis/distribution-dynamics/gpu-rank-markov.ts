// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_RANK_MARKOV_ROUTES, GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS} from './rank-markov-routes';
export {GPU_RANK_MARKOV_ROUTES, GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS} from './rank-markov-routes';

import type {
  GPUAnalysisRoute,
  GPUAnalysisExecutionPlanner
} from '../ontology/spatial-analysis-ontology';
import {
  evaluateStatisticsRoute,
  getStatisticsRequirementCheck,
  getStatisticsViewBindingByteLength,
  getStatisticsExecutionPlan,
  validateStatisticsExecutionPlan,
  type GPUStatisticsPlanDeviceLimits
} from '../geographic-distribution/statistics-execution-plan';
import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
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
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  getGraphOutputViews,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  GPUTransitionMatrix,
  GPU_TRANSITION_MATRIX_MAXIMUM_CLASS_COUNT,
  type GPUTransitionMatrixOutput
} from './gpu-transition-matrix';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createSortedRankNodes, createRankMobilityNodes} from './rank-markov-sorted';
import {getLoadBalancedTileCount} from '../../analysis-core/kernels/load-balanced-for-each';

/** Period sort keys and the balanced-panel API retain the existing 65,536-period envelope. */
export const GPU_RANK_MARKOV_MAXIMUM_PERIOD_COUNT = 65536;
/** Published stable sorting indexes at most 2^31 panel entries; bindings impose a tighter cap. */
export const GPU_RANK_MARKOV_MAXIMUM_PANEL_VALUES = 0x80000000;
/** Two radix sorts, complete cases, scatter, transitions and mobility (including sort scans). */
const RANK_MARKOV_COMMAND_NODE_ALLOWANCE = 512;
/** Published stable radix sort uses 256-row workgroups and at most 16 histogram buckets. */
const RANK_MARKOV_SORT_WORKGROUP_SIZE = 256;
const RANK_MARKOV_SORT_BUCKET_COUNT = 16;

export type GPURankMarkovProps = {
  id?: string;
  /** Period-major balanced panel. Any nonfinite observation excludes its row from ALL periods. */
  values: GraphDataView<'float32'>;
  periodCount: number;
  mask?: GraphDataView<'uint32'>;
  mode?: 'full-rank' | 'geographic-rank';
  /** auto uses pairwise through 64 rows, stable GPU sorting above it. Compile-time. */
  rankingMethod?: 'auto' | 'pairwise' | 'sorted';
  /** giddy reverses ascending ordinal ranks: later row IDs win descending ties. */
  tieOrder?: 'giddy' | 'row-id';
  output: GPUTransitionMatrixOutput & {
    /** Zero-based descending ranks; ties follow tieOrder (giddy by default). Excluded rows: 0xffffffff. */
    ranks: GraphDataView<'uint32'>;
    /** Mean absolute rank change per adjacent period, NaN when fewer than two included rows. */
    mobility?: GraphDataView<'float32'>;
  };
};

/**
 * giddy full-rank and geographic-rank transition conventions, pinned to giddy 2.3.6 (the version
 * that generated the reference in `test/oracles/fixtures/statistics-upstream.json`; the other
 * Markov contributors are pinned to giddy 2.3.9 in `statistics-markov.json`). Full rank follows each
 * location through rank states. Geographic rank follows the occupant of each fixed rank through
 * location states. No discretization, rank averaging or synthetic self-loops. Common complete
 * cases keep the population comparable over time. The n-by-n matrix retains original capacity
 * even when rows are excluded; zero transition rows remain zero. Existing transition diagnostics
 * explicitly report nonergodicity instead of inventing a unique stationary distribution.
 * Pairwise ranking and optional reference diagnostics support at most 64 logical states.
 * Sorted ranking supports up to 65,535 logical states over the domain, subject to dense n-by-n
 * output bindings. Request sorted without homogeneity, steadyStates, stateSparsity or status
 * diagnostics above 64 states.
 */
export class GPURankMarkov implements GPUCommandNodeProducer {
  /** Numeric route metadata; hard limits and automatic crossovers are distinct. */
  static get routes(): readonly GPUAnalysisRoute[] {
    return GPU_RANK_MARKOV_ROUTES;
  }
  /** Preflight topology and, when supplied, actual device binding/allocation limits. */
  static getExecutionPlan(props: GPURankMarkovProps, deviceLimits?: GPUStatisticsPlanDeviceLimits) {
    return getGPURankMarkovExecutionPlan(props, deviceLimits);
  }

  readonly id: string;
  readonly rowCount: number;
  readonly rankingMethod: 'pairwise' | 'sorted';
  readonly estimatedCommandNodeCount = RANK_MARKOV_COMMAND_NODE_ALLOWANCE;
  readonly props: GPURankMarkovProps;
  constructor(props: GPURankMarkovProps) {
    props = {
      ...props,
      output: {...props.output, status: props.output.status ? {...props.output.status} : undefined}
    };
    this.props = props;
    this.id = props.id ?? 'rank-markov';
    validateCapacity(
      this.id,
      'periodCount',
      props.periodCount,
      2,
      GPU_RANK_MARKOV_MAXIMUM_PERIOD_COUNT
    );
    this.rowCount = props.values.length / props.periodCount;
    validateRequirement(
      this.id,
      ['auto', 'pairwise', 'sorted'].includes(props.rankingMethod ?? 'auto'),
      'rankingMethod must be auto, pairwise or sorted'
    );
    const routePlan = getGPURankMarkovExecutionPlan(props);
    this.rankingMethod = routePlan.route;
    const hasDiagnostics = Boolean(
      props.output.homogeneity ||
        props.output.steadyStates ||
        props.output.stateSparsity ||
        props.output.status
    );
    if (
      this.rankingMethod === 'pairwise' &&
      this.rowCount > GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS
    ) {
      const alternative = getGPURankMarkovExecutionPlan({...props, rankingMethod: 'sorted'});
      const failed = alternative.constraints.find(
        check => check.kind !== 'auto-crossover' && !check.satisfied
      );
      validateRequirement(
        this.id,
        false,
        `rowCount received ${this.rowCount}; pairwise ranking supports at most 64 rows${alternative.feasible ? "; use rankingMethod: 'sorted' subject to dense output bindings" : `; ${hasDiagnostics ? 'requested diagnostics: ' : ''}sorted alternative fails ${failed?.quantity ?? 'numeric admission checks'}`}`
      );
    }
    validateCapacity(
      this.id,
      'rowCount',
      this.rowCount,
      1,
      this.rankingMethod === 'pairwise'
        ? GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS
        : GPU_TRANSITION_MATRIX_MAXIMUM_CLASS_COUNT
    );
    validateCapacity(
      this.id,
      'values.length',
      props.values.length,
      1,
      GPU_RANK_MARKOV_MAXIMUM_PANEL_VALUES
    );
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateStatisticsExecutionPlan(this.id, routePlan);
    const outputs = getGraphOutputViews({
      ...props.output,
      status: {converged: props.output.status?.converged}
    });
    validateSpatialOutputViews(
      this.id,
      Object.fromEntries(outputs.map((view, index) => [`output[${index}]`, view])),
      [{read: [props.output.ranks], write: [props.output.counts, props.output.ignored]}]
    );
    if (props.tieOrder && !['giddy', 'row-id'].includes(props.tieOrder))
      throw new Error(`${this.id}: invalid tie order`);
    if (props.mode && props.mode !== 'full-rank' && props.mode !== 'geographic-rank')
      throw new Error(`${this.id}: invalid rank mode`);
    for (const [view, format, length] of [
      [props.values, 'float32', props.values.length],
      [props.mask, 'uint32', this.rowCount],
      [props.output.ranks, 'uint32', props.values.length],
      [props.output.mobility, 'float32', props.periodCount - 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [props.values, props.mask]);
    // Validate optional diagnostics at declaration, before constructing transient rank occupants.
    new GPUTransitionMatrix({
      id: `${this.id}-transitions`,
      classes: props.output.ranks,
      periodCount: props.periodCount,
      classCount: this.rowCount,
      output: props.output
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateStatisticsExecutionPlan(
      this.id,
      getGPURankMarkovExecutionPlan(this.props, graph.device.limits)
    );
    const {id, props, rowCount: rows} = this;
    const ranks = props.output.ranks;
    const outputViews = getGraphOutputViews({
      ...props.output,
      status: {converged: props.output.status?.converged}
    });
    validateGraphViewsBelongToGraph(id, graph, [props.values, props.mask, ...outputViews]);
    validateGraphChunkBindingRanges(graph, id, {
      values: props.values,
      mask: props.mask,
      ...Object.fromEntries(outputViews.map((view, index) => [`output[${index}]`, view]))
    });
    for (const [name, byteLength] of [
      ['panel scratch', ranks.length * 4],
      ['complete cases', rows * 4],
      [
        'sort histogram',
        this.rankingMethod === 'sorted'
          ? Math.ceil(ranks.length / RANK_MARKOV_SORT_WORKGROUP_SIZE) *
            RANK_MARKOV_SORT_BUCKET_COUNT *
            4
          : 0
      ],
      ['mobility offsets', props.periodCount * 4],
      ['mobility totals', props.output.mobility ? (props.periodCount - 1) * 8 : 0],
      [
        'mobility tile carries',
        props.output.mobility
          ? getLoadBalancedTileCount(props.periodCount - 1, rows * (props.periodCount - 1)) * 12
          : 0
      ]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce rowCount or periodCount'
      );
    }
    const valid = createTransientView(graph, `${id}-complete-cases`, 'uint32', rows);
    const classes =
      props.mode === 'geographic-rank'
        ? createTransientView(graph, `${id}-occupants`, 'uint32', ranks.length)
        : ranks;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-complete`,
        operation: 'GPURankMarkov',
        variant: 'complete-cases',
        invocationCount: rows,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'valid', view: valid, type: 'u32', access: 'read_write'}
        ],
        body: `valid[validOffset + index] = ${props.mask ? 'select(0u, 1u, mask[maskOffset + index] != 0u)' : '1u'};`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-exclude-incomplete`,
        operation: 'GPURankMarkov',
        variant: 'exclude-incomplete',
        invocationCount: ranks.length,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'valid', view: valid, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `if (!isFiniteBits(values[valuesOffset + index])) { atomicStore(&valid[validOffset + index % ${rows}u], 0u); }`
      })
    ];
    if (this.rankingMethod === 'sorted')
      nodes.push(...createSortedRankNodes(graph, id, props, valid));
    else
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-rank`,
          operation: 'GPURankMarkov',
          variant: 'rank',
          invocationCount: ranks.length,
          bindings: [
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'valid', view: valid, type: 'u32', access: 'read'},
            {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'}
          ],
          body: `let row = index % ${rows}u; let base = index - row; var rank = 0xffffffffu;
if (valid[validOffset + row] != 0u) { rank = 0u; let value = values[valuesOffset + index];
  for (var other = 0u; other < ${rows}u; other++) { let candidate = values[valuesOffset + base + other];
    if (valid[validOffset + other] != 0u && (candidate > value || (candidate == value && other ${props.tieOrder === 'row-id' ? '<' : '>'} row))) { rank++; }
  }
}
ranks[ranksOffset + index] = rank;`
        })
      );
    if (props.mode === 'geographic-rank') {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-clear-occupants`,
          operation: 'GPURankMarkov',
          variant: 'clear-occupants',
          invocationCount: ranks.length,
          bindings: [{name: 'classes', view: classes, type: 'u32', access: 'read_write'}],
          body: 'classes[classesOffset + index] = 0xffffffffu;'
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-occupants`,
          operation: 'GPURankMarkov',
          variant: 'occupants',
          invocationCount: ranks.length,
          bindings: [
            {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
            {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
          ],
          body: `let row = index % ${rows}u; let rank = ranks[ranksOffset + index];
if (rank < ${rows}u) { classes[classesOffset + index - row + rank] = row; }`
        })
      );
    }
    nodes.push(
      ...new GPUTransitionMatrix({
        id: `${id}-transitions`,
        classes,
        periodCount: props.periodCount,
        classCount: rows,
        output: props.output
      }).getCommandNodes(graph)
    );
    nodes.push(...createRankMobilityNodes(graph, id, props));
    return nodes;
  }
}

/** Includes the dense state-square outputs and period-panel sorting scratch. */
export function getGPURankMarkovExecutionPlan(
  props: GPURankMarkovProps,
  deviceLimits?: GPUStatisticsPlanDeviceLimits
) {
  const id = props.id ?? 'rank-markov';
  validateRequirement(
    id,
    ['full-rank', 'geographic-rank'].includes(props.mode ?? 'full-rank'),
    `mode must be full-rank or geographic-rank; received ${props.mode}`
  );
  validateRequirement(
    id,
    ['giddy', 'row-id'].includes(props.tieOrder ?? 'giddy'),
    `tieOrder must be giddy or row-id; received ${props.tieOrder}`
  );
  const requested = props.rankingMethod ?? 'auto';
  validateRequirement(
    id,
    ['auto', 'pairwise', 'sorted'].includes(requested),
    `rankingMethod must be auto, pairwise or sorted; received ${requested}`
  );
  const rows = props.values.length / props.periodCount;
  const route =
    requested === 'pairwise' ||
    (requested === 'auto' && rows <= GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS)
      ? 'pairwise'
      : 'sorted';
  const diagnostics = Boolean(
    props.output.homogeneity ||
      props.output.steadyStates ||
      props.output.stateSparsity ||
      props.output.status
  );
  const descriptor = GPU_RANK_MARKOV_ROUTES[0];
  const checks = [
    evaluateStatisticsRoute(descriptor, rows),
    evaluateStatisticsRoute(
      {
        ...descriptor,
        name: 'route-state-maximum',
        kind: 'hard-limit',
        appliesWhen: `rankingMethod ${route}`,
        minimum: 1,
        maximum:
          route === 'pairwise'
            ? GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS
            : GPU_TRANSITION_MATRIX_MAXIMUM_CLASS_COUNT
      },
      rows
    ),
    evaluateStatisticsRoute(
      {
        ...descriptor,
        name: 'periodCount',
        kind: 'hard-limit',
        quantity: 'periodCount',
        unit: 'elements',
        minimum: 2,
        maximum: GPU_RANK_MARKOV_MAXIMUM_PERIOD_COUNT,
        appliesWhen: 'all routes'
      },
      props.periodCount
    ),
    evaluateStatisticsRoute(
      {
        ...descriptor,
        name: 'panel-product',
        kind: 'hard-limit',
        quantity: 'rows * periodCount',
        unit: 'elements',
        maximum: GPU_RANK_MARKOV_MAXIMUM_PANEL_VALUES,
        appliesWhen: 'all routes'
      },
      props.values.length
    ),
    evaluateStatisticsRoute(
      {
        ...descriptor,
        name: 'transition-opportunities',
        kind: 'hard-limit',
        quantity: 'rows * (periodCount - 1)',
        unit: 'elements',
        maximum: 0xffffffff,
        appliesWhen: 'all routes'
      },
      rows * (props.periodCount - 1)
    )
  ];
  if (diagnostics)
    checks.push(
      evaluateStatisticsRoute(
        {
          ...descriptor,
          name: 'diagnostic-state-maximum',
          quantity: 'classCount',
          kind: 'hard-limit',
          maximum: 64,
          appliesWhen: 'any diagnostic output or status group'
        },
        rows
      )
    );
  const bindings: Record<string, number> = {
    values: getStatisticsViewBindingByteLength(props.values),
    ...(props.mask ? {mask: getStatisticsViewBindingByteLength(props.mask)} : {}),
    'panel scratch: rows * periodCount * 4': props.values.length * 4,
    'complete cases: rows * 4': rows * 4,
    'dense counts: rows * rows * 4': rows * rows * 4,
    'mobility offsets: periodCount * 4': props.periodCount * 4,
    ...(route === 'sorted'
      ? {
          'radix histogram: ceil(panelValues / 256) * 16 * 4':
            Math.ceil(props.values.length / RANK_MARKOV_SORT_WORKGROUP_SIZE) *
            RANK_MARKOV_SORT_BUCKET_COUNT *
            4
        }
      : {}),
    ...(props.output.mobility
      ? {
          'mobility totals': (props.periodCount - 1) * 8,
          'mobility tile carries':
            getLoadBalancedTileCount(props.periodCount - 1, rows * (props.periodCount - 1)) * 12
        }
      : {})
  };
  const outputViews = {
    ...props.output,
    status: undefined,
    'status.converged': props.output.status?.converged
  };
  for (const [name, view] of Object.entries(outputViews))
    if (view) bindings[`output.${name}`] = getStatisticsViewBindingByteLength(view);
  for (const [name, view, minimum, exact] of [
    ['ranks', props.output.ranks, props.values.length, true],
    ['counts', props.output.counts, rows * rows, false],
    ['probabilities', props.output.probabilities, rows * rows, false],
    ['rowTotals', props.output.rowTotals, rows, false],
    ['ignored', props.output.ignored, 1, false],
    ['mobility', props.output.mobility, props.periodCount - 1, true],
    ['steadyStates', props.output.steadyStates, rows, false],
    ['stateSparsity', props.output.stateSparsity, 2, false],
    ['status.converged', props.output.status?.converged, 1, true]
  ] as const)
    if (view)
      checks.push(
        getStatisticsRequirementCheck(
          descriptor,
          `output.${name} ${exact ? 'length must equal' : 'must hold at least'} ${minimum}`,
          exact ? view.length === minimum : view.length >= minimum
        )
      );
  checks.push(
    getStatisticsRequirementCheck(
      descriptor,
      'output.homogeneity needs at least 2 conditions; Rank Markov has 1',
      !props.output.homogeneity
    )
  );
  checks.push(
    getStatisticsRequirementCheck(
      descriptor,
      'output.status.converged needs output.steadyStates',
      !props.output.status?.converged || Boolean(props.output.steadyStates)
    )
  );
  checks.push(
    getStatisticsRequirementCheck(
      descriptor,
      'mask length must equal rowCount',
      !props.mask || props.mask.length === rows
    )
  );
  return getStatisticsExecutionPlan(
    route,
    checks,
    RANK_MARKOV_COMMAND_NODE_ALLOWANCE,
    bindings,
    deviceLimits,
    descriptor
  );
}
export const GPU_RANK_MARKOV_PLANNER: GPUAnalysisExecutionPlanner<
  GPURankMarkovProps,
  GPUStatisticsPlanDeviceLimits | undefined
> = {getExecutionPlan: getGPURankMarkovExecutionPlan};
