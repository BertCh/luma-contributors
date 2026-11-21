// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createFixedPointClearNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {Device} from '@luma.gl/core';
import {validateRasterGrid} from '../raster-band-utils';
import {validateGPUStatusPort} from '../../analysis-core/contracts/status';
import {
  createTransientView,
  getViewBindingRange,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateRasterWritableDispatches,
  type RasterWritableDispatch
} from '../raster-time-series/raster-time-series-chunks-b';
import {
  createRasterRegionBoruvkaNodes,
  getRasterRegionBoruvkaRoundCount
} from './raster-region-boruvka';
import {RASTER_FINITE_WGSL, rasterBinding, validateRasterGraph} from './raster-kernel-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterRegionGrid} from './raster-region-snapshot';
import type {
  GPUAnalysisExecutionPlanner,
  GPUAnalysisRoute
} from '../../analysis-core/contracts/analysis-execution-planning';

/** Regions: adjacency and weights are dense regionCount squared matrices and the merge scans a whole matrix row per round in one invocation (reference tier). */
export const MAXIMUM_REGION_GRAPH_REGION_COUNT = 256;
/** Dense public matrix offsets must fit uint32; actual matrix and sparse scratch bindings must also fit the device. */
export const MAXIMUM_BORUVKA_REGION_GRAPH_REGION_COUNT = 65535;
/** Merge rounds: every round is an unrolled command node (adaptive rounds merge one pair each), so the count bounds the node total. */
export const MAXIMUM_REGION_GRAPH_ITERATIONS = 256;
/** Cells: labels index cells with 30 bits so the boundary pass can count neighbour pairs in a u32 atomic without overflow. */
export const MAXIMUM_REGION_GRAPH_CELL_COUNT = 0x3fffffff;

export type GPURasterRegionGraphProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Dense IDs 0..regionCount-1; other labels are background. */
  labels: GraphDataView<'uint32'>;
  /** One uint32 word, nonzero for a complete source partition. Requires output.status.complete. */
  labelsComplete?: GraphDataView<'uint32'>;
  values: GraphDataView<'float32'>;
  regionCount: number;
  /** Fixed single linkage (default) or one globally closest mean-updating merge per round. */
  mergeMode?: 'fixed' | 'adaptive';
  /** Boruvka applies only to fixed linkage. Auto preserves bounded dense propagation for small unfinished runs; otherwise uses Boruvka. Dense retains the 256-region reference route. */
  searchStructure?: 'auto' | 'dense' | 'boruvka';
  /** Merge mean-difference edges at or below this nonnegative threshold. */
  mergeThreshold: number;
  /** Dense propagation/adaptive rounds, or Boruvka phases (at most ceil(log2(regionCount))). */
  maximumIterations: number;
  output: {
    /** Symmetric dense regionCount squared matrix; four-neighbor shared-edge count. */
    adjacency: GraphDataView<'uint32'>;
    /** Symmetric dense mean-difference matrix; absent/undefined edges are NaN. */
    weights: GraphDataView<'float32'>;
    /** Original region to smallest connected source region after the bounded rounds. */
    mapping: GraphDataView<'uint32'>;
    /** Named status words (`GPUIterationStatusPort` names). */
    status: {
      /** One word, 1 if another fixed propagation or adaptive merge is possible, otherwise 0. */
      iterationLimitReached: GraphDataView<'uint32'>;
      /** Source-label completeness, independent of local merge-budget exhaustion. Incumbents remain available. */
      complete?: GraphDataView<'uint32'>;
    };
  };
};
export type GPURasterRegionGraphOutput = GPURasterRegionGraphProps['output'];

/** Pure device-specific admission and provenance, without graph mutation or cached device state. */
export function getRasterRegionGraphExecutionPlan(
  props: Pick<
    GPURasterRegionGraphProps,
    | 'id'
    | 'grid'
    | 'regionCount'
    | 'maximumIterations'
    | 'mergeMode'
    | 'searchStructure'
    | 'mergeThreshold'
  > & {output?: {status: {complete?: GraphDataView<'uint32'>}}},
  device: Pick<Device, 'limits'>
) {
  const id = props.id ?? 'raster-region-graph';
  const count = validateRasterGrid(id, props.grid);
  const requestedSearchStructure = props.searchStructure ?? 'auto';
  validateRequirement(
    id,
    ['auto', 'dense', 'boruvka'].includes(requestedSearchStructure),
    "searchStructure must be 'auto', 'dense' or 'boruvka'"
  );
  validateRequirement(
    id,
    ['fixed', 'adaptive'].includes(props.mergeMode ?? 'fixed'),
    "mergeMode must be 'fixed' or 'adaptive'"
  );
  validateRequirement(
    id,
    props.mergeMode !== 'adaptive' || requestedSearchStructure !== 'boruvka',
    "searchStructure 'boruvka' requires mergeMode 'fixed'"
  );
  validateCapacity(id, 'grid cell count', count, 1, MAXIMUM_REGION_GRAPH_CELL_COUNT);
  validateCapacity(
    id,
    'maximumIterations',
    props.maximumIterations,
    1,
    MAXIMUM_REGION_GRAPH_ITERATIONS
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(props.mergeThreshold)) && props.mergeThreshold >= 0,
    'mergeThreshold must be a nonnegative finite float32 value'
  );
  const bindingMaximum = Math.min(
    device.limits.maxStorageBufferBindingSize,
    device.limits.maxBufferSize
  );
  const selectedSearchStructure: 'dense' | 'boruvka' =
    requestedSearchStructure === 'dense' ||
    props.mergeMode === 'adaptive' ||
    (requestedSearchStructure === 'auto' &&
      props.regionCount <= MAXIMUM_REGION_GRAPH_REGION_COUNT &&
      (props.maximumIterations < props.regionCount - 1 || count * 8 > bindingMaximum))
      ? 'dense'
      : 'boruvka';
  validateCapacity(
    id,
    'regionCount',
    props.regionCount,
    1,
    selectedSearchStructure === 'dense'
      ? MAXIMUM_REGION_GRAPH_REGION_COUNT
      : MAXIMUM_BORUVKA_REGION_GRAPH_REGION_COUNT
  );
  const resourceBounds = {
    labelsByteLength: count * 4,
    valuesByteLength: count * 4,
    matrixByteLength: props.regionCount ** 2 * 4,
    regionByteLength: props.regionCount * 4,
    fixedPointSumsByteLength: props.regionCount * FIXED_POINT_WORD_COUNT * 4,
    edgeEndpointsByteLength: selectedSearchStructure === 'boruvka' ? count * 8 : 0,
    bindingMaximumByteLength: bindingMaximum
  };
  // Dense accumulation/means use five bindings; adaptive final publication and Boruvka's
  // minimum-edge pass use six. Check this before any recipe assembly or inner allocation.
  const peakStorageBindingCount =
    props.mergeMode === 'adaptive' ||
    (selectedSearchStructure === 'boruvka' &&
      getRasterRegionBoruvkaRoundCount(props.regionCount, props.maximumIterations) > 0)
      ? 6
      : 5;
  validateRequirement(
    id,
    peakStorageBindingCount <= device.limits.maxStorageBuffersPerShaderStage,
    `${selectedSearchStructure} route needs ${peakStorageBindingCount} storage buffers per shader stage; received limit ${device.limits.maxStorageBuffersPerShaderStage}; use searchStructure 'dense' when its five bindings fit`
  );
  const constraints: (GPUAnalysisRoute & {received: number; satisfied: boolean})[] = [];
  for (const [name, bytes] of Object.entries(resourceBounds)) {
    if (name === 'bindingMaximumByteLength' || bytes === 0) continue;
    const subject = name === 'edgeEndpointsByteLength' ? 'edge endpoints' : name;
    validateGPUStorageBindingByteLength(
      device as Device,
      id,
      subject,
      bytes,
      'reduce grid dimensions or regionCount'
    );
    validateCapacity(id, `${subject} (maxBufferSize)`, bytes, 0, device.limits.maxBufferSize);
    constraints.push({
      name: subject,
      kind: 'allocation',
      quantity: subject,
      unit: 'bytes',
      appliesWhen: `selectedSearchStructure === '${selectedSearchStructure}'`,
      maximum: bindingMaximum,
      scope: 'binding',
      received: bytes,
      satisfied: true,
      accuracyImplications: 'Binding admission does not change linkage semantics.',
      source: 'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph.ts'
    });
  }
  const estimatedCommandNodeCount =
    (props.mergeMode === 'adaptive'
      ? 8 + 2 * (props.maximumIterations + 1)
      : selectedSearchStructure === 'boruvka'
        ? 13 + 4 * getRasterRegionBoruvkaRoundCount(props.regionCount, props.maximumIterations)
        : 10 + props.maximumIterations) + Number(Boolean(props.output?.status.complete));
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  return {
    route: selectedSearchStructure,
    feasible: true,
    constraints,
    accuracyImplications: [
      props.mergeMode === 'adaptive'
        ? 'Adaptive mode merges one closest component pair per round and updates component means.'
        : 'Fixed linkage uses original means; dense rounds propagate labels, Boruvka phases merge components. An unfinished budget can yield different incumbents.'
    ],
    requestedSearchStructure,
    selectedSearchStructure,
    resourceBounds,
    peakStorageBindingCount,
    estimatedCommandNodeCount
  };
}

export type GPURasterRegionGraphExecutionPlan = ReturnType<
  typeof getRasterRegionGraphExecutionPlan
>;
// Check the shared catalog planner contract without a runtime dependency on the catalog.
const plannerContract: GPUAnalysisExecutionPlanner<
  Parameters<typeof getRasterRegionGraphExecutionPlan>[0],
  Pick<Device, 'limits'>
> = {getExecutionPlan: getRasterRegionGraphExecutionPlan};
void plannerContract;
/** Four-neighbor region adjacency with bounded fixed or adaptive mean-difference merging. Explicit rounds use ping-pong graph scratch; no cross-workgroup barrier.
 *
 * Fixed linkage can use raster-boundary edge-list Boruvka: each component chooses its minimum
 * outgoing eligible edge, parallel atomic union-find merges the choices, and at most
 * ceil(log2(regionCount)) phases reach closure. The public dense matrices are preserved and
 * remain subject to device binding limits. `searchStructure: 'dense'` keeps the 256-region
 * reference route, including its bounded propagation semantics. Auto uses Boruvka for larger
 * inputs and for small runs whose dense iteration budget guarantees closure (regionCount - 1).
 * Auto also falls back to dense for small runs when boundary scratch exceeds the device binding
 * limit. Adaptive mode keeps its dense, one-pair-at-a-time reference route.
 *
 * Labels remain source IDs; missing observations exclude only mean contributions. Tile seams
 * must be reconciled before this contributor. Supply labelsComplete for provisional partitions
 * together with output.status.complete; this certificate covers source labels for adjacency and
 * the incumbent mapping, independently of iterationLimitReached. Omission assumes complete labels.
 * getExecutionPlan(device) reports the actual selected route and bounds without declaring nodes.
 * All adaptive merges recompute observation-count-weighted component means; the closest
 * eligible pair wins, with lexicographic source-root ties. One adaptive round performs one merge.
 */
export class GPURasterRegionGraph implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly preferredSearchStructure: 'dense' | 'boruvka';
  /** Upper bound: auto may fall back to dense on devices whose bindings cannot hold the boundary list. */
  readonly estimatedCommandNodeCount: number;
  readonly props: GPURasterRegionGraphProps;
  constructor(inputProps: GPURasterRegionGraphProps) {
    const captured = snapshotConstructionFields(inputProps, [
      'id',
      'grid',
      'labels',
      'labelsComplete',
      'values',
      'regionCount',
      'mergeMode',
      'searchStructure',
      'mergeThreshold',
      'maximumIterations',
      'output'
    ]);
    const output = snapshotConstructionFields(captured.output, [
      'adjacency',
      'weights',
      'mapping',
      'status'
    ]);
    const props = {
      ...captured,
      grid: snapshotRasterRegionGrid(captured.grid),
      output: {
        ...output,
        status: snapshotConstructionFields(output.status, ['iterationLimitReached', 'complete'])
      }
    };
    this.props = props;
    this.id = props.id ?? 'raster-region-graph';
    validateRequirement(
      this.id,
      ['auto', 'dense', 'boruvka'].includes(props.searchStructure ?? 'auto'),
      "searchStructure must be 'auto', 'dense' or 'boruvka'"
    );
    validateRequirement(
      this.id,
      props.mergeMode !== 'adaptive' || props.searchStructure !== 'boruvka',
      "searchStructure 'boruvka' requires mergeMode 'fixed'"
    );
    this.preferredSearchStructure =
      props.searchStructure === 'dense' ||
      props.mergeMode === 'adaptive' ||
      ((props.searchStructure ?? 'auto') === 'auto' &&
        props.regionCount <= MAXIMUM_REGION_GRAPH_REGION_COUNT &&
        props.maximumIterations < props.regionCount - 1)
        ? 'dense'
        : 'boruvka';
    validateRasterGrid(this.id, props.grid);
    validateCapacity(this.id, 'grid.width', props.grid.width);
    validateCapacity(this.id, 'grid.height', props.grid.height);
    validateCapacity(
      this.id,
      'grid cell count',
      props.grid.width * props.grid.height,
      1,
      MAXIMUM_REGION_GRAPH_CELL_COUNT
    );
    validateCapacity(
      this.id,
      'regionCount',
      props.regionCount,
      1,
      this.preferredSearchStructure === 'dense'
        ? MAXIMUM_REGION_GRAPH_REGION_COUNT
        : MAXIMUM_BORUVKA_REGION_GRAPH_REGION_COUNT
    );
    validateCapacity(
      this.id,
      'maximumIterations',
      props.maximumIterations,
      1,
      MAXIMUM_REGION_GRAPH_ITERATIONS
    );
    validateRequirement(
      this.id,
      ['fixed', 'adaptive'].includes(props.mergeMode ?? 'fixed'),
      "mergeMode must be 'fixed' or 'adaptive'"
    );
    validateRequirement(
      this.id,
      Number.isFinite(Math.fround(props.mergeThreshold)) && props.mergeThreshold >= 0,
      'mergeThreshold must be a nonnegative finite float32 value'
    );
    validateRequirement(
      this.id,
      props.labels.length >= props.grid.width * props.grid.height &&
        props.values.length >= props.grid.width * props.grid.height,
      'labels and values must hold at least grid.width times grid.height entries'
    );
    validateRequirement(
      this.id,
      props.output.adjacency.length >= props.regionCount ** 2 &&
        props.output.weights.length >= props.regionCount ** 2 &&
        props.output.mapping.length >= props.regionCount &&
        props.output.status.iterationLimitReached.length >= 1,
      'output.adjacency and output.weights must hold regionCount squared entries, output.mapping regionCount entries and output.status.iterationLimitReached one'
    );
    this.estimatedCommandNodeCount =
      props.mergeMode === 'adaptive'
        ? 8 + 2 * (props.maximumIterations + 1)
        : this.preferredSearchStructure === 'boruvka'
          ? 13 + 4 * getRasterRegionBoruvkaRoundCount(props.regionCount, props.maximumIterations)
          : 10 + props.maximumIterations;
    if (
      this.preferredSearchStructure === 'boruvka' &&
      props.searchStructure !== 'boruvka' &&
      props.regionCount <= MAXIMUM_REGION_GRAPH_REGION_COUNT
    ) {
      this.estimatedCommandNodeCount = Math.max(
        this.estimatedCommandNodeCount,
        10 + props.maximumIterations
      );
    }
    this.estimatedCommandNodeCount += Number(Boolean(props.output.status.complete));
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateGPUStatusPort(this.id, props.output.status);
    validateRequirement(
      this.id,
      !props.labelsComplete || Boolean(props.output.status.complete),
      'labelsComplete requires output.status.complete to retain source-label completeness'
    );
    for (const [name, view] of [
      ['labelsComplete', props.labelsComplete],
      ['output.status.complete', props.output.status.complete]
    ] as const) {
      if (view)
        validateRequirement(
          this.id,
          view.length === 1,
          `${name} must hold exactly one uint32 word; received ${view.length}`
        );
    }
    const inputs = [
      props.labels,
      props.values,
      ...(props.labelsComplete ? [props.labelsComplete] : [])
    ];
    const outputs = [
      props.output.adjacency,
      props.output.weights,
      props.output.mapping,
      props.output.status.iterationLimitReached,
      ...(props.output.status.complete ? [props.output.status.complete] : [])
    ];
    for (const [name, view, format] of [
      ['labels', props.labels, 'uint32'],
      ['values', props.values, 'float32'],
      ['output.adjacency', props.output.adjacency, 'uint32'],
      ['output.weights', props.output.weights, 'float32'],
      ['output.mapping', props.output.mapping, 'uint32'],
      ['output.status.iterationLimitReached', props.output.status.iterationLimitReached, 'uint32'],
      ['labelsComplete', props.labelsComplete, 'uint32'],
      ['output.status.complete', props.output.status.complete, 'uint32']
    ] as const)
      if (view)
        validateSinglePackedView(
          view,
          [format],
          {id: this.id, contributor: 'GPURasterRegionGraph'},
          name
        );
    validateRasterWritableDispatches(
      this.id,
      inputs,
      outputs,
      this.getDispatches(this.preferredSearchStructure)
    );
  }

  private getDispatches(searchStructure: 'dense' | 'boruvka'): RasterWritableDispatch[] {
    const {props} = this;
    const {adjacency, weights, mapping, status} = props.output;
    const boundary = {inputs: [props.labels], outputs: [adjacency]};
    const completion: RasterWritableDispatch[] = status.complete
      ? [{inputs: props.labelsComplete ? [props.labelsComplete] : [], outputs: [status.complete]}]
      : [];
    if (props.mergeMode === 'adaptive')
      return [
        ...completion,
        boundary,
        {inputs: [adjacency], outputs: [weights, mapping, status.iterationLimitReached]}
      ];
    return [
      ...completion,
      boundary,
      {inputs: [adjacency], outputs: [weights]},
      ...(searchStructure === 'boruvka'
        ? [
            {inputs: [weights], outputs: [status.iterationLimitReached]},
            {inputs: [], outputs: [mapping]}
          ]
        : [{inputs: [weights], outputs: [mapping, status.iterationLimitReached]}])
    ];
  }

  /** Device-resolved route and resource bounds for recording or admission, without declaration. */
  getExecutionPlan(device: Pick<Device, 'limits'>): GPURasterRegionGraphExecutionPlan {
    const plan = getRasterRegionGraphExecutionPlan(this.props, device);
    const {props} = this;
    for (const [name, view] of Object.entries({
      labels: props.labels,
      values: props.values,
      labelsComplete: props.labelsComplete,
      adjacency: props.output.adjacency,
      weights: props.output.weights,
      mapping: props.output.mapping,
      ...props.output.status
    })) {
      if (!view) continue;
      const range = getViewBindingRange(view);
      validateGPUStorageBindingByteLength(
        device as Device,
        this.id,
        `${name} aligned binding`,
        range.size,
        'reduce the view length or use an aligned byteOffset'
      );
      validateCapacity(
        this.id,
        `${name} aligned binding bytes (maxBufferSize)`,
        range.size,
        1,
        device.limits.maxBufferSize
      );
    }
    return plan;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const count = props.grid.width * props.grid.height;
    const regions = props.regionCount;
    // Auto must not reject an existing small dense workload merely because sparse edge scratch
    // needs twice the input binding size. Explicit Boruvka keeps its predictable scratch contract.
    const searchStructure = this.getExecutionPlan(graph.device).selectedSearchStructure;
    if (searchStructure !== this.preferredSearchStructure)
      validateRasterWritableDispatches(
        id,
        [props.labels, props.values, ...(props.labelsComplete ? [props.labelsComplete] : [])],
        [
          props.output.adjacency,
          props.output.weights,
          props.output.mapping,
          props.output.status.iterationLimitReached,
          ...(props.output.status.complete ? [props.output.status.complete] : [])
        ],
        this.getDispatches(searchStructure)
      );
    validateRasterGraph(
      id,
      graph,
      [props.labels, props.values, ...(props.labelsComplete ? [props.labelsComplete] : [])],
      [
        props.output.adjacency,
        props.output.weights,
        props.output.mapping,
        props.output.status.iterationLimitReached,
        ...(props.output.status.complete ? [props.output.status.complete] : [])
      ]
    );
    validateGraphChunkBindingRanges(graph, id, {
      labels: props.labels,
      values: props.values,
      adjacency: props.output.adjacency,
      weights: props.output.weights,
      mapping: props.output.mapping,
      iterationLimitReached: props.output.status.iterationLimitReached,
      ...(props.labelsComplete ? {labelsComplete: props.labelsComplete} : {}),
      ...(props.output.status.complete ? {complete: props.output.status.complete} : {})
    });
    // Check every derived allocation before declaring any graph resource or node.
    for (const [name, entries] of [
      ['means, counts, mapping, other and component means', regions],
      ['fixed-point scale', 1],
      ['fixed-point sums', regions * FIXED_POINT_WORD_COUNT],
      ...(searchStructure === 'boruvka'
        ? ([
            ['edge endpoints', count * 2],
            ['component minima', regions]
          ] as const)
        : [])
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        entries * 4,
        'reduce grid dimensions or regionCount'
      );
      validateCapacity(
        id,
        `${name} scratch byteLength (maxBufferSize)`,
        entries * 4,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    const means = createTransientView(graph, `${id}-means`, 'float32', regions);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', regions);
    const mapping = createTransientView(graph, `${id}-mapping`, 'uint32', regions);
    const other = createTransientView(graph, `${id}-other`, 'uint32', regions);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (props.output.status.complete)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-labels-complete`,
          operation: 'GPURasterRegionGraph',
          invocationCount: 1,
          bindings: [
            ...(props.labelsComplete
              ? [rasterBinding('labelsComplete', props.labelsComplete)]
              : []),
            rasterBinding('complete', props.output.status.complete, true)
          ],
          body: `complete[completeOffset] = ${props.labelsComplete ? 'select(0u, 1u, labelsComplete[labelsCompleteOffset] != 0u)' : '1u'};`
        })
      );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-clear`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions * regions,
        bindings: [rasterBinding('adjacency', props.output.adjacency, true)],
        body: 'adjacency[adjacencyOffset + index] = 0u;'
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-boundaries`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count,
        bindings: [
          rasterBinding('labels', props.labels),
          {
            name: 'adjacency',
            view: props.output.adjacency,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        body: `let label = labels[labelsOffset + index]; if (label >= ${regions}u) { return; }
let neighbors = array<u32, 2>(index + 1u, index + ${props.grid.width}u);
for (var direction = 0u; direction < 2u; direction++) { if (direction == 0u && index % ${props.grid.width}u == ${props.grid.width - 1}u) { continue; } let neighbor = neighbors[direction]; if (neighbor >= ${count}u) { continue; } let other = labels[labelsOffset + neighbor]; if (other < ${regions}u && other != label) { atomicAdd(&adjacency[adjacencyOffset + label * ${regions}u + other], 1u); atomicAdd(&adjacency[adjacencyOffset + other * ${regions}u + label], 1u); } }`
      })
    );
    // Per-pixel exact fixed-point accumulation: O(N) total work, order independent, no per-region full scans.
    const scale = createTransientView(graph, `${id}-scale`, 'uint32', 1);
    const sums = createTransientView(
      graph,
      `${id}-sums`,
      'uint32',
      regions * FIXED_POINT_WORD_COUNT
    );
    nodes.push(
      ...createFixedPointScaleNodes(graph, {
        id: `${id}-means-scale`,
        operation: 'GPURasterRegionGraph',
        values: graph.createDataView(props.values.buffer, {
          format: 'float32',
          byteOffset: props.values.byteOffset,
          length: count
        }),
        scale
      }),
      createFixedPointClearNode(graph, {
        id: `${id}-means-clear-sums`,
        operation: 'GPURasterRegionGraph',
        words: sums
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means-clear-counts`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [rasterBinding('counts', counts, true)],
        body: 'counts[countsOffset + index] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means-accumulate`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count,
        bindings: [
          rasterBinding('labels', props.labels),
          rasterBinding('values', props.values),
          rasterBinding('scale', scale),
          {name: 'sums', view: sums, type: 'atomic<u32>', access: 'read_write'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}${getFixedPointAtomicAddWGSL('sums')}`,
        body: `let label = labels[labelsOffset + index]; let value = values[valuesOffset + index];
if (label >= ${regions}u || !finite(value)) { return; }
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], ${getFixedPointHeadroomBits(count)}u);
addFixedPointToSums(label, quantizeFixedPoint(value, unitExponent));
atomicAdd(&counts[countsOffset + label], 1u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('scale', scale),
          rasterBinding('sums', sums),
          rasterBinding('means', means, true),
          rasterBinding('counts', counts),
          rasterBinding('mapping', mapping, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}`,
        body: `let count = counts[countsOffset + index];
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], ${getFixedPointHeadroomBits(count)}u);
let words = vec3<u32>(sums[sumsOffset + index * 3u], sums[sumsOffset + index * 3u + 1u], sums[sumsOffset + index * 3u + 2u]);
var mean = getFixedPointFloat(words, unitExponent) / f32(max(count, 1u));
// A sum beyond the f32 range: decode 2^32 times smaller, divide, and scale back.
if (!finite(mean) && count > 0u) { mean = getFixedPointFloat(words, unitExponent - 32) / f32(count) * 4294967296.0; }
means[meansOffset + index] = select(missing(), mean, count > 0u && finite(mean)); mapping[mappingOffset + index] = index;`
      })
    );
    if (props.mergeMode === 'adaptive') {
      return [...nodes, ...getAdaptiveRegionNodes(graph, props, id, means, counts, mapping, other)];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions * regions,
        bindings: [
          rasterBinding('adjacency', props.output.adjacency),
          rasterBinding('means', means),
          rasterBinding('weights', props.output.weights, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `let first = means[meansOffset + index / ${regions}u]; let second = means[meansOffset + index % ${regions}u]; weights[weightsOffset + index] = select(missing(), abs(first - second), adjacency[adjacencyOffset + index] > 0u && finite(first) && finite(second));`
      })
    );
    if (searchStructure === 'boruvka')
      return [...nodes, ...createRasterRegionBoruvkaNodes(graph, props, id, mapping)];
    let source = mapping;
    let destination = other;
    for (let round = 0; round < props.maximumIterations; round++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-merge-${round}`,
          operation: 'GPURasterRegionGraph',
          invocationCount: regions,
          bindings: [
            rasterBinding('weights', props.output.weights),
            rasterBinding('source', source),
            rasterBinding('destination', destination, true)
          ],
          declarations: RASTER_FINITE_WGSL,
          body: `var selected = source[sourceOffset + index]; for (var neighbor = 0u; neighbor < ${regions}u; neighbor++) { let weight = weights[weightsOffset + index * ${regions}u + neighbor]; if (finite(weight) && weight <= ${getWGSLFloatLiteral(props.mergeThreshold)}) { selected = min(selected, source[sourceOffset + neighbor]); } } destination[destinationOffset + index] = selected;`
        })
      );
      [source, destination] = [destination, source];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPURasterRegionGraph',
        // contract-allow: single-invocation dense reference diagnostic scans at most MAXIMUM_REGION_GRAPH_REGION_COUNT squared pairs.
        invocationCount: 1,
        bindings: [
          rasterBinding('weights', props.output.weights),
          rasterBinding('source', source),
          rasterBinding('mapping', props.output.mapping, true),
          rasterBinding('iterationLimitReached', props.output.status.iterationLimitReached, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var pending = 0u; for (var region = 0u; region < ${regions}u; region++) { let current = source[sourceOffset + region]; mapping[mappingOffset + region] = current; for (var neighbor = 0u; neighbor < ${regions}u; neighbor++) { let weight = weights[weightsOffset + region * ${regions}u + neighbor]; if (finite(weight) && weight <= ${getWGSLFloatLiteral(props.mergeThreshold)} && source[sourceOffset + neighbor] < current) { pending = 1u; } } } iterationLimitReached[iterationLimitReachedOffset] = pending;`
      })
    );
    return nodes;
  }
}

function getAdaptiveRegionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegionGraphProps,
  id: string,
  originalMeans: GraphDataView<'float32'>,
  originalCounts: GraphDataView<'uint32'>,
  initialMapping: GraphDataView<'uint32'>,
  temporaryMapping: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const regions = props.regionCount;
  const componentMeans = createTransientView(graph, `${id}-component-means`, 'float32', regions);
  const nodes: GPUCommandNode<Parameters>[] = [];
  let source = initialMapping;
  let destination = temporaryMapping;
  // Include a final diagnostic round which checks eligibility but does not apply another merge.
  for (let round = 0; round <= props.maximumIterations; round++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-adaptive-means-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('source', source),
          rasterBinding('originalMeans', originalMeans),
          rasterBinding('originalCounts', originalCounts),
          rasterBinding('componentMeans', componentMeans, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var count = 0u; var mean = 0.0;
for (var region = 0u; region < ${regions}u; region++) {
  let observations = originalCounts[originalCountsOffset + region];
  if (source[sourceOffset + region] != index || observations == 0u) { continue; }
  let combined = count + observations;
  mean = mean * (f32(count) / f32(combined)) + originalMeans[originalMeansOffset + region] * (f32(observations) / f32(combined)); count = combined;
}
componentMeans[componentMeansOffset + index] = select(missing(), mean, count > 0u);`
      })
    );
    const finalRound = round === props.maximumIterations;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-adaptive-merge-${round}`,
        operation: 'GPURasterRegionGraph',
        // contract-allow: single-invocation adaptive reference selects one edge from at most MAXIMUM_REGION_GRAPH_REGION_COUNT squared pairs.
        invocationCount: 1,
        bindings: [
          rasterBinding('source', source),
          rasterBinding('componentMeans', componentMeans),
          rasterBinding('adjacency', props.output.adjacency),
          rasterBinding('destination', finalRound ? props.output.mapping : destination, true),
          ...(finalRound
            ? [
                rasterBinding('weights', props.output.weights, true),
                rasterBinding(
                  'iterationLimitReached',
                  props.output.status.iterationLimitReached,
                  true
                )
              ]
            : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var selectedFirst = 0xffffffffu; var selectedSecond = 0xffffffffu; var selectedWeight: f32 = ${getWGSLFloatLiteral(props.mergeThreshold)};
for (var first = 0u; first < ${regions}u; first++) {
  let firstRoot = source[sourceOffset + first];
  for (var second = 0u; second < ${regions}u; second++) {
    let secondRoot = source[sourceOffset + second]; let edge = first * ${regions}u + second;
    var weight = missing();
    if (firstRoot != secondRoot && adjacency[adjacencyOffset + edge] > 0u) { weight = abs(componentMeans[componentMeansOffset + firstRoot] - componentMeans[componentMeansOffset + secondRoot]); }
    ${finalRound ? 'weights[weightsOffset + edge] = weight;' : ''}
    let lower = min(firstRoot, secondRoot); let upper = max(firstRoot, secondRoot);
    if (finite(weight) && weight <= ${getWGSLFloatLiteral(props.mergeThreshold)} && (weight < selectedWeight || (weight == selectedWeight && (lower < selectedFirst || (lower == selectedFirst && upper < selectedSecond))))) { selectedWeight = weight; selectedFirst = lower; selectedSecond = upper; }
  }
}
for (var region = 0u; region < ${regions}u; region++) { let root = source[sourceOffset + region]; destination[destinationOffset + region] = ${finalRound ? 'root' : 'select(root, selectedFirst, root == selectedSecond)'}; }
${finalRound ? 'iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, selectedFirst != 0xffffffffu);' : ''}`
      })
    );
    [source, destination] = [destination, source];
  }
  return nodes;
}
