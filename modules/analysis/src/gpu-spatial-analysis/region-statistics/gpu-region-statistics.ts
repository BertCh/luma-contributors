// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  createTransientView,
  GPUHistogram,
  GPUPointSpatialFilter,
  GPUReduction,
  GPUVisibilityWorkflow,
  GraphVectorView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import type {GPUUint32Rows} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  isGraphVectorView,
  validateGraphPorts,
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validateCompactOutput,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUPickRegionMask, validateRegionPicks} from './gpu-pick-region-mask';
import {
  GPURegionMask,
  validateRegionShape,
  REGION_STATISTICS_WGSL_HELPERS,
  getRegionLonLatPolygonCheckNodes
} from './gpu-region-mask';
import {
  getGridCandidateNodes,
  getGridMaskScatterNode,
  validateGridIndexProps,
  type GridCandidates
} from './region-statistics-grid';
import {
  GPU_REGION_STATISTICS_FLAGS,
  GPU_REGION_STATISTICS_HEADER_LENGTH,
  GPU_REGION_STATISTICS_SUMMARY_LAYOUT,
  resolveGPURegionSpatialContext,
  validateGPURegionSecondarySpatialContext,
  type GPURegionHistogramProps,
  type GPURegionPolygon,
  type GPURegionRadius,
  type GPURegionRectangle,
  type GPURegionSelection,
  type GPURegionStatisticsGridIndex,
  type GPURegionStatisticsOutput
} from './region-statistics-types';

import {validateGPUStatusPort} from '../../analysis-core/contracts/index';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {
  snapshotRegionSelection,
  snapshotRegionStatisticsOutput
} from './region-construction-snapshot';

/** Workgroup size of the counting kernel: lane 0 publishes one global atomic per workgroup. */
const COUNT_WORKGROUP_SIZE = 256;
const OPERATION = 'GPURegionStatistics';
const MAXIMUM_BIN_COUNT = 65536;

/**
 * Properties for {@link GPURegionStatistics}.
 *
 * Per-frame: region bounds, circle, lasso vertices and vertex count, screen transform, a view
 * histogram domain, and every input buffer's contents. Compile-time: the row count, the lasso
 * vertex capacity, `histogram.binCount`, a literal domain, `batchIndex`, the region kind, which
 * optional views exist, `output.selectedRows.ids.length`, and `spatialIndex`.
 */
export type GPURegionStatisticsProps = {
  /** ID prefix for every node and transient. Defaults to `'region-statistics'`. */
  id?: string;
  /**
   * Coordinate space of `positions` and of the region. Omitted means native planar. Rectangle,
   * lasso and pick regions also accept longitude/latitude with linear edges in degree space.
   * Prepare polygons that cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`;
   * optional status counters report unprepared rings. A `'radius'` region is a planar distance
   * and throws in longitude/latitude at construction; project positions to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `region` in position coordinates when declared separately from `spatialContext`.
   * Must be compatible (datum, coordinate space, metric, units, planar frame), with linear
   * longitude/latitude edges. Omitted means `spatialContext` describes both inputs.
   * Screen-space, pick-region and mask selections cannot declare a region context.
   */
  regionSpatialContext?: GPUSpatialContext;
  /** Region input: a shape, a pick-region result or a caller-computed mask. */
  region: GPURegionSelection;
  /** Packed 2D positions. Required for `rectangle`, `polygon`, and `radius`. */
  positions?: GPUFloat32Positions;
  /** Optional `float32` value column aligned with the source rows. */
  values?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional stable IDs used in `output.selectedRows.ids`. Row indices are used when omitted. */
  sourceIds?: GPUUint32Rows;
  /** Optional histogram. Requires `values`. */
  histogram?: GPURegionHistogramProps;
  /** Caller-owned outputs: the packed summary plus optional mask, selected rows and draw count. */
  output: GPURegionStatisticsOutput;
  /**
   * Compile-time. Gathers candidates from a uniform grid index instead of testing every row. Only
   * for world-space `rectangle`, `radius`, and `polygon` regions with packed `positions`,
   * `values`, and `sourceIds`. See {@link GPURegionStatisticsGridIndex} for the identity guarantee
   * and the new `candidatesTruncated` summary flag.
   */
  spatialIndex?: GPURegionStatisticsGridIndex;
};

/**
 * Selects points inside a per-frame rectangle, circle, lasso, or pick region and reduces them
 * into counts, value statistics, a histogram, and stable selected IDs.
 *
 * Every statistic lands in one small `summary` buffer so the application reads back a single
 * bounded copy, for example through `GPURegionStatisticsReadback`.
 */
export class GPURegionStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURegionStatisticsProps;
  /** Resolved source row count. */
  readonly rowCount: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPURegionStatisticsProps) {
    let construction = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'regionSpatialContext',
      'region',
      'positions',
      'values',
      'sourceIds',
      'histogram',
      'output',
      'spatialIndex'
    ]);
    props = construction;
    validateGraphPorts(props.id ?? 'region-statistics', props, {
      positions: {required: false, viewType: 'either', formats: ['float32x2']},
      values: {required: false, viewType: 'either', formats: ['float32']},
      sourceIds: {required: false, viewType: 'either', formats: ['uint32']}
    });
    // Rewrap only public vector metadata for upstream consumers that use local constructors.
    // Every chunk and GPU buffer keeps its identity and order.
    props = {
      ...props,
      positions: isGraphVectorView(props.positions)
        ? new GraphVectorView({...props.positions})
        : props.positions,
      values: isGraphVectorView(props.values)
        ? new GraphVectorView({...props.values})
        : props.values,
      sourceIds: isGraphVectorView(props.sourceIds)
        ? new GraphVectorView({...props.sourceIds})
        : props.sourceIds
    };
    construction = props;
    const constructionHistogram =
      construction.histogram &&
      snapshotConstructionFields(construction.histogram, ['binCount', 'domain']);
    const spatialIndex =
      construction.spatialIndex &&
      snapshotConstructionFields(construction.spatialIndex, ['kind', 'index', 'candidateCapacity']);
    const index =
      spatialIndex &&
      snapshotConstructionFields(spatialIndex.index, [
        'cellOffsets',
        'objectIds',
        'count',
        'overflow',
        'gridSize',
        'bounds'
      ]);
    props = Object.freeze({
      ...construction,
      ...(construction.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(construction.spatialContext)}
        : {}),
      ...(construction.regionSpatialContext
        ? {
            regionSpatialContext: snapshotConstructionSpatialContext(
              construction.regionSpatialContext
            )
          }
        : {}),
      region: snapshotRegionSelection(construction.region),
      output: snapshotRegionStatisticsOutput(construction.output),
      ...(constructionHistogram
        ? {
            histogram: Object.freeze({
              ...constructionHistogram,
              ...(Array.isArray(constructionHistogram.domain)
                ? {
                    domain: Object.freeze([...constructionHistogram.domain]) as readonly [
                      number,
                      number
                    ]
                  }
                : {})
            })
          }
        : {}),
      ...(spatialIndex && index
        ? {
            spatialIndex: Object.freeze({
              ...spatialIndex,
              index: Object.freeze({
                ...index,
                gridSize: Object.freeze([...index.gridSize]) as typeof index.gridSize,
                bounds: Object.freeze([...index.bounds]) as typeof index.bounds
              })
            })
          }
        : {})
    });
    this.id = props.id ?? 'region-statistics';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPURegionSpatialContext(
      id,
      props.spatialContext,
      props.region.kind
    );
    validateGPURegionSecondarySpatialContext(
      this.id,
      props.regionSpatialContext,
      this.spatialContext,
      props.region
    );
    const {region: selection, positions, values, sourceIds, histogram} = props;
    const {mask: outputMask, selectedRows: output, summary, drawInstanceCount} = props.output;
    const rowCount =
      selection.kind === 'mask'
        ? selection.mask.length
        : (positions?.length ?? values?.length ?? sourceIds?.length ?? outputMask?.length);
    if (rowCount === undefined) {
      throw new Error(`${id} cannot infer the source row count`);
    }
    if (rowCount === 0) {
      throw new Error(`${id} requires at least one source row`);
    }
    this.rowCount = rowCount;
    if (
      (selection.kind === 'rectangle' ||
        selection.kind === 'polygon' ||
        selection.kind === 'radius') &&
      !positions
    ) {
      throw new Error(`${id} positions are required for ${selection.kind} regions`);
    }
    for (const [name, view] of [
      ['positions', positions],
      ['values', values],
      ['sourceIds', sourceIds],
      ['output.mask', outputMask],
      ['region.mask', selection.kind === 'mask' ? selection.mask : undefined]
    ] as const) {
      if (view && view.length !== rowCount) {
        throw new Error(`${id} ${name} length must equal the source row count`);
      }
    }
    if (
      positions &&
      (selection.kind === 'rectangle' ||
        selection.kind === 'polygon' ||
        selection.kind === 'radius')
    ) {
      for (const chunk of getGraphViewChunks(positions))
        validateSinglePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    for (const chunk of values ? getGraphViewChunks(values) : []) {
      validateSinglePackedView(chunk, ['float32'], `${id} values`);
    }
    if (histogram) {
      if (!values) {
        throw new Error(`${id} histogram requires values`);
      }
      validateCapacity(id, 'histogram.binCount', histogram.binCount, 1, MAXIMUM_BIN_COUNT);
      const {domain} = histogram;
      if (Array.isArray(domain)) {
        const [minimum, maximum] = domain as readonly number[];
        if (
          !Number.isFinite(Math.fround(minimum)) ||
          !Number.isFinite(Math.fround(maximum)) ||
          minimum > maximum
        ) {
          throw new Error(
            `${id} histogram.domain must have finite float32 endpoints with min <= max`
          );
        }
      } else if (domain && domain !== 'selection') {
        validateSinglePackedView(domain as GraphDataView, ['float32'], `${id} histogram.domain`);
        if ((domain as GraphDataView).length !== 2) {
          throw new Error(`${id} histogram.domain view must hold two float32 values`);
        }
      }
    }
    if (selection.kind === 'rectangle' || selection.kind === 'polygon') {
      validateRegionShape(id, selection);
    }
    if (selection.kind === 'pick-region') {
      validateRegionPicks(id, selection.picks, selection.batchIndex);
    }
    if (selection.kind === 'radius') {
      validateSinglePackedView(selection.circle, ['float32'], `${id} region.circle`);
      if (selection.circle.length !== 3) {
        throw new Error(`${id} region.circle must contain three float32 values`);
      }
    }
    if (selection.kind === 'mask') {
      validatePackedUint32View(selection.mask, `${id} region.mask`);
      if (selection.summary) {
        validatePackedUint32View(selection.summary, `${id} region.summary`);
        validateCapacity(
          id,
          'region.summary.length',
          selection.summary.length,
          GPU_REGION_STATISTICS_HEADER_LENGTH,
          Number.MAX_SAFE_INTEGER
        );
      }
    }
    if (selection.kind === 'mask' && outputMask) {
      throw new Error(`${id} output.mask cannot be combined with a mask region`);
    }
    if (selection.kind === 'mask' && selection.status) {
      validateGPUStatusPort(`${id} region.status`, selection.status);
    }
    if (outputMask) {
      validatePackedUint32View(outputMask, `${id} output.mask`);
    }
    validatePackedUint32View(summary, `${id} output.summary`);
    const binCount = histogram?.binCount ?? 0;
    if (summary.length !== GPU_REGION_STATISTICS_HEADER_LENGTH + binCount) {
      throw new Error(`${id} output.summary must hold 8 + binCount uint32 words`);
    }
    if (output) {
      validateCompactOutput(id, output);
      if (output.ids.length < 1) {
        throw new Error(`${id} output.selectedRows.ids must hold at least one row`);
      }
    }
    if (drawInstanceCount) {
      validatePackedUint32View(drawInstanceCount, `${id} output.drawInstanceCount`);
      if (drawInstanceCount.length < 1) {
        throw new Error(`${id} output.drawInstanceCount must contain one uint32 row`);
      }
      if (!output) {
        throw new Error(`${id} output.drawInstanceCount requires output.selectedRows`);
      }
    }
    if (props.output.status) validateGPUStatusPort(`${id} output.status`, props.output.status);
    if (props.spatialIndex) {
      validateGridIndexProps(id, {...props, spatialIndex: props.spatialIndex});
    }
    validateDisjointWritableViews(id, {
      summary,
      mask: outputMask,
      ids: output?.ids,
      selectedCount: output?.status.count,
      selectedOverflow: output?.status.overflow,
      selectedCandidateOverflow: output?.status.candidateOverflow,
      selectedRequiredCount: output?.status.requiredCount,
      drawInstanceCount,
      ...props.output.status
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        summary,
        outputMask,
        output?.ids,
        output?.status.count,
        output?.status.overflow,
        output?.status.candidateOverflow,
        output?.status.requiredCount,
        drawInstanceCount,
        props.output.status?.antimeridianUncertainCount,
        props.output.status?.poleUncertainCount,
        props.output.status?.overflow,
        props.output.status?.candidateOverflow
      ],
      getStatisticsInputs(props)
    );
  }

  /** Returns optional polygon checks, selection, reductions, compaction and summary nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpatialStatisticsBindings(graph, this.id, this.props, {
      'row scratch': this.rowCount * 4,
      'candidate positions': this.props.spatialIndex
        ? Math.min(this.props.spatialIndex.candidateCapacity, this.rowCount) * 8
        : 0
    });
    const {id, props, rowCount} = this;
    const {region: selection, positions, values, sourceIds, histogram} = props;
    const {mask: outputMask, selectedRows: output, summary, drawInstanceCount} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      summary,
      outputMask,
      output?.ids,
      output?.status.count,
      output?.status.overflow,
      output?.status.candidateOverflow,
      output?.status.requiredCount,
      drawInstanceCount,
      props.output.status?.antimeridianUncertainCount,
      props.output.status?.poleUncertainCount,
      props.output.status?.overflow,
      props.output.status?.candidateOverflow,
      ...getStatisticsInputs(props)
    ]);
    const nodes = getRegionLonLatPolygonCheckNodes(
      graph,
      id,
      OPERATION,
      this.spatialContext,
      selection,
      props.output.status
    );
    const regionOverflow = createTransientView(graph, `${id}-region-overflow`, 'uint32', 1);
    const {spatialIndex} = props;
    const needsSourceMask = !spatialIndex || Boolean(outputMask || output);
    const mask =
      selection.kind === 'mask'
        ? selection.mask
        : (outputMask ??
          (needsSourceMask
            ? createTransientView(graph, `${id}-mask`, 'uint32', rowCount)
            : undefined));
    let candidates: GridCandidates<Parameters> | undefined;

    // Stage 1: selection mask.
    if (spatialIndex) {
      candidates = getGridCandidateNodes(graph, {
        id,
        region: selection as GPURegionRectangle | GPURegionRadius | GPURegionPolygon,
        positions: positions as GPUFloat32Positions,
        values: values as GraphDataView<'float32'> | undefined,
        spatialIndex,
        rowCount,
        regionOverflow
      });
      nodes.push(...candidates.nodes);
      if (mask) {
        // O(rowCount): clear the source-aligned mask, then scatter the candidate mask into it.
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-mask-clear`,
            operation: OPERATION,
            view: mask,
            type: 'u32',
            value: '0u'
          }),
          getGridMaskScatterNode(graph, {id: `${id}-mask-scatter`, candidates, mask, rowCount})
        );
      }
    } else
      switch (selection.kind) {
        case 'rectangle':
        case 'radius':
          if (selection.kind === 'radius' || !selection.screenTransform) {
            nodes.push(
              ...new GPUPointSpatialFilter({
                id: `${id}-filter`,
                // The public upstream filter still dispatches with its local vector constructor.
                positions: (isGraphVectorView(positions)
                  ? new GraphVectorView({...positions})
                  : positions) as GPUFloat32Positions,
                kind: selection.kind === 'radius' ? 'radius' : 'bounds',
                query: selection.kind === 'radius' ? selection.circle : selection.bounds,
                outputMask: mask as GraphDataView<'uint32'>,
                overflow: regionOverflow
              }).getCommandNodes(graph)
            );
            break;
          }
          nodes.push(
            ...new GPURegionMask({
              id: `${id}-region`,
              spatialContext: this.spatialContext,
              regionSpatialContext: props.regionSpatialContext,
              positions: positions as GPUFloat32Positions,
              region: selection,
              output: {mask: mask as GraphDataView<'uint32'>, status: {overflow: regionOverflow}}
            }).getCommandNodes(graph)
          );
          break;
        case 'polygon':
          nodes.push(
            ...new GPURegionMask({
              id: `${id}-region`,
              spatialContext: this.spatialContext,
              regionSpatialContext: props.regionSpatialContext,
              positions: positions as GPUFloat32Positions,
              region: selection,
              output: {mask: mask as GraphDataView<'uint32'>, status: {overflow: regionOverflow}}
            }).getCommandNodes(graph)
          );
          break;
        case 'pick-region':
          nodes.push(
            ...new GPUPickRegionMask({
              id: `${id}-pick`,
              picks: selection.picks,
              batchIndex: selection.batchIndex,
              output: {mask: mask as GraphDataView<'uint32'>, status: {overflow: regionOverflow}}
            }).getCommandNodes(graph)
          );
          break;
        case 'mask':
          if (selection.summary) {
            nodes.push(
              createWGSLKernelNode<Parameters>(graph, {
                id: `${id}-region-overflow`,
                operation: OPERATION,
                variant: 'mask-membership-flags',
                bindings: [
                  {name: 'upstreamSummary', view: selection.summary, type: 'u32', access: 'read'},
                  {name: 'regionOverflow', view: regionOverflow, type: 'u32', access: 'read_write'}
                ],
                invocationCount: 1,
                body: 'regionOverflow[regionOverflowOffset] = upstreamSummary[upstreamSummaryOffset + 7u] & 6u;'
              })
            );
            break;
          }
          if (selection.status?.overflow) {
            nodes.push(
              createWGSLKernelNode<Parameters>(graph, {
                id: `${id}-region-overflow`,
                operation: OPERATION,
                variant: 'mask-overflow',
                bindings: [
                  {
                    name: 'upstreamOverflow',
                    view: selection.status.overflow,
                    type: 'u32',
                    access: 'read'
                  },
                  {name: 'regionOverflow', view: regionOverflow, type: 'u32', access: 'read_write'}
                ],
                invocationCount: 1,
                body: 'regionOverflow[regionOverflowOffset] = select(0u, 1u, upstreamOverflow[upstreamOverflowOffset] != 0u);'
              })
            );
            break;
          }
          nodes.push(
            createFillNode<Parameters>(graph, {
              id: `${id}-region-overflow-clear`,
              operation: OPERATION,
              view: regionOverflow,
              type: 'u32',
              value: '0u'
            })
          );
          break;
      }

    // Stage 2: selected and finite-value counts, plus the selected-and-finite value mask.
    // With a spatial index the stages below run over the candidate slots instead of source rows.
    const statisticsRowCount = candidates ? candidates.capacity : rowCount;
    const statisticsMask = candidates ? candidates.mask : (mask as GraphDataView<'uint32'>);
    const statisticsValues = candidates ? candidates.values : values;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', candidates ? 3 : 2);
    const valueMask = statisticsValues
      ? createTransientView(graph, `${id}-value-mask`, 'uint32', statisticsRowCount)
      : undefined;
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      })
    );
    if (candidates) {
      nodes.push(candidates.getStatusNode(counts));
    }
    nodes.push(
      ...getCountNodes(
        graph,
        id,
        statisticsMask,
        counts,
        statisticsRowCount,
        statisticsValues,
        valueMask
      )
    );

    // Stage 3: value statistics.
    let extentValue: GraphDataView<'float32'> | undefined;
    let sumValue: GraphDataView<'float32'> | undefined;
    if (statisticsValues && valueMask) {
      sumValue = createTransientView(graph, `${id}-sum-value`, 'float32', 1);
      extentValue = createTransientView(graph, `${id}-extent-value`, 'float32', 2);
      nodes.push(
        ...new GPUReduction({
          id: `${id}-sum`,
          input: statisticsValues,
          mask: valueMask,
          output: sumValue,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-extent`,
          input: statisticsValues,
          mask: valueMask,
          output: extentValue,
          operation: 'extent'
        }).getCommandNodes(graph)
      );
    }

    // Stage 4: histogram.
    let bins: GraphDataView<'uint32'> | undefined;
    let binsTotal: GraphDataView<'uint32'> | undefined;
    if (histogram && statisticsValues && valueMask && extentValue) {
      bins = createTransientView(graph, `${id}-bins`, 'uint32', histogram.binCount);
      binsTotal = createTransientView(graph, `${id}-bins-total-value`, 'uint32', 1);
      const domain =
        histogram.domain === undefined || histogram.domain === 'selection'
          ? extentValue
          : histogram.domain;
      nodes.push(
        ...new GPUHistogram<'float32'>({
          id: `${id}-histogram`,
          input: statisticsValues,
          mask: valueMask,
          output: bins,
          domain
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-bins-total`,
          input: bins,
          output: binsTotal,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    // Stage 5: stable selected IDs, clamped to the caller's capacity.
    let selectionTotal: GraphDataView<'uint32'> | undefined;
    if (output) {
      const selectionIds = createTransientView(graph, `${id}-selection-ids`, 'uint32', rowCount);
      selectionTotal = createTransientView(graph, `${id}-selection-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask: mask as GraphDataView<'uint32'>}],
          output: selectionIds,
          count: selectionTotal,
          sourceIds
        }).getCommandNodes(graph)
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-selection-finalize`,
          operation: OPERATION,
          requiredCount: selectionTotal,
          compactIds: selectionIds,
          output: {
            ids: output.ids,
            status: {
              count: output.status.count,
              overflow: output.status.overflow,
              requiredCount: output.status.requiredCount
            }
          },
          extraCounts: drawInstanceCount ? [drawInstanceCount] : []
        })
      );
    }

    // Stage 6: pack the summary.
    nodes.push(
      getSummaryNode(graph, {
        id: `${id}-summary`,
        counts,
        hasCandidateFlag: Boolean(candidates),
        regionOverflowIsFlags: selection.kind === 'mask' && Boolean(selection.summary),
        regionOverflow,
        sumValue,
        extentValue,
        bins,
        binsTotal,
        selectionTotal,
        selectionCapacity: output?.ids.length ?? 0,
        summary
      })
    );
    // Stage 7: named status scalars from the flags word (the summary node binds eight buffers).
    const {overflow, candidateOverflow} = props.output.status ?? {};
    if (overflow || candidateOverflow) {
      nodes.push(getStatusNode(graph, {id: `${id}-status`, summary, overflow, candidateOverflow}));
    }
    // Separate publication keeps the selection copy below the eight-storage-binding limit.
    if (output?.status.candidateOverflow) {
      nodes.push(
        getStatusNode(graph, {
          id: `${id}-selection-candidate-status`,
          summary,
          includeRegionLoss: true,
          candidateOverflow: output.status.candidateOverflow
        })
      );
    }
    return nodes;
  }
}

/** Builds the node that publishes the summary flag bits as named status scalars. */
function getStatusNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    summary: GraphDataView<'uint32'>;
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    includeRegionLoss?: boolean;
  }
): GPUCommandNode<Parameters> {
  const {overflow, candidateOverflow} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'summary', view: props.summary, type: 'u32', access: 'read'}
  ];
  if (overflow) {
    bindings.push({name: 'overflow', view: overflow, type: 'u32', access: 'read_write'});
  }
  if (candidateOverflow) {
    bindings.push({
      name: 'candidateOverflow',
      view: candidateOverflow,
      type: 'u32',
      access: 'read_write'
    });
  }
  const {flags} = GPU_REGION_STATISTICS_SUMMARY_LAYOUT;
  const {selectionTruncated, regionTruncated, candidatesTruncated} = GPU_REGION_STATISTICS_FLAGS;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `status-${overflow ? 'overflow' : 'none'}-${candidateOverflow ? 'candidate' : 'none'}`,
    bindings,
    invocationCount: 1,
    body: `let flags = summary[summaryOffset + ${flags}u];
  ${overflow ? `overflow[overflowOffset] = select(0u, 1u, (flags & ${selectionTruncated | regionTruncated}u) != 0u);` : ''}
  ${candidateOverflow ? `candidateOverflow[candidateOverflowOffset] = select(0u, 1u, (flags & ${candidatesTruncated | (props.includeRegionLoss ? regionTruncated : 0)}u) != 0u);` : ''}`
  });
}

/** Returns every read-only view used by a region statistics contributor. */
function getStatisticsInputs(
  props: GPURegionStatisticsProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {region: selection, histogram} = props;
  const selectionViews: (GraphDataView | undefined)[] = [];
  switch (selection.kind) {
    case 'rectangle':
      selectionViews.push(selection.bounds, selection.screenTransform);
      break;
    case 'polygon':
      selectionViews.push(selection.vertices, selection.vertexCount, selection.screenTransform);
      break;
    case 'radius':
      selectionViews.push(selection.circle);
      break;
    case 'pick-region':
      selectionViews.push(selection.picks);
      break;
    case 'mask':
      selectionViews.push(selection.mask, selection.status?.overflow, selection.summary);
      break;
  }
  const domain =
    histogram?.domain && typeof histogram.domain === 'object' && !Array.isArray(histogram.domain)
      ? (histogram.domain as GraphDataView)
      : undefined;
  const {index} = props.spatialIndex ?? {};
  return [
    props.positions,
    props.values,
    props.sourceIds,
    domain,
    ...selectionViews,
    ...(index ? [index.cellOffsets, index.objectIds, index.count, index.overflow] : [])
  ];
}

/** Builds the fused count pass, one node per nonempty values chunk. */
function getCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  mask: GraphDataView<'uint32'>,
  counts: GraphDataView<'uint32'>,
  rowCount: number,
  values: GraphDataView<'float32'> | GraphVectorView<'float32'> | undefined,
  valueMask: GraphDataView<'uint32'> | undefined
): GPUCommandNode<Parameters>[] {
  const chunks = values ? getGraphViewChunks(values) : [undefined];
  const isVector = isGraphVectorView(values);
  const nodes: GPUCommandNode<Parameters>[] = [];
  let rowStart = 0;
  for (const [chunkIndex, chunk] of chunks.entries()) {
    const chunkLength = chunk ? chunk.length : rowCount;
    if (chunkLength > 0) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'mask', view: mask, type: 'u32', access: 'read'}
      ];
      if (chunk && valueMask) {
        bindings.push({name: 'values', view: chunk, type: 'f32', access: 'read'});
        bindings.push({name: 'valueMask', view: valueMask, type: 'u32', access: 'read_write'});
      }
      bindings.push({name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'});
      nodes.push(
        createSpatialStatisticsCollectiveNode<Parameters>(graph, {
          id: isVector ? `${id}-count-chunk-${chunkIndex}` : `${id}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings,
          invocationCount: chunkLength,
          workgroupSize: COUNT_WORKGROUP_SIZE,
          parameterValues: [rowStart],
          declarations: `
var<workgroup> localSelectedCount: atomic<u32>;
var<workgroup> localValueCount: atomic<u32>;
${REGION_STATISTICS_WGSL_HELPERS}`,
          body: `if (localInvocationIndex == 0u) {
    atomicStore(&localSelectedCount, 0u);
    atomicStore(&localValueCount, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let row = getChunkParameter(36u) + index;
    let selected = mask[maskOffset + row] != 0u;
    ${
      chunk
        ? `let value = values[valuesOffset + index];
    let valued = selected && isFiniteValue(value);
    valueMask[valueMaskOffset + row] = select(0u, 1u, valued);
    if (valued) { atomicAdd(&localValueCount, 1u); }`
        : ''
    }
    if (selected) { atomicAdd(&localSelectedCount, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let selectedTotal = atomicLoad(&localSelectedCount);
    let valueTotal = atomicLoad(&localValueCount);
    if (selectedTotal > 0u) { atomicAdd(&counts[countsOffset], selectedTotal); }
    if (valueTotal > 0u) { atomicAdd(&counts[countsOffset + 1u], valueTotal); }
  }`
        })
      );
    }
    rowStart += chunkLength;
  }
  return nodes;
}

/** Builds the node that packs counts, value statistics, flags, and bins into the summary. */
function getSummaryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    counts: GraphDataView<'uint32'>;
    hasCandidateFlag: boolean;
    regionOverflowIsFlags: boolean;
    regionOverflow: GraphDataView<'uint32'>;
    sumValue?: GraphDataView<'float32'>;
    extentValue?: GraphDataView<'float32'>;
    bins?: GraphDataView<'uint32'>;
    binsTotal?: GraphDataView<'uint32'>;
    selectionTotal?: GraphDataView<'uint32'>;
    selectionCapacity: number;
    summary: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {sumValue, extentValue, bins, binsTotal, selectionTotal} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'regionOverflow', view: props.regionOverflow, type: 'u32', access: 'read'}
  ];
  if (sumValue) bindings.push({name: 'sumValue', view: sumValue, type: 'f32', access: 'read'});
  if (extentValue) {
    bindings.push({name: 'extentValue', view: extentValue, type: 'f32', access: 'read'});
  }
  if (bins) bindings.push({name: 'bins', view: bins, type: 'u32', access: 'read'});
  if (binsTotal) bindings.push({name: 'binsTotal', view: binsTotal, type: 'u32', access: 'read'});
  if (selectionTotal) {
    bindings.push({name: 'selectionTotal', view: selectionTotal, type: 'u32', access: 'read'});
  }
  bindings.push({name: 'summary', view: props.summary, type: 'u32', access: 'read_write'});
  const binCount = bins?.length ?? 0;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'summary',
    bindings,
    invocationCount: Math.max(binCount, 1),
    declarations: `${SPATIAL_STATISTICS_COUNTS_WGSL}
const BIN_COUNT: u32 = ${binCount}u;
const SELECTION_CAPACITY: u32 = ${props.selectionCapacity}u;`,
    body: `${bins ? 'if (index < BIN_COUNT) { summary[summaryOffset + 8u + index] = bins[binsOffset + index]; }' : ''}
  if (index == 0u) {
    let selectedCount = counts[countsOffset];
    let valueCount = counts[countsOffset + 1u];
    let hasValues = valueCount > 0u;
    ${
      sumValue && extentValue
        ? `let sum = select(0.0, sumValue[sumValueOffset], hasValues);
    let mean = select(0.0, divideBySpatialCount(sum, max(valueCount, 1u)), hasValues);
    let minimum = select(0.0, extentValue[extentValueOffset], hasValues);
    let maximum = select(0.0, extentValue[extentValueOffset + 1u], hasValues);`
        : `let sum = 0.0;
    let mean = 0.0;
    let minimum = 0.0;
    let maximum = 0.0;`
    }
    ${
      binsTotal
        ? `let binnedCount = binsTotal[binsTotalOffset];
    let outsideCount = select(0u, valueCount - binnedCount, valueCount > binnedCount);`
        : 'let outsideCount = 0u;'
    }
    let selectionTruncated = ${selectionTotal ? 'selectionTotal[selectionTotalOffset] > SELECTION_CAPACITY' : 'false'};
    let regionTruncated = ${props.regionOverflowIsFlags ? '(regionOverflow[regionOverflowOffset] & 2u) != 0u' : 'regionOverflow[regionOverflowOffset] != 0u'};
    let candidatesTruncated = ${props.hasCandidateFlag ? 'counts[countsOffset + 2u] != 0u' : 'false'}${props.regionOverflowIsFlags ? ' || (regionOverflow[regionOverflowOffset] & 4u) != 0u' : ''};
    summary[summaryOffset + 0u] = selectedCount;
    summary[summaryOffset + 1u] = valueCount;
    summary[summaryOffset + 2u] = bitcast<u32>(sum);
    summary[summaryOffset + 3u] = bitcast<u32>(mean);
    summary[summaryOffset + 4u] = bitcast<u32>(minimum);
    summary[summaryOffset + 5u] = bitcast<u32>(maximum);
    summary[summaryOffset + 6u] = outsideCount;
    summary[summaryOffset + 7u] = select(0u, 1u, selectionTruncated) | select(0u, 2u, regionTruncated) | select(0u, 4u, candidatesTruncated);
  }`
  });
}
