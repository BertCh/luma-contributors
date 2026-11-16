// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  getStorageBindingElementLimit,
  validateCapacity
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  validateCellDispatchBindings,
  validateCellReadWriteDispatch
} from '../cell-indexing/cell-dispatch-validation';
import {
  getDefaultEdgeSlabs,
  getDefaultQuerySort,
  getPolygonEdgeLayoutLength
} from '../spatial-join/spatial-join-passes';
import {
  planSpatialJoinCandidatePages,
  SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT
} from '../spatial-join/spatial-join-candidate-pages';
import type {GPUZonalStatisticsProps} from './gpu-zonal-statistics';
import type {getZonalStatisticsPlan} from './zonal-statistics-passes';

/** Reject only public views actually co-bound as read-only and writable storage. @internal */
export function validateZonalStatisticsReadWriteDispatches(
  id: string,
  props: GPUZonalStatisticsProps,
  plan: ReturnType<typeof getZonalStatisticsPlan>,
  atomicSums: boolean
): void {
  const {output, features} = props;
  const rows =
    features.kind === 'feature-rows' ? features.pointFeatureRows : output.pointFeatureRows;
  const rowChunks = rows ? getGraphViewChunks(rows) : [];
  for (const row of rowChunks) {
    validateCellReadWriteDispatch(
      id,
      {pointFeatureRows: row},
      {
        counts: output.counts,
        valueCounts: output.valueCounts,
        minima: output.minima,
        maxima: output.maxima
      }
    );
    if (atomicSums)
      validateCellReadWriteDispatch(
        id,
        {pointFeatureRows: row},
        {
          sums: output.sums,
          weightSums: output.weightSums
        }
      );
  }
  for (const [name, source] of [
    ['values', props.values],
    ['weights', props.weights]
  ] as const)
    for (const chunk of source ? getGraphViewChunks(source) : [])
      validateCellReadWriteDispatch(
        id,
        {[name]: chunk},
        {
          sums: atomicSums && !props.weights ? output.sums : undefined,
          weightSums: atomicSums && name === 'weights' ? output.weightSums : undefined,
          minima: name === 'values' ? output.minima : undefined,
          maxima: name === 'values' ? output.maxima : undefined
        }
      );
  if (plan.needMeans)
    validateCellReadWriteDispatch(
      id,
      {
        sums: output.sums,
        valueCounts: props.weights ? undefined : output.valueCounts,
        weightSums: props.weights ? output.weightSums : undefined
      },
      {means: output.means}
    );
  if (plan.needDensities)
    validateCellReadWriteDispatch(
      id,
      {counts: output.counts, areas: props.areas ?? output.featureAreas},
      {
        densities: output.densities
      }
    );
  if (plan.needWeightedDensities)
    validateCellReadWriteDispatch(
      id,
      {weightSums: output.weightSums, areas: props.areas ?? output.featureAreas},
      {
        weightedDensities: output.weightedDensities
      }
    );
  if (plan.needVariances)
    validateCellReadWriteDispatch(
      id,
      {valueCounts: output.valueCounts, weightSums: props.weights ? output.weightSums : undefined},
      {
        variances: output.variances,
        standardDeviations: output.standardDeviations
      }
    );
}

/** Preflights row, reduction and nested polygon-join storage before the parent declares anything. @internal */
export function validateZonalStatisticsBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUZonalStatisticsProps,
  pointCount: number,
  featureCount: number,
  plan: ReturnType<typeof getZonalStatisticsPlan>,
  sortedSums: boolean,
  outputs: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  const {features, values, weights} = props;
  const views: Record<string, GraphDataView | GraphVectorView | undefined> = {
    positions: features.kind === 'polygons' ? props.positions : undefined,
    values: plan.needValueStatistics ? values : undefined,
    weights: plan.needValueStatistics ? weights : undefined,
    areas: plan.needDensities || plan.needWeightedDensities ? props.areas : undefined
  };
  if (features.kind === 'polygons')
    Object.assign(views, {
      featurePositions: features.positions,
      featureOffsets: features.featureOffsets,
      polygonOffsets: features.polygonOffsets,
      ringOffsets: features.ringOffsets
    });
  else views['pointFeatureRows'] = features.pointFeatureRows;
  outputs.forEach((view, index) => {
    views[`output ${index}`] = view;
  });
  validateGraphChunkBindingRanges(graph, id, views);
  const scratch: [string, number][] = [];
  if (
    plan.needCounts ||
    plan.needValueStatistics ||
    plan.needGPUAreas ||
    plan.needDensities ||
    plan.needWeightedDensities
  )
    scratch.push(['feature statistic scratch', Math.max(featureCount, 1) * 4]);
  if (values && plan.needValueStatistics) {
    for (const [index, chunk] of getGraphViewChunks(values).entries())
      scratch.push([`value contribution scratch ${index}`, chunk.length * 4]);
    if (values instanceof GraphVectorView) {
      // The prepare stage binds only one 12-byte slot, but allocates one 256-byte slot per chunk.
      validateCapacity(
        id,
        'prepare parameter buffer byteLength',
        values.data.length * 256,
        0,
        graph.device.limits.maxBufferSize
      );
      scratch.push(['prepare parameter slot', 12]);
    }
  }
  if (sortedSums || plan.needVariances)
    scratch.push(
      ['point sort and deviation scratch', pointCount * 4],
      ['segment offsets scratch', (featureCount + 1) * 4]
    );
  if (features.kind === 'polygons') {
    const chunks = getGraphViewChunks(props.positions!);
    const vertexCount = features.positions.length;
    const leafCapacity =
      features.leafCapacity ?? 2 ** Math.ceil(Math.log2(Math.max(featureCount, 1)));
    const edgeSlabs = getDefaultEdgeSlabs(vertexCount, features.candidateCapacity, featureCount);
    const pages = planSpatialJoinCandidatePages(
      graph.device,
      id,
      features.candidateCapacity,
      undefined,
      edgeSlabs ? 16 : 8,
      0,
      chunks.length
    );
    scratch.push(
      [
        'join assignment and page metadata',
        (pointCount + SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT * chunks.length) * 4
      ],
      ['join state', 20],
      ['join candidate page', pages.capacity * (edgeSlabs ? 16 : 8)],
      ['join BVH node bounds', (2 * leafCapacity - 1) * 8],
      ['join feature bounds', Math.max(featureCount, 1) * 8],
      ['join feature radix histogram', 16 * Math.ceil(featureCount / 256) * 4]
    );
    if (getDefaultQuerySort(pointCount, featureCount))
      scratch.push(
        ['join query sort', pointCount * 4],
        ['join query radix histogram', 16 * Math.ceil(pointCount / 256) * 4]
      );
    if (edgeSlabs) {
      const partCount = features.polygonOffsets.length - 1;
      const entryCapacity = Math.min(
        getStorageBindingElementLimit(graph.device, 8, vertexCount),
        Math.max(4 * vertexCount, 1024)
      );
      scratch.push(
        ['join slab parameters', Math.max(partCount, 1) * 24],
        ['join slab owner scan', (partCount + 1) * 4],
        ['join slab ring owners', Math.max(features.ringOffsets.length - 1, 1) * 4],
        ['join slab vertex owners', vertexCount * 4],
        ['join slab offsets', (Math.floor(vertexCount / 4) + 2) * 4],
        ['join slab base edges', vertexCount * 8],
        ['join slab entries', (vertexCount + entryCapacity) * 8],
        ['join slab layout', getPolygonEdgeLayoutLength(vertexCount, featureCount) * 4]
      );
    }
    validateGraphChunkGroupKernelSupport(graph, id, 8);
  }
  validateCellDispatchBindings(graph, id, {}, scratch);
}
