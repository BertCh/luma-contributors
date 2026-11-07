// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRecipeBoundary, getRecipeView as getOrCreateView} from './expansion-recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import type {
  GPUSpatialContext,
  GPURecipeStageStatusPort
} from '../../analysis-core/contracts/index';
import {
  GPUPointInPolygonJoin,
  type GPUPointInPolygonJoinProps
} from '../spatial-join/gpu-point-in-polygon-join';
import {RecipeBuilder, type GPURecipeResult} from './recipe-utils';
import {getSpatialJoinGeometryViews} from '../spatial-join/spatial-join-geometry';
import {getGraphViewChunks} from '../../utils/gpu-contributor-utils';

/** Dense zone summaries. Counts exclude non-finite values; rowCounts retain assigned rows. */
export type GPUZonalDistributionOutput = {
  zoneIds: GraphDataView<'uint32'>;
  rowCounts: GraphDataView<'uint32'>;
  validCounts: GraphDataView<'uint32'>;
  medians: GraphDataView<'float32'>;
  modes: GraphDataView<'float32'>;
  uniqueCounts: GraphDataView<'uint32'>;
  /** Zone-major values, one row per zone and requested probability. */
  quantiles?: GraphDataView<'float32'>;
};

/** Already assigned rows or points requiring assignment to polygon features. */
export type GPUZonalDistributionSource =
  | {
      kind: 'zones';
      zones: GraphDataView<'uint32'>;
      zoneCount: number;
      mask?: GraphDataView<'uint32'>;
    }
  | {kind: 'points'; join: Omit<GPUPointInPolygonJoinProps, 'id' | 'output'>};

export type GPUZonalDistributionRecipeProps = {
  id?: string;
  source: GPUZonalDistributionSource;
  /** Calibrated values aligned with zone rows or points. NaN and infinities are omitted. */
  values: GraphDataView<'float32'>;
  /** Per-frame probabilities in [0, 1], using the linear quantile convention. */
  probabilities?: GraphDataView<'float32'>;
  outputs?: Partial<GPUZonalDistributionOutput>;
};

export type GPUZonalDistributionRecipeResult = GPURecipeResult<GPUZonalDistributionOutput> & {
  spatialContext?: GPUSpatialContext;
};

/**
 * Assigns each point to one feature (the join's deterministic ownership rule), then computes
 * unweighted zone distributions. Prepared zone rows support raster cells or already expanded
 * overlapping memberships. Empty zones retain their ID, zero counts and NaN order statistics.
 * Raster callers must calibrate and apply validity before supplying values.
 */
export function addZonalDistributionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUZonalDistributionRecipeProps
): GPUZonalDistributionRecipeResult {
  const id = props.id ?? 'zonal-distribution-recipe';
  const builder = new RecipeBuilder(graph);
  const stages: GPURecipeStageStatusPort[] = [];
  const source = props.source;
  const zoneCount =
    source.kind === 'zones' ? source.zoneCount : source.join.polygons.featureOffsets.length - 1;
  const zones =
    source.kind === 'zones'
      ? source.zones
      : getOrCreateView(graph, `${id}-assignments`, 'uint32', props.values.length);
  if (zones.length !== props.values.length) {
    throw new Error('Zone values must align with assignments');
  }
  if (source.kind === 'points') {
    const joinStatus = {
      overflow: getOrCreateView(graph, `${id}-join-overflow`, 'uint32', 1),
      uncertainCount: getOrCreateView(graph, `${id}-join-uncertain`, 'uint32', 1)
    };
    builder.add(
      new GPUPointInPolygonJoin({
        ...source.join,
        id: `${id}-join`,
        points: {...source.join.points, sourceIds: undefined},
        polygons: {...source.join.polygons, sourceIds: undefined},
        output: {pointFeatureIds: zones, status: joinStatus}
      })
    );
    stages.push({stage: 'zone-assignment', status: joinStatus});
  }
  const output = props.outputs ?? {};
  const outputs: GPUZonalDistributionOutput = {
    zoneIds: getOrCreateView(graph, `${id}-zone-ids`, 'uint32', zoneCount, output.zoneIds),
    rowCounts: getOrCreateView(graph, `${id}-row-counts`, 'uint32', zoneCount, output.rowCounts),
    validCounts: getOrCreateView(
      graph,
      `${id}-valid-counts`,
      'uint32',
      zoneCount,
      output.validCounts
    ),
    medians: getOrCreateView(graph, `${id}-medians`, 'float32', zoneCount, output.medians),
    modes: getOrCreateView(graph, `${id}-modes`, 'float32', zoneCount, output.modes),
    uniqueCounts: getOrCreateView(
      graph,
      `${id}-unique-counts`,
      'uint32',
      zoneCount,
      output.uniqueCounts
    ),
    ...(props.probabilities
      ? {
          quantiles: getOrCreateView(
            graph,
            `${id}-quantiles`,
            'float32',
            zoneCount * props.probabilities.length,
            output.quantiles
          )
        }
      : {})
  };
  validateRecipeBoundary(
    graph,
    id,
    [
      props.values,
      props.probabilities,
      zones,
      ...(source.kind === 'zones'
        ? [source.mask]
        : [
            ...getGraphViewChunks(source.join.points.positions),
            ...getSpatialJoinGeometryViews(source.join.polygons)
          ])
    ],
    Object.values(outputs)
  );
  if (output.quantiles && !props.probabilities) {
    throw new Error('Quantile output requires probabilities');
  }
  const status = {
    count: getOrCreateView(graph, `${id}-count`, 'uint32', 1),
    requiredCount: getOrCreateView(graph, `${id}-required`, 'uint32', 1),
    overflow: getOrCreateView(graph, `${id}-overflow`, 'uint32', 1)
  };
  builder.add(
    new GPUGroupStatistics({
      id: `${id}-statistics`,
      keys: zones,
      keyCount: zoneCount,
      mask: source.kind === 'zones' ? source.mask : undefined,
      percentiles: props.probabilities,
      columns: [
        {
          values: props.values,
          statistics: [
            'count',
            'median',
            'mode',
            'uniqueCount',
            ...(props.probabilities ? ['percentiles' as const] : [])
          ],
          output: {
            counts: outputs.validCounts,
            medians: outputs.medians,
            modes: outputs.modes,
            uniqueCounts: outputs.uniqueCounts,
            percentiles: outputs.quantiles
          }
        }
      ],
      output: {keys: outputs.zoneIds, counts: outputs.rowCounts, status}
    })
  );
  stages.push({stage: 'zone-distributions', status});
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {zones},
    status: {stages},
    spatialContext: source.kind === 'points' ? source.join.spatialContext : undefined
  };
}
