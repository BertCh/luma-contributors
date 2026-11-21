// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/graph/gpu-core';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import type {GPUUint32Rows} from '../../analysis-core/kernels/gpu-contributor-types';
import type {
  GPULonLatPolygonStatusPort,
  GPULineGeometryPort,
  GPUPolygonGeometryPort,
  GPUPackedGeometryPort,
  GPUMultipartGeometryPort
} from '../../analysis-core/contracts/index';
import type {
  GPUNearestFeatureJoinProps,
  GPUNearestFeatureJoinOutput
} from './gpu-nearest-feature-join';
import type {GPUNearestQueryGeometry} from './nearest-types';
import {isMultipartJoinGeometry} from './spatial-join-geometry';
import {getNearestGeometryViews} from './nearest-nodes';

export type NearestDistanceRows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/** Flat internal view of the props that the node builders read. */
export type NearestFeatureJoinViews = Omit<
  GPUNearestFeatureJoinProps<GPUUint32Rows, NearestDistanceRows, GPUNearestQueryGeometry, never>,
  'queries' | 'output'
> &
  Omit<GPUNearestFeatureJoinOutput<GPUUint32Rows, NearestDistanceRows>, 'status'> &
  GPULonLatPolygonStatusPort & {
    /** Point queries, possibly chunked. */
    points?: GPUFloat32Positions;
    /** Line or polygon queries. */
    queries?: GPUPackedGeometryPort<
      GPULineGeometryPort | GPUPolygonGeometryPort | GPUMultipartGeometryPort
    >;
    /** Point query IDs with the chunk topology of `points`. */
    sourceIds?: GPUUint32Rows;
    /** Feature IDs written instead of feature rows. */
    featureIds?: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };

export function getNearestFeatureJoinViews(
  props: GPUNearestFeatureJoinProps<
    GPUUint32Rows,
    NearestDistanceRows,
    GPUNearestQueryGeometry,
    never
  >
): NearestFeatureJoinViews {
  const {queries, output, ...scalars} = props;
  const {status, ...outputs} = output;
  return {
    ...scalars,
    ...outputs,
    points:
      queries.kind === 'points' && !isMultipartJoinGeometry(queries)
        ? queries.positions
        : undefined,
    queries:
      queries.kind === 'points' && !isMultipartJoinGeometry(queries)
        ? undefined
        : (queries as NearestFeatureJoinViews['queries']),
    sourceIds: queries.sourceIds,
    featureIds: props.features.sourceIds,
    overflow: status.overflow,
    candidateCount: status.candidateCount,
    candidateOverflow: status.candidateOverflow,
    antimeridianUncertainCount: status.antimeridianUncertainCount,
    poleUncertainCount: status.poleUncertainCount
  };
}

/** Returns every read-only view of a nearest-feature join. */
export function getNearestFeatureJoinInputViews(
  props: NearestFeatureJoinViews
): (GraphDataView | GraphVectorView | undefined)[] {
  return [
    props.points,
    ...(props.queries ? getNearestGeometryViews(props.queries) : []),
    props.sourceIds,
    props.featureIds,
    props.queryIds,
    props.onAttribute?.left,
    props.onAttribute?.right,
    props.radius,
    ...getNearestGeometryViews(props.features)
  ];
}

/** Returns every writable view of a nearest-feature join. */
export function getNearestFeatureJoinOutputViews(
  props: NearestFeatureJoinViews
): (GraphDataView | GraphVectorView | undefined)[] {
  return [
    props.nearestFeatureIds,
    props.nearestFeatureRows,
    props.nearestDistances,
    props.featureCounts,
    props.overflow,
    props.candidateCount,
    props.candidateOverflow,
    props.candidateTruncated,
    props.antimeridianUncertainCount,
    props.poleUncertainCount,
    props.neighborIds,
    props.neighborRows,
    props.neighborCounts,
    props.neighborDistances,
    props.neighborFootPoints,
    props.neighborSegmentIndices,
    props.neighborQueryPoints,
    props.matches?.ids,
    props.matches?.status.count,
    props.matches?.status.overflow,
    props.matches?.status.requiredCount,
    props.matches?.status.candidateOverflow
  ];
}
