// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GPUFlowAggregationProps} from './gpu-flow-aggregation';

/** Capture nested zone literals and port selections; graph view contents remain live. @internal */
export function snapshotFlowAggregationConstructionProps(
  props: GPUFlowAggregationProps
): GPUFlowAggregationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'zones',
    'origins',
    'destinations',
    'originZoneIds',
    'destinationZoneIds',
    'sumOrder',
    'weights',
    'mask',
    'timeWindow',
    'excludeSelfFlows',
    'pairCapacity',
    'maximumProbeCount',
    'output'
  ]);
  // Read the discriminant once, then only the active variant's fields. Inactive fields may
  // be getters that cannot be evaluated for this mode. Unknown kinds reach normal validation.
  const {kind} = snapshotConstructionFields(snapshot.zones, ['kind']);
  const fields =
    kind === 'ids'
      ? ['zoneCount']
      : kind === 'grid'
        ? ['bounds', 'gridSize', 'activeGridSize']
        : kind === 'hexagon'
          ? ['bounds', 'gridSize', 'activeGridSize', 'radius']
          : [];
  const zones = Object.freeze({
    ...snapshotConstructionFields(snapshot.zones, fields as (keyof typeof snapshot.zones)[]),
    kind
  }) as GPUFlowAggregationProps['zones'];
  const output = snapshotConstructionFields(snapshot.output, [
    'ids',
    'status',
    'flowOriginZoneIds',
    'flowDestinationZoneIds',
    'flowCounts',
    'flowWeights',
    'zoneOutCounts',
    'zoneInCounts',
    'zoneOutWeights',
    'zoneInWeights',
    'drawInstanceCount',
    'zoneOutCountExtent',
    'zoneInCountExtent'
  ]);
  return Object.freeze({
    ...snapshot,
    zones:
      zones.kind !== 'grid' && zones.kind !== 'hexagon'
        ? zones
        : Object.freeze({
            ...zones,
            gridSize: Object.freeze([...zones.gridSize]) as readonly [number, number],
            bounds: Array.isArray(zones.bounds)
              ? (Object.freeze([...zones.bounds]) as readonly [number, number, number, number])
              : zones.bounds
          }),
    ...(snapshot.timeWindow
      ? {
          timeWindow: snapshotConstructionFields(snapshot.timeWindow, [
            'timestamps',
            'timestampsLow',
            'window'
          ])
        }
      : {}),
    output: Object.freeze({
      ...output,
      status: snapshotConstructionFields(output.status, [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow',
        'pairOverflow'
      ])
    })
  });
}
