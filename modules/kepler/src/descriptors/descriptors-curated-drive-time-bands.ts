// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor
} from './operation-descriptor-types';
import {
  CURATED_DRIVE_TIME_BANDS_BINDING,
  DRIVE_TIME_BANDS_LIMITS
} from '../curated/bindings/curated-drive-time-bands-binding';

const NETWORK_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    role: 'forward-cost',
    label: 'Street travel time',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: true,
    parameter: 'forwardCostProperty'
  },
  {
    role: 'reverse-cost',
    label: 'Reverse travel time (empty uses forward)',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: false,
    parameter: 'reverseCostProperty',
    autoAssign: false
  },
  {
    role: 'one-way',
    label: 'One-way restriction',
    accepts: ['boolean', 'category', 'integer', 'text'],
    required: false,
    parameter: 'onewayProperty',
    autoAssign: false
  }
];
const DEMAND_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    role: 'weight',
    label: 'Demand value (empty counts points)',
    accepts: ['number', 'integer'],
    measurement: ['count', 'amount'],
    required: false,
    parameter: 'weightProperty',
    autoAssign: false
  }
];

export const CURATED_DRIVE_TIME_BANDS_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: CURATED_DRIVE_TIME_BANDS_BINDING.id,
  questions: ['what-is-near-or-reachable'],
  action: 'Map drive-time bands',
  prominence: 1,
  inputs: [
    {name: 'network', kind: 'network', fields: NETWORK_FIELDS},
    {name: 'facilities', kind: 'points', rowSemantics: ['objects'], fields: []},
    {name: 'demand', kind: 'points', rowSemantics: ['objects', 'events'], fields: DEMAND_FIELDS}
  ],
  parameters: CURATED_DRIVE_TIME_BANDS_BINDING.parameters
    .filter(
      parameter =>
        ![...NETWORK_FIELDS, ...DEMAND_FIELDS].some(field => field.parameter === parameter.name)
    )
    .map(parameter => ({
      name: parameter.name,
      level: parameter.name === 'gridSize' ? 'advanced' : 'basic',
      apply: 'rebuild',
      defaultPolicy: {from: 'constant'}
    })),
  requirements: [
    ...CURATED_DRIVE_TIME_BANDS_BINDING.inputs.flatMap(input => [
      {
        id: 'minimum-rows' as const,
        input: input.name,
        severity: 'block' as const,
        limit: 1,
        remedy: `Supply nonempty ${input.name}`
      },
      {
        id: 'maximum-rows' as const,
        input: input.name,
        severity: 'block' as const,
        limit: input.maximumFeatures,
        remedy: 'Filter to a smaller study population'
      },
      {
        id: 'maximum-vertices' as const,
        input: input.name,
        severity: 'block' as const,
        limit: input.maximumVertices,
        remedy: 'Reduce the street vertices or supply one Point per demand/facility row'
      },
      {
        id: 'declared-coordinates' as const,
        input: input.name,
        severity: 'block' as const,
        remedy:
          'Confirm a common coordinate system; snap distance and corridor radius use metres for geographic data or planar coordinate units'
      }
    ]),
    {
      id: 'maximum-network-nodes',
      input: 'network',
      severity: 'block',
      limit: DRIVE_TIME_BANDS_LIMITS.nodes,
      remedy: 'Profile and reduce the network node count'
    },
    {
      id: 'maximum-network-arcs',
      input: 'network',
      severity: 'block',
      limit: DRIVE_TIME_BANDS_LIMITS.arcs,
      remedy: 'Profile and reduce directed segments after splitting every bend'
    },
    ...['forward-cost', 'reverse-cost'].map(role => ({
      id: 'non-negative-values' as const,
      input: 'network',
      role,
      severity: 'block' as const,
      remedy:
        'Use non-negative travel time costs in the declared time unit; missing costs remove that direction with a diagnostic'
    })),
    {
      id: 'non-negative-values',
      input: 'demand',
      role: 'weight',
      severity: 'block',
      remedy: 'Use non-negative demand values'
    },
    {
      id: 'no-missing-values',
      input: 'demand',
      role: 'weight',
      severity: 'block',
      remedy: 'Fill every demand value or leave the optional role empty to count points'
    }
  ],
  outputs: [
    {
      tableId: 'demand-bands',
      kind: 'points',
      kindFrom: 'demand',
      presentation: {column: 'band', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['how-much-per-area']
    },
    {
      tableId: 'band-summary',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    },
    {
      tableId: 'isobands',
      kind: 'polygons',
      presentation: {column: 'band', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: [
    'Requires travel time properties, in declared seconds, minutes or hours. Band breaks use the same unit; no speed or time is inferred from street length. Defaults [5,10,15] minutes, 100 distance units for snapping and corridor radius, and a 32 by 32 map raster are stated constants, not estimates from the data.',
    'Travel is outward from facilities; the minimum time across all snapped facilities determines each demand band. Bands are disjoint, include their upper bound, and include zero in the first band. Empty bands are retained. Demand summaries count each point once and optionally sum its non-negative value.',
    'Points snap to the nearest directed street segment. Snapping adds no off-road travel time. Unsnapped demand and demand with no directed path have null travel times; reachable demand beyond all breaks keeps its time and has a null band. Unsnapped facilities seed no routes and are counted in the report.',
    'Shared vertices connect streets; crossings without a shared vertex remain disconnected. Bends retain their vertices and proportionate travel times. Reverse costs and one-way restrictions are honored. Missing street costs remove affected directions with a diagnostic; negative costs reject preparation.',
    'Map polygons are triangles from a buffered raster approximation of network costs, including direct paths from snapped facilities inside streets. Their grid size, corridor radius and finite edge-sampling budget affect their outline. The raster extent is padded so the requested radius and minimum pixel-diagonal radius fit the supported 16-pixel splat window. They are not exact catchment polygons and do not determine demand membership. Geographic inputs use the catalog local azimuthal-equidistant metre frame; geographic map output is transformed back to longitude/latitude.',
    'GPU required. Input, derived segment/node/arc and output bounds apply separately. A search that does not converge in 64 rounds or a band/triangle overflow refuses publication. No sampled significance, per-facility demand totals or turn restrictions are computed.'
  ]
};
