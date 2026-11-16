// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor
} from './operation-descriptor-types';
import {
  CURATED_NETWORK_DISRUPTION_BINDING,
  MAXIMUM_NETWORK_DISRUPTION_ARC_COST,
  NETWORK_DISRUPTION_LIMITS
} from '../curated/bindings/curated-network-disruption-binding';

const NETWORK_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    role: 'closure',
    label: 'Closed street (boolean or numeric 0/1)',
    accepts: ['boolean', 'integer', 'number'],
    required: true,
    parameter: 'closureProperty'
  },
  {
    role: 'forward-cost',
    label: 'Forward routing cost',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: {whenParameter: 'costMode', equals: ['property']},
    parameter: 'forwardCostProperty'
  },
  {
    role: 'reverse-cost',
    label: 'Reverse routing cost (empty uses forward)',
    accepts: ['number', 'integer'],
    measurement: ['amount'],
    required: false,
    parameter: 'reverseCostProperty'
  },
  {
    role: 'one-way',
    label: 'One-way restriction',
    accepts: ['boolean', 'category', 'integer', 'text'],
    required: false,
    parameter: 'onewayProperty'
  }
];
const FACILITY_FIELDS: readonly KeplerOperationFieldDescriptor[] = [
  {
    role: 'weight',
    label: 'Facility opportunity weight (empty counts facilities)',
    accepts: ['number', 'integer'],
    measurement: ['count', 'amount'],
    required: false,
    parameter: 'weightProperty',
    autoAssign: false
  }
];

export const CURATED_NETWORK_DISRUPTION_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: CURATED_NETWORK_DISRUPTION_BINDING.id,
  questions: ['what-is-near-or-reachable'],
  action: 'Compare street closures',
  prominence: 2,
  inputs: [
    {name: 'network', kind: 'network', fields: NETWORK_FIELDS},
    {name: 'facilities', kind: 'points', rowSemantics: ['objects'], fields: FACILITY_FIELDS},
    {name: 'demand', kind: 'points', rowSemantics: ['objects', 'events'], fields: []}
  ],
  parameters: CURATED_NETWORK_DISRUPTION_BINDING.parameters
    .filter(
      parameter =>
        ![...NETWORK_FIELDS, ...FACILITY_FIELDS].some(field => field.parameter === parameter.name)
    )
    .map(parameter => ({
      name: parameter.name,
      level: ['costMode', 'maximumCost', 'maximumSnapDistance', 'costUnit'].includes(parameter.name)
        ? 'basic'
        : 'advanced',
      apply: 'rebuild',
      defaultPolicy: {from: 'constant'},
      ...(parameter.name === 'costUnit'
        ? {activeWhen: {parameter: 'costMode', equals: ['property']}}
        : {})
    })),
  requirements: [
    ...CURATED_NETWORK_DISRUPTION_BINDING.inputs.flatMap(input => [
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
        remedy: 'Filter the inputs to the study area'
      },
      {
        id: 'maximum-vertices' as const,
        input: input.name,
        severity: 'block' as const,
        limit: input.maximumVertices,
        remedy: 'Reduce vertices or supply one Point per demand/facility row'
      },
      {
        id: 'declared-coordinates' as const,
        input: input.name,
        severity: 'block' as const,
        remedy: 'Confirm a common coordinate system and the endpoint-snapping distance units'
      }
    ]),
    {
      id: 'maximum-network-edges',
      input: 'network',
      severity: 'block',
      limit: NETWORK_DISRUPTION_LIMITS.edges,
      remedy: 'Profile the chosen shared-vertex/snap topology and reduce physical edges'
    },
    {
      id: 'maximum-network-nodes',
      input: 'network',
      severity: 'block',
      limit: NETWORK_DISRUPTION_LIMITS.nodes,
      remedy: 'Profile and reduce the chosen network node count'
    },
    {
      id: 'no-missing-values',
      input: 'network',
      role: 'closure',
      severity: 'block',
      remedy: 'Supply true/false or numeric 0/1 on every street'
    },
    {
      id: 'maximum-network-arcs',
      input: 'network',
      severity: 'block',
      limit: NETWORK_DISRUPTION_LIMITS.arcs,
      remedy: 'Profile and reduce the directed arc count'
    },
    ...['forward-cost', 'reverse-cost'].map(role => ({
      id: 'non-negative-values' as const,
      input: 'network',
      role,
      severity: 'block' as const,
      activeWhen: {parameter: 'costMode', equals: ['property']},
      remedy:
        'Correct negative routing costs; missing/non-numeric values exclude that direction and are reported'
    })),
    {
      id: 'non-negative-values',
      input: 'facilities',
      role: 'weight',
      severity: 'block',
      remedy: 'Use non-negative numeric opportunity weights'
    },
    {
      id: 'no-missing-values',
      input: 'facilities',
      role: 'weight',
      severity: 'block',
      remedy: 'Fill every facility weight or leave the optional role empty to count facilities'
    }
  ],
  outputs: [
    {
      tableId: 'demand-access',
      kind: 'points',
      presentation: {column: 'accessibility_difference', scale: 'diverging'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['what-is-near-or-reachable']
    },
    {tableId: 'pairs', kind: 'pairs', interpretation: {summary: 'row-count', cautions: []}},
    {
      tableId: 'scenario-arcs',
      kind: 'network',
      presentation: {column: 'closed', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: [
    `The prepared float32 arc cost ceiling is ${MAXIMUM_NETWORK_DISRUPTION_ARC_COST}. Rescale larger routing costs before execution; finite individual arcs alone do not prevent path-cost overflow. This ceiling keeps paths through the supported 4096-node network finite.`,
    'Demand points route toward facilities on the original directed topology. Nearest-endpoint snapping adds no connector cost; use endpoint-aligned points or inspect snap distances. Crossings without a shared vertex remain disconnected. Shared vertices split source-part property costs proportionally by length.',
    'A true or numeric 1 closure masks every permitted direction of the source street. Nodes, split edge IDs, parallel edges and CSR slots remain unchanged. Forward/reverse one-way restrictions are honored. Missing property costs exclude that direction with a diagnostic; negative costs reject the run.',
    'Cumulative accessibility sums facility weights within maximumCost. Empty weights count facilities; zero weights contribute zero. The difference is scenario minus baseline. Cut-off and newly-unreachable mean loss of directed reachability, regardless of the threshold or weights; an already unreachable pair is distinguished. Unreachable costs and undefined Infinity differences are null.',
    'Length costs are metres for EPSG:4326 and declared coordinate units for planar data. Property costs use the analyst-declared costUnit, including the abstract cost-unit when no physical unit is supplied. Threshold 100 and node snap distance 100 are stated constants; set them for the data and chosen units.',
    'All settings rebuild. Separate bounds apply to 4096 physical edges, 4096 nodes, 8192 directed arcs, 128 demand rows and 128 facilities, with at most 16384 pairs. Two searches use 16 lanes and 64 iterations; either nonconverged search prevents publication. No sampled significance or causal inference is implied.'
  ]
};
