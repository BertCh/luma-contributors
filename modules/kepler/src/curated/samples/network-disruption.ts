// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';
import {CURATED_NETWORK_DISRUPTION_DESCRIPTOR} from '../../descriptors/descriptors-curated-network-disruption';

/** Integration-ready record; shared registries are owned by stream X. */
export const CURATED_NETWORK_DISRUPTION_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'network-disruption',
  question: 'what-is-near-or-reachable',
  title: 'What does a closure cut off?',
  name: 'Network disruption',
  wave: 2,
  method:
    'Demand points route toward facilities on a directed network before and after a closure mask. Nodes and CSR arc slots remain stable. Cumulative accessibility sums the facility weights within a declared cost threshold, while cut-off flags distinguish loss of every directed path from a threshold change. Points snap to the nearest routing endpoint without a connector cost; unreachable pair costs and undefined differences are null.',
  slots: CURATED_NETWORK_DISRUPTION_DESCRIPTOR.inputs.map(input => ({
    name: input.name,
    label:
      input.name === 'network'
        ? 'Street network'
        : input.name === 'facilities'
          ? 'Facilities'
          : 'Demand points',
    kind: input.kind,
    rowSemantics: input.rowSemantics,
    fields: input.fields.map(({parameter, autoAssign, ...field}) => field)
  })),
  execution: {
    status: 'available',
    bindingId: 'curated-network-disruption',
    inputs: {network: 'network', facilities: 'facilities', demand: 'demand'},
    fieldParameters: {
      'network.closure': 'closureProperty',
      'network.forward-cost': 'forwardCostProperty',
      'network.reverse-cost': 'reverseCostProperty',
      'network.one-way': 'onewayProperty',
      'facilities.weight': 'weightProperty'
    },
    parameters: {costMode: 'property', costUnit: 'minutes'}
  },
  exposedParameters: ['costMode', 'maximumCost', 'maximumSnapDistance', 'costUnit'],
  sampleId: 'network-disruption-bridge'
};

export const NETWORK_DISRUPTION_SAMPLE = {
  id: 'network-disruption-bridge',
  license: 'MIT',
  attribution:
    'vis.gl contributors; repository-authored synthetic network, no third-party observations',
  reference: 'THIRD-PARTY-DATA.md#curated-network-disruption-synthetic-samples',
  columns: {
    'network.closure': 'closed',
    'network.forward-cost': 'minutes',
    'network.one-way': 'direction',
    'facilities.weight': 'supply'
  },
  parameters: {
    coordinateSystem: 'planar',
    costMode: 'property',
    costUnit: 'minutes',
    closureProperty: 'closed',
    forwardCostProperty: 'minutes',
    onewayProperty: 'direction',
    weightProperty: 'supply',
    maximumCost: 100,
    maximumSnapDistance: 0
  }
} as const;

/** Same one-way 0→1→2→3 chain as expansion-recipes.spec.ts (the middle cost-2 arc closes).
 * An isolated source line supplies a real disconnected target without inventing a fake point edge.
 * Two demand rows route to five facilities at nodes 0..4; the isolated line also creates node 5.
 */
export function createNetworkDisruptionSample(): Record<string, CatalogFeatureCollection> {
  const nodes = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
    [10, 0],
    [11, 0]
  ];
  return {
    network: {
      type: 'FeatureCollection',
      features: [
        [0, 1, 1, false],
        [1, 2, 2, true],
        [2, 3, 1, false],
        [4, 5, 1, false]
      ].map(([from, to, cost, closed], index) => ({
        id: `street-${index}`,
        geometry: {type: 'LineString', coordinates: [nodes[Number(from)], nodes[Number(to)]]},
        properties: {minutes: cost, closed, direction: 'forward'}
      }))
    },
    demand: {
      type: 'FeatureCollection',
      features: [0, 2].map((node, index) => ({
        id: `demand-${index}`,
        geometry: {type: 'Point', coordinates: nodes[node]},
        properties: {}
      }))
    },
    facilities: {
      type: 'FeatureCollection',
      features: [0, 1, 2, 3, 4].map(node => ({
        id: `facility-${node}`,
        geometry: {type: 'Point', coordinates: nodes[node]},
        properties: {supply: node + 1}
      }))
    }
  };
}

/** Hand-computed recipe fixture expectations; matrices are demand-major. */
export const NETWORK_DISRUPTION_EXPECTATIONS = {
  baselineCosts: [0, 1, 3, 4, Infinity, Infinity, Infinity, 0, 1, Infinity],
  scenarioCosts: [0, 1, Infinity, Infinity, Infinity, Infinity, Infinity, 0, 1, Infinity],
  costDifferences: [0, 0, NaN, NaN, NaN, NaN, NaN, 0, 0, NaN],
  newlyUnreachable: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
  baselineAccessibility: [10, 7],
  scenarioAccessibility: [3, 7],
  accessibilityDifferences: [-7, 0],
  stages: [
    {stage: 'baseline-routing', converged: 1},
    {stage: 'scenario-routing', converged: 1}
  ]
} as const;

export const NETWORK_DISRUPTION_SECOND_SAMPLE = {
  id: 'renamed-parallel-streets',
  columns: {
    'network.closure': 'blocked',
    'network.forward-cost': 'travelTime',
    'network.one-way': 'restriction',
    'facilities.weight': 'capacity'
  },
  parameters: {
    coordinateSystem: 'planar',
    costMode: 'property',
    costUnit: 'minutes',
    closureProperty: 'blocked',
    forwardCostProperty: 'travelTime',
    onewayProperty: 'restriction',
    weightProperty: 'capacity',
    maximumCost: 3,
    maximumSnapDistance: 0
  }
} as const;

/** Independent topology: parallel 0→1 arcs cost 1/4; close the cheap one. Demand is node 0;
 * facilities are deliberately ordered [node 2, node 0, node 1] with weights [5,1,2].
 * This is the parallel-edge fixture from backlog-recipe-composition.spec.ts.
 */
export function createNetworkDisruptionSecondDataset(): Record<string, CatalogFeatureCollection> {
  const nodes = [
    [100, 200],
    [110, 200],
    [120, 200]
  ];
  return {
    network: {
      type: 'FeatureCollection',
      features: [
        [0, 1, 1, 1],
        [0, 1, 4, 0],
        [1, 2, 2, 0]
      ].map(([from, to, travelTime, blocked], index) => ({
        id: `alternate-road-${index}`,
        geometry: {type: 'LineString', coordinates: [nodes[from], nodes[to]]},
        properties: {travelTime, blocked, restriction: 'forward'}
      }))
    },
    demand: {
      type: 'FeatureCollection',
      features: [
        {id: 'warehouse', geometry: {type: 'Point', coordinates: nodes[0]}, properties: {}}
      ]
    },
    facilities: {
      type: 'FeatureCollection',
      features: [2, 0, 1].map((node, index) => ({
        id: ['clinic', 'depot', 'station'][index],
        geometry: {type: 'Point', coordinates: nodes[node]},
        properties: {capacity: [5, 1, 2][index]}
      }))
    }
  };
}

export const NETWORK_DISRUPTION_SECOND_EXPECTATIONS = {
  baselineCosts: [3, 0, 1],
  scenarioCosts: [6, 0, 4],
  costDifferences: [3, 0, 3],
  newlyUnreachable: [0, 0, 0],
  baselineAccessibility: [8],
  scenarioAccessibility: [1],
  accessibilityDifferences: [-7],
  stages: [
    {stage: 'baseline-routing', converged: 1},
    {stage: 'scenario-routing', converged: 1}
  ]
} as const;
