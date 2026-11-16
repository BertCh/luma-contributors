// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';
import {CURATED_DRIVE_TIME_BANDS_BINDING} from '../bindings/curated-drive-time-bands-binding';
import {CURATED_DRIVE_TIME_BANDS_DESCRIPTOR} from '../../descriptors/descriptors-curated-drive-time-bands';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';
/** Integration-ready record; the shared registries remain owned by stream X. */
export const CURATED_DRIVE_TIME_BANDS_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'drive-time-bands',
  question: 'what-is-near-or-reachable',
  title: 'Who is within each drive-time band?',
  name: 'Drive-time bands',
  wave: 1,
  method:
    'Facilities and demand snap to street segments. Directed minimum travel times outward from all facilities classify each demand point once into disjoint bands with inclusive upper bounds. The band table counts demand and sums optional demand values, while map triangles show approximate buffered raster corridors. Travel times come from a declared time-cost property, not street length. This is separate from the reachable-demand total per facility.',
  slots: CURATED_DRIVE_TIME_BANDS_DESCRIPTOR.inputs.map(input => ({
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
    bindingId: CURATED_DRIVE_TIME_BANDS_BINDING.id,
    inputs: {network: 'network', facilities: 'facilities', demand: 'demand'},
    fieldParameters: {
      'network.forward-cost': 'forwardCostProperty',
      'network.reverse-cost': 'reverseCostProperty',
      'network.one-way': 'onewayProperty',
      'demand.weight': 'weightProperty'
    },
    parameters: {costUnit: 'minutes'}
  },
  exposedParameters: ['costUnit', 'costBreaks', 'maximumSnapDistance', 'bufferRadius'],
  sampleId: 'drive-time-bands-streets'
};

export const DRIVE_TIME_BANDS_SAMPLE = {
  id: 'drive-time-bands-streets',
  license: 'MIT',
  attribution: 'vis.gl contributors; repository-authored synthetic streets and demand',
  reference: 'THIRD-PARTY-DATA.md#curated-drive-time-bands-synthetic-samples',
  columns: {
    'network.forward-cost': 'minutes',
    'network.one-way': 'direction',
    'demand.weight': 'residents'
  },
  parameters: {
    coordinateSystem: 'planar',
    forwardCostProperty: 'minutes',
    onewayProperty: 'direction',
    weightProperty: 'residents',
    costUnit: 'minutes',
    costBreaks: [1, 4, 8],
    maximumSnapDistance: 1,
    bufferRadius: 5,
    gridSize: 32
  }
} as const;

/** Two facilities on a bidirectional chain, plus a disconnected road and an unsnapped point.
 * Direct same-road travel is shorter than going to an endpoint and back. Breaks are exact.
 */
export function createDriveTimeBandsSample(): Record<string, CatalogFeatureCollection> {
  return {
    network: {
      type: 'FeatureCollection',
      features: [
        [0, 8],
        [8, 16],
        [16, 24],
        [24, 32],
        [80, 88]
      ].map(([from, to], index) => ({
        id: `street-${index}`,
        geometry: {
          type: 'LineString',
          coordinates: [
            [from, 0],
            [to, 0]
          ]
        },
        properties: {minutes: 8, direction: 'both'}
      }))
    },
    facilities: {
      type: 'FeatureCollection',
      features: [2, 22].map((horizontal, index) => ({
        id: `facility-${index}`,
        geometry: {type: 'Point', coordinates: [horizontal, 0]},
        properties: {}
      }))
    },
    demand: {
      type: 'FeatureCollection',
      features: [
        [2, 0],
        [3, 0],
        [6, 0],
        [8, 0],
        [14, 0],
        [32, 0],
        [84, 0],
        [8, 50]
      ].map((coordinates, index) => ({
        id: `household-${index}`,
        geometry: {type: 'Point', coordinates},
        properties: {residents: index + 1}
      }))
    }
  };
}

/** Hand-computed outward travel on the straight streets; no solver generated these numbers. */
export const DRIVE_TIME_BANDS_EXPECTATIONS = {
  times: [0, 1, 4, 6, 8, 10, Infinity, Infinity],
  demandBands: [0, 0, 1, 2, 2, 0xffffffff, 0xffffffff, 0xffffffff],
  keys: [0, 1, 2],
  counts: [2, 1, 2],
  sums: [3, 3, 9],
  means: [1.5, 3, 4.5]
} as const;

export const DRIVE_TIME_BANDS_SECOND_SAMPLE = {
  id: 'drive-time-bands-renamed-oneway',
  columns: {
    'network.forward-cost': 'outbound',
    'network.reverse-cost': 'inbound',
    'network.one-way': 'restriction',
    'demand.weight': 'customers'
  },
  parameters: {
    coordinateSystem: 'planar',
    forwardCostProperty: 'outbound',
    reverseCostProperty: 'inbound',
    onewayProperty: 'restriction',
    weightProperty: 'customers',
    costUnit: 'seconds',
    costBreaks: [2, 4, 6],
    maximumSnapDistance: 1,
    bufferRadius: 5,
    gridSize: 24
  }
} as const;

/** Different topology/IDs/fields: a one-way street plus an asymmetric bent two-way road.
 * From [105,200], travel to [100,200] is impossible. The bend is a routing vertex;
 * outbound cost 8 splits into 4+4, while inbound 16 splits into 8+8.
 */
export function createDriveTimeBandsSecondDataset(): Record<string, CatalogFeatureCollection> {
  return {
    network: {
      type: 'FeatureCollection',
      features: [
        {
          id: 'one-way',
          geometry: {
            type: 'LineString',
            coordinates: [
              [100, 200],
              [110, 200]
            ]
          },
          properties: {outbound: 4, inbound: 40, restriction: 'forward'}
        },
        {
          id: 'bent-street',
          geometry: {
            type: 'LineString',
            coordinates: [
              [110, 200],
              [110, 210],
              [120, 210]
            ]
          },
          properties: {outbound: 8, inbound: 16, restriction: 'both'}
        }
      ]
    },
    facilities: {
      type: 'FeatureCollection',
      features: [{id: 'depot', geometry: {type: 'Point', coordinates: [105, 200]}, properties: {}}]
    },
    demand: {
      type: 'FeatureCollection',
      features: [
        [100, 200],
        [110, 200],
        [110, 205],
        [110, 210],
        [120, 210]
      ].map((coordinates, index) => ({
        id: ['upstream', 'junction', 'on-bend', 'corner', 'far-end'][index],
        geometry: {type: 'Point', coordinates},
        properties: {customers: [100, 2, 3, 4, 5][index]}
      }))
    }
  };
}

export const DRIVE_TIME_BANDS_SECOND_EXPECTATIONS = {
  times: [Infinity, 2, 4, 6, 10],
  demandBands: [0xffffffff, 0, 1, 2, 0xffffffff],
  keys: [0, 1, 2],
  counts: [1, 1, 1],
  sums: [2, 3, 4],
  means: [2, 3, 4]
} as const;
