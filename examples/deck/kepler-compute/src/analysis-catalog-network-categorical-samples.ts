// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '@luma-contributors/kepler';

function makePoints(
  coordinates: number[][],
  properties: Record<string, unknown>[]
): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map((position, index) => ({
      id: `zone-${index}`,
      geometry: {type: 'Point', coordinates: position},
      properties: properties[index]
    }))
  };
}

/** Small explicit fixtures for every network/categorical operation; all defaults are runnable. */
export function getNetworkCategoricalSampleInputs(
  operationId: string
): Record<string, string> | undefined {
  const points = makePoints(
    [
      [-74, 40.7],
      [-73.999, 40.7],
      [-73.99, 40.7],
      [-73.989, 40.7]
    ],
    [
      {category: 'A', value: 0, height: 10},
      {category: 'A', value: 2, height: 10},
      {category: 'B', value: 8, height: 10},
      {category: 'B', value: 10, height: 10}
    ]
  );
  const lines: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      {
        id: 'first',
        geometry: {
          type: 'LineString',
          coordinates: [
            [0, 0],
            [1, 0]
          ]
        },
        properties: {cost: 1, slope: 0.1}
      },
      {
        id: 'second',
        geometry: {
          type: 'LineString',
          coordinates: [
            [1, 0],
            [3, 0]
          ]
        },
        properties: {cost: 2, slope: 0.1}
      }
    ]
  };
  const zones = makePoints(
    [
      [-74, 40.7],
      [-73.99, 40.7],
      [-73.97, 40.7]
    ],
    [
      {population: 10, outflow: 6},
      {population: 20, outflow: 4},
      {population: 30, outflow: 2}
    ]
  );
  const collections: Record<string, CatalogFeatureCollection> = {};
  switch (operationId) {
    case 'colocation-quotient':
    case 'local-category-dependence':
    case 'spatial-entropy':
    case 'category-pair-counts':
    case 'geographical-detector':
    case 'redcap-single-linkage':
    case 'healpix-point-to-cell':
    case 'healpix-cell-aggregation':
      collections['points'] = points;
      break;
    case 'network-potential-path-area':
    case 'network-travel-time-percentiles':
      collections['lines'] = lines;
      break;
    case 'kernel-flow-smoothing':
      collections['flows'] = {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'LineString',
              coordinates: [
                zones.features[0].geometry!.coordinates,
                zones.features[1].geometry!.coordinates
              ]
            },
            properties: {origin: 'zone-0', destination: 'zone-1', weight: 5}
          }
        ]
      };
      collections['zones'] = zones;
      break;
    case 'radiation-flow-model':
      collections['zones'] = zones;
      break;
    case 'map-matched-edge-volumes':
      collections['edges'] = lines;
      collections['tracks'] = {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'LineString',
              coordinates: [
                [0, 0],
                [1, 0],
                [3, 0]
              ]
            },
            properties: {traversedEdges: [0, 1, null, 0]}
          }
        ]
      };
      break;
    case 'datum-grid-transform':
      collections['points'] = points;
      collections['samples'] = makePoints(
        [
          [-180, -90],
          [180, -90],
          [-180, 90],
          [180, 90]
        ],
        Array.from({length: 4}, () => ({
          longitudeShift: 0.001,
          latitudeShift: 0.002,
          heightShift: 3
        }))
      );
      break;
    case 'sinkhorn-divergence':
      collections['cells'] = makePoints(
        [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, 1]
        ],
        [
          {source: 1, target: 0},
          {source: 0, target: 1},
          {source: 0, target: 0},
          {source: 0, target: 0}
        ]
      );
      break;
    default:
      return undefined;
  }
  return Object.fromEntries(
    Object.entries(collections).map(([name, collection]) => [name, JSON.stringify(collection)])
  );
}
