// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';

export const SPACETIME_TEST_POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: Array.from({length: 6}, (_, location) =>
    Array.from({length: 8}, (_, period) => ({
      id: `location-${location}-period-${period}`,
      geometry: {
        type: 'Point' as const,
        coordinates: [(location % 3) * 2, Math.floor(location / 3) * 2]
      },
      properties: {
        location: `location-${location}`,
        time: 1700000000000 + period * 60000,
        value: (location + 1) * (period < 4 ? 1 : 3) + (period % 2),
        second: (location + 1) * (period < 3 ? 1 : 3) + ((period + 1) % 2),
        baseline: location + 1,
        case: (location + period) % 2
      }
    }))
  ).flat()
};

export const SPACETIME_TEST_PARAMETERS: Record<string, string | number> = {
  coordinateSystem: 'planar',
  timeProperty: 'time',
  timeUnit: 'milliseconds',
  spatialBandwidth: 2,
  temporalBandwidth: 120,
  spatialThreshold: 3,
  temporalThreshold: 120,
  gridSize: 2,
  sliceCount: 2,
  permutations: 3,
  seed: 7,
  neighborCount: 2,
  bandCount: 2,
  classCount: 2,
  maximumDistance: 3,
  maximumTimeLag: 120,
  maximumIterations: 3,
  maximumParents: 8,
  maximumWindowLength: 3,
  locationProperty: 'location',
  valueProperty: 'value',
  secondValueProperty: 'second',
  baselineProperty: 'baseline',
  caseProperty: 'case',
  sensitivity: 0.5,
  factorLimit: 4,
  lag: 1,
  variant: 'lagged',
  cost: 'mean',
  penalty: 10,
  minimumSegmentLength: 2
};

/** Two opposite populations observed 100 ms apart, below the old one-second slice floor. */
export const SPACETIME_SUBSECOND_TEST_POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [
    {
      id: 'subsecond-control',
      geometry: {type: 'Point', coordinates: [0, 0]},
      properties: {time: 1700000000000, case: 0}
    },
    {
      id: 'subsecond-case',
      geometry: {type: 'Point', coordinates: [1, 1]},
      properties: {time: 1700000000100, case: 1}
    }
  ]
};
