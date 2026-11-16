// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRowSemantics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: threshold only reclassifies computed values in a
// post-pass. Other settings change prepared inputs, allocation, graph construction or compiled
// shader constants (including neighbourhood sizes and permutation loops), so require rebuilding.

// Descriptors for the five bindings of the `clustering` catalog domain. Distances (`epsilon`,
// `maximumDistance`, `extractionDistance`, `cutDistance`, `distanceBand`) are in metres in the
// local frame for EPSG:4326 data and in coordinate units for planar data, the unit the
// `k-distance-knee` and `outlier-resistant-nearest-neighbor-distance` rules return.
//
// Encodings the descriptors rely on (read from the binding code, checked by the spec): the
// `cluster` column of `points` tables is an integer, -1 for noise and 0, 1, 2, ... for clusters
// (`makeClusterOutput` in `catalog-platform3-b-clusters.ts`); SKATER's `region` column is an
// integer 0 to k-1 with no noise, so it is shown as categorical.

/**
 * Margin on the outlier-resistant nearest-neighbor distance (see the statistics file). The
 * inclusive binding boundary does not remove the geographic profile/local-frame metric mismatch.
 */
const DISTANCE_BAND_MARGIN = 1.01;

/**
 * Margin on the knee of the k-distance curve. Fixed planar DBSCAN includes exact-radius pairs,
 * but geographic profile distances and local-frame distances still differ (AX-87 probes). Keep
 * the margin for that mismatch and upward rounding independently: on a regular lattice a knee
 * of 111 m rounded down to 110 m made every point noise.
 */
const KNEE_MARGIN = 1.01;

/**
 * The knee rule reads the sorted distance to the 3rd nearest other point (k = 3, two dimensions:
 * Schubert et al. 2017, k = 2 * dimension - 1), `GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK`.
 */
const KNEE_NEIGHBOR_RANK = 3;

function describeConstant(
  name: string,
  level: 'basic' | 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen']
): KeplerOperationParameterDescriptor {
  return {
    name,
    apply: ['threshold'].includes(name) ? 'live' : 'rebuild',
    defaultPolicy: {from: 'constant'},
    level,
    ...(activeWhen ? {activeWhen} : {})
  };
}

function describeCapacity(
  input: string,
  maximumRows: number,
  maximumVertices?: number
): KeplerOperationRequirement[] {
  const requirements: KeplerOperationRequirement[] = [
    {
      input,
      id: 'maximum-rows',
      severity: 'block',
      limit: maximumRows,
      remedy: 'Filter the layer to the area of interest or aggregate it first'
    }
  ];
  if (maximumVertices !== undefined) {
    requirements.push({
      input,
      id: 'maximum-vertices',
      severity: 'block',
      limit: maximumVertices,
      remedy: 'Simplify the geometry or split the layer into tiles'
    });
  }
  return requirements;
}

function describeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {
    input,
    id: 'declared-coordinates',
    severity: 'warn',
    remedy: 'State whether the coordinates are longitude and latitude or planar'
  };
}

/**
 * Radius used when the k-distance curve has no clear knee (half of 32 realistic datasets): three
 * times the median nearest-neighbor distance. The binding's constant (200 in whatever units)
 * labelled every point noise on weather stations and on points across the antimeridian.
 *
 * Chosen against DBSCAN with 4 minimum points (the binding, checked against scikit-learn 1.6.1,
 * which agreed on the cluster counts to within one border-point assignment) on eight data sets:
 * uniform random 30, 500 and 3,000 points, a regular lattice, three tight clusters in 40 percent
 * background noise, one Gaussian blob of 500 and of 30 points, and points along lines. Factor 2
 * invents structure on uniform data (42 clusters of about ten points and 20 percent noise at 500
 * points, 231 clusters at 3,000) and breaks lines into 58 fragments with 42 percent noise.
 * Factor 3 finds the three planted clusters (4 clusters, 39 percent noise for 40 percent planted),
 * keeps lines together better (52 fragments, 14 percent noise) and on uniform data merges almost
 * everything into one cluster (95 to 100 percent) with at most 2 percent noise: a result that
 * says "no separation", not a finding. Factor 3.5 and above merge the lines and blobs; the mean
 * point spacing times 0.5 to 1.5 was worse (all noise, or one cluster, on lines and lattices).
 * When repeats make the ordinary median zero, the rule uses the median spacing of distinct locations; the original multiplicities still count toward density in the binding.
 */
const KNEE_FALLBACK_FACTOR = 3;

/** A radius taken from the knee of the sorted k-distance curve of the points, times `factor`. */
function describeKneeDistance(
  name: string,
  factor: number,
  level: 'basic' | 'advanced',
  fallbackFactor: number = KNEE_FALLBACK_FACTOR
): KeplerOperationParameterDescriptor {
  return {
    name,
    apply: 'rebuild',
    defaultPolicy: {
      from: 'data',
      input: 'points',
      rule: 'k-distance-knee',
      factor,
      fallback: {rule: 'median-nearest-neighbor-distance', factor: fallbackFactor}
    },
    level
  };
}

/** Rows that stand for places or events; clustering locations of surface samples means nothing. */
const CLUSTERED_ROW_SEMANTICS: readonly GPUAnalysisRowSemantics[] = ['events', 'objects'];

function describePointClusterRequirements(
  maximumRows: number,
  extra: KeplerOperationRequirement[] = []
): KeplerOperationRequirement[] {
  return [
    ...describeCapacity('points', maximumRows),
    ...extra,
    describeDeclaredCoordinates('points'),
    {
      input: 'points',
      id: 'row-semantics',
      severity: 'warn',
      expects: CLUSTERED_ROW_SEMANTICS,
      remedy:
        'Interpolate measured values with IDW or kriging instead of clustering their locations'
    }
  ];
}

export const CLUSTERING_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'density-clustering',
    questions: ['which-groups-or-regions', 'where-concentrated'],
    action: 'Find clusters of points',
    prominence: 1,
    inputs: [{name: 'points', kind: 'points', rowSemantics: CLUSTERED_ROW_SEMANTICS, fields: []}],
    parameters: [
      // The knee of the sorted distance to the 3rd nearest neighbor. The binding's minimum points
      // counts the point itself, so 4 minimum points means 3 other neighbors, the same k. The
      // plot often has no clear knee; the distinct-location spacing fallback then follows scale.
      describeKneeDistance('epsilon', KNEE_MARGIN, 'basic'),
      // 4 for two-dimensional data (Ester et al. 1996); a count, so constant.
      describeConstant('minimumPoints', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describePointClusterRequirements(20000, [
      {
        input: 'points',
        id: 'no-duplicate-locations',
        severity: 'warn',
        remedy: 'Coincident points count toward density; aggregate them or raise the minimum points'
      }
    ]),
    outputs: [
      {
        tableId: 'points',
        kind: 'points',
        kindFrom: 'points',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'cluster-membership',
          classEncoding: 'catalog-signed-label'
        },
        interpretation: {summary: 'cluster-count-and-noise', cautions: []},
        next: ['how-much-per-area', 'how-related']
      },
      {
        tableId: 'clusters',
        kind: 'points',
        presentation: {column: 'size', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'The input cap of 20,000 points is the GPU engine’s. The CPU reference compares every pair and throws “Work capacity exceeded” above 6,324 points (count times count above 40 million: `catalog-platform3-b-clusters.ts` lines 33 and 36 to 40, checked by `computeDbscanLabels` at line 63).',
      'Results with noise outside 1 to 30 percent of the points, or one cluster holding most of them, mean the radius is off: lower it, or use OPTICS or the density cut.'
    ]
  },
  {
    bindingId: 'optics-ordering',
    questions: ['which-groups-or-regions'],
    action: 'Order points by density (OPTICS)',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', rowSemantics: CLUSTERED_ROW_SEMANTICS, fields: []}],
    parameters: [
      // The extraction distance is the DBSCAN-style radius, so it takes the same knee. The
      // maximum distance only bounds the work and must not fall below the extraction distance, so
      // it is derived from the same knee times 3 (this descriptor's choice; the binding's own
      // defaults are 1000 m and 300 m, a ratio of about 3.3). Both stay together or the
      // extraction could exceed the maximum.
      describeKneeDistance('extractionDistance', KNEE_MARGIN, 'basic'),
      describeKneeDistance('maximumDistance', 3, 'basic', 3 * KNEE_FALLBACK_FACTOR),
      {
        name: 'minimumSamples',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'preset',
          value: KNEE_NEIGHBOR_RANK + 1,
          reason: `The knee is read from the distance to the ${KNEE_NEIGHBOR_RANK}rd other point. The binding’s minimum samples counts the point itself, so ${KNEE_NEIGHBOR_RANK + 1} gives a core distance to the ${KNEE_NEIGHBOR_RANK}rd other point, the same k (the binding’s 5 would read k = 4)`
        },
        level: 'basic'
      },
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describePointClusterRequirements(3000),
    outputs: [
      {
        tableId: 'points',
        kind: 'points',
        kindFrom: 'points',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'cluster-membership',
          classEncoding: 'catalog-signed-label'
        },
        interpretation: {summary: 'cluster-count-and-noise', cautions: []},
        next: ['how-much-per-area']
      },
      {
        tableId: 'clusters',
        kind: 'points',
        presentation: {column: 'size', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'local-outlier-factor',
    // Unusual points are the significance question; the threshold is a convention, not a test.
    questions: ['where-significant'],
    action: 'Find isolated points',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', rowSemantics: CLUSTERED_ROW_SEMANTICS, fields: []}],
    parameters: [
      // Counts and a score threshold: constants.
      describeConstant('neighborCount', 'basic'),
      describeConstant('threshold', 'basic'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describePointClusterRequirements(3000, [
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'block',
        // More points than neighbors; 11 for the default of 10 neighbors.
        limit: 11,
        remedy: 'Use more points or fewer neighbors (the minimum is the neighbor count plus one)'
      },
      {
        input: 'points',
        id: 'no-duplicate-locations',
        severity: 'warn',
        remedy: 'Points whose neighbors all coincide get a score of one; aggregate them first'
      }
    ]),
    outputs: [
      {
        tableId: 'outliers',
        kind: 'points',
        kindFrom: 'points',
        presentation: {column: 'lof', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['which-groups-or-regions']
      }
    ],
    notes: [
      'Exploratory: the outlier flag is a score threshold (1.5 by default), a convention, not a significance test.'
    ]
  },
  {
    bindingId: 'hierarchical-density-clustering',
    // A single flat cut of the mutual-reachability tree, not HDBSCAN's stability selection.
    questions: ['which-groups-or-regions'],
    action: 'Cluster by density (one cut)',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', rowSemantics: CLUSTERED_ROW_SEMANTICS, fields: []}],
    parameters: [
      // The cut of the mutual-reachability tree plays the part of DBSCAN's radius at the same
      // minimum samples, so it takes the knee of the k-distance curve.
      describeKneeDistance('cutDistance', KNEE_MARGIN, 'basic'),
      describeConstant('minimumClusterSize', 'basic'),
      {
        name: 'minimumSamples',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'preset',
          value: KNEE_NEIGHBOR_RANK,
          reason: `The knee is read from the distance to the ${KNEE_NEIGHBOR_RANK}rd other point. The binding’s core distance is the distance to its minimum-samples-th OTHER point, so ${KNEE_NEIGHBOR_RANK} is the same k (the binding’s 5 would read k = 5)`
        },
        level: 'basic'
      },
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describePointClusterRequirements(3000),
    outputs: [
      {
        tableId: 'points',
        kind: 'points',
        kindFrom: 'points',
        presentation: {
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'cluster-membership',
          classEncoding: 'catalog-signed-label'
        },
        interpretation: {summary: 'cluster-count-and-noise', cautions: []},
        next: ['how-much-per-area']
      },
      {
        tableId: 'clusters',
        kind: 'points',
        presentation: {column: 'size', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'A single flat cut of the density hierarchy at the cut distance, not HDBSCAN’s stability selection: clusters of different densities are not found together.'
    ]
  },
  {
    bindingId: 'skater-regions',
    questions: ['which-groups-or-regions'],
    action: 'Build regions of similar areas',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'attributes',
            label: 'Attributes that regions should share',
            accepts: ['number', 'integer'],
            // Attributes are standardised and compared by squared distance. A total per polygon
            // mostly measures its size, so polygons take rates only; points may carry amounts.
            measurement: ['rate', 'amount', 'ordinal'],
            measurementByGeometry: {polygon: ['rate'], point: ['rate', 'amount', 'ordinal']},
            required: true,
            multiple: {minimum: 1, maximum: 16},
            parameter: 'properties'
          }
        ]
      }
    ],
    parameters: [
      describeConstant('regionCount', 'basic'),
      // The size floor counts features, not population, so it does not guarantee a minimum total.
      describeConstant('minimumRegionSize', 'basic'),
      describeConstant('neighborhood', 'basic'),
      describeConstant('neighborCount', 'advanced', {parameter: 'neighborhood', equals: ['knn']}),
      {
        name: 'distanceBand',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'features',
          rule: 'outlier-resistant-nearest-neighbor-distance',
          factor: DISTANCE_BAND_MARGIN
        },
        level: 'advanced',
        activeWhen: {parameter: 'neighborhood', equals: ['distance-band']}
      },
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('features', 1500, 200000),
      {
        input: 'features',
        id: 'minimum-rows',
        severity: 'block',
        // At least as many features as regions; 4 for the default of 4 regions.
        limit: 4,
        remedy: 'Use more features or ask for fewer regions'
      },
      {
        input: 'features',
        id: 'every-row-has-a-neighbor',
        severity: 'warn',
        remedy: 'Features without neighbors become one-feature regions; raise the distance band'
      },
      {
        input: 'features',
        id: 'connected-weights',
        severity: 'warn',
        remedy: 'A disconnected neighborhood starts with one region per component'
      },
      describeDeclaredCoordinates('features'),
      {
        input: 'features',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['objects'],
        remedy: 'Aggregate the points to areas or cells first'
      }
    ],
    outputs: [
      {
        tableId: 'regions',
        kind: 'regions',
        kindFrom: 'features',
        // Regions have no noise label, which the cluster legend would show as an empty class.
        presentation: {column: 'region', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-related', 'how-much-per-area']
      }
    ]
  }
];
