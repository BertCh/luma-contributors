// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Descriptors for the second half of the geometry domain: what each binding means to a user.
// The numbers in `requirements` repeat the binding's own input capacities; the spec compares
// them with the binding so the two cannot drift apart.

/** Both capacity limits of one binding input, as blocking requirements. */
function limitCapacity(
  input: string,
  maximumRows: number,
  maximumVertices: number
): KeplerOperationRequirement[] {
  return [
    {
      id: 'maximum-rows',
      severity: 'block',
      input,
      limit: maximumRows,
      remedy: 'Filter to the area of interest or sample the features first'
    },
    {
      id: 'maximum-vertices',
      severity: 'block',
      input,
      limit: maximumVertices,
      remedy: 'Simplify the geometry first'
    }
  ];
}

/** A parameter whose binding default stands whatever the data looks like. */
function describeConstant(
  name: string,
  level: 'basic' | 'advanced' = 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen']
): KeplerOperationParameterDescriptor {
  return {name, defaultPolicy: {from: 'constant'}, level, ...(activeWhen ? {activeWhen} : {})};
}

/**
 * The binding reads coordinates as longitude and latitude unless told they are planar, and it
 * does not detect which they are. Lengths, areas and distances come out null or the result comes
 * out empty when the declaration is wrong (probed: label-point clearance null, Voronoi 0 rows).
 */
function declareCoordinates(input: string): KeplerOperationRequirement {
  return {
    id: 'declared-coordinates',
    severity: 'warn',
    input,
    remedy:
      'Set the coordinate system to match the data: EPSG:4326 for longitude and latitude, planar for projected coordinates. A wrong choice gives empty or null results'
  };
}

/**
 * The coordinate system parameter. It is advanced where another parameter carries the decision
 * and basic where it is the only one, because the metric frame depends on it.
 */
function describeCoordinateSystem(
  level: 'basic' | 'advanced' = 'advanced'
): KeplerOperationParameterDescriptor {
  return describeConstant('coordinateSystem', level);
}

// The code reads a group through String(value), so any column type works as a group.
const GROUP_ROLE_ACCEPTS = ['category', 'text', 'integer', 'boolean'] as const;

/** A missing group value silently becomes the group named by the empty string. */
function describeGroupRequirement(input: string): KeplerOperationRequirement {
  return {
    id: 'no-missing-values',
    severity: 'warn',
    input,
    role: 'group',
    remedy: 'Fill the group column; rows without a value are put in one unnamed group'
  };
}

/** What an S2 level means on the ground: average cell edge, from the sphere area over 6 * 4^level. */
const S2_LEVEL_NOTE =
  'An S2 cell level sets the cell size on the ground and is not chosen from the data: level 3 cells are about 1,150 km across, level 6 about 145 km, level 10 about 9 km and level 12 about 2.3 km. Choose the level for the extent of your data.';

// The binding allows 100,000 dots by default. Target 50% to leave headroom for default
// rounding and estimated column sums. It counts sum(floor(value / dotValue)) over valid
// non-negative polygon values, losing less than one dot per polygon to flooring. Thus
// dotValue = columnSum / 50,000 targets 50,000 dots before per-polygon rounding.
const DOT_DENSITY_TARGET_DOT_COUNT = 50000;

export const GEOMETRY_B_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'geometry-minimum-bounds',
    questions: ['prepare-data'],
    action: 'Enclose features in a box or circle',
    prominence: 3,
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    parameters: [describeConstant('shape', 'basic'), describeCoordinateSystem()],
    requirements: [declareCoordinates('features'), ...limitCapacity('features', 20000, 200000)],
    outputs: [
      {
        tableId: 'minimum-bounds',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ],
    notes: [
      'Each feature needs extent: a feature with a single location gives no row, so a layer of single points returns nothing. Use lines or polygons, or outline groups of points with the concave hull.',
      'A rectangle needs at least three vertices that are not on one line; a circle needs at least two different vertices.'
    ]
  },
  {
    bindingId: 'geometry-label-point',
    questions: ['prepare-data'],
    action: 'Find a label spot inside each polygon',
    prominence: 2,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeConstant('precision'), describeCoordinateSystem('basic')],
    requirements: [declareCoordinates('polygons'), ...limitCapacity('polygons', 20000, 200000)],
    outputs: [
      {
        tableId: 'label-points',
        kind: 'points',
        presentation: {column: 'clearance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ]
  },
  {
    bindingId: 'geometry-maximum-inscribed-circle',
    questions: ['prepare-data'],
    action: 'Find the largest circle in a polygon',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeConstant('precision'), describeCoordinateSystem('basic')],
    requirements: [declareCoordinates('polygons'), ...limitCapacity('polygons', 20000, 200000)],
    outputs: [
      {
        tableId: 'inscribed-circles',
        kind: 'polygons',
        presentation: {column: 'clearance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ]
  },
  {
    bindingId: 'geometry-minimum-clearance',
    // A data-quality measure: how close a feature is to being invalid or collapsing.
    questions: ['prepare-data'],
    action: 'Find narrow gaps within shapes',
    prominence: 3,
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    parameters: [describeCoordinateSystem('basic')],
    requirements: [declareCoordinates('features'), ...limitCapacity('features', 5000, 50000)],
    outputs: [
      {
        tableId: 'minimum-clearance',
        kind: 'lines',
        presentation: {column: 'clearance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ],
    notes: [
      'A feature with more than 5,000 vertices is not measured: it gets status skipped-too-many-vertices and a warning. A feature with fewer than three vertices has no clearance.'
    ]
  },
  {
    bindingId: 'geographic-distribution',
    // Mean centre, standard distance and the ellipse describe where a pattern is centred and how
    // far it spreads, which is the concentration question.
    questions: ['where-concentrated'],
    action: 'Find the centre and spread',
    // A summary of where features are, useful for any layer but not a first analysis of any, and
    // the input accepts every geometry kind, so it must not lead the suggestions.
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'points',
        fields: [
          {
            role: 'group',
            parameter: 'groupProperty',
            label: 'Group by',
            accepts: [...GROUP_ROLE_ACCEPTS],
            measurement: ['nominal'],
            required: false
          },
          {
            role: 'weight',
            parameter: 'weightProperty',
            label: 'Weight',
            accepts: ['number', 'integer'],
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      }
    ],
    parameters: [describeConstant('standardDeviations', 'basic'), describeCoordinateSystem()],
    requirements: [
      declareCoordinates('features'),
      ...limitCapacity('features', 20000, 200000),
      describeGroupRequirement('features'),
      // Features with a negative or non-numeric weight are skipped and counted.
      {
        id: 'non-negative-values',
        severity: 'warn',
        input: 'features',
        role: 'weight',
        remedy: 'Choose a weight column without negative values'
      },
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'features',
        role: 'weight',
        remedy: 'Remove or fill rows without a weight'
      }
    ],
    outputs: [
      {
        tableId: 'distribution-ellipses',
        kind: 'polygons',
        presentation: {column: 'standard_distance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['which-groups-or-regions', 'what-changed']
      },
      {
        tableId: 'distribution-centers',
        kind: 'points',
        presentation: {column: 'kind', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'spatial-hilbert-order',
    questions: ['prepare-data'],
    action: 'Sort features by location',
    prominence: 3,
    inputs: [{name: 'features', kind: 'points', fields: []}],
    parameters: [describeConstant('order')],
    requirements: limitCapacity('features', 20000, 200000),
    outputs: [
      {
        tableId: 'hilbert-order',
        kind: 'points',
        presentation: {column: 'rank', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'spatial-join-bounds-candidates',
    // Growing the left bounds by a distance finds features near each other; the pairs are a
    // first filter for a nearness join, so the primary question is nearness.
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Find features whose boxes overlap',
    prominence: 3,
    inputs: [
      {name: 'left', kind: 'points', fields: []},
      {name: 'right', kind: 'polygons', fields: []}
    ],
    parameters: [describeConstant('distance', 'basic'), describeCoordinateSystem()],
    requirements: [
      declareCoordinates('left'),
      ...limitCapacity('left', 20000, 200000),
      ...limitCapacity('right', 20000, 200000)
    ],
    outputs: [
      {
        tableId: 'join-candidates',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      'At most 200,000 candidate pairs are returned; more than that fails before any result is produced. Use a smaller distance or fewer features.',
      'These are bounding-box overlaps, not exact matches. Refine them with the predicate join.'
    ]
  },
  {
    bindingId: 'spatial-join-predicate',
    // Declare overlap only when disjoint inputs yield no useful result for any parameter value.
    // Every predicate here tests planar intersection or containment, with no distance allowance.
    overlappingInputs: [['left', 'right']],
    // Which points fall in which areas is the first step of counting per area; with `contains`
    // it also answers which areas hold which features.
    questions: ['how-much-per-area'],
    action: 'Match features to areas',
    prominence: 2,
    inputs: [
      {name: 'left', kind: 'points', fields: []},
      {name: 'right', kind: 'polygons', fields: []}
    ],
    parameters: [describeConstant('predicate', 'basic')],
    requirements: [
      ...limitCapacity('left', 5000, 50000),
      ...limitCapacity('right', 5000, 50000),
      // The tests use raw coordinates and ignore the curvature of longitude and latitude.
      {
        id: 'planar-coordinates',
        severity: 'warn',
        input: 'left',
        remedy: 'Project both layers with the local-frame projection first'
      }
    ],
    outputs: [
      {
        tableId: 'spatial-join-pairs',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      'At most 200,000 matching pairs are returned; more than that fails before any result is produced.',
      'The tests use the coordinates as they are and treat longitude and latitude as planar; a point on a polygon edge counts as inside.'
    ]
  },
  {
    bindingId: 'geometry-coverage-validity',
    // The result says which polygons overlap; it prepares a coverage for analysis.
    questions: ['prepare-data'],
    action: 'Check polygons for overlaps',
    prominence: 2,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [],
    // `disjoint-coverage` is not declared: this operation is the check for it.
    requirements: [
      ...limitCapacity('polygons', 3000, 50000),
      // Overlaps are found on the raw coordinates, also for longitude and latitude.
      {
        id: 'planar-coordinates',
        severity: 'warn',
        input: 'polygons',
        remedy: 'Project the polygons with the local-frame projection first'
      }
    ],
    outputs: [
      {
        tableId: 'coverage-overlaps',
        kind: 'pairs',
        interpretation: {summary: 'row-count', cautions: []}
      },
      {
        tableId: 'coverage-features',
        kind: 'polygons',
        kindFrom: 'polygons',
        // The text `valid` / `overlaps` is not the 0/1 validity flag, so it is a plain category.
        presentation: {column: 'status', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'At most 200,000 overlapping pairs are reported; more than that fails before any result is produced.',
      'Only overlaps are found. Gaps and slivers between polygons are not detected.'
    ]
  },
  {
    bindingId: 'voronoi-diagram',
    // Each cell is the area closer to its site than to any other: a catchment of nearest site.
    questions: ['what-is-near-or-reachable'],
    action: 'Draw the area nearest each point',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [describeConstant('padding'), describeCoordinateSystem()],
    requirements: [
      declareCoordinates('points'),
      ...limitCapacity('points', 1500, 1500),
      // A point at an existing position gets no cell.
      {
        id: 'no-duplicate-locations',
        severity: 'warn',
        input: 'points',
        remedy: 'Remove points that share a location'
      }
    ],
    outputs: [
      {
        tableId: 'voronoi-cells',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ]
  },
  {
    bindingId: 'delaunay-triangulation',
    // The triangles and edges say which points are neighbours; nearness is their main use.
    questions: ['what-is-near-or-reachable'],
    action: 'Connect points into triangles',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [describeCoordinateSystem('basic')],
    requirements: [
      declareCoordinates('points'),
      ...limitCapacity('points', 3000, 3000),
      {
        id: 'minimum-rows',
        severity: 'block',
        input: 'points',
        limit: 3,
        remedy: 'Provide at least three points that are not on one line'
      },
      {
        id: 'no-duplicate-locations',
        severity: 'warn',
        input: 'points',
        remedy: 'Remove points that share a location'
      }
    ],
    outputs: [
      {
        tableId: 'delaunay-triangles',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      },
      {
        tableId: 'delaunay-edges',
        kind: 'lines',
        presentation: {column: 'length', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['which-groups-or-regions']
      }
    ]
  },
  {
    bindingId: 'concave-hull',
    // The hull outlines where each group lies; its use is to compare or delimit groups.
    questions: ['which-groups-or-regions'],
    action: 'Outline the extent of groups',
    prominence: 2,
    inputs: [
      {
        name: 'features',
        kind: 'points',
        fields: [
          {
            role: 'group',
            parameter: 'groupProperty',
            label: 'Group by',
            accepts: [...GROUP_ROLE_ACCEPTS],
            measurement: ['nominal'],
            required: false
          }
        ]
      }
    ],
    parameters: [
      describeConstant('ratio', 'basic', {parameter: 'maximumEdgeLength', equals: [0]}),
      // Zero means "use the ratio". A spacing rule could give a plausible edge length, but it would silently turn off the ratio, which is the other way to set concavity, so the zero stays.
      describeConstant('maximumEdgeLength'),
      describeCoordinateSystem()
    ],
    requirements: [
      declareCoordinates('features'),
      ...limitCapacity('features', 5000, 5000),
      describeGroupRequirement('features'),
      {
        id: 'minimum-rows',
        severity: 'block',
        input: 'features',
        limit: 3,
        remedy: 'A hull needs at least three points that are not on one line'
      }
    ],
    outputs: [
      {
        tableId: 'concave-hulls',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area', 'what-changed']
      }
    ],
    notes: [
      'A row without a group value is put in one unnamed group, so a missing value is a group of its own. Groups with fewer than three distinct points give no hull.'
    ]
  },
  {
    bindingId: 'polygon-triangulation',
    questions: ['prepare-data'],
    action: 'Split polygons into triangles',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeCoordinateSystem('basic')],
    requirements: [declareCoordinates('polygons'), ...limitCapacity('polygons', 2000, 50000)],
    outputs: [
      {
        tableId: 'polygon-triangles',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'A polygon part with more than 1,500 vertices is skipped: it gives no rows and a warning. An invalid or self-crossing ring may stop early and keep the triangles found so far.'
    ]
  },
  {
    bindingId: 'polygon-outline',
    questions: ['prepare-data'],
    action: 'Turn polygons into outlines',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeCoordinateSystem('basic')],
    requirements: [declareCoordinates('polygons'), ...limitCapacity('polygons', 20000, 200000)],
    outputs: [
      {
        tableId: 'polygon-outlines',
        kind: 'lines',
        presentation: {column: 'length', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'geometry-shape-generator',
    questions: ['prepare-data'],
    action: 'Draw shapes around points',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            role: 'radius',
            parameter: 'radiusProperty',
            label: 'Radius from a column',
            accepts: ['number', 'integer'],
            // A size in the units of the coordinate system, not a count or a rate.
            measurement: ['amount'],
            required: false
          }
        ]
      }
    ],
    parameters: [
      describeConstant('shape', 'basic'),
      // Shapes that do not overlap their neighbours need a radius below the spacing of the
      // points; both are metres for geographic data and coordinate units for planar data, as
      // the binding's own unit (`crs`) is.
      {
        name: 'radius',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'median-nearest-neighbor-distance',
          factor: 0.4
        },
        level: 'basic'
      },
      describeConstant('rotationDegrees'),
      describeConstant('circleSegments', 'advanced', {parameter: 'shape', equals: ['circle']}),
      describeCoordinateSystem()
    ],
    requirements: [
      declareCoordinates('points'),
      ...limitCapacity('points', 5000, 5000),
      {
        id: 'positive-values',
        severity: 'warn',
        input: 'points',
        role: 'radius',
        remedy: 'Rows with a missing or non-positive radius get no shape'
      }
    ],
    outputs: [
      {
        tableId: 'generated-shapes',
        kind: 'polygons',
        presentation: {column: 'radius', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'The radius column, when chosen, replaces the single radius for every point. Points with a missing or non-positive radius get no shape.'
    ]
  },
  {
    bindingId: 'geometry-dot-density',
    // Dots show a per-area total as a pattern of locations; the input question is the totals.
    questions: ['how-much-per-area', 'where-concentrated'],
    action: 'Draw dots for area totals',
    prominence: 2,
    inputs: [
      {
        name: 'polygons',
        kind: 'polygons',
        fields: [
          {
            role: 'value',
            parameter: 'valueProperty',
            label: 'Total to show as dots',
            accepts: ['number', 'integer'],
            // One dot stands for a fixed amount, so the column must be a total that adds up
            // over the area. A rate or density would draw dots in proportion to the wrong thing.
            measurement: ['count', 'amount'],
            required: true
          }
        ]
      }
    ],
    parameters: [
      {
        name: 'dotValue',
        defaultPolicy: {
          from: 'data',
          input: 'polygons',
          rule: 'column-sum',
          role: 'value',
          factor: 1 / DOT_DENSITY_TARGET_DOT_COUNT
        },
        level: 'basic'
      },
      describeConstant('maximumDots', 'basic'),
      describeConstant('seed'),
      describeCoordinateSystem()
    ],
    requirements: [
      declareCoordinates('polygons'),
      ...limitCapacity('polygons', 5000, 100000),
      {
        id: 'non-negative-values',
        severity: 'block',
        input: 'polygons',
        role: 'value',
        remedy:
          'Use non-negative area totals: remove negative rows or choose a non-negative total column before running. Negative rows get no dots in the binding, so they cannot safely offset the total used to size the dots. The value per dot is derived from the column total to target about 50,000 dots, half the default maximum of 100,000. If you lower the maximum dots or change the value per dot, raise the value per dot until the sum of each polygon total divided by it, rounded down per polygon, fits the cap'
      },
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'polygons',
        role: 'value',
        remedy: 'Rows without a value get no dots'
      }
    ],
    outputs: [
      {
        tableId: 'dots',
        kind: 'points',
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-concentrated']
      }
    ],
    notes: [
      'One dot stands for the value per dot, derived as the selected column total divided by 50,000 to target about half the default maximum of 100,000 dots. Each polygon rounds its dot count down, so small polygons may get no dots. If the column total is unavailable or not positive, the binding fallback is 1. Sampled totals are estimates; check the value per dot and increase it if the actual counts exceed the cap. The derived value per dot assumes the default dot cap; raise it in proportion when the cap is lowered (for example, divide the cap by 10 and multiply the value per dot by 10).',
      'Dots are placed at random inside each polygon, with the same seed giving the same dots.'
    ]
  },
  {
    bindingId: 'geometry-local-frame-projection',
    questions: ['prepare-data'],
    action: 'Convert to or from local metres',
    prominence: 3,
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    // Origin coordinates are degrees, not distances, so no distance rule fits them; the data
    // centre is already used when `useDataCenter` is on.
    parameters: [
      describeConstant('direction', 'basic'),
      describeConstant('useDataCenter', 'basic', {parameter: 'direction', equals: ['forward']}),
      describeConstant('originLongitude'),
      describeConstant('originLatitude')
    ],
    // No coordinate requirement: forward needs longitude and latitude and inverse needs metres,
    // and a requirement cannot depend on the direction.
    requirements: limitCapacity('features', 20000, 200000),
    outputs: [
      {
        tableId: 'local-frame-features',
        kind: 'polygons',
        kindFrom: 'features',
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'Forward turns longitude and latitude into metres east and north of the origin; inverse turns metres back. The inverse always uses the origin below, so enter the origin the forward step used.',
      'The origin longitude and latitude apply when converting back, or when the data centre is not used going forward. They are not hidden by the form because that depends on two parameters at once.',
      'Forward outputs planar metres; inverse outputs longitude and latitude. Declare the output coordinate system accordingly. Properties are not copied.'
    ]
  },
  {
    bindingId: 'line-split-at-points',
    questions: ['prepare-data'],
    action: 'Split lines at points',
    prominence: 3,
    inputs: [
      {name: 'lines', kind: 'lines', fields: []},
      {name: 'splitters', kind: 'points', fields: []}
    ],
    // The tolerance is how far a point may be from a line and still cut it; that is the
    // positional accuracy of the points, not a property of the extent, so it stays constant.
    parameters: [describeConstant('tolerance', 'basic'), describeCoordinateSystem()],
    requirements: [
      declareCoordinates('lines'),
      ...limitCapacity('lines', 20000, 200000),
      ...limitCapacity('splitters', 5000, 5000)
    ],
    outputs: [
      {
        tableId: 'line-pieces',
        kind: 'lines',
        presentation: {column: 'length', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['what-is-near-or-reachable']
      }
    ],
    notes: [
      'The tolerance is how far a point may lie from a line and still cut it. It reflects how accurately the points were located, not the size of the area, so it keeps its fixed default; set it to your positioning accuracy.'
    ]
  },
  {
    bindingId: 'line-clip-by-polygon',
    // Outside and both modes retain disjoint lines, so overlap is not an unconditional requirement.
    questions: ['prepare-data', 'how-much-per-area'],
    action: 'Clip lines to polygons',
    prominence: 3,
    inputs: [
      {name: 'lines', kind: 'lines', fields: []},
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: [describeConstant('keep', 'basic'), describeCoordinateSystem()],
    requirements: [
      declareCoordinates('lines'),
      ...limitCapacity('lines', 5000, 20000),
      ...limitCapacity('polygons', 2000, 20000),
      // Crossings use raw coordinates, also for longitude and latitude.
      {
        id: 'planar-coordinates',
        severity: 'warn',
        input: 'lines',
        remedy: 'Project both layers with the local-frame projection first'
      }
    ],
    outputs: [
      {
        tableId: 'clipped-lines',
        kind: 'lines',
        presentation: {column: 'side', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      'Crossings are found on the coordinates as they are, so longitude and latitude edges are treated as straight lines.'
    ]
  },
  {
    bindingId: 'line-segment-intersections',
    questions: ['prepare-data'],
    action: 'Find where lines cross',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [],
    requirements: [
      ...limitCapacity('lines', 5000, 20000),
      {
        id: 'planar-coordinates',
        severity: 'warn',
        input: 'lines',
        remedy: 'Project the lines with the local-frame projection first'
      }
    ],
    outputs: [
      {
        tableId: 'segment-intersections',
        kind: 'points',
        presentation: {column: 'kind', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'More than 50,000 intersections fails before any result is produced. Touching and crossing are told apart by the kind column.'
    ]
  },
  {
    bindingId: 'line-shared-paths',
    questions: ['prepare-data'],
    action: 'Find stretches shared by lines',
    prominence: 3,
    inputs: [
      {name: 'left', kind: 'lines', fields: []},
      {name: 'right', kind: 'lines', fields: []}
    ],
    // Tolerances are the positional accuracy of the lines and an angle, not the data extent.
    parameters: [
      describeConstant('tolerance', 'basic'),
      describeConstant('angleToleranceDegrees', 'basic'),
      describeCoordinateSystem()
    ],
    requirements: [
      declareCoordinates('left'),
      ...limitCapacity('left', 3000, 20000),
      ...limitCapacity('right', 3000, 20000)
    ],
    outputs: [
      {
        tableId: 'shared-paths',
        kind: 'lines',
        presentation: {column: 'direction', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'At most 50,000 shared segments are returned; more than that fails before any result is produced.',
      'Each row is one pair of nearly parallel segments, not a merged path.'
    ]
  },
  {
    bindingId: 'line-great-circle-arcs',
    // Arcs between paired origins and destinations are a flow map.
    questions: ['how-it-moves'],
    action: 'Draw arcs between places',
    prominence: 2,
    inputs: [
      {name: 'origins', kind: 'points', fields: []},
      {name: 'destinations', kind: 'points', fields: []}
    ],
    parameters: [describeConstant('segments')],
    requirements: [
      ...limitCapacity('origins', 5000, 5000),
      ...limitCapacity('destinations', 5000, 5000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'origins',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 'great-circle-arcs',
        kind: 'lines',
        presentation: {column: 'distance', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ],
    notes: [
      'Origins and destinations are paired by their order in the two layers; extra features on either side are ignored. Coincident and opposite pairs are skipped.'
    ]
  },
  {
    bindingId: 'geometry-offset-curve',
    questions: ['prepare-data'],
    action: 'Draw parallel copies of lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [
      // An offset is a physical distance (a lane, a setback), not a cell size: a rule over the
      // extent gave 25 m for city streets and 286 m for a coastline, which means nothing for it.
      describeConstant('distance', 'basic'),
      describeConstant('mitreLimit'),
      describeCoordinateSystem()
    ],
    requirements: [declareCoordinates('lines'), ...limitCapacity('lines', 20000, 200000)],
    outputs: [
      {
        tableId: 'offset-curves',
        kind: 'lines',
        presentation: {column: 'length', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'Positive distances offset to the left of the line direction and negative to the right. Inner corners can form small loops; remove them with line loop removal.'
    ]
  },
  {
    bindingId: 'line-loop-removal',
    questions: ['prepare-data'],
    action: 'Remove loops from lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [describeCoordinateSystem('basic')],
    requirements: [
      declareCoordinates('lines'),
      ...limitCapacity('lines', 20000, 200000),
      {
        id: 'planar-coordinates',
        severity: 'warn',
        input: 'lines',
        remedy: 'Project the lines with the local-frame projection first'
      }
    ],
    outputs: [
      {
        tableId: 'loop-free-lines',
        kind: 'lines',
        kindFrom: 'lines',
        presentation: {column: 'removed_loops', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'geometry-cleanup',
    questions: ['prepare-data'],
    action: 'Remove duplicate and spike vertices',
    prominence: 2,
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    // The duplicate-vertex tolerance defaults to zero (exact duplicates only); a distance
    // derived from the data would change what the operation does.
    parameters: [
      describeConstant('tolerance'),
      describeConstant('removeSpikes', 'basic'),
      describeCoordinateSystem()
    ],
    requirements: [declareCoordinates('features'), ...limitCapacity('features', 20000, 200000)],
    outputs: [
      {
        tableId: 'cleaned-geometry',
        kind: 'polygons',
        kindFrom: 'features',
        // The values are unchanged, cleaned and dropped, which is not the validity scheme.
        presentation: {column: 'status', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'polygon-precision-reduction',
    questions: ['prepare-data'],
    action: 'Snap polygons to a grid',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    // The grid size is in coordinate units (degrees for longitude and latitude) and the binding
    // has no coordinate system parameter, while the extent rules return metres for geographic
    // data, so the units disagree and the default stays constant.
    parameters: [describeConstant('gridSize', 'basic')],
    requirements: limitCapacity('polygons', 20000, 200000),
    outputs: [
      {
        tableId: 'reduced-polygons',
        kind: 'polygons',
        kindFrom: 'polygons',
        presentation: {column: 'status', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ]
  },
  {
    bindingId: 'geometry-polygonize',
    questions: ['prepare-data'],
    action: 'Build polygons from lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    // The snap tolerance defaults to zero (vertices must already coincide), so it stays constant.
    parameters: [
      describeConstant('nodeCrossings', 'basic'),
      describeConstant('snapTolerance'),
      describeCoordinateSystem()
    ],
    requirements: [declareCoordinates('lines'), ...limitCapacity('lines', 5000, 20000)],
    outputs: [
      {
        tableId: 'polygonized-faces',
        kind: 'polygons',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      'The lines must meet at shared vertices unless crossings are noded, and more than 20,000 segments after noding fails before any result is produced. The faces are not checked for validity.'
    ]
  },
  {
    bindingId: 'spherical-point-in-polygon',
    // Great-circle edges can enclose points beyond the raw vertex extent used by the profiler.
    questions: ['how-much-per-area', 'prepare-data'],
    action: 'Test points in polygons on a sphere',
    prominence: 3,
    inputs: [
      {name: 'points', kind: 'points', fields: []},
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: [describeConstant('boundary', 'basic')],
    requirements: [
      ...limitCapacity('points', 20000, 200000),
      ...limitCapacity('polygons', 2000, 100000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'points',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 'spherical-containment',
        kind: 'points',
        kindFrom: 'points',
        presentation: {column: 'on_boundary', scale: 'categorical'},
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      'Every point is tested against every polygon part, at most 500,000 pairs; more than that fails before any result is produced.',
      'Polygons that cross themselves, have a zero-length edge or span more than a hemisphere contain nothing and are counted.'
    ]
  },
  {
    bindingId: 's2-cell-hierarchy',
    questions: ['prepare-data'],
    action: 'Look up related global grid cells',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('operation', 'basic'),
      describeConstant('level', 'basic'),
      describeConstant('parentLevel', 'advanced', {parameter: 'operation', equals: ['parent']})
    ],
    requirements: [
      ...limitCapacity('points', 20000, 200000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'points',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 's2-hierarchy',
        kind: 'cells',
        // No column to colour by: the relation is the position among the children or neighbours.
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      S2_LEVEL_NOTE,
      'The level here is the level of the cell each point falls in; the parent level is the coarser level asked for when the relation is parent.'
    ]
  },
  {
    bindingId: 's2-cell-measures',
    questions: ['prepare-data'],
    action: 'Measure global grid cells',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [describeConstant('level', 'basic'), describeConstant('radius')],
    requirements: [
      ...limitCapacity('points', 20000, 200000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'points',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 's2-measures',
        kind: 'cells',
        presentation: {column: 'area', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []}
      }
    ],
    notes: [S2_LEVEL_NOTE, 'The level is at most 22: finer cells cannot be checked on the CPU.']
  },
  {
    bindingId: 's2-cap-covering',
    // The cells inside a radius of each point are the cells that are near it.
    questions: ['what-is-near-or-reachable'],
    action: 'Cover a radius with global grid cells',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    // The radius is a fixed metric distance on the sphere; the cell level is bounded to 0-6.
    parameters: [
      describeConstant('radiusMeters', 'basic'),
      describeConstant('level', 'basic'),
      describeConstant('mode'),
      describeConstant('maximumCells')
    ],
    requirements: [
      ...limitCapacity('points', 200, 200000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'points',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 's2-cap-cover',
        kind: 'cells',
        interpretation: {summary: 'row-count', cautions: []}
      }
    ],
    notes: [
      S2_LEVEL_NOTE,
      'Cells are all of one level and the level is at most 6, so a small radius needs a larger level than 6 allows and is covered by few, large cells. Match the level to the radius: a cell should be a fraction of the radius across.'
    ]
  },
  {
    bindingId: 's2-polygon-covering',
    questions: ['prepare-data'],
    action: 'Cover polygons with global grid cells',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [
      describeConstant('mode', 'basic'),
      describeConstant('maximumLevel', 'basic'),
      describeConstant('minimumLevel'),
      describeConstant('maximumCells'),
      describeConstant('maximumVisitedCells')
    ],
    requirements: [
      ...limitCapacity('polygons', 200, 20000),
      {
        id: 'geographic-coordinates',
        severity: 'block',
        input: 'polygons',
        remedy: 'Use longitude and latitude in degrees'
      }
    ],
    outputs: [
      {
        tableId: 's2-polygon-cover',
        kind: 'cells',
        presentation: {column: 'level', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [
      S2_LEVEL_NOTE,
      'The default maximum level 6 is about 145 km: a city polygon is covered by a single cell. Raise the maximum level until the cells are a fraction of the polygon size.'
    ]
  }
];
