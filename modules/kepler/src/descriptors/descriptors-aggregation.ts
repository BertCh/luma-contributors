// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: these settings change prepared geometry, row/vertex allocation, graph construction
// or compiled shader constants. Rebuild keeps those resources consistent with the chosen value.

// Descriptors of the aggregation bindings (`catalog-platform3-b-bindings.ts`,
// `catalog-platform4-dggs.ts`, `catalog-platform5-recipes.ts`): counting and summarising points
// per area, per lattice cell, per discrete-global-grid cell, and the two recipes that do it per
// polygon.
//
// Shared choices:
// - Cell sizes of the lattice bindings are in coordinate units (degrees for longitude and
//   latitude data), so they are derived with the rule that stays in source units. The binding's
//   constant 0.01 gives 1 cm cells for data in metres (every count is 1) and one cell for a city
//   in degrees.
// - The value column of the cell and zonal bindings is optional: without it the binding only
//   counts. Rows without a numeric value are skipped and counted in a diagnostic.

/**
 * Cell size of the lattice bindings: the shorter side of the extent in source units over 250,
 * times 10, that is the shorter side over 25. The grid has about 25 cells across its shorter
 * side, so 3,000 evenly spread points fall about 4 to a cell. The factor is a choice: the rule is
 * based on the extent and not on the count, so fewer than about 600 points give mostly single
 * points per cell, and the cell size must then be set by hand (the notes say so). The occupied
 * cells are at most the points (20,000) and at most about 25 by 25 times the aspect ratio of the
 * extent, far under the 100,000 cells (200,000 rollup rows) the bindings stop at.
 */
const LATTICE_CELL_SIZE_FACTOR = 10;

function describeLatticeCellSize(input: string): KeplerOperationParameterDescriptor {
  return {
    name: 'cellSize',
    apply: 'rebuild',
    defaultPolicy: {
      from: 'data',
      input,
      rule: 'extent-shorter-side-over-250-in-source-units',
      factor: LATTICE_CELL_SIZE_FACTOR
    },
    level: 'basic'
  };
}

const LATTICE_NOTE =
  'The cell size is in the units of the coordinates (degrees for longitude and latitude) and is derived from the extent, about 25 cells across the shorter side; with fewer than about 600 points most cells hold one point, so set a larger size. Cells are a square lattice and are not equal in area on longitude and latitude data.';
const VALUE_DROP_NOTE =
  'When a value column is named, points without a number are dropped from the count as well, not only from the sum.';
const DGGS_RESOLUTION_NOTE =
  'The resolution has no rule that maps an extent to a level, so it keeps the default of 8 and must be chosen for the extent: at the default a Quadbin cell is a web-mercator tile about 156 km wide at the equator (a whole city falls in one row), while an H3 cell at 8 is a hexagon with an edge of about 0.46 km. Each family has its own scale, so the same number means very different cell sizes.';

function describeParameters(
  names: readonly string[],
  basicNames: readonly string[]
): KeplerOperationParameterDescriptor[] {
  return names.map(name => ({
    name,
    apply: 'rebuild',
    defaultPolicy: {from: 'constant'},
    level: basicNames.includes(name) ? 'basic' : 'advanced'
  }));
}

/** Capacity limits of one input, with the binding's own numbers. */
function makeCapacityRequirements(
  input: string,
  maximumRows: number,
  maximumVertices: number,
  remedy: string
): KeplerOperationRequirement[] {
  const requirements: KeplerOperationRequirement[] = [
    {id: 'maximum-rows', severity: 'block', input, limit: maximumRows, remedy}
  ];
  // Point inputs have as many vertices as rows, so one limit says it all.
  if (maximumVertices !== maximumRows) {
    requirements.push({
      id: 'maximum-vertices',
      severity: 'block',
      input,
      limit: maximumVertices,
      remedy
    });
  }
  return requirements;
}

/** Count-only cell maps describe the sampling layout rather than a continuous surface. */
function makeCountSemanticsRequirement(input: string): KeplerOperationRequirement {
  return {
    input,
    id: 'row-semantics',
    severity: 'warn',
    expects: ['events', 'objects'],
    activeWhen: {parameter: 'valueProperty', equals: ['']},
    remedy: 'For surface samples choose a measured-value analysis, or explicitly map valueProperty'
  };
}

/** Rows without a numeric value are skipped (or, for the recipes, left out of the statistic). */
function makeMissingValueRequirement(input: string, remedy: string): KeplerOperationRequirement {
  return {id: 'no-missing-values', severity: 'warn', input, role: 'value', remedy};
}

/** The coordinate system is a parameter the user must set; the binding never infers it. */
function makeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {
    id: 'declared-coordinates',
    severity: 'warn',
    input,
    remedy:
      'Set the coordinate system to EPSG:4326 for longitude and latitude, or planar for projected coordinates'
  };
}

/** The grid-cell families take longitude and latitude only; other values are dropped. */
function makeGeographicCoordinates(input: string): KeplerOperationRequirement {
  return {
    id: 'geographic-coordinates',
    severity: 'block',
    input,
    remedy: 'Reproject the points to longitude and latitude (EPSG:4326)'
  };
}

const POINT_REMEDY = 'Aggregate a sample or filter the points to the area of interest';
/**
 * The optional value column of a count-or-summarise binding. All of count, sum, mean, minimum and
 * maximum are computed together, so no single measurement level is right and none is declared.
 */
const OPTIONAL_VALUE_FIELD: KeplerOperationFieldDescriptor = {
  parameter: 'valueProperty',
  role: 'value',
  label: 'Value to summarise',
  accepts: ['number', 'integer'],
  required: false
};

/**
 * The value column of the lattice compare, read from both layers with one parameter: the same
 * role on both inputs. The contract cannot tie the measurement level to a statistic (a sum wants
 * a count or an amount, a mean also accepts a rate), so the level is left open.
 */
const LATTICE_COMPARE_VALUE_FIELD: KeplerOperationFieldDescriptor = {
  ...OPTIONAL_VALUE_FIELD,
  required: {whenParameter: 'statistic', equals: ['sum', 'mean']}
};

/** The value column of the grid-cell compare: read from both layers, for the sum measure only. */
const DGGS_COMPARE_VALUE_FIELD: KeplerOperationFieldDescriptor = {
  parameter: 'valueProperty',
  role: 'value',
  label: 'Value to add up',
  accepts: ['number', 'integer'],
  // The column is read only for the sum measure, which adds it up per cell.
  measurement: ['count', 'amount'],
  required: {whenParameter: 'measure', equals: ['sum']}
};

const COMPARE_COLUMN_NOTE =
  'The value column is read from both layers, so a column with that name must exist in both; both inputs declare the role, and the curated mapping offers only columns present in both. A cell present in only one period counts as 0 there, for a mean as well, so a mean can show a change that is only an absence.';
const OVERLAP_NOTE =
  'A point that lies in two polygons joins only the polygon that comes first in the layer; the later polygon gets 0 for it, silently.';

const OVERLAP_COUNTED_IN_EACH_NOTE =
  'A point that lies in two overlapping polygons counts in each of them, so the polygon counts can add up to more than the number of points; multipleMatchCount in the summary says how many points that is.';

const MISSING_VALUE_REMEDY = 'Fill or drop the rows without a value; they are skipped';

export const AGGREGATION_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'line-length-per-polygon',
    // Declare overlap only when disjoint inputs yield no useful result for any parameter value.
    // Counts, lengths and zone statistics have no memberships when the inputs are disjoint.
    overlappingInputs: [['lines', 'polygons']],
    questions: ['how-much-per-area'],
    action: 'Measure line length per area',
    prominence: 2,
    inputs: [
      {name: 'polygons', kind: 'polygons', fields: []},
      {name: 'lines', kind: 'lines', fields: []}
    ],
    parameters: describeParameters(['coordinateSystem'], ['coordinateSystem']),
    requirements: [
      ...makeCapacityRequirements('polygons', 2000, 50000, 'Use fewer or simpler polygons'),
      ...makeCapacityRequirements('lines', 5000, 100000, 'Filter or simplify the lines'),
      makeDeclaredCoordinates('polygons'),
      {
        // Overlapping polygons each count the shared length, so the totals double count.
        id: 'disjoint-coverage',
        severity: 'warn',
        input: 'polygons',
        remedy: 'Dissolve or clip overlapping polygons'
      }
    ],
    outputs: [
      {
        tableId: 'polygon-lengths',
        kind: 'polygons',
        kindFrom: 'polygons',
        presentation: {column: 'length_density', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [
      'Overlapping polygons each count the shared length, so totals over polygons can exceed the length of the lines.'
    ]
  },
  {
    bindingId: 'cell-aggregation',
    questions: ['how-much-per-area', 'where-concentrated'],
    action: 'Count points per cell',
    prominence: 1,
    inputs: [{name: 'points', kind: 'points', fields: [OPTIONAL_VALUE_FIELD]}],
    parameters: [describeLatticeCellSize('points')],
    requirements: [
      makeCountSemanticsRequirement('points'),
      ...makeCapacityRequirements('points', 20000, 20000, POINT_REMEDY),
      makeMissingValueRequirement('points', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'cells',
        kind: 'cells',
        presentation: {column: 'count', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    // Leads for point data of any extent because its default size adapts to the extent. The
    // equal-area grid (`dggs-cell-aggregation`) is better for comparing cell values across wide
    // longitude and latitude extents, but its resolution cannot be derived, so it is offered second.
    notes: [
      LATTICE_NOTE,
      'Every point contributes to count; points without a numeric value are excluded only from value statistics and reported in missingValueCount.'
    ]
  },
  {
    bindingId: 'cell-rollup',
    questions: ['how-much-per-area'],
    action: 'Roll cells up to coarser sizes',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: [OPTIONAL_VALUE_FIELD]}],
    parameters: [
      describeLatticeCellSize('points'),
      ...describeParameters(['levels', 'factor'], ['levels'])
    ],
    requirements: [
      makeCountSemanticsRequirement('points'),
      ...makeCapacityRequirements('points', 20000, 20000, POINT_REMEDY),
      makeMissingValueRequirement('points', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'rollup',
        kind: 'cells',
        presentationException: {
          type: 'uniform-style',
          reason:
            'Rollup levels overlap and have different cell areas; a common count scale would compare unlike spatial supports. Show a selected level uniformly or choose a scale after filtering to one level.'
        },
        // Every level is in one table and the levels overlap, so one colour scale over the counts
        // would compare cells of different sizes and a level number is not worth colouring: the
        // host shows one level at a time.
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      LATTICE_NOTE,
      VALUE_DROP_NOTE,
      'All levels are in one table and overlap: show one level at a time.'
    ]
  },
  {
    bindingId: 'cell-compare',
    questions: ['what-changed'],
    action: 'Compare two periods by cell',
    prominence: 2,
    inputs: [
      {name: 'before', kind: 'points', fields: [LATTICE_COMPARE_VALUE_FIELD]},
      {name: 'after', kind: 'points', fields: [LATTICE_COMPARE_VALUE_FIELD]}
    ],
    parameters: [
      describeLatticeCellSize('before'),
      ...describeParameters(['statistic'], ['statistic'])
    ],
    requirements: [
      ...makeCapacityRequirements('before', 20000, 20000, POINT_REMEDY),
      ...makeCapacityRequirements('after', 20000, 20000, POINT_REMEDY),
      makeMissingValueRequirement('before', MISSING_VALUE_REMEDY),
      makeMissingValueRequirement('after', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'comparison',
        kind: 'cells',
        presentation: {column: 'change', scale: 'diverging', center: 0},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'which-groups-or-regions']
      }
    ],
    notes: [
      LATTICE_NOTE,
      'The cell size is derived from the extent of the before layer only; the after layer can reach beyond it.',
      COMPARE_COLUMN_NOTE
    ]
  },
  {
    bindingId: 'zonal-point-statistics',
    overlappingInputs: [['points', 'polygons']],
    questions: ['how-much-per-area'],
    action: 'Summarise points per zone',
    prominence: 2,
    inputs: [
      {name: 'polygons', kind: 'polygons', fields: []},
      {name: 'points', kind: 'points', fields: [OPTIONAL_VALUE_FIELD]}
    ],
    parameters: [],
    requirements: [
      ...makeCapacityRequirements('polygons', 5000, 100000, 'Use fewer or simpler zones'),
      ...makeCapacityRequirements('points', 50000, 50000, POINT_REMEDY),
      makeMissingValueRequirement('points', MISSING_VALUE_REMEDY),
      {
        // A point in overlapping zones counts for each of them.
        id: 'disjoint-coverage',
        severity: 'warn',
        input: 'polygons',
        remedy: 'Dissolve or clip overlapping zones so that no point counts twice'
      }
    ],
    outputs: [
      {
        tableId: 'zones',
        kind: 'polygons',
        presentation: {column: 'count', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: ['counts-where-rates-expected']},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [VALUE_DROP_NOTE]
  },
  {
    bindingId: 'dggs-point-index',
    questions: ['prepare-data'],
    action: 'Index points to grid cells',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: describeParameters(['family', 'resolution'], ['family', 'resolution']),
    requirements: [
      ...makeCapacityRequirements('points', 20000, 200000, POINT_REMEDY),
      makeGeographicCoordinates('points')
    ],
    outputs: [
      {
        // Cell identifiers are labels with no order, so there is nothing to colour by.
        tableId: 'point-cells',
        kind: 'points',
        presentationException: {
          type: 'uniform-style',
          reason:
            'Cell identifiers label the cell containing each point and have no quantitative order; show points uniformly until values are aggregated by cell.'
        },
        interpretation: {summary: 'row-count', cautions: []},
        next: ['how-much-per-area']
      }
    ],
    notes: [DGGS_RESOLUTION_NOTE]
  },
  {
    bindingId: 'dggs-cell-aggregation',
    questions: ['how-much-per-area', 'where-concentrated'],
    action: 'Count points per grid cell',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: [OPTIONAL_VALUE_FIELD]}],
    parameters: describeParameters(['family', 'resolution'], ['family', 'resolution']),
    requirements: [
      makeCountSemanticsRequirement('points'),
      ...makeCapacityRequirements('points', 20000, 200000, POINT_REMEDY),
      makeGeographicCoordinates('points'),
      makeMissingValueRequirement('points', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'cells',
        kind: 'cells',
        presentation: {column: 'count', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    // Prominence 2: for longitude and latitude data over a wide extent this grid should lead,
    // because its cells are comparable in area and the lattice's are not. It cannot lead in
    // general because its resolution cannot be derived from the data (see the note).
    notes: [
      DGGS_RESOLUTION_NOTE,
      'Use this instead of the square lattice when cell values are compared across a wide extent of longitude and latitude; for one city the lattice with its derived size is the easier start.',
      VALUE_DROP_NOTE
    ]
  },
  {
    bindingId: 'dggs-cell-rollup',
    questions: ['how-much-per-area'],
    action: 'Roll grid cells up',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: [OPTIONAL_VALUE_FIELD]}],
    parameters: describeParameters(
      ['family', 'resolution', 'levels', 'levelStep'],
      ['family', 'resolution', 'levels']
    ),
    requirements: [
      makeCountSemanticsRequirement('points'),
      ...makeCapacityRequirements('points', 20000, 200000, POINT_REMEDY),
      makeGeographicCoordinates('points'),
      makeMissingValueRequirement('points', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'rollup',
        kind: 'cells',
        presentationException: {
          type: 'uniform-style',
          reason:
            'Rollup levels overlap and have different cell areas; a common count scale would compare unlike spatial supports. Show a selected level uniformly or choose a scale after filtering to one level.'
        },
        // Levels overlap in one table; see cell-rollup.
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant']
      }
    ],
    notes: [
      DGGS_RESOLUTION_NOTE,
      'All levels are in one table and overlap: show one level at a time. Parent cells are published for Quadbin only.',
      VALUE_DROP_NOTE
    ]
  },
  {
    bindingId: 'dggs-cell-compare',
    questions: ['what-changed'],
    action: 'Compare two periods by grid cell',
    prominence: 2,
    inputs: [
      {name: 'before', kind: 'points', fields: [DGGS_COMPARE_VALUE_FIELD]},
      {name: 'after', kind: 'points', fields: [DGGS_COMPARE_VALUE_FIELD]}
    ],
    parameters: describeParameters(
      ['family', 'resolution', 'measure'],
      ['family', 'resolution', 'measure']
    ),
    requirements: [
      ...makeCapacityRequirements('before', 20000, 200000, POINT_REMEDY),
      ...makeCapacityRequirements('after', 20000, 200000, POINT_REMEDY),
      makeGeographicCoordinates('before'),
      makeGeographicCoordinates('after'),
      makeMissingValueRequirement('before', MISSING_VALUE_REMEDY),
      makeMissingValueRequirement('after', MISSING_VALUE_REMEDY)
    ],
    outputs: [
      {
        tableId: 'cell-comparison',
        kind: 'cells',
        presentation: {column: 'delta', scale: 'diverging', center: 0},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'which-groups-or-regions']
      }
    ],
    notes: [DGGS_RESOLUTION_NOTE, COMPARE_COLUMN_NOTE]
  },
  {
    bindingId: 'recipe-points-in-polygons-choropleth',
    overlappingInputs: [['points', 'polygons']],
    questions: ['how-much-per-area'],
    action: 'Count points per area',
    prominence: 1,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'valueProperty',
            role: 'value',
            label: 'Value to summarise',
            accepts: ['number', 'integer'],
            // The contract cannot tie the measurement level to a statistic (a sum wants a count or an
            // amount, a mean also accepts a rate), so the level is left open.
            required: {whenParameter: 'statistic', equals: ['sum', 'mean', 'minimum', 'maximum']}
          }
        ]
      },
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: describeParameters(
      ['statistic', 'coordinateSystem'],
      ['statistic', 'coordinateSystem']
    ),
    requirements: [
      ...makeCapacityRequirements('points', 100000, 100000, POINT_REMEDY),
      ...makeCapacityRequirements('polygons', 5000, 200000, 'Use fewer or simpler polygons'),
      makeDeclaredCoordinates('points'),
      {
        // These points still count toward point_count but add nothing to the statistic.
        id: 'no-missing-values',
        severity: 'warn',
        input: 'points',
        role: 'value',
        remedy: 'Fill the missing values; the statistic leaves those points out'
      }
    ],
    outputs: [
      {
        tableId: 'recipe-choropleth',
        kind: 'polygons',
        kindFrom: 'polygons',
        // The map reads the selected statistic from value; the sentence reads point_count.
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: ['counts-where-rates-expected']},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [
      'The sentence counts joined points; the map shows the selected statistic.',
      OVERLAP_COUNTED_IN_EACH_NOTE,
      'Points without a value still count toward point_count but add nothing to the statistic.'
    ]
  },
  {
    bindingId: 'recipe-zonal-distribution',
    overlappingInputs: [['points', 'polygons']],
    questions: ['how-much-per-area', 'which-groups-or-regions'],
    action: 'Compare value spread across areas',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'valueProperty',
            role: 'value',
            label: 'Value to describe',
            accepts: ['number', 'integer'],
            // Median, mode and quantiles are defined for counts, amounts, rates and ranks alike.
            measurement: ['count', 'amount', 'rate', 'ordinal'],
            required: true
          }
        ]
      },
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: describeParameters(
      ['probabilities', 'coordinateSystem'],
      ['probabilities', 'coordinateSystem']
    ),
    requirements: [
      ...makeCapacityRequirements('points', 100000, 100000, POINT_REMEDY),
      ...makeCapacityRequirements('polygons', 5000, 200000, 'Use fewer or simpler polygons'),
      makeDeclaredCoordinates('points'),
      {
        // A point in two polygons joins the first; the later one gets 0 (`assignRecipePoints`).
        id: 'disjoint-coverage',
        severity: 'warn',
        input: 'polygons',
        remedy: 'Dissolve or clip overlapping polygons so that no point lies in two'
      },
      {
        // Points without a numeric value are left out of the statistics (valid_count).
        id: 'no-missing-values',
        severity: 'warn',
        input: 'points',
        role: 'value',
        remedy: 'Fill the missing values; the distribution leaves those points out'
      }
    ],
    outputs: [
      {
        tableId: 'recipe-zonal-distribution',
        kind: 'polygons',
        kindFrom: 'polygons',
        presentation: {column: 'median', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [OVERLAP_NOTE]
  }
];
