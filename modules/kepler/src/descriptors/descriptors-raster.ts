// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Descriptors of the four grid-based bindings that the catalog files under `raster`:
// `raster-join-points-in-polygons` (`catalog-platform4-raster.ts`), `recipe-straight-line-catchments`
// (`catalog-platform5-recipes.ts`), `raster-density-smoothing` and `raster-cell-clusters`
// (`catalog-platform5-raster.ts`). None of them takes a raster: each bins its points onto a square
// lattice in a local metre frame first.
//
// The cell size of all four is in the `crs` unit, which is metres for geographic data and
// coordinate units for planar data, the same units the data-derived rules return. The size is
// derived from the extent, because the fixed default of 10 makes the grid of any city-sized
// dataset larger than the binding allows and the run throws.

/**
 * The cell-count limits of the grids, and the factor that keeps the derived cell size inside
 * them. `extent-shorter-side-over-250` alone gives about 250 cells on the shorter side, so a
 * grid of 250 by 250 is 62,500 cells and every dataset would exceed the 40,000 cells of the
 * smoothing and cluster bindings. The factors make the shorter side about 62 cells (40,000
 * cell limit, up to an aspect ratio of 10) and about 125 cells (250,000 cell limit, up to an
 * aspect ratio of 15).
 */
const SMALL_GRID_CELL_SIZE_FACTOR = 4;
const LARGE_GRID_CELL_SIZE_FACTOR = 2;

function describeParameters(
  names: readonly string[],
  basicNames: readonly string[]
): KeplerOperationParameterDescriptor[] {
  return names.map(name => ({
    name,
    defaultPolicy: {from: 'constant'},
    level: basicNames.includes(name) ? 'basic' : 'advanced'
  }));
}

/** A cell size derived from the extent of the named input. */
function describeCellSize(
  input: string,
  factor: number,
  level: 'basic' | 'advanced' = 'basic'
): KeplerOperationParameterDescriptor {
  return {
    name: 'cellSize',
    defaultPolicy: {from: 'data', input, rule: 'extent-shorter-side-over-250', factor},
    level
  };
}

/**
 * Isolated points inflate the grid: the cell size follows the extent, and two far outliers took a
 * city-sized grid from 530 m to 1,993 m cells in the reviewer's probe, so the city fell in a few
 * cells. `mean-point-spacing` was tried instead: it is also computed from the extent (the side of
 * the square that holds one point on average), so isolated points inflate it too, by the square
 * root of the area. The outlier-resistant nearest-neighbour rule is not inflated, but it knows
 * nothing about the extent, so with outliers the grid it implies exceeds the cell cap and the run
 * throws. The extent rule is kept because it is the one that cannot exceed the cap.
 */
const OUTLIER_NOTE =
  'Isolated points far from the rest inflate the grid, because the cell size follows the extent: two far outliers can turn 500 m cells into 2 km cells and put a whole city in a few cells. Remove outliers or clip the layer to the area of interest, or set the cell size.';

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

const POINT_REMEDY = 'Aggregate a sample or filter the points to the area of interest';

export const RASTER_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'raster-join-points-in-polygons',
    overlappingInputs: [['points', 'polygons']],
    // The CPU engine joins exactly and ignores the cell size; only the GPU engine rasterizes the
    // polygons on that grid, and the cell size there is what the derived default protects. The
    // size is advanced for that reason.
    questions: ['how-much-per-area'],
    action: 'Count points per polygon',
    prominence: 3,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'valueProperty',
            role: 'value',
            label: 'Value to add up',
            accepts: ['number', 'integer'],
            // The join adds the values up per polygon, so it wants counts or amounts.
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      },
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: [
      describeCellSize('polygons', LARGE_GRID_CELL_SIZE_FACTOR, 'advanced'),
      ...describeParameters(['coordinateSystem'], ['coordinateSystem'])
    ],
    requirements: [
      ...makeCapacityRequirements('points', 20000, 200000, POINT_REMEDY),
      ...makeCapacityRequirements('polygons', 20000, 200000, 'Use fewer or simpler polygons'),
      makeDeclaredCoordinates('points'),
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'points',
        role: 'value',
        remedy: 'Fill or drop the rows without a value; they are skipped'
      }
    ],
    outputs: [
      {
        tableId: 'raster-join',
        kind: 'polygons',
        kindFrom: 'polygons',
        presentation: {column: 'count', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: ['counts-where-rates-expected']},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [
      'Without a GPU device the count is exact (a point-in-polygon test per point); the cell size is then not used. On the GPU the polygons are rasterized, and the count is exact except for points in boundary cells, whose number is bounded by boundary_count.',
      'A point that lies in two overlapping polygons counts in each of them, so the polygon counts can add up to more than the number of points; multipleMatchCount in the summary says how many points that is. Such data always runs on the exact CPU join.',
      'Points without a number are skipped when a value column is named, and are then left out of the count too.'
    ]
  },
  {
    bindingId: 'recipe-straight-line-catchments',
    questions: ['what-is-near-or-reachable', 'how-much-per-area'],
    action: 'Find what each facility serves',
    prominence: 2,
    inputs: [
      {name: 'facilities', kind: 'points', fields: []},
      {
        name: 'observations',
        kind: 'points',
        fields: [
          {
            // Without a value column every cell holds 1, so the statistics describe the grid only.
            parameter: 'valueProperty',
            role: 'value',
            label: 'Observation value',
            accepts: ['number', 'integer'],
            required: false
          }
        ]
      }
    ],
    parameters: [
      describeCellSize('observations', LARGE_GRID_CELL_SIZE_FACTOR),
      ...describeParameters(['coordinateSystem'], ['coordinateSystem'])
    ],
    requirements: [
      ...makeCapacityRequirements('facilities', 1000, 1000, 'Use fewer facilities'),
      ...makeCapacityRequirements('observations', 100000, 100000, POINT_REMEDY),
      {
        // At least one facility is needed to allocate cells.
        id: 'minimum-rows',
        severity: 'block',
        input: 'facilities',
        limit: 1,
        remedy: 'Provide at least one facility'
      },
      makeDeclaredCoordinates('facilities'),
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'observations',
        role: 'value',
        remedy: 'Fill the missing values; the cells they fall in count as empty'
      }
    ],
    outputs: [
      {
        tableId: 'recipe-catchments',
        kind: 'points',
        kindFrom: 'facilities',
        // The number of grid cells nearest to each facility is the size of its catchment.
        presentation: {column: 'cell_count', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['how-much-per-area', 'where-significant']
      }
    ],
    notes: [
      OUTLIER_NOTE,
      'The cell size is in metres for longitude and latitude data. Distances are straight lines in a local frame, not travel along streets, and a facility that shares a cell with another gets an empty catchment.'
    ]
  },
  {
    bindingId: 'raster-density-smoothing',
    questions: ['where-concentrated'],
    action: 'Smooth point density',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'valueProperty',
            role: 'value',
            label: 'Value to add up',
            accepts: ['number', 'integer'],
            // Cells hold the sum of the values and the stencil averages those sums.
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      }
    ],
    parameters: [
      describeCellSize('points', SMALL_GRID_CELL_SIZE_FACTOR),
      ...describeParameters(
        ['kernel', 'radius', 'coordinateSystem'],
        ['radius', 'kernel', 'coordinateSystem']
      )
    ],
    requirements: [
      ...makeCapacityRequirements('points', 100000, 100000, POINT_REMEDY),
      makeDeclaredCoordinates('points'),
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'points',
        role: 'value',
        remedy: 'Fill or drop the rows without a value; they are skipped'
      }
    ],
    outputs: [
      {
        tableId: 'raster-smoothed-density',
        kind: 'cells',
        presentation: {column: 'smoothed_value', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['which-groups-or-regions', 'where-significant']
      }
    ],
    notes: [
      OUTLIER_NOTE,
      'The grid is at most 40,000 cells, which the derived size (about 62 cells across the shorter side) keeps for extents up to an aspect ratio of 10. Cells near the border lose mass to the outside.'
    ]
  },
  {
    bindingId: 'raster-cell-clusters',
    questions: ['which-groups-or-regions', 'where-concentrated'],
    action: 'Group busy cells into clusters',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'valueProperty',
            role: 'value',
            label: 'Value to add up',
            accepts: ['number', 'integer'],
            // Only the value sum of each cluster uses the column, and it adds values up.
            measurement: ['count', 'amount'],
            required: false
          }
        ]
      }
    ],
    parameters: [
      describeCellSize('points', SMALL_GRID_CELL_SIZE_FACTOR),
      ...describeParameters(
        ['minimumPoints', 'connectivity', 'coordinateSystem'],
        ['minimumPoints', 'connectivity', 'coordinateSystem']
      )
    ],
    requirements: [
      ...makeCapacityRequirements('points', 100000, 100000, POINT_REMEDY),
      makeDeclaredCoordinates('points'),
      {
        id: 'no-missing-values',
        severity: 'warn',
        input: 'points',
        role: 'value',
        remedy: 'Fill or drop the rows without a value; they are skipped'
      }
    ],
    outputs: [
      {
        tableId: 'raster-components',
        kind: 'points',
        // Points per cluster: a number worth colouring. The id itself has as many values as
        // there are clusters.
        presentation: {column: 'point_count', scale: 'sequential'},
        interpretation: {summary: 'row-count', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [
      OUTLIER_NOTE,
      'The grid is at most 40,000 cells, which the derived size (about 62 cells across the shorter side) keeps for extents up to an aspect ratio of 10. Cells holding fewer than the minimum points are not part of any cluster.'
    ]
  }
];
