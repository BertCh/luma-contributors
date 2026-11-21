// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Apply policy: IDW search radius, power and minimum count, and kriging model, range and search
// radius are packed runtime parameters in catalog-platform3-b-grids.ts. includeEmptyCells only
// filters publication rows. They are live. Neighbour count also sizes maximumNeighborCount,
// so it rebuilds. KDE's retained graph fixes indexGridSize across live bandwidth changes;
// kernel, grid and coordinate settings
// change shader constants, allocation or prepared inputs and also rebuild. Kriging sill and
// nugget change the normalization of prepared sample values, so they rebuild.

// Descriptors for the three `interpolation` and three `density` catalog bindings. All of their
// grids are built in the local metre frame for EPSG:4326 data (`preparePointGrid` and
// `prepareLineDensity` in `catalog-platform3-b-grids.ts`) and in coordinate units for planar
// data, so a cell size, radius or bandwidth of unit `crs` takes the metre or coordinate-unit
// value the extent and spacing rules return. The grid covers the extent of the input points, so
// no bounds parameter exists to derive.
//
// Grid capacity: IDW, kernel, point and line density publish at most 250,000 cells; kriging at
// most 40,000. Shorter side over 250 gives 62,500 cells for a square extent, so the kriging cell
// size is the rule times 2 (15,625 cells for a square extent). The resolver also raises a cell
// size that would exceed the cap of the binding (`KEPLER_GRID_CELL_CAPS`). The CPU engine also
// refuses cells times samples above 60 million; the GPU engine, used when a device is available,
// does not.
//
// Grids for DISPLAY (IDW, kriging, kernel density, line density) keep the extent over 250: the
// cell only sets the resolution of a smooth surface. A grid that is READ for counts or feeds a
// test (point density, the hot spot lattice in the statistics file) follows point density.

function describeConstant(
  name: string,
  level: 'basic' | 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen']
): KeplerOperationParameterDescriptor {
  return {
    name,
    apply: ['power', 'minimumNeighborCount', 'includeEmptyCells', 'model', 'searchRadius'].includes(
      name
    )
      ? 'live'
      : 'rebuild',
    defaultPolicy: {from: 'constant'},
    level,
    ...(activeWhen ? {activeWhen} : {})
  };
}

/** Cell size of a display grid over the extent of `input`, scaled to stay within the cell cap. */
function describeCellSize(
  input: string,
  factor: number,
  level: 'basic' | 'advanced' = 'basic'
): KeplerOperationParameterDescriptor {
  return {
    name: 'cellSize',
    apply: 'rebuild',
    defaultPolicy: {from: 'data', input, rule: 'extent-shorter-side-over-250', factor},
    level
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

/** The column of a surface-sample interpolation: a measurement at a point. */
function describeSampleValue(label: string) {
  return {
    role: 'value',
    label,
    accepts: ['number', 'integer'],
    // A quantity that exists everywhere (rainfall, elevation, a price per area), measured at a
    // point. A count of events at a point says how many happened there, not what the surface is.
    measurement: ['amount', 'rate'],
    required: true,
    parameter: 'valueProperty'
  } as const;
}

function describeSampleRequirements(
  maximumRows: number,
  extra: KeplerOperationRequirement[] = []
): KeplerOperationRequirement[] {
  return [
    ...describeCapacity('points', maximumRows),
    {
      input: 'points',
      id: 'varying-values',
      severity: 'warn',
      role: 'value',
      remedy: 'A constant column gives a constant surface; choose a column that varies'
    },
    ...extra,
    describeDeclaredCoordinates('points'),
    {
      input: 'points',
      id: 'row-semantics',
      severity: 'warn',
      expects: ['surface-samples'],
      remedy: 'For the locations of events use a density analysis instead'
    }
  ];
}

/** Canonical-bandwidth scales of the compact kernels (`getGPUKernelDensityCanonicalBandwidthScale`). */
const EPANECHNIKOV_BANDWIDTH_SCALE = 192 ** (1 / 6);
const QUARTIC_BANDWIDTH_SCALE = (2304 / 5) ** (1 / 6);

export const INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'idw-grid',
    questions: ['value-where-unmeasured'],
    action: 'Estimate values between samples',
    prominence: 1,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['surface-samples'],
        fields: [describeSampleValue('Measured value')]
      }
    ],
    parameters: [
      describeCellSize('points', 1),
      // Cells with fewer than the minimum samples inside the radius are nodata. A radius of 3
      // times the outlier-resistant nearest-neighbor distance reaches a few samples from a cell
      // between them: every sample (outliers apart) has a neighbor within 1 times that distance,
      // so 3 times covers a sample, its neighbor and the cells between with some margin. The
      // factor 3 is this descriptor's choice, not a published rule (GDAL and Esri state no
      // numeric radius). The rule ignores isolated samples, which then leave nodata cells.
      {
        name: 'searchRadius',
        apply: 'live',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'outlier-resistant-nearest-neighbor-distance',
          factor: 3
        },
        level: 'basic'
      },
      // Power 2 is the default of Esri and GDAL; it is a modelling choice, not a data statistic.
      describeConstant('power', 'basic'),
      describeConstant('neighborCount', 'advanced'),
      describeConstant('minimumNeighborCount', 'advanced'),
      describeConstant('includeEmptyCells', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describeSampleRequirements(20000),
    outputs: [
      {
        tableId: 'idw-cells',
        kind: 'surface',
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['shape-of-the-surface']
      }
    ]
  },
  {
    bindingId: 'kriging-grid',
    questions: ['value-where-unmeasured'],
    action: 'Krige with a stated variogram',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['surface-samples'],
        fields: [describeSampleValue('Measured value')]
      }
    ],
    parameters: [
      describeCellSize('points', 2, 'advanced'),
      describeConstant('model', 'basic'),
      // One third of the extent diagonal, the maximum lag the design uses for variogram
      // estimation. A fitted range is usually below the maximum lag, so this is an upper
      // stand-in; the binding's 1000 m is arbitrary for data of any other size.
      {
        name: 'range',
        apply: 'live',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'extent-diagonal',
          factor: 1 / 3
        },
        level: 'basic'
      },
      // The partial sill is the variance of the values (with the nugget at 0): the sill of a
      // variogram that levels off at the variance of the data. The binding's sill of 1 drew a
      // confident surface with a variance of 0.002 to 0.5 from white noise of variance 100.
      // Still a stand-in, not a fit.
      {
        name: 'sill',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'column-variance',
          role: 'value',
          factor: 1
        },
        level: 'basic'
      },
      describeConstant('nugget', 'basic'),
      describeConstant('neighborCount', 'advanced'),
      describeConstant('searchRadius', 'advanced'),
      describeConstant('includeEmptyCells', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: describeSampleRequirements(2000, [
      {
        input: 'points',
        id: 'no-duplicate-locations',
        severity: 'warn',
        remedy: 'Coincident samples make the kriging system singular; average them first'
      }
    ]),
    outputs: [
      {
        tableId: 'kriging-cells',
        kind: 'surface',
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['shape-of-the-surface']
      }
    ],
    notes: [
      'No variogram is fitted. The surface and the variance come from the model, nugget, sill and range that are set; the preset sill is the variance of the values and the preset range a third of the extent diagonal, neither a fit.',
      'The variance uses the squared units specified by the variogram. Its calibration depends on how well that variogram describes the data.',
      'This differs from the host kriging panel, which fits the variogram to the data.'
    ]
  },
  {
    bindingId: 'areal-interpolation',
    questions: ['prepare-data', 'how-much-per-area'],
    action: 'Move values between zone sets',
    prominence: 2,
    inputs: [
      {
        name: 'sources',
        kind: 'polygons',
        rowSemantics: ['objects'],
        fields: [
          {
            role: 'value',
            label: 'Value on the source zones',
            accepts: ['number', 'integer'],
            // Counts and amounts are extensive (split by area share), rates intensive (averaged
            // by area). The variable type follows the level of the column.
            measurement: ['count', 'amount', 'rate'],
            required: true,
            parameter: 'valueProperty'
          }
        ]
      },
      {name: 'targets', kind: 'polygons', fields: []}
    ],
    overlappingInputs: [['sources', 'targets']],
    parameters: [
      // A rate moved as extensive is summed: two halves with rates 10 and 20 gave 30.
      {
        name: 'variableType',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'field-measurement',
          input: 'sources',
          role: 'value',
          values: {rate: 'intensive', count: 'extensive', amount: 'extensive'}
        },
        level: 'basic'
      },
      describeConstant('samplesPerSide', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('sources', 1000, 100000),
      ...describeCapacity('targets', 1000, 100000),
      {
        input: 'sources',
        id: 'disjoint-coverage',
        severity: 'warn',
        remedy: 'Overlapping source zones count the shared area twice'
      },
      {
        input: 'targets',
        id: 'disjoint-coverage',
        severity: 'warn',
        remedy: 'A sample inside overlapping targets goes to the first one only'
      },
      describeDeclaredCoordinates('sources')
    ],
    outputs: [
      {
        tableId: 'targets',
        kind: 'polygons',
        kindFrom: 'targets',
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'how-related']
      }
    ],
    notes: [
      'Overlaps are estimated from a sample grid inside each source zone, so slivers are approximate. Source area that falls outside every target is lost for extensive values; `coverage` says how much of each target was reached.'
    ]
  },
  {
    bindingId: 'kernel-density-grid',
    questions: ['where-concentrated'],
    action: 'Map where points concentrate',
    prominence: 1,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        rowSemantics: ['events'],
        fields: [
          {
            role: 'weight',
            label: 'Weight (empty counts each point once)',
            accepts: ['number', 'integer'],
            // The density sums weights, so they must be totals; a rate would add up to nothing.
            measurement: ['count', 'amount'],
            required: false,
            parameter: 'weightProperty'
          }
        ]
      }
    ],
    parameters: [
      describeCellSize('points', 1),
      // Silverman's normal-reference rule gives the standard deviation of a Gaussian kernel.
      // Only the `gaussian` kernel (the binding's default) takes its bandwidth as a standard
      // deviation: `createCatalogKernelDensityEvaluator` in `catalog-platform3-b-grids.ts` is exp(-d^2 / (2 h^2)) /
      // (2 pi h^2), truncated at 5 h. The `quartic` and `epanechnikov` kernels take `h` as the
      // support radius (zero beyond d = h); bisquare uses the quartic support radius and canonical factor, so the rule's value undersmooths them; the factor that
      // converts a Gaussian-scale bandwidth to equal smoothing is
      // `getGPUKernelDensityCanonicalBandwidthScale` in the analysis package: epanechnikov
      // 192^(1/6), about 2.40, and quartic (2304/5)^(1/6), about 2.78 (the 2D canonical bandwidth
      // of Silverman 1986). `factorByParameter` applies them when the kernel is switched. The rule
      // also assumes one unimodal cloud and over-smooths clustered data.
      {
        name: 'bandwidth',
        apply: 'live',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'normal-reference-bandwidth-silverman',
          factor: 1,
          factorByParameter: {
            parameter: 'kernel',
            factors: {
              gaussian: 1,
              quartic: QUARTIC_BANDWIDTH_SCALE,
              bisquare: QUARTIC_BANDWIDTH_SCALE,
              epanechnikov: EPANECHNIKOV_BANDWIDTH_SCALE
            }
          }
        },
        level: 'basic'
      },
      describeConstant('kernel', 'basic'),
      // Density falls off near the grid edge unless the padding covers the bandwidth; the rules
      // cannot express a multiple of another parameter, so the padding stays at zero.
      describeConstant('padding', 'advanced'),
      describeConstant('includeEmptyCells', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('points', 20000),
      {
        input: 'points',
        id: 'non-negative-values',
        severity: 'warn',
        role: 'weight',
        remedy: 'Points with a negative weight are excluded'
      },
      describeDeclaredCoordinates('points'),
      {
        input: 'points',
        id: 'row-semantics',
        severity: 'warn',
        expects: ['events'],
        remedy: 'For measured values at places, interpolate them with IDW or kriging instead'
      }
    ],
    outputs: [
      {
        tableId: 'density-cells',
        kind: 'surface',
        presentation: {column: 'density', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant', 'which-groups-or-regions']
      }
    ],
    notes: [
      'No edge correction: density falls off near the border of the grid unless the padding covers the bandwidth.'
    ]
  },
  {
    bindingId: 'point-density-grid',
    questions: ['where-concentrated'],
    action: 'Count points per grid cell',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            role: 'weight',
            label: 'Value to sum or average',
            accepts: ['number', 'integer'],
            // A sum needs a total; a mean of a rate is rarely meaningful.
            measurement: ['count', 'amount'],
            required: {whenParameter: 'statistic', equals: ['sum', 'mean']},
            parameter: 'weightProperty'
          }
        ]
      }
    ],
    parameters: [
      // The counts of this grid are read, so the cell follows point density: twice the mean point
      // spacing holds about four points per cell. Shorter side over 250 gave 0.1 points per cell,
      // a map of zeros and ones. The cell count is the point count over four, far below the cap.
      {
        name: 'cellSize',
        apply: 'rebuild',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'mean-point-spacing',
          factor: 2
        },
        level: 'basic'
      },
      describeConstant('statistic', 'basic'),
      describeConstant('includeEmptyCells', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [...describeCapacity('points', 20000), describeDeclaredCoordinates('points')],
    outputs: [
      {
        tableId: 'density-cells',
        kind: 'surface',
        presentation: {column: 'value', scale: 'sequential'},
        interpretation: {summary: 'count-per-area', cautions: []},
        next: ['where-significant', 'which-groups-or-regions']
      }
    ],
    notes: [
      'Values are per-cell counts, sums or means, not intensity per area, and change with the cell size.'
    ]
  },
  {
    bindingId: 'line-density-grid',
    questions: ['where-concentrated'],
    action: 'Map line length per area',
    prominence: 2,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [
      describeCellSize('lines', 1),
      describeConstant('includeEmptyCells', 'advanced'),
      describeConstant('coordinateSystem', 'advanced')
    ],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [
      {
        tableId: 'line-density-cells',
        kind: 'surface',
        presentation: {column: 'density', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['where-significant']
      }
    ]
  }
];
