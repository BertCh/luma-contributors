// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisPresentationSpec,
  GPUAnalysisQuestion,
  GPUSpatialAnalysisDataKind
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {
  KeplerOperationDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

// Descriptors for the first half of the `geometry` catalog domain. Only one binding here reads a
// feature property by name (`group-convex-hull`, parameter `groupProperty`); every other string
// is absent because these operations work on geometry alone. Capacity numbers repeat the
// binding's own input limits and the descriptor spec checks that they still agree.
//
// Input and output kinds follow `data-kind-vocabulary.ts`: `points`, `lines` and `polygons` are
// features with geometry. `regions` (a label or mask table) and `cells` (H3 or Quadbin cells) are
// not used, because no binding here produces or consumes them. A mixed-geometry input gets the
// kind the binding mainly serves.

/** Sentence for a binding that reads coordinates as degrees unless told they are planar. */
const DECLARE_COORDINATES_REMEDY =
  'Set the coordinate system: projected data in metres must be set to planar, or its values are read as degrees and the results are wrong';

/** A parameter whose binding default stands. */
function describeConstant(
  name: string,
  level: 'basic' | 'advanced',
  activeWhen?: KeplerOperationParameterDescriptor['activeWhen']
): KeplerOperationParameterDescriptor {
  return {name, defaultPolicy: {from: 'constant'}, level, ...(activeWhen ? {activeWhen} : {})};
}

/** The coordinate system parameter, which every binding here that measures distance has. */
function describeCoordinateSystem(): KeplerOperationParameterDescriptor {
  return describeConstant('coordinateSystem', 'advanced');
}

/** Warns that the coordinate system is declared by the user and not read from the data. */
function describeDeclaredCoordinates(input: string): KeplerOperationRequirement {
  return {
    input,
    id: 'declared-coordinates',
    severity: 'warn',
    remedy: DECLARE_COORDINATES_REMEDY
  };
}

/** Warns that the method works on raw coordinates, so only projected data gives equal distances. */
function describePlanarCoordinates(input: string, remedy: string): KeplerOperationRequirement {
  return {input, id: 'planar-coordinates', severity: 'warn', remedy};
}

/** Block requirements for the row (and vertex) capacity of one input. */
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
      remedy: 'Filter the layer to the area of interest or split it into tiles'
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

/** One output table described by what its rows are and the column to colour by. */
function describeOutput(
  tableId: string,
  kind: GPUSpatialAnalysisDataKind,
  presentation: GPUAnalysisPresentationSpec | undefined,
  summary: 'row-count' | 'value-range' | 'cluster-count-and-noise',
  next: readonly GPUAnalysisQuestion[] = [],
  kindFrom?: string
): KeplerOperationOutputDescriptor {
  return {
    tableId,
    kind,
    ...(kindFrom ? {kindFrom} : {}),
    ...(presentation ? {presentation} : {}),
    interpretation: {summary, cautions: []},
    ...(next.length ? {next} : {})
  };
}

const SEQUENTIAL = (column: string): GPUAnalysisPresentationSpec => ({
  column,
  scale: 'sequential'
});
// Colouring by the text `status` of a validity or repair table: three or four values, not an id.
const CATEGORICAL = (column: string): GPUAnalysisPresentationSpec => ({
  column,
  scale: 'categorical'
});

// The k-means table writes the cluster index as an int32 from 0 and never labels noise, which is
// the signed-label encoding of the `cluster-membership` scheme.
const CLUSTER_MEMBERSHIP = (column: string): GPUAnalysisPresentationSpec => ({
  column,
  scale: 'class-scheme',
  classScheme: 'cluster-membership',
  classEncoding: 'catalog-signed-label'
});

// Reserve rounding headroom below the generator's 20,000-cell output cap for aspect ratios <= 5.
const GRID_CELL_SIZE_FACTOR = 4.5;

const NEEDS_GPU_NOTE = 'This operation has no CPU engine: it needs a GPU device.';

/**
 * The two GPU buffer bindings share these. `mitreLimit` only matters for mitre joins. For
 * polygons the cap style is accepted but unused (polygons have no open ends), and the arc
 * segments only shape round joins; a line buffer also rounds its caps, so `quadSegments` there
 * is active for either style and is left without `activeWhen`.
 */
function describeBufferStyleParameters(
  geometry: 'line' | 'polygon'
): KeplerOperationParameterDescriptor[] {
  return [
    describeConstant(
      'quadSegments',
      'advanced',
      geometry === 'polygon' ? {parameter: 'joinStyle', equals: ['round']} : undefined
    ),
    describeConstant('joinStyle', 'advanced'),
    describeConstant('capStyle', 'advanced'),
    describeConstant('mitreLimit', 'advanced', {parameter: 'joinStyle', equals: ['mitre']}),
    describeConstant('dissolve', 'basic'),
    describeCoordinateSystem()
  ];
}

function describeOverlay(
  bindingId: string,
  action: string,
  prominence: 2 | 3,
  next: readonly GPUAnalysisQuestion[]
): KeplerOperationDescriptor {
  return {
    bindingId,
    // Declare overlap only when disjoint inputs yield no useful result for any parameter value.
    // Union and difference retain separated areas; intersection cannot.
    overlappingInputs:
      bindingId === 'geometry-overlay-intersection' ? [['left', 'right']] : undefined,
    questions: ['prepare-data'],
    action,
    prominence,
    inputs: [
      {name: 'left', kind: 'polygons', fields: []},
      {name: 'right', kind: 'polygons', fields: []}
    ],
    parameters: [describeConstant('fillRule', 'advanced'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('left', 5000, 50000),
      ...describeCapacity('right', 5000, 50000),
      describeDeclaredCoordinates('left')
    ],
    outputs: [describeOutput('overlay', 'polygons', undefined, 'row-count', next)],
    notes: [
      NEEDS_GPU_NOTE,
      'Each layer is dissolved into one shape first, so result rows do not say which source polygon they came from.'
    ]
  };
}

/** One descriptor per binding id, in the order of the task that wrote them. */
export const GEOMETRY_A_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  {
    bindingId: 'geometry-validity',
    questions: ['prepare-data'],
    action: 'Check polygon validity',
    prominence: 2,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [],
    requirements: [
      ...describeCapacity('polygons', 20000, 200000),
      describePlanarCoordinates(
        'polygons',
        'Checks use raw coordinates, so they are the same for projected and geographic data; polygons that cross the antimeridian are not handled'
      )
    ],
    outputs: [
      describeOutput('validity', 'polygons', CATEGORICAL('status'), 'row-count', [], 'polygons')
    ],
    notes: ['Nothing is repaired or dropped; invalid polygons are reported as they are.']
  },
  {
    bindingId: 'geometry-measures',
    questions: ['prepare-data'],
    action: 'Measure area and length',
    prominence: 2,
    // The features can be points, lines or polygons, and no single measure exists for all three
    // (area is empty for lines and points, length for points), so the table has no presentation.
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    parameters: [describeConstant('coordinateSystem', 'basic')],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      describeDeclaredCoordinates('features')
    ],
    outputs: [describeOutput('measures', 'polygons', undefined, 'row-count', [], 'features')],
    notes: [
      'Area is filled for polygons only; length is the line length or the polygon perimeter and is empty for points.',
      'Areas and lengths are spherical (not ellipsoidal) when the coordinate system is EPSG:4326.'
    ]
  },
  {
    bindingId: 'geometry-point-buffer',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Buffer points',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      {
        name: 'radius',
        defaultPolicy: {
          from: 'data',
          input: 'points',
          rule: 'median-nearest-neighbor-distance',
          factor: 0.5
        },
        level: 'basic'
      },
      describeConstant('segmentsPerQuarter', 'advanced'),
      describeConstant('coordinateSystem', 'basic')
    ],
    requirements: [...describeCapacity('points', 20000), describeDeclaredCoordinates('points')],
    outputs: [describeOutput('buffers', 'polygons', undefined, 'row-count', ['how-much-per-area'])],
    notes: [
      'Overlapping buffers stay separate; nothing is merged.',
      'The buffer is an inscribed polygon, a little smaller than a true circle.'
    ]
  },
  {
    bindingId: 'geometry-line-segmentize',
    questions: ['prepare-data'],
    action: 'Add vertices to long segments',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    // A length derived from the extent could exceed the 500,000-vertex output cap for a layer of
    // many long lines, and the cap depends on total length, which no rule reads: constant.
    parameters: [describeConstant('maximumSegmentLength', 'basic'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [
      describeOutput('segmentized', 'lines', SEQUENTIAL('length'), 'value-range', [], 'lines')
    ],
    notes: [
      'An output of more than 500,000 vertices fails; use a longer maximum segment length for long lines.'
    ]
  },
  {
    bindingId: 'geometry-convex-clip',
    overlappingInputs: [['features', 'clip']],
    questions: ['prepare-data'],
    action: 'Clip to a convex area',
    prominence: 3,
    inputs: [
      {name: 'features', kind: 'polygons', fields: []},
      {name: 'clip', kind: 'polygons', fields: []}
    ],
    parameters: [],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      ...describeCapacity('clip', 1, 1000),
      describePlanarCoordinates(
        'features',
        'Edges are clipped as straight lines in coordinate space, which is not a great-circle path for geographic data over long distances'
      )
    ],
    outputs: [describeOutput('clipped', 'polygons', undefined, 'row-count', [], 'features')],
    notes: [
      'The clip must be a single valid convex polygon; a concave or invalid clip fails.',
      'Features entirely outside the clip produce no row.'
    ]
  },
  {
    bindingId: 'line-simplification',
    questions: ['prepare-data'],
    action: 'Simplify lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [
      {
        name: 'tolerance',
        defaultPolicy: {from: 'data', input: 'lines', rule: 'extent-shorter-side-over-250'},
        level: 'basic'
      },
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [
      describeOutput(
        'simplified-lines',
        'lines',
        SEQUENTIAL('vertices_after'),
        'value-range',
        [],
        'lines'
      )
    ],
    notes: ['Lines are simplified one at a time, so lines that share a boundary can drift apart.']
  },
  {
    bindingId: 'line-smooth',
    questions: ['prepare-data'],
    action: 'Smooth lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [describeConstant('iterations', 'basic')],
    requirements: describeCapacity('lines', 20000),
    outputs: [describeOutput('smoothed-lines', 'lines', undefined, 'row-count', [], 'lines')],
    notes: ['Each iteration roughly doubles the number of vertices.']
  },
  {
    bindingId: 'line-precision-reduction',
    questions: ['prepare-data'],
    action: 'Round line coordinates',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [describeConstant('decimals', 'basic')],
    requirements: describeCapacity('lines', 20000, 200000),
    outputs: [describeOutput('reduced-lines', 'lines', undefined, 'row-count', [], 'lines')],
    notes: ['Decimals count coordinate units: decimal degrees for geographic data, not metres.']
  },
  {
    bindingId: 'line-merge',
    questions: ['prepare-data'],
    action: 'Join connected lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [describeOutput('merged-lines', 'lines', SEQUENTIAL('member_count'), 'row-count')],
    notes: [
      'Ends must match exactly; there is no snap distance, and junctions of three or more lines are not merged.'
    ]
  },
  {
    bindingId: 'line-chunk',
    questions: ['prepare-data'],
    action: 'Split lines into pieces',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    // The piece count is the total line length over this value, which no extent rule sees, and
    // more than 50,000 pieces fail: constant.
    parameters: [describeConstant('maximumChunkLength', 'basic'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [describeOutput('line-chunks', 'lines', SEQUENTIAL('length'), 'row-count')],
    notes: ['More than 50,000 pieces fail; use a longer maximum length for long lines.']
  },
  {
    bindingId: 'line-locate',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Snap points to lines',
    prominence: 3,
    inputs: [
      {name: 'lines', kind: 'lines', fields: []},
      {name: 'points', kind: 'points', fields: []}
    ],
    parameters: [
      {
        // A cut-off for points too far from any line. The diagonal of the lines' extent is the
        // largest distance any point inside that extent can be from a line, so no nearby point
        // is dropped by the default; a tighter cut-off is the user's choice.
        name: 'maximumOffset',
        defaultPolicy: {from: 'data', input: 'lines', rule: 'extent-diagonal'},
        level: 'basic'
      },
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      ...describeCapacity('points', 5000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [
      describeOutput('line-locations', 'points', SEQUENTIAL('offset'), 'value-range', [
        'where-concentrated'
      ])
    ],
    notes: ['The search checks every point against every segment, so large layers are slow.']
  },
  {
    bindingId: 'linear-referencing',
    questions: ['prepare-data'],
    action: 'Place points along lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [
      describeConstant('measures', 'basic'),
      describeConstant('measureMode', 'basic'),
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [describeOutput('measure-points', 'points', SEQUENTIAL('measure'), 'row-count')],
    notes: ['A measure beyond the end of a line gives a row with no point; nothing is clamped.']
  },
  {
    bindingId: 'group-convex-hull',
    questions: ['which-groups-or-regions'],
    action: 'Outline groups of points',
    prominence: 2,
    inputs: [
      {
        name: 'points',
        kind: 'points',
        fields: [
          {
            parameter: 'groupProperty',
            role: 'group',
            label: 'Group',
            // Any value is accepted (it is read as text), so every storage type that can name a
            // group is listed, the same as the other grouping bindings.
            accepts: ['category', 'text', 'integer', 'boolean'],
            measurement: ['nominal', 'ordinal'],
            // An empty value puts every point in one group, so the column is optional.
            required: false
          }
        ]
      }
    ],
    parameters: [describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('points', 20000, 200000),
      {
        input: 'points',
        id: 'minimum-rows',
        severity: 'warn',
        limit: 3,
        remedy:
          'This checks the whole layer; the outline is made per group, and a group with fewer than three distinct points, or points on one line, gets none. Use a coarser group column'
      },
      {
        input: 'points',
        id: 'no-missing-values',
        severity: 'warn',
        role: 'group',
        remedy:
          'Points with no group value are put together in one group with an empty name; fill in the group or remove those rows'
      },
      describePlanarCoordinates(
        'points',
        'The outline is built from raw coordinates, so on geographic data it can differ from the true outline of the points; hulls that cross the antimeridian or a pole are not handled'
      ),
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      describeOutput('group-hulls', 'polygons', SEQUENTIAL('area'), 'row-count', [
        'how-much-per-area'
      ])
    ]
  },
  {
    bindingId: 'nearest-feature-join',
    questions: ['what-is-near-or-reachable'],
    action: 'Find nearest feature',
    prominence: 2,
    inputs: [
      {name: 'points', kind: 'points', fields: []},
      // Points, lines or polygons are accepted; a nearest facility is most often a point.
      {name: 'targets', kind: 'points', fields: []}
    ],
    parameters: [describeConstant('maximumDistance', 'basic'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('points', 5000),
      ...describeCapacity('targets', 5000, 50000),
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      describeOutput(
        'nearest-feature-join',
        'points',
        SEQUENTIAL('distance'),
        'value-range',
        ['how-much-per-area'],
        'points'
      )
    ],
    notes: ['The search checks every point against every target, so large layers are slow.']
  },
  {
    bindingId: 'point-in-polygon-join',
    overlappingInputs: [['points', 'polygons']],
    questions: ['how-much-per-area', 'prepare-data'],
    action: 'Join points to areas',
    prominence: 2,
    inputs: [
      {name: 'points', kind: 'points', fields: []},
      {name: 'polygons', kind: 'polygons', fields: []}
    ],
    parameters: [],
    requirements: [
      ...describeCapacity('points', 5000),
      ...describeCapacity('polygons', 5000, 200000),
      {
        input: 'polygons',
        id: 'disjoint-coverage',
        severity: 'warn',
        remedy:
          'Where areas overlap only the lowest-numbered one is reported; match_count shows the overlap'
      },
      describePlanarCoordinates(
        'polygons',
        'Containment is tested in raw coordinates, which is exact for edges that are straight in the data; polygons that cross the antimeridian are not handled'
      )
    ],
    outputs: [
      describeOutput(
        'point-in-polygon-join',
        'points',
        SEQUENTIAL('match_count'),
        'row-count',
        ['where-significant'],
        'points'
      )
    ]
  },
  {
    bindingId: 'buffer-selection',
    questions: ['what-is-near-or-reachable'],
    action: 'Select features near points',
    prominence: 2,
    inputs: [
      {name: 'selectors', kind: 'points', fields: []},
      {name: 'features', kind: 'points', fields: []}
    ],
    // The radius is the question being asked ("within 500 m of a station"), not a property of
    // either layer's spacing: a distance from the selectors' spacing gave 1,000 m for one
    // selector and 10 km for three selectors 20 km apart. It stays the binding's constant.
    parameters: [describeConstant('distance', 'basic'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('selectors', 500),
      ...describeCapacity('features', 5000, 50000),
      describeDeclaredCoordinates('features')
    ],
    outputs: [
      describeOutput(
        'buffer-selection',
        'points',
        SEQUENTIAL('distance'),
        'value-range',
        ['how-much-per-area'],
        'features'
      )
    ],
    notes: ['Only the selected features are listed, with their original geometry.']
  },
  {
    bindingId: 'geodesic-destination',
    questions: ['prepare-data'],
    action: 'Move points by bearing',
    prominence: 3,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('bearing', 'basic'),
      describeConstant('distance', 'basic'),
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('points', 20000, 200000),
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      describeOutput('geodesic-destinations', 'points', undefined, 'row-count', [], 'points')
    ]
  },
  {
    bindingId: 'geodesic-pairs',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Measure between point pairs',
    prominence: 3,
    inputs: [
      {name: 'origins', kind: 'points', fields: []},
      {name: 'destinations', kind: 'points', fields: []}
    ],
    parameters: [describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('origins', 20000, 200000),
      ...describeCapacity('destinations', 20000, 200000),
      describeDeclaredCoordinates('origins')
    ],
    outputs: [
      describeOutput('geodesic-pairs', 'lines', SEQUENTIAL('distance'), 'value-range', [
        'how-it-moves'
      ])
    ],
    notes: ['The nth origin is paired with the nth destination; surplus points are ignored.']
  },
  {
    bindingId: 'geometry-orientation',
    questions: ['prepare-data'],
    action: 'Set ring direction',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeConstant('shellOrientation', 'basic')],
    requirements: [
      ...describeCapacity('polygons', 20000, 200000),
      describePlanarCoordinates(
        'polygons',
        'Direction is read from raw coordinates with x east and y north; data stored with y pointing south reverses the result'
      )
    ],
    outputs: [
      describeOutput('oriented-polygons', 'polygons', undefined, 'row-count', [], 'polygons')
    ]
  },
  {
    bindingId: 'affine-transform',
    questions: ['prepare-data'],
    action: 'Scale, rotate or shift',
    prominence: 3,
    inputs: [{name: 'features', kind: 'polygons', fields: []}],
    parameters: [
      describeConstant('scaleX', 'basic'),
      describeConstant('scaleY', 'basic'),
      describeConstant('rotationDegrees', 'basic'),
      describeConstant('translateX', 'advanced'),
      describeConstant('translateY', 'advanced'),
      describeConstant('pivot', 'advanced')
    ],
    requirements: [
      ...describeCapacity('features', 20000, 200000),
      describePlanarCoordinates(
        'features',
        'The matrix is applied to raw coordinates, so on geographic data it is not a metric transform'
      )
    ],
    outputs: [
      describeOutput('transformed-features', 'polygons', undefined, 'row-count', [], 'features')
    ]
  },
  {
    bindingId: 'vertex-snap',
    questions: ['prepare-data'],
    action: 'Snap nearby line vertices',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    // No rule reads line vertex spacing, which is what a snap distance should be a small part
    // of; the extent or the anchor spacing would be far too large: constant.
    parameters: [describeConstant('tolerance', 'basic'), describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('lines', 20000, 200000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [
      describeOutput(
        'snapped-lines',
        'lines',
        SEQUENTIAL('vertices_moved'),
        'value-range',
        [],
        'lines'
      )
    ]
  },
  {
    bindingId: 'grid-generator',
    questions: ['prepare-data', 'how-much-per-area'],
    action: 'Make a grid of cells',
    prominence: 3,
    // Any geometry is accepted; a study area is most often a polygon.
    inputs: [{name: 'extent', kind: 'polygons', fields: []}],
    // The cell size is in coordinate units (degrees for geographic data), so the rule that keeps
    // source units is used. A factor of 4.5 reserves room for two-significant-digit rounding:
    // even a 5% downward round gives at most ceil(250 / 4.275) * ceil(1250 / 4.275)
    // = 59 * 293 = 17,287 cells for aspect ratios up to five. More elongated or wrapped
    // extents still need the engine's exact raw-coordinate cell-cap check after rounding.
    parameters: [
      {
        name: 'cellWidth',
        defaultPolicy: {
          from: 'data',
          input: 'extent',
          rule: 'extent-shorter-side-over-250-in-source-units',
          factor: GRID_CELL_SIZE_FACTOR
        },
        level: 'basic'
      },
      {
        name: 'cellHeight',
        defaultPolicy: {
          from: 'data',
          input: 'extent',
          rule: 'extent-shorter-side-over-250-in-source-units',
          factor: GRID_CELL_SIZE_FACTOR
        },
        level: 'basic'
      }
    ],
    requirements: [
      ...describeCapacity('extent', 20000, 200000),
      describePlanarCoordinates(
        'extent',
        'Cells are rectangles in coordinate space, equal in area only for projected data'
      )
    ],
    outputs: [
      describeOutput('grid-cells', 'polygons', undefined, 'row-count', ['how-much-per-area'])
    ],
    notes: [
      'More than 20,000 cells fail; the cell count is ceil(extent width / cell width) times ceil(extent height / cell height), so an elongated extent needs larger cells.',
      'Cell sizes are in coordinate units: degrees for geographic data.'
    ]
  },
  {
    bindingId: 'random-points-in-polygon',
    questions: ['prepare-data'],
    action: 'Scatter random points in areas',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [
      describeConstant('pointsPerPolygon', 'basic'),
      describeConstant('seed', 'advanced')
    ],
    requirements: [
      ...describeCapacity('polygons', 1000, 100000),
      describePlanarCoordinates(
        'polygons',
        'Sampling is uniform in coordinate space, so project geographic areas first for equal-area samples'
      )
    ],
    outputs: [describeOutput('random-points', 'points', undefined, 'row-count')],
    notes: [
      'More than 20,000 points in total fail: the number of polygons times points per polygon must stay within it.',
      'The same seed gives the same points.'
    ]
  },
  {
    bindingId: 'random-points-on-line',
    questions: ['prepare-data'],
    action: 'Scatter random points on lines',
    prominence: 3,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    parameters: [
      describeConstant('pointsPerLine', 'basic'),
      describeConstant('seed', 'advanced'),
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('lines', 1000, 100000),
      describeDeclaredCoordinates('lines')
    ],
    outputs: [describeOutput('random-points', 'points', undefined, 'row-count')],
    notes: [
      'More than 20,000 points in total fail: the number of lines times points per line must stay within it.',
      'The same seed gives the same points.'
    ]
  },
  {
    bindingId: 'k-means',
    questions: ['which-groups-or-regions'],
    action: 'Group points into clusters',
    prominence: 2,
    inputs: [{name: 'points', kind: 'points', fields: []}],
    parameters: [
      describeConstant('clusterCount', 'basic'),
      describeConstant('maximumIterations', 'advanced'),
      describeCoordinateSystem()
    ],
    requirements: [
      ...describeCapacity('points', 20000, 200000),
      describeDeclaredCoordinates('points')
    ],
    outputs: [
      describeOutput(
        'kmeans-assignments',
        'points',
        CLUSTER_MEMBERSHIP('cluster'),
        'cluster-count-and-noise',
        ['how-much-per-area'],
        'points'
      ),
      describeOutput('kmeans-centroids', 'points', SEQUENTIAL('point_count'), 'value-range')
    ],
    notes: [
      'The result is one local optimum from a fixed start; the number of clusters is yours to choose.'
    ]
  },
  {
    bindingId: 'geometry-line-buffer',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Buffer lines',
    prominence: 2,
    inputs: [{name: 'lines', kind: 'lines', fields: []}],
    // The distance stays constant: no rule says how wide a corridor should be.
    parameters: [describeConstant('distance', 'basic'), ...describeBufferStyleParameters('line')],
    requirements: [...describeCapacity('lines', 5000, 20000), describeDeclaredCoordinates('lines')],
    outputs: [describeOutput('buffers', 'polygons', undefined, 'row-count', ['how-much-per-area'])],
    notes: [NEEDS_GPU_NOTE, 'Without dissolve, at most 256 features are buffered, each on its own.']
  },
  {
    bindingId: 'geometry-polygon-buffer',
    questions: ['what-is-near-or-reachable', 'prepare-data'],
    action: 'Grow or shrink areas',
    prominence: 2,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    // A negative distance shrinks, so no positive scale rule can stand for it: constant.
    parameters: [
      describeConstant('distance', 'basic'),
      ...describeBufferStyleParameters('polygon')
    ],
    requirements: [
      ...describeCapacity('polygons', 5000, 50000),
      describeDeclaredCoordinates('polygons')
    ],
    outputs: [describeOutput('buffers', 'polygons', undefined, 'row-count', ['how-much-per-area'])],
    notes: [
      NEEDS_GPU_NOTE,
      'A negative distance shrinks the areas; thin parts disappear.',
      'Without dissolve, at most 256 features are buffered, each on its own. The cap style has no effect on polygons.'
    ]
  },
  describeOverlay('geometry-overlay-intersection', 'Find where areas overlap', 2, [
    'how-much-per-area'
  ]),
  describeOverlay('geometry-overlay-union', 'Merge two area layers', 3, []),
  describeOverlay('geometry-overlay-difference', 'Cut one layer from another', 3, []),
  {
    bindingId: 'geometry-topology-repair',
    questions: ['prepare-data'],
    action: 'Repair invalid polygons',
    prominence: 3,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('polygons', 5000, 50000),
      describeDeclaredCoordinates('polygons')
    ],
    outputs: [
      describeOutput('repaired', 'polygons', CATEGORICAL('status'), 'row-count', [], 'polygons')
    ],
    notes: [
      NEEDS_GPU_NOTE,
      'Each polygon is repaired on its own; shared boundaries between neighbours are not reconciled.'
    ]
  },
  {
    bindingId: 'geometry-shape-descriptors',
    questions: ['prepare-data'],
    action: 'Describe polygon shapes',
    prominence: 2,
    inputs: [{name: 'polygons', kind: 'polygons', fields: []}],
    parameters: [describeCoordinateSystem()],
    requirements: [
      ...describeCapacity('polygons', 20000, 200000),
      describeDeclaredCoordinates('polygons')
    ],
    outputs: [
      describeOutput(
        'shape-descriptors',
        'polygons',
        SEQUENTIAL('compactness'),
        'value-range',
        [],
        'polygons'
      )
    ]
  }
];
