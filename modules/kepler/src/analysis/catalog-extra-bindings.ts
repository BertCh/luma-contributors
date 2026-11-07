// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  COORDINATE_SYSTEM_PARAMETER,
  LINE_INPUT,
  MAXIMUM_NETWORK_EDGES,
  NETWORK_BEHAVIOR,
  NETWORK_PARAMETERS,
  SNAP_DISTANCE_PARAMETER,
  getNetworkParameters,
  type CatalogInputSpec,
  type CatalogOperationBinding,
  type CatalogOutputSpec,
  type CatalogParameterSpec
} from './catalog-binding-types';
import type {CatalogCoordinateSystem} from './catalog-geojson';
import {
  runAffineTransform,
  runBufferSelection,
  runGeodesicDestination,
  runGeodesicPairs,
  runGeometryOrientation,
  runGridGenerator,
  runGroupConvexHull,
  runKMeans,
  runLineChunk,
  runLineLocate,
  runLineMerge,
  runLinePrecisionReduction,
  runLineSimplification,
  runLineSmooth,
  runLinearReferencing,
  runNearestFeatureJoin,
  runPointInPolygonJoin,
  runRandomPointsInPolygon,
  runRandomPointsOnLine,
  runVertexSnap
} from './catalog-extra-geometry';
import {
  runNetworkCentrality,
  runNetworkCostMatrix,
  runNetworkLineGraph,
  runNetworkNeighborhood,
  runNetworkStructure
} from './catalog-extra-network';

/** CPU-bound operation before a GPU runner (if any) is attached. */
export type ExtraCpuBinding = Omit<CatalogOperationBinding, 'gpuContributorNames'>;

const CPU_CLEANUP =
  'CPU only; no retained GPU resources. Published results are removed through native Sources.';

const MAXIMUM_FEATURES = 20000;
const MAXIMUM_VERTICES = 200000;

function makeInput(
  name: string,
  label: string,
  accepts: CatalogInputSpec['accepts'],
  maximumFeatures = MAXIMUM_FEATURES,
  maximumVertices = MAXIMUM_VERTICES
): CatalogInputSpec {
  return {name, label, accepts, maximumFeatures, maximumVertices};
}

function makeOutput(
  tableId: string,
  label: string,
  geometry: CatalogOutputSpec['geometry'],
  identityColumns: readonly string[],
  maximumRows = MAXIMUM_FEATURES
): CatalogOutputSpec {
  return {tableId, label, geometry, identityColumns, maximumRows};
}

const POINTS_ANY: CatalogInputSpec['accepts'] = ['point', 'line', 'polygon'];
const LINES_INPUT = makeInput('lines', 'Lines', ['line']);
const POINTS_INPUT = makeInput('points', 'Points', ['point'], 5000, 5000);
const POLYGONS_INPUT = makeInput('polygons', 'Polygons', ['polygon']);

const LINE_PART_BEHAVIOR =
  'Line parts with fewer than two vertices are skipped and counted; MultiLineString parts become separate rows.';

const CRS_DISTANCE_NOTE =
  'Distances are meters for EPSG:4326 (local azimuthal equidistant projection around the data center) and coordinate units for planar.';

function makeDistanceParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum = 0
): CatalogParameterSpec {
  return {name, label, kind: 'number', unit: 'crs', minimum, maximum: 1e9, defaultValue};
}

const SEED_PARAMETER: CatalogParameterSpec = {
  name: 'seed',
  label: 'Random seed',
  kind: 'integer',
  minimum: 0,
  maximum: 4294967295,
  defaultValue: 1
};

const MAXIMUM_CENTRALITY_NODES = 1500;

export const EXTRA_CPU_BINDINGS: readonly ExtraCpuBinding[] = [
  {
    id: 'line-simplification',
    title: 'Line simplification',
    domain: 'geometry',
    summary:
      'Douglas-Peucker simplification per line part; endpoints are kept and vertices are dropped within the tolerance.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineSimplification'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [makeDistanceParameter('tolerance', 'Tolerance', 10), COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      makeOutput('simplified-lines', 'Simplified lines', 'line', [
        'line_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      CRS_DISTANCE_NOTE,
      'Topology between lines is not preserved.'
    ],
    limitation:
      'Planar Douglas-Peucker on locally projected vertices; shared boundaries between features can diverge.',
    run: (inputs, parameters) =>
      runLineSimplification(inputs['lines'], {
        tolerance: parameters['tolerance'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'line-smooth',
    title: 'Line smoothing',
    domain: 'geometry',
    summary:
      'Chaikin corner cutting per line part; endpoints of open lines are kept and closed lines stay closed.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineSmooth'],
    engine: 'cpu-reference',
    inputs: [makeInput('lines', 'Lines', ['line'], MAXIMUM_FEATURES, 20000)],
    parameters: [
      {
        name: 'iterations',
        label: 'Iterations',
        kind: 'integer',
        minimum: 1,
        maximum: 6,
        defaultValue: 2
      }
    ],
    outputs: [
      makeOutput('smoothed-lines', 'Smoothed lines', 'line', [
        'line_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      'Each iteration replaces every segment by points at 1/4 and 3/4; vertex count roughly doubles per iteration.',
      'A line whose first and last vertices are equal and with at least four vertices is treated as closed.'
    ],
    limitation:
      'Smoothed lines may deviate from the source and are not checked for self-intersection.',
    run: (inputs, parameters) =>
      runLineSmooth(inputs['lines'], {iterations: parameters['iterations'] as number})
  },
  {
    id: 'line-precision-reduction',
    title: 'Line precision reduction',
    domain: 'geometry',
    summary:
      'Rounds line coordinates to a number of decimals and removes repeated consecutive vertices.',
    capabilityIds: ['coverage-geometry-refinement'],
    contributorNames: ['GPULinePrecisionReduction'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [
      {
        name: 'decimals',
        label: 'Decimals',
        kind: 'integer',
        minimum: 0,
        maximum: 15,
        defaultValue: 5
      }
    ],
    outputs: [
      makeOutput('reduced-lines', 'Precision-reduced lines', 'line', [
        'line_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      'Parts that collapse below two distinct vertices are dropped and counted.',
      'Rounding is in coordinate units (decimal degrees for EPSG:4326).'
    ],
    limitation: 'Decimal rounding only; no topology repair of lines that cross after rounding.',
    run: (inputs, parameters) =>
      runLinePrecisionReduction(inputs['lines'], {decimals: parameters['decimals'] as number})
  },
  {
    id: 'line-merge',
    title: 'Line merge',
    domain: 'geometry',
    summary: 'Joins line parts that meet end to end at a vertex shared by exactly two line ends.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineMerge'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      makeOutput('merged-lines', 'Merged lines', 'line', ['merged_id', 'member_source_indices'])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      'Endpoints must be exactly equal; there is no snap tolerance. Junctions of three or more ends are not merged.',
      'Merged ids are assigned in input order; member_source_indices lists the contributing features.'
    ],
    limitation: 'Exact endpoint matching only; attributes are not combined.',
    run: (inputs, parameters) =>
      runLineMerge(inputs['lines'], {
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'line-chunk',
    title: 'Line chunking',
    domain: 'geometry',
    summary: 'Splits each line part into consecutive pieces no longer than a maximum length.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineChunk'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [
      makeDistanceParameter('maximumChunkLength', 'Maximum chunk length', 1000, 1e-9),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput(
        'line-chunks',
        'Line chunks',
        'line',
        ['chunk_id', 'source_index', 'source_id', 'part_index', 'chunk_index'],
        50000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      'Cuts fall at multiples of the maximum length; the last chunk holds the remainder.',
      'Lengths are meters for EPSG:4326 (haversine) and coordinate units for planar; cut points interpolate linearly in coordinates.',
      'More than 50,000 chunks throws before rows are produced.'
    ],
    limitation: 'Linear interpolation in coordinates, not along a geodesic.',
    run: (inputs, parameters) =>
      runLineChunk(
        inputs['lines'],
        {
          maximumChunkLength: parameters['maximumChunkLength'] as number,
          coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
        },
        50000
      )
  },
  {
    id: 'line-locate',
    title: 'Locate points on lines',
    domain: 'geometry',
    summary:
      'Projects each point onto the nearest line and reports distance along, fraction, offset, side and the snapped point.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineLocate'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT, POINTS_INPUT],
    parameters: [
      makeDistanceParameter('maximumOffset', 'Maximum offset', 1000),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput(
        'line-locations',
        'Located points',
        'point',
        ['location_id', 'point_index', 'line_index', 'part_index'],
        5000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      CRS_DISTANCE_NOTE,
      'Points beyond the maximum offset keep a row with status beyond-maximum-offset and null measures.',
      'side is left or right of the line direction; on when the offset is zero.',
      'Exhaustive O(points x segments) search.'
    ],
    limitation: 'Nearest line part only; ties resolve to the first part.',
    run: (inputs, parameters) =>
      runLineLocate(inputs['lines'], inputs['points'], {
        maximumOffset: parameters['maximumOffset'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'linear-referencing',
    title: 'Points at measure',
    domain: 'geometry',
    summary: 'Interpolates a point on each line part at given distances or fractions along it.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULinearReferencing'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [
      {
        name: 'measures',
        label: 'Measures',
        kind: 'number-list',
        minimum: 0,
        maximum: 1e9,
        maximumLength: 16,
        defaultValue: [0, 0.5, 1]
      },
      {
        name: 'measureMode',
        label: 'Measure is',
        kind: 'enum',
        options: ['fraction', 'distance'],
        defaultValue: 'fraction'
      },
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput(
        'measure-points',
        'Points at measure',
        'point',
        ['measure_id', 'source_index', 'source_id', 'part_index'],
        20000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      'distance measures are meters for EPSG:4326 and coordinate units for planar; fractions are of part length.',
      'Measures beyond the line extent produce status outside-line and no geometry; nothing is clamped.'
    ],
    limitation: 'Measures are length along the part; M values stored in coordinates are not read.',
    run: (inputs, parameters) =>
      runLinearReferencing(inputs['lines'], {
        measures: parameters['measures'] as readonly number[],
        measureMode: parameters['measureMode'] as 'distance' | 'fraction',
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'group-convex-hull',
    title: 'Convex hull per group',
    domain: 'geometry',
    summary:
      'Monotone-chain convex hull of the points in each group of a property value (or all points).',
    capabilityIds: ['group-and-regional-analysis'],
    contributorNames: ['GPUGroupConvexHull'],
    engine: 'cpu-reference',
    inputs: [makeInput('points', 'Points', ['point'])],
    parameters: [
      {
        name: 'groupProperty',
        label: 'Group property',
        kind: 'string',
        maximumLength: 64,
        defaultValue: ''
      },
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput('group-hulls', 'Group convex hulls', 'polygon', ['hull_id', 'group_value'], 20000)
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'An empty group property puts every point in one group; missing property values form the empty-string group.',
      'Groups with fewer than three distinct points or only collinear points have no hull and are omitted and counted.',
      'The hull is built in coordinate space; area is square meters (spherical) for EPSG:4326 or coordinate units squared for planar.'
    ],
    limitation:
      'Planar hull in longitude/latitude; hulls spanning the antimeridian or poles are not handled.',
    run: (inputs, parameters) =>
      runGroupConvexHull(inputs['points'], {
        groupProperty: parameters['groupProperty'] as string,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'nearest-feature-join',
    title: 'Nearest feature join',
    domain: 'geometry',
    summary:
      'Joins each point to the nearest target feature (point, line or polygon) with the distance.',
    capabilityIds: ['spatial-joins'],
    contributorNames: ['GPUNearestFeatureJoin'],
    engine: 'cpu-reference',
    inputs: [POINTS_INPUT, makeInput('targets', 'Target features', POINTS_ANY, 5000, 50000)],
    parameters: [
      makeDistanceParameter('maximumDistance', 'Maximum distance', 1e6),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput(
        'nearest-feature-join',
        'Nearest feature join',
        'point',
        ['join_id', 'point_index', 'target_index'],
        5000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      CRS_DISTANCE_NOTE,
      'A point inside a polygon has distance 0; otherwise the distance is to the nearest boundary segment.',
      'Points with no target within the maximum distance keep a row with null target columns; ties resolve to the first target.',
      'Exhaustive O(points x target vertices) search.'
    ],
    limitation: 'Brute force with local planar distances; no spatial index.',
    run: (inputs, parameters) =>
      runNearestFeatureJoin(inputs['points'], inputs['targets'], {
        maximumDistance: parameters['maximumDistance'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'point-in-polygon-join',
    title: 'Point in polygon join',
    domain: 'geometry',
    summary:
      'Assigns each point the first polygon that contains it and counts all containing polygons.',
    capabilityIds: ['spatial-joins'],
    contributorNames: ['GPUPointInPolygonJoin'],
    engine: 'cpu-reference',
    inputs: [
      makeInput('points', 'Points', ['point'], 5000, 5000),
      makeInput('polygons', 'Polygons', ['polygon'], 5000, 200000)
    ],
    parameters: [],
    outputs: [
      makeOutput(
        'point-in-polygon-join',
        'Point in polygon join',
        'point',
        ['join_id', 'point_index', 'polygon_index'],
        5000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'Containment is planar in coordinate space; holes exclude points and boundary points count as inside.',
      'The lowest polygon index is reported; match_count shows overlaps. Unmatched points keep a row with status outside.'
    ],
    limitation: 'Planar containment; polygons crossing the antimeridian are not handled.',
    run: inputs => runPointInPolygonJoin(inputs['points'], inputs['polygons'])
  },
  {
    id: 'buffer-selection',
    title: 'Select within distance',
    domain: 'geometry',
    summary: 'Selects the features that lie within a distance of any selector point.',
    capabilityIds: ['spatial-joins'],
    contributorNames: ['GPUBufferSelection'],
    engine: 'cpu-reference',
    inputs: [
      makeInput('selectors', 'Selector points', ['point'], 500, 500),
      makeInput('features', 'Features', POINTS_ANY, 5000, 50000)
    ],
    parameters: [makeDistanceParameter('distance', 'Distance', 1000), COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      makeOutput(
        'buffer-selection',
        'Selected features',
        'none',
        ['source_index', 'source_id', 'nearest_selector_index'],
        5000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      CRS_DISTANCE_NOTE,
      'Selection is exact distance to the geometry (no polygonal buffer is built); distance equal to the limit is selected.',
      'Selected features are returned with their original geometry; the others are not listed.'
    ],
    limitation: 'Distance test only; no buffer polygons are produced.',
    run: (inputs, parameters) =>
      runBufferSelection(inputs['selectors'], inputs['features'], {
        distance: parameters['distance'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'geodesic-destination',
    title: 'Destination point',
    domain: 'geometry',
    summary:
      'Moves each point by a bearing and distance along a great circle (EPSG:4326) or a straight line (planar).',
    capabilityIds: ['geodesic-operations'],
    contributorNames: ['GPUGeodesicDestination'],
    engine: 'cpu-reference',
    inputs: [makeInput('points', 'Points', ['point'])],
    parameters: [
      {
        name: 'bearing',
        label: 'Bearing (degrees clockwise from north)',
        kind: 'number',
        unit: 'degrees',
        minimum: -360,
        maximum: 360,
        defaultValue: 90
      },
      makeDistanceParameter('distance', 'Distance', 1000),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput('geodesic-destinations', 'Destination points', 'point', [
        'destination_id',
        'source_index',
        'source_id'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'EPSG:4326 uses a spherical earth (mean radius 6371008.8 m); longitudes are normalized to [-180, 180].',
      'Planar moves by distance x (sin bearing, cos bearing) in coordinate units.'
    ],
    limitation: 'Spherical, not ellipsoidal (errors up to about 0.5 percent).',
    run: (inputs, parameters) =>
      runGeodesicDestination(inputs['points'], {
        bearing: parameters['bearing'] as number,
        distance: parameters['distance'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'geodesic-pairs',
    title: 'Point pair distance and bearing',
    domain: 'geometry',
    summary:
      'Distance and initial and final bearing between origin and destination points paired by position.',
    capabilityIds: ['geodesic-operations'],
    contributorNames: ['GPUGeodesicPairs'],
    engine: 'cpu-reference',
    inputs: [
      makeInput('origins', 'Origin points', ['point']),
      makeInput('destinations', 'Destination points', ['point'])
    ],
    parameters: [COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      makeOutput('geodesic-pairs', 'Point pairs', 'line', [
        'pair_id',
        'origin_index',
        'destination_index'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'The nth origin is paired with the nth destination; surplus points are ignored with a diagnostic.',
      'Distance is haversine meters (EPSG:4326) or Euclidean (planar); bearings are degrees clockwise from north.',
      'The output line is a straight segment between the pair, not a great-circle arc.'
    ],
    limitation: 'Spherical earth; the line geometry is not densified along the great circle.',
    run: (inputs, parameters) =>
      runGeodesicPairs(inputs['origins'], inputs['destinations'], {
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'geometry-orientation',
    title: 'Polygon ring orientation',
    domain: 'geometry',
    summary:
      'Rewinds polygon rings so shells use the chosen orientation and holes the opposite one.',
    capabilityIds: ['geometry-editing'],
    contributorNames: ['GPUGeometryOrientation'],
    engine: 'cpu-reference',
    inputs: [POLYGONS_INPUT],
    parameters: [
      {
        name: 'shellOrientation',
        label: 'Shell orientation',
        kind: 'enum',
        options: ['counterclockwise', 'clockwise'],
        defaultValue: 'counterclockwise'
      }
    ],
    outputs: [
      makeOutput('oriented-polygons', 'Oriented polygons', 'polygon', ['source_index', 'source_id'])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'Orientation uses the planar shoelace sign in coordinate space (x east, y north).',
      'Rings with zero area are unchanged and counted; geometry kind (Polygon or MultiPolygon) is preserved.'
    ],
    limitation: 'Planar orientation only; the result is not validated for self-intersection.',
    run: (inputs, parameters) =>
      runGeometryOrientation(inputs['polygons'], {
        shellOrientation: parameters['shellOrientation'] as 'counterclockwise' | 'clockwise'
      })
  },
  {
    id: 'affine-transform',
    title: 'Affine transform',
    domain: 'geometry',
    summary:
      'Scales, rotates and translates all vertices about the origin or the input bounds center.',
    capabilityIds: ['geometry-editing'],
    contributorNames: ['GPUAffineTransform'],
    engine: 'cpu-reference',
    inputs: [makeInput('features', 'Features', POINTS_ANY)],
    parameters: [
      {
        name: 'scaleX',
        label: 'Scale X',
        kind: 'number',
        minimum: -1e6,
        maximum: 1e6,
        defaultValue: 1
      },
      {
        name: 'scaleY',
        label: 'Scale Y',
        kind: 'number',
        minimum: -1e6,
        maximum: 1e6,
        defaultValue: 1
      },
      {
        name: 'rotationDegrees',
        label: 'Rotation (counterclockwise degrees)',
        kind: 'number',
        unit: 'degrees',
        minimum: -360,
        maximum: 360,
        defaultValue: 0
      },
      {
        name: 'translateX',
        label: 'Translate X',
        kind: 'number',
        unit: 'coordinate units',
        minimum: -1e9,
        maximum: 1e9,
        defaultValue: 0
      },
      {
        name: 'translateY',
        label: 'Translate Y',
        kind: 'number',
        unit: 'coordinate units',
        minimum: -1e9,
        maximum: 1e9,
        defaultValue: 0
      },
      {
        name: 'pivot',
        label: 'Pivot',
        kind: 'enum',
        options: ['origin', 'bounds-center'],
        defaultValue: 'bounds-center'
      }
    ],
    outputs: [
      makeOutput('transformed-features', 'Transformed features', 'none', [
        'source_index',
        'source_id'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'Order is scale, then rotate about the pivot, then translate; all values are in coordinate units (degrees for EPSG:4326).',
      'Geometry kinds are preserved; features without geometry are skipped and counted.'
    ],
    limitation:
      'Applies the matrix in coordinate space; on geographic data this is not a metric transform.',
    run: (inputs, parameters) =>
      runAffineTransform(inputs['features'], {
        translateX: parameters['translateX'] as number,
        translateY: parameters['translateY'] as number,
        scaleX: parameters['scaleX'] as number,
        scaleY: parameters['scaleY'] as number,
        rotationDegrees: parameters['rotationDegrees'] as number,
        pivot: parameters['pivot'] as 'origin' | 'bounds-center'
      })
  },
  {
    id: 'vertex-snap',
    title: 'Vertex snap',
    domain: 'geometry',
    summary:
      'Snaps line vertices that lie within a tolerance of an earlier vertex onto that vertex.',
    capabilityIds: ['geometry-editing'],
    contributorNames: ['GPUVertexSnap'],
    engine: 'cpu-reference',
    inputs: [LINES_INPUT],
    parameters: [
      makeDistanceParameter('tolerance', 'Snap tolerance', 1),
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput('snapped-lines', 'Vertex-snapped lines', 'line', [
        'line_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      LINE_PART_BEHAVIOR,
      CRS_DISTANCE_NOTE,
      'The first vertex seen at a location is the anchor; later vertices snap to the nearest anchor within tolerance.',
      'Consecutive vertices that snap together are merged; parts collapsing below two vertices are dropped and counted.'
    ],
    limitation:
      'Order-dependent greedy clustering; vertices are not snapped onto segment interiors.',
    run: (inputs, parameters) =>
      runVertexSnap(inputs['lines'], {
        tolerance: parameters['tolerance'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'grid-generator',
    title: 'Grid generator',
    domain: 'geometry',
    summary: 'Rectangular fishnet polygons covering the bounds of the input extent.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUGridGenerator'],
    engine: 'cpu-reference',
    inputs: [makeInput('extent', 'Extent features', POINTS_ANY)],
    parameters: [
      {
        name: 'cellWidth',
        label: 'Cell width',
        kind: 'number',
        unit: 'coordinate units',
        minimum: 1e-9,
        maximum: 1e9,
        defaultValue: 1
      },
      {
        name: 'cellHeight',
        label: 'Cell height',
        kind: 'number',
        unit: 'coordinate units',
        minimum: 1e-9,
        maximum: 1e9,
        defaultValue: 1
      }
    ],
    outputs: [makeOutput('grid-cells', 'Grid cells', 'polygon', ['cell_id', 'row', 'column'])],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'Cells start at the lower-left of the bounds and are sized in coordinate units (degrees for EPSG:4326).',
      'The last row and column extend past the bounds when the size does not divide the extent.',
      'More than 20,000 cells throws before any row is produced.'
    ],
    limitation:
      'Axis-aligned rectangles in coordinate space; cells are not equal-area on geographic data.',
    run: (inputs, parameters) =>
      runGridGenerator(
        inputs['extent'],
        {
          cellWidth: parameters['cellWidth'] as number,
          cellHeight: parameters['cellHeight'] as number
        },
        MAXIMUM_FEATURES
      )
  },
  {
    id: 'random-points-in-polygon',
    title: 'Random points in polygons',
    domain: 'geometry',
    summary: 'Seeded rejection-sampled points inside each polygon feature, holes excluded.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPURandomPointsInPolygon'],
    engine: 'cpu-reference',
    inputs: [makeInput('polygons', 'Polygons', ['polygon'], 1000, 100000)],
    parameters: [
      {
        name: 'pointsPerPolygon',
        label: 'Points per polygon',
        kind: 'integer',
        minimum: 1,
        maximum: 1000,
        defaultValue: 10
      },
      SEED_PARAMETER
    ],
    outputs: [
      makeOutput(
        'random-points',
        'Random points',
        'point',
        ['point_id', 'source_index', 'source_id', 'sample_index'],
        20000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'The same seed and input always give the same points (mulberry32 stream per feature).',
      'Sampling is uniform in coordinate space over the bounding box, so geographic polygons are not area-uniform on the sphere.',
      'At most 200 attempts per requested point; a shortfall is reported in a diagnostic.',
      'More than 20,000 output rows throws after sampling.'
    ],
    limitation: 'Rejection sampling; very thin polygons may under-fill.',
    run: (inputs, parameters) =>
      runRandomPointsInPolygon(inputs['polygons'], {
        pointsPerPolygon: parameters['pointsPerPolygon'] as number,
        seed: parameters['seed'] as number
      })
  },
  {
    id: 'random-points-on-line',
    title: 'Random points on lines',
    domain: 'geometry',
    summary: 'Seeded points at uniformly random distances along each line feature.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPURandomPointsOnLine'],
    engine: 'cpu-reference',
    inputs: [makeInput('lines', 'Lines', ['line'], 1000, 100000)],
    parameters: [
      {
        name: 'pointsPerLine',
        label: 'Points per line',
        kind: 'integer',
        minimum: 1,
        maximum: 1000,
        defaultValue: 10
      },
      SEED_PARAMETER,
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput(
        'random-points',
        'Random points',
        'point',
        ['point_id', 'source_index', 'source_id', 'sample_index'],
        20000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'The same seed and input always give the same points.',
      'Distance is uniform over the total length of all parts of a feature; zero-length features produce no rows.',
      'More than 20,000 output rows throws after sampling.'
    ],
    limitation: 'Linear interpolation in coordinates between vertices.',
    run: (inputs, parameters) =>
      runRandomPointsOnLine(inputs['lines'], {
        pointsPerLine: parameters['pointsPerLine'] as number,
        seed: parameters['seed'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'k-means',
    title: 'K-means clustering',
    domain: 'geometry',
    summary:
      'Lloyd k-means on points with deterministic farthest-first initialization; assignments and centroids.',
    capabilityIds: ['group-and-regional-analysis'],
    contributorNames: ['GPUKMeans'],
    engine: 'cpu-reference',
    inputs: [makeInput('points', 'Points', ['point'])],
    parameters: [
      {
        name: 'clusterCount',
        label: 'Clusters',
        kind: 'integer',
        minimum: 1,
        maximum: 64,
        defaultValue: 3
      },
      {
        name: 'maximumIterations',
        label: 'Maximum iterations',
        kind: 'integer',
        minimum: 1,
        maximum: 200,
        defaultValue: 50
      },
      COORDINATE_SYSTEM_PARAMETER
    ],
    outputs: [
      makeOutput('kmeans-assignments', 'Cluster assignments', 'point', [
        'point_index',
        'point_source_id'
      ]),
      makeOutput('kmeans-centroids', 'Cluster centroids', 'point', ['cluster'], 64)
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      'Initialization starts at the first point and repeatedly adds the point farthest from existing centroids; results are deterministic.',
      'Distances are Euclidean in local projected meters for EPSG:4326 and coordinate units for planar.',
      'Fewer points than clusters reduce the cluster count with a diagnostic; empty clusters keep their centroid.'
    ],
    limitation:
      'Local optimum from a single deterministic start; no multi-start or silhouette scoring.',
    run: (inputs, parameters) =>
      runKMeans(inputs['points'], {
        clusterCount: parameters['clusterCount'] as number,
        maximumIterations: parameters['maximumIterations'] as number,
        coordinateSystem: parameters['coordinateSystem'] as CatalogCoordinateSystem
      })
  },
  {
    id: 'network-structure',
    title: 'Network structure',
    domain: 'network',
    summary:
      'Node degree, connected component and node kind (dead end, pass-through, junction) of the line network.',
    capabilityIds: ['gpu-network'],
    contributorNames: ['GPUNetworkStructure'],
    engine: 'cpu-reference',
    inputs: [LINE_INPUT],
    parameters: NETWORK_PARAMETERS,
    outputs: [
      makeOutput(
        'network-structure',
        'Network nodes',
        'point',
        ['node_id', 'component'],
        MAXIMUM_NETWORK_EDGES * 2
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Degree counts edge ends at a node regardless of one-way restrictions; self-loops count twice.'
    ],
    limitation: 'Topological summary of the built graph; no betweenness or block analysis.',
    run: (inputs, parameters) =>
      runNetworkStructure(inputs['lines'], getNetworkParameters(parameters), MAXIMUM_NETWORK_EDGES)
  },
  {
    id: 'network-centrality',
    title: 'Network centrality',
    domain: 'network',
    summary:
      'Degree centrality and Wasserman-Faust closeness centrality per node over directed arc costs.',
    capabilityIds: ['gpu-network'],
    contributorNames: ['GPUNetworkCentrality'],
    engine: 'cpu-reference',
    inputs: [{...LINE_INPUT, maximumFeatures: 5000, maximumVertices: 50000}],
    parameters: NETWORK_PARAMETERS,
    outputs: [
      makeOutput(
        'network-centrality',
        'Node centrality',
        'point',
        ['node_id'],
        MAXIMUM_CENTRALITY_NODES
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Closeness runs one shortest-path search per node, so networks over 1,500 nodes throw before any search.',
      'Closeness = (r/(n-1)) x (r/sum of distances) with r reachable nodes; isolated nodes score 0.'
    ],
    limitation: 'Degree and closeness only; bounded to 1,500 nodes; no betweenness.',
    run: (inputs, parameters) =>
      runNetworkCentrality(inputs['lines'], getNetworkParameters(parameters), {
        maximumEdges: MAXIMUM_NETWORK_EDGES,
        maximumNodes: MAXIMUM_CENTRALITY_NODES
      })
  },
  {
    id: 'network-cost-matrix',
    title: 'Network cost matrix',
    domain: 'network',
    summary:
      'Least network cost for every origin and destination pair after snapping each point to the network.',
    capabilityIds: ['gpu-network'],
    contributorNames: ['GPUNetworkCostMatrix'],
    engine: 'cpu-reference',
    inputs: [
      LINE_INPUT,
      makeInput('origins', 'Origin points', ['point'], 100, 100),
      makeInput('destinations', 'Destination points', ['point'], 100, 100)
    ],
    parameters: [...NETWORK_PARAMETERS, SNAP_DISTANCE_PARAMETER],
    outputs: [
      makeOutput(
        'network-cost-matrix',
        'Origin-destination costs',
        'none',
        ['pair_id', 'origin_index', 'destination_index'],
        10000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Statuses: ok, origin-not-snapped, destination-not-snapped, unreachable.',
      'Partial-edge cost is proportional to length along the snapped edge; same-edge pairs use the direct span.'
    ],
    limitation: 'One search per origin; not time-dependent or turn-restricted; no path geometry.',
    run: (inputs, parameters) =>
      runNetworkCostMatrix(
        inputs['lines'],
        inputs['origins'],
        inputs['destinations'],
        {
          ...getNetworkParameters(parameters),
          maximumSnapDistance: parameters['maximumSnapDistance'] as number
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  },
  {
    id: 'network-neighborhood',
    title: 'Network neighborhood',
    domain: 'network',
    summary: 'Network nodes reachable from an origin within a maximum cost, with their least cost.',
    capabilityIds: ['gpu-network'],
    contributorNames: ['GPUNetworkNeighborhood'],
    engine: 'cpu-reference',
    inputs: [LINE_INPUT, makeInput('origin', 'Origin point', ['point'], 1, 1)],
    parameters: [
      ...NETWORK_PARAMETERS,
      SNAP_DISTANCE_PARAMETER,
      {
        name: 'maximumCost',
        label: 'Maximum cost',
        kind: 'number',
        unit: 'cost',
        minimum: 0,
        maximum: 1e9,
        defaultValue: 1000
      }
    ],
    outputs: [
      makeOutput(
        'network-neighborhood',
        'Neighborhood nodes',
        'point',
        ['node_id', 'component'],
        MAXIMUM_NETWORK_EDGES * 2
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Only graph nodes are listed; positions along edges between nodes are not (see the service area binding).',
      'An unsnappable origin returns an error diagnostic and no nodes.'
    ],
    limitation: 'Node set within cost; no edge bands or boundary polygons.',
    run: (inputs, parameters) =>
      runNetworkNeighborhood(
        inputs['lines'],
        inputs['origin'],
        {
          ...getNetworkParameters(parameters),
          maximumSnapDistance: parameters['maximumSnapDistance'] as number,
          maximumCost: parameters['maximumCost'] as number
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  },
  {
    id: 'network-line-graph',
    title: 'Network line graph',
    domain: 'network',
    summary: 'Adjacency between network edges that share a node (the edge-to-edge dual graph).',
    capabilityIds: ['gpu-network'],
    contributorNames: ['GPUNetworkLineGraph'],
    engine: 'cpu-reference',
    inputs: [LINE_INPUT],
    parameters: NETWORK_PARAMETERS,
    outputs: [
      makeOutput(
        'network-line-graph',
        'Edge adjacency',
        'none',
        ['link_id', 'first_edge_id', 'second_edge_id'],
        100000
      )
    ],
    cleanup: CPU_CLEANUP,
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Adjacency is undirected and ignores one-way restrictions; each edge pair is listed once per shared node.',
      'More than 100,000 links throws before rows are produced.'
    ],
    limitation: 'Topological adjacency only; no turn costs or restrictions.',
    run: (inputs, parameters) =>
      runNetworkLineGraph(inputs['lines'], getNetworkParameters(parameters), {
        maximumEdges: MAXIMUM_NETWORK_EDGES,
        maximumRows: 100000
      })
  }
];
