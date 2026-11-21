// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  COORDINATE_SYSTEM_PARAMETER,
  LINE_INPUT,
  NETWORK_BEHAVIOR,
  NETWORK_PARAMETERS,
  SNAP_DISTANCE_PARAMETER,
  type CatalogOperationBinding,
  type CatalogParameterSpec
} from './catalog-binding-types';
import {
  validateCatalogCoordinateSystem,
  validatePolygonTriangulationCapacity,
  type CatalogCoordinateSystem
} from './catalog-geojson';
import {runGpuHilbertOrder} from './catalog-platform3-a-gpu';
import {
  runBoundsCandidates,
  runCoverageValidity,
  runPolygonize,
  runSpatialPredicateJoin,
  type SpatialPredicate
} from './catalog-platform3-a-join';
import {
  runDotDensity,
  runGeometryCleanup,
  runGreatCircleArcs,
  runLineClipByPolygon,
  runLineLoopRemoval,
  runLineSplit,
  runLocalFrameProjection,
  runOffsetCurve,
  runPolygonOutline,
  runPolygonPrecisionReduction,
  runSegmentIntersections,
  runShapeGenerator,
  runSharedPaths
} from './catalog-platform3-a-lines';
import {
  runGeographicDistribution,
  runHilbertOrder,
  runLabelPoints,
  runMinimumBounds,
  runMinimumClearance,
  runShapeDescriptors
} from './catalog-platform3-a-measure';
import {
  runAlternativeRoutes,
  runEdgeBundling,
  runMapMatching,
  runNetworkAccessibility,
  runNetworkCoarsening,
  runNetworkFlowAggregation,
  runNetworkKFunction,
  runNetworkSubgraphFilter
} from './catalog-platform3-a-network';
import {
  CPU_CLEANUP,
  GPU_OR_CPU_CLEANUP,
  SEED_PARAMETER_SPEC,
  makeInputSpec,
  makeIntegerParameter,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter
} from './catalog-platform3-a-shared';
import {
  runConcaveHull,
  runDelaunayTriangulation,
  runPolygonTriangulation,
  runVoronoiDiagram
} from './catalog-platform3-a-tessellation';

type BindingSpec = Omit<
  CatalogOperationBinding,
  'engine' | 'cleanup' | 'gpuContributorNames' | 'domain'
>;

function defineCpuBinding(spec: BindingSpec, domain: 'geometry' | 'network' = 'geometry') {
  return {
    ...spec,
    run: spec.run
      ? (inputs, values) => {
          validateCatalogCoordinateSystem(spec.id, values['coordinateSystem'], inputs);
          if (spec.id === 'polygon-triangulation')
            validatePolygonTriangulationCapacity(inputs['polygons']);
          return spec.run!(inputs, values);
        }
      : undefined,
    domain,
    engine: 'cpu-reference',
    cleanup: CPU_CLEANUP,
    gpuContributorNames: []
  } as CatalogOperationBinding;
}

const CRS = COORDINATE_SYSTEM_PARAMETER;
const FEATURES_ANY = ['point', 'line', 'polygon'] as const;
const METRIC_NOTE =
  'EPSG:4326 measures in a local meter frame around the data center (azimuthal equidistant, or Lambert equal-area for operations that publish areas); planar uses coordinate units.';
const SKIP_NOTE = 'Features without usable geometry are skipped and counted in a diagnostic.';

const distanceParameter = (name: string, label: string, defaultValue: number, minimum = 0) =>
  makeNumberParameter(name, label, defaultValue, minimum, 1e9, 'crs');

const POLYGONS_INPUT = makeInputSpec('polygons', 'Polygons', ['polygon']);
const LINES_INPUT = makeInputSpec('lines', 'Lines', ['line']);
const POINTS_INPUT = makeInputSpec('points', 'Points', ['point'], 5000, 5000);

const networkParameters = (extra: readonly CatalogParameterSpec[], withSnap = true) => [
  ...NETWORK_PARAMETERS,
  ...(withSnap ? [SNAP_DISTANCE_PARAMETER] : []),
  ...extra
];

const GEOMETRY_BINDINGS: readonly CatalogOperationBinding[] = [
  defineCpuBinding({
    id: 'geometry-shape-descriptors',
    title: 'Polygon shape descriptors',
    summary:
      'Area, perimeter, compactness, convexity, rectangularity and elongation per polygon feature.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPUShapeDescriptors'],
    inputs: [POLYGONS_INPUT],
    parameters: [CRS],
    outputs: [
      makeOutputSpec('shape-descriptors', 'Shape descriptors', 'polygon', [
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Holes subtract from the area and add to the perimeter; compactness is the Polsby-Popper score 4 pi A / P^2.',
      'Convexity is area over the area of the convex hull of the shell vertices; rectangularity is area over the minimum rotated rectangle.',
      SKIP_NOTE
    ],
    limitation: 'Planar metrics in a local frame; no ellipsoidal correction.',
    run: (inputs, values) =>
      runShapeDescriptors(inputs['polygons'], values['coordinateSystem'] as CatalogCoordinateSystem)
  }),
  defineCpuBinding({
    id: 'geometry-minimum-bounds',
    title: 'Minimum bounding rectangle or circle',
    summary: 'Smallest rotated rectangle or enclosing circle of every feature.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPUMinimumBounds'],
    inputs: [makeInputSpec('features', 'Features', FEATURES_ANY)],
    parameters: [
      {
        name: 'shape',
        label: 'Shape',
        kind: 'enum',
        options: ['rectangle', 'circle'],
        defaultValue: 'rectangle'
      },
      CRS
    ],
    outputs: [
      makeOutputSpec('minimum-bounds', 'Minimum bounds', 'polygon', [
        'bounds_id',
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Rectangles come from the convex hull edge directions (rotating calipers); circles from incremental smallest-enclosing-circle construction.',
      'Features with fewer than three non-collinear vertices (rectangle) or a single vertex (circle) produce no row.'
    ],
    limitation: 'Circles are written as 64-gons; no ellipsoidal geometry.',
    run: (inputs, values) =>
      runMinimumBounds(
        inputs['features'],
        values['shape'] as 'rectangle' | 'circle',
        values['coordinateSystem'] as CatalogCoordinateSystem
      )
  }),
  defineCpuBinding({
    id: 'geometry-label-point',
    title: 'Label point (pole of inaccessibility)',
    summary: 'The interior point farthest from the polygon boundary, with its clearance.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPULabelPoint'],
    inputs: [POLYGONS_INPUT],
    parameters: [
      makeNumberParameter('precision', 'Precision (fraction of size)', 0.001, 1e-6, 0.5),
      CRS
    ],
    outputs: [
      makeOutputSpec('label-points', 'Label points', 'point', [
        'label_id',
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Quad-tree refinement stops when no cell can beat the best distance by more than the precision.',
      'MultiPolygon features report the part with the largest clearance.'
    ],
    limitation: 'Planar in a local frame; the search is capped at 20,000 cell visits per part.',
    run: (inputs, values) =>
      runLabelPoints(
        inputs['polygons'],
        false,
        values['precision'] as number,
        values['coordinateSystem'] as CatalogCoordinateSystem
      )
  }),
  defineCpuBinding({
    id: 'geometry-maximum-inscribed-circle',
    title: 'Maximum inscribed circle',
    summary: 'The largest circle inside each polygon, centered at the pole of inaccessibility.',
    capabilityIds: ['coverage-geometry-refinement'],
    contributorNames: ['GPUMaximumInscribedCircle'],
    inputs: [POLYGONS_INPUT],
    parameters: [
      makeNumberParameter('precision', 'Precision (fraction of size)', 0.001, 1e-6, 0.5),
      CRS
    ],
    outputs: [
      makeOutputSpec('inscribed-circles', 'Maximum inscribed circles', 'polygon', [
        'label_id',
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'clearance is the exact radius; the polygon is a 64-gon inscribed in that circle.',
      'MultiPolygon features report the part with the largest circle.'
    ],
    limitation: 'Planar in a local frame; precision-limited center.',
    run: (inputs, values) =>
      runLabelPoints(
        inputs['polygons'],
        true,
        values['precision'] as number,
        values['coordinateSystem'] as CatalogCoordinateSystem
      )
  }),
  defineCpuBinding({
    id: 'geometry-minimum-clearance',
    title: 'Minimum clearance',
    summary: 'Smallest vertex-to-segment separation per feature with the line that realizes it.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPUMinimumClearance'],
    inputs: [makeInputSpec('features', 'Lines or polygons', ['line', 'polygon'], 5000, 50000)],
    parameters: [CRS],
    outputs: [
      makeOutputSpec(
        'minimum-clearance',
        'Minimum clearance',
        'line',
        ['source_index', 'source_id'],
        5000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Vertices are compared with every segment that does not own them; coincident vertices are ignored.',
      'Features over 5,000 vertices are not measured and report status skipped-too-many-vertices.'
    ],
    limitation:
      'Brute-force vertex-segment pairs; features with fewer than three vertices have no clearance.',
    run: (inputs, values) =>
      runMinimumClearance(inputs['features'], values['coordinateSystem'] as CatalogCoordinateSystem)
  }),
  defineCpuBinding({
    id: 'geographic-distribution',
    title: 'Geographic distribution',
    summary:
      'Mean center, median center, standard distance and standard deviational ellipse, optionally per group and weighted.',
    capabilityIds: ['geometry-measurement'],
    contributorNames: ['GPUGeographicDistribution'],
    inputs: [makeInputSpec('features', 'Features', FEATURES_ANY)],
    parameters: [
      makeStringParameter('groupProperty', 'Group property'),
      makeStringParameter('weightProperty', 'Weight property'),
      makeNumberParameter('standardDeviations', 'Ellipse standard deviations', 1, 0.1, 3),
      CRS
    ],
    outputs: [
      makeOutputSpec('distribution-ellipses', 'Standard deviational ellipses', 'polygon', [
        'group'
      ]),
      makeOutputSpec('distribution-centers', 'Mean and median centers', 'point', [
        'center_id',
        'group'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Non-point features contribute their vertex average; points of a MultiPoint each count with the feature weight.',
      'Features with a missing or negative weight are skipped; groups with fewer than three points have no ellipse polygon.',
      'The median center is the Weiszfeld geometric median.'
    ],
    limitation: 'Planar in a local frame; no small-sample or projection correction of the ellipse.',
    run: (inputs, values) =>
      runGeographicDistribution(inputs['features'], {
        groupProperty: values['groupProperty'] as string,
        weightProperty: values['weightProperty'] as string,
        standardDeviations: values['standardDeviations'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  {
    ...defineCpuBinding({
      id: 'spatial-hilbert-order',
      title: 'Hilbert curve order',
      summary:
        'Orders features along a Hilbert curve through their bounding-box centers, with the curve key and rank.',
      capabilityIds: ['spatial-ordering'],
      contributorNames: ['GPUHilbertKeys'],
      inputs: [makeInputSpec('features', 'Features', FEATURES_ANY)],
      parameters: [makeIntegerParameter('order', 'Curve order (bits per axis)', 16, 1, 16)],
      outputs: [
        makeOutputSpec('hilbert-order', 'Hilbert order', 'point', [
          'rank',
          'source_index',
          'source_id'
        ])
      ],
      declaredBehavior: [
        'Keys quantize the float32 box center against the bounds of all features; ties keep input order.',
        'Features without finite geometry get the invalid key and sort last.',
        'Longitude and latitude are used as plain coordinates; nothing wraps at the antimeridian.'
      ],
      limitation:
        'Float32 quantization: centers within a float32 rounding of a cell border may land in the neighboring cell. Results are an ordering, not a spatial index.',
      run: (inputs, values) => runHilbertOrder(inputs['features'], values['order'] as number)
    }),
    engine: 'gpu-or-cpu',
    cleanup: GPU_OR_CPU_CLEANUP,
    gpuContributorNames: ['GPUHilbertKeys'],
    runGpu: (context, inputs, values) =>
      runGpuHilbertOrder(context, inputs['features'], values['order'] as number)
  },
  defineCpuBinding({
    id: 'spatial-join-bounds-candidates',
    title: 'Spatial join candidates (bounds)',
    summary:
      'Pairs of left and right features whose bounding boxes overlap, optionally grown by a distance.',
    capabilityIds: ['spatial-joins'],
    contributorNames: ['GPUSpatialJoinCandidates', 'GPUBoundsFilter'],
    inputs: [
      makeInputSpec('left', 'Left features', FEATURES_ANY),
      makeInputSpec('right', 'Right features', FEATURES_ANY)
    ],
    parameters: [distanceParameter('distance', 'Distance to grow left bounds', 0), CRS],
    outputs: [
      makeOutputSpec(
        'join-candidates',
        'Join candidate pairs',
        'none',
        ['left_index', 'right_index'],
        200000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Candidate pairs are bounding-box overlaps only (borders count); they are not exact matches.',
      'More than 200,000 pairs throws before any result is produced.'
    ],
    limitation:
      'Bounds test only; refine with the predicate join. Longitude and latitude do not wrap at the antimeridian.',
    run: (inputs, values) =>
      runBoundsCandidates(inputs['left'], inputs['right'], {
        distance: values['distance'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'spatial-join-predicate',
    title: 'Spatial predicate join',
    summary:
      'Left features that intersect, lie within, or contain right polygons, as identity pairs.',
    capabilityIds: ['spatial-joins'],
    contributorNames: ['GPUSpatialPredicateJoin', 'GPUSpatialJoinPrepared'],
    inputs: [
      makeInputSpec('left', 'Left features', FEATURES_ANY, 5000, 50000),
      makeInputSpec('right', 'Right polygons', ['polygon'], 5000, 50000)
    ],
    parameters: [
      {
        name: 'predicate',
        label: 'Predicate',
        kind: 'enum',
        options: ['intersects', 'within', 'contains'],
        defaultValue: 'intersects'
      }
    ],
    outputs: [
      makeOutputSpec(
        'spatial-join-pairs',
        'Spatial join pairs',
        'none',
        ['join_id', 'left_index', 'right_index'],
        200000
      )
    ],
    declaredBehavior: [
      'Planar tests relative to the minimum corner of each candidate pair, so translating the pair or adding unrelated features does not change the result; bounds filtering includes the boundary tolerance; a point within 1e-12 times the larger of 1 and its local coordinate magnitude of an edge is on the boundary, and boundary contact counts as inside.',
      'within: every part of the left geometry lies in the right polygons (holes respected). contains: the left polygons cover the right polygon.',
      'Left features without geometry match nothing; more than 200,000 pairs throws.'
    ],
    limitation:
      'Pairs only, no attribute transfer; bounding boxes prefilter and the exact tests are brute force over segments, so very large inputs are slow. Geographic data is treated as planar.',
    run: (inputs, values) =>
      runSpatialPredicateJoin(
        inputs['left'],
        inputs['right'],
        values['predicate'] as SpatialPredicate
      )
  }),
  defineCpuBinding({
    id: 'geometry-coverage-validity',
    title: 'Polygon coverage validity',
    summary:
      'Detects overlapping polygons in a coverage and reports overlaps per pair and per feature.',
    capabilityIds: ['geometry-validation'],
    contributorNames: ['GPUCoverageValidity'],
    inputs: [makeInputSpec('polygons', 'Coverage polygons', ['polygon'], 3000, 50000)],
    parameters: [],
    outputs: [
      makeOutputSpec(
        'coverage-overlaps',
        'Coverage overlaps',
        'none',
        ['overlap_id', 'first_index', 'second_index'],
        200000
      ),
      makeOutputSpec(
        'coverage-features',
        'Coverage feature status',
        'polygon',
        ['source_index', 'source_id'],
        3000
      )
    ],
    declaredBehavior: [
      'Pairs overlap when their edges properly cross or a vertex, edge midpoint or verified interior point of a polygon part is strictly inside the other; shared edges are valid.',
      'Tests use binary64 coordinates relative to each candidate pair’s minimum corner, with boundary tolerance 1e-12 times the larger of 1 and the local coordinate magnitude. Features are never modified or repaired.'
    ],
    limitation:
      'Overlap detection only on planar coordinates in a candidate-pair local frame; gaps and slivers between polygons are not detected.',
    run: inputs => runCoverageValidity(inputs['polygons'])
  }),
  defineCpuBinding({
    id: 'voronoi-diagram',
    title: 'Voronoi diagram',
    summary: 'Voronoi cells of the input points clipped to the padded point extent.',
    capabilityIds: ['tessellation'],
    contributorNames: ['GPUVoronoiDiagram'],
    inputs: [makeInputSpec('points', 'Points', ['point'], 1500, 1500)],
    parameters: [makeNumberParameter('padding', 'Extent padding (fraction)', 0.1, 0, 10), CRS],
    outputs: [
      makeOutputSpec(
        'voronoi-cells',
        'Voronoi cells',
        'polygon',
        ['cell_id', 'source_index', 'source_id'],
        1500
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Points at a duplicate position get no cell; the first occurrence owns it.',
      'Cells are built by half-plane clipping of the padded extent against the nearest sites.'
    ],
    limitation:
      'At most 1,500 points; cells are clipped to the padded extent, not to a boundary polygon.',
    run: (inputs, values) =>
      runVoronoiDiagram(inputs['points'], {
        padding: values['padding'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'delaunay-triangulation',
    title: 'Delaunay triangulation',
    summary: 'Delaunay triangles and unique edges of the input points with source point identity.',
    capabilityIds: ['tessellation'],
    contributorNames: ['GPUDelaunayTessellation'],
    inputs: [makeInputSpec('points', 'Points', ['point'], 3000, 3000)],
    parameters: [CRS],
    outputs: [
      makeOutputSpec(
        'delaunay-triangles',
        'Delaunay triangles',
        'polygon',
        ['triangle_id', 'source_a', 'source_b', 'source_c'],
        20000
      ),
      makeOutputSpec(
        'delaunay-edges',
        'Delaunay edges',
        'line',
        ['edge_id', 'point_from', 'point_to'],
        30000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Bowyer-Watson on the local-frame coordinates; duplicate positions are ignored and counted.',
      'Fewer than three distinct points or only collinear points produce no triangles.'
    ],
    limitation:
      'At most 3,000 points; a large bounding triangle can clip nearly-collinear hull triangles. Not constrained and not refined.',
    run: (inputs, values) =>
      runDelaunayTriangulation(
        inputs['points'],
        values['coordinateSystem'] as CatalogCoordinateSystem
      )
  }),
  defineCpuBinding({
    id: 'concave-hull',
    title: 'Concave hull',
    summary:
      'Concave hull of all vertices, optionally per group, by trimming long Delaunay boundary triangles.',
    capabilityIds: ['expansion-footprints'],
    contributorNames: ['GPUConcaveHull', 'GPURatioConcaveHull', 'GPUGroupConcaveHull'],
    inputs: [makeInputSpec('features', 'Features', FEATURES_ANY, 5000, 5000)],
    parameters: [
      makeNumberParameter('ratio', 'Concavity ratio (1 = convex hull)', 0.3, 0, 1),
      distanceParameter('maximumEdgeLength', 'Maximum boundary edge length (0 = use ratio)', 0),
      makeStringParameter('groupProperty', 'Group property'),
      CRS
    ],
    outputs: [
      makeOutputSpec('concave-hulls', 'Concave hulls', 'polygon', ['hull_id', 'group'], 5000)
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Boundary triangles are removed while their boundary edge exceeds min + ratio * (max - min) of the Delaunay edge lengths, or the explicit maximum edge length.',
      'Removal that would pinch the hull is skipped, so the result stays one connected ring per group in normal cases.',
      'Groups with fewer than three non-collinear distinct points produce no hull.'
    ],
    limitation:
      'At most 5,000 vertices; interior holes are not created; heuristic, not an alpha shape.',
    run: (inputs, values) =>
      runConcaveHull(inputs['features'], {
        ratio: values['ratio'] as number,
        maximumEdgeLength: values['maximumEdgeLength'] as number,
        groupProperty: values['groupProperty'] as string,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'polygon-triangulation',
    title: 'Polygon triangulation',
    summary: 'Ear-clipping triangulation of polygons with holes into triangle rows.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUPolygonTriangulation'],
    inputs: [makeInputSpec('polygons', 'Polygons', ['polygon'], 2000, 50000)],
    parameters: [CRS],
    outputs: [
      makeOutputSpec(
        'polygon-triangles',
        'Polygon triangles',
        'mixed',
        ['triangle_id', 'source_index', 'source_id'],
        100000
      )
    ],
    declaredBehavior: [
      "Counted triangle rows preserve their IDs; degeneracy is 'none', 'line' or 'point', with Polygon, LineString or Point geometry respectively.",
      'Holes are joined to the shell by visible bridges before ear clipping.',
      'Each part is limited to 1,500 vertices (excluding closing duplicates); larger parts fail validation before triangulation. Invalid or self-intersecting rings may stall and keep the triangles found so far.'
    ],
    limitation:
      'At most 1,500 vertices per polygon part and 50,000 input vertices in total. Cubic worst case; no constrained Delaunay quality and no Steiner points.',
    run: (inputs, values) =>
      runPolygonTriangulation(
        inputs['polygons'],
        values['coordinateSystem'] as CatalogCoordinateSystem
      )
  }),
  defineCpuBinding({
    id: 'polygon-outline',
    title: 'Polygon outlines',
    summary: 'Splits polygons into one closed line per ring with hole flags.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUOutlineGeometry'],
    inputs: [POLYGONS_INPUT],
    parameters: [CRS],
    outputs: [
      makeOutputSpec(
        'polygon-outlines',
        'Polygon outlines',
        'line',
        ['outline_id', 'source_index', 'source_id'],
        100000
      )
    ],
    declaredBehavior: ['Rings with fewer than four positions are skipped.'],
    limitation: 'Lines only; no stroke geometry (use the line buffer for widths).',
    run: (inputs, values) =>
      runPolygonOutline(inputs['polygons'], values['coordinateSystem'] as CatalogCoordinateSystem)
  }),
  defineCpuBinding({
    id: 'geometry-shape-generator',
    title: 'Shape generator',
    summary: 'Generates circles, squares, triangles, pentagons, hexagons or stars around points.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUShapeGenerator'],
    inputs: [POINTS_INPUT],
    parameters: [
      {
        name: 'shape',
        label: 'Shape',
        kind: 'enum',
        options: ['circle', 'square', 'triangle', 'pentagon', 'hexagon', 'star'],
        defaultValue: 'circle'
      },
      distanceParameter('radius', 'Radius', 100, 1e-9),
      makeStringParameter('radiusProperty', 'Radius property (overrides radius)'),
      makeNumberParameter('rotationDegrees', 'Rotation (degrees)', 0, -360, 360),
      makeIntegerParameter('circleSegments', 'Circle segments', 32, 8, 128),
      CRS
    ],
    outputs: [
      makeOutputSpec(
        'generated-shapes',
        'Generated shapes',
        'polygon',
        ['shape_id', 'source_index', 'source_id'],
        5000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE + ' The frame is centered at each point.',
      'Points with a missing, non-numeric or non-positive radius property produce no shape.'
    ],
    limitation:
      'Vertices circumscribe at the radius; polygons are regular, not true geodesic circles.',
    run: (inputs, values) =>
      runShapeGenerator(inputs['points'], {
        shape: values['shape'] as string,
        radius: values['radius'] as number,
        radiusProperty: values['radiusProperty'] as string,
        rotationDegrees: values['rotationDegrees'] as number,
        circleSegments: values['circleSegments'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'geometry-dot-density',
    title: 'Dot density',
    summary: 'Random dots inside polygons, one per dot value of a numeric property.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUDotDensity'],
    inputs: [makeInputSpec('polygons', 'Polygons', ['polygon'], 5000, 100000)],
    parameters: [
      makeStringParameter('valueProperty', 'Value property', 'value'),
      makeNumberParameter('dotValue', 'Value per dot', 1, 1e-9, 1e12),
      makeIntegerParameter('maximumDots', 'Maximum dots', 100000, 1, 1000000),
      SEED_PARAMETER_SPEC,
      CRS
    ],
    outputs: [
      makeOutputSpec('dots', 'Dots', 'point', ['dot_id', 'source_index', 'source_id'], 1000000)
    ],
    declaredBehavior: [
      'Each feature gets floor(value / dotValue) dots placed uniformly by area in the local metric frame; the seed makes runs reproducible.',
      'Features with a missing or negative value get no dots; exceeding the maximum dots throws.',
      'Dots that cannot be placed in 1,000 attempts are dropped and counted.'
    ],
    limitation: 'Rejection sampling per part; very thin polygons may drop dots.',
    run: (inputs, values) =>
      runDotDensity(inputs['polygons'], {
        valueProperty: values['valueProperty'] as string,
        dotValue: values['dotValue'] as number,
        maximumDots: values['maximumDots'] as number,
        seed: values['seed'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'geometry-local-frame-projection',
    title: 'Local-frame projection',
    summary:
      'Projects geometries to meters in an azimuthal-equidistant frame around an origin, or back.',
    capabilityIds: ['geodesic-operations'],
    contributorNames: ['GPULocalFrameProjection'],
    inputs: [makeInputSpec('features', 'Features', FEATURES_ANY)],
    parameters: [
      {
        name: 'direction',
        label: 'Direction',
        kind: 'enum',
        options: ['forward', 'inverse'],
        defaultValue: 'forward'
      },
      {
        name: 'originLongitude',
        label: 'Origin longitude',
        kind: 'number',
        minimum: -180,
        maximum: 180,
        defaultValue: 0
      },
      {
        name: 'originLatitude',
        label: 'Origin latitude',
        kind: 'number',
        minimum: -90,
        maximum: 90,
        defaultValue: 0
      },
      {
        name: 'useDataCenter',
        label: 'Use the data bounds center as origin (forward)',
        kind: 'boolean',
        defaultValue: true
      }
    ],
    outputs: [
      makeOutputSpec('local-frame-features', 'Local-frame features', 'polygon', [
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      'Forward: longitude/latitude to meters east/north of the origin on a sphere; distances from the origin are exact.',
      'Inverse: meters back to longitude/latitude around the supplied origin.',
      'Geometry types and structure are preserved; properties are not copied.'
    ],
    limitation:
      'Spherical (not ellipsoidal); precision degrades near the antipode of the origin; the output is planar meters and must be published as planar data.',
    run: (inputs, values) =>
      runLocalFrameProjection(inputs['features'], {
        direction: values['direction'] as 'forward' | 'inverse',
        originLongitude: values['originLongitude'] as number,
        originLatitude: values['originLatitude'] as number,
        useDataCenter: values['useDataCenter'] as boolean
      })
  }),
  defineCpuBinding({
    id: 'line-split-at-points',
    title: 'Split lines at points',
    summary: 'Cuts each line at the points nearest to it within a tolerance.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineSplit'],
    inputs: [LINES_INPUT, makeInputSpec('splitters', 'Split points', ['point'], 5000, 5000)],
    parameters: [distanceParameter('tolerance', 'Snap tolerance', 10), CRS],
    outputs: [
      makeOutputSpec(
        'line-pieces',
        'Line pieces',
        'line',
        ['piece_id', 'source_index', 'source_id', 'part_index'],
        100000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Each point cuts every line part it lies within the tolerance of, at the nearest position; cuts at the ends or on top of each other are merged.',
      'Splitter points that cut nothing are counted in a diagnostic; unsplit lines return one piece.'
    ],
    limitation:
      'Points cut by projection onto the line; the geometry is not changed to pass through the point.',
    run: (inputs, values) =>
      runLineSplit(inputs['lines'], inputs['splitters'], {
        tolerance: values['tolerance'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'line-clip-by-polygon',
    title: 'Clip lines by polygons',
    summary: 'Splits lines at polygon boundaries and keeps the inside, outside or both parts.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPULineClipByPolygon'],
    inputs: [
      makeInputSpec('lines', 'Lines', ['line'], 5000, 20000),
      makeInputSpec('polygons', 'Clip polygons', ['polygon'], 2000, 20000)
    ],
    parameters: [
      {
        name: 'keep',
        label: 'Keep',
        kind: 'enum',
        options: ['inside', 'outside', 'both'],
        defaultValue: 'inside'
      },
      CRS
    ],
    outputs: [
      makeOutputSpec(
        'clipped-lines',
        'Clipped lines',
        'line',
        ['piece_id', 'source_index', 'source_id', 'part_index'],
        100000
      )
    ],
    declaredBehavior: [
      'Overlapping clip polygons count as one region; holes are respected.',
      'Consecutive pieces on the same side are merged into one line.'
    ],
    limitation:
      'Crossings use raw planar coordinates, also for EPSG:4326 (edges are not great-circle).',
    run: (inputs, values) =>
      runLineClipByPolygon(inputs['lines'], inputs['polygons'], {
        keep: values['keep'] as 'inside' | 'outside' | 'both',
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'line-segment-intersections',
    title: 'Segment intersections',
    summary: 'Points where line segments cross or touch, with both feature identities.',
    capabilityIds: ['segment-and-coverage-topology'],
    contributorNames: ['GPUSegmentIntersection', 'GPULineNoding'],
    inputs: [makeInputSpec('lines', 'Lines', ['line'], 5000, 20000)],
    parameters: [],
    outputs: [
      makeOutputSpec(
        'segment-intersections',
        'Segment intersections',
        'point',
        ['intersection_id', 'first_index', 'second_index'],
        50000
      )
    ],
    declaredBehavior: [
      'Consecutive segments of one line (and the closing pair of a closed line) are not compared.',
      'kind is crossing when both segments are cut in their interior, touch at an endpoint.',
      'More than 50,000 intersections throws.'
    ],
    limitation: 'Parallel and collinear segments produce no point; planar raw coordinates.',
    run: inputs => runSegmentIntersections(inputs['lines'])
  }),
  defineCpuBinding({
    id: 'line-shared-paths',
    title: 'Shared paths between lines',
    summary: 'Overlapping collinear stretches of two line sets with their direction.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPUSharedPaths'],
    inputs: [
      makeInputSpec('left', 'Left lines', ['line'], 3000, 20000),
      makeInputSpec('right', 'Right lines', ['line'], 3000, 20000)
    ],
    parameters: [
      distanceParameter('tolerance', 'Distance tolerance', 1),
      makeNumberParameter('angleToleranceDegrees', 'Angle tolerance (degrees)', 1, 0, 45),
      CRS
    ],
    outputs: [
      makeOutputSpec(
        'shared-paths',
        'Shared paths',
        'line',
        ['shared_id', 'left_index', 'right_index'],
        50000
      )
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Segment pairs count when nearly parallel and both right endpoints lie within the tolerance of the left line; the overlap is reported on the left segment.',
      'direction is same or opposite; more than 50,000 rows throws.'
    ],
    limitation: 'Per segment pair, not merged along a path; brute force over segment pairs.',
    run: (inputs, values) =>
      runSharedPaths(inputs['left'], inputs['right'], {
        tolerance: values['tolerance'] as number,
        angleToleranceDegrees: values['angleToleranceDegrees'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'line-great-circle-arcs',
    title: 'Great-circle arcs',
    summary:
      'Great-circle lines between paired origin and destination points with spherical distance.',
    capabilityIds: ['line-processing'],
    contributorNames: ['GPUGreatCircleArcs'],
    inputs: [
      makeInputSpec('origins', 'Origins', ['point'], 5000, 5000),
      makeInputSpec('destinations', 'Destinations', ['point'], 5000, 5000)
    ],
    parameters: [makeIntegerParameter('segments', 'Segments per arc', 32, 1, 256)],
    outputs: [
      makeOutputSpec(
        'great-circle-arcs',
        'Great-circle arcs',
        'line',
        ['arc_id', 'origin_index', 'destination_index'],
        5000
      )
    ],
    declaredBehavior: [
      'Origins and destinations pair by feature order on EPSG:4326 longitude/latitude; distance is the spherical great-circle length in meters.',
      'Coincident and antipodal pairs are skipped; arcs crossing the antimeridian keep continuous longitudes instead of being split.'
    ],
    limitation: 'Spherical, not ellipsoidal geodesics.',
    run: (inputs, values) =>
      runGreatCircleArcs(inputs['origins'], inputs['destinations'], values['segments'] as number)
  }),
  defineCpuBinding({
    id: 'geometry-offset-curve',
    title: 'Offset curve',
    summary:
      'Parallel offset of lines to the left (positive) or right (negative) with mitre joins.',
    capabilityIds: ['geometry-editing', 'coverage-geometry-refinement'],
    contributorNames: ['GPUOffsetCurve', 'GPUOffsetCurveComponents'],
    inputs: [LINES_INPUT],
    parameters: [
      makeNumberParameter('distance', 'Offset distance (left positive)', 10, -1e9, 1e9, 'crs'),
      makeNumberParameter('mitreLimit', 'Mitre limit', 5, 1, 20),
      CRS
    ],
    outputs: [
      makeOutputSpec('offset-curves', 'Offset curves', 'line', [
        'offset_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Corners beyond the mitre limit are beveled; degenerate lines are skipped.'
    ],
    limitation:
      'The raw offset is not cleaned: inner corners can form small loops (see line loop removal).',
    run: (inputs, values) =>
      runOffsetCurve(inputs['lines'], {
        distance: values['distance'] as number,
        mitreLimit: values['mitreLimit'] as number,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'line-loop-removal',
    title: 'Line loop removal',
    summary: 'Cuts self-intersection loops out of lines, replacing them by the crossing point.',
    capabilityIds: ['coverage-geometry-refinement'],
    contributorNames: ['GPULineLoopRemoval'],
    inputs: [LINES_INPUT],
    parameters: [CRS],
    outputs: [
      makeOutputSpec('loop-free-lines', 'Loop-free lines', 'line', [
        'line_id',
        'source_index',
        'source_id',
        'part_index'
      ])
    ],
    declaredBehavior: [
      'The first self-crossing is cut repeatedly (up to 1,000 times per part); lines without loops are returned unchanged.',
      'removed_loops and removed_vertices report what changed.'
    ],
    limitation: 'Planar raw coordinates; keeps the first loop-free route, not the shortest.',
    run: (inputs, values) =>
      runLineLoopRemoval(inputs['lines'], values['coordinateSystem'] as CatalogCoordinateSystem)
  }),
  defineCpuBinding({
    id: 'geometry-cleanup',
    title: 'Geometry cleanup',
    summary:
      'Removes duplicate, non-finite and spike vertices and collapsed rings from lines and polygons.',
    capabilityIds: ['geometry-editing'],
    contributorNames: ['GPUGeometryCleanup'],
    inputs: [makeInputSpec('features', 'Lines or polygons', ['line', 'polygon'])],
    parameters: [
      distanceParameter('tolerance', 'Duplicate vertex tolerance', 0),
      {name: 'removeSpikes', label: 'Remove a-b-a spikes', kind: 'boolean', defaultValue: true},
      CRS
    ],
    outputs: [
      makeOutputSpec('cleaned-geometry', 'Cleaned geometry', 'polygon', [
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      METRIC_NOTE,
      'Every input feature keeps a row: status unchanged, cleaned or dropped (geometry collapsed).',
      'Rings with fewer than three distinct vertices are removed; a collapsed shell drops the polygon part.'
    ],
    limitation: 'Local clean-up only; self-intersections and orientation are not repaired.',
    run: (inputs, values) =>
      runGeometryCleanup(inputs['features'], {
        tolerance: values['tolerance'] as number,
        removeSpikes: values['removeSpikes'] as boolean,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  }),
  defineCpuBinding({
    id: 'polygon-precision-reduction',
    title: 'Polygon precision reduction',
    summary: 'Snaps polygon vertices to a grid and removes rings that collapse.',
    capabilityIds: ['coverage-geometry-refinement'],
    contributorNames: ['GPUPolygonPrecisionReduction'],
    inputs: [POLYGONS_INPUT],
    parameters: [
      makeNumberParameter('gridSize', 'Grid size (coordinate units)', 0.001, 1e-12, 1e6)
    ],
    outputs: [
      makeOutputSpec('reduced-polygons', 'Precision-reduced polygons', 'polygon', [
        'source_index',
        'source_id'
      ])
    ],
    declaredBehavior: [
      'Vertices round to multiples of the grid size; repeated vertices are merged.',
      'Rings with fewer than three distinct vertices or zero area are removed; every feature keeps a row with status unchanged, cleaned or dropped.'
    ],
    limitation:
      'Topology is not repaired: snapped rings can become invalid or touch other polygons.',
    run: (inputs, values) =>
      runPolygonPrecisionReduction(inputs['polygons'], values['gridSize'] as number)
  }),
  defineCpuBinding({
    id: 'geometry-polygonize',
    title: 'Polygonize lines',
    summary: 'Builds polygon faces from a network of lines that meet at shared vertices.',
    capabilityIds: ['geometry-repair'],
    contributorNames: ['GPUPolygonize'],
    inputs: [makeInputSpec('lines', 'Lines', ['line'], 5000, 20000)],
    parameters: [
      makeNumberParameter('snapTolerance', 'Vertex snap tolerance', 0, 0, 1e9, 'coordinate units'),
      {
        name: 'nodeCrossings',
        label: 'Node crossing lines first',
        kind: 'boolean',
        defaultValue: false
      },
      CRS
    ],
    outputs: [
      makeOutputSpec('polygonized-faces', 'Polygonized faces', 'polygon', ['face_id'], 20000)
    ],
    declaredBehavior: [
      'Faces are traced around a planar graph of the line vertices; dangles are pruned and the unbounded face is dropped.',
      'Islands are attached as holes of the smallest enclosing face.',
      'snapTolerance uses raw coordinate units: degrees for EPSG:4326 and input units for planar; it is not a meter tolerance.',
      'More than 20,000 segments (after noding) throws.'
    ],
    limitation:
      'Lines must share vertices (snap tolerance is in coordinate units) unless crossings are noded; the noding is brute force; polygons are not validated.',
    run: (inputs, values) =>
      runPolygonize(inputs['lines'], {
        snapTolerance: values['snapTolerance'] as number,
        nodeCrossings: values['nodeCrossings'] as boolean,
        coordinateSystem: values['coordinateSystem'] as CatalogCoordinateSystem
      })
  })
];

const NETWORK_CAPABILITY = ['gpu-network'];

const NETWORK_BINDINGS: readonly CatalogOperationBinding[] = [
  defineCpuBinding(
    {
      id: 'network-accessibility',
      title: 'Network accessibility',
      summary:
        'Opportunities reachable from each origin within a network cost, with cumulative or decaying weights.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUNetworkAccessibility'],
      inputs: [
        LINE_INPUT,
        makeInputSpec('origins', 'Origins', ['point'], 500, 500),
        makeInputSpec('opportunities', 'Opportunities', ['point'], 5000, 5000)
      ],
      parameters: networkParameters([
        makeNumberParameter('maximumCost', 'Maximum cost', 1000, 0, 1e9, 'cost'),
        {
          name: 'decay',
          label: 'Decay',
          kind: 'enum',
          options: ['cumulative', 'linear', 'exponential', 'gaussian'],
          defaultValue: 'cumulative'
        },
        makeNumberParameter('decayScale', 'Decay scale', 500, 1e-9, 1e9, 'cost'),
        makeStringParameter('weightProperty', 'Opportunity weight property')
      ]),
      outputs: [
        makeOutputSpec(
          'accessibility',
          'Network accessibility',
          'point',
          ['origin_index', 'origin_id', 'point_index'],
          500
        )
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Origins and opportunities beyond the maximum snap distance are not routed (status origin-not-snapped / counted).',
        'Score = sum of weight times decay(least network cost); cumulative counts weights within the maximum cost.',
        'When weightProperty is set, missing or non-numeric weights are excluded from reachable counts and scores and counted in diagnostics; negative weights throw a named validation error.'
      ],
      limitation:
        'One Dijkstra per origin on the CPU; partial-edge costs are proportional; no turn penalties; costs are not multi-modal.',
      run: (inputs, values) => runNetworkAccessibility(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-subgraph-filter',
      title: 'Network subgraph filter',
      summary:
        'Keeps network edges by cost range, feature property and optionally only the largest connected component.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUNetworkSubgraphFilter'],
      inputs: [LINE_INPUT],
      parameters: networkParameters(
        [
          makeNumberParameter('minimumCost', 'Minimum edge cost', 0, 0, 1e12, 'cost'),
          makeNumberParameter('maximumCost', 'Maximum edge cost', 1e9, 0, 1e12, 'cost'),
          makeStringParameter('propertyName', 'Required property'),
          makeStringParameter('propertyValue', 'Required property value'),
          {
            name: 'largestComponentOnly',
            label: 'Keep only the largest component',
            kind: 'boolean',
            defaultValue: false
          }
        ],
        false
      ),
      outputs: [
        makeOutputSpec(
          'subgraph-edges',
          'Retained network edges',
          'line',
          ['edge_id', 'source_index', 'source_id'],
          50000
        )
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'The edge cost filter uses the forward cost (reverse for one-way reverse edges).',
        'Components are undirected over the retained edges; the largest is by total length.'
      ],
      limitation:
        'No spatial boundary filter (all inputs are required); property match is exact text.',
      run: (inputs, values) => runNetworkSubgraphFilter(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-coarsening',
      title: 'Network coarsening',
      summary:
        'Merges chains of edges through degree-two nodes into single edges with summed costs.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUNetworkCoarsening'],
      inputs: [LINE_INPUT],
      parameters: networkParameters([], false),
      outputs: [
        makeOutputSpec(
          'coarse-edges',
          'Coarsened edges',
          'line',
          ['coarse_id', 'from_node', 'to_node'],
          50000
        )
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Chains merge at nodes with exactly two incident edges regardless of attributes; closed loops of degree-two nodes become one edge.',
        'Forward and reverse costs are summed along the chain and are null when a member forbids that direction.'
      ],
      limitation:
        'Attribute differences (road class, name) do not stop a merge; node identities are graph-local integers.',
      run: (inputs, values) => runNetworkCoarsening(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-flow-aggregation',
      title: 'Network flow aggregation',
      summary:
        'Routes weighted origin-destination pairs on least-cost paths and sums the flow per edge.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUFlowAggregation'],
      inputs: [
        LINE_INPUT,
        makeInputSpec('origins', 'Origins', ['point'], 2000, 2000),
        makeInputSpec('destinations', 'Destinations', ['point'], 2000, 2000)
      ],
      parameters: networkParameters([
        makeStringParameter('weightProperty', 'Flow weight property (on origins)')
      ]),
      outputs: [
        makeOutputSpec(
          'flow-edges',
          'Edge flows',
          'line',
          ['edge_id', 'source_index', 'source_id'],
          50000
        ),
        makeOutputSpec(
          'flow-pairs',
          'Routed origin-destination pairs',
          'none',
          ['pair_index'],
          2000
        )
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Origins and destinations pair by feature order; unpaired features, missing weights and unreachable or unsnapped pairs carry no flow and are listed with a status.',
        'Every edge on a route, including the partial edges at the snapped ends, receives the full weight once.'
      ],
      limitation:
        'One Dijkstra per distinct origin snap on the CPU; a single least-cost route per pair (no assignment splitting); no capacity or congestion.',
      run: (inputs, values) => runNetworkFlowAggregation(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-alternative-routes',
      title: 'Alternative routes',
      summary:
        'Up to six distinct least-cost routes between two points using an edge penalty method.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUAlternativeRoutes'],
      inputs: [
        LINE_INPUT,
        makeInputSpec('origin', 'Origin', ['point'], 1, 1),
        makeInputSpec('destination', 'Destination', ['point'], 1, 1)
      ],
      parameters: networkParameters([
        makeIntegerParameter('routeCount', 'Routes', 3, 1, 6),
        makeNumberParameter('penalty', 'Edge penalty factor', 1, 0.05, 10),
        makeNumberParameter('maximumOverlap', 'Maximum overlap with an accepted route', 0.9, 0, 1)
      ]),
      outputs: [
        makeOutputSpec('alternative-routes', 'Alternative routes', 'line', ['route_rank'], 6)
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Each accepted route multiplies the cost of its edges by 1 + penalty before the next search; reported costs are the unpenalized costs.',
        'Routes sharing more than the maximum overlap of their length with an accepted route are discarded; fewer routes than requested may result.'
      ],
      limitation: 'Penalty heuristic: alternatives are plausible, not the exact k shortest paths.',
      run: (inputs, values) => runAlternativeRoutes(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-map-matching',
      title: 'Map matching',
      summary:
        'Matches an ordered GPS trace to the network with a hidden Markov model and returns the matched route.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUMapMatching'],
      inputs: [
        makeInputSpec('lines', 'Network lines', ['line'], 5000, 50000),
        makeInputSpec('trace', 'Trace points (in order)', ['point', 'line'], 500, 500)
      ],
      parameters: networkParameters(
        [
          makeNumberParameter('searchRadius', 'Candidate search radius', 50, 0, 1e6, 'crs'),
          makeIntegerParameter('candidatesPerPoint', 'Candidates per point', 4, 1, 8),
          makeNumberParameter('sigma', 'GPS noise sigma', 10, 1e-3, 1e6, 'crs'),
          makeNumberParameter('beta', 'Transition scale', 5, 1e-3, 1e6, 'crs')
        ],
        false
      ),
      outputs: [
        makeOutputSpec(
          'matched-points',
          'Matched points',
          'point',
          ['point_index', 'source_index', 'edge_id'],
          500
        ),
        makeOutputSpec('matched-route', 'Matched route', 'line', ['route_segment'], 500)
      ],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Trace points are the vertices of the input features in order; points with no edge within the search radius are unmatched and break the route.',
        'Emission is Gaussian in the snap distance; transitions penalize the difference between network cost and straight-line distance; the chain restarts where no connected transition exists.'
      ],
      limitation:
        'At most 500 trace points and 50,000 edges (brute-force candidates and one Dijkstra per candidate); timestamps, headings and speeds are not used.',
      run: (inputs, values) => runMapMatching(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-k-function',
      title: 'Network K function',
      summary: 'Network-distance Ripley K of points snapped onto the network at chosen radii.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUNetworkKFunction'],
      inputs: [LINE_INPUT, makeInputSpec('points', 'Points', ['point'], 600, 600)],
      parameters: networkParameters([
        {
          name: 'radii',
          label: 'Radii',
          kind: 'number-list',
          unit: 'cost',
          minimum: 0,
          maximum: 1e9,
          maximumLength: 16,
          defaultValue: [100, 200, 400]
        }
      ]),
      outputs: [makeOutputSpec('network-k', 'Network K function', 'none', ['radius'], 16)],
      declaredBehavior: [
        ...NETWORK_BEHAVIOR,
        'Points beyond the maximum snap distance are ignored; fewer than two snapped points leave K null.',
        'Only costMode=length is supported; property-cost mode throws because radii and physical network length must share units.',
        'K(r) = L / (n (n - 1)) * ordered pairs within network cost r; expected_k is the 2r reference of an unbounded line.'
      ],
      limitation:
        'No edge correction, Monte Carlo envelope or significance test; at most 600 points.',
      run: (inputs, values) => runNetworkKFunction(inputs, values)
    },
    'network'
  ),
  defineCpuBinding(
    {
      id: 'network-edge-bundling',
      title: 'Edge bundling',
      summary: 'Force-directed bundling of origin-destination lines into shared corridors.',
      capabilityIds: NETWORK_CAPABILITY,
      contributorNames: ['GPUEdgeBundling'],
      inputs: [makeInputSpec('lines', 'Origin-destination lines', ['line'], 400, 4000)],
      parameters: [
        makeIntegerParameter('cycles', 'Subdivision cycles', 4, 1, 6),
        makeIntegerParameter('iterations', 'Iterations in the first cycle', 40, 1, 120),
        makeNumberParameter('stiffness', 'Spring stiffness', 0.1, 0.01, 1),
        makeNumberParameter('compatibilityThreshold', 'Compatibility threshold', 0.5, 0, 1),
        CRS
      ],
      outputs: [
        makeOutputSpec('bundled-edges', 'Bundled edges', 'line', ['source_index', 'source_id'], 400)
      ],
      declaredBehavior: [
        METRIC_NOTE,
        'Each line is reduced to its first and last vertex; lines with identical ends are skipped.',
        'Attraction acts only between edges whose angle, scale and position compatibility reach the threshold.'
      ],
      limitation:
        'At most 400 lines; the bundling is deterministic but not visibility-compatibility aware.',
      run: (inputs, values) => runEdgeBundling(inputs['lines'], values)
    },
    'network'
  )
];

/** PLATFORM3 bindings (family A): complete bindings, with gpuContributorNames and runGpu when GPU-backed. */
export const PLATFORM3_A_BINDINGS: readonly CatalogOperationBinding[] = [
  ...GEOMETRY_BINDINGS,
  ...NETWORK_BINDINGS
];
