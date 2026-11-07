// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "geometry-processing": measure, test, repair and normalize geometry.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

const POLYGON_COMMENT =
  "// Views are graph views on `graph`; ports need `kind` ('polygons' or 'lines'); polygons add featureOffsets, polygonOffsets, ringOffsets.";

/** Builds the shared graph.add, compile, encode example around a constructor body. */
function makeExample(symbol: string, body: string[], before: string[] = []): string {
  return [
    `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    `import {${symbol}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    POLYGON_COMMENT,
    ...before,
    `graph.add(new ${symbol}({`,
    ...body.map(line => `  ${line}`),
    '}));',
    'const compiled = await graph.compileAsync();',
    'compiled.encode(device.commandEncoder, {parameters: undefined});'
  ].join('\n');
}

function makeParam(
  name: string,
  type: string,
  description: string,
  options: {required?: boolean; default?: string} = {}
): ParamDoc {
  return {name, type, description, ...options};
}

function makeOutput(name: string, type: string, description: string): OutputDoc {
  return {name, type, description};
}

const POLYGON_POSITIONS = makeParam(
  'positions',
  'GraphDataView<"float32x2">',
  'Packed planar vertex positions.',
  {required: true}
);
const FEATURE_OFFSETS = makeParam(
  'featureOffsets',
  'GraphDataView<"uint32">',
  'Feature-to-polygon offsets with one terminal entry.',
  {required: true}
);
const POLYGON_OFFSETS = makeParam(
  'polygonOffsets',
  'GraphDataView<"uint32">',
  'Polygon-to-ring offsets with one terminal entry.',
  {required: true}
);
const RING_OFFSETS = makeParam(
  'ringOffsets',
  'GraphDataView<"uint32">',
  'Ring-to-vertex offsets with one terminal entry.',
  {required: true}
);
const SPATIAL_CONTEXT = makeParam(
  'spatialContext',
  'GPUSpatialContext | undefined',
  'Coordinate and metric contract; omitted means native planar.'
);

export const entries: CatalogEntry[] = [
  {
    id: 'geometry-measures',
    symbol: 'GPUGeometryMeasures',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometry measures',
    summary: 'Length, area, centroid, bounds and vertex count of every line or polygon at once.',
    explanation:
      'Sizes, ranks or filters features without leaving the GPU. Polygon area is shell minus holes, never negative. Planar, spherical and ellipsoidal metrics come from `spatialContext`.',
    dataTypes: {input: ['lines', 'polygons'], output: ['table']},
    useCases: ['measure'],
    domains: ['cartography', 'urban'],
    params: [
      makeParam('geometryType', '"polygons"', 'Feature kind: lines or polygons.', {required: true}),
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      SPATIAL_CONTEXT,
      makeParam('output', 'GPUGeometryMeasuresOutput | undefined', 'Per-feature columns to write.'),
      makeParam(
        'groupIds',
        'GraphDataView<"uint32"> | undefined',
        'Optional per-feature group ids for grouped totals.'
      ),
      makeParam('groupCount', 'number | undefined', 'Number of groups.'),
      makeParam(
        'ellipsoidalEdgeModel',
        '"coordinate-linear" | "geodesic" | undefined',
        'Edge interpretation for ellipsoidal polygon areas.'
      )
    ],
    outputs: [
      makeOutput('lengths', 'GraphDataView<"float32">', 'Line length or polygon perimeter.'),
      makeOutput('areas', 'GraphDataView<"float32">', 'Polygon area, shell minus holes.'),
      makeOutput('centroids', 'GraphDataView<"float32x2">', 'Area or length weighted centroid.'),
      makeOutput('bounds', 'GraphDataView<"float32x4">', '[minX, minY, maxX, maxY] per feature.'),
      makeOutput('vertexCounts', 'GraphDataView<"uint32">', 'Vertex rows per feature.')
    ],
    example: makeExample('GPUGeometryMeasures', [
      `geometryType: 'polygons',`,
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'output: {areas, lengths, centroids, bounds}',
      '// longitude/latitude: spatialContext with metric ellipsoidal'
    ]),
    preview: {
      kind: 'polygons',
      seed: 4201,
      caption: 'Area and perimeter of each parcel, in one dispatch.'
    },
    related: ['shape-descriptors', 'label-point', 'minimum-bounds'],
    lensScenario: 'gpu-spatial-analysis.geometry-measures',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'shape-descriptors',
    symbol: 'GPUShapeDescriptors',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Shape descriptors',
    summary: 'Compactness, convexity, elongation, orientation and sliver flags for every polygon.',
    explanation:
      'Separates compact districts from slivers and sprawl. Polsby-Popper and Schwartzberg score roundness; convexity compares area to the convex hull. A sliver threshold is a per-frame parameter.',
    dataTypes: {input: ['polygons'], output: ['table']},
    useCases: ['measure'],
    domains: ['cartography', 'urban'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'Packed settings from getGPUShapeDescriptorsParameterValues.',
        {required: true}
      ),
      makeParam('output', 'GPUShapeDescriptorsOutput', 'Per-feature output columns.', {
        required: true
      }),
      makeParam(
        'convexityMethod',
        'GPUShapeDescriptorsConvexityMethod | undefined',
        'Hull algorithm behind the convexity column.'
      )
    ],
    outputs: [
      makeOutput(
        'polsbyPopper',
        'GraphDataView<"float32">',
        'Compactness in (0, 1], 1 is a circle.'
      ),
      makeOutput('schwartzberg', 'GraphDataView<"float32">', 'Perimeter-ratio compactness.'),
      makeOutput('convexity', 'GraphDataView<"float32">', 'Area over convex hull area.'),
      makeOutput('elongation', 'GraphDataView<"float32">', 'Long over short axis.'),
      makeOutput('orientation', 'GraphDataView<"float32">', 'Principal axis angle.'),
      makeOutput('sliver', 'GraphDataView<"uint32">', '1 when compactness is below the threshold.')
    ],
    example: makeExample('GPUShapeDescriptors', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'parameters, // getGPUShapeDescriptorsParameterValues({sliverThreshold: 0.1})',
      'output: {polsbyPopper, convexity, elongation, sliver}'
    ]),
    preview: {kind: 'polygons', seed: 4202, caption: 'Round, convex and sliver shapes, scored.'},
    related: ['geometry-measures', 'minimum-bounds', 'maximum-inscribed-circle'],
    lensScenario: 'gpu-spatial-analysis.shape-descriptors',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'label-point',
    symbol: 'GPULabelPoint',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Label point',
    summary:
      'The pole of inaccessibility of each polygon: the interior point farthest from its edge.',
    explanation:
      'Anchors labels and pins inside awkward shapes where the centroid falls outside. A coarse grid finds the best cells, then refinement rounds sharpen the answer.',
    dataTypes: {input: ['polygons'], output: ['points']},
    useCases: ['measure', 'classify'],
    domains: ['cartography'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam('output', 'GPULabelPointOutput', 'Output columns.', {required: true}),
      makeParam('initialGridSize', 'number | undefined', 'Samples per axis of the first grid.'),
      makeParam('refinementRounds', 'number | undefined', 'Refinement rounds.'),
      makeParam(
        'refinementCandidates',
        'number | undefined',
        'Best first-grid cells that may be refined, 1 to 16.'
      )
    ],
    outputs: [
      makeOutput('points', 'GraphDataView<"float32x2">', 'Label anchor per polygon.'),
      makeOutput(
        'distances',
        'GraphDataView<"float32">',
        'Distance from the anchor to the boundary.'
      ),
      makeOutput(
        'degenerate',
        'GraphDataView<"uint32">',
        '1 when the polygon has no usable interior.'
      )
    ],
    example: makeExample('GPULabelPoint', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'refinementRounds: 3,',
      'output: {points, distances}'
    ]),
    preview: {
      kind: 'polygons',
      seed: 4203,
      caption: 'Each label sits at the widest part of its shape.'
    },
    related: ['maximum-inscribed-circle', 'geometry-measures'],
    lensScenario: 'gpu-spatial-analysis.label-point',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'maximum-inscribed-circle',
    symbol: 'GPUMaximumInscribedCircle',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Maximum inscribed circle',
    summary:
      'The largest circle that fits inside each polygon, with its radius and touching points.',
    explanation:
      'Best for guaranteed-fit placement and tight label anchors. A best-first quadtree search runs over all rings under even-odd fill and stops at `tolerance`.',
    dataTypes: {input: ['polygons'], output: ['points', 'table']},
    useCases: ['measure'],
    domains: ['cartography', 'urban'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam('tolerance', 'number', 'Positive stopping tolerance in planar units.', {
        required: true
      }),
      makeParam('cellCapacity', 'number | undefined', 'Live quadtree cells per feature.', {
        default: '256'
      }),
      makeParam('maximumIterations', 'number | undefined', 'Cell splits per feature.', {
        default: '1024'
      }),
      makeParam('output', '{circles, witnesses}', 'Caller-owned circle outputs.', {required: true})
    ],
    outputs: [
      makeOutput('circles', 'GraphDataView<"float32x4">', 'Center and radius per polygon.'),
      makeOutput(
        'witnesses',
        'GraphDataView<"float32x4">',
        'Boundary points that touch the circle.'
      ),
      makeOutput('status', "GraphDataView<'uint32'>", 'Per-feature search status code.')
    ],
    example: makeExample('GPUMaximumInscribedCircle', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'tolerance: 0.5,',
      'output: {circles, witnesses, status}'
    ]),
    preview: {kind: 'polygons', seed: 4204, caption: 'The biggest circle that still fits.'},
    related: ['label-point', 'minimum-clearance', 'shape-descriptors'],
    lensScenario: 'gpu-spatial-analysis.maximum-inscribed-circle',
    reference: 'api/gpu-geometry-refinement',
    maturity: 'stable'
  },
  {
    id: 'minimum-bounds',
    symbol: 'GPUMinimumBounds',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Minimum bounds',
    summary:
      'Minimum rotated rectangle, bounding circle and longest line of each feature or group.',
    explanation:
      'Best for oriented footprints, enclosing circles and feature extent. It builds a convex hull per group, then rotating calipers find the tightest shapes.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['polygons', 'table']},
    useCases: ['measure'],
    domains: ['cartography', 'urban'],
    params: [
      POLYGON_POSITIONS,
      makeParam(
        'featureOffsets',
        'GraphDataView<"uint32"> | undefined',
        'Feature-to-polygon offsets with one terminal entry.'
      ),
      makeParam('output', 'GPUMinimumBoundsOutput', 'Caller-owned outputs.', {required: true}),
      makeParam('maximumHullVertices', 'number | undefined', 'Largest hull per group.', {
        default: '2048'
      }),
      SPATIAL_CONTEXT
    ],
    outputs: [
      makeOutput(
        'rectangleCorners',
        'GraphDataView<"float32x2">',
        'Corners of the rotated rectangle.'
      ),
      makeOutput(
        'rectangleSizes',
        'GraphDataView<"float32x4">',
        'Rectangle width, height and axis.'
      ),
      makeOutput('circles', 'GraphDataView<"float32x3">', 'Minimum bounding circle: x, y, radius.'),
      makeOutput('longestLines', 'GraphDataView<"float32x4">', 'Diameter segment endpoints.'),
      makeOutput('diameters', 'GraphDataView<"float32">', 'Length of the longest line.'),
      makeOutput('status', '{overflow}', 'Nonzero when the hull vertex cap was exceeded.')
    ],
    example: makeExample('GPUMinimumBounds', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'output: {rectangleCorners, circles, diameters, status}'
    ]),
    preview: {kind: 'polygons', seed: 4205, caption: 'Tightest rotated box around each shape.'},
    related: ['geometry-measures', 'shape-descriptors', 'ratio-concave-hull'],
    lensScenario: 'gpu-spatial-analysis.minimum-bounds',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'minimum-clearance',
    symbol: 'GPUMinimumClearance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Minimum clearance',
    summary:
      'How far each feature can be perturbed before it becomes invalid, with the witness line.',
    explanation:
      'Finds fragile geometry before simplifying or rounding. It is the smallest distance between a vertex and a nonadjacent edge, found with a BVH over segments.',
    dataTypes: {input: ['lines', 'polygons'], output: ['table', 'lines']},
    useCases: ['measure'],
    domains: ['cartography', 'data-engineering'],
    params: [
      makeParam('geometry', 'GPUSegmentGeometry', 'Linestring or polygon features.', {
        required: true
      }),
      makeParam('output', 'GPUMinimumClearanceOutput', 'Outputs; at least one column.', {
        required: true
      }),
      makeParam('leafCapacity', 'number | undefined', 'Power-of-two BVH leaf slots.'),
      makeParam('spatialSort', 'boolean | undefined', 'Sort segments spatially (compile-time).')
    ],
    outputs: [
      makeOutput('clearances', 'GraphDataView<"float32">', 'Minimum clearance per feature.'),
      makeOutput('lines', 'GraphDataView<"float32x4">', 'Segment realizing the clearance.'),
      makeOutput('vertexIds', 'GraphDataView<"uint32">', 'Vertex rows that define it.')
    ],
    example: makeExample('GPUMinimumClearance', [
      'geometry, // line or polygon rows in the GeoArrow layout',
      'output: {clearances, lines}'
    ]),
    preview: {kind: 'lines', seed: 4206, caption: 'The thinnest gap in each shape.'},
    related: ['geometry-validity', 'maximum-inscribed-circle', 'polygon-precision-reduction'],
    lensScenario: 'gpu-spatial-analysis.minimum-clearance',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'geometry-predicates',
    symbol: 'GPUGeometryPredicates',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometry predicates',
    summary:
      'Per-feature is-simple, is-ring, is-closed and winding flags plus coordinate and part counts.',
    explanation:
      'Profiles a layer before analysis: which lines self-cross, which rings are closed, how many parts. Give it a second geometry for element-wise `equalsExact`.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['relate', 'measure'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('geometry', 'GPUSpatialJoinGeometry', 'Point, line or polygon features.', {
        required: true
      }),
      makeParam('output', 'GPUGeometryPredicatesOutput', 'Caller-owned columns and status.', {
        required: true
      }),
      makeParam(
        'other',
        'GPUSpatialJoinGeometry | undefined',
        'Second geometry for element-wise tests.'
      ),
      makeParam(
        'parameters',
        'GraphDataView<"float32"> | undefined',
        'Packed settings, for example the exact-equality tolerance.'
      ),
      makeParam(
        'intersectionCapacity',
        'number | undefined',
        'Capacity of the self-intersection list.'
      ),
      makeParam('leafCapacity', 'number | undefined', 'Power-of-two BVH leaf slots.')
    ],
    outputs: [
      makeOutput(
        'isSimple',
        'GraphDataView<"uint32">',
        '1 when the feature has no self-intersection.'
      ),
      makeOutput('isRing', 'GraphDataView<"uint32">', '1 for a closed simple line.'),
      makeOutput('isCcw', 'GraphDataView<"uint32">', '1 for counter-clockwise rings.'),
      makeOutput('numCoordinates', 'GraphDataView<"uint32">', 'Coordinate count.'),
      makeOutput('equalsExact', 'GraphDataView<"uint32">', 'Exact match against `other`.')
    ],
    example: makeExample('GPUGeometryPredicates', [
      'geometry,',
      'output: {isSimple, isRing, isCcw, numCoordinates, status}'
    ]),
    preview: {
      kind: 'lines',
      seed: 4207,
      caption: 'Simple lines pass; self-crossing lines are flagged.'
    },
    related: ['geometry-validity', 'segment-intersection'],
    lensScenario: 'gpu-spatial-analysis.geometry-predicates',
    reference: 'api/gpu-spatial-analysis/operators-validity',
    maturity: 'stable'
  },
  {
    id: 'geometry-validity',
    symbol: 'GPUGeometryValidity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometry validity',
    summary: 'A validity bitmask per feature: self-intersections, bad rings, holes outside shells.',
    explanation:
      'The gate before overlay, buffering or joins, the GPU analog of `ST_IsValidReason`. Each bit names one failure so you can count, filter or route to repair.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['relate', 'transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam(
        'polygons',
        'GPUPackedGeometryPort<GPUPolygonGeometryPort> | undefined',
        'Polygon features.'
      ),
      makeParam(
        'lines',
        'GPUPackedGeometryPort<GPULineGeometryPort> | undefined',
        'Linestring features.'
      ),
      makeParam(
        'points',
        'GPUPackedGeometryPort<GPUPointGeometryPort> | undefined',
        'Point features.'
      ),
      makeParam('output', 'GPUGeometryValidityOutput', 'Validity `mask` and optional status.', {
        required: true
      }),
      makeParam(
        'ringClosure',
        'GPUGeometryValidityRingClosure | undefined',
        'Ring closure convention.'
      ),
      makeParam(
        'orientation',
        'GPUGeometryValidityOrientation | undefined',
        'Ring orientation convention.'
      ),
      makeParam(
        'intersectionCapacity',
        'number | undefined',
        'Capacity of the self-intersection list.'
      )
    ],
    outputs: [
      makeOutput('mask', 'GraphDataView<"uint32">', 'Bitmask per feature; 0 is valid.'),
      makeOutput('status', '{count, overflow}', 'Overflow and uncertainty counters.')
    ],
    example: makeExample('GPUGeometryValidity', [
      "polygons: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},",
      'output: {mask}'
    ]),
    preview: {
      kind: 'polygon-overlay',
      seed: 4208,
      caption: 'Bow-ties and overlaps light up as invalid.'
    },
    related: ['make-valid', 'geometry-predicates', 'polygonize'],
    lensScenario: 'gpu-spatial-analysis.geometry-validity',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'make-valid',
    symbol: 'GPUMakeValid',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Make valid',
    summary: 'Rebuilds valid polygons from invalid boundaries, one source feature at a time.',
    explanation:
      'Repairs bow-ties and overlapping rings before overlay. It nodes every boundary exactly, then polygonizes the bounded faces under even-odd fill.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      makeParam('polygons', 'GPUPolygonGeometryPort', 'Polygon boundaries to node and rebuild.', {
        required: true
      }),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam(
        'precision',
        'GPUTopologyPrecisionPolicy',
        'Vertex identity and predicate policy.',
        {
          required: true
        }
      ),
      makeParam('output', 'GPUPolygonizeOutput', 'Caller-owned outputs and diagnostics.', {
        required: true
      }),
      makeParam(
        'snapRounding',
        '{ gridSize: number; lattice: GraphDataView<"float32">; } | undefined',
        'Optional hot-pixel snap rounding.'
      ),
      makeParam('leafCapacity', 'number | undefined', 'Power-of-two BVH leaf slots.')
    ],
    outputs: [
      makeOutput('nodedSegments', 'GPUNodedSegmentPort', 'Noded input segments.'),
      makeOutput(
        'directedEdges',
        'GPUDirectedEdgePort',
        'Directed half-edges built from the noded segments.'
      ),
      makeOutput('polygons', 'GPUPolygonGeometryPort', 'Repaired valid polygons.'),
      makeOutput(
        'diagnostics',
        'GPUPolygonizeDiagnosticsPort',
        'Counts of dangles, cuts and invalid rings.'
      ),
      makeOutput('status', 'GPUTopologyStatusPort', 'Capacity and completeness flags.')
    ],
    example: makeExample('GPUMakeValid', [
      "polygons: {kind: 'polygons', featureOffsets, polygonOffsets, ringOffsets, positions},",
      'intersectionCapacity: 4096,',
      `precision: {vertexTolerance: 1e-6, predicates: 'exact-or-uncertain', storage: 'float32'},`,
      'output // nodedSegments, directedEdges, polygons, diagnostics, status'
    ]),
    preview: {
      kind: 'polygon-overlay',
      seed: 4209,
      caption: 'A bow-tie becomes two valid triangles.'
    },
    related: ['geometry-validity', 'polygonize', 'line-noding'],
    lensScenario: 'gpu-spatial-analysis.make-valid',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'polygonize',
    symbol: 'GPUPolygonize',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Polygonize',
    summary: 'Turns arbitrary planar linework into bounded polygons.',
    explanation:
      'Use it to build areas from street, boundary or cut lines. Lines are noded exactly, linked into half-edges, and every bounded face becomes a polygon.',
    dataTypes: {input: ['lines'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      makeParam('lines', 'GPULineGeometryPort', 'Arbitrary finite planar linework.', {
        required: true
      }),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam(
        'precision',
        'GPUTopologyPrecisionPolicy',
        'Explicit precision and vertex policy.',
        {
          required: true
        }
      ),
      makeParam(
        'output',
        'GPUPolygonizeOutput',
        'Noding, half-edge, polygon and diagnostic outputs.',
        {
          required: true
        }
      ),
      makeParam(
        'groupIds',
        'GraphDataView<"uint32"> | undefined',
        'Polygonization group per input line.'
      ),
      makeParam(
        'snapRounding',
        '{ gridSize: number; lattice: GraphDataView<"float32">; } | undefined',
        'Optional hot-pixel snap rounding.'
      )
    ],
    outputs: [
      makeOutput('nodedSegments', 'GPUNodedSegmentPort', 'Noded input segments.'),
      makeOutput(
        'directedEdges',
        'GPUDirectedEdgePort',
        'Directed half-edges built from the noded segments.'
      ),
      makeOutput('polygons', 'GPUPolygonGeometryPort', 'Polygons assembled from closed rings.'),
      makeOutput(
        'diagnostics',
        'GPUPolygonizeDiagnosticsPort',
        'Counts of dangles, cuts and invalid rings.'
      ),
      makeOutput('status', 'GPUTopologyStatusPort', 'Capacity and completeness flags.')
    ],
    example: makeExample('GPUPolygonize', [
      "lines: {kind: 'lines', positions, lineOffsets},",
      'intersectionCapacity: 4096,',
      `precision: {vertexTolerance: 1e-6, predicates: 'exact-or-uncertain', storage: 'float32'},`,
      'output // nodedSegments, directedEdges, polygons, diagnostics, status'
    ]),
    preview: {kind: 'polygon-overlay', seed: 4210, caption: 'Crossing lines enclose faces.'},
    related: ['line-noding', 'make-valid', 'segment-intersection'],
    lensScenario: 'gpu-spatial-analysis.polygonize',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'line-noding',
    symbol: 'GPULineNoding',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line noding',
    summary: 'Splits planar linework at every crossing into atomic segments with provenance.',
    explanation:
      'The first step of any topology build. Exact intersection events cut each line, and every output row remembers its source feature, ring and edge.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform', 'relate'],
    domains: ['cartography', 'transport'],
    params: [
      makeParam('lines', 'GPULineGeometryPort', 'Arbitrary finite planar linework.', {
        required: true
      }),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam('output', 'GPUNodedSegmentPort', 'Atomic noded segments and provenance.', {
        required: true
      }),
      makeParam('spatialSort', 'boolean | undefined', 'Sort segments spatially (compile-time).'),
      makeParam('leafCapacity', 'number | undefined', 'Power-of-two BVH leaf slots.')
    ],
    outputs: [
      makeOutput('endpoints', 'GraphDataView<"float32x4">', 'Segment endpoints x0, y0, x1, y1.'),
      makeOutput('sourceFeatureIds', 'GraphDataView<"uint32">', 'Source feature per segment.'),
      makeOutput('sourceEdgeIds', 'GraphDataView<"uint32">', 'Source edge per segment.'),
      makeOutput('status', 'GPUTopologyStatusPort', 'Capacity and overflow counters.'),
      makeOutput('sourceRingIds', "GraphDataView<'uint32'>", 'Source ring of each noded segment.'),
      makeOutput(
        'sourceStartParameters',
        "GraphDataView<'float32'>",
        'Start parameter along the source edge, 0 to 1.'
      ),
      makeOutput(
        'sourceEndParameters',
        "GraphDataView<'float32'>",
        'End parameter along the source edge, 0 to 1.'
      )
    ],
    example: makeExample('GPULineNoding', [
      "lines: {kind: 'lines', positions, lineOffsets},",
      'intersectionCapacity: 4096,',
      'output // endpoints, sourceFeatureIds, sourceRingIds, sourceEdgeIds, status'
    ]),
    preview: {kind: 'network', seed: 4211, caption: 'Every crossing becomes a node.'},
    related: ['polygonize', 'segment-intersection', 'line-loop-removal'],
    lensScenario: 'gpu-spatial-analysis.line-noding',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'segment-intersection',
    symbol: 'GPUSegmentIntersection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Segment intersection',
    summary:
      'Finds every intersecting pair of line segments, between two geometries or within one.',
    explanation:
      'Finds crossings, overlaps and touch points exactly. A BVH over one side prunes pairs; predicates are exact with an explicit uncertain count.',
    dataTypes: {input: ['lines', 'polygons'], output: ['table', 'points']},
    useCases: ['relate'],
    domains: ['cartography', 'transport'],
    params: [
      makeParam('left', 'GPUSegmentGeometry', 'Left segments.', {required: true}),
      makeParam('right', 'GPUSegmentGeometry | undefined', 'Right segments; omit for self mode.'),
      makeParam(
        'sameFeatureOnly',
        'boolean | undefined',
        'Self mode: keep same-feature pairs only.'
      ),
      makeParam('output', 'GPUSegmentIntersectionOutput', 'Caller-owned outputs.', {
        required: true
      }),
      makeParam('leafCapacity', 'number | undefined', 'Power-of-two BVH leaf slots.'),
      makeParam('spatialSort', 'boolean | undefined', 'Sort segments spatially (compile-time).')
    ],
    outputs: [
      makeOutput('pairs', 'GPUSegmentIntersectionPairs', 'Left and right segment ids per hit.'),
      makeOutput('status', '{count, overflow}', 'Count, overflow and uncertain counters.'),
      makeOutput('kinds', "GraphDataView<'uint32'> | undefined", 'Intersection kind per pair.'),
      makeOutput(
        'points',
        "GraphDataView<'float32x2'> | undefined",
        'Intersection point, or start of an overlap.'
      ),
      makeOutput(
        'endPoints',
        "GraphDataView<'float32x2'> | undefined",
        'End of an overlap; equals points otherwise.'
      ),
      makeOutput(
        'leftFeatures',
        "GraphDataView<'uint32'> | undefined",
        'Feature row of the left segment.'
      ),
      makeOutput(
        'rightFeatures',
        "GraphDataView<'uint32'> | undefined",
        'Feature row of the right segment.'
      ),
      {
        name: 'leftRings, rightRings',
        type: "GraphDataView<'uint32'> | undefined",
        description: 'Further result fields; see the reference.'
      }
    ],
    example: makeExample('GPUSegmentIntersection', [
      'left, // streets',
      'right, // rivers; omit for self-intersections',
      'output // pairs plus intersection columns and status'
    ]),
    preview: {kind: 'lines', seed: 4212, caption: 'Crossings between two line sets.'},
    related: ['line-noding', 'geometry-validity', 'polygonize'],
    lensScenario: 'gpu-spatial-analysis.segment-intersection',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'bounds-filter',
    symbol: 'GPUBoundsFilter',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Bounds filter',
    summary:
      'Selects features whose bounding boxes intersect, sit within or contain a per-frame box.',
    explanation:
      'Best for map-window culling and fast pre-filters, a GPU `GeoDataFrame.cx[...]`. The box is a parameter, so panning rewrites four floats and never recompiles.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['relate', 'index'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('bounds', 'GraphDataView<"float32x4">', 'Per-feature [minX, minY, maxX, maxY].', {
        required: true
      }),
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'Box from getGPUBoundsFilterParameterValues.',
        {
          required: true
        }
      ),
      makeParam('mode', 'GPUBoundsFilterMode | undefined', 'intersects, within or contains.', {
        default: "'intersects'"
      }),
      makeParam('output', 'GPUBoundsFilterOutput', 'Bounded list of accepted rows.', {
        required: true
      }),
      SPATIAL_CONTEXT
    ],
    outputs: [
      makeOutput('ids', 'GraphDataView<"uint32">', 'Accepted feature rows, ascending.'),
      makeOutput('status', '{count, overflow}', 'Overflow count of the bounded list.'),
      makeOutput('outputMask', 'GraphDataView<"uint32">', 'Optional 0 or 1 mask per feature.')
    ],
    example: makeExample(
      'GPUBoundsFilter',
      [`bounds, parameters: box.importToGraph(graph), mode: 'intersects', output`],
      [
        "const box = new GPUParameterBuffer(device, {id: 'box', format: 'float32', length: 4});",
        'box.write(getGPUBoundsFilterParameterValues({minX, minY, maxX, maxY}));'
      ]
    ),
    preview: {kind: 'points', seed: 4213, caption: 'Only features touching the window survive.'},
    related: ['rectangle-clip', 'geometry-measures'],
    lensScenario: 'gpu-spatial-analysis.bounds-filter',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'rectangle-clip',
    symbol: 'GPURectangleClip',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Rectangle clip',
    summary: 'Clips lines and polygon rings to an axis-aligned rectangle.',
    explanation:
      'Cuts features to a tile, viewport or study box (`clip_by_rect`). Lines use Liang-Barsky; the rectangle is a per-frame parameter.',
    dataTypes: {input: ['lines', 'polygons'], output: ['lines', 'polygons']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      makeParam('geometryType', '"lines"', 'Open lines clipped with Liang-Barsky, or polygons.', {
        required: true
      }),
      POLYGON_POSITIONS,
      makeParam('lineOffsets', 'GraphDataView<"uint32"> | undefined', 'Feature-to-vertex offsets.'),
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'Rectangle from getGPURectangleClipParameterValues.',
        {
          required: true
        }
      ),
      makeParam('output', 'GPULineGeometryOutput', 'Bounded output.', {required: true})
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Clipped vertices.'),
      makeOutput('lineOffsets', 'GraphDataView<"uint32">', 'Clipped feature offsets.'),
      makeOutput('status', '{count, overflow}', 'Count and overflow counters.')
    ],
    example: makeExample(
      'GPURectangleClip',
      [`geometryType: 'lines', positions, lineOffsets, parameters, output`],
      ['// parameters: getGPURectangleClipParameterValues({minX, minY, maxX, maxY})']
    ),
    preview: {
      kind: 'polygon-overlay',
      seed: 4214,
      caption: 'Everything outside the window is cut away.'
    },
    related: ['bounds-filter', 'lon-lat-polygon-preparation'],
    lensScenario: 'gpu-spatial-analysis.rectangle-clip',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'affine-transform',
    symbol: 'GPUAffineTransform',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Affine transform',
    summary:
      'Translates, scales, rotates or skews many features at once with per-feature matrices.',
    explanation:
      'Best for layout, animation and alignment (`shapely.affinity`). Per-feature `[a, b, d, e, xoff, yoff]` rows can be rewritten every frame without recompiling.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['points', 'lines', 'polygons']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      POLYGON_POSITIONS,
      makeParam('geometryType', '"polygons"', 'Feature kind: points, lines or polygons.', {
        required: true
      }),
      FEATURE_OFFSETS,
      makeParam('output', 'GPUAffineTransformOutput', 'Output positions.', {required: true}),
      makeParam(
        'featureTransforms',
        'GraphDataView<"float32"> | undefined',
        '6 floats per feature.'
      ),
      makeParam(
        'parameters',
        'GraphDataView<"float32"> | undefined',
        'Per-frame packed transform.'
      ),
      makeParam(
        'origins',
        'readonly GPUAffineTransformOrigin[] | undefined',
        'Origins that parameters may select.'
      )
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Transformed vertices, same layout.')
    ],
    example: makeExample('GPUAffineTransform', [
      `geometryType: 'polygons',`,
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'featureTransforms, // [a, b, d, e, xoff, yoff] per feature',
      'output: {positions: transformed}'
    ]),
    preview: {
      kind: 'affine-transform',
      seed: 4215,
      caption: 'One matrix: gridlines and basis vectors show [a b; d e].'
    },
    related: ['geometry-orientation', 'local-frame-projection'],
    lensScenario: 'gpu-spatial-analysis.affine-transform',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'geometry-orientation',
    symbol: 'GPUGeometryOrientation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometry orientation',
    summary: 'Reverses rings or forces polygon winding to GeoJSON or shapefile convention.',
    explanation:
      'Best when a renderer or format needs a fixed winding (`ST_ForcePolygonCW`, `orient_polygons`). `exteriorClockwise` is a per-frame setting; counter-clockwise exteriors are the default.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['points', 'lines', 'polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      makeParam('mode', '"orient-polygons"', 'Orient polygons, or reverse lines and rings.', {
        required: true
      }),
      POLYGON_POSITIONS,
      makeParam(
        'featureOffsets',
        'GraphDataView<"uint32"> | undefined',
        'Feature-to-polygon offsets.'
      ),
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'From getGPUGeometryOrientationParameterValues.',
        {
          required: true
        }
      ),
      makeParam('output', 'GPUGeometryOrientationOutput', 'Output views.', {required: true})
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Reoriented vertices, same layout.'),
      makeOutput('reversedRings', 'GraphDataView<"uint32">', 'Optional 1 per reversed ring.')
    ],
    example: makeExample(
      'GPUGeometryOrientation',
      [
        `mode: 'orient-polygons',`,
        'positions, featureOffsets, polygonOffsets, ringOffsets,',
        'parameters, output: {positions: oriented, reversedRings}'
      ],
      ['// parameters: getGPUGeometryOrientationParameterValues({exteriorClockwise: false})']
    ),
    preview: {
      kind: 'polygons',
      seed: 4216,
      caption: 'Exteriors counter-clockwise, holes clockwise.'
    },
    related: ['geometry-cleanup', 'normalize-polygons', 'geometry-predicates'],
    lensScenario: 'gpu-spatial-analysis.geometry-orientation',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'geometry-cleanup',
    symbol: 'GPUGeometryCleanup',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometry cleanup',
    summary:
      'Snaps coordinates to a grid and drops repeated vertices from points, lines or polygons.',
    explanation:
      'Sheds redundant vertices and noise before storage or rendering. Grid size and repeat tolerance are per-frame parameters, so you can scrub them live.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['points', 'lines', 'polygons']},
    useCases: ['transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('geometryType', '"points"', 'Feature kind: points, lines or polygons.', {
        required: true
      }),
      POLYGON_POSITIONS,
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'From getGPUGeometryCleanupParameterValues.',
        {
          required: true
        }
      ),
      makeParam('output', 'GPUGeometryCleanupPointOutput', 'Output views for the geometry kind.', {
        required: true
      }),
      makeParam('lineOffsets', 'GraphDataView<"uint32">', 'Feature-to-vertex offsets for lines.'),
      makeParam(
        'forceSequentialRepeatRemoval',
        'boolean | undefined',
        'Always compile the repeat pass.'
      )
    ],
    outputs: [
      makeOutput('positions', "GraphDataView<'float32x2'>", 'Snapped, deduplicated vertices.'),
      makeOutput(
        'lineOffsets',
        "GraphDataView<'uint32'> | undefined",
        'Optional line start offsets.'
      ),
      makeOutput(
        'ringOffsets',
        "GraphDataView<'uint32'> | undefined",
        'Optional ring start offsets.'
      ),
      makeOutput('status', 'GPUGeometryCleanupStatus', 'Written counts and overflow flags.')
    ],
    example: makeExample(
      'GPUGeometryCleanup',
      [`geometryType: 'lines', positions, lineOffsets, parameters, output`],
      ['// parameters: getGPUGeometryCleanupParameterValues({gridSize: 0.01, tolerance: 0.001})']
    ),
    preview: {kind: 'lines', seed: 4217, caption: 'Noisy vertices collapse onto a grid.'},
    related: ['line-precision-reduction', 'polygon-precision-reduction', 'vertex-snap'],
    lensScenario: 'gpu-spatial-analysis.geometry-cleanup',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'vertex-snap',
    symbol: 'GPUVertexSnap',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Vertex snap',
    summary: 'Moves each vertex onto the nearest reference vertex within a tolerance.',
    explanation:
      'Closes gaps between layers that should share corners (`shapely.snap`). The layout is unchanged, so attributes stay aligned; tolerance is a per-frame parameter.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['points', 'lines', 'polygons']},
    useCases: ['transform', 'relate'],
    domains: ['cartography', 'data-engineering'],
    params: [
      makeParam('positions', 'GraphDataView<"float32x2">', 'Positions of the geometry to snap.', {
        required: true
      }),
      makeParam(
        'referencePositions',
        'GraphDataView<"float32x2">',
        'Reference vertices to snap to.',
        {
          required: true
        }
      ),
      makeParam('parameters', 'GraphDataView<"float32">', 'From getGPUVertexSnapParameterValues.', {
        required: true
      }),
      makeParam('output', 'GPUVertexSnapOutput', 'Output views.', {required: true}),
      makeParam(
        'featureOffsets',
        'GraphDataView<"uint32"> | undefined',
        'Pairwise mode: feature offsets.'
      ),
      makeParam(
        'referenceOffsets',
        'GraphDataView<"uint32"> | undefined',
        'Pairwise mode: reference offsets.'
      )
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Snapped vertices, same layout.'),
      makeOutput(
        'referenceRows',
        'GraphDataView<"uint32">',
        'Matched reference row, or no-reference sentinel.'
      )
    ],
    example: makeExample(
      'GPUVertexSnap',
      ['positions, referencePositions, parameters,', 'output: {positions: snapped, referenceRows}'],
      ['// parameters: getGPUVertexSnapParameterValues({tolerance: 50})']
    ),
    preview: {kind: 'lines', seed: 4218, caption: 'Park corners jump onto neighbor corners.'},
    related: ['segment-snap', 'geometry-cleanup'],
    lensScenario: 'gpu-spatial-analysis.vertex-snap',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'segment-snap',
    symbol: 'GPUSegmentSnap',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Segment snap',
    summary:
      'Snaps line endpoints to reference vertices and insert nearby reference vertices along segments.',
    explanation:
      'Makes lines pass exactly through shared nodes, as in coverage cleanup. Endpoints snap first, then every reference vertex within `tolerance` is inserted in projection order.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      POLYGON_POSITIONS,
      makeParam('lineOffsets', 'GraphDataView<"uint32">', 'Feature-to-vertex offsets.', {
        required: true
      }),
      makeParam('referencePositions', 'GraphDataView<"float32x2">', 'Reference vertices.', {
        required: true
      }),
      makeParam('tolerance', 'number', 'Nonnegative inclusive distance in planar units.', {
        required: true
      }),
      makeParam('maximumVerticesPerLine', 'number', 'Fixed output slots per input line.', {
        required: true
      }),
      makeParam('output', '{positions, sourceEdges}', 'Snapped lines.', {
        required: true
      })
    ],
    outputs: [
      makeOutput(
        'positions',
        'GraphDataView<"float32x2">',
        'Snapped vertices with inserted nodes.'
      ),
      makeOutput('sourceEdges', 'GraphDataView<"uint32">', 'Source edge of each output vertex.'),
      makeOutput(
        'counts',
        "GraphDataView<'uint32x4'>",
        'Per line: written, required, overflow and invalid flags.'
      )
    ],
    example: makeExample('GPUSegmentSnap', [
      'positions, lineOffsets, referencePositions,',
      'tolerance: 2,',
      'maximumVerticesPerLine: 64,',
      'output // positions, lineOffsets, sourceEdges, status'
    ]),
    preview: {kind: 'lines', seed: 4219, caption: 'Lines pick up the nodes they pass near.'},
    related: ['vertex-snap', 'line-noding'],
    lensScenario: 'gpu-spatial-analysis.segment-snap',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'line-precision-reduction',
    symbol: 'GPULinePrecisionReduction',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line precision reduction',
    summary: 'Rounds lines to a grid with an explicit policy for lines that collapse.',
    explanation:
      'Shrinks coordinates for storage or matching while knowing what happened to every line. `valid-output` drops collapsed lines; `keep-collapsed` keeps them with a status.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      POLYGON_POSITIONS,
      makeParam('lineOffsets', 'GraphDataView<"uint32">', 'Feature-to-vertex offsets.', {
        required: true
      }),
      makeParam('gridSize', 'number', 'Grid step to round to.', {required: true}),
      makeParam('mode', '"valid-output" | "keep-collapsed"', 'Collapse policy.', {required: true}),
      makeParam('output', '{positions, lineOffsets}', 'Rounded lines and status.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Rounded vertices.'),
      makeOutput('lineOffsets', 'GraphDataView<"uint32">', 'Offsets of the rounded lines.'),
      makeOutput('status', '{count, overflow}', 'Per-source line status.')
    ],
    example: makeExample('GPULinePrecisionReduction', [
      'positions, lineOffsets,',
      'gridSize: 0.5,',
      `mode: 'valid-output',`,
      'output // positions, lineOffsets, status'
    ]),
    preview: {kind: 'lines', seed: 4220, caption: 'Lines rounded to a coarse grid.'},
    related: ['polygon-precision-reduction', 'geometry-cleanup'],
    lensScenario: 'gpu-spatial-analysis.line-precision-reduction',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'polygon-precision-reduction',
    symbol: 'GPUPolygonPrecisionReduction',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Polygon precision reduction',
    summary: 'Rounds polygons to a grid and rebuild them valid, like GEOS set_precision.',
    explanation:
      'Snaps polygons to a lattice without creating gaps or bow-ties. Hot-pixel noding rounds the linework, then even-odd reconstruction rebuilds each polygon.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('polygons', 'GPUPolygonGeometryPort', 'Polygon boundaries to round and rebuild.', {
        required: true
      }),
      makeParam('gridSize', 'number', 'Positive finite grid step.', {required: true}),
      makeParam('lattice', 'GraphDataView<"float32">', '[originX, originY, actualGridSize, 0].', {
        required: true
      }),
      makeParam(
        'mode',
        '"valid-output" | "keep-collapsed" | undefined',
        'Collapsed polygons become empty.'
      ),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam('output', 'GPUPolygonizeOutput', 'Caller-owned outputs.', {required: true})
    ],
    outputs: [
      makeOutput('nodedSegments', 'GPUNodedSegmentPort', 'Noded input segments.'),
      makeOutput(
        'directedEdges',
        'GPUDirectedEdgePort',
        'Directed half-edges built from the noded segments.'
      ),
      makeOutput('polygons', 'GPUPolygonGeometryPort', 'Polygons with reduced precision.'),
      makeOutput(
        'diagnostics',
        'GPUPolygonizeDiagnosticsPort',
        'Counts of dangles, cuts and invalid rings.'
      ),
      makeOutput('status', 'GPUTopologyStatusPort', 'Capacity and completeness flags.')
    ],
    example: makeExample('GPUPolygonPrecisionReduction', [
      "polygons: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},",
      'gridSize: 1, lattice, intersectionCapacity: 4096,',
      'output // polygons, nodedSegments, directedEdges, status'
    ]),
    preview: {kind: 'cells', seed: 4221, caption: 'Polygons snapped onto a visible lattice.'},
    related: ['line-precision-reduction', 'make-valid', 'geometry-cleanup'],
    lensScenario: 'gpu-spatial-analysis.polygon-precision-reduction',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'line-loop-removal',
    symbol: 'GPULineLoopRemoval',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line loop removal',
    summary: 'Erases self-loops from open curves in the order they were drawn.',
    explanation:
      'Cleans offset curves and tracks that curl back on themselves. Curves are noded first, then each loop is cut out chronologically.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      makeParam('lines', 'GPULineGeometryPort', 'Prepared offset curves or open planar lines.', {
        required: true
      }),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam('output', '{positions, lineOffsets}', 'Loop-free lines.', {
        required: true
      }),
      SPATIAL_CONTEXT
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Vertices after loop erasure.'),
      makeOutput('lineOffsets', 'GraphDataView<"uint32">', 'Offsets of the cleaned lines.'),
      makeOutput('sourceEdges', "GraphDataView<'uint32'>", 'Source edge of each output vertex.'),
      makeOutput(
        'status',
        "GraphDataView<'uint32x4'>",
        'Written, required, overflow and incomplete-noding words.'
      )
    ],
    example: makeExample('GPULineLoopRemoval', [
      "lines: {kind: 'lines', positions, lineOffsets},",
      'intersectionCapacity: 1024,',
      'output // positions, lineOffsets, status'
    ]),
    preview: {kind: 'lines', seed: 4222, caption: 'A curl is cut out of the path.'},
    related: ['offset-curve-components', 'line-noding'],
    lensScenario: 'gpu-spatial-analysis.line-loop-removal',
    reference: 'api/gpu-geometry-refinement',
    maturity: 'stable'
  },
  {
    id: 'offset-curve-components',
    symbol: 'GPUOffsetCurveComponents',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Offset curve components',
    summary: 'Turns raw one-sided offsets into clean multipart curves at the requested distance.',
    explanation:
      'Use it to finish an offset: node the raw curve, discard parts closer to the source than `distance`, keep the requested side and compact the components.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      makeParam('original', 'GPULineGeometryFields', 'Source lines.', {required: true}),
      makeParam('raw', 'GPULineGeometryFields', 'Raw offset of each source line, same order.', {
        required: true
      }),
      makeParam('distance', 'number', 'Offset distance.', {required: true}),
      makeParam('tolerance', 'number', 'Absolute geometric slack, including chord error.', {
        required: true
      }),
      makeParam('intersectionCapacity', 'number', 'Capacity of the intersection event list.', {
        required: true
      }),
      makeParam('segmentCapacity', 'number', 'Capacity of kept segments.', {required: true}),
      makeParam('output', '{positions, components}', 'Compacted components.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Vertices of the kept components.'),
      makeOutput('components', 'GraphDataView<"uint32">', 'Component offsets per source line.'),
      makeOutput('sourceEdges', "GraphDataView<'uint32'>", 'Source edge of each output vertex.'),
      makeOutput(
        'status',
        "GraphDataView<'uint32x4'>",
        'Written and required component and vertex counts.'
      )
    ],
    example: makeExample('GPUOffsetCurveComponents', [
      'original, raw, // GPULineGeometryFields',
      'distance: 20, tolerance: 0.5,',
      'intersectionCapacity: 4096, segmentCapacity: 4096,',
      'output // positions, components, ...'
    ]),
    preview: {kind: 'buffers', seed: 4223, caption: 'A street offset 20 m to one side.'},
    related: ['line-loop-removal', 'line-noding'],
    lensScenario: 'gpu-spatial-analysis.offset-curve-components',
    reference: 'api/gpu-geometry-refinement',
    maturity: 'stable'
  },
  {
    id: 'normalize-polygons',
    symbol: 'GPUNormalizePolygons',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Normalize polygons',
    summary: 'Puts polygons in one canonical form: ring order, start vertex and winding.',
    explanation:
      'Best before comparing or hashing polygons that describe the same shape differently. The output is stable, so equal shapes produce equal arrays.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam('output', '{positions, ringOffsets}', 'Canonical polygons.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Canonical vertices.'),
      makeOutput('ringOffsets', 'GraphDataView<"uint32">', 'Canonical ring offsets.'),
      makeOutput(
        'polygonOffsets',
        "GraphDataView<'uint32'>",
        'Polygon-to-ring offsets after normalization.'
      )
    ],
    example: makeExample('GPUNormalizePolygons', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'output // positions, featureOffsets, polygonOffsets, ringOffsets'
    ]),
    preview: {
      kind: 'polygons',
      seed: 4224,
      caption: 'Same shape, many encodings, one canonical form.'
    },
    related: ['canonical-polygon-equality', 'normalize-geometry', 'geometry-orientation'],
    lensScenario: 'gpu-spatial-analysis.normalize-polygons',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'canonical-polygon-equality',
    symbol: 'GPUCanonicalPolygonEquality',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Canonical polygon equality',
    summary: 'Compares two normalized polygon sets feature by feature, without hashing.',
    explanation:
      'Tests that two pipelines produced the same polygons. Inputs are `GPUNormalizePolygons` outputs; coordinates are compared within an absolute tolerance.',
    dataTypes: {input: ['polygons'], output: ['table']},
    useCases: ['relate'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('left', 'GPUPolygonGeometryFields', 'Normalized polygons, left side.', {
        required: true
      }),
      makeParam('right', 'GPUPolygonGeometryFields', 'Normalized polygons, right side.', {
        required: true
      }),
      makeParam('tolerance', 'number | undefined', 'Absolute per-coordinate tolerance.'),
      makeParam(
        'output',
        'GraphDataView<"uint32">',
        'Per feature: equal 1, unequal 0, nonfinite 2.',
        {
          required: true
        }
      )
    ],
    outputs: [makeOutput('output', 'GraphDataView<"uint32">', 'Equal 1, unequal 0, nonfinite 2.')],
    example: makeExample('GPUCanonicalPolygonEquality', [
      'left: normalizedA, right: normalizedB, // GPUNormalizePolygons outputs',
      'tolerance: 1e-6,',
      'output: equalFlags'
    ]),
    preview: {kind: 'polygon-overlay', seed: 4225, caption: 'Matching shapes overlay exactly.'},
    related: ['normalize-polygons', 'canonical-geometry-equality'],
    lensScenario: 'gpu-spatial-analysis.canonical-polygon-equality',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'normalize-geometry',
    symbol: 'GPUNormalizeGeometry',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Normalize geometry',
    summary:
      'Canonicalizes points, lines, polygons, multis and nested collections, keeping their structure.',
    explanation:
      'Best for mixed geometry trees where `GPUNormalizePolygons` is too narrow. Nodes are `[type, firstVertexOrChildSlot, count, sourceId]` rows and nothing is flattened.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['points', 'lines', 'polygons']},
    useCases: ['transform'],
    domains: ['data-engineering', 'cartography'],
    params: [
      POLYGON_POSITIONS,
      makeParam(
        'nodes',
        'GraphDataView<"uint32x4">',
        '[type, firstVertexOrChildSlot, count, sourceId].',
        {
          required: true
        }
      ),
      makeParam('children', 'GraphDataView<"uint32">', 'Child node ids, grouped by parent.', {
        required: true
      }),
      makeParam(
        'maximumNestingDepth',
        'number | undefined',
        'Maximum collection depth (compile-time).'
      ),
      makeParam('output', '{positions, children}', 'Canonical tree.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Canonical vertices.'),
      makeOutput('children', 'GraphDataView<"uint32">', 'Canonical child ids.'),
      makeOutput('status', "GraphDataView<'uint32'>", 'Per node: 0 success, nonzero error code.')
    ],
    example: makeExample('GPUNormalizeGeometry', [
      'positions, nodes, children,',
      'maximumNestingDepth: 4,',
      'output // positions, nodes, children'
    ]),
    preview: {kind: 'polygons', seed: 4226, caption: 'Nested collections keep their shape.'},
    related: ['canonical-geometry-equality', 'normalize-polygons'],
    lensScenario: 'gpu-spatial-analysis.normalize-geometry',
    reference: 'api/gpu-geometry-refinement',
    maturity: 'stable'
  },
  {
    id: 'canonical-geometry-equality',
    symbol: 'GPUCanonicalGeometryEquality',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Canonical geometry equality',
    summary:
      'Structural equality of normalized geometry trees, including nested empty collections.',
    explanation:
      'Compares any normalized geometry, not only polygons. Give it root node pairs from two `GPUNormalizeGeometry` trees and read one flag per pair.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['table']},
    useCases: ['relate'],
    domains: ['data-engineering'],
    params: [
      makeParam('left', 'GPUCanonicalGeometryTree', 'Normalized tree, left side.', {
        required: true
      }),
      makeParam('right', 'GPUCanonicalGeometryTree', 'Normalized tree, right side.', {
        required: true
      }),
      makeParam(
        'pairs',
        'GraphDataView<"uint32x2">',
        'Root node ids in the left and right trees.',
        {
          required: true
        }
      ),
      makeParam('tolerance', 'number | undefined', 'Absolute per-coordinate tolerance.'),
      makeParam('output', 'GraphDataView<"uint32">', 'Per pair: equal 1, unequal 0, invalid 2.', {
        required: true
      })
    ],
    outputs: [makeOutput('output', 'GraphDataView<"uint32">', 'Equal 1, unequal 0, invalid 2.')],
    example: makeExample('GPUCanonicalGeometryEquality', [
      'left: treeA, right: treeB, // GPUNormalizeGeometry outputs',
      'pairs, // [leftRoot, rightRoot] per comparison',
      'output: equalFlags'
    ]),
    preview: {kind: 'polygon-overlay', seed: 4227, caption: 'Two trees, one verdict per pair.'},
    related: ['normalize-geometry', 'canonical-polygon-equality'],
    lensScenario: 'gpu-spatial-analysis.canonical-geometry-equality',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'subdivide-polygons',
    symbol: 'GPUSubdividePolygons',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Subdivide polygons',
    summary: 'Splits complex polygons into small convex or rectilinear pieces.',
    explanation:
      'Speeds up joins, clipping and rendering of huge polygons. Interiors are triangulated with holes, then edge-adjacent triangles merge while the union stays convex.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform', 'index'],
    domains: ['cartography', 'data-engineering'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam('method', '"convex" | "rectilinear" | undefined', 'Piece shape.', {
        default: "'convex'"
      }),
      makeParam('maximumVertices', 'number | undefined', 'Vertices per convex piece, 3 to 128.'),
      makeParam('output', '{positions, pieces}', 'Pieces and offsets.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Vertices of all pieces.'),
      makeOutput('pieces', 'GraphDataView<"uint32">', 'Piece offsets with source feature ids.'),
      makeOutput(
        'status',
        "GraphDataView<'uint32x4'>",
        'Written pieces, required pieces and overflow flags.'
      )
    ],
    example: makeExample('GPUSubdividePolygons', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      `method: 'convex', maximumVertices: 16,`,
      'output // positions, pieces, ...'
    ]),
    preview: {kind: 'tessellation', seed: 4228, caption: 'One coastline, many convex pieces.'},
    related: ['convex-straight-skeleton', 'vector-straight-skeleton'],
    lensScenario: 'gpu-spatial-analysis.subdivide-polygons',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'ratio-concave-hull',
    symbol: 'GPURatioConcaveHull',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Ratio concave hull',
    summary: 'A concave hull around points, tightened by an edge-length ratio.',
    explanation:
      'Outlines clusters more honestly than a convex hull. Delaunay border triangles are eroded while their edges exceed `ratio`, a fraction of the edge length range.',
    dataTypes: {input: ['points'], output: ['polygons', 'lines']},
    useCases: ['transform', 'find-patterns'],
    domains: ['cartography', 'ecology'],
    params: [
      POLYGON_POSITIONS,
      makeParam('ratio', 'number', 'Fraction of the Delaunay edge length range, 0 to 1.', {
        required: true
      }),
      makeParam('allowHoles', 'boolean | undefined', 'Allow interior erosion to start holes.'),
      makeParam('maximumIterations', 'number | undefined', 'Maximum triangle removals.'),
      makeParam('output', '{boundaryEdges, edgeValidity}', 'Hull edges.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('boundaryEdges', 'GraphDataView<"uint32x2">', 'Hull edges as point index pairs.'),
      makeOutput('edgeValidity', 'GraphDataView<"uint32">', '1 for edges that are on the hull.'),
      makeOutput('triangles', "GraphDataView<'uint32x3'>", 'Retained Delaunay triangles.'),
      makeOutput(
        'triangleValidity',
        "GraphDataView<'uint32'>",
        '1 for each triangle kept in the hull.'
      ),
      makeOutput('status', '{converged}', 'One row; 0 if tessellation did not finish.')
    ],
    example: makeExample('GPURatioConcaveHull', [
      'positions, // float32x2 points',
      'ratio: 0.3,',
      'output // boundaryEdges, edgeValidity, ...'
    ]),
    preview: {
      kind: 'point-clusters',
      seed: 4229,
      caption: 'The hull hugs the cluster at ratio 0.3.'
    },
    related: ['minimum-bounds', 'subdivide-polygons'],
    lensScenario: 'gpu-spatial-analysis.ratio-concave-hull',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'stable'
  },
  {
    id: 'vector-straight-skeleton',
    symbol: 'GPUVectorStraightSkeleton',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Vector straight skeleton',
    summary: 'The straight skeleton and inward offsets of polygons with holes and reflex corners.',
    explanation:
      'Best for medial axes, roof lines and inset outlines. A wavefront shrinks the boundary, handling edge collapses, reflex splits and hole merges as events.',
    dataTypes: {input: ['polygons'], output: ['lines']},
    useCases: ['transform', 'measure'],
    domains: ['cartography', 'urban'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam('offsetDistance', 'number', 'Inward offset in planar units.', {required: true}),
      makeParam('segmentCapacity', 'number', 'Skeleton arcs reserved per polygon.', {
        required: true
      }),
      makeParam('maximumEvents', 'number | undefined', 'Collapse and split events per polygon.', {
        default: '128'
      }),
      makeParam('output', '{segments, sourceEdges}', 'Skeleton arcs.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('segments', 'GraphDataView<"float32x4">', 'Skeleton arcs x0, y0, x1, y1.'),
      makeOutput('sourceEdges', 'GraphDataView<"uint32">', 'Source edges that generate each arc.'),
      makeOutput(
        'offsetPositions',
        "GraphDataView<'float32x2'>",
        'Offset wavefront vertices, fixed block per polygon.'
      ),
      makeOutput(
        'offsetLinks',
        "GraphDataView<'uint32x2'>",
        'Previous and next offset vertex ids.'
      ),
      makeOutput(
        'status',
        "GraphDataView<'uint32x4'>",
        'Written and required arcs, active vertices, flags.'
      )
    ],
    example: makeExample('GPUVectorStraightSkeleton', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'offsetDistance: 5, segmentCapacity: 256,',
      'output // segments, sourceEdges, ...'
    ]),
    preview: {
      kind: 'tessellation',
      seed: 4230,
      caption: 'Skeleton arcs run down the middle of a block.'
    },
    related: ['convex-straight-skeleton', 'maximum-inscribed-circle'],
    lensScenario: 'gpu-spatial-analysis.vector-straight-skeleton',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'partial'
  },
  {
    id: 'convex-straight-skeleton',
    symbol: 'GPUConvexStraightSkeleton',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Convex straight skeleton',
    summary: 'Straight skeleton and inward offset for independent strictly convex polygons.',
    explanation:
      'Best for the fast convex case of roof lines and insets. It is a wavefront prototype; reach for the vector skeleton when polygons have reflex corners or holes.',
    dataTypes: {input: ['polygons'], output: ['lines']},
    useCases: ['transform', 'measure'],
    domains: ['cartography', 'urban'],
    params: [
      POLYGON_POSITIONS,
      makeParam(
        'ringOffsets',
        'GraphDataView<"uint32">',
        'One strictly convex simple polygon per run.',
        {
          required: true
        }
      ),
      makeParam('offsetDistance', 'number', 'Inward offset in planar units.', {required: true}),
      makeParam('maximumVertices', 'number | undefined', 'Vertices per polygon.'),
      makeParam('maximumIterations', 'number | undefined', 'Wavefront iteration cap.'),
      makeParam('eventTolerance', 'number | undefined', 'Relative simultaneous-event tolerance.', {
        default: '1e-5'
      }),
      makeParam('output', '{segments, sourceEdges}', 'Skeleton arcs.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('segments', 'GraphDataView<"float32x4">', 'Skeleton arcs x0, y0, x1, y1.'),
      makeOutput('sourceEdges', 'GraphDataView<"uint32">', 'Source edges that generate each arc.'),
      makeOutput(
        'offsetPositions',
        "GraphDataView<'float32x2'>",
        'Inward offset vertices, fixed block per polygon.'
      ),
      makeOutput(
        'status',
        "GraphDataView<'uint32x4'>",
        'Written and required arcs, offset vertices, flags.'
      )
    ],
    example: makeExample('GPUConvexStraightSkeleton', [
      'positions, ringOffsets, // one convex polygon per ring',
      'offsetDistance: 5,',
      'output // segments, sourceEdges, ...'
    ]),
    preview: {kind: 'tessellation', seed: 4231, caption: 'A convex lot shrinks to a ridge.'},
    related: ['vector-straight-skeleton', 'subdivide-polygons'],
    lensScenario: 'gpu-spatial-analysis.convex-straight-skeleton',
    reference: 'api/gpu-spatial-analysis/geometry-refinement-lenses',
    maturity: 'experimental'
  },
  {
    id: 'geodesic-pairs',
    symbol: 'GPUGeodesicPairs',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geodesic pairs',
    summary: 'Distance, bearings and midpoint between many origin and target pairs on the globe.',
    explanation:
      'Best for great-circle or ellipsoidal distances between aligned point columns (`turf.distance`, `ST_Distance`). `spatialContext` sets the sphere or WGS84 model and the output unit.',
    dataTypes: {input: ['points'], output: ['table', 'points']},
    useCases: ['measure'],
    domains: ['transport', 'urban'],
    params: [
      makeParam('spatialContext', 'GPUSpatialContext', 'Longitude/latitude metric and unit.', {
        required: true
      }),
      makeParam('origins', 'GraphDataView<"float32x2">', 'Origins, longitude/latitude degrees.', {
        required: true
      }),
      makeParam('targets', 'GraphDataView<"float32x2">', 'Targets aligned with origins.', {
        required: true
      }),
      makeParam('maximumIterations', 'number | undefined', 'Vincenty iteration cap for wgs84.'),
      makeParam('output', 'GPUGeodesicPairsOutput', 'Per-pair outputs; at least one.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('distances', 'GraphDataView<"float32">', 'Geodesic distance in the context unit.'),
      makeOutput('initialBearings', 'GraphDataView<"float32">', 'Degrees clockwise from north.'),
      makeOutput('finalBearings', 'GraphDataView<"float32">', 'Arrival bearing in degrees.'),
      makeOutput('midpoints', 'GraphDataView<"float32x2">', 'Geodesic midpoint per pair.')
    ],
    example: makeExample('GPUGeodesicPairs', [
      `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'},`,
      'origins, targets,',
      'output: {distances, initialBearings, midpoints}'
    ]),
    preview: {kind: 'flows', seed: 4232, caption: 'Great-circle arcs between city pairs.'},
    related: ['geodesic-destination', 'local-frame-projection', 'geometry-measures'],
    lensScenario: 'gpu-spatial-analysis.geodesic-pairs',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'geodesic-destination',
    symbol: 'GPUGeodesicDestination',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geodesic destination',
    summary: 'The point reached from each origin along a bearing for a distance on the globe.',
    explanation:
      'Best for range rings, sector wedges and direct geodesic problems (`turf.destination`, `ST_Project`). Bearings are degrees clockwise from north.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['measure', 'transform'],
    domains: ['transport', 'urban'],
    params: [
      makeParam('spatialContext', 'GPUSpatialContext', 'Longitude/latitude metric and unit.', {
        required: true
      }),
      makeParam('origins', 'GraphDataView<"float32x2">', 'Origins, longitude/latitude degrees.', {
        required: true
      }),
      makeParam('bearings', 'GraphDataView<"float32">', 'Initial bearings in degrees.', {
        required: true
      }),
      makeParam('distances', 'GraphDataView<"float32">', 'Distances in the context unit.', {
        required: true
      }),
      makeParam('maximumIterations', 'number | undefined', 'Vincenty iteration cap for wgs84.'),
      makeParam('output', 'GPUGeodesicDestinationOutput', 'Per-row outputs; at least one.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('destinations', 'GraphDataView<"float32x2">', 'Destination longitude/latitude.'),
      makeOutput('finalBearings', 'GraphDataView<"float32">', 'Bearing on arrival, in degrees.'),
      makeOutput('status', '{converged?}', 'Per-row convergence flag.')
    ],
    example: makeExample('GPUGeodesicDestination', [
      `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'},`,
      'origins, bearings, distances,',
      'output: {destinations}'
    ]),
    preview: {
      kind: 'buffers',
      seed: 4233,
      caption: 'A 50 km range ring, one destination per bearing.'
    },
    related: ['geodesic-pairs', 'local-frame-projection'],
    lensScenario: 'gpu-spatial-analysis.geodesic-destination',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'local-frame-projection',
    symbol: 'GPULocalFrameProjection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local frame projection',
    summary:
      'Projects longitude/latitude into a meter-based local frame so planar operators can use it.',
    explanation:
      'Best for sub-17 m work where raw float32 degrees are too coarse. Create a frame at your study origin, project on the GPU, run planar operators, then read meters.',
    dataTypes: {input: ['points'], output: ['points']},
    useCases: ['transform'],
    domains: ['cartography', 'urban'],
    params: [
      makeParam('frame', 'GPULocalFrame', 'Frame from createGPULocalFrame.', {required: true}),
      makeParam('positions', 'GPUGeospatialPositions', 'Longitude/latitude positions in degrees.', {
        required: true
      }),
      makeParam(
        'bounds',
        'ProjectionBounds',
        '[minLongitude, minLatitude, maxLongitude, maxLatitude].',
        {
          required: true
        }
      ),
      makeParam('output', 'GPUFloat32Positions', 'Caller-owned float32x2 meters.', {
        required: true
      }),
      makeParam('tolerance', 'number | undefined', 'Maximum projection error in meters.'),
      makeParam(
        'validity',
        'GPUProjectionValidity | undefined',
        '1 for projected rows, 0 for outside bounds.'
      )
    ],
    outputs: [
      makeOutput(
        'output',
        'GPUFloat32Positions',
        'Frame coordinates in meters, one row per input.'
      ),
      makeOutput('validity', 'GPUProjectionValidity', 'Optional uint32 inside-bounds flags.')
    ],
    example: makeExample(
      'GPULocalFrameProjection',
      [
        'frame, positions, // lon/lat degrees',
        'bounds: [-87.8, 41.8, -87.6, 42.0],',
        'output: meters'
      ],
      [`const frame = createGPULocalFrame({origin: [-87.7, 41.9]});`]
    ),
    preview: {kind: 'points', seed: 4234, caption: 'Degrees in, meters out, no CPU round trip.'},
    related: ['geodesic-pairs', 'geometry-measures', 'lon-lat-polygon-preparation'],
    lensScenario: 'gpu-spatial-analysis.local-frame-projection',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'lon-lat-polygon-preparation',
    symbol: 'GPULonLatPolygonPreparation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Lon/lat polygon preparation',
    summary:
      'Densifies, splits at the antimeridian, caps at poles and clips longitude/latitude polygons.',
    explanation:
      'Best before gridding or rendering polygons that cross +/-180 degrees or enclose a pole. Each ring is classified, then split or capped by the chosen policy.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam(
        'parameters',
        'GraphDataView<"float32">',
        'Four floats from the parameter packer.',
        {
          required: true
        }
      ),
      makeParam(
        'antimeridian',
        '"split" | "reject" | undefined',
        'Split at the antimeridian or reject.',
        {
          default: "'split'"
        }
      ),
      makeParam('poleRings', '"reject" | "cap" | undefined', 'Cap at the nearer pole or reject.', {
        default: "'cap'"
      }),
      makeParam('output', 'GPULonLatPolygonPreparationOutput', 'Bounded output and diagnostics.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('positions', 'GraphDataView<"float32x2">', 'Vertices in [-180, 180].'),
      makeOutput(
        'ringFlags',
        'GraphDataView<"uint32">',
        'Crossing, capped and invalid flags per ring.'
      ),
      makeOutput('status', '{count, overflow}', 'Completeness and validity diagnostics.'),
      makeOutput(
        'featureOffsets',
        "GraphDataView<'uint32'>",
        'Feature-to-polygon offsets, one row per input feature.'
      ),
      makeOutput(
        'polygonOffsets',
        "GraphDataView<'uint32'>",
        'Polygon-to-ring offsets, main then east copies.'
      ),
      makeOutput(
        'ringOffsets',
        "GraphDataView<'uint32'>",
        'Ring-to-vertex offsets; empty copies keep ring roles.'
      )
    ],
    example: makeExample('GPULonLatPolygonPreparation', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'parameters, // getGPULonLatPolygonPreparationParameterValues(...)',
      `antimeridian: 'split', poleRings: 'cap',`,
      'output'
    ]),
    preview: {
      kind: 'polygons',
      seed: 4235,
      caption: 'A polygon across the date line splits in two.'
    },
    related: ['rectangle-clip', 'local-frame-projection'],
    lensScenario: 'gpu-spatial-analysis.lon-lat-polygon-preparation',
    reference: 'api/gpu-spatial-analysis/operators-cells',
    maturity: 'stable'
  },
  {
    id: 'study-window-preparation',
    symbol: 'GPUStudyWindowPreparation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Study window preparation',
    summary:
      'Prepares one polygon as an analysis window: oriented edges, area, perimeter, bounds and mask.',
    explanation:
      'Gives point-pattern and density operators a shared boundary. It orients edges with the interior on the left and rasterizes a sampling grid for fast inside tests.',
    dataTypes: {input: ['polygons'], output: ['raster', 'table']},
    useCases: ['measure', 'model'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      POLYGON_POSITIONS,
      FEATURE_OFFSETS,
      POLYGON_OFFSETS,
      RING_OFFSETS,
      makeParam(
        'samplingGrid',
        'readonly [number, number] | undefined',
        'Columns and rows of the mask.',
        {
          default: '[256, 256]'
        }
      ),
      makeParam('crossingCapacity', 'number | undefined', 'Raster crossing capacity.', {
        default: '4 * samplingGrid[1] * ringCount'
      }),
      makeParam(
        'output',
        '{ summary?: GraphDataView<"float32"> | undefined; } | undefined',
        'Optional summary copy.'
      )
    ],
    outputs: [
      makeOutput(
        'summary',
        "GraphDataView<'float32'> | undefined",
        'Eight floats: area, perimeter, bounds and more.'
      )
    ],
    example: makeExample('GPUStudyWindowPreparation', [
      'positions, featureOffsets, polygonOffsets, ringOffsets,',
      'samplingGrid: [256, 256],',
      'output: {summary}'
    ]),
    preview: {
      kind: 'polygons',
      seed: 4236,
      caption: 'One boundary, shared by every point-pattern test.'
    },
    related: ['geometry-measures', 'shape-descriptors'],
    lensScenario: 'gpu-spatial-analysis.study-window-preparation',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  }
];
