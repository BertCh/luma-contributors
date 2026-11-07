// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: build and combine geometry (buffers, hulls, overlay, tessellation, Voronoi, line
// operations, shape generators and urban-form recipes).

import type {CatalogEntry} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------------------------------------
  // Buffers, offsets and overlay
  // ---------------------------------------------------------------------------------------------
  {
    id: 'buffer-surface',
    symbol: 'GPUBufferSurface',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Buffer surface',
    summary: 'Grows a polygon buffer of a given distance around lines and polygons.',
    explanation:
      'Best for proximity zones, setbacks and exclusion areas. Offset curves are generated per side, then merged by a nonzero polygon overlay into one clean surface. Distance is a per-frame parameter, so the buffer can animate without recompiling.',
    dataTypes: {input: ['lines', 'polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'urban'],
    params: [
      {
        name: 'geometry',
        type: 'GPUPackedGeometryPort<GPUPolygonGeometryPort | GPULineGeometryPort>',
        required: true,
        description: 'Lines or polygons to buffer.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed distance and mitre limit, rewritten per frame.'
      },
      {
        name: 'joinStyle',
        type: 'GPUOffsetCurveJoinStyle',
        default: "'round'",
        description: "Corner join: 'round', 'mitre' or 'bevel'."
      },
      {
        name: 'capStyle',
        type: 'GPUBufferCapStyle',
        default: "'round'",
        description: "Line end cap: 'round', 'flat' or 'square'."
      },
      {
        name: 'quadSegments',
        type: 'number',
        default: '8',
        description: 'Arc segments per quarter circle on round joins.'
      },
      {
        name: 'vertexTolerance',
        type: 'number',
        required: true,
        description: 'Snap tolerance for the overlay, in position units.'
      },
      {
        name: 'capacity',
        type: 'GPUPolygonOverlayCapacity',
        required: true,
        description: 'Upper bounds for intersections and noded segments.'
      },
      {
        name: 'output',
        type: 'GPUPolygonOverlayOutput',
        required: true,
        description: 'Caller-owned views that receive the buffer polygons.'
      }
    ],
    outputs: [
      {
        name: 'geometry',
        type: 'GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        description: 'Packed polygon rings of the buffer surface.'
      },
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Count, overflow and required counts for capacity checks.'
      },
      {
        name: 'boundary',
        type: 'GPUPolygonOverlayBoundaryPort',
        description: 'Boundary rings of the buffer, for downstream assembly.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUBufferSurface,
  GPUParameterBuffer,
  getGPUOffsetCurveParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'buffer'});
const parameters = new GPUParameterBuffer(device, {id: 'buffer-parameters', format: 'float32', length: 4});
// \`roads\` (packed lines) and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUBufferSurface({
  geometry: roads,
  parameters: parameters.importToGraph(graph),
  joinStyle: 'round',
  vertexTolerance: 1e-3,
  capacity: {intersections: 20000, nodedSegments: 40000},
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUOffsetCurveParameterValues({distance: 50}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'buffers',
      seed: 4101,
      caption: 'Round-joined 50 m buffers around street centerlines.'
    },
    related: ['offset-curve', 'polygon-overlay', 'outline-geometry'],
    lensScenario: 'gpu-spatial-analysis.buffer-surface',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'offset-curve',
    symbol: 'GPUOffsetCurve',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Offset curve',
    summary: 'Shifts lines or rings sideways by a signed distance, keeping them one-sided.',
    explanation:
      'Best for parallel lanes, lane edges and inset rings (Shapely `offset_curve`). Positive distances go left of the line direction, negative go right. Joins are round, mitre or bevel, and inside turns are resolved per vertex.',
    dataTypes: {input: ['lines', 'polygons'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed signed distance and mitre limit.'
      },
      {
        name: 'joinStyle',
        type: 'GPUOffsetCurveJoinStyle',
        default: "'round'",
        description: "Corner join: 'round', 'mitre' or 'bevel'."
      },
      {
        name: 'quadSegments',
        type: 'number',
        description: 'Arc segments per quarter circle on round joins.'
      },
      {
        name: 'insideTurns',
        type: 'GPUOffsetCurveInsideTurns',
        description: "Inside-turn policy: 'intersection' or 'buffer'."
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar by default; longitude-latitude is supported.'
      },
      {
        name: 'output',
        type: 'GPUOffsetCurveOutput',
        required: true,
        description: 'Caller-owned views for offset lines.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Offset points, a fixed number of rows per input vertex.'
      },
      {
        name: 'counts',
        type: "GraphDataView<'uint32'> | undefined",
        description: 'Optional distinct offset points per input vertex.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUOffsetCurve,
  GPUParameterBuffer,
  getGPUOffsetCurveParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'offset'});
const parameters = new GPUParameterBuffer(device, {id: 'offset-parameters', format: 'float32', length: 4});
// \`positions\`, \`lineOffsets\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUOffsetCurve({
  positions,
  lineOffsets,
  joinStyle: 'mitre',
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUOffsetCurveParameterValues({distance: -12, mitreLimit: 5}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'buffers', seed: 4102, caption: 'A street offset 12 m to its right.'},
    related: ['buffer-surface', 'outline-geometry', 'line-smooth'],
    lensScenario: 'gpu-spatial-analysis.offset-curve',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'outline-geometry',
    symbol: 'GPUOutlineGeometry',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Outline geometry',
    summary:
      'Builds render-only buffer outlines for points, lines and rings, one round join per vertex.',
    explanation:
      'Draws halos and buffer-like glows fast, without computing a true dissolved buffer. Each vertex gets a join disc and each segment a stroke, ready for instanced drawing. For analysis-grade buffers use the buffer surface.',
    dataTypes: {input: ['points', 'lines', 'polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      {
        name: 'geometryType',
        type: '"points" | "lines" | "polygons"',
        required: true,
        description: 'Input kind; lines need lineOffsets, polygons need ringOffsets.'
      },
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Line start rows; lines only.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Ring start rows; polygons only.'
      },
      {name: 'joinSegments', type: 'number', description: 'Segments per round join disc.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed distance, rewritten per frame.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPUOutlineGeometryOutput',
        required: true,
        description: 'Caller-owned views for outline vertices.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Outline triangle vertices, render only.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUOutlineGeometry,
  GPUParameterBuffer,
  getGPUOutlineGeometryParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'outline'});
const parameters = new GPUParameterBuffer(device, {id: 'outline-parameters', format: 'float32', length: 4});
// \`positions\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUOutlineGeometry({
  geometryType: 'points',
  positions,
  joinSegments: 16,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUOutlineGeometryParameterValues({distance: 30}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'buffers',
      seed: 4103,
      caption: 'Round halos around points, drawn as triangles.'
    },
    related: ['buffer-surface', 'offset-curve'],
    lensScenario: 'gpu-spatial-analysis.outline-geometry',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'polygon-overlay',
    symbol: 'GPUPolygonOverlay',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Polygon overlay',
    summary:
      'Intersects, unions, subtracts or dissolves polygon sets over a snap-rounded planar arrangement.',
    explanation:
      'Best for clip, merge and erase workflows on parcels, zones and footprints. Edges are noded and snapped, then faces are kept by a winding rule per operand. Capacities are explicit, and status reports when they were too small.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['relate', 'transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'left',
        type: 'PackedPolygons',
        required: true,
        description: 'First operand, packed polygons.'
      },
      {name: 'right', type: 'PackedPolygons', description: 'Second operand; omit for dissolve.'},
      {
        name: 'operation',
        type: 'GPUPolygonOverlayOperation',
        required: true,
        description: "'intersection', 'union', 'difference', 'symmetric-difference' or 'dissolve'."
      },
      {
        name: 'fillRule',
        type: 'GPUPolygonOverlayFillRule',
        default: "'even-odd'",
        description: "Inside test: 'even-odd', 'positive', 'negative' or 'nonzero'."
      },
      {
        name: 'vertexTolerance',
        type: 'number',
        required: true,
        description: 'Snap-rounding tolerance in position units.'
      },
      {
        name: 'capacity',
        type: 'GPUPolygonOverlayCapacity',
        required: true,
        description: 'Upper bounds for intersections and noded segments.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar or spherical frame of the left operand.'
      },
      {
        name: 'output',
        type: 'GPUPolygonOverlayOutput',
        required: true,
        description: 'Caller-owned views for result polygons and status.'
      }
    ],
    outputs: [
      {
        name: 'geometry',
        type: 'GPUPackedGeometryPort<GPUPolygonGeometryPort>',
        description: 'Result polygons with ring offsets and source ids.'
      },
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Count, overflow and required counts.'
      },
      {
        name: 'boundary',
        type: 'GPUPolygonOverlayBoundaryPort',
        description: 'Result boundary segments with operand ids.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPolygonOverlay} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'overlay'});
// \`parcels\`, \`floodZone\` (packed polygons) and \`output\` are graph views on \`graph\`.
graph.add(new GPUPolygonOverlay({
  left: parcels,
  right: floodZone,
  operation: 'intersection',
  vertexTolerance: 1e-4,
  capacity: {intersections: 50000, nodedSegments: 100000},
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// then read \`output.status.count\` and check \`output.status.overflow\` before using the rings`,
    preview: {
      kind: 'polygon-overlay',
      seed: 4104,
      caption: 'Parcels intersected with a flood zone.'
    },
    related: ['buffer-surface', 'coverage-dissolve', 'segment-ring-assembly'],
    lensScenario: 'gpu-spatial-analysis.polygon-overlay',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'coverage-dissolve',
    symbol: 'GPUCoverageDissolve',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Coverage dissolve',
    summary: 'Merges the polygons of a gap-free coverage by label, dropping shared interior edges.',
    explanation:
      'Rolls tracts up into districts or counties (GeoPandas `dissolve`). Shared edges cancel, and the remaining boundary segments are chained into rings. It assumes a clean coverage; validate first if unsure.',
    dataTypes: {input: ['polygons', 'table'], output: ['polygons']},
    useCases: ['summarize', 'transform'],
    domains: ['cartography', 'demographics'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed ring vertices.'
      },
      {
        name: 'featureOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Polygon range of every feature.'
      },
      {
        name: 'polygonOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Ring range of every polygon.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Vertex range of every ring.'
      },
      {
        name: 'labels',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Dissolve label of every feature.'
      },
      {
        name: 'snapTolerance',
        type: 'number',
        description: 'Vertex matching tolerance in position units.'
      },
      {
        name: 'splitTouchingRings',
        type: 'boolean',
        description: 'Split rings that touch at a single vertex.'
      },
      {
        name: 'output',
        type: 'GPUCoverageDissolveOutput',
        required: true,
        description: 'Caller-owned ring views and status.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x2'>", description: 'Closed ring vertices.'},
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex range of every merged ring.'
      },
      {
        name: 'ringIsHole',
        type: "GraphDataView<'uint32'>",
        description: 'Optional hole flag per ring.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {openSegmentCount?}',
        description: 'Ring count, overflow and boundary segments left open.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCoverageDissolve} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'dissolve'});
// Tract rings and a district \`labels\` column are graph views imported on \`graph\`.
graph.add(new GPUCoverageDissolve({
  positions,
  featureOffsets,
  polygonOffsets,
  ringOffsets,
  labels: districtIds,
  snapTolerance: 1e-5,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'polygons', seed: 4105, caption: 'Tracts dissolved into districts.'},
    related: [
      'coverage-validity',
      'coverage-simplification',
      'segment-ring-assembly',
      'polygon-overlay'
    ],
    lensScenario: 'gpu-spatial-analysis.coverage-dissolve',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'coverage-simplification',
    symbol: 'GPUCoverageSimplification',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Coverage simplification',
    summary: 'Simplifies a polygon coverage without opening gaps or overlaps between neighbors.',
    explanation:
      'Generalizes administrative boundaries for small scales (Shapely `simplify_coverage`). Shared edges are simplified once, so both neighbors move together. Tolerance is a per-frame parameter.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed ring vertices.'
      },
      {
        name: 'featureOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Polygon range of every feature.'
      },
      {
        name: 'polygonOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Ring range of every polygon.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Vertex range of every ring.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed tolerance, rewritten per frame.'
      },
      {
        name: 'simplifyBoundary',
        type: 'boolean',
        description: 'Also simplify the outer coverage boundary.'
      },
      {
        name: 'snapTolerance',
        type: 'number',
        description: 'Vertex matching tolerance in position units.'
      },
      {
        name: 'output',
        type: 'GPUCoverageSimplificationOutput',
        required: true,
        description: 'Caller-owned simplified rings.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Simplified ring vertices.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex range of every simplified ring.'
      },
      {
        name: 'outputMask',
        type: "GraphDataView<'uint32'>",
        description: 'Optional kept-vertex flag per source row.'
      },
      {
        name: 'status',
        type: '{converged?, overflow?}',
        description: 'Convergence and overflow scalars.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCoverageSimplification,
  GPUParameterBuffer,
  getGPULineSimplificationParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'simplify-coverage'});
const parameters = new GPUParameterBuffer(device, {id: 'tolerance', format: 'float32', length: 4});
// Packed polygon views and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUCoverageSimplification({
  positions, featureOffsets, polygonOffsets, ringOffsets,
  parameters: parameters.importToGraph(graph),
  simplifyBoundary: true,
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPULineSimplificationParameterValues({tolerance: 250}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygons',
      seed: 4106,
      caption: 'Counties simplified; shared borders stay shared.'
    },
    related: ['line-simplification', 'coverage-dissolve', 'coverage-validity'],
    lensScenario: 'gpu-spatial-analysis.coverage-simplification',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'coverage-validity',
    symbol: 'GPUCoverageValidity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Coverage validity',
    summary: 'Flags overlaps, gaps and mismatched edges in a polygon coverage, segment by segment.',
    explanation:
      'Checks that a boundary dataset is a clean coverage before dissolving or simplifying (Shapely `coverage_is_valid`). Each segment gets a bit mask of problems and each feature a count. A gap wider than `gapWidth` is an error.',
    dataTypes: {input: ['polygons'], output: ['lines', 'table']},
    useCases: ['transform', 'measure'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed ring vertices.'
      },
      {
        name: 'featureOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Polygon range of every feature.'
      },
      {
        name: 'polygonOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Ring range of every polygon.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Vertex range of every ring.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed gap width, rewritten per frame.'
      },
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort segments spatially; defaults on for large inputs.'
      },
      {name: 'leafCapacity', type: 'number', description: 'Segments per BVH leaf.'},
      {
        name: 'output',
        type: 'GPUCoverageValidityOutput',
        required: true,
        description: 'Caller-owned flag views.'
      }
    ],
    outputs: [
      {
        name: 'segmentFlags',
        type: "GraphDataView<'uint32'>",
        description: 'Problem bit mask per segment; 0 is valid.'
      },
      {
        name: 'featureInvalidCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Optional invalid segment count per feature.'
      },
      {
        name: 'isValid',
        type: "GraphDataView<'uint32'>",
        description: 'Optional one-row flag: 1 when the coverage is valid.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCoverageValidity,
  GPUParameterBuffer,
  getGPUCoverageValidityParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'coverage-check'});
const parameters = new GPUParameterBuffer(device, {id: 'gap-width', format: 'float32', length: 4});
// Packed polygon views and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUCoverageValidity({
  positions, featureOffsets, polygonOffsets, ringOffsets,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUCoverageValidityParameterValues({gapWidth: 0.5}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'polygons',
      seed: 4107,
      caption: 'Invalid edges of a county coverage, highlighted.'
    },
    related: ['coverage-dissolve', 'coverage-simplification', 'shared-paths'],
    lensScenario: 'gpu-spatial-analysis.coverage-validity',
    reference: 'api/gpu-spatial-analysis/operators-validity',
    maturity: 'stable'
  },
  {
    id: 'segment-ring-assembly',
    symbol: 'GPUSegmentRingAssembly',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Segment ring assembly',
    summary: 'Chains loose directed boundary segments into closed rings, with hole and area flags.',
    explanation:
      'The last step of overlays, dissolves and cell outlines, where you end up with unordered edges. Endpoints are matched within a tolerance and walked into rings. Optional outputs give winding, area and hole ownership.',
    dataTypes: {input: ['lines'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'endpoints',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Directed segments as (x0, y0, x1, y1).'
      },
      {
        name: 'count',
        type: "GraphDataView<'uint32'>",
        description: 'One-row count of live segments.'
      },
      {
        name: 'groups',
        type: "GraphDataView<'uint32'>",
        description: 'Group id per segment; rings never cross groups.'
      },
      {name: 'vertexTolerance', type: 'number', description: 'Endpoint matching tolerance.'},
      {
        name: 'cancelOpposingSegments',
        type: 'boolean',
        description: 'Drop segment pairs that run in opposite directions.'
      },
      {
        name: 'normalizeWinding',
        type: 'boolean',
        description: 'Rewind rings to a canonical orientation.'
      },
      {
        name: 'splitTouchingRings',
        type: 'boolean',
        description: 'Split rings that touch at a single vertex.'
      },
      {
        name: 'output',
        type: 'GPUSegmentRingAssemblyOutput',
        required: true,
        description: 'Caller-owned ring views and status.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Ring vertices; each ring repeats its first vertex.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex range of every ring.'
      },
      {
        name: 'ringIsHole',
        type: "GraphDataView<'uint32'>",
        description: 'Optional hole flag per ring.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {openSegmentCount?}',
        description: 'Ring count, overflow and open segment count.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSegmentRingAssembly} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'rings'});
// \`edges\` (float32x4 segments), \`segmentCount\` and \`output\` are graph views on \`graph\`.
graph.add(new GPUSegmentRingAssembly({
  endpoints: edges,
  count: segmentCount,
  vertexTolerance: 1e-6,
  cancelOpposingSegments: true,
  normalizeWinding: true,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'polygons', seed: 4108, caption: 'Loose boundary edges closed into rings.'},
    related: ['polygon-overlay', 'coverage-dissolve', 'concave-hull'],
    lensScenario: 'gpu-spatial-analysis.segment-ring-assembly',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Hulls and group geometry
  // ---------------------------------------------------------------------------------------------
  {
    id: 'concave-hull',
    symbol: 'GPUConcaveHull',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Concave hull',
    summary:
      'Wraps a point set in an alpha-shape footprint that follows its real outline, holes included.',
    explanation:
      'Best for building footprints from points, habitat extents and service areas. Delaunay triangles with a circumradius above `maximumCircumradius` are dropped, and the oriented boundary of the rest is returned. It is radius-based alpha-shape, not the ratio-based GEOS hull.',
    dataTypes: {input: ['points'], output: ['polygons']},
    useCases: ['transform', 'measure'],
    domains: ['cartography', 'ecology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Input points.'
      },
      {
        name: 'maximumCircumradius',
        type: 'number',
        required: true,
        description: 'Largest kept triangle circumradius, in position units.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar coordinates only.'},
      {
        name: 'output',
        type: 'GPUConcaveHull output',
        required: true,
        description: 'Boundary edges, triangles, their validity masks, and status.'
      }
    ],
    outputs: [
      {
        name: 'boundaryEdges',
        type: "GraphDataView<'uint32x2'>",
        description: 'Oriented hull edges as point index pairs.'
      },
      {
        name: 'edgeValidity',
        type: "GraphDataView<'uint32'>",
        description: 'Mask for real edges versus padding.'
      },
      {
        name: 'triangles',
        type: "GraphDataView<'uint32x3'>",
        description: 'Retained Delaunay triangles.'
      },
      {
        name: 'status',
        type: "{converged: GraphDataView<'uint32'>}",
        description: 'Zero if tessellation did not finish.'
      },
      {
        name: 'triangleValidity',
        type: "GraphDataView<'uint32'>",
        description: '1 for each triangle kept in the hull.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUConcaveHull} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'concave-hull'});
// \`positions\` and the four output views are graph views imported on \`graph\`.
graph.add(new GPUConcaveHull({
  positions,
  maximumCircumradius: 150,
  output: {boundaryEdges, edgeValidity, triangles, triangleValidity, status: {converged}}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// feed \`boundaryEdges\` to GPUSegmentRingAssembly for polygon rings`,
    preview: {
      kind: 'point-clusters',
      seed: 4109,
      caption: 'A concave footprint around scattered points.'
    },
    related: ['group-concave-hull', 'group-convex-hull', 'delaunay-tessellation'],
    lensScenario: 'gpu-spatial-analysis.concave-hull',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'group-concave-hull',
    symbol: 'GPUGroupConcaveHull',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Group concave hull',
    summary: 'Builds one alpha-shape hull per cluster label, all in a single GPU pass.',
    explanation:
      'Outlines clusters from DBSCAN-style labels, or any group id column. Each group gets its own Delaunay complex, so hulls never bridge between groups. Boundaries are oriented edges ready for ring assembly.',
    dataTypes: {input: ['points', 'table'], output: ['polygons']},
    useCases: ['find-patterns', 'transform'],
    domains: ['spatial-statistics', 'cartography'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Input points.'
      },
      {
        name: 'groupIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Group id of every point.'
      },
      {name: 'groupCount', type: 'number', required: true, description: 'Number of groups.'},
      {
        name: 'maximumCircumradius',
        type: 'number',
        required: true,
        description: 'Largest kept triangle circumradius, in position units.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar coordinates only.'},
      {
        name: 'output',
        type: 'GPUGroupConcaveHull output',
        required: true,
        description: 'Boundary edges, triangles, validity masks and status.'
      }
    ],
    outputs: [
      {
        name: 'boundaryEdges',
        type: "GraphDataView<'uint32x2'>",
        description: 'Oriented hull edges across all groups.'
      },
      {
        name: 'edgeValidity',
        type: "GraphDataView<'uint32'>",
        description: 'Mask for real edges versus padding.'
      },
      {
        name: 'triangles',
        type: "GraphDataView<'uint32x3'>",
        description: 'Retained triangles across groups.'
      },
      {
        name: 'triangleValidity',
        type: "GraphDataView<'uint32'>",
        description: '1 for each triangle kept in the hull.'
      },
      {
        name: 'status',
        type: '{converged}',
        description: 'One row; 0 if tessellation did not finish.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUGroupConcaveHull} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'cluster-hulls'});
// \`positions\`, \`labels\` and the output views are graph views imported on \`graph\`.
graph.add(new GPUGroupConcaveHull({
  positions,
  groupIds: labels,
  groupCount: 12,
  maximumCircumradius: 80,
  output: {boundaryEdges, edgeValidity, triangles, triangleValidity, status: {converged}}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'point-clusters', seed: 4110, caption: 'One concave hull per cluster.'},
    related: ['concave-hull', 'group-convex-hull', 'group-geometry'],
    lensScenario: 'gpu-spatial-analysis.group-concave-hull',
    reference: 'api/gpu-raster/expansion',
    maturity: 'stable'
  },
  {
    id: 'group-convex-hull',
    symbol: 'GPUGroupConvexHull',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Group convex hull',
    summary:
      'Finds the convex hull of every labeled group of points, with exact orientation tests.',
    explanation:
      "Best for quick cluster outlines, extents and minimum bounds. It runs Andrew's monotone chain per label after an optional extreme-point prefilter. Noise labels can be skipped.",
    dataTypes: {input: ['points', 'table'], output: ['polygons']},
    useCases: ['find-patterns', 'measure'],
    domains: ['spatial-statistics', 'cartography'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Input points.'
      },
      {
        name: 'labels',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Group label of every point.'
      },
      {name: 'groupCount', type: 'number', required: true, description: 'Number of groups.'},
      {name: 'noiseLabel', type: 'number', description: 'Label to skip, such as DBSCAN noise.'},
      {
        name: 'maximumVerticesPerGroup',
        type: 'number',
        required: true,
        description: 'Hull vertex cap per group.'
      },
      {
        name: 'vertexCapacity',
        type: 'number',
        required: true,
        description: 'Total hull vertex capacity.'
      },
      {name: 'prefilterLevels', type: 'number', description: 'Extreme-point prefilter depth.'},
      {
        name: 'output',
        type: 'GPUGroupConvexHullOutput',
        required: true,
        description: 'Caller-owned hull views.'
      }
    ],
    outputs: [
      {
        name: 'vertexIndices',
        type: "GraphDataView<'uint32'>",
        description: 'Hull vertices as source point indices.'
      },
      {
        name: 'vertexPositions',
        type: "GraphDataView<'float32x2'>",
        description: 'Optional hull vertex coordinates.'
      },
      {name: 'offsets', type: "GraphDataView<'uint32'>", description: 'Start of every group hull.'},
      {
        name: 'counts',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex count per group hull.'
      },
      {
        name: 'status',
        type: '{overflow}',
        description: 'Nonzero when the output capacity was too small.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUGroupConvexHull} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'convex-hulls'});
// \`positions\`, \`labels\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUGroupConvexHull({
  positions,
  labels,
  groupCount: 12,
  noiseLabel: 0xffffffff,
  maximumVerticesPerGroup: 64,
  vertexCapacity: 12 * 64,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'point-clusters', seed: 4111, caption: 'Convex hulls of labeled clusters.'},
    related: ['group-concave-hull', 'group-geometry', 'concave-hull'],
    lensScenario: 'gpu-spatial-analysis.group-convex-hull',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'group-geometry',
    symbol: 'GPUGroupGeometry',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Group geometry',
    summary:
      'Summarizes each labeled point group: bounds, centers, medoid and standard-distance ellipse.',
    explanation:
      'Describes clusters: where they are, how big and which way they lean. All requested summaries come from one pass over the label column. Weights give weighted centers.',
    dataTypes: {input: ['points', 'table'], output: ['table', 'points']},
    useCases: ['measure', 'summarize'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Input points.'
      },
      {
        name: 'labels',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Group label of every point.'
      },
      {name: 'groupCount', type: 'number', required: true, description: 'Number of groups.'},
      {
        name: 'weights',
        type: "GraphDataView<'float32'>",
        description: 'Point weights for weighted centers.'
      },
      {name: 'noiseLabel', type: 'number', description: 'Label to skip, such as DBSCAN noise.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed per-frame settings.'
      },
      {
        name: 'medoidMaximumGroupSize',
        type: 'number',
        description: 'Largest group for exact medoids.'
      },
      {
        name: 'output',
        type: 'GPUGroupGeometryOutput',
        required: true,
        description: 'Choose the summaries to compute.'
      }
    ],
    outputs: [
      {name: 'counts', type: "GraphDataView<'uint32'>", description: 'Points per group.'},
      {name: 'bounds', type: "GraphDataView<'float32x4'>", description: 'Bounding box per group.'},
      {
        name: 'meanCenters',
        type: "GraphDataView<'float32x2'>",
        description: 'Mean center per group.'
      },
      {
        name: 'medoidIndices',
        type: "GraphDataView<'uint32'>",
        description: 'Point index of each group medoid.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUGroupGeometry, GPUParameterBuffer} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'group-geometry'});
const parameters = new GPUParameterBuffer(device, {id: 'group-parameters', format: 'float32', length: 4});
// \`positions\`, \`labels\` and the output views are graph views imported on \`graph\`.
graph.add(new GPUGroupGeometry({
  positions,
  labels,
  groupCount: 12,
  parameters: parameters.importToGraph(graph),
  output: {counts, bounds, meanCenters, medoidIndices}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'point-clusters',
      seed: 4112,
      caption: 'Bounds, centers and medoids of three groups.'
    },
    related: ['group-convex-hull', 'group-concave-hull', 'dorling-cartogram'],
    lensScenario: 'gpu-spatial-analysis.group-geometry',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Tessellation, triangulation and Voronoi
  // ---------------------------------------------------------------------------------------------
  {
    id: 'delaunay-tessellation',
    symbol: 'GPUDelaunayTessellation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Delaunay tessellation',
    summary: 'Triangulates a point set into a Delaunay mesh on the GPU, optionally per group.',
    explanation:
      'Best for neighbor graphs, surface meshes and as the base of Voronoi and alpha hulls. Triangles are built by parallel insertion and flipping, with exact predicates. Groups triangulate independently in one pass.',
    dataTypes: {input: ['points'], output: ['polygons', 'spatial-weights']},
    useCases: ['transform', 'relate'],
    domains: ['cartography', 'spatial-statistics'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Input sites.'
      },
      {
        name: 'groups',
        type: '{ids: GraphDataView<"uint32">; count: number}',
        description: 'Triangulate each group independently.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPUDelaunayTessellationOutput',
        required: true,
        description: 'Caller-owned triangle views and status.'
      }
    ],
    outputs: [
      {
        name: 'triangles',
        type: "GraphDataView<'uint32x3'>",
        description: 'Triangle corners as site indices.'
      },
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Triangle count, overflow and required count.'
      },
      {
        name: 'groups',
        type: '{triangleCounts, triangleOffsets?} | undefined',
        description: 'Per-group triangle counts, when grouped.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDelaunayTessellation} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'delaunay'});
// \`sites\` and \`output\` (triangles, status) are graph views imported on \`graph\`.
graph.add(new GPUDelaunayTessellation({
  positions: sites,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// \`output.triangles\` is a uint32x3 index buffer, ready for a mesh or Voronoi`,
    preview: {kind: 'tessellation', seed: 4113, caption: 'Delaunay triangles over random sites.'},
    related: ['voronoi-diagram', 'constrained-delaunay-triangulation', 'concave-hull'],
    lensScenario: 'gpu-spatial-analysis.delaunay-tessellation',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'voronoi-diagram',
    symbol: 'GPUVoronoiDiagram',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Voronoi diagram',
    summary:
      'Derives clipped Voronoi cell edges from a Delaunay triangulation, entirely on the GPU.',
    explanation:
      'Best for service areas, nearest-site regions and proximity partitions. Circumcenters of adjacent triangles become edges, clipped to `clipBounds`. Each edge reports the pair of sites it separates.',
    dataTypes: {input: ['points'], output: ['lines', 'polygons']},
    useCases: ['transform', 'route'],
    domains: ['cartography', 'urban'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'The sites.'
      },
      {
        name: 'triangles',
        type: "GraphDataView<'uint32x3'>",
        required: true,
        description: 'Delaunay triangles over the sites.'
      },
      {
        name: 'triangleCount',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'One-row count of live triangles.'
      },
      {
        name: 'clipBounds',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Clip rectangle as (minX, minY, maxX, maxY).'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPUVoronoiDiagramOutput',
        required: true,
        description: 'Caller-owned edge views and status.'
      }
    ],
    outputs: [
      {
        name: 'segments',
        type: "GraphDataView<'float32x4'>",
        description: 'Voronoi edges as (x0, y0, x1, y1).'
      },
      {
        name: 'siteIds',
        type: "GraphDataView<'uint32x2'>",
        description: 'The two sites each edge separates.'
      },
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Edge count, overflow and required count.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDelaunayTessellation, GPUVoronoiDiagram} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'voronoi'});
// \`sites\`, \`clipBounds\`, \`delaunay\` and \`voronoi\` output views live on \`graph\`.
graph.add(new GPUDelaunayTessellation({positions: sites, output: delaunay}));
graph.add(new GPUVoronoiDiagram({
  positions: sites,
  triangles: delaunay.triangles,
  triangleCount: delaunay.status.count,
  clipBounds,
  output: voronoi
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'tessellation', seed: 4114, caption: 'Voronoi cells around 40 sites.'},
    related: ['delaunay-tessellation', 'building-tessellation-recipe', 'dorling-cartogram'],
    lensScenario: 'gpu-spatial-analysis.voronoi-diagram',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'polygon-triangulation',
    symbol: 'GPUPolygonTriangulation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Polygon triangulation',
    summary: 'Triangulates polygons with holes into index buffers on the GPU, the Earcut way.',
    explanation:
      'Renders large polygon sets without CPU tessellation. Every polygon gets its own triangle slice, with a validity flag. Large polygons can use a monotone method to bound the work.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'three-d'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed ring vertices.'
      },
      {
        name: 'featureOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Polygon range of every feature.'
      },
      {
        name: 'polygonOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Ring range of every polygon.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Vertex range of every ring.'
      },
      {
        name: 'maximumWork',
        type: 'number',
        description: 'Per-polygon work cap before it is flagged invalid.'
      },
      {
        name: 'useZOrderHash',
        type: 'boolean',
        description: 'Speed up ear tests on big rings with a z-order hash.'
      },
      {
        name: 'largePolygonMethod',
        type: '"none" | "monotone"',
        description: 'Strategy for very large polygons.'
      },
      {
        name: 'output',
        type: 'GPUPolygonTriangulationOutput',
        required: true,
        description: 'Caller-owned index views.'
      }
    ],
    outputs: [
      {
        name: 'indices',
        type: "GraphDataView<'uint32'>",
        description: 'Triangle indices into positions, per polygon slice.'
      },
      {
        name: 'valid',
        type: "GraphDataView<'uint32'>",
        description: 'Per polygon: 1 when triangulation completed.'
      },
      {
        name: 'triangleOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Optional triangle slice offsets per polygon.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUPolygonTriangulation} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'triangulate'});
// Packed polygon views and \`output\` (indices, valid) are graph views on \`graph\`.
graph.add(new GPUPolygonTriangulation({
  positions,
  featureOffsets,
  polygonOffsets,
  ringOffsets,
  largePolygonMethod: 'monotone',
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// draw \`output.indices\` with the same \`positions\` as vertex buffer`,
    preview: {
      kind: 'tessellation',
      seed: 4115,
      caption: 'Polygons with holes, split into triangles.'
    },
    related: ['constrained-delaunay-triangulation', 'delaunay-tessellation'],
    lensScenario: 'gpu-spatial-analysis.polygon-triangulation',
    reference: 'api/gpu-spatial-analysis/operators-density',
    maturity: 'stable'
  },
  {
    id: 'constrained-delaunay-triangulation',
    symbol: 'GPUConstrainedDelaunayTriangulation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Constrained Delaunay triangulation',
    summary: 'Triangulates polygons, then flips shared edges toward the Delaunay criterion.',
    explanation:
      'Best when triangle quality matters: meshes for simulation, interpolation or smooth shading. Shell and hole edges are never flipped, so they stay in the mesh. Flips are bounded by `maximumFlips`.',
    dataTypes: {input: ['polygons'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'three-d'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed ring vertices.'
      },
      {
        name: 'featureOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Polygon range of every feature.'
      },
      {
        name: 'polygonOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Ring range of every polygon.'
      },
      {
        name: 'ringOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Vertex range of every ring.'
      },
      {name: 'maximumFlips', type: 'number', description: 'Upper bound on Lawson flip rounds.'},
      {
        name: 'maximumWork',
        type: 'number',
        description: 'Per-polygon work cap before it is flagged invalid.'
      },
      {
        name: 'largePolygonMethod',
        type: '"none" | "monotone"',
        description: 'Strategy for very large polygons.'
      },
      {
        name: 'output',
        type: 'GPUPolygonTriangulationOutput & {status}',
        required: true,
        description: 'Triangle views plus a converged flag.'
      }
    ],
    outputs: [
      {
        name: 'indices',
        type: "GraphDataView<'uint32'>",
        description: 'Triangle indices into positions, per polygon slice.'
      },
      {
        name: 'valid',
        type: "GraphDataView<'uint32'>",
        description: 'Per polygon: 1 when triangulation completed.'
      },
      {
        name: 'status',
        type: "{converged: GraphDataView<'uint32'>}",
        description: 'Whether flipping converged within maximumFlips.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUConstrainedDelaunayTriangulation} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'cdt'});
// Packed polygon views and \`output\` (indices, valid, status) are graph views on \`graph\`.
graph.add(new GPUConstrainedDelaunayTriangulation({
  positions,
  featureOffsets,
  polygonOffsets,
  ringOffsets,
  maximumFlips: 64,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'tessellation', seed: 4116, caption: 'A polygon mesh after Delaunay flips.'},
    related: ['polygon-triangulation', 'delaunay-tessellation'],
    lensScenario: 'gpu-spatial-analysis.constrained-delaunay-triangulation',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Line operations
  // ---------------------------------------------------------------------------------------------
  {
    id: 'line-segmentize',
    symbol: 'GPULineSegmentize',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line segmentize',
    summary: 'Densifies lines so no segment is longer than a maximum length.',
    explanation:
      'Best before reprojecting, snapping or sampling along lines (PostGIS `ST_Segmentize`). Original vertices are kept and long segments are cut into equal pieces. In a longitude-latitude context, new points follow the great circle.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed maximum segment length.'
      },
      {
        name: 'closedLines',
        type: 'boolean',
        default: 'false',
        description: 'Treat lines as rings; densify the closing edge.'
      },
      {name: 'maximumPieceCount', type: 'number', description: 'Cap on pieces per input segment.'},
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar by default; great-circle for lon/lat.'
      },
      {
        name: 'output',
        type: 'GPULineGeometryOutput',
        required: true,
        description: 'Caller-owned line views.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x2'>", description: 'Densified vertices.'},
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Row range of every output line.'
      },
      {
        name: 'measures',
        type: "GraphDataView<'float32'>",
        description: 'Optional distance along the source line per vertex.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {lineCount?}',
        description: 'Written vertex count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineSegmentize,
  GPUParameterBuffer,
  getGPULineSegmentizeParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'segmentize'});
const parameters = new GPUParameterBuffer(device, {id: 'segmentize-parameters', format: 'float32', length: 4});
// \`positions\`, \`lineOffsets\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineSegmentize({
  positions,
  lineOffsets,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: 25}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4117, caption: 'Streets densified to 25 m segments.'},
    related: ['line-chunk', 'line-smooth', 'great-circle-arcs'],
    lensScenario: 'gpu-spatial-analysis.line-segmentize',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'great-circle-arcs',
    symbol: 'GPUGreatCircleArcs',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Great-circle arcs',
    summary: 'Tessellates a great-circle arc for every origin and destination pair.',
    explanation:
      'Best for flight paths and flow maps on a globe (turf `greatCircle`). Each pair becomes one line with equal-angle vertices and exact endpoints. Longitudes are unwrapped, never split at the antimeridian.',
    dataTypes: {input: ['points'], output: ['lines', 'network']},
    useCases: ['transform'],
    domains: ['transport', 'cartography'],
    params: [
      {
        name: 'sources',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Origin longitude and latitude per arc.'
      },
      {
        name: 'targets',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Destination longitude and latitude per arc.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed segment length and minimum segments.'
      },
      {name: 'maximumSegmentCount', type: 'number', description: 'Cap on segments per arc.'},
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Great-circle by default; planar throws.'
      },
      {
        name: 'output',
        type: 'GPULineGeometryOutput',
        required: true,
        description: 'Caller-owned line views.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Arc vertices, segments + 1 per pair.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Row range of every arc.'
      },
      {name: 'status', type: 'GPUBoundedResultStatusPort', description: 'Count and overflow.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGreatCircleArcs,
  GPUParameterBuffer,
  getGPUGreatCircleArcsParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'arcs'});
const parameters = new GPUParameterBuffer(device, {id: 'arc-parameters', format: 'float32', length: 4});
// \`origins\`, \`destinations\` (lon/lat) and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUGreatCircleArcs({
  sources: origins,
  targets: destinations,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 32}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'flows', seed: 4118, caption: 'Arcs from one hub to 40 destinations.'},
    related: ['line-segmentize', 'line-chunk'],
    lensScenario: 'gpu-spatial-analysis.great-circle-arcs',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-smooth',
    symbol: 'GPULineSmooth',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line smooth',
    summary: 'Rounds the corners of lines or rings by Chaikin corner cutting.',
    explanation:
      'Softens digitized tracks, contours and generalized boundaries (turf `polygonSmooth`). Each pass replaces every edge with two points at the cut ratio, so vertex counts double. Planar only.',
    dataTypes: {input: ['lines', 'polygons'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'iterationCount',
        type: 'number',
        required: true,
        description: 'Cutting passes, 1 to 10.'
      },
      {name: 'closed', type: 'boolean', description: 'Treat lines as rings.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed cut ratio, rewritten per frame.'
      },
      {
        name: 'output',
        type: 'GPULineGeometryOutput',
        required: true,
        description: 'Caller-owned line views.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x2'>", description: 'Smoothed vertices.'},
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Row range of every smoothed line.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {lineCount?}',
        description: 'Written vertex count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineSmooth,
  GPUParameterBuffer,
  getGPULineSmoothParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'smooth'});
const parameters = new GPUParameterBuffer(device, {id: 'smooth-parameters', format: 'float32', length: 4});
// \`positions\`, \`lineOffsets\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineSmooth({
  positions,
  lineOffsets,
  iterationCount: 2,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPULineSmoothParameterValues({ratio: 0.25}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4119, caption: 'Two Chaikin rounds on angular streets.'},
    related: ['line-simplification', 'line-segmentize', 'offset-curve'],
    lensScenario: 'gpu-spatial-analysis.line-smooth',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-chunk',
    symbol: 'GPULineChunk',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line chunk',
    summary: 'Splits lines into equal-length pieces, or cuts a measure range from every line.',
    explanation:
      'Best for fixed-length road segments, stationing and extracting sub-lines (turf `lineChunk`, PostGIS `ST_LineSubstring`). Mode `chunk` yields pieces of `chunkLength`; `substring` keeps the range between a start and end measure.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform', 'measure'],
    domains: ['transport', 'cartography'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {name: 'mode', type: 'LinePieceMode', required: true, description: "'chunk' or 'substring'."},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed chunk length or measure range.'
      },
      {
        name: 'spatialContext',
        type: 'GPUSpatialContext',
        description: 'Planar by default; great-circle for lon/lat.'
      },
      {
        name: 'output',
        type: 'GPULineGeometryOutput',
        required: true,
        description: 'Line views, status and source line indices.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x2'>", description: 'Piece vertices.'},
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Row range of every piece.'
      },
      {
        name: 'sourceLineIndices',
        type: "GraphDataView<'uint32'>",
        description: 'Input line of every piece.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort & {lineCount?}',
        description: 'Written vertex count and overflow flag.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineChunk,
  GPUParameterBuffer,
  getGPULineChunkParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'chunk'});
const parameters = new GPUParameterBuffer(device, {id: 'chunk-parameters', format: 'float32', length: 4});
// \`positions\`, \`lineOffsets\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineChunk({
  positions,
  lineOffsets,
  mode: 'chunk',
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPULineChunkParameterValues({chunkLength: 100}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4120, caption: 'Streets cut into 100 m chunks.'},
    related: ['line-segmentize', 'line-locate', 'linear-referencing'],
    lensScenario: 'gpu-spatial-analysis.line-chunk',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-split',
    symbol: 'GPULineSplit',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line split',
    summary: 'Splits lines at every crossing, touch or overlap with each other or themselves.',
    explanation:
      'Nodes a street layer so each piece runs between intersections (turf `lineSplit`, PostGIS `ST_Node`). It is the first step of building a routable network from raw centerlines. Capacity must cover the intersection count.',
    dataTypes: {input: ['lines'], output: ['lines', 'network']},
    useCases: ['transform', 'relate'],
    domains: ['transport', 'data-engineering'],
    params: [
      {
        name: 'lines',
        type: 'GPULineGeometryPort',
        required: true,
        description: 'Packed positions, line offsets and ids.'
      },
      {
        name: 'intersectionCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on found intersections.'
      },
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort segments spatially; speeds large inputs.'
      },
      {name: 'leafCapacity', type: 'number', description: 'Segments per BVH leaf.'},
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPULineSplitOutput',
        required: true,
        description: 'Caller-owned piece views and status.'
      }
    ],
    outputs: [
      {
        name: 'pieces',
        type: 'GPULineSplitPieces',
        description: 'Split lines with source ids and bounded-result status.'
      },
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Optional counts, overflow and uncertain pairs.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULineSplit} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'split'});
// \`streets\` (a packed line port) and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineSplit({
  lines: streets,
  intersectionCapacity: 50000,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// \`output.pieces.geometry\` holds the noded lines, \`pieces.sourceIds\` the origin line`,
    preview: {kind: 'lines', seed: 4121, caption: 'Streets noded at every intersection.'},
    related: ['line-merge', 'shared-paths', 'line-clip-by-polygon'],
    lensScenario: 'gpu-spatial-analysis.line-split',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-merge',
    symbol: 'GPULineMerge',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line merge',
    summary: 'Joins lines that share endpoints into maximal chains.',
    explanation:
      'Reassembles a road or river from segments (Shapely `linemerge`, turf `lineMerge`). Chains stop at junctions where three or more lines meet. Set `directed` to respect line direction.',
    dataTypes: {input: ['lines'], output: ['lines']},
    useCases: ['transform'],
    domains: ['transport', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'directed',
        type: 'boolean',
        description: 'Only join head to tail along line direction.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPULineMergeOutput',
        required: true,
        description: 'Caller-owned chain views and status.'
      }
    ],
    outputs: [
      {name: 'positions', type: "GraphDataView<'float32x2'>", description: 'Chain vertices.'},
      {
        name: 'chainOffsets',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex range of every chain.'
      },
      {
        name: 'lineChains',
        type: "GraphDataView<'uint32'>",
        description: 'Optional chain index of every input line.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort',
        description: '`count` is the number of merged chains.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULineMerge} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'merge'});
// \`positions\`, \`lineOffsets\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineMerge({
  positions,
  lineOffsets,
  directed: false,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4122, caption: 'Segments joined into continuous chains.'},
    related: ['line-split', 'line-simplification', 'shared-paths'],
    lensScenario: 'gpu-spatial-analysis.line-merge',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-simplification',
    symbol: 'GPULineSimplification',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line simplification',
    summary: 'Simplifies many polylines or tracks with Douglas-Peucker, TD-TR or Visvalingam.',
    explanation:
      'Thins tracks and boundaries for drawing or storage. It scores every vertex by importance once, then keeps the vertices above the tolerance, so changing the tolerance is cheap. With timestamps and the `time-ratio` metric it becomes TD-TR for trajectories.',
    dataTypes: {input: ['lines', 'trajectories'], output: ['lines']},
    useCases: ['transform'],
    domains: ['cartography', 'transport'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed vertices, sorted by line.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'importance',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Per-vertex importance, filled by the operator.'
      },
      {
        name: 'method',
        type: 'GPULineSimplificationMethod',
        default: "'douglas-peucker'",
        description: "'douglas-peucker' or 'visvalingam'."
      },
      {
        name: 'metric',
        type: 'GPULineSimplificationMetric',
        default: "'segment'",
        description: "'segment' or 'time-ratio' (TD-TR)."
      },
      {
        name: 'timestamps',
        type: "GraphDataView<'float32'>",
        description: 'Vertex times, for the time-ratio metric.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        description: 'Packed tolerance, rewritten per frame.'
      },
      {
        name: 'output',
        type: 'GPULineSimplificationOutput',
        description: 'Kept vertex ids, mask and per-line counts.'
      }
    ],
    outputs: [
      {
        name: 'ids',
        type: "GraphDataView<'uint32'>",
        description: 'Kept vertex rows in line order.'
      },
      {
        name: 'outputMask',
        type: "GraphDataView<'uint32'>",
        description: 'Optional 1/0 kept flag per vertex.'
      },
      {
        name: 'lineCounts',
        type: "GraphDataView<'uint32'>",
        description: 'Optional kept vertices per line.'
      },
      {
        name: 'status',
        type: 'GPULineSimplificationStatus & GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Kept-row count, overflow and convergence diagnostics.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineSimplification,
  GPUParameterBuffer,
  getGPULineSimplificationParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'simplify'});
const parameters = new GPUParameterBuffer(device, {id: 'tolerance', format: 'float32', length: 4});
// \`positions\`, \`lineOffsets\`, \`importance\` and \`output\` are graph views on \`graph\`.
graph.add(new GPULineSimplification({
  positions, lineOffsets, importance,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPULineSimplificationParameterValues({tolerance: 15}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4123, caption: 'A GPS track thinned at 15 m tolerance.'},
    related: ['coverage-simplification', 'line-smooth', 'line-segmentize'],
    lensScenario: 'gpu-spatial-analysis.line-simplification',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-clip-by-polygon',
    symbol: 'GPULineClipByPolygon',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line clip by polygon',
    summary: 'Clips lines to the inside or outside of polygons.',
    explanation:
      'Cuts roads by a district, or remove stretches inside water (GeoPandas `clip`). Lines are split at polygon edges and each piece is kept by its side. Pieces keep their source line id.',
    dataTypes: {input: ['lines', 'polygons'], output: ['lines']},
    useCases: ['transform', 'relate'],
    domains: ['cartography', 'transport'],
    params: [
      {
        name: 'lines',
        type: 'GPULineGeometryPort',
        required: true,
        description: 'Packed lines to clip.'
      },
      {
        name: 'polygons',
        type: 'GPUPolygonGeometryPort',
        required: true,
        description: 'Packed clip polygons.'
      },
      {
        name: 'mode',
        type: '"inside" | "outside"',
        default: "'inside'",
        description: 'Which side to keep.'
      },
      {
        name: 'intersectionCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on found intersections.'
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on candidate segment pairs.'
      },
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort segments spatially; speeds large inputs.'
      },
      {
        name: 'output',
        type: 'GPULineClipByPolygonOutput',
        required: true,
        description: 'Caller-owned piece views and status.'
      }
    ],
    outputs: [
      {name: 'pieces', type: 'GPULineSplitPieces', description: 'Clipped lines with source ids.'},
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'Optional counts, overflow and uncertain pairs.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULineClipByPolygon} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'clip-lines'});
// \`roads\`, \`district\` (packed ports) and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineClipByPolygon({
  lines: roads,
  polygons: district,
  mode: 'inside',
  intersectionCapacity: 20000,
  candidateCapacity: 200000,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4124, caption: 'Streets clipped to a district.'},
    related: ['line-split', 'polygon-overlay', 'shared-paths'],
    lensScenario: 'gpu-spatial-analysis.line-clip-by-polygon',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'shared-paths',
    symbol: 'GPUSharedPaths',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Shared paths',
    summary: 'Finds the stretches that two sets of lines have in common.',
    explanation:
      'Finds shared walls between buildings, duplicate road geometry or overlapping routes (Shapely `shared_paths`). Each run reports both source lines and its direction on each. Results feed adjacency analyses.',
    dataTypes: {input: ['lines'], output: ['lines', 'table']},
    useCases: ['relate'],
    domains: ['urban', 'data-engineering'],
    params: [
      {
        name: 'left',
        type: 'GPULineGeometryPort',
        required: true,
        description: 'First packed line set.'
      },
      {
        name: 'right',
        type: 'GPULineGeometryPort',
        required: true,
        description: 'Second packed line set.'
      },
      {
        name: 'intersectionCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on found intersections.'
      },
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort segments spatially; speeds large inputs.'
      },
      {name: 'leafCapacity', type: 'number', description: 'Segments per BVH leaf.'},
      {
        name: 'output',
        type: 'GPUSharedPathsOutput',
        required: true,
        description: 'Caller-owned run views and status.'
      }
    ],
    outputs: [
      {
        name: 'runs',
        type: 'GPUSharedPathsRuns',
        description: 'Shared runs with source lines on both sides.'
      },
      {
        name: 'status',
        type: '{count}',
        description: 'Run count, overflow and required counts.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSharedPaths} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'shared-paths'});
// \`wallsA\`, \`wallsB\` (packed line ports) and \`output\` are graph views on \`graph\`.
graph.add(new GPUSharedPaths({
  left: wallsA,
  right: wallsB,
  intersectionCapacity: 20000,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// \`output.runs\` lists each shared stretch; \`output.status.count\` says how many`,
    preview: {
      kind: 'lines',
      seed: 4125,
      caption: 'Party walls shared between neighboring buildings.'
    },
    related: ['building-adjacency-recipe', 'line-split', 'coverage-validity'],
    lensScenario: 'gpu-spatial-analysis.shared-paths',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'linear-referencing',
    symbol: 'GPULinearReferencing',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Linear referencing',
    summary: 'Snaps points to the nearest line and reports line, measure, offset and side.',
    explanation:
      'Places crashes, sensors or addresses along a road network (PostGIS `ST_LineLocatePoint`). Each query finds the nearest line within its radius using a BVH. Outputs include the foot point and signed offset.',
    dataTypes: {input: ['points', 'lines'], output: ['table', 'points']},
    useCases: ['relate', 'measure'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed line vertices.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'queryPositions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Points to snap.'
      },
      {
        name: 'radius',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Search radius per query point.'
      },
      {
        name: 'candidateCapacity',
        type: 'number',
        description: 'Candidate segments kept per query.'
      },
      {
        name: 'spatialSort',
        type: 'boolean',
        description: 'Sort segments spatially; speeds large inputs.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPULinearReferencingOutput',
        required: true,
        description: 'Choose the columns to compute.'
      }
    ],
    outputs: [
      {
        name: 'lineIndices',
        type: "GraphDataView<'uint32'>",
        description: 'Nearest line per query.'
      },
      {
        name: 'measures',
        type: "GraphDataView<'float32'>",
        description: 'Distance along the line to the foot point.'
      },
      {
        name: 'footPoints',
        type: "GraphDataView<'float32x2'>",
        description: 'Snapped point on the line.'
      },
      {
        name: 'signedOffsets',
        type: "GraphDataView<'float32'>",
        description: 'Signed distance from the line; left is positive.'
      },
      {
        name: 'status',
        type: '{overflow, candidateCount?}',
        description: 'Whether the nearest-line search was complete.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPULinearReferencing} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'snap'});
// Road views, \`crashes\`, \`radii\` and the output columns are graph views on \`graph\`.
graph.add(new GPULinearReferencing({
  positions,
  lineOffsets,
  queryPositions: crashes,
  radius: radii,
  output: {lineIndices, measures, footPoints, signedOffsets, status}
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'lines', seed: 4126, caption: 'Points snapped to the nearest street.'},
    related: ['line-locate', 'line-chunk'],
    lensScenario: 'gpu-spatial-analysis.linear-referencing',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },
  {
    id: 'line-locate',
    symbol: 'GPULineLocate',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Line locate',
    summary: 'Places events at a measure along their line and returns position and heading.',
    explanation:
      'Animates vehicles or place markers by distance or fraction along a route (turf `along`, PostGIS `ST_LineInterpolatePoint`). Measure scale and offset are per-frame parameters, so playback never recompiles. Tangent and angle help orient icons.',
    dataTypes: {input: ['lines', 'table'], output: ['points']},
    useCases: ['relate', 'simulate'],
    domains: ['transport', 'urban'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Packed line vertices.'
      },
      {
        name: 'lineOffsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every line plus a terminal row.'
      },
      {
        name: 'eventLineIndices',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Line of every event.'
      },
      {
        name: 'eventMeasures',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Measure of every event along its line.'
      },
      {
        name: 'eventOffsets',
        type: "GraphDataView<'float32'>",
        description: 'Sideways offset of every event.'
      },
      {
        name: 'measureMode',
        type: '"distance" | "fraction"',
        description: 'Measures are distances or 0 to 1 fractions.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        description: 'Packed measure scale and offset.'
      },
      {
        name: 'output',
        type: 'GPULineLocateOutput',
        required: true,
        description: 'Caller-owned event views.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Located point of every event.'
      },
      {
        name: 'tangents',
        type: "GraphDataView<'float32x2'>",
        description: 'Optional unit direction at the point.'
      },
      {
        name: 'angles',
        type: "GraphDataView<'float32'>",
        description: 'Optional heading angle at the point.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineLocate,
  GPUParameterBuffer,
  getGPULineLocateParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'locate'});
const parameters = new GPUParameterBuffer(device, {id: 'locate-parameters', format: 'float32', length: 4});
// Route views, event columns and \`output\` are graph views imported on \`graph\`.
graph.add(new GPULineLocate({
  positions, lineOffsets, eventLineIndices, eventMeasures,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
// every frame: advance the vehicles by 12 m
parameters.write(getGPULineLocateParameterValues({measureOffset: frame * 12}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'trajectories', seed: 4127, caption: 'Markers advancing along routes.'},
    related: ['linear-referencing', 'line-chunk'],
    lensScenario: 'gpu-spatial-analysis.line-locate',
    reference: 'api/gpu-spatial-analysis/operators-lines',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Generators, cartograms and coloring
  // ---------------------------------------------------------------------------------------------
  {
    id: 'grid-generator',
    symbol: 'GPUGridGenerator',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Grid generator',
    summary: 'Generates square, hexagon, triangle or point grids from an origin and cell size.',
    explanation:
      'Best for fishnets, hex bins and sampling lattices (turf `squareGrid`, GeoPandas `make_grid`). Origin and cell size are per-frame parameters, so a grid can pan and zoom without recompiling. Optionally clip to an extent.',
    dataTypes: {input: ['table'], output: ['polygons', 'points']},
    useCases: ['index', 'summarize'],
    domains: ['cartography', 'data-engineering'],
    params: [
      {
        name: 'gridType',
        type: 'GPUGridType',
        required: true,
        description: "'square', 'hex', 'triangle' or 'point'."
      },
      {name: 'columns', type: 'number', required: true, description: 'Cells across.'},
      {name: 'rows', type: 'number', required: true, description: 'Cells down.'},
      {name: 'hexOrientation', type: '"pointy" | "flat"', description: 'Hexagon orientation.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed origin and cell size.'
      },
      {
        name: 'extent',
        type: 'GPUPolygonGeometryPort',
        description: 'Clip cells to this polygon extent.'
      },
      {name: 'spatialContext', type: 'GPUSpatialContext', description: 'Planar by default.'},
      {
        name: 'output',
        type: 'GPUGridGeneratorOutput',
        required: true,
        description: 'Choose centers, corners or positions.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Optional grid polygon vertices.'
      },
      {name: 'centers', type: "GraphDataView<'float32x2'>", description: 'Optional cell centers.'},
      {
        name: 'corners',
        type: "GraphDataView<'float32x2'>",
        description: 'Optional cell corner positions.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUGridGenerator,
  GPUParameterBuffer,
  getGPUGridGeneratorParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'grid'});
const parameters = new GPUParameterBuffer(device, {id: 'grid-parameters', format: 'float32', length: 4});
// \`centers\`, \`positions\` and the output views are graph views imported on \`graph\`.
graph.add(new GPUGridGenerator({
  gridType: 'hex',
  columns: 64,
  rows: 48,
  parameters: parameters.importToGraph(graph),
  output: {centers, positions}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUGridGeneratorParameterValues({minX: -122.5, minY: 37.7, cellWidth: 0.01, cellHeight: 0.01}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'cells', seed: 4128, caption: 'A hexagon grid generated on the GPU.'},
    related: ['shape-generator', 'delaunay-tessellation'],
    lensScenario: 'gpu-spatial-analysis.grid-generator',
    reference: 'api/gpu-spatial-analysis',
    maturity: 'stable'
  },
  {
    id: 'shape-generator',
    symbol: 'GPUShapeGenerator',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Shape generator',
    summary: 'Generates circle, sector or ellipse rings around per-feature centers.',
    explanation:
      'Best for radius rings, coverage wedges and error ellipses (turf `circle`, `sector`, `ellipse`). Each feature has its own radius, bearings and rotation. Segment count and radius scale are per-frame parameters.',
    dataTypes: {input: ['points', 'table'], output: ['polygons']},
    useCases: ['transform'],
    domains: ['cartography', 'urban'],
    params: [
      {
        name: 'shape',
        type: 'GPUShapeType',
        required: true,
        description: "'circle', 'sector' or 'ellipse'."
      },
      {
        name: 'centers',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Center of every shape.'
      },
      {
        name: 'radii',
        type: "GraphDataView<'float32'> | GraphDataView<'float32x2'>",
        required: true,
        description: 'Radius, or x and y radii for ellipses.'
      },
      {
        name: 'bearings',
        type: "GraphDataView<'float32x2'>",
        description: 'Start and end bearing for sectors.'
      },
      {
        name: 'rotations',
        type: "GraphDataView<'float32'>",
        description: 'Ellipse rotation per feature.'
      },
      {name: 'maximumSegments', type: 'number', description: 'Cap on segments per shape.'},
      {
        name: 'parameters',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Packed segment count and radius scale.'
      },
      {
        name: 'output',
        type: 'GPUShapeGeneratorOutput',
        required: true,
        description: 'Caller-owned ring views.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Ring vertices of every shape.'
      },
      {
        name: 'offsets',
        type: "GraphDataView<'uint32'>",
        description: 'Vertex range of every shape.'
      },
      {
        name: 'vertexCount',
        type: "GraphDataView<'uint32'> | undefined",
        description: 'Optional one-row total vertex count.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUShapeGenerator,
  GPUParameterBuffer,
  getGPUShapeGeneratorParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'shapes'});
const parameters = new GPUParameterBuffer(device, {id: 'shape-parameters', format: 'float32', length: 2});
// \`centers\`, \`radii\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUShapeGenerator({
  shape: 'circle',
  centers,
  radii,
  maximumSegments: 64,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUShapeGeneratorParameterValues({segmentCount: 48}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'buffers', seed: 4129, caption: 'Circles and sectors around towers.'},
    related: ['grid-generator', 'buffer-surface', 'outline-geometry'],
    lensScenario: 'gpu-spatial-analysis.shape-generator',
    reference: 'api/gpu-spatial-analysis/operators-geometry',
    maturity: 'stable'
  },
  {
    id: 'dorling-cartogram',
    symbol: 'GPUDorlingCartogram',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Dorling cartogram',
    summary:
      'Turns areas into circles sized by value and relaxes them until they stop overlapping.',
    explanation:
      'Shows a count or rate without large empty regions dominating the map. Circle area follows the value, and circles are pushed apart while a weak pull keeps them near their true location. Iterations are bounded and reported.',
    dataTypes: {input: ['points', 'table'], output: ['points']},
    useCases: ['classify', 'summarize'],
    domains: ['cartography', 'demographics'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'True location of every area.'
      },
      {
        name: 'values',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Value that sets circle area.'
      },
      {name: 'areaScale', type: 'number', description: 'Scale from value to circle area.'},
      {
        name: 'maximumIterations',
        type: 'number',
        description: 'Relaxation rounds before stopping.'
      },
      {name: 'overlapTolerance', type: 'number', description: 'Overlap accepted as converged.'},
      {name: 'attraction', type: 'number', description: 'Pull back toward the true location.'},
      {
        name: 'neighborGrid',
        type: 'boolean | "auto"',
        description: 'Use a grid for neighbor search on large inputs.'
      },
      {
        name: 'output',
        type: 'GPUDorlingCartogramOutput',
        required: true,
        description: 'Caller-owned circle views.'
      }
    ],
    outputs: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        description: 'Circle centers after relaxation.'
      },
      {name: 'radii', type: "GraphDataView<'float32'>", description: 'Circle radius per area.'},
      {
        name: 'metrics',
        type: "GraphDataView<'float32x2'>",
        description: 'Overlap and displacement diagnostics.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32'>",
        description: 'Converged flag and iteration count.'
      },
      {
        name: 'sourceIds',
        type: "GraphDataView<'uint32'>",
        description: 'Source feature id of each circle.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUDorlingCartogram} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'dorling'});
// \`centroids\`, \`population\` and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUDorlingCartogram({
  positions: centroids,
  values: population,
  areaScale: 0.5,
  maximumIterations: 200,
  attraction: 0.05,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'point-clusters', seed: 4130, caption: 'Counties as non-overlapping circles.'},
    related: ['group-geometry', 'map-coloring', 'voronoi-diagram'],
    lensScenario: 'gpu-spatial-analysis.dorling-cartogram',
    reference: 'api/gpu-spatial-analysis/gpu-dorling-cartogram',
    maturity: 'stable'
  },
  {
    id: 'map-coloring',
    symbol: 'GPUMapColoring',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Map coloring',
    summary: 'Assigns each polygon a class so that no two neighbors share one.',
    explanation:
      'Colors a categorical map with few distinct colors. A parallel greedy method (Jones-Plassmann) colors rows from contiguity weights, largest degree first. A seed picks among valid colorings.',
    dataTypes: {input: ['spatial-weights', 'polygons'], output: ['table']},
    useCases: ['classify'],
    domains: ['cartography'],
    params: [
      {
        name: 'weights',
        type: 'GPUSpatialWeights | GPUChunkedSpatialWeights',
        required: true,
        description: 'Contiguity weights between polygons.'
      },
      {
        name: 'parameters',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Packed seed.'
      },
      {
        name: 'priority',
        type: 'GPUMapColoringPriority',
        default: "'largest-degree-first'",
        description: "'largest-degree-first' or 'random'."
      },
      {
        name: 'maximumIterations',
        type: 'number',
        description: 'Rounds before rows stay uncolored.'
      },
      {
        name: 'output',
        type: 'GPUMapColoringOutput',
        required: true,
        description: 'Color indices and diagnostics.'
      }
    ],
    outputs: [
      {
        name: 'colors',
        type: "GraphDataView<'uint32'>",
        description: 'Color index per polygon, from 0.'
      },
      {
        name: 'colorCount',
        type: "GraphDataView<'uint32'>",
        description: 'Optional number of colors used.'
      },
      {
        name: 'conflictCount',
        type: "GraphDataView<'uint32'>",
        description: 'Optional neighbor pairs with equal color.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUMapColoring,
  GPUParameterBuffer,
  getGPUMapColoringParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'map-coloring'});
const parameters = new GPUParameterBuffer(device, {id: 'seed', format: 'uint32', length: 1});
// \`contiguity\` (spatial weights) and \`output\` are graph views imported on \`graph\`.
graph.add(new GPUMapColoring({
  weights: contiguity,
  parameters: parameters.importToGraph(graph),
  output: {colors, colorCount}
}));
const compiled = await graph.compileAsync();
parameters.write(getGPUMapColoringParameterValues({seed: 4131}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'classification',
      seed: 4132,
      caption: 'Counties in five colors, no neighbors alike.'
    },
    related: ['dorling-cartogram', 'coverage-dissolve'],
    lensScenario: 'gpu-spatial-analysis.map-coloring',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'offset-expansion',
    symbol: 'GPUOffsetExpansion',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Offset expansion',
    summary: 'Expands an offsets array into an owner index per child row.',
    explanation:
      'Turns ragged layouts (lines of vertices, polygons of rings) into flat per-child columns. It is the GPU form of `repeat` with counts. Joins and gathers build on it.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['index', 'relate'],
    domains: ['data-engineering'],
    params: [
      {
        name: 'offsets',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Start row of every parent plus a terminal row.'
      },
      {
        name: 'ownerMap',
        type: "GraphDataView<'uint32'>",
        description: 'Optional mapping from parent slot to owner id.'
      },
      {
        name: 'output',
        type: 'GPUOffsetExpansionOutput',
        required: true,
        description: 'Caller-owned owner views and status.'
      }
    ],
    outputs: [
      {
        name: 'owners',
        type: "GraphDataView<'uint32'>",
        description: 'Owner index of every child row.'
      },
      {
        name: 'localIndex',
        type: "GraphDataView<'uint32'>",
        description: 'Optional child index within its owner.'
      },
      {
        name: 'status',
        type: 'GPUCardinalityPort & GPUOverflowStatusPort',
        description: 'Row count and overflow.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUOffsetExpansion} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'expand'});
// \`lineOffsets\` and \`output\` (owners, localIndex, status) are graph views on \`graph\`.
graph.add(new GPUOffsetExpansion({
  offsets: lineOffsets,
  output
}));
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// offsets [0, 3, 5] expand to owners [0, 0, 0, 1, 1]`,
    preview: {kind: 'pipeline', seed: 4133, caption: 'Offsets expanded to one owner per row.'},
    related: ['parameter-buffer', 'line-split'],
    lensScenario: 'gpu-spatial-analysis.offset-expansion',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'parameter-buffer',
    symbol: 'GPUParameterBuffer',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Parameter buffer',
    summary:
      'A small buffer for per-frame values, so tolerances and distances change without recompiling.',
    explanation:
      'Best for any setting you want to animate or tweak: distances, tolerances, seeds. Import it into the graph once, compile once, then write new values before each encode. Values can be float32, uint32 or sint32.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['index'],
    domains: ['data-engineering'],
    params: [
      {name: 'id', type: 'string', required: true, description: 'Buffer and graph resource name.'},
      {
        name: 'format',
        type: 'Format',
        required: true,
        description: "'float32', 'uint32' or 'sint32'."
      },
      {name: 'length', type: 'number', required: true, description: 'Number of values.'},
      {
        name: 'values',
        type: 'Float32Array | Uint32Array | Int32Array',
        description: 'Initial contents.'
      }
    ],
    outputs: [
      {
        name: 'importToGraph(graph)',
        type: 'GraphDataView',
        description: 'A graph view to pass as a parameters prop.'
      },
      {name: 'write(values)', type: 'void', description: 'Updates the buffer between encodings.'}
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULineSimplification,
  GPUParameterBuffer,
  getGPULineSimplificationParameterValues
} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'tolerance-slider'});
const parameters = new GPUParameterBuffer(device, {id: 'tolerance', format: 'float32', length: 4});
graph.add(new GPULineSimplification({
  positions, lineOffsets, importance,
  parameters: parameters.importToGraph(graph),
  output
}));
const compiled = await graph.compileAsync(); // once

// on every slider change: write, encode, no recompile
parameters.write(getGPULineSimplificationParameterValues({tolerance}));
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {kind: 'pipeline', seed: 4134, caption: 'One compiled graph, new values every frame.'},
    related: ['offset-expansion', 'line-simplification', 'buffer-surface'],
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },

  // ---------------------------------------------------------------------------------------------
  // Urban-form recipes
  // ---------------------------------------------------------------------------------------------
  {
    id: 'building-adjacency-recipe',
    symbol: 'addBuildingAdjacencyRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Building adjacency',
    summary: 'Finds shared walls between buildings and summarizes adjacency and axial alignment.',
    explanation:
      'Best for urban-form studies of terraces, blocks and contiguity. It consumes shared-path runs, pairs buildings, and measures the wall lengths and orientation agreement. Components group connected buildings.',
    dataTypes: {input: ['lines', 'polygons'], output: ['table', 'network']},
    useCases: ['relate', 'measure'],
    domains: ['urban'],
    params: [
      {
        name: 'sharedPaths',
        type: 'GPUSharedPathsOutput',
        required: true,
        description: 'Shared-wall runs from GPUSharedPaths.'
      },
      {
        name: 'lineBuildingIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Building id of every wall line.'
      },
      {
        name: 'orientations',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Axial orientation per building, radians.'
      },
      {
        name: 'contiguity',
        type: "GraphDataView<'uint32x2'>",
        description: 'Optional precomputed building pairs.'
      },
      {
        name: 'structures',
        type: 'GPUBuildingStructureGeometry',
        description: 'Optional structure geometry for extra measures.'
      },
      {
        name: 'id',
        type: 'string',
        default: "'building-adjacency'",
        description: 'Prefix for graph resources.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: '{pairs, lengths, morphology, components}',
        description: 'Adjacent building pairs, shared wall lengths, morphology and components.'
      },
      {
        name: 'structures',
        type: '{perimeters, overlays}',
        description: 'Intermediate perimeters and overlay results.'
      },
      {
        name: 'status',
        type: '{complete}',
        description: '1 when every stage finished within capacity.'
      },
      {
        name: 'provenance',
        type: '{sharedPaths, lineBuildingIds}',
        description: 'Source ids tying outputs back to buildings.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addBuildingAdjacencyRecipe} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'adjacency'});
// \`sharedPaths\` comes from a GPUSharedPaths node added to the same graph; the
// other inputs are graph views imported on \`graph\`.
const {outputs, status} = addBuildingAdjacencyRecipe(graph, {
  sharedPaths,
  lineBuildingIds,
  orientations
});
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// read outputs.pairs, outputs.morphology and outputs.components`,
    preview: {kind: 'polygons', seed: 4135, caption: 'Buildings linked across shared walls.'},
    related: ['shared-paths', 'building-tessellation-recipe', 'street-profiles-recipe'],
    lensScenario: 'gpu-spatial-analysis.building-adjacency-recipe',
    reference: 'api/gpu-spatial-analysis/urban-ecology-lenses',
    maturity: 'stable'
  },
  {
    id: 'building-tessellation-recipe',
    symbol: 'addBuildingTessellationRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Building tessellation',
    summary:
      'Divides the space around buildings into one cell per building, clipped to an enclosure.',
    explanation:
      'Best for morphological tessellation: the plot of land each building would claim. Building boundaries are sampled, Voronoi fragments are clipped to the enclosure, then dissolved by owner. Sample and fragment capacities are explicit.',
    dataTypes: {input: ['polygons', 'lines'], output: ['polygons']},
    useCases: ['transform', 'measure'],
    domains: ['urban'],
    params: [
      {
        name: 'segments',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Building wall segments as (x0, y0, x1, y1).'
      },
      {
        name: 'segmentBuildingIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Building id of every wall segment.'
      },
      {name: 'buildingCount', type: 'number', required: true, description: 'Number of buildings.'},
      {
        name: 'enclosure',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Outer boundary vertices.'
      },
      {
        name: 'enclosureMode',
        type: '"convex" | "simple"',
        default: "'convex'",
        description: 'How the enclosure is clipped.'
      },
      {
        name: 'sampleSpacing',
        type: 'number',
        required: true,
        description: 'Distance between sampled wall points.'
      },
      {
        name: 'sampleCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on sampled points.'
      },
      {
        name: 'vertexTolerance',
        type: 'number',
        required: true,
        description: 'Snap tolerance in position units.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: 'GPUPolygonOverlayOutput-like polygons',
        description: 'One tessellation polygon per building.'
      },
      {
        name: 'intermediates',
        type: '{samples, owners, fragments, fragmentCounts}',
        description: 'Sampled points and Voronoi fragments.'
      },
      {
        name: 'status',
        type: '{sampling, complete}',
        description: 'Sampling status and a completion flag.'
      },
      {
        name: 'provenance',
        type: '{segments, segmentBuildingIds}',
        description: 'Boundary segments and their owning buildings.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addBuildingTessellationRecipe} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'tessellation'});
// Wall \`segments\`, \`segmentBuildingIds\` and \`enclosure\` are graph views on \`graph\`.
const {outputs, status} = addBuildingTessellationRecipe(graph, {
  segments,
  segmentBuildingIds,
  buildingCount: 480,
  enclosure,
  sampleSpacing: 2,
  sampleCapacity: 100000,
  vertexTolerance: 1e-3,
  capacity: {intersections: 200000, nodedSegments: 400000, vertices: 400000, rings: 200000}
});
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});`,
    preview: {
      kind: 'tessellation',
      seed: 4136,
      caption: 'One cell per building, clipped to the block.'
    },
    related: ['voronoi-diagram', 'building-adjacency-recipe', 'polygon-overlay'],
    lensScenario: 'gpu-spatial-analysis.building-tessellation-recipe',
    reference: 'api/gpu-spatial-analysis/urban-ecology-lenses',
    maturity: 'stable'
  },
  {
    id: 'isovists-recipe',
    symbol: 'addIsovistsRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Isovists',
    summary:
      'Computes the visible area from each viewpoint among obstacles, as a polygon fan of rays.',
    explanation:
      'Best for visibility, openness and sightline studies in streets and plazas. Rays are cast counterclockwise from +X to the nearest obstacle within `maximumDistance`. Vertex sampling adds rays at obstacle corners.',
    dataTypes: {input: ['points', 'lines'], output: ['polygons', 'table']},
    useCases: ['measure', 'relate'],
    domains: ['urban', 'three-d'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Viewpoints.'
      },
      {
        name: 'segments',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Obstacle walls as (x0, y0, x1, y1).'
      },
      {name: 'rayCount', type: 'number', required: true, description: 'Rays per viewpoint.'},
      {
        name: 'maximumDistance',
        type: 'number',
        required: true,
        description: 'Longest visible distance.'
      },
      {
        name: 'sampling',
        type: '"vertices" | "uniform"',
        default: "'uniform'",
        description: 'Even angles, or add rays toward obstacle corners.'
      },
      {name: 'angularEpsilon', type: 'number', description: 'Angle nudge around corner rays.'},
      {name: 'tolerance', type: 'number', description: 'Geometric tolerance.'},
      {
        name: 'region',
        type: 'UrbanFreeSpaceRegion',
        description: 'Restrict viewpoints to free space.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: '{endpoints, distances, measures, validity}',
        description: 'Ray hits, ray lengths, isovist measures and viewpoint validity.'
      },
      {
        name: 'provenance',
        type: '{positions, obstacles}',
        description: 'Resolved viewpoints and obstacle inputs.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addIsovistsRecipe} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'isovists'});
// \`viewpoints\` (float32x2) and wall \`segments\` (float32x4) are graph views on \`graph\`.
const {outputs} = addIsovistsRecipe(graph, {
  positions: viewpoints,
  segments,
  rayCount: 360,
  maximumDistance: 200,
  sampling: 'uniform'
});
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// outputs.endpoints are ray hits; outputs.measures hold area and perimeter`,
    preview: {kind: 'polygons', seed: 4137, caption: 'What is visible from one street corner.'},
    related: ['visibility-graph-recipe', 'street-profiles-recipe'],
    lensScenario: 'gpu-spatial-analysis.isovists-recipe',
    reference: 'api/gpu-spatial-analysis/urban-ecology-lenses',
    maturity: 'stable'
  },
  {
    id: 'street-profiles-recipe',
    symbol: 'addStreetProfilesRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Street profiles',
    summary:
      'Measures distance and height of the first building on each side of every street sample.',
    explanation:
      'Best for canyon width, height-to-width ratio and frontage studies. Rays cast left and right of each sample along its normal. A side with no building stays range-censored and is never reported as open space.',
    dataTypes: {input: ['points', 'lines', 'table'], output: ['table']},
    useCases: ['measure'],
    domains: ['urban'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Street sample points.'
      },
      {
        name: 'normals',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Unit normal at every sample.'
      },
      {
        name: 'segments',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Facade segments as (x0, y0, x1, y1).'
      },
      {
        name: 'segmentBuildingIds',
        type: "GraphDataView<'uint32'>",
        required: true,
        description: 'Building id of every facade segment.'
      },
      {
        name: 'buildingHeights',
        type: "GraphDataView<'float32'>",
        required: true,
        description: 'Height of every building.'
      },
      {name: 'maximumDistance', type: 'number', required: true, description: 'Longest ray length.'},
      {
        name: 'sampleSpacing',
        type: 'number',
        required: true,
        description: 'Spacing used for frontage sampling.'
      },
      {
        name: 'streetSegment',
        type: "GraphDataView<'float32x4'>",
        description: 'Optional street segment per sample.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: '{profiles, buildings, canyon, validity}',
        description: 'Side distances, building ids, canyon measures and validity.'
      },
      {
        name: 'facades',
        type: '{lengths, frontage}',
        description: 'Facade lengths and frontage per sample.'
      },
      {
        name: 'provenance',
        type: '{positions, normals}',
        description: 'Sample positions and normals along each street.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addStreetProfilesRecipe} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'street-profiles'});
// Samples, facade \`segments\` and building columns are graph views on \`graph\`.
const {outputs} = addStreetProfilesRecipe(graph, {
  positions: samples,
  normals,
  segments,
  segmentBuildingIds,
  buildingHeights,
  maximumDistance: 60,
  sampleSpacing: 5
});
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// outputs.profiles: distance to the building on each side`,
    preview: {
      kind: 'lines',
      seed: 4138,
      caption: 'Perpendicular rays from a street to its facades.'
    },
    related: ['isovists-recipe', 'building-adjacency-recipe', 'visibility-graph-recipe'],
    lensScenario: 'gpu-spatial-analysis.street-profiles-recipe',
    reference: 'api/gpu-spatial-analysis/urban-ecology-lenses',
    maturity: 'stable'
  },
  {
    id: 'visibility-graph-recipe',
    symbol: 'addVisibilityGraphRecipe',
    kind: 'recipe',
    importPath: IMPORT_PATH,
    title: 'Visibility graph',
    summary:
      'Links every pair of mutually visible points among obstacles and derives graph metrics.',
    explanation:
      'Best for space syntax: integration, choice and depth over a plaza or building plan. Closed obstacles block sightlines, and visible pairs become edges within `maximumDistance`. Edge capacity bounds the result.',
    dataTypes: {input: ['points', 'lines'], output: ['network', 'table']},
    useCases: ['relate', 'measure'],
    domains: ['urban', 'transport'],
    params: [
      {
        name: 'positions',
        type: "GraphDataView<'float32x2'>",
        required: true,
        description: 'Candidate viewpoints.'
      },
      {
        name: 'segments',
        type: "GraphDataView<'float32x4'>",
        required: true,
        description: 'Obstacle walls as (x0, y0, x1, y1).'
      },
      {name: 'maximumDistance', type: 'number', required: true, description: 'Longest sightline.'},
      {
        name: 'edgeCapacity',
        type: 'number',
        required: true,
        description: 'Upper bound on visible pairs.'
      },
      {name: 'tolerance', type: 'number', description: 'Geometric tolerance.'},
      {
        name: 'region',
        type: 'UrbanFreeSpaceRegion',
        description: 'Restrict viewpoints to free space.'
      },
      {
        name: 'id',
        type: 'string',
        default: "'visibility-graph'",
        description: 'Prefix for graph resources.'
      }
    ],
    outputs: [
      {
        name: 'outputs',
        type: '{edges, adjacency, metrics, integration, choice}',
        description: 'Visible pairs, adjacency lists and per-viewpoint graph measures.'
      },
      {
        name: 'status',
        type: "GraphDataView<'uint32x4'>",
        description: 'Completeness and overflow words.'
      },
      {
        name: 'provenance',
        type: '{positions, obstacles}',
        description: 'Resolved viewpoints and obstacle inputs.'
      }
    ],
    example: `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {addVisibilityGraphRecipe} from '${IMPORT_PATH}';

const graph = new GPUCommandGraph(device, {id: 'visibility'});
// \`grid\` (float32x2 points) and wall \`segments\` (float32x4) are graph views on \`graph\`.
const {outputs, status} = addVisibilityGraphRecipe(graph, {
  positions: grid,
  segments,
  maximumDistance: 120,
  edgeCapacity: 500000
});
const compiled = await graph.compileAsync();
compiled.encode(device.commandEncoder, {parameters: undefined});
// outputs.edges lists visible pairs; outputs.integration scores each point`,
    preview: {kind: 'network', seed: 4139, caption: 'Sightlines between points on a plaza.'},
    related: ['isovists-recipe', 'street-profiles-recipe'],
    lensScenario: 'gpu-spatial-analysis.visibility-graph-recipe',
    reference: 'api/gpu-spatial-analysis/urban-ecology-lenses',
    maturity: 'stable'
  }
];
