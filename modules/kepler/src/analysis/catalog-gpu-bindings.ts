// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogGpuContext,
  CatalogInputSpec,
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogParameterValue,
  CatalogRunOutput
} from './catalog-binding-types';
import {
  COORDINATE_SYSTEM_PARAMETER,
  LINE_INPUT,
  MAXIMUM_NETWORK_EDGES,
  NETWORK_BEHAVIOR,
  NETWORK_PARAMETERS,
  SNAP_DISTANCE_PARAMETER,
  getNetworkParameters
} from './catalog-binding-types';
import type {CatalogCoordinateSystem, CatalogFeatureCollection} from './catalog-geojson';
import {
  runGpuBuffer,
  runGpuGeometryMeasures,
  runGpuGeometryValidity,
  runGpuLineSegmentize,
  runGpuPolygonOverlay,
  runGpuRectangleClip,
  runGpuTopologyRepair
} from './catalog-gpu-geometry';
import {runGpuPointBuffer} from './catalog-gpu-point-buffer';
import {runGpuSnapPoints} from './catalog-snap-binding';
import {
  readCatalogCrossings,
  runGpuNetworkBuild,
  runGpuNetworkIsochrones,
  runGpuNetworkMultiOrigin,
  runGpuNetworkNoding,
  runGpuNetworkServiceArea,
  runGpuNetworkShortestPaths
} from './catalog-gpu-network';

type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;
type Values = Readonly<Record<string, CatalogParameterValue>>;

/** GPU path of an existing CPU-bound operation. */
export type CatalogGpuRunner = {
  contributorNames: readonly string[];
  run(context: CatalogGpuContext, inputs: Inputs, parameters: Values): Promise<CatalogRunOutput>;
};

function getCoordinateSystem(values: Values): CatalogCoordinateSystem {
  return values['coordinateSystem'] as CatalogCoordinateSystem;
}

function getGpuNetworkParameters(values: Values) {
  return {...getNetworkParameters(values), crossings: readCatalogCrossings(values)};
}

/** GPU runners for the eight CPU-bound operations, keyed by binding id. */
export const CATALOG_GPU_RUNNERS: Readonly<Record<string, CatalogGpuRunner>> = {
  'network-snap-points': {
    contributorNames: ['GPUNetworkSnapping'],
    run: (context, inputs, values) => runGpuSnapPoints(context, inputs, values)
  },
  'geometry-validity': {
    contributorNames: ['GPUGeometryValidity'],
    run: (context, inputs) => runGpuGeometryValidity(context, inputs['polygons'])
  },
  'geometry-measures': {
    contributorNames: ['GPUGeometryMeasures'],
    run: (context, inputs, values) =>
      runGpuGeometryMeasures(context, inputs['features'], getCoordinateSystem(values))
  },
  'geometry-point-buffer': {
    contributorNames: ['GPUPointBuffer'],
    run: (context, inputs, values) =>
      runGpuPointBuffer(context, inputs['points'], {
        radius: values['radius'] as number,
        segmentsPerQuarter: values['segmentsPerQuarter'] as number,
        coordinateSystem: getCoordinateSystem(values)
      })
  },
  'geometry-line-segmentize': {
    contributorNames: ['GPULineSegmentize'],
    run: (context, inputs, values) =>
      runGpuLineSegmentize(
        context,
        inputs['lines'],
        {
          maximumSegmentLength: values['maximumSegmentLength'] as number,
          coordinateSystem: getCoordinateSystem(values)
        },
        500000
      )
  },
  'geometry-convex-clip': {
    contributorNames: ['GPURectangleClip'],
    run: (context, inputs) => runGpuRectangleClip(context, inputs['features'], inputs['clip'])
  },
  'network-build-graph': {
    contributorNames: ['GPUNetworkNoding'],
    run: (context, inputs, values) =>
      runGpuNetworkBuild(
        context,
        inputs['lines'],
        getGpuNetworkParameters(values),
        MAXIMUM_NETWORK_EDGES
      )
  },
  'network-shortest-path': {
    contributorNames: ['GPUNetworkReachability'],
    run: (context, inputs, values) =>
      runGpuNetworkShortestPaths(
        context,
        inputs['lines'],
        inputs['origin'],
        inputs['destinations'],
        {
          ...getGpuNetworkParameters(values),
          maximumSnapDistance: values['maximumSnapDistance'] as number
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES, maximumDestinations: 500}
      )
  },
  'network-service-area': {
    contributorNames: ['GPUNetworkReachability'],
    run: (context, inputs, values) =>
      runGpuNetworkServiceArea(
        context,
        inputs['lines'],
        inputs['origin'],
        {
          ...getGpuNetworkParameters(values),
          maximumSnapDistance: values['maximumSnapDistance'] as number,
          budgets: values['budgets'] as readonly number[],
          includeUnreachable: values['includeUnreachable'] as boolean
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  }
};

const GPU_CLEANUP =
  'The GPU session creates storage buffers and compiled graphs per run and destroys them before the result returns; results are plain arrays and publication goes through native Sources.';

const POLYGON_INPUT = (name: string, label: string): CatalogInputSpec => ({
  name,
  label,
  accepts: ['polygon'],
  maximumFeatures: 5000,
  maximumVertices: 50000
});

const BUFFER_PARAMETERS: readonly CatalogParameterSpec[] = [
  {
    name: 'quadSegments',
    label: 'Segments per quarter circle',
    kind: 'integer',
    minimum: 1,
    maximum: 16,
    defaultValue: 8
  },
  {
    name: 'joinStyle',
    label: 'Join style',
    kind: 'enum',
    options: ['round', 'mitre', 'bevel'],
    defaultValue: 'round'
  },
  {
    name: 'capStyle',
    label: 'Cap style (lines)',
    kind: 'enum',
    options: ['round', 'flat', 'square'],
    defaultValue: 'round'
  },
  {
    name: 'mitreLimit',
    label: 'Mitre limit',
    kind: 'number',
    minimum: 1,
    maximum: 20,
    defaultValue: 5
  },
  {
    name: 'dissolve',
    label: 'Dissolve into one union',
    kind: 'boolean',
    defaultValue: true
  },
  COORDINATE_SYSTEM_PARAMETER
];

function readBufferParameters(values: Values) {
  return {
    distance: values['distance'] as number,
    quadSegments: values['quadSegments'] as number,
    joinStyle: values['joinStyle'] as 'round' | 'mitre' | 'bevel',
    capStyle: values['capStyle'] as 'round' | 'flat' | 'square',
    mitreLimit: values['mitreLimit'] as number,
    dissolve: values['dissolve'] as boolean,
    coordinateSystem: getCoordinateSystem(values)
  };
}

function makeGpuBinding(
  binding: Omit<CatalogOperationBinding, 'engine' | 'cleanup' | 'run'>
): CatalogOperationBinding {
  return {...binding, engine: 'gpu', cleanup: GPU_CLEANUP};
}

function makeOverlayBinding(
  operation: 'intersection' | 'union' | 'difference',
  title: string,
  summary: string
): CatalogOperationBinding {
  return makeGpuBinding({
    id: `geometry-overlay-${operation}`,
    title,
    domain: 'geometry',
    summary,
    capabilityIds: ['geometry-overlay'],
    contributorNames: ['GPUPolygonOverlay'],
    gpuContributorNames: ['GPUPolygonOverlay'],
    inputs: [POLYGON_INPUT('left', 'Left polygons'), POLYGON_INPUT('right', 'Right polygons')],
    parameters: [
      COORDINATE_SYSTEM_PARAMETER,
      {
        name: 'fillRule',
        label: 'Inside rule',
        kind: 'enum',
        options: ['even-odd', 'nonzero'],
        defaultValue: 'even-odd'
      }
    ],
    outputs: [
      {
        tableId: 'overlay',
        label: `Overlay ${operation}`,
        geometry: 'polygon',
        identityColumns: ['overlay_id', 'component_index'],
        maximumRows: 20000
      }
    ],
    declaredBehavior: [
      'Each layer is one operand: polygons inside a layer are dissolved, so result rows carry no source feature identity.',
      'Degrees are projected into a local azimuthal-equidistant frame; vertices snap to a lattice about 2^-20 of the extent.',
      'Capacities grow automatically up to the device storage limit; beyond it the run fails instead of truncating.'
    ],
    limitation:
      'GPU only; no CPU reference engine. Whole-layer operands, planar arrangement, area in the local frame.',
    runGpu: (context, inputs, values) =>
      runGpuPolygonOverlay(context, inputs['left'], inputs['right'], {
        operation,
        coordinateSystem: getCoordinateSystem(values),
        fillRule: values['fillRule'] as 'even-odd' | 'nonzero'
      })
  });
}

const MULTI_ORIGIN_INPUT: CatalogInputSpec = {
  name: 'origins',
  label: 'Origin points',
  accepts: ['point'],
  maximumFeatures: 32,
  maximumVertices: 32
};

const BUDGETS_PARAMETER: CatalogParameterSpec = {
  name: 'budgets',
  label: 'Cost budgets',
  kind: 'number-list',
  unit: 'cost',
  minimum: 0,
  maximum: 1e9,
  maximumLength: 8,
  defaultValue: [500, 1000]
};

/** Operations that exist only as GPU contributors; they need a device. */
export const CATALOG_GPU_ONLY_BINDINGS: readonly CatalogOperationBinding[] = [
  makeGpuBinding({
    id: 'geometry-line-buffer',
    title: 'Line buffer',
    domain: 'geometry',
    summary:
      'Buffers lines with round, flat or square caps and round, mitre or bevel joins, optionally dissolved into one union.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUBufferSurface'],
    gpuContributorNames: ['GPUBufferSurface', 'GPUPolygonOverlay'],
    inputs: [
      {
        name: 'lines',
        label: 'Lines',
        accepts: ['line'],
        maximumFeatures: 5000,
        maximumVertices: 20000
      }
    ],
    parameters: [
      {
        name: 'distance',
        label: 'Buffer distance',
        kind: 'number',
        unit: 'crs',
        minimum: 1e-9,
        maximum: 1e7,
        defaultValue: 100
      },
      ...BUFFER_PARAMETERS
    ],
    outputs: [
      {
        tableId: 'buffers',
        label: 'Line buffers',
        geometry: 'polygon',
        identityColumns: ['buffer_id', 'component_index', 'source_index', 'source_id'],
        maximumRows: 20000
      }
    ],
    declaredBehavior: [
      'Distance is meters for EPSG:4326 (local azimuthal-equidistant frame) and coordinate units for planar.',
      'Dissolved output is one union; without dissolve each feature is buffered in its own arrangement (at most 256 features).',
      'Non-line features are skipped and counted; a non-positive distance gives an empty surface.'
    ],
    limitation:
      'GPU only; planar arrangement in a local frame, so extents beyond a few hundred kilometers distort. Pole-crossing and antimeridian data are not handled.',
    runGpu: (context, inputs, values) =>
      runGpuBuffer(context, 'line', inputs['lines'], {
        ...readBufferParameters(values)
      })
  }),
  makeGpuBinding({
    id: 'geometry-polygon-buffer',
    title: 'Polygon buffer',
    domain: 'geometry',
    summary:
      'Grows (positive) or shrinks (negative) polygons; collapsed holes vanish and necks thinner than twice the distance split.',
    capabilityIds: ['geometry-construction'],
    contributorNames: ['GPUBufferSurface'],
    gpuContributorNames: ['GPUBufferSurface', 'GPUPolygonOverlay'],
    inputs: [POLYGON_INPUT('polygons', 'Polygons')],
    parameters: [
      {
        name: 'distance',
        label: 'Signed buffer distance',
        kind: 'number',
        unit: 'crs',
        minimum: -1e7,
        maximum: 1e7,
        defaultValue: 100
      },
      ...BUFFER_PARAMETERS
    ],
    outputs: [
      {
        tableId: 'buffers',
        label: 'Polygon buffers',
        geometry: 'polygon',
        identityColumns: ['buffer_id', 'component_index', 'source_index', 'source_id'],
        maximumRows: 20000
      }
    ],
    declaredBehavior: [
      'Positive distance grows shells and shrinks holes; negative does the reverse.',
      'Dissolved output is one union; without dissolve each feature is buffered in its own arrangement (at most 256 features).',
      'Non-polygon features are skipped and counted.'
    ],
    limitation:
      'GPU only; planar arrangement in a local frame (meters for EPSG:4326). Extents beyond a few hundred kilometers distort.',
    runGpu: (context, inputs, values) =>
      runGpuBuffer(context, 'polygon', inputs['polygons'], readBufferParameters(values))
  }),
  makeOverlayBinding(
    'intersection',
    'Overlay intersection',
    'Area covered by both polygon layers.'
  ),
  makeOverlayBinding('union', 'Overlay union', 'Area covered by either polygon layer.'),
  makeOverlayBinding(
    'difference',
    'Overlay difference',
    'Area of the left layer not covered by the right layer.'
  ),
  makeGpuBinding({
    id: 'geometry-topology-repair',
    title: 'Topology repair (make valid)',
    domain: 'geometry',
    summary:
      'Nodes the rings of each polygon feature and rebuilds valid polygons by even-odd source parity; bow ties split and doubly covered lenses drop.',
    capabilityIds: ['geometry-validation', 'geometry-editing'],
    contributorNames: ['GPUMakeValid'],
    gpuContributorNames: ['GPUMakeValid'],
    inputs: [POLYGON_INPUT('polygons', 'Polygons to repair')],
    parameters: [COORDINATE_SYSTEM_PARAMETER],
    outputs: [
      {
        tableId: 'repaired',
        label: 'Repaired polygons',
        geometry: 'polygon',
        identityColumns: ['source_index', 'source_id'],
        maximumRows: 5000
      }
    ],
    declaredBehavior: [
      'One row per input polygon feature with status repaired, unchanged or empty.',
      'Repair is independent per feature; shared boundaries between features are not reconciled.',
      'Capacities grow automatically; beyond four growths the run fails instead of truncating.'
    ],
    limitation: 'GPU only; planar arrangement in a local frame; areas are measured in that frame.',
    runGpu: (context, inputs, values) =>
      runGpuTopologyRepair(context, inputs['polygons'], {
        coordinateSystem: getCoordinateSystem(values)
      })
  }),
  makeGpuBinding({
    id: 'network-crossing-noding',
    title: 'Node crossing lines',
    domain: 'network',
    summary:
      'Cuts lines at crossings, touches and overlaps, snaps piece ends within a tolerance and numbers the nodes.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkNoding'],
    gpuContributorNames: ['GPUNetworkNoding'],
    inputs: [LINE_INPUT],
    parameters: [
      COORDINATE_SYSTEM_PARAMETER,
      {
        name: 'snapTolerance',
        label: 'Snap tolerance',
        kind: 'number',
        unit: 'crs',
        minimum: 0,
        maximum: 1e6,
        defaultValue: 0
      }
    ],
    outputs: [
      {
        tableId: 'noded-edges',
        label: 'Noded edges',
        geometry: 'line',
        identityColumns: ['piece_id', 'source_index', 'source_id', 'part_index'],
        maximumRows: MAXIMUM_NETWORK_EDGES
      },
      {
        tableId: 'noded-nodes',
        label: 'Noded nodes',
        geometry: 'point',
        identityColumns: ['node_id'],
        maximumRows: MAXIMUM_NETWORK_EDGES
      }
    ],
    declaredBehavior: [
      'Every piece keeps its source feature, part and GPU node ids; lengths are meters in a local frame for EPSG:4326.',
      'Bridges and tunnels are not modeled: every crossing becomes a node.',
      'Capacities grow automatically up to four times; beyond that the run fails.'
    ],
    limitation: 'GPU only; planar noding in a local frame.',
    runGpu: (context, inputs, values) =>
      runGpuNetworkNoding(context, inputs['lines'], {
        coordinateSystem: getCoordinateSystem(values),
        snapTolerance: values['snapTolerance'] as number
      })
  }),
  makeGpuBinding({
    id: 'network-multi-origin-service-area',
    title: 'Service areas for several origins',
    domain: 'network',
    summary:
      'One search lane per origin on a shared network; edges carry first-reach cost, band and whether the origin is the nearest.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkReachability'],
    gpuContributorNames: ['GPUNetworkReachability'],
    inputs: [LINE_INPUT, MULTI_ORIGIN_INPUT],
    parameters: [...NETWORK_PARAMETERS, SNAP_DISTANCE_PARAMETER, BUDGETS_PARAMETER],
    outputs: [
      {
        tableId: 'network-multi-origin',
        label: 'Service areas by origin',
        geometry: 'line',
        identityColumns: ['origin_index', 'edge_id', 'source_index', 'source_id'],
        maximumRows: MAXIMUM_NETWORK_EDGES * 4
      }
    ],
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'At most 32 origins; unreachable edges are not listed; origins without a snap are counted.'
    ],
    limitation:
      'GPU only; edge bands, not polygons; lanes multiply node count by origin count within the device storage limit.',
    runGpu: (context, inputs, values) =>
      runGpuNetworkMultiOrigin(
        context,
        inputs['lines'],
        inputs['origins'],
        {
          ...getGpuNetworkParameters(values),
          maximumSnapDistance: values['maximumSnapDistance'] as number,
          budgets: values['budgets'] as readonly number[],
          includeUnreachable: false
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  }),
  makeGpuBinding({
    id: 'network-isochrone-polygons',
    title: 'Isochrone (service-area) polygons',
    domain: 'network',
    summary:
      'Contours travel cost around the network into one polygon band per budget, from edge costs splatted to a raster.',
    capabilityIds: [],
    contributorNames: ['GPUNetworkIsochrones'],
    gpuContributorNames: ['GPUNetworkReachability', 'GPUNetworkIsochrones'],
    inputs: [LINE_INPUT, {...MULTI_ORIGIN_INPUT, maximumFeatures: 16, maximumVertices: 16}],
    parameters: [
      ...NETWORK_PARAMETERS,
      SNAP_DISTANCE_PARAMETER,
      BUDGETS_PARAMETER,
      {
        name: 'bufferRadius',
        label: 'Street buffer radius',
        kind: 'number',
        unit: 'crs',
        minimum: 0,
        maximum: 1e6,
        defaultValue: 50
      },
      {
        name: 'rasterSize',
        label: 'Raster size (pixels per side)',
        kind: 'integer',
        minimum: 64,
        maximum: 1024,
        defaultValue: 256
      }
    ],
    outputs: [
      {
        tableId: 'network-isochrones',
        label: 'Isochrone bands',
        geometry: 'polygon',
        identityColumns: ['polygon_id', 'band'],
        maximumRows: 20000
      }
    ],
    declaredBehavior: [
      ...NETWORK_BEHAVIOR,
      'Band k holds costs from budget k-1 up to budget k; polygons are exclusive annuli, not cumulative.',
      'Accuracy is about one raster pixel; edges are straight between nodes and the buffer follows the street.',
      'Several origins act as facilities and share bands.'
    ],
    limitation:
      'GPU only; raster contours, not exact offset curves; ring capacity is bounded by raster size and budget count.',
    runGpu: (context, inputs, values) =>
      runGpuNetworkIsochrones(
        context,
        inputs['lines'],
        inputs['origins'],
        {
          ...getGpuNetworkParameters(values),
          maximumSnapDistance: values['maximumSnapDistance'] as number,
          budgets: values['budgets'] as readonly number[],
          bufferRadius: values['bufferRadius'] as number,
          rasterSize: values['rasterSize'] as number
        },
        {maximumEdges: MAXIMUM_NETWORK_EDGES}
      )
  })
];
