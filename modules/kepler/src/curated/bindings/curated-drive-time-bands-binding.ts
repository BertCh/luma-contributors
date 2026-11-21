// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  addDriveTimeCatchmentRecipe,
  getGPUNetworkIsochronesParameterValues,
  GPU_DRIVE_TIME_CATCHMENT_NO_BAND
} from '@luma-contributors/analysis/gpu-network';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {NETWORK_PARAMETERS, SNAP_DISTANCE_PARAMETER} from '../../analysis/catalog-binding-types';
import {
  appendCatalogRow,
  countFeatureVertices,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';
import {buildNetworkGraph} from '../../analysis/catalog-network';
import {buildCatalogDirectedCsr, type CatalogArc} from '../../analysis/catalog-gpu-network';
import {makeLocalFrame} from '../../analysis/catalog-local-frame';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';

const BINDING_ID = 'curated-drive-time-bands';
/** Separate bounds on CPU preparation, exact snapping work, routing and publication readback. */
export const DRIVE_TIME_BANDS_LIMITS = {
  features: 1024,
  vertices: 8192,
  segments: 4096,
  nodes: 4096,
  arcs: 8192,
  facilities: 128,
  demand: 2048,
  bands: 8,
  gridSize: 64,
  /** Published isochrone splatter half-window limit per axis. */
  bufferPixels: 16,
  triangles: 65536,
  iterations: 64
} as const;

const PARAMETERS: CatalogOperationBinding['parameters'] = [
  ...NETWORK_PARAMETERS.filter(parameter =>
    ['coordinateSystem', 'forwardCostProperty', 'reverseCostProperty', 'onewayProperty'].includes(
      parameter.name
    )
  ),
  {
    name: 'costUnit',
    label: 'Travel time unit of the street costs and band breaks',
    kind: 'enum',
    options: ['seconds', 'minutes', 'hours'],
    defaultValue: 'minutes'
  },
  {
    name: 'costBreaks',
    label: 'Ascending travel time upper bounds',
    kind: 'number-list',
    minimum: 0,
    maximum: 1e9,
    maximumLength: DRIVE_TIME_BANDS_LIMITS.bands,
    defaultValue: [5, 10, 15]
  },
  {
    name: 'weightProperty',
    label: 'Demand value column (empty counts points)',
    kind: 'string',
    maximumLength: 64,
    defaultValue: ''
  },
  SNAP_DISTANCE_PARAMETER,
  {
    name: 'bufferRadius',
    label: 'Map corridor radius (not added to travel time)',
    kind: 'number',
    minimum: 0.001,
    maximum: 1e6,
    unit: 'crs',
    defaultValue: 100
  },
  {
    name: 'gridSize',
    label: 'Isoband raster width and height',
    kind: 'integer',
    minimum: 8,
    maximum: DRIVE_TIME_BANDS_LIMITS.gridSize,
    defaultValue: 32
  }
];

export type PreparedCuratedDriveTimeBands = {
  graph: ReturnType<typeof buildNetworkGraph>;
  topology: ReturnType<typeof buildCatalogDirectedCsr>;
  frame: ReturnType<typeof makeLocalFrame>;
  nodePositions: Float32Array;
  facilities: CatalogFeatureCollection;
  demand: CatalogFeatureCollection;
  facilityPositions: Float32Array;
  demandPositions: Float32Array;
  demandValues: Float32Array;
  costBreaks: Float32Array;
  costUnit: string;
  maximumSnapDistance: number;
  bufferRadius: number;
  gridSize: number;
  extent: [number, number, number, number];
};

/** Prepare geometry and declared time costs once, before allocating any GPU buffers. */
export function prepareCuratedDriveTimeBands(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
): PreparedCuratedDriveTimeBands {
  const values = {
    ...Object.fromEntries(PARAMETERS.map(parameter => [parameter.name, parameter.defaultValue!])),
    ...parameters
  };
  for (const specification of PARAMETERS) {
    const value = values[specification.name];
    if (specification.kind === 'number-list') {
      validateRequirement(BINDING_ID, Array.isArray(value), 'costBreaks must be a numeric list');
      const breaks = value as readonly number[];
      validateCapacity(
        BINDING_ID,
        'costBreaks.length',
        breaks.length,
        1,
        DRIVE_TIME_BANDS_LIMITS.bands
      );
      breaks.forEach((upperBound, index) => {
        validateFiniteNumber(BINDING_ID, 'costBreaks', upperBound, 0, 1e9);
        validateRequirement(
          BINDING_ID,
          index === 0 || Math.fround(upperBound) > Math.fround(breaks[index - 1]),
          'costBreaks must be strictly ascending after float32 conversion'
        );
      });
    } else if (specification.kind === 'number' || specification.kind === 'integer') {
      validateRequirement(
        BINDING_ID,
        typeof value === 'number',
        `${specification.name} must be numeric`
      );
      validateFiniteNumber(
        BINDING_ID,
        specification.name,
        value as number,
        specification.minimum,
        specification.maximum
      );
      validateRequirement(
        BINDING_ID,
        specification.kind !== 'integer' || Number.isInteger(value),
        `${specification.name} must be an integer`
      );
    } else {
      validateRequirement(
        BINDING_ID,
        typeof value === 'string' &&
          value.length <= (specification.maximumLength ?? Infinity) &&
          (!specification.options || specification.options.includes(value)),
        `${specification.name} must be a string within its declared length and allowed values`
      );
    }
  }
  validateRequirement(
    BINDING_ID,
    Boolean(values['forwardCostProperty']),
    'forwardCostProperty must name travel time costs; length costs are not drive time'
  );
  for (const [name, maximumFeatures, maximumVertices, kinds] of [
    [
      'network',
      DRIVE_TIME_BANDS_LIMITS.features,
      DRIVE_TIME_BANDS_LIMITS.vertices,
      ['LineString', 'MultiLineString']
    ],
    [
      'facilities',
      DRIVE_TIME_BANDS_LIMITS.facilities,
      DRIVE_TIME_BANDS_LIMITS.facilities,
      ['Point']
    ],
    ['demand', DRIVE_TIME_BANDS_LIMITS.demand, DRIVE_TIME_BANDS_LIMITS.demand, ['Point']]
  ] as const) {
    const collection = inputs[name];
    validateRequirement(
      BINDING_ID,
      collection?.type === 'FeatureCollection' && Array.isArray(collection.features),
      `${name} must be a FeatureCollection`
    );
    validateCapacity(
      BINDING_ID,
      `${name} feature count`,
      collection.features.length,
      1,
      maximumFeatures
    );
    for (const feature of collection.features) {
      validateRequirement(
        BINDING_ID,
        (kinds as readonly string[]).includes(feature.geometry?.type ?? ''),
        `${name} geometry must be ${kinds.join(' or ')}`
      );
      // Inspect raw arrays before catalog readers can filter malformed vertices. Filtering here
      // would replace a broken street with a shortcut joining its surviving vertices.
      const coordinates = feature.geometry?.coordinates;
      const parts =
        name !== 'network'
          ? [[coordinates]]
          : feature.geometry?.type === 'MultiLineString'
            ? coordinates
            : [coordinates];
      validateRequirement(
        BINDING_ID,
        Array.isArray(parts) &&
          parts.length > 0 &&
          parts.every(
            part =>
              Array.isArray(part) &&
              part.length >= (name === 'network' ? 2 : 1) &&
              part.every(
                position =>
                  Array.isArray(position) &&
                  position.length >= 2 &&
                  position.every(
                    ordinate => typeof ordinate === 'number' && Number.isFinite(ordinate)
                  ) &&
                  (values['coordinateSystem'] !== 'EPSG:4326' ||
                    (Math.abs(position[0]) <= 180 && Math.abs(position[1]) <= 90))
              )
          ),
        `${name} coordinates must contain only finite numeric positions in the declared coordinateSystem`
      );
    }
    validateCapacity(
      BINDING_ID,
      `${name} vertex count`,
      collection.features.reduce((count, feature) => count + countFeatureVertices(feature), 0),
      1,
      maximumVertices
    );
  }
  // The existing builder owns one-way rules, shared-vertex connectivity, proportional cost
  // splitting and diagnostics. Exact vertex identities are retained (no vertex clustering).
  const graph = buildNetworkGraph(
    inputs['network'],
    {
      coordinateSystem: values['coordinateSystem'] as 'planar' | 'EPSG:4326',
      snapTolerance: 0,
      splitAtSharedVertices: true,
      costMode: 'property',
      forwardCostProperty: String(values['forwardCostProperty']),
      reverseCostProperty: String(values['reverseCostProperty']),
      onewayProperty: String(values['onewayProperty'])
    },
    DRIVE_TIME_BANDS_LIMITS.segments,
    BINDING_ID
  );
  validateCapacity(BINDING_ID, 'nodeCount', graph.nodes.length, 2, DRIVE_TIME_BANDS_LIMITS.nodes);
  // The recipe snaps to straight CSR arcs. Preserve every bend as a routing vertex instead of
  // shortcutting a polyline to its chord. Prorate its already-prepared cost by segment length.
  const nodeByPosition = new Map(
    graph.nodes.map((position, index) => [JSON.stringify([position[0], position[1]]), index])
  );
  const arcs: CatalogArc[] = [];
  let segmentCount = 0;
  for (const edge of graph.edges) {
    for (let index = 1; index < edge.coordinates.length; index++) {
      const from = nodeByPosition.get(JSON.stringify(edge.coordinates[index - 1].slice(0, 2)))!;
      const to = nodeByPosition.get(JSON.stringify(edge.coordinates[index].slice(0, 2)))!;
      const share = (edge.cumulative[index] - edge.cumulative[index - 1]) / edge.length;
      if (edge.forwardCost !== null) arcs.push({from, to, cost: edge.forwardCost * share});
      if (edge.reverseCost !== null)
        arcs.push({from: to, to: from, cost: edge.reverseCost * share});
      segmentCount++;
    }
  }
  validateCapacity(BINDING_ID, 'segmentCount', segmentCount, 1, DRIVE_TIME_BANDS_LIMITS.segments);
  validateCapacity(BINDING_ID, 'arcCount', arcs.length, 1, DRIVE_TIME_BANDS_LIMITS.arcs);
  const topology = buildCatalogDirectedCsr(graph.nodes.length, arcs);
  // Costs below this bound keep any supported simple path, including snap connectors, finite.
  validateRequirement(
    BINDING_ID,
    topology.weights.every(cost => Number.isFinite(cost) && cost >= 0 && cost <= 1e9),
    'travel time arc costs must be finite float32 values in [0, 1000000000]; rescale costs and costUnit'
  );
  const frame = makeLocalFrame(graph.coordinateSystem, graph.nodes, {shift: 'center'});
  const projectPositions = (positions: readonly (readonly number[])[]) =>
    Float32Array.from(
      positions.flatMap(position => {
        const projected = frame.forward(position as [number, number]);
        validateRequirement(
          BINDING_ID,
          projected.every(Number.isFinite),
          'coordinates must fit the local planar frame'
        );
        return projected;
      })
    );
  const nodePositions = projectPositions(graph.nodes);
  validateRequirement(
    BINDING_ID,
    nodePositions.every(Number.isFinite),
    'nodePositions must fit finite float32'
  );
  for (const arc of arcs)
    validateRequirement(
      BINDING_ID,
      nodePositions[2 * arc.from] !== nodePositions[2 * arc.to] ||
        nodePositions[2 * arc.from + 1] !== nodePositions[2 * arc.to + 1],
      'nodePositions must remain distinct along every arc after float32 conversion; reduce the study extent'
    );
  const demandValues = Float32Array.from(inputs['demand'].features, feature => {
    const value = values['weightProperty']
      ? feature.properties?.[String(values['weightProperty'])]
      : 1;
    validateRequirement(
      BINDING_ID,
      typeof value === 'number',
      'weightProperty must contain numeric values on every demand row'
    );
    validateFiniteNumber(BINDING_ID, 'weightProperty', value as number, 0, 1e12);
    return value as number;
  });
  const facilityPositions = projectPositions(
    inputs['facilities'].features.map(feature => getPointPositions(feature.geometry)[0])
  );
  const demandPositions = projectPositions(
    inputs['demand'].features.map(feature => getPointPositions(feature.geometry)[0])
  );
  validateRequirement(
    BINDING_ID,
    facilityPositions.every(Number.isFinite) && demandPositions.every(Number.isFinite),
    'facilityPositions and demandPositions must fit finite float32 in the local frame'
  );
  const bufferRadius = Number(values['bufferRadius']);
  const horizontal = Array.from(nodePositions).filter((_, index) => index % 2 === 0);
  const vertical = Array.from(nodePositions).filter((_, index) => index % 2 === 1);
  const gridSize = Number(values['gridSize']);
  // Keep one pixel of rounding headroom within the published 16-pixel half-window. The
  // splatter also expands a tiny radius to half a pixel diagonal, so bound raster aspect ratio.
  const supportedReach = DRIVE_TIME_BANDS_LIMITS.bufferPixels - 1;
  const minimumSpan = (gridSize * bufferRadius) / supportedReach;
  const horizontalSpan = Math.max(...horizontal) - Math.min(...horizontal) + 2 * bufferRadius;
  const verticalSpan = Math.max(...vertical) - Math.min(...vertical) + 2 * bufferRadius;
  const aspectSpan = Math.max(horizontalSpan, verticalSpan) / supportedReach;
  const width = Math.max(horizontalSpan, minimumSpan, aspectSpan);
  const height = Math.max(verticalSpan, minimumSpan, aspectSpan);
  const centerHorizontal = (Math.min(...horizontal) + Math.max(...horizontal)) / 2;
  const centerVertical = (Math.min(...vertical) + Math.max(...vertical)) / 2;
  return {
    graph,
    topology,
    frame,
    nodePositions,
    facilities: inputs['facilities'],
    demand: inputs['demand'],
    facilityPositions,
    demandPositions,
    demandValues,
    costBreaks: Float32Array.from(values['costBreaks'] as readonly number[]),
    costUnit: String(values['costUnit']),
    maximumSnapDistance: Number(values['maximumSnapDistance']),
    bufferRadius,
    gridSize,
    extent: [
      centerHorizontal - width / 2,
      centerVertical - height / 2,
      centerHorizontal + width / 2,
      centerVertical + height / 2
    ]
  };
}

export type CuratedDriveTimeBandsReadback = {
  times: ArrayLike<number>;
  demandBands: ArrayLike<number>;
  demandEdges: ArrayLike<number>;
  facilityEdges: ArrayLike<number>;
  keys: ArrayLike<number>;
  counts: ArrayLike<number>;
  sums: ArrayLike<number>;
  means: ArrayLike<number>;
  triangles: ArrayLike<number>;
  triangleBands: ArrayLike<number>;
  converged: number;
  bandCount: number;
  requiredBandCount: number;
  bandOverflow: number;
  triangleCount: number;
  triangleOverflow: number;
};

/** Publish exact network-space demand bands and explicitly approximate map isobands. */
export function publishCuratedDriveTimeBands(
  prepared: PreparedCuratedDriveTimeBands,
  readback: CuratedDriveTimeBandsReadback
): CatalogRunOutput {
  validateRequirement(
    BINDING_ID,
    readback.converged === 1,
    `service-areas did not converge within ${DRIVE_TIME_BANDS_LIMITS.iterations} iterations; reduce the network before publication`
  );
  validateRequirement(
    BINDING_ID,
    readback.bandOverflow === 0 && readback.triangleOverflow === 0,
    'band-statistics or isochrone-bands overflow; reduce the grid or band count before publication'
  );
  validateCapacity(BINDING_ID, 'bandCount', readback.bandCount, 0, prepared.costBreaks.length);
  validateRequirement(
    BINDING_ID,
    readback.requiredBandCount === readback.bandCount,
    'requiredBandCount must equal bandCount without overflow'
  );
  validateCapacity(
    BINDING_ID,
    'triangleCount',
    readback.triangleCount,
    0,
    DRIVE_TIME_BANDS_LIMITS.triangles
  );
  for (const [name, length] of [
    ['times', prepared.demand.features.length],
    ['demandBands', prepared.demand.features.length],
    ['demandEdges', prepared.demand.features.length],
    ['facilityEdges', prepared.facilities.features.length],
    ['keys', readback.bandCount],
    ['counts', readback.bandCount],
    ['sums', readback.bandCount],
    ['means', readback.bandCount],
    ['triangles', readback.triangleCount * 6],
    ['triangleBands', readback.triangleCount]
  ] as const)
    validateRequirement(
      BINDING_ID,
      readback[name].length >= length,
      `${name} readback must contain ${length} values`
    );
  for (let group = 0; group < readback.bandCount; group++) {
    validateCapacity(
      BINDING_ID,
      'band key',
      readback.keys[group],
      0,
      prepared.costBreaks.length - 1
    );
    validateRequirement(
      BINDING_ID,
      group === 0 || readback.keys[group] > readback.keys[group - 1],
      'band keys must be distinct and ascending'
    );
    validateCapacity(
      BINDING_ID,
      'band demand count',
      readback.counts[group],
      1,
      prepared.demand.features.length
    );
    validateFiniteNumber(
      BINDING_ID,
      'band demand sum',
      readback.sums[group],
      0,
      1e12 * prepared.demand.features.length
    );
    validateFiniteNumber(BINDING_ID, 'band demand mean', readback.means[group], 0, 1e12);
  }
  const demand = makeCatalogTable(
    'demand-bands',
    'Travel time to demand',
    {
      source_index: 'int32',
      source_id: 'string',
      snapped_arc: 'int32',
      travel_time: 'float64',
      band: 'int32',
      upper_bound: 'float64',
      demand_value: 'float64',
      cost_unit: 'string',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  prepared.demand.features.forEach((feature, row) => {
    const time = readback.times[row];
    const band = readback.demandBands[row];
    validateRequirement(
      BINDING_ID,
      (Number.isFinite(time) && time >= 0) || time === Infinity,
      'demandTimes must be non-negative finite times or positive Infinity'
    );
    const expectedBand = Number.isFinite(time)
      ? prepared.costBreaks.findIndex(limit => time <= limit)
      : -1;
    validateRequirement(
      BINDING_ID,
      band === (expectedBand < 0 ? GPU_DRIVE_TIME_CATCHMENT_NO_BAND : expectedBand),
      'demandBands must match the first inclusive costBreaks upper bound'
    );
    const snapped = readback.demandEdges[row] !== GPU_DRIVE_TIME_CATCHMENT_NO_BAND;
    validateRequirement(
      BINDING_ID,
      snapped || time === Infinity,
      'unsnapped demand must have an infinite travel time'
    );
    if (snapped)
      validateCapacity(
        BINDING_ID,
        'demand snapped arc',
        readback.demandEdges[row],
        0,
        prepared.topology.neighbors.length - 1
      );
    appendCatalogRow(demand, {
      source_index: row,
      source_id: getFeatureIdentity(feature),
      snapped_arc: snapped ? readback.demandEdges[row] : null,
      travel_time: Number.isFinite(time) ? time : null,
      band: expectedBand < 0 ? null : band,
      upper_bound: expectedBand < 0 ? null : prepared.costBreaks[band],
      demand_value: prepared.demandValues[row],
      cost_unit: prepared.costUnit,
      status: !snapped
        ? 'unsnapped'
        : !Number.isFinite(time)
          ? 'unreachable'
          : expectedBand < 0
            ? 'beyond-bands'
            : 'within-band',
      geometry: JSON.stringify(feature.geometry)
    });
  });
  const bands = makeCatalogTable('band-summary', 'Demand by disjoint travel time band', {
    band: 'int32',
    lower_bound: 'float64',
    upper_bound: 'float64',
    demand_count: 'int32',
    demand_sum: 'float64',
    demand_mean: 'float64',
    cost_unit: 'string'
  });
  for (let band = 0; band < prepared.costBreaks.length; band++) {
    const group = Array.from(readback.keys).slice(0, readback.bandCount).indexOf(band);
    appendCatalogRow(bands, {
      band,
      lower_bound: band === 0 ? 0 : prepared.costBreaks[band - 1],
      upper_bound: prepared.costBreaks[band],
      demand_count: group < 0 ? 0 : readback.counts[group],
      demand_sum: group < 0 ? 0 : readback.sums[group],
      demand_mean: group < 0 ? null : readback.means[group],
      cost_unit: prepared.costUnit
    });
  }
  const isobands = makeCatalogTable(
    'isobands',
    'Approximate travel time corridors',
    {
      triangle_index: 'int32',
      band: 'int32',
      upper_bound: 'float64',
      cost_unit: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  for (let triangle = 0; triangle < readback.triangleCount; triangle++) {
    const band = readback.triangleBands[triangle];
    validateCapacity(BINDING_ID, 'triangle band', band, 0, prepared.costBreaks.length - 1);
    const coordinates = Array.from({length: 3}, (_, vertex) => {
      const offset = triangle * 6 + vertex * 2;
      const point = prepared.frame.inverse(
        readback.triangles[offset],
        readback.triangles[offset + 1]
      );
      validateRequirement(
        BINDING_ID,
        point.every(Number.isFinite),
        'isoband coordinates must be finite'
      );
      return point;
    });
    appendCatalogRow(isobands, {
      triangle_index: triangle,
      band,
      upper_bound: prepared.costBreaks[band],
      cost_unit: prepared.costUnit,
      geometry: makeCatalogGeometryText('Polygon', [[...coordinates, coordinates[0]]])
    });
  }
  return {
    tables: [demand, bands, isobands],
    diagnostics: [
      ...prepared.graph.diagnostics,
      {
        code: 'drive-time-method',
        severity: 'info',
        count: prepared.demand.features.length,
        message:
          'Travel uses the minimum outward time across facilities along directed arcs. Edge snapping adds no off-road connector time. Exact network-space demand bands are disjoint and include their upper bound; unsnapped and unreachable times are null. Map triangles approximate buffered raster corridors, not exact catchment boundaries or demand containment.'
      }
    ],
    summary: {
      demandCount: demand.rowCount,
      facilityCount: prepared.facilities.features.length,
      unsnappedFacilityCount: Array.from(readback.facilityEdges).filter(
        edge => edge === GPU_DRIVE_TIME_CATCHMENT_NO_BAND
      ).length,
      bandCount: bands.rowCount,
      triangleCount: isobands.rowCount,
      costUnit: prepared.costUnit,
      coordinateSystem: prepared.graph.coordinateSystem,
      gridSize: prepared.gridSize,
      bufferRadius: prepared.bufferRadius,
      maximumSnapDistance: prepared.maximumSnapDistance,
      converged: 1
    }
  };
}

export const CURATED_DRIVE_TIME_BANDS_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Drive-time bands',
  domain: 'network',
  summary:
    'Snap facilities and demand to streets, compute directed travel times and summarize demand in disjoint bands with approximate map corridors.',
  capabilityIds: ['network-accessibility'],
  contributorNames: [
    'GPUNetworkSnapping',
    'GPUNetworkServiceAreas',
    'GPUNetworkIsochrones',
    'GPUGroupStatistics'
  ],
  gpuContributorNames: [
    'GPUNetworkSnapping',
    'GPUNetworkServiceAreas',
    'GPUNetworkIsochrones',
    'GPUGroupStatistics'
  ],
  engine: 'gpu',
  inputs: [
    {
      name: 'network',
      label: 'Street network',
      accepts: ['line'],
      maximumFeatures: DRIVE_TIME_BANDS_LIMITS.features,
      maximumVertices: DRIVE_TIME_BANDS_LIMITS.vertices
    },
    {
      name: 'facilities',
      label: 'Facilities',
      accepts: ['point'],
      maximumFeatures: DRIVE_TIME_BANDS_LIMITS.facilities,
      maximumVertices: DRIVE_TIME_BANDS_LIMITS.facilities
    },
    {
      name: 'demand',
      label: 'Demand points',
      accepts: ['point'],
      maximumFeatures: DRIVE_TIME_BANDS_LIMITS.demand,
      maximumVertices: DRIVE_TIME_BANDS_LIMITS.demand
    }
  ],
  parameters: PARAMETERS,
  outputs: [
    {
      tableId: 'demand-bands',
      label: 'Travel time to demand',
      geometry: 'point',
      identityColumns: ['source_index', 'source_id'],
      maximumRows: DRIVE_TIME_BANDS_LIMITS.demand
    },
    {
      tableId: 'band-summary',
      label: 'Demand by band',
      geometry: 'none',
      identityColumns: ['band'],
      maximumRows: DRIVE_TIME_BANDS_LIMITS.bands
    },
    {
      tableId: 'isobands',
      label: 'Approximate travel time corridors',
      geometry: 'polygon',
      identityColumns: ['triangle_index', 'band'],
      maximumRows: DRIVE_TIME_BANDS_LIMITS.triangles
    }
  ],
  cleanup:
    'The catalog GPU session releases all uploaded buffers, readbacks and compiled graphs in finally. Returned tables own plain arrays.',
  declaredBehavior: [
    'Travel costs must be a property in the declared seconds, minutes or hours; length is never labelled drive time.',
    'Shared vertices connect roads; crossings without a vertex remain disconnected. Bends become routing segments with proportionate time costs. One-way restrictions and asymmetric reverse costs are honored.',
    'Exact bounded edge snapping; points outside maximumSnapDistance are unsnapped. Demand uses the least outward travel time across all snapped facilities, with direct same-road travel included.',
    'The first band includes zero. Each later band excludes its lower bound and includes its upper bound. Empty bands are reported with count/sum zero and mean null.',
    'A nonconverged search or band/triangle overflow prevents publication.'
  ],
  limitation:
    'GPU required; no CPU fallback, turn penalties, traffic model, off-road connector times or crossing noding. Map triangles approximate a buffered raster and do not decide demand membership. Settings rebuild; exact snapping is bounded by input caps and routing by 64 iterations.',
  async runGpu(context, inputs, parameters) {
    const prepared = prepareCuratedDriveTimeBands(inputs, parameters);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const times = session.output('times', 'float32', prepared.demand.features.length);
      const demandBands = session.output('demand-bands', 'uint32', prepared.demand.features.length);
      const demandEdges = session.output('demand-edges', 'uint32', prepared.demand.features.length);
      const facilityEdges = session.output(
        'facility-edges',
        'uint32',
        prepared.facilities.features.length
      );
      const keys = session.output('keys', 'uint32', prepared.costBreaks.length);
      const counts = session.output('counts', 'uint32', prepared.costBreaks.length);
      const sums = session.output('sums', 'float32', prepared.costBreaks.length);
      const means = session.output('means', 'float32', prepared.costBreaks.length);
      const bandCount = session.output('band-count', 'uint32', 1);
      const requiredBandCount = session.output('required-band-count', 'uint32', 1);
      const bandOverflow = session.output('band-overflow', 'uint32', 1);
      const converged = session.output('converged', 'uint32', 1);
      const triangles = session.output(
        'triangles',
        'float32x2',
        3 * DRIVE_TIME_BANDS_LIMITS.triangles
      );
      const triangleBands = session.output(
        'triangle-bands',
        'uint32',
        DRIVE_TIME_BANDS_LIMITS.triangles
      );
      const triangleCount = session.output('triangle-count', 'uint32', 1);
      const triangleOverflow = session.output('triangle-overflow', 'uint32', 1);
      const costBreaks = session.input('breaks', prepared.costBreaks, 'float32');
      const recipe = addDriveTimeCatchmentRecipe(session.graph, {
        topology: {
          offsets: session.input('offsets', prepared.topology.offsets, 'uint32'),
          neighbors: session.input('neighbors', prepared.topology.neighbors, 'uint32'),
          weights: session.input('weights', prepared.topology.weights, 'float32'),
          nodePositions: session.input('positions', prepared.nodePositions, 'float32x2')
        },
        facilities: session.input('facilities', prepared.facilityPositions, 'float32x2'),
        demand: session.input('demand', prepared.demandPositions, 'float32x2'),
        demandValues: session.input('demand-values', prepared.demandValues, 'float32'),
        maximumSnapDistance: session.input(
          'snap-distance',
          Float32Array.of(prepared.maximumSnapDistance),
          'float32'
        ),
        maximumIterations: DRIVE_TIME_BANDS_LIMITS.iterations,
        costBreaks,
        bandCapacity: prepared.costBreaks.length,
        scratch: {facilityEdges: facilityEdges.view, demandEdges: demandEdges.view},
        output: {
          demandTimes: times.view,
          demandBands: demandBands.view,
          status: {serviceAreaConverged: converged.view},
          bands: {
            keys: keys.view,
            counts: counts.view,
            sumValues: sums.view,
            means: means.view,
            status: {
              count: bandCount.view,
              requiredCount: requiredBandCount.view,
              overflow: bandOverflow.view
            }
          }
        },
        isochrones: {
          costBreaks,
          parameters: session.input(
            'isochrone-parameters',
            getGPUNetworkIsochronesParameterValues({
              breakCount: prepared.costBreaks.length,
              extent: prepared.extent,
              bufferRadius: prepared.bufferRadius
            }),
            'float32'
          ),
          raster: {
            width: prepared.gridSize,
            height: prepared.gridSize,
            maximumBufferPixels: DRIVE_TIME_BANDS_LIMITS.bufferPixels,
            maximumSamplesPerEdge: 1024,
            output: {
              triangles: triangles.view,
              triangleBands: triangleBands.view,
              status: {count: triangleCount.view, overflow: triangleOverflow.view}
            }
          }
        }
      });
      session.run();
      const readback: CuratedDriveTimeBandsReadback = {
        times: await session.readFloat32(times),
        demandBands: await session.readUint32(demandBands),
        demandEdges: await session.readUint32(demandEdges),
        facilityEdges: await session.readUint32(facilityEdges),
        keys: await session.readUint32(keys),
        counts: await session.readUint32(counts),
        sums: await session.readFloat32(sums),
        means: await session.readFloat32(means),
        converged: await session.readScalar(converged),
        bandCount: await session.readScalar(bandCount),
        requiredBandCount: await session.readScalar(requiredBandCount),
        bandOverflow: await session.readScalar(bandOverflow),
        triangleCount: await session.readScalar(triangleCount),
        triangleOverflow: await session.readScalar(triangleOverflow),
        triangles: [],
        triangleBands: []
      };
      if (readback.triangleCount <= DRIVE_TIME_BANDS_LIMITS.triangles) {
        readback.triangles = await session.readFloat32(triangles, readback.triangleCount * 3);
        readback.triangleBands = await session.readUint32(triangleBands, readback.triangleCount);
      }
      return {
        ...publishCuratedDriveTimeBands(prepared, readback),
        dispatchedContributors: recipe.contributors.map(contributor => contributor.constructor.name)
      };
    });
  }
};
