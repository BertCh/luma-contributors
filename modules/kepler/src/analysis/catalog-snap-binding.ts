// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  LINE_INPUT,
  MAXIMUM_NETWORK_EDGES,
  NETWORK_PARAMETERS,
  SNAP_DISTANCE_PARAMETER,
  getNetworkParameters,
  type CatalogGpuContext,
  type CatalogInputSpec,
  type CatalogOperationBinding,
  type CatalogParameterValue,
  type CatalogRunOutput
} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {snapCatalogPointsOnGpu} from './catalog-gpu-snapping';
import {buildNetworkGraph, snapPointToNetwork, type SnappedPoint} from './catalog-network';

type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;
type Values = Readonly<Record<string, CatalogParameterValue>>;

const POINTS_INPUT: CatalogInputSpec = {
  name: 'points',
  label: 'Points to snap',
  accepts: ['point'],
  maximumFeatures: 5000,
  maximumVertices: 5000
};

function makeSnapOutput(
  inputs: Inputs,
  values: Values,
  snap: (
    graph: ReturnType<typeof buildNetworkGraph>,
    points: CatalogPosition[]
  ) => (SnappedPoint | null)[]
): CatalogRunOutput;
function makeSnapOutput(
  inputs: Inputs,
  values: Values,
  snap: (
    graph: ReturnType<typeof buildNetworkGraph>,
    points: CatalogPosition[]
  ) => Promise<(SnappedPoint | null)[]>
): Promise<CatalogRunOutput>;
function makeSnapOutput(
  inputs: Inputs,
  values: Values,
  snap: (
    graph: ReturnType<typeof buildNetworkGraph>,
    points: CatalogPosition[]
  ) => (SnappedPoint | null)[] | Promise<(SnappedPoint | null)[]>
): CatalogRunOutput | Promise<CatalogRunOutput> {
  const graph = buildNetworkGraph(
    inputs['lines'],
    {...getNetworkParameters(values), splitAtSharedVertices: true},
    MAXIMUM_NETWORK_EDGES
  );
  const rows: {featureIndex: number; position: CatalogPosition}[] = [];
  inputs['points'].features.forEach((feature, featureIndex) => {
    for (const position of getPointPositions(feature.geometry)) rows.push({featureIndex, position});
  });
  const finish = (snapped: (SnappedPoint | null)[]): CatalogRunOutput => {
    const table = makeCatalogTable(
      'snapped-points',
      'Snapped points',
      {
        point_index: 'int32',
        source_index: 'int32',
        source_id: 'string',
        status: 'string',
        edge_id: 'string',
        edge_source_index: 'int32',
        edge_source_id: 'string',
        fraction: 'float64',
        distance: 'float64',
        geometry: 'string'
      },
      'geometry'
    );
    const diagnostics: CatalogDiagnostic[] = [...graph.diagnostics];
    rows.forEach((row, pointIndex) => {
      const result = snapped[pointIndex];
      const edge = result ? graph.edges[result.edge] : undefined;
      if (!result) {
        addCatalogDiagnostic(
          diagnostics,
          'point-not-snapped',
          'warning',
          'These points have no network edge within the maximum snap distance; they keep their own position.'
        );
      }
      appendCatalogRow(table, {
        point_index: pointIndex,
        source_index: row.featureIndex,
        source_id: getFeatureIdentity(inputs['points'].features[row.featureIndex]),
        status: result ? 'snapped' : 'unsnapped',
        edge_id: edge?.edgeId ?? null,
        edge_source_index: edge?.sourceIndex ?? null,
        edge_source_id: edge?.sourceId ?? null,
        fraction: result?.fraction ?? null,
        distance: result?.distance ?? null,
        geometry: makeCatalogGeometryText('Point', result?.position ?? row.position)
      });
    });
    addCatalogDiagnostic(
      diagnostics,
      'snap-distance-units',
      'info',
      graph.coordinateSystem === 'planar'
        ? 'Distances are in coordinate units.'
        : 'Distances are meters in a local equirectangular frame around the network.'
    );
    return {
      tables: [table],
      diagnostics,
      summary: {
        pointCount: rows.length,
        snappedCount: snapped.filter(Boolean).length,
        edgeCount: graph.edges.length
      }
    };
  };
  const points = rows.map(row => row.position);
  const snapped = snap(graph, points);
  return Array.isArray(snapped) ? finish(snapped) : snapped.then(finish);
}

/** CPU snap used as the oracle and as the fallback of the GPUNetworkSnapping path. */
export function runSnapPoints(inputs: Inputs, values: Values): CatalogRunOutput {
  return makeSnapOutput(inputs, values, (graph, points) =>
    points.map(point => snapPointToNetwork(graph, point, values['maximumSnapDistance'] as number))
  );
}

export function runGpuSnapPoints(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  let search = 'scan';
  return makeSnapOutput(inputs, values, (graph, points) =>
    snapCatalogPointsOnGpu(context, graph, points, values['maximumSnapDistance'] as number, {
      onSearch: ran => {
        search = ran;
      }
    })
  ).then(output => ({
    ...output,
    summary: {...output.summary, snap_search: search},
    dispatchedContributors: ['GPUNetworkSnapping']
  }));
}

export const SNAP_POINTS_CPU_BINDING: Omit<CatalogOperationBinding, 'gpuContributorNames'> = {
  id: 'network-snap-points',
  title: 'Snap points to network',
  domain: 'network',
  summary:
    'Moves points onto the nearest network edge with the edge identity, fraction along it and snap distance.',
  capabilityIds: [],
  contributorNames: ['GPUNetworkSnapping'],
  engine: 'cpu-reference',
  inputs: [LINE_INPUT, POINTS_INPUT],
  parameters: [...NETWORK_PARAMETERS, SNAP_DISTANCE_PARAMETER],
  outputs: [
    {
      tableId: 'snapped-points',
      label: 'Snapped points',
      geometry: 'point',
      identityColumns: ['point_index', 'source_index', 'source_id', 'edge_id'],
      maximumRows: 5000
    }
  ],
  cleanup:
    'CPU only; no retained GPU resources. Published results are removed through native Sources.',
  declaredBehavior: [
    'Every point is snapped to the nearest polyline segment within the maximum snap distance; ties go to the lowest edge and segment.',
    'Unsnapped points stay at their own position with status unsnapped and null edge fields.',
    'fraction is measured along the whole edge from its first node; distance is meters (EPSG:4326, local frame) or coordinate units (planar).',
    'More than 50,000 edges throws before any result is produced.'
  ],
  limitation:
    'The GPU engine scans every segment for every point (points x segments, no spatial index); float32 distances and fractions, so near-ties can choose a different segment than the float64 CPU engine. At most 5,000 points.',
  run: (inputs, values) => runSnapPoints(inputs, values)
};
