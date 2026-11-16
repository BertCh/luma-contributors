// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateFiniteNumber, validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {getNetworkParameters, type CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../analysis/catalog-geojson';
import {buildNetworkGraph} from '../analysis/catalog-network';
import type {KeplerDatasetNetworkProfile, KeplerDatasetProfile} from './dataset-profile-types';

/** Match preparation that a binding applies regardless of its exposed parameter values. */
function getEffectiveNetworkProfileParameters(
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  bindingId?: string
): Readonly<Record<string, CatalogParameterValue>> {
  if (bindingId === 'curated-drive-time-bands')
    return {
      ...parameters,
      snapTolerance: 0,
      splitAtSharedVertices: true,
      costMode: 'property',
      crossings: 'not-noded'
    };
  // Both snap engines share makeSnapOutput, which forces splitting and never nodes crossings.
  return bindingId === 'network-snap-points'
    ? {...parameters, splitAtSharedVertices: true, crossings: 'not-noded'}
    : parameters;
}

/**
 * Both routing engines reuse the catalog builder until GPU crossing noding changes the input
 * geometry. Its counts stay unknown until that output is profiled. CPU ignores `crossings`.
 */
export function getKeplerDatasetNetworkTopologyKey(
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  executionEngine: 'cpu' | 'gpu',
  bindingId?: string
): string | undefined {
  const effectiveParameters = getEffectiveNetworkProfileParameters(parameters, bindingId);
  if (executionEngine === 'gpu' && effectiveParameters['crossings'] === 'noded') return undefined;
  const network = getNetworkParameters(effectiveParameters);
  return JSON.stringify([
    network.coordinateSystem,
    network.snapTolerance,
    network.splitAtSharedVertices,
    network.costMode,
    network.costMode === 'property' ? network.forwardCostProperty : '',
    network.costMode === 'property' ? network.reverseCostProperty : '',
    network.onewayProperty,
    ...(bindingId === 'curated-drive-time-bands' ? ['street-segments'] : [])
  ]);
}

/**
 * Exact facts computed only on request, using the catalog builder's multipart, duplicate,
 * greedy input-order snapping, shared-vertex splitting and excluded-arc semantics.
 * Pass resolved parameters and bindingId for operation-specific preparation such as snap's
 * forced splitting. No binding edge cap is imposed: the full count must reject oversized graphs.
 * The host admits raw input before materializing a resident layer.
 */
export function profileKeplerDatasetNetwork(
  profile: KeplerDatasetProfile,
  lines: CatalogFeatureCollection,
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  executionEngine: 'cpu' | 'gpu' = 'cpu',
  bindingId?: string
): KeplerDatasetNetworkProfile | undefined {
  const topologyKey = getKeplerDatasetNetworkTopologyKey(parameters, executionEngine, bindingId);
  if (topologyKey === undefined) return undefined;
  const contributorId = 'profileKeplerDatasetNetwork';
  const network = getNetworkParameters(getEffectiveNetworkProfileParameters(parameters, bindingId));
  validateRequirement(
    contributorId,
    profile.catalogGeometry === 'line' || profile.kind === 'lines' || profile.kind === 'network',
    'profile.kind must describe line geometry'
  );
  validateRequirement(
    contributorId,
    network.coordinateSystem === 'planar' || network.coordinateSystem === 'EPSG:4326',
    'coordinateSystem must be planar or EPSG:4326'
  );
  validateFiniteNumber(contributorId, 'snapTolerance', network.snapTolerance, 0);
  validateRequirement(
    contributorId,
    typeof network.splitAtSharedVertices === 'boolean',
    'splitAtSharedVertices must be a boolean'
  );
  validateRequirement(
    contributorId,
    network.costMode === 'length' || network.costMode === 'property',
    'costMode must be length or property'
  );
  const graph = buildNetworkGraph(lines, network, Number.MAX_SAFE_INTEGER, contributorId);
  return Object.freeze({
    datasetId: profile.datasetId,
    revision: profile.revision,
    topologyKey,
    nodeCount: graph.nodes.length,
    edgeCount:
      bindingId === 'curated-drive-time-bands'
        ? graph.edges.reduce((count, edge) => count + edge.coordinates.length - 1, 0)
        : graph.edges.length,
    arcCount: graph.outgoing.reduce(
      (count, outgoing) =>
        count +
        outgoing.reduce(
          (arcCount, arc) =>
            arcCount +
            (bindingId === 'curated-drive-time-bands'
              ? graph.edges[arc.edge].coordinates.length - 1
              : 1),
          0
        ),
      0
    )
  });
}

/** Select only facts for this dataset revision, engine preparation and resolved settings. */
export function getKeplerDatasetNetworkProfile(
  profile: KeplerDatasetProfile,
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  executionEngine: 'cpu' | 'gpu',
  bindingId?: string
): KeplerDatasetNetworkProfile | undefined {
  const network = profile.network;
  if (!network) return undefined;
  const topologyKey = getKeplerDatasetNetworkTopologyKey(parameters, executionEngine, bindingId);
  return topologyKey !== undefined &&
    network.datasetId === profile.datasetId &&
    network.revision === profile.revision &&
    network.topologyKey === topologyKey
    ? network
    : undefined;
}
