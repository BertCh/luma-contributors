// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {test} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {getNetworkParameters} from '../../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {getNetworkGraphArcs, nodeCatalogLines} from '../../src/analysis/catalog-gpu-network';
import {buildNetworkGraph} from '../../src/analysis/catalog-network';
import {profileKeplerDatasetNetwork} from '../../src/profile/dataset-network-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

// Readable failure reports are necessary because the headless runner hides console output.
test('network profile matches GPU routing preparation and refuses crossing-noded counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [
        [0, 0],
        [2, 0]
      ],
      [
        [1, -1],
        [1, 1]
      ]
    ].map(coordinates => ({geometry: {type: 'LineString', coordinates}, properties: {}}))
  };
  const source: KeplerDatasetProfile = {
    datasetId: 'crossings',
    label: 'Crossings',
    revision: 1,
    kind: 'lines',
    rowCount: 2,
    geometryColumns: [],
    columns: [],
    catalogGeometry: 'line'
  };
  const parameters = {
    coordinateSystem: 'planar',
    snapTolerance: 0,
    splitAtSharedVertices: true,
    costMode: 'length',
    forwardCostProperty: '',
    reverseCostProperty: '',
    onewayProperty: '',
    crossings: 'not-noded'
  };
  const facts = profileKeplerDatasetNetwork(source, lines, parameters, 'gpu');
  const graph = buildNetworkGraph(lines, getNetworkParameters(parameters), 50000);
  const noded = await nodeCatalogLines({device}, lines, {
    coordinateSystem: 'planar',
    snapTolerance: 0
  });
  const nodedGraph = buildNetworkGraph(noded.pieces, getNetworkParameters(parameters), 50000);
  const unsupported = profileKeplerDatasetNetwork(
    source,
    lines,
    {...parameters, crossings: 'noded'},
    'gpu'
  );
  const report = {
    facts,
    ordinary: {
      nodes: graph.nodes.length,
      edges: graph.edges.length,
      arcs: getNetworkGraphArcs(graph).length
    },
    noded: {
      nodes: nodedGraph.nodes.length,
      edges: nodedGraph.edges.length,
      arcs: getNetworkGraphArcs(nodedGraph).length
    },
    unsupported
  };
  if (
    facts?.nodeCount !== 4 ||
    facts.edgeCount !== 2 ||
    facts.arcCount !== 4 ||
    graph.nodes.length !== facts.nodeCount ||
    graph.edges.length !== facts.edgeCount ||
    getNetworkGraphArcs(graph).length !== facts.arcCount ||
    nodedGraph.nodes.length !== 5 ||
    nodedGraph.edges.length !== 4 ||
    getNetworkGraphArcs(nodedGraph).length !== 8 ||
    unsupported !== undefined
  )
    throw new Error(`AX-82 GPU network profile mismatch: ${JSON.stringify(report)}`);
});
