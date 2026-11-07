// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './edge-bundling.example';
import {
  CHICAGO_OD_FLOWS,
  CONSTRUCTION_DATASETS,
  gatherRows,
  getKeySpaces,
  sliceCapture
} from './construction-lens-support';

const scenarioId = 'gpu-network.edge-bundling';
const flows = CHICAGO_OD_FLOWS.annual;
const POINTS_PER_EDGE = 16;
const MAXIMUM_ITERATIONS = 8;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'lengthMeters',
  withoutRoads: true,
  seed: 0,
  datasetVersion: `${CONSTRUCTION_DATASETS.odFlows.id}-v1`,
  retain: ['bundledPaths', 'bundleOffsets'],
  formulas: [],
  keySpaces: getKeySpaces([
    {id: 'flow', label: 'origin-destination row', size: flows.trips.length},
    {id: 'zone', label: 'community-area centroid', size: CHICAGO_OD_FLOWS.zoneCount}
  ]),
  entityKeySpace: 'flow',
  choices: [
    {
      stage: 'presentation',
      name: 'maximumIterations',
      value: MAXIMUM_ITERATIONS,
      alternatives: [4, 15],
      sweepable: true
    },
    {
      stage: 'presentation',
      name: 'weights',
      value: 'ignored: one line per origin-destination row',
      alternatives: [],
      sweepable: false
    }
  ],
  async build(context) {
    const zonePositions = context.upload(
      'zonePositions',
      Float32Array.from(CHICAGO_OD_FLOWS.zonePositions),
      'float32x2'
    );
    const originZones = context.upload(
      'originZones',
      Uint32Array.from(flows.originZones),
      'uint32'
    );
    const destinationZones = context.upload(
      'destinationZones',
      Uint32Array.from(flows.destinationZones),
      'uint32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      zonePositions,
      originZones,
      destinationZones
    });
    return {
      views: {
        zonePositions,
        originZones,
        destinationZones,
        bundledPaths: example.paths,
        bundleOffsets: example.lineOffsets
      }
    };
  },
  capture(final) {
    const offsets = final('bundleOffsets');
    const lineGeometry = {
      kind: 'lines' as const,
      coordinateModel: 'geographic' as const,
      offsetsCapture: 'bundleOffsets'
    };
    const zones = final('zonePositions');
    const origins = final('originZones');
    const destinations = final('destinationZones');
    const flowCount = flows.trips.length;
    const straight = Float32Array.from(
      Array.from({length: flowCount}, (_, flow) => [
        ...gatherRows(zones, [origins.values[flow]]),
        ...gatherRows(zones, [destinations.values[flow]])
      ]).flat()
    );
    return [
      {
        ...zones,
        id: 'straightFlows',
        name: 'straightFlows',
        keySpace: undefined,
        values: straight,
        role: 'input',
        meaning:
          'input: one straight line per origin-destination row between community-area centroids (positions gathered from the zonePositions readback); trip counts are not read',
        geometry: {
          kind: 'lines',
          coordinateModel: 'geographic',
          offsetsCapture: 'straightFlowOffsets'
        }
      },
      {
        ...offsets,
        id: 'straightFlowOffsets',
        name: 'straightFlowOffsets',
        keySpace: undefined,
        elementType: 'u32',
        values: Uint32Array.from({length: flowCount + 1}, (_, flow) => flow * 2),
        meaning: 'derived: flow k spans vertices 2k to 2k + 2 of the straight input lines'
      },
      offsets,
      {
        ...final('bundledPaths'),
        role: 'output',
        meaning: `output: bundled paths after ${MAXIMUM_ITERATIONS} iterations (${POINTS_PER_EDGE} control points per flow, endpoints fixed)`,
        geometry: lineGeometry
      },
      {...origins, keySpace: 'flow', meaning: 'origin zone of each flow row'},
      {...destinations, keySpace: 'flow', meaning: 'destination zone of each flow row'},
      sliceCapture(offsets, 0, flowCount, {
        id: 'bundleStart',
        keySpace: 'flow',
        meaning: 'first path vertex of each flow (GPU line offsets)'
      })
    ];
  }
};

/** Executes the density edge bundling on taxi flows between Chicago community areas. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
