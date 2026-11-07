// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './track-quality-recipe.example';

const scenario: NetworkElementScenario = {
  id: 'gpu-network.track-quality-recipe',
  weights: 'lengthMeters',
  withoutRoads: true,
  roadContext: false,
  seed: 0,
  datasetVersion: 'synthetic-four-observation-trajectory-v1',
  retain: [
    'track-quality-recipe-speeds',
    'track-quality-recipe-lengths',
    'track-quality-recipe-durations',
    'track-quality-recipe-maximum-speeds',
    'track-quality-recipe-flags',
    'track-quality-recipe-flagged-counts'
  ],
  formulas: [],
  keySpaces: [
    {id: 'observation', label: 'Raw observation', size: 4, indexSpace: '1d'},
    {id: 'track', label: 'Track', size: 1, indexSpace: '1d'},
    {id: 'track-boundary', label: 'Track boundary offset', size: 2, indexSpace: '1d'},
    {id: 'threshold', label: 'Speed and time-gap thresholds', size: 2, indexSpace: '1d'}
  ],
  entityKeySpace: 'observation',
  choices: [
    {stage: 'representation', name: 'coordinateModel', value: 'projected', sweepable: false},
    {
      stage: 'summary',
      name: 'qualityThresholds',
      value: 'maximumSpeed=1.5; maximumGap=5',
      sweepable: false
    }
  ],
  async build(context) {
    const positions = context.upload(
      'positions',
      new Float32Array([0, 0, 1, 0, 2, 0, 20, 0]),
      'float32x2'
    );
    const timestamps = context.upload('timestamps', new Float32Array([0, 0, 1, 10]), 'float32');
    const trackOffsets = context.upload('trackOffsets', new Uint32Array([0, 4]), 'uint32');
    const thresholds = context.upload('thresholds', new Float32Array([1.5, 5]), 'float32');
    const result = await runExample(context.graph, {
      trajectory: {positions, timestamps, trackOffsets},
      thresholds
    });
    return {
      views: {
        positions,
        timestamps,
        trackOffsets,
        thresholds,
        ...result.outputs.metrics,
        flags: result.outputs.flags,
        flaggedCounts: result.outputs.flaggedCounts
      }
    };
  },
  capture(final) {
    return [
      {
        ...final('positions'),
        keySpace: 'observation',
        role: 'input',
        geometry: {kind: 'points', coordinateModel: 'projected'}
      },
      {...final('timestamps'), keySpace: 'observation', role: 'input'},
      {...final('trackOffsets'), keySpace: 'track-boundary', role: 'input'},
      {...final('thresholds'), keySpace: 'threshold', role: 'input'},
      ...['stepSpeeds', 'flags'].map(name => ({
        ...final(name),
        keySpace: 'observation',
        role: 'output' as const
      })),
      ...['trackLengths', 'trackDurations', 'maximumSpeeds', 'flaggedCounts'].map(name => ({
        ...final(name),
        keySpace: 'track',
        role: 'output' as const
      }))
    ];
  }
};
/** Executes and reads the actual network recipe on raw observations. */
export function run(device: Device) {
  return recordNetworkElement(device, scenario);
}
/** Builds the same graph without GPU execution. */
export function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
