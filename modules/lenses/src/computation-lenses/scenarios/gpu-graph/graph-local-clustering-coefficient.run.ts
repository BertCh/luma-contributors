// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  buildGPUGraphScenario,
  compileGPUGraphScenario,
  describeGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  type GPUGraphScenarioRecordOptions
} from './gpu-graph-scenario';
import {runExample} from './graph-local-clustering-coefficient.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-local-clustering-coefficient',
    directed: false,
    example: runExample
  });
  try {
    const outputCaptures = {
      'clustering-coefficient': {
        view: getGPUGraphCaptureView(fixture, 'clustering-coefficient', outputs.clustering),
        keySpace: 'vertex',
        stationValues: true
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-local-clustering-coefficient',
      element: 'GPUGraphAlgorithmsLocalClusteringCoefficient',
      fixture,
      ...compileGPUGraphScenario(fixture, outputCaptures),
      outputCaptures,
      choices: [
        {
          stage: 'parameters',
          name: 'algorithm',
          value: 'auto',
          alternatives: ['direct', 'canonical'],
          sweepable: false
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public clustering example and captures one coefficient per station. */
export async function run(device: Device) {
  return recordGPUGraphScenario(await buildScenario(device));
}

/** Builds the same compiled scenario on NullDevice, with no submission or fabricated values. */
export async function describeStructure(device: Device) {
  const options = await buildScenario(device);
  try {
    return describeGPUGraphScenario(options);
  } finally {
    try {
      options.snapshot.arena.destroy();
    } finally {
      try {
        options.compiled.destroy();
      } finally {
        options.fixture.destroy();
      }
    }
  }
}
