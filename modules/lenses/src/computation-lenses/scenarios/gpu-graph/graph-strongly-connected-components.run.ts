// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import {
  buildGPUGraphScenario,
  describeGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  type GPUGraphScenarioRecordOptions
} from './gpu-graph-scenario';
import {runExample} from './graph-strongly-connected-components.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-strongly-connected-components',
    directed: true,
    example: runExample
  });
  try {
    // The public contributor returns the caller-supplied convergence scalar in this example.
    assert(outputs.converged);
    const outputCaptures = {
      'strongly-connected-components': {
        view: getGPUGraphCaptureView(
          fixture,
          'strongly-connected-components',
          outputs.componentLabels
        ),
        keySpace: 'vertex',
        stationValues: true
      },
      'strongly-connected-components-converged': {
        view: getGPUGraphCaptureView(
          fixture,
          'strongly-connected-components-converged',
          outputs.converged
        )
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-strongly-connected-components',
      element: 'GPUGraphAlgorithmsStronglyConnectedComponents',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      iterationFrames: [
        {
          name: 'strongly-connected-components',
          loopId: 'gpu-graph-strongly-connected-components-iterations'
        }
      ],
      choices: [{stage: 'parameters', name: 'iterations', value: 48, sweepable: true}],
      highlight: {
        capture: 'strongly-connected-components',
        select: (values, railFixture) => {
          // Labels are the lowest vertex id of each strong component, so the most frequent label names
          // the largest component's root station; ties resolve to the lowest label.
          const sizes = new Map<number, number>();
          for (let index = 0; index < values.length; index++) {
            sizes.set(values[index], (sizes.get(values[index]) ?? 0) + 1);
          }
          let key = -1;
          let size = 0;
          for (const [label, count] of sizes) {
            if (count > size || (count === size && label < key)) [key, size] = [label, count];
          }
          // A converged run labels every station with a vertex id; the overflow sentinel is not one.
          assert(key >= 0 && key < railFixture.stations.vertexCount);
          return {key, label: `${railFixture.stations.names[key]}: root of ${size} stations`};
        }
      },
      series: captures => [
        {
          name: 'converged (1 = fixed point)',
          axis: 'publication',
          values: [captures['strongly-connected-components-converged'].values[0]]
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public strongly-connected-components example and captures its real buffers and rounds. */
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
      options.compiled.destroy();
    } finally {
      options.fixture.destroy();
    }
  }
}
