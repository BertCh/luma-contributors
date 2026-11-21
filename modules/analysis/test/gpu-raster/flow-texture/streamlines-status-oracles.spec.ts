// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {GPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {GPUStreamlines} from '../../../src/raster-analysis/flow-texture/gpu-streamlines';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
const CASES = [
  {
    name: 'pending',
    maximumIterations: 1,
    pointCapacity: 3,
    velocity: 1,
    expected: [1, 1, 3, 0, 1, 1]
  },
  {
    name: 'complete',
    maximumIterations: 2,
    pointCapacity: 3,
    velocity: 1,
    expected: [1, 1, 3, 0, 0, 0]
  },
  {
    name: 'short-points',
    maximumIterations: 2,
    pointCapacity: 2,
    velocity: 1,
    expected: [0, 1, 0, 1, 0, 0]
  },
  {name: 'empty', maximumIterations: 2, pointCapacity: 3, velocity: 0, expected: [0, 0, 0, 0, 0, 0]}
];
const SETTINGS = {
  fieldPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
  gridPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
  stepLength: 0,
  seed: 1,
  minimumPoints: 3
};
// Closed form: four seeds compete for the single occupancy cell. A zero integration step
// gives exactly three identical points per valid seed. The first pruning round accepts only
// the priority winner and leaves other valid seeds undecided; round two rejects those occupied
// starts. Two point slots cannot hold the accepted three-point line. A zero field rejects all.

for (const scene of CASES) {
  it(`streamline forms pin complete status words for ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const options = {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 4,
      seedRows: 1,
      stepsPerDirection: 1,
      lineCapacity: 1,
      pointCapacity: scene.pointCapacity,
      maximumIterations: scene.maximumIterations
    };
    const parameters = getGPUStreamlinesParameterValues(SETTINGS);
    const words = getGPUStreamlinesUint32ParameterValues(SETTINGS);
    const rig = createParityGraph(device);
    const ports = ['count', 'required', 'points', 'overflow', 'candidate', 'limit'].map(name =>
      rig.output(name, 1, 'uint32')
    );
    rig.graph.add(
      new GPUStreamlines({
        ...options,
        velocities: rig.input('velocities', [scene.velocity, 0], 'float32x2').view,
        parameters: rig.input('parameters', Array.from(parameters), 'float32').view,
        uint32Parameters: rig.input('words', Array.from(words), 'uint32').view,
        output: {
          lines: {
            ids: rig.output('ids', 1, 'uint32').view,
            status: {
              count: ports[0].view,
              requiredCount: ports[1].view,
              overflow: ports[3].view,
              candidateOverflow: ports[4].view
            }
          },
          positionCount: ports[2].view,
          positions: rig.output('positions', scene.pointCapacity, 'float32x2').view,
          lineOffsets: rig.output('offsets', 2, 'uint32').view,
          status: {iterationLimitReached: ports[5].view}
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const port of ports) port.buffer.write(Uint32Array.of(12345));
      submitGraph(device, compiled, undefined);
      const actual = [];
      for (const port of ports) actual.push((await readUint32(port.buffer, 1))[0]);
      expect(actual).toEqual(scene.expected);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
    for (const Operation of [GPUStreamlinesOperation, CPUStreamlinesOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          velocities: {format: 'float32x2', values: Float32Array.of(scene.velocity, 0)},
          parameters: {format: 'float32', values: parameters},
          words: {format: 'uint32', values: words}
        },
        operations: [
          new Operation({
            ...options,
            velocities: 'velocities',
            parameters: 'parameters',
            uint32Parameters: 'words',
            output: {
              lines: {
                ids: 'ids',
                status: {
                  count: 'count',
                  requiredCount: 'required',
                  overflow: 'overflow',
                  candidateOverflow: 'candidate'
                }
              },
              positions: 'positions',
              lineOffsets: 'offsets',
              positionCount: 'points',
              status: {iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: ['count', 'required', 'points', 'overflow', 'candidate', 'limit']
      });
      expect(
        ['count', 'required', 'points', 'overflow', 'candidate', 'limit'].map(
          name => result[name][0]
        )
      ).toEqual(scene.expected);
    }
  });
}
