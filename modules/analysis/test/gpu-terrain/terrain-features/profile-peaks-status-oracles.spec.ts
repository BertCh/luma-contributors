// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUProfilePeaks} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {GPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  PROFILE_STATUS_PORTS,
  PROFILE_STATUS_VALUES,
  PROFILE_STATUS_CASES,
  PROFILE_EMPTY_STATUS,
  createProfileStatusProps
} from './profile-peaks-status-oracles-fixtures';

for (const testCase of PROFILE_STATUS_CASES) {
  it(`GPU profile status overwrites poison and clears on replay: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const values = createInputBuffer(device, new Float32Array(PROFILE_STATUS_VALUES));
    const validity = createInputBuffer(device, new Uint32Array(10).fill(1));
    const offsets = createInputBuffer(device, new Uint32Array([0, 10]));
    const parameters = createInputBuffer(device, new Float32Array([1]));
    const ids = createOutputBuffer(device, testCase.capacity);
    ids.write(new Uint32Array(testCase.capacity).fill(17));
    const statusBuffers = Object.fromEntries(
      PROFILE_STATUS_PORTS.map(port => {
        const buffer = createOutputBuffer(device, 1);
        buffer.write(new Uint32Array([17]));
        return [port, buffer];
      })
    );
    const statusViews = Object.fromEntries(
      PROFILE_STATUS_PORTS.map(port => [
        port,
        importGraphBuffer(graph, port, statusBuffers[port], 'uint32', 1)
      ])
    );
    graph.add(
      new GPUProfilePeaks({
        values: importGraphBuffer(graph, 'values', values, 'float32', 10),
        validity: importGraphBuffer(graph, 'validity', validity, 'uint32', 10),
        offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 1),
        window: 2,
        minimumSide: 1,
        suppressionRadius: 0,
        maximumSuppressionIterations: testCase.maximumSuppressionIterations,
        output: {
          peaks: {
            ids: importGraphBuffer(graph, 'ids', ids, 'uint32', testCase.capacity),
            status: {
              count: statusViews['count'],
              requiredCount: statusViews['requiredCount'],
              overflow: statusViews['overflow'],
              candidateOverflow: statusViews['candidateOverflow']
            }
          },
          status: {
            converged: statusViews['converged'],
            iterationLimitReached: statusViews['iterationLimitReached']
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const hasCandidates of [true, false, true]) {
        validity.write(new Uint32Array(10).fill(Number(hasCandidates)));
        submitGraph(device, compiled, undefined);
        const actual = await Promise.all(
          PROFILE_STATUS_PORTS.map(port => readUint32(statusBuffers[port], 1))
        );
        const expected = hasCandidates ? testCase.expected : PROFILE_EMPTY_STATUS;
        expect(actual).toEqual(expected.map(value => [value]));
      }
    } finally {
      compiled.destroy();
      for (const buffer of [
        values,
        validity,
        offsets,
        parameters,
        ids,
        ...Object.values(statusBuffers)
      ])
        buffer.destroy();
    }
  });

  it(`GPU profile operation status has the same independent oracle: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const hasCandidates of [true, false]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: new Float32Array(PROFILE_STATUS_VALUES)},
          validity: {format: 'uint32', values: new Uint32Array(10).fill(Number(hasCandidates))},
          offsets: {format: 'uint32', values: new Uint32Array([0, 10])},
          parameters: {format: 'float32', values: new Float32Array([1])}
        },
        operations: [new GPUProfilePeaksOperation(createProfileStatusProps(testCase))],
        outputs: PROFILE_STATUS_PORTS
      });
      const expected = hasCandidates ? testCase.expected : PROFILE_EMPTY_STATUS;
      expect(PROFILE_STATUS_PORTS.map(port => Array.from(result[port]))).toEqual(
        expected.map(value => [value])
      );
    }
  });
}
