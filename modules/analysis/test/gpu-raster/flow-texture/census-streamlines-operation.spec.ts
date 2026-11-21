// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {CPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {GPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {decodeStreamlineCase, streamlineCases} from '../../oracles/parity-wp4-round3-fixtures';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectCensusValues} from '../census-raster-oracle-utils';

for (const withCandidates of [false, true]) {
  for (const Operation of [GPUStreamlinesOperation, CPUStreamlinesOperation]) {
    it(`census ${Operation.name} reads every port against SciPy/Philox (candidates=${withCandidates})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Existing committed oracle: scipy.solve_ivp/map_coordinates and randomgen.Philox,
      // with independent greedy spacing; no contributor/CPU-twin output supplies expectations.
      const encoded = streamlineCases.find(
        scene => scene.name === 'horizontal-boundaries-and-competing-seeds'
      )!;
      const scene = await decodeStreamlineCase(encoded);
      const result = await runOperationGraph(device, {
        inputs: {
          velocities: {format: 'float32x2', values: Float32Array.from(scene.field)},
          parameters: {
            format: 'float32',
            values: getGPUStreamlinesParameterValues(scene.settings)
          },
          words: {format: 'uint32', values: getGPUStreamlinesUint32ParameterValues(scene.settings)}
        },
        operations: [
          new Operation({
            ...scene.config,
            fieldWidth: scene.fieldWidth,
            fieldHeight: scene.fieldHeight,
            maximumIterations: 64,
            velocities: 'velocities',
            parameters: 'parameters',
            uint32Parameters: 'words',
            output: {
              lines: {
                ids: 'ids',
                status: {
                  count: 'count',
                  overflow: 'overflow',
                  ...(withCandidates
                    ? {requiredCount: 'requiredCount', candidateOverflow: 'candidateOverflow'}
                    : {})
                }
              },
              lineOffsets: 'offsets',
              positions: 'positions',
              positionCount: 'positionCount',
              ...(withCandidates ? {status: {iterationLimitReached: 'limit'}} : {}),
              ...(withCandidates
                ? {candidates: {positions: 'candidatePositions', spans: 'candidateSpans'}}
                : {})
            }
          })
        ],
        outputs: [
          'ids',
          'count',
          'overflow',
          'offsets',
          'positions',
          'positionCount',
          ...(withCandidates
            ? [
                'requiredCount',
                'candidateOverflow',
                'limit',
                'candidatePositions',
                'candidateSpans'
              ]
            : [])
        ]
      });
      expectCensusValues(result['count'], [scene.ids.length]);
      expectCensusValues(result['overflow'], [0]);
      expectCensusValues(result['positionCount'], [scene.points.length / 2]);
      expectCensusValues(result['ids'].slice(0, scene.ids.length), scene.ids);
      expectCensusValues(result['offsets'].slice(0, scene.offsets.length), scene.offsets);
      // Fixed 32x32 field, 32 steps per direction: 5e-4 world units allows accumulated
      // float32 interpolation/seed rounding; integer topology/status comparisons remain exact.
      expectCensusValues(result['positions'].slice(0, scene.points.length), scene.points, 5e-4);
      if (withCandidates) {
        expectCensusValues(result['requiredCount'], [scene.ids.length]);
        expectCensusValues(result['candidateOverflow'], [0]);
        expectCensusValues(result['limit'], [0]);
        expectCensusValues(result['candidatePositions'], scene.candidates, 5e-4);
        const seedCount = scene.config.seedColumns * scene.config.seedRows;
        for (let seed = 0; seed < seedCount; seed++)
          expectCensusValues(
            result['candidateSpans'].slice(seed * 4, seed * 4 + 2),
            scene.spans.slice(seed * 2, seed * 2 + 2)
          );
        for (const seed of scene.ids)
          expectCensusValues(
            result['candidateSpans'].slice(seed * 4 + 2, seed * 4 + 4),
            scene.kept.slice(seed * 2, seed * 2 + 2)
          );
      }
    });
  }
}
