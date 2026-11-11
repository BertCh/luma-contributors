// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUStreamlinesOperation} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  GPUStreamlinesOperation,
  type StreamlinesOperationProps
} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ['ordinary', 'varying', 'overflow', 'empty'] as const)
  it(`streamline operations agree on ${scene} traces`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: StreamlinesOperationProps = {
      fieldWidth: 8,
      fieldHeight: 8,
      gridWidth: scene === 'varying' ? 1 : 8,
      gridHeight: scene === 'varying' ? 1 : 8,
      seedColumns: scene === 'varying' ? 1 : 8,
      seedRows: scene === 'varying' ? 1 : 8,
      stepsPerDirection: scene === 'varying' ? 10 : 16,
      maximumIterations: 64,
      lineCapacity: scene === 'overflow' ? 1 : 64,
      pointCapacity: 2112,
      velocities: 'velocities',
      parameters: 'parameters',
      uint32Parameters: 'uint32Parameters',
      output: {
        lines: {
          ids: 'ids',
          status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
        },
        positions: 'positions',
        lineOffsets: 'lineOffsets',
        positionCount: 'positionCount',
        status: {iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const outputs = [
      'ids',
      'count',
      'requiredCount',
      'overflow',
      'positions',
      'lineOffsets',
      'positionCount',
      'iterationLimitReached'
    ];
    const results = [];
    for (const Operation of [GPUStreamlinesOperation, CPUStreamlinesOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            velocities: {
              format: 'float32x2',
              values: Float32Array.from({length: 128}, (_, component) => {
                if (scene === 'varying')
                  return component % 2 === 0 ? 1 : (Math.floor(component / 2) % 8) + 0.5;
                return component % 2 === 0 && scene !== 'empty' ? 3 : 0;
              })
            },
            parameters: {
              format: 'float32',
              values: getGPUStreamlinesParameterValues({
                fieldExtent: [0, 0, 1, 1],
                gridExtent: scene === 'varying' ? [0, 0, 4, 4] : [0, 0, 1, 1],
                stepLength: scene === 'varying' ? 0.05 : 0.25,
                seed: 1,
                minimumPoints: 2
              })
            },
            uint32Parameters: {
              format: 'uint32',
              values: getGPUStreamlinesUint32ParameterValues({
                seed: scene === 'varying' ? 3 : 1,
                minimumPoints: 2
              })
            }
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    if (scene === 'ordinary') {
      expect(results[1]['count'][0]).toBeGreaterThan(1);
      expect(results[1]['count'][0]).toBeLessThan(64);
      expect(results[1]['positionCount'][0]).toBeGreaterThan(results[1]['count'][0] * 2);
    }
    if (scene === 'varying') {
      expect(results[1]['count'][0]).toBe(1);
      expect(results[1]['positionCount'][0]).toBe(21);
      for (const result of results) {
        const centerX = result['positions'][20];
        const centerY = result['positions'][21];
        const invariant = centerY - (centerX * centerX) / 2;
        for (let point = 0; point < 21; point++)
          expect(
            Math.abs(
              result['positions'][point * 2 + 1] -
                result['positions'][point * 2] ** 2 / 2 -
                invariant
            )
          ).toBeLessThan(2e-5);
      }
    }
    for (const port of outputs) {
      const count = results[1]['count'][0];
      const length =
        port === 'ids'
          ? count
          : port === 'lineOffsets'
            ? count + 1
            : port === 'positions'
              ? results[1]['positionCount'][0] * 2
              : 1;
      for (let index = 0; index < length; index++) {
        const actual = results[0][port][index];
        const expected = results[1][port][index];
        // Placement magnitude at most eight: absolute position tolerance 2.4e-4.
        if (port === 'positions')
          expect(Math.abs(actual - expected)).toBeLessThan(scene === 'varying' ? 1e-5 : 2.4e-4);
        else expect(actual === expected).toBe(true);
      }
    }
  });

for (const scene of ['interior', 'boundary-seed', 'boundary-step'] as const)
  it(`streamline candidate diagnostic ports with ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const boundary = scene === 'boundary-seed';
    const extent = boundary
      ? ([16777216, 0, 2, 2] as const)
      : scene === 'boundary-step'
        ? ([0, 0, 25, 25] as const)
        : ([0, 0, 10, 10] as const);
    const props: StreamlinesOperationProps = {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 1,
      gridHeight: 1,
      seedColumns: 1,
      seedRows: 1,
      stepsPerDirection: 1,
      lineCapacity: 1,
      pointCapacity: 3,
      velocities: 'velocities',
      parameters: 'parameters',
      uint32Parameters: 'uint32Parameters',
      output: {
        lines: {
          ids: 'ids',
          status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
        },
        positions: 'positions',
        lineOffsets: 'lineOffsets',
        positionCount: 'positionCount',
        candidates: {positions: 'candidatePositions', spans: 'candidateSpans'},
        status: {iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const outputs = [
      'ids',
      'count',
      'requiredCount',
      'overflow',
      'positions',
      'lineOffsets',
      'positionCount',
      'candidatePositions',
      'candidateSpans',
      'iterationLimitReached'
    ];
    const results = [];
    for (const Operation of [GPUStreamlinesOperation, CPUStreamlinesOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            velocities: {format: 'float32x2', values: Float32Array.of(1, 0)},
            parameters: {
              format: 'float32',
              values: getGPUStreamlinesParameterValues({
                fieldExtent: extent,
                gridExtent: extent,
                stepLength: scene === 'boundary-step' ? 7.925207614898682 : 0.125
              })
            },
            uint32Parameters: {
              format: 'uint32',
              values: getGPUStreamlinesUint32ParameterValues({
                seed: boundary ? 0 : 3,
                minimumPoints: 1
              })
            }
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const result of results) {
      expect(result['count'][0]).toBe(boundary ? 0 : 1);
      expect(result['requiredCount'][0]).toBe(boundary ? 0 : 1);
      expect(result['positionCount'][0]).toBe(boundary ? 0 : scene === 'boundary-step' ? 2 : 3);
      expect(Array.from(result['candidateSpans'])).toEqual(
        boundary ? [0, 0, 0, 0] : scene === 'boundary-step' ? [1, 0, 1, 0] : [1, 1, 1, 1]
      );
      if (boundary) expect(result['candidatePositions'].every(Number.isNaN)).toBe(true);
    }
    for (const port of outputs) {
      const count = results[1]['count'][0];
      const length =
        port === 'ids'
          ? count
          : port === 'lineOffsets'
            ? count + 1
            : port === 'positions'
              ? results[1]['positionCount'][0] * 2
              : port === 'candidatePositions'
                ? 6
                : port === 'candidateSpans'
                  ? 4
                  : 1;
      for (let index = 0; index < length; index++) {
        const actual = results[0][port][index];
        const expected = results[1][port][index];
        if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
        else if (port === 'positions' || port === 'candidatePositions')
          expect(Math.abs(actual - expected)).toBeLessThan(2e-6);
        else expect(actual === expected).toBe(true);
      }
    }
  });
