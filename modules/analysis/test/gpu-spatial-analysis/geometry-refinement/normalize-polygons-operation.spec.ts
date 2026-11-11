// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNormalizePolygonsOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation';
import {
  GPUNormalizePolygonsOperation,
  type NormalizePolygonsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from './normalize-polygons-fixture.json';

for (const empty of [false, true])
  it(`compares every polygon-normalization output (${empty ? 'empty rings and parts' : 'features, holes and polygon reordering'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = empty
      ? {positions: [0, 0], ringOffsets: [0, 0], polygonOffsets: [0, 0], featureOffsets: [0, 1]}
      : fixture.input;
    const props: NormalizePolygonsOperationProps = {
      positions: 'positions',
      ringOffsets: 'rings',
      polygonOffsets: 'polygons',
      featureOffsets: 'features',
      rowCount: input.positions.length / 2,
      ringCount: input.ringOffsets.length - 1,
      polygonCount: input.polygonOffsets.length - 1,
      featureCount: input.featureOffsets.length - 1,
      output: {
        positions: 'normalized',
        ringOffsets: 'normalizedRings',
        polygonOffsets: 'normalizedPolygons'
      }
    };
    const outputs = Object.values(props.output);
    const results = [];
    for (const Operation of [GPUNormalizePolygonsOperation, CPUNormalizePolygonsOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: new Float32Array(input.positions)},
            rings: {format: 'uint32', values: new Uint32Array(input.ringOffsets)},
            polygons: {format: 'uint32', values: new Uint32Array(input.polygonOffsets)},
            features: {format: 'uint32', values: new Uint32Array(input.featureOffsets)}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    // Normalization only copies coordinates and integer offsets: exact numeric comparison.
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++)
        expect(results[0][port][index] === results[1][port][index]).toBe(true);
    }
  });
