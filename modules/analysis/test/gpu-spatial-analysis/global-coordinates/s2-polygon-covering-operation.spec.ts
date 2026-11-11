// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {
  GPUS2PolygonCoveringOperation,
  type S2PolygonCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './s2-polygon-covering-operation-fixture.json';

for (const refinement of ['serial', 'level-synchronous'] as const) {
  for (const mode of ['exterior', 'interior'] as const) {
    for (const complement of [false, true]) {
      it(`polygon operations agree on a shell and hole: ${refinement}, ${mode}, complement ${complement}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = fixtures.find(scene => scene.ringOffsets.length === 3)!;
        const props: S2PolygonCoveringOperationProps = {
          positions: 'positions',
          ringOffsets: 'ringOffsets',
          positionCount: fixture.positions.length / 3,
          ringOffsetCount: fixture.ringOffsets.length,
          maximumLevel: 3,
          minimumLevel: 0,
          mode,
          complement,
          refinement,
          capacity: 512,
          maximumVisitedCells: 2048,
          output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visitedCount'}
        };
        const results = [];
        for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation])
          results.push(
            await runOperationGraph(device, {
              inputs: {
                positions: {format: 'float32x3', values: Float32Array.from(fixture.positions)},
                ringOffsets: {format: 'uint32', values: Uint32Array.from(fixture.ringOffsets)}
              },
              operations: [new Operation(props)],
              outputs: Object.values(props.output) as string[]
            })
          );
        // Every output is integer or a copied cell key: tolerance is exactly zero, including
        // counts, status, visitedCount and the unused zero-filled output capacity.
        for (const output of ['cells', 'counts', 'status', 'visitedCount'])
          expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
        expect(results[0]['status'][0]).toBe(0);
        expect(results[0]['counts'][0]).toBeGreaterThan(1);
        expect(results[0]['visitedCount'][0]).toBeGreaterThan(6);
      });
    }
  }
  for (const invalid of [false, true]) {
    it(`polygon operations agree on ${invalid ? 'an invalid ring' : 'output overflow'} with ${refinement}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[0];
      const props: S2PolygonCoveringOperationProps = {
        positions: 'positions',
        ringOffsets: 'ringOffsets',
        positionCount: 4,
        ringOffsetCount: 2,
        maximumLevel: 3,
        minimumLevel: 3,
        capacity: 2,
        refinement,
        maximumVisitedCells: 2048,
        output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visitedCount'}
      };
      const results = [];
      for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x3', values: Float32Array.from(fixture.positions)},
              ringOffsets: {format: 'uint32', values: Uint32Array.of(0, invalid ? 2 : 4)}
            },
            operations: [new Operation(props)],
            outputs: Object.values(props.output) as string[]
          })
        );
      for (const output of ['cells', 'counts', 'status', 'visitedCount'])
        expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
      expect(results[0]['status'][0]).toBe(invalid ? 1 : 2);
      expect(results[0]['counts'][0]).toBe(invalid ? 0 : 2);
      if (!invalid) expect(results[0]['counts'][1]).toBeGreaterThan(2);
    });
  }
}
