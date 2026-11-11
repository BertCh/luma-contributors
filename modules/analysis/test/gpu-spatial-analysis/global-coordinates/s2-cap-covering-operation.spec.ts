// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {
  GPUS2CapCoveringOperation,
  type S2CapCoveringOperationProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './s2-cap-covering-operation-fixture.json';

for (const refinement of ['uniform', 'level-synchronous'] as const) {
  for (const mode of ['exterior', 'interior'] as const) {
    for (const edgeCase of [false, true]) {
      it(`cap operations agree exactly: ${refinement}, ${mode}, ${edgeCase ? 'overflow and invalid caps' : 'ordinary caps on multiple faces'}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = fixtures.find(scene => scene.level === 2)!;
        const caps = Float32Array.from(
          edgeCase ? [...fixture.caps, 0, 0, 0, 1, 1, 0, 0, -1] : fixture.caps
        );
        const props: S2CapCoveringOperationProps = {
          caps: 'caps',
          capCount: caps.length / 4,
          level: 2,
          mode,
          refinement,
          capacity: edgeCase ? 3 : 96,
          ...(refinement === 'level-synchronous' ? {frontierCapacity: 512} : {}),
          output: {cells: 'cells', counts: 'counts', status: 'status'}
        };
        const results = [];
        for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation])
          results.push(
            await runOperationGraph(device, {
              inputs: {caps: {format: 'float32x4', values: caps}},
              operations: [new Operation(props)],
              outputs: ['cells', 'counts', 'status']
            })
          );
        // Keys, counts and status are all integer outputs: exact equality (tolerance zero),
        // across every cap and every unused capacity row.
        for (const output of ['cells', 'counts', 'status'])
          expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
        if (edgeCase) {
          expect(Array.from(results[0]['status'].slice(-2))).toEqual([1, 1]);
          expect(Array.from(results[0]['status']).some(value => value === 2)).toBe(true);
        } else {
          expect(results[0]['status'].every(value => value === 0)).toBe(true);
          expect(new Set(Array.from(results[0]['counts'])).size).toBeGreaterThan(3);
        }
      });
    }
  }
}
