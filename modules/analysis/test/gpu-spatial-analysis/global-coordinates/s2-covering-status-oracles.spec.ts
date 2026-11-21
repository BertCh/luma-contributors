// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';
import {CPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {GPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {CPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// At level 0 a radius-0 cap at +x intersects only the +x cube face; a pi
// cap covers all six faces. Capacity 4 therefore gives complete/capacity/invalid
// codes [0,2,1] for that point, the globe, and a non-unit zero center.
for (const refinement of ['uniform', 'level-synchronous'] as const) {
  it(`cap operation status has an independent full-word oracle (${refinement})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          caps: {
            format: 'float32x4',
            values: Float32Array.of(1, 0, 0, 0, 1, 0, 0, Math.PI, 0, 0, 0, 1)
          }
        },
        operations: [
          new Operation({
            caps: 'caps',
            capCount: 3,
            level: 0,
            capacity: 4,
            refinement,
            mode: 'exterior',
            output: {cells: 'cells', counts: 'counts', status: 'status'}
          })
        ],
        outputs: ['counts', 'status']
      });
      expect(Array.from(result['status'])).toEqual([0, 2, 1]);
      expect(Array.from(result['counts'])).toEqual([1, 1, 4, 6, 0, 0]);
    }
  });
}

it('S2 measure status clears poisoned valid rows and rejects invalid IDs in both precisions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const precision of ['float32', 'double-single'] as const) {
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    // Face-0 root and leaf ID 1 are valid; ID 0 and an impossible face 7 are invalid.
    const status = output('status', 'uint32', 4);
    graph.add(
      new GPUS2CellMeasures({
        precision,
        cells: input('cells', 'uint32x2', Uint32Array.of(0, 0x10000000, 0, 0, 1, 0, 0, 0xf0000000)),
        output: {measures: output('measures', 'float32x2', 4), status}
      })
    );
    const compiled = graph.compile();
    try {
      expect(await fixture.submitTwice(compiled, () => fixture.read(status))).toEqual([0, 1, 0, 1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

// The positive xyz octant intersects exactly the +x,+y,+z root faces.
// Level 0 therefore requires three cells: capacity 1 sets only bit 2; capacity
// 6 completes with status 0. A two-vertex ring sets only invalid bit 1.
// At level 1, six visits cannot finish: after the first intersecting root,
// children remain to classify, as do the +y/+z roots. Capacity 24 cannot
// overflow in six visits, so the full status must be exactly traversal bit 4.
// At capacity 1 the serial walk visits +x and its four reportable children,
// then +y with children still unresolved: retained/required [1,4], bits 2|4=6.
// The level-synchronous walk spends six visits on roots and discards its
// unfinished frontier, publishing [0,0] with only bit 4. Neither is a CPU oracle.
for (const refinement of ['serial', 'level-synchronous'] as const) {
  for (const scenario of [
    {name: 'complete', maximumLevel: 0, maximumVisitedCells: 32, capacity: 6, end: 3, status: 0},
    {name: 'capacity', maximumLevel: 0, maximumVisitedCells: 32, capacity: 1, end: 3, status: 2},
    {name: 'invalid', maximumLevel: 0, maximumVisitedCells: 32, capacity: 6, end: 2, status: 1},
    {name: 'unfinished', maximumLevel: 1, maximumVisitedCells: 6, capacity: 24, end: 3, status: 4},
    {
      name: 'capacity-and-unfinished',
      maximumLevel: 1,
      maximumVisitedCells: 6,
      capacity: 1,
      end: 3,
      status: refinement === 'serial' ? 6 : 4,
      counts: refinement === 'serial' ? [1, 4] : [0, 0]
    }
  ]) {
    it(`polygon full status follows root-face geometry (${refinement}, ${scenario.name})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const positions = Float32Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
      const ringOffsets = Uint32Array.of(0, scenario.end);
      const options = {
        maximumLevel: scenario.maximumLevel,
        maximumVisitedCells: scenario.maximumVisitedCells,
        capacity: scenario.capacity,
        refinement
      };
      for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x3', values: positions},
            ringOffsets: {format: 'uint32', values: ringOffsets}
          },
          operations: [
            new Operation({
              ...options,
              positions: 'positions',
              ringOffsets: 'ringOffsets',
              positionCount: 3,
              ringOffsetCount: 2,
              output: {cells: 'cells', counts: 'counts', status: 'status'}
            })
          ],
          outputs: ['status', 'counts']
        });
        expect(Array.from(result['status'])).toEqual([scenario.status]);
        if (scenario.counts) expect(Array.from(result['counts'])).toEqual(scenario.counts);
      }
      const fixture = createOutputOracleFixture(device);
      const {graph, input, output} = fixture;
      const status = output('status', 'uint32', 1);
      const counts = output('counts', 'uint32x2', 1);
      graph.add(
        new GPUS2PolygonCovering({
          ...options,
          positions: input('positions', 'float32x3', positions),
          ringOffsets: input('rings', 'uint32', ringOffsets),
          output: {
            cells: output('cells', 'uint32x2', scenario.capacity),
            counts,
            status
          }
        })
      );
      const compiled = graph.compile();
      try {
        const result = await fixture.submitTwice(compiled, async () => ({
          status: await fixture.read(status),
          counts: await fixture.read(counts)
        }));
        expect(result.status).toEqual([scenario.status]);
        if (scenario.counts) expect(result.counts).toEqual(scenario.counts);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
