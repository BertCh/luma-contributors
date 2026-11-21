// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation';
import {GPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {CPUS2PolygonCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {CPUS2CapCoveringOperation} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createOutputOracleFixture} from './output-oracle-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('both cap operation forms short circuit an empty interior cap before frontier expansion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        caps: {
          format: 'float32x4',
          values: new Float32Array([Math.SQRT1_2, Math.SQRT1_2, 0, 9.5e-6])
        }
      },
      operations: [
        new Operation({
          caps: 'caps',
          capCount: 1,
          capacity: 6,
          level: 1,
          refinement: 'level-synchronous',
          frontierCapacity: 6,
          mode: 'interior',
          tolerance: 1e-5,
          output: {cells: 'cells', counts: 'counts', status: 'status'}
        })
      ],
      outputs: ['cells', 'counts', 'status']
    });
    // Effective radius 9.5e-6 - 1e-5 is negative: the interior cover is empty.
    expect(Array.from(actual['counts'])).toEqual([0, 0]);
    expect(Array.from(actual['status'])).toEqual([0]);
    expect(Array.from(actual['cells'])).toEqual(Array(12).fill(0));
  }
});

it('both cap operations compare encoded tolerance before a tight frontier expansion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation])
    for (const radius of [0.000011, 0.000012]) {
      const actual = await runOperationGraph(device, {
        inputs: {
          caps: {
            format: 'float32x4',
            values: Float32Array.of(Math.SQRT1_2, Math.SQRT1_2, 0, radius)
          }
        },
        operations: [
          new Operation({
            caps: 'caps',
            capCount: 1,
            capacity: 6,
            level: 1,
            refinement: 'level-synchronous',
            frontierCapacity: 6,
            mode: 'interior',
            tolerance: 0.000011,
            output: {cells: 'cells', counts: 'counts', status: 'status'}
          })
        ],
        outputs: ['cells', 'counts', 'status']
      });
      // Equality is empty. The positive residual keeps two face roots, whose eight
      // children exceed the six-slot frontier even though the final interior is empty.
      expect(Array.from(actual['counts'])).toEqual([0, 0]);
      expect(Array.from(actual['status'])).toEqual([radius === 0.000011 ? 0 : 4]);
      expect(Array.from(actual['cells'])).toEqual(Array(12).fill(0));
    }
});

it('both polygon operation forms publish six visits when the next frontier cannot fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = Float32Array.from(
    [
      [1, -0.1, -0.1],
      [1, 0.1, -0.1],
      [1, 0, 0.1]
    ].flatMap(point => point.map(value => value / Math.hypot(...point)))
  );
  for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x3', values: positions},
        rings: {format: 'uint32', values: new Uint32Array([0, 3])}
      },
      operations: [
        new Operation({
          positions: 'positions',
          ringOffsets: 'rings',
          positionCount: 3,
          ringOffsetCount: 2,
          capacity: 1,
          minimumLevel: 1,
          maximumLevel: 1,
          complement: true,
          maximumVisitedCells: 8,
          refinement: 'level-synchronous',
          output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visited'}
        })
      ],
      outputs: ['cells', 'counts', 'status', 'visited']
    });
    // Six root visits queue 24 children; none can be visited in an eight-slot frontier.
    expect(Array.from(actual['counts'])).toEqual([0, 0]);
    expect(Array.from(actual['status'])).toEqual([4]);
    expect(Array.from(actual['visited'])).toEqual([6]);
    expect(Array.from(actual['cells'])).toEqual([0, 0]);
  }
});

it('keeps omitted CPU cap frontiers complete alongside explicitly bounded frontiers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const explicit of [false, true]) {
    const actual = await runOperationGraph(device, {
      inputs: {caps: {format: 'float32x4', values: new Float32Array([1, 0, 0, Math.PI])}},
      operations: [
        new CPUS2CapCoveringOperation({
          caps: 'caps',
          capCount: 1,
          capacity: 1,
          level: 2,
          refinement: 'level-synchronous',
          ...(explicit ? {frontierCapacity: 72} : {}),
          output: {cells: 'cells', counts: 'counts', status: 'status'}
        })
      ],
      outputs: ['counts', 'status']
    });
    expect(Array.from(actual['counts'])).toEqual(explicit ? [0, 0] : [1, 96]);
    expect(Array.from(actual['status'])).toEqual([explicit ? 4 : 2]);
  }
});

for (const refinement of ['serial', 'level-synchronous'] as const) {
  it(`both public polygon operations bind the explicit six-visit budget: ${refinement}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = Float32Array.from(
      [
        [1, -0.1, -0.1],
        [1, 0.1, -0.1],
        [1, 0, 0.1]
      ].flatMap(point => point.map(value => value / Math.hypot(...point)))
    );
    for (const Operation of [GPUS2PolygonCoveringOperation, CPUS2PolygonCoveringOperation]) {
      const actual = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x3', values: positions},
          ringOffsets: {format: 'uint32', values: new Uint32Array([0, 3])}
        },
        operations: [
          new Operation({
            positions: 'positions',
            ringOffsets: 'ringOffsets',
            positionCount: 3,
            ringOffsetCount: 2,
            maximumLevel: 1,
            capacity: 6,
            maximumVisitedCells: 6,
            refinement,
            output: {cells: 'cells', counts: 'counts', status: 'status', visitedCount: 'visited'}
          })
        ],
        outputs: ['cells', 'counts', 'status', 'visited']
      });
      expect(Array.from(actual['counts'])).toEqual(refinement === 'serial' ? [4, 4] : [0, 0]);
      expect(Array.from(actual['status'])).toEqual([4]);
      expect(Array.from(actual['visited'])).toEqual([6]);
      expect(Array.from(actual['cells'])).toEqual(
        refinement === 'serial'
          ? [0, 0x04000000, 0, 0x0c000000, 0, 0x14000000, 0, 0x1c000000, 0, 0, 0, 0]
          : Array(12).fill(0)
      );
    }
  });
}
it('both public cap operations suppress a whole-globe cover after frontier exhaustion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [GPUS2CapCoveringOperation, CPUS2CapCoveringOperation]) {
    const actual = await runOperationGraph(device, {
      inputs: {caps: {format: 'float32x4', values: new Float32Array([1, 0, 0, Math.PI])}},
      operations: [
        new Operation({
          caps: 'caps',
          capCount: 1,
          level: 1,
          capacity: 6,
          frontierCapacity: 6,
          refinement: 'level-synchronous',
          output: {cells: 'cells', counts: 'counts', status: 'status'}
        })
      ],
      outputs: ['cells', 'counts', 'status']
    });
    expect(Array.from(actual['counts'])).toEqual([0, 0]);
    expect(Array.from(actual['status'])).toEqual([4]);
    expect(Array.from(actual['cells'])).toEqual(Array(12).fill(0));
  }
});

for (const refinement of ['uniform', 'level-synchronous'] as const) {
  for (const radius of [0, Math.PI]) {
    it(`writes analytic level-zero cap cells and counts: ${refinement}, radius ${radius}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createOutputOracleFixture(device);
      const views = {
        caps: fixture.input('caps', 'float32x4', new Float32Array([1, 0, 0, radius])),
        cells: fixture.output('cells', 'uint32x2', 6),
        counts: fixture.output('counts', 'uint32x2', 1),
        status: fixture.output('status', 'uint32', 1)
      };
      const operation = new GPUS2CapCoveringOperation({
        caps: 'caps',
        capCount: 1,
        level: 0,
        capacity: 6,
        refinement,
        output: {cells: 'cells', counts: 'counts', status: 'status'}
      });
      fixture.graph.add(
        operation.createContributor(port => views[port as keyof typeof views] as never)
      );
      const compiled = fixture.graph.compile();
      try {
        const actual = await fixture.submitTwice(compiled, async () => ({
          cells: await fixture.readRows([views.cells]),
          counts: await fixture.readRows([views.counts]),
          status: await fixture.readRows([views.status])
        }));
        // Independent S2 encoding: (2*face+1)<<60; +X point is on face 0.
        expect(actual.cells).toEqual(
          radius === 0
            ? [0, 0x10000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            : [
                0, 0x10000000, 0, 0x30000000, 0, 0x50000000, 0, 0x70000000, 0, 0x90000000, 0,
                0xb0000000
              ]
        );
        expect(actual.counts).toEqual(radius === 0 ? [1, 1] : [6, 6]);
        expect(actual.status).toEqual([0]);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const refinement of ['serial', 'level-synchronous'] as const) {
  for (const operationForm of [true, false]) {
    for (const includeVisitedCount of [true, false]) {
      it(`writes analytic face-zero triangle counts and visits: ${refinement}, operation ${operationForm}, visits ${includeVisitedCount}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixture = createOutputOracleFixture(device);
        const views = {
          positions: fixture.input(
            'positions',
            'float32x3',
            Float32Array.from(
              [
                [1, -0.1, -0.1],
                [1, 0.1, -0.1],
                [1, 0, 0.1]
              ].flatMap(point => point.map(value => value / Math.hypot(...point)))
            )
          ),
          ringOffsets: fixture.input('rings', 'uint32', new Uint32Array([0, 3])),
          cells: fixture.output('cells', 'uint32x2', 2),
          counts: fixture.output('counts', 'uint32x2', 1),
          status: fixture.output('status', 'uint32', 1),
          visitedCount: includeVisitedCount ? fixture.output('visited', 'uint32', 1) : undefined
        };
        const options = {maximumLevel: 0, capacity: 2, maximumVisitedCells: 6, refinement};
        const operation = new GPUS2PolygonCoveringOperation({
          ...options,
          positions: 'positions',
          ringOffsets: 'rings',
          positionCount: 3,
          ringOffsetCount: 2,
          output: {
            cells: 'cells',
            counts: 'counts',
            status: 'status',
            visitedCount: includeVisitedCount ? 'visited' : undefined
          }
        });
        fixture.graph.add(
          operationForm
            ? operation.createContributor(port => views[port as keyof typeof views] as never)
            : new GPUS2PolygonCovering({
                ...options,
                positions: views.positions,
                ringOffsets: views.ringOffsets,
                output: {
                  cells: views.cells,
                  counts: views.counts,
                  status: views.status,
                  visitedCount: views.visitedCount
                }
              })
        );
        const compiled = fixture.graph.compile();
        try {
          const actual = await fixture.submitTwice(compiled, async () => ({
            cells: await fixture.readRows([views.cells]),
            counts: await fixture.readRows([views.counts]),
            status: await fixture.readRows([views.status]),
            visits: views.visitedCount ? await fixture.readRows([views.visitedCount]) : []
          }));
          // Triangle lies strictly within cube face 0; all six root cells are visited.
          expect(actual).toEqual({
            cells: [0, 0x10000000, 0, 0],
            counts: [1, 1],
            status: [0],
            visits: includeVisitedCount ? [6] : []
          });
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      });
    }
  }
}
