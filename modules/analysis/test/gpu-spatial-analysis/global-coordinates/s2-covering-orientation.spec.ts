// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUS2CapCovering,
  type GPUS2CapCoveringProps
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import fixtures from './s2-polygon-covering-operation-fixture.json';
import {createGlobalCoordinateFixture} from './global-coordinate-fixture';
import {OFF_AXIS_CLOCKWISE_TRIANGLE} from './off-axis-ring-fixture';
import {createFineCoveringRectangle} from './s2-covering-float32-predicate';

// Literal cell IDs come from s2-polygon-covering-operation-fixture.py: independent GEOS
// clipping of S2 rectangles in the x=1 gnomonic chart, plus an xy-to-Hilbert encoder.
// Reversal changes no planar region. These fixtures cover the antipodal face 3 only for
// complement=true; absolute winding would incorrectly include the antipodal shell image.
for (const refinement of ['serial', 'level-synchronous'] as const) {
  it(`S2 ${refinement} narrow face-boundary triangle covers exactly faces 0 and 1`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const longitude = Math.PI / 4;
    const halfWidth = 0.75e-6; // Radians, not degrees.
    const vertices = [
      [Math.cos(longitude - halfWidth), Math.sin(longitude - halfWidth), 0],
      [Math.cos(longitude + halfWidth), Math.sin(longitude + halfWidth), 0],
      [
        Math.cos(Math.PI / 18) * Math.cos(longitude),
        Math.cos(Math.PI / 18) * Math.sin(longitude),
        Math.sin(Math.PI / 18)
      ]
    ];
    // The equatorial edge straddles x=y>0; the top vertex is (45 degrees,10 degrees).
    // The whole triangle has x,y>z>=0. Only face roots 0 and 1 intersect it; the
    // antipodal faces 3 and 4 cannot intersect. Reversal describes the same region.
    for (const reversed of [false, true]) {
      const fixture = createGlobalCoordinateFixture(device);
      const cells = fixture.output('uint32x2', 6);
      const counts = fixture.output('uint32x2', 1);
      const status = fixture.output('uint32', 1);
      fixture.graph.add(
        new GPUS2PolygonCovering({
          positions: fixture.input(
            'float32x3',
            (reversed ? [...vertices].reverse() : vertices).flat()
          ),
          ringOffsets: fixture.input('uint32', [0, 3]),
          maximumLevel: 0,
          capacity: 6,
          maximumVisitedCells: 6,
          refinement,
          mode: 'exterior',
          output: {cells: cells.view, counts: counts.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        for (let submission = 0; submission < 2; submission++) {
          cells.buffer.write(new Uint32Array(12).fill(0xdeadbeef));
          counts.buffer.write(Uint32Array.of(99, 99));
          status.buffer.write(Uint32Array.of(99));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(status.buffer, 1)).toEqual([0]);
          expect(await readUint32(counts.buffer, 2)).toEqual([2, 2]);
          expect(await readUint32(cells.buffer, 12)).toEqual([
            0, 0x10000000, 0, 0x30000000, 0, 0, 0, 0, 0, 0, 0, 0
          ]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
  });
  it(`S2 ${refinement} pins off-axis tiny-ring root IDs without antipodal roots (B1)`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // The triangle strictly contains the (1,1,1) cube corner and crosses faces 0,1,2.
    // It lies in the positive xyz octant, so faces 3,4,5 are outside. At level zero no
    // root fits inside it. For its complement exactly roots 3,4,5 fit inside.
    for (const mode of ['exterior', 'interior'] as const)
      for (const complement of [false, true])
        for (const reversed of [false, true]) {
          const fixture = createGlobalCoordinateFixture(device);
          const vertices = reversed
            ? [...OFF_AXIS_CLOCKWISE_TRIANGLE].reverse()
            : OFF_AXIS_CLOCKWISE_TRIANGLE;
          const cells = fixture.output('uint32x2', 6);
          const counts = fixture.output('uint32x2', 1);
          const status = fixture.output('uint32', 1);
          fixture.graph.add(
            new GPUS2PolygonCovering({
              positions: fixture.input('float32x3', vertices.flat()),
              ringOffsets: fixture.input('uint32', [0, 3]),
              maximumLevel: 0,
              capacity: 6,
              maximumVisitedCells: 6,
              mode,
              complement,
              refinement,
              output: {cells: cells.view, counts: counts.view, status: status.view}
            })
          );
          const compiled = fixture.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            const expected =
              mode === 'interior'
                ? complement
                  ? [0x70000000, 0x90000000, 0xb0000000]
                  : []
                : complement
                  ? [0x10000000, 0x30000000, 0x50000000, 0x70000000, 0x90000000, 0xb0000000]
                  : [0x10000000, 0x30000000, 0x50000000];
            expect(await readUint32(status.buffer, 1)).toEqual([0]);
            expect(await readUint32(counts.buffer, 2)).toEqual([expected.length, expected.length]);
            expect(await readUint32(cells.buffer, 12)).toEqual([
              ...expected.flatMap(high => [0, high]),
              ...new Array((6 - expected.length) * 2).fill(0)
            ]);
          } finally {
            compiled.destroy();
            fixture.destroy();
          }
        }
  });
  it(`S2 ${refinement} classifies known face cells in the antipodal shell image`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // The shell is [-2,2]^2 in x=1. Face 0 is [-1,1]^2 there, strictly inside.
    // Face 3 is wholly in x<0, strictly outside. Its corners lie in the antipodal image
    // of the shell, so a reversed ring's sign-only winding or abs(winding) falsely covers it.
    // These two root IDs follow the S2 face encoding (2*face+1)<<60, independently of traversal.
    for (const mode of ['exterior', 'interior'] as const) {
      for (const complement of [false, true])
        for (const reversed of [false, true]) {
          const fixture = createGlobalCoordinateFixture(device);
          const vertices = [
            [1, -2, -2],
            [1, 2, -2],
            [1, 2, 2],
            [1, -2, 2]
          ];
          if (reversed) vertices.reverse();
          const positions = vertices.flatMap(point =>
            point.map(value => value / Math.hypot(...point))
          );
          const cells = fixture.output('uint32x2', 512);
          const counts = fixture.output('uint32x2', 1);
          const status = fixture.output('uint32', 1);
          fixture.graph.add(
            new GPUS2PolygonCovering({
              positions: fixture.input('float32x3', positions),
              ringOffsets: fixture.input('uint32', [0, 4]),
              maximumLevel: 2,
              maximumVisitedCells: 512,
              capacity: 512,
              mode,
              complement,
              refinement,
              output: {cells: cells.view, counts: counts.view, status: status.view}
            })
          );
          const compiled = fixture.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            expect(await readUint32(status.buffer, 1)).toEqual([0]);
            const [written, required] = await readUint32(counts.buffer, 2);
            expect(written).toBe(required);
            const words = await readUint32(cells.buffer, written * 2);
            const identifiers = Array.from(
              {length: written},
              (_, row) => BigInt(words[row * 2]) | (BigInt(words[row * 2 + 1]) << 32n)
            );
            expect(identifiers.includes(0x1000000000000000n)).toBe(!complement);
            expect(identifiers.includes(0x7000000000000000n)).toBe(complement);
            if (!complement) expect(identifiers.some(cell => cell >> 61n === 3n)).toBe(false);
          } finally {
            compiled.destroy();
            fixture.destroy();
          }
        }
    }
  });
  it(`S2 ${refinement} covers ignore shell/hole orientation and reject the antipodal image`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const fixture of fixtures.filter(scene => scene.minimumLevel === 0)) {
      for (const reverseShell of [false, true]) {
        const graph = new GPUCommandGraph(device);
        const rings = fixture.ringOffsets.slice(0, -1).map((begin, ring) => {
          const end = fixture.ringOffsets[ring + 1];
          const vertices = Array.from({length: end - begin}, (_, vertex) =>
            fixture.positions.slice((begin + vertex) * 3, (begin + vertex + 1) * 3)
          );
          // CCW shell/CW hole, or CW shell/CCW hole.
          return (ring === 0 ? reverseShell : !reverseShell) ? vertices.reverse() : vertices;
        });
        const values = [new Float32Array(rings.flat(2)), new Uint32Array(fixture.ringOffsets)];
        const buffers = [
          ...values.map(value => createInputBuffer(device, value)),
          createOutputBuffer(device, 1024),
          createOutputBuffer(device, 2),
          createOutputBuffer(device, 1),
          createOutputBuffer(device, 1)
        ];
        const view = <Format extends 'float32x3' | 'uint32' | 'uint32x2'>(
          index: number,
          format: Format,
          length: number
        ) => importGraphBuffer(graph, `buffer-${index}`, buffers[index], format, length);
        const contributor = new GPUS2PolygonCovering({
          positions: view(0, 'float32x3', fixture.positions.length / 3),
          ringOffsets: view(1, 'uint32', fixture.ringOffsets.length),
          minimumLevel: fixture.minimumLevel,
          maximumLevel: fixture.maximumLevel,
          mode: fixture.mode as 'exterior' | 'interior',
          complement: fixture.complement,
          capacity: 512,
          maximumVisitedCells: 512,
          refinement,
          output: {
            cells: view(2, 'uint32x2', 512),
            counts: view(3, 'uint32x2', 1),
            status: view(4, 'uint32', 1),
            visitedCount: view(5, 'uint32', 1)
          }
        });
        graph.add(contributor);
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(buffers[4], 1), fixture.name).toEqual([0]);
          expect(await readUint32(buffers[3], 2), fixture.name).toEqual([
            fixture.cells.length,
            fixture.cells.length
          ]);
          expect(await readUint32(buffers[5], 1), fixture.name).toEqual([fixture.visitedCount]);
          const cells = await readUint32(buffers[2], 1024);
          const actual = Array.from({length: fixture.cells.length}, (_, row) =>
            (BigInt(cells[row * 2]) | (BigInt(cells[row * 2 + 1]) << 32n)).toString(16)
          );
          expect(actual, fixture.name).toEqual(fixture.cells);
          expect(cells.slice(fixture.cells.length * 2).every(value => value === 0)).toBe(true);
          expect(actual.includes('7000000000000000')).toBe(fixture.complement);
        } finally {
          compiled.destroy();
          buffers.forEach(buffer => buffer.destroy());
        }
      }
    }
  });
}

for (const refinement of ['serial', 'level-synchronous'] as const) {
  it(`S2 ${refinement} retains tiny level-15 face-zero interior cells and rejects crossings (review B1/B2)`, {
    timeout: 300000
  }, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGlobalCoordinateFixture(device);
    const positionStorage = fixture.output('float32x3', 4);
    const cells = fixture.output('uint32x2', 4);
    const counts = fixture.output('uint32x2', 1);
    const status = fixture.output('uint32', 1);
    const visited = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUS2PolygonCovering({
        positions: positionStorage.view,
        ringOffsets: fixture.input('uint32', [0, 4]),
        minimumLevel: 15,
        maximumLevel: 15,
        mode: 'interior',
        capacity: 4,
        maximumVisitedCells: 512,
        refinement,
        output: {
          cells: cells.view,
          counts: counts.view,
          status: status.view,
          visitedCount: visited.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      // Each rectangle contains exactly the central grid square. Keys come from an independent
      // xy-to-Hilbert encoder; crossed rings have no promised cover and must fail before visits.
      for (const scene of [
        {first: 23000, second: 21000, low: 0x40000000, high: 0x1123fdc0},
        {first: 24503, second: 19621, low: 0x40000000, high: 0x10d2c5c2},
        {first: 19170, second: 31311, low: 0xc0000000, high: 0x12c46378}
      ])
        for (const reversed of [false, true])
          for (const crossed of [false, true]) {
            const ring = createFineCoveringRectangle(scene.first, scene.second);
            const ordered = crossed ? [ring[0], ring[2], ring[1], ring[3]] : ring;
            if (reversed) ordered.reverse();
            positionStorage.buffer.write(Float32Array.from(ordered.flat()));
            cells.buffer.write(new Uint32Array(8).fill(0xdeadbeef));
            submitGraph(device, compiled, undefined);
            expect(await readUint32(status.buffer, 1)).toEqual([crossed ? 1 : 0]);
            expect(await readUint32(counts.buffer, 2)).toEqual(crossed ? [0, 0] : [1, 1]);
            expect(await readUint32(cells.buffer, 8)).toEqual(
              crossed ? new Array(8).fill(0) : [scene.low, scene.high, 0, 0, 0, 0, 0, 0]
            );
            if (crossed) expect(await readUint32(visited.buffer, 1)).toEqual([0]);
          }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

it('S2 rejects collapsed rings before visits on both GPU routes (review N2)', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const refinement of ['serial', 'level-synchronous'] as const) {
    const fixture = createGlobalCoordinateFixture(device);
    const cells = fixture.output('uint32x2', 6);
    const counts = fixture.output('uint32x2', 1);
    const status = fixture.output('uint32', 1);
    const visited = fixture.output('uint32', 1);
    cells.buffer.write(new Uint32Array(12).fill(0xdeadbeef));
    const positions = [-0.4, 0, 0.4].flatMap(value => {
      const length = Math.hypot(1, value);
      return [1 / length, value / length, 0];
    });
    fixture.graph.add(
      new GPUS2PolygonCovering({
        positions: fixture.input('float32x3', positions),
        ringOffsets: fixture.input('uint32', [0, 3]),
        maximumLevel: 0,
        capacity: 6,
        maximumVisitedCells: 6,
        refinement,
        output: {
          cells: cells.view,
          counts: counts.view,
          status: status.view,
          visitedCount: visited.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 1)).toEqual([1]);
      expect(await readUint32(counts.buffer, 2)).toEqual([0, 0]);
      expect(await readUint32(visited.buffer, 1)).toEqual([0]);
      expect(await readUint32(cells.buffer, 12)).toEqual(new Array(12).fill(0));
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('S2 cap captures both routes and output selections while buffer contents stay live (W2-04)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const refinement of ['uniform', 'level-synchronous'] as const) {
    const fixture = createGlobalCoordinateFixture(device);
    const capStorage = fixture.output('float32x4', 1);
    capStorage.buffer.write(Float32Array.of(1, 0, 0, 0));
    const cells = fixture.output('uint32x2', 6);
    const counts = fixture.output('uint32x2', 1);
    const status = fixture.output('uint32', 1);
    const props: GPUS2CapCoveringProps = {
      caps: capStorage.view,
      level: 0,
      capacity: 6,
      refinement,
      ...(refinement === 'level-synchronous' ? {frontierCapacity: 6} : {}),
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'none',
        edgeModel: 'great-circle'
      },
      output: {cells: cells.view, counts: counts.view, status: status.view}
    };
    const contributor = new GPUS2CapCovering(props);
    props.refinement = refinement === 'uniform' ? 'level-synchronous' : 'uniform';
    props.level = 8;
    props.capacity = 3;
    props.frontierCapacity = 999;
    props.mode = 'interior';
    props.spatialContext!.coordinateSpace = 'planar';
    props.output.cells = fixture.output('uint32x2', 3).view;
    props.caps = fixture.input('float32x4', [0, 0, 1, 0]);
    fixture.graph.add(contributor);
    const compiled = fixture.graph.compile();
    try {
      for (const [values, high] of [
        [[0, 1, 0, 0], 0x30000000],
        [[1, 0, 0, 0], 0x10000000]
      ] as const) {
        capStorage.buffer.write(Float32Array.from(values));
        cells.buffer.write(new Uint32Array(12).fill(0xdeadbeef));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(status.buffer, 1)).toEqual([0]);
        expect(await readUint32(counts.buffer, 2)).toEqual([1, 1]);
        expect(await readUint32(cells.buffer, 12)).toEqual([0, high, ...new Array(10).fill(0)]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
