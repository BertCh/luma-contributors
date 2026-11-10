// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../../../src/analysis-core/ring-assembly/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Adversarial turn-choice cases for the face walk of GPUSegmentRingAssembly on lattice input.
//
// Two outgoing lattice edges whose directions differ by one lattice unit at a far end (the pair
// (N - 1, N - 2) and (N, N - 1) with N = 2^21 has cross product 1, an angle of about 2^-43) cannot
// be ordered by an f32 turn key: the keys are equal under every f32 evaluation order, with or
// without fused multiply-add, so a lowest-index tie-break used to pick the wrong edge. The
// expected successor of every segment is computed here with exact BigInt arithmetic.

const NONE = GPU_SEGMENT_RING_ASSEMBLY_NONE;
/** Half the lattice range: candidate directions are this long, their sum reaches the lattice limit. */
const HALF_LATTICE = 2 ** 21;
/** Mirror axis for the right-interior variant: keeps every mirrored coordinate in [1, 2^22]. */
const MIRROR_SUM = 2 ** 22 + 1;

type Segment = [number, number, number, number];
type InteriorSide = 'left' | 'right';

type AssemblyRun = {
  count: number;
  open: number;
  segmentRings: number[];
  segmentVertices: number[];
  flags: number[];
};

/**
 * Two triangles that touch at vertex v = (1, 1). Triangle A (rows 0-2) is a sliver between
 * directions a = (N - 1, N - 2) and c = a + b; triangle B (rows 3-5) spans from b = (N, N - 1) to
 * straight up. Row 5 (B's edge into v) must continue with row 3 (direction b), which is 2^-43
 * radians counter-clockwise of row 0 (direction a).
 */
function createTieScene(): Segment[] {
  const n = HALF_LATTICE;
  const v: [number, number] = [1, 1];
  const a: [number, number] = [n - 1, n - 2];
  const b: [number, number] = [n, n - 1];
  const c: [number, number] = [a[0] + b[0], a[1] + b[1]];
  const q: [number, number] = [0, 2 * n - 2];
  const at = (offset: [number, number]): [number, number] => [v[0] + offset[0], v[1] + offset[1]];
  const edge = (from: [number, number], to: [number, number]): Segment => [...from, ...to];
  return [
    edge(v, at(a)),
    edge(at(a), at(c)),
    edge(at(c), v),
    edge(v, at(b)),
    edge(at(b), at(q)),
    edge(at(q), v)
  ];
}

/**
 * Two triangles that touch at v = (1, 1), where the edge into v of triangle A (row 2, from
 * v + (N - 1, N - 2)) is anti-parallel, within one lattice unit, to triangle B's outgoing edge
 * (row 3, direction (N, N - 1)). The f32 cross product of that pair is -1 exactly but rounds to 0
 * without fused multiply-add, which turns a near u-turn to the right into a full u-turn to the left.
 */
function createSignLossScene(): Segment[] {
  const n = HALF_LATTICE;
  const v: [number, number] = [1, 1];
  const at = (offset: [number, number]): [number, number] => [v[0] + offset[0], v[1] + offset[1]];
  const edge = (from: [number, number], to: [number, number]): Segment => [...from, ...to];
  const p1 = at([n, 0]);
  const q1 = at([n - 1, n - 2]);
  const p2 = at([n, n - 1]);
  const q2 = at([0, 2 * n - 2]);
  return [edge(v, p1), edge(p1, q1), edge(q1, v), edge(v, p2), edge(p2, q2), edge(q2, v)];
}

/** Mirrors x so counter-clockwise triangles become clockwise (interior on the right). */
function mirrorScene(scene: Segment[]): Segment[] {
  return scene.map(([x0, y0, x1, y1]) => [MIRROR_SUM - x0, y0, MIRROR_SUM - x1, y1]);
}

/**
 * Turn class of outgoing direction `outgoing` after incoming direction `incoming`, in the order
 * of the turn angle in (-pi, pi]: 0 for a right turn, 1 for straight on (or a zero vector),
 * 2 for a left turn or a u-turn.
 */
function getTurnClass(incoming: [bigint, bigint], outgoing: [bigint, bigint]): number {
  const cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0];
  if (cross < 0n) {
    return 0;
  }
  if (cross > 0n) {
    return 2;
  }
  const dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1];
  return dot < 0n ? 2 : 1;
}

/** Sign of turnAngle(first) - turnAngle(second), exactly. */
function compareTurns(
  incoming: [bigint, bigint],
  first: [bigint, bigint],
  second: [bigint, bigint]
): number {
  const firstClass = getTurnClass(incoming, first);
  const secondClass = getTurnClass(incoming, second);
  if (firstClass !== secondClass) {
    return Math.sign(firstClass - secondClass);
  }
  if (firstClass === 1) {
    return 0;
  }
  // Inside one open half-plane the counter-clockwise direction has the larger turn angle.
  const cross = first[0] * second[1] - first[1] * second[0];
  return cross > 0n ? -1 : cross < 0n ? 1 : 0;
}

/** Face-walk successor of every segment: the tightest turn toward the interior, lowest index on ties. */
function getExactSuccessors(scene: Segment[], interiorSide: InteriorSide): number[] {
  const integers = scene.map(segment => segment.map(value => BigInt(value)));
  const sideSign = interiorSide === 'left' ? 1 : -1;
  return integers.map(([x0, y0, x1, y1], index) => {
    const incoming: [bigint, bigint] = [x1 - x0, y1 - y0];
    let best = NONE;
    let bestDirection: [bigint, bigint] = [0n, 0n];
    integers.forEach(([cx0, cy0, cx1, cy1], candidate) => {
      if (candidate === index || cx0 !== x1 || cy0 !== y1) {
        return;
      }
      const direction: [bigint, bigint] = [cx1 - cx0, cy1 - cy0];
      if (best === NONE || sideSign * compareTurns(incoming, direction, bestDirection) > 0) {
        best = candidate;
        bestDirection = direction;
      }
    });
    return best;
  });
}

/** Successors implied by GPU ring output (rings follow the segment direction from slot to slot). */
function getAssembledSuccessors(run: AssemblyRun): number[] {
  return run.segmentRings.map((ring, index) => {
    if (ring === NONE) {
      return NONE;
    }
    const members = run.segmentRings
      .map((other, row) => (other === ring ? row : -1))
      .filter(row => row >= 0);
    const slots = members.map(row => run.segmentVertices[row]);
    const firstSlot = Math.min(...slots);
    const nextSlot = firstSlot + ((run.segmentVertices[index] - firstSlot + 1) % members.length);
    return members[slots.indexOf(nextSlot)];
  });
}

async function runAssembly(
  device: Device,
  scene: Segment[],
  interiorSide: InteriorSide
): Promise<AssemblyRun> {
  const buffers: Buffer[] = [];
  const track = <B extends Buffer>(buffer: B): B => {
    buffers.push(buffer);
    return buffer;
  };
  const rows = scene.length;
  const ringCapacity = 4;
  const vertexCapacity = rows + ringCapacity;
  const output = {
    ringOffsets: track(createOutputBuffer(device, ringCapacity + 1)),
    positions: track(createOutputBuffer(device, 2 * vertexCapacity)),
    segmentRings: track(createOutputBuffer(device, rows)),
    segmentVertices: track(createOutputBuffer(device, rows)),
    flags: track(createOutputBuffer(device, rows)),
    count: track(createOutputBuffer(device, 1)),
    overflow: track(createOutputBuffer(device, 1)),
    open: track(createOutputBuffer(device, 1))
  };
  const graph = new GPUCommandGraph(device, {id: 'exact-turn-graph'});
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(
        graph,
        'endpoints',
        track(createInputBuffer(device, Float32Array.from(scene.flat()))),
        'float32x4',
        rows
      ),
      // Distinct lattice points are at least one unit apart, as in GPUPolygonOverlay.
      vertexTolerance: 0.25,
      interiorSide,
      output: {
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          output.ringOffsets,
          'uint32',
          ringCapacity + 1
        ),
        positions: importGraphBuffer(
          graph,
          'positions',
          output.positions,
          'float32x2',
          vertexCapacity
        ),
        segmentRings: importGraphBuffer(
          graph,
          'segment-rings',
          output.segmentRings,
          'uint32',
          rows
        ),
        segmentVertices: importGraphBuffer(
          graph,
          'segment-vertices',
          output.segmentVertices,
          'uint32',
          rows
        ),
        segmentFlags: importGraphBuffer(graph, 'segment-flags', output.flags, 'uint32', rows),
        status: {
          count: importGraphBuffer(graph, 'count', output.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', output.overflow, 'uint32', 1),
          openSegmentCount: importGraphBuffer(graph, 'open', output.open, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const run: AssemblyRun = {
    count: (await readUint32(output.count, 1))[0],
    open: (await readUint32(output.open, 1))[0],
    segmentRings: await readUint32(output.segmentRings, rows),
    segmentVertices: await readUint32(output.segmentVertices, rows),
    flags: await readUint32(output.flags, rows)
  };
  compiled.destroy?.();
  buffers.forEach(buffer => buffer.destroy());
  return run;
}

function expectExactFaceWalk(run: AssemblyRun, scene: Segment[], interiorSide: InteriorSide) {
  const expected = getExactSuccessors(scene, interiorSide);
  // The scenes are two triangles: the exact walk closes each one on its own.
  expect(expected).toEqual([1, 2, 0, 4, 5, 3]);
  expect(run.flags.map(flags => flags & GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT)).toEqual(
    scene.map(() => 0)
  );
  expect(run.open).toBe(0);
  expect(run.count).toBe(2);
  expect(getAssembledSuccessors(run)).toEqual(expected);
}

for (const interiorSide of ['left', 'right'] as const) {
  it(`GPUSegmentRingAssembly orders lattice turns 2^-43 radians apart exactly (${interiorSide})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = interiorSide === 'left' ? createTieScene() : mirrorScene(createTieScene());
    expectExactFaceWalk(await runAssembly(device, scene, interiorSide), scene, interiorSide);
  });

  it(`GPUSegmentRingAssembly keeps the sign of a near u-turn on lattice input (${interiorSide})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene =
      interiorSide === 'left' ? createSignLossScene() : mirrorScene(createSignLossScene());
    expectExactFaceWalk(await runAssembly(device, scene, interiorSide), scene, interiorSide);
  });
}
