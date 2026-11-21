// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CPUSphericalArcOverlayInput,
  CPUSphericalArcOverlayOptions
} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';

function createOperand(ring: number[][], referenceInside = false) {
  return {
    positions: Float32Array.from(
      ring.flatMap(([horizontal, vertical]) => {
        const length = Math.hypot(horizontal, vertical, 1);
        return [horizontal / length, vertical / length, 1 / length];
      })
    ),
    ringOffsets: Uint32Array.of(0, ring.length),
    referenceInside
  };
}

function createRectangle(left: number, bottom: number, right: number, top: number) {
  return [
    [left, bottom],
    [right, bottom],
    [right, top],
    [left, top]
  ];
}

type StatusScene = {
  name: string;
  input: CPUSphericalArcOverlayInput;
  options: CPUSphericalArcOverlayOptions;
  counts: number[];
  status: number[];
  ringOffsets?: number[];
};

export function createArcStatusScenes(): StatusScene[] {
  // Inverse gnomonic projection preserves intersections and boundary combinatorics.
  // A = [-.6,.4] x [-.4,.5], B = [-.2,.7] x [-.6,.2]. Their boundaries
  // cross twice. A union B has eight edges; A xor B has two six-edge rings.
  // Capacity failures suppress all written geometry, retaining required edges.
  const first = createOperand(createRectangle(-0.6, -0.4, 0.4, 0.5));
  const second = createOperand(createRectangle(-0.2, -0.6, 0.7, 0.2));
  const options: CPUSphericalArcOverlayOptions = {
    referencePoint: [0, 0, -1],
    operation: 'intersection',
    capacity: 16,
    ringCapacity: 4
  };
  return [
    {
      name: 'square with spare ring destination slots',
      input: {left: first, right: first},
      options: {...options, operation: 'union', capacity: 4, ringCapacity: 8},
      counts: [4, 1, 4, 0],
      status: [0],
      ringOffsets: [0, 4, 4, 4, 4, 4, 4, 4, 4]
    },
    {
      name: 'impossible input ring count',
      input: {left: {...first, ringOffsets: Uint32Array.of(0, 0, 0, 0, 4)}, right: first},
      options,
      counts: [0, 0, 0, 0],
      status: [1],
      ringOffsets: [0, 0, 0, 0, 0]
    },
    {
      name: 'four-edge intersection',
      input: {left: first, right: first},
      options,
      counts: [4, 1, 4, 0],
      status: [0]
    },
    // Coincident operands describe the same four boundary edges, independent of
    // whether output capacity can retain them. Count each edge once, including overflow.
    {
      name: 'coincident union at capacity three',
      input: {left: first, right: first},
      options: {...options, operation: 'union', capacity: 3},
      counts: [0, 0, 4, 0],
      status: [2]
    },
    {
      name: 'coincident union at capacity four',
      input: {left: first, right: first},
      options: {...options, operation: 'union', capacity: 4},
      counts: [4, 1, 4, 0],
      status: [0]
    },
    // Complementing both identical operands reverses their boundary and contains the
    // south-pole reference. This observes the nonzero fourth counts word independently.
    {
      name: 'complement reference membership',
      input: {
        left: {...first, referenceInside: true},
        right: {...first, referenceInside: true}
      },
      options,
      counts: [4, 1, 4, 1],
      status: [0]
    },
    {
      name: 'eight-edge union exceeds vertex capacity',
      input: {left: first, right: second},
      options: {...options, operation: 'union', capacity: 3},
      counts: [0, 0, 8, 0],
      status: [2]
    },
    {
      name: 'two-ring xor exceeds ring capacity',
      input: {left: first, right: second},
      options: {...options, operation: 'symmetric-difference', ringCapacity: 1},
      counts: [0, 0, 12, 0],
      status: [2]
    },
    // A packed ring must start at zero: no geometric solver is needed to diagnose this.
    {
      name: 'invalid starting offset',
      input: {
        left: {...first, ringOffsets: Uint32Array.of(1, 4)},
        right: second
      },
      options,
      counts: [0, 0, 0, 0],
      status: [1]
    },
    // The first edge subtends atan(3e-6) radians: above the 2*tolerance merge band,
    // below the documented 4*tolerance piece separation. The first unresolved edge
    // occurs before any retained boundary; this is a geometric limit, not a round cap.
    {
      name: 'unresolved three-microradian edge',
      input: {
        left: createOperand([
          [0, 0],
          [3e-6, 0],
          [0, 0.5]
        ]),
        right: second
      },
      options,
      counts: [0, 0, 0, 0],
      status: [4]
    },
    {
      name: 'empty disjoint intersection',
      input: {
        left: first,
        right: createOperand(createRectangle(0.5, -0.6, 0.8, -0.2))
      },
      options,
      counts: [0, 0, 0, 0],
      status: [0]
    }
  ];
}
