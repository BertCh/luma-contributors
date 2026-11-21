// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export function createSphericalPredicateScenes() {
  const normalize = (point: number[]) => point.map(value => value / Math.hypot(...point));
  const direction = (longitudeDegrees: number, latitudeDegrees: number) => {
    const longitude = (longitudeDegrees * Math.PI) / 180;
    const latitude = (latitudeDegrees * Math.PI) / 180;
    return [
      Math.cos(latitude) * Math.cos(longitude),
      Math.cos(latitude) * Math.sin(longitude),
      Math.sin(latitude)
    ];
  };
  const packedVertex = normalize([10, 6, 1]).map(Math.fround);
  const first = direction(44.95, 35);
  const second = direction(45.05, 35.05);
  const halfWidth = 0.01015;
  return [
    // An exact packed endpoint remains boundary even when incident plane arithmetic
    // rounds to 2^-23 radians, slightly larger than the requested 1e-7 tolerance.
    {
      name: 'exact packed (3,-11,9) vertex with rounded incident planes',
      vertices: [
        [3, -11, 9],
        [-13, -9, 14],
        [-12, -9, -2]
      ].map(point => normalize(point).map(Math.fround)),
      query: normalize([3, -11, 9]).map(Math.fround),
      tolerance: 1e-7,
      status: 2,
      contains: 0
    },
    // Query bytes exactly equal the first vertex: distance zero independently of edge planes.
    {
      name: 'exact packed (10,6,1) vertex',
      vertices: [packedVertex, normalize([0, 1, -1]), [0, 1, 0]],
      query: packedVertex,
      tolerance: 1e-7,
      status: 2,
      contains: 0
    },
    // Gnomonic projection onto x=1 preserves straight-edge crossings: these two diagonals cross.
    {
      name: 'h=.01015 invalid bow-tie',
      vertices: [
        [-1, -1],
        [1, 1],
        [-1, 1],
        [1, -1]
      ].map(([horizontal, vertical]) =>
        normalize([1, 1 + halfWidth * horizontal, 1 + halfWidth * vertical])
      ),
      query: normalize([1, 1, 1]),
      tolerance: 1e-6,
      status: 1,
      contains: 0
    },
    // +x is the center of a regular radius-5e-7 triangle, with inradius about 2.5e-7 > tolerance.
    {
      name: 'radius-5e-7 valid triangle',
      vertices: [0, 1, 2].map(index => [
        Math.cos(5e-7),
        Math.sin(5e-7) * Math.cos((index * 2 * Math.PI) / 3),
        Math.sin(5e-7) * Math.sin((index * 2 * Math.PI) / 3)
      ]),
      query: [1, 0, 0],
      tolerance: 1e-7,
      status: 0,
      contains: 1
    },
    // normalize(a+b) is the exact minor great-circle midpoint. Float32 packing error is
    // well below the 1e-6 angular boundary band; the north pole completes a simple triangle.
    {
      name: 'oblique-edge midpoint',
      vertices: [first, second, [0, 0, 1]],
      query: normalize(first.map((value, index) => value + second[index])),
      tolerance: 1e-6,
      status: 2,
      contains: 0
    }
  ];
}
