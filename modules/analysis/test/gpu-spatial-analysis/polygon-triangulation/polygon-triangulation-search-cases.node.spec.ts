// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Tests of the tiling checker itself (`getTilingDefect`, `getBoundaryChainDefect`), on triangle
// sets whose status is known by construction. The checker is the oracle of the randomized search
// of `GPUPolygonTriangulation`, so it must reject every set that is not a tiling, also where the
// areas agree. Finding frc-geometry-05 of
// `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-geometry.md`.

import {describe, expect, it} from 'vitest';
import {
  METRE_RESOLUTION_FRAME,
  getBoundaryChainDefect,
  getPolygonDefect,
  getTilingDefect,
  getTriangleAreaSum,
  mapToFrame,
  type SearchPoint
} from './polygon-triangulation-search-cases';

describe('the tiling checker of the triangulation search', () => {
  // The counterexample of the review. Polygon A V C B E (clockwise, area 10.5):
  //
  //        C(3,3)
  //       /  \
  //   A--V    B      A(0,0) V(1,0) B(4,0)
  //    \     /
  //     E(2,-3)
  //
  // Triangle ABC covers AVC, which is outside the polygon, and leaves AVE uncovered. The two
  // pieces have equal area (1.5), no triangle edge crosses a polygon edge properly, and no polygon
  // vertex is strictly inside a triangle: V is on side AB of triangle ABC, and the polygon edge VC
  // runs through ABC from V to the corner C.
  const reviewPolygon: SearchPoint[][] = [
    [
      [0, 0],
      [1, 0],
      [3, 3],
      [4, 0],
      [2, -3]
    ]
  ];

  it('rejects two triangles of the right total area that cover a piece outside the polygon', () => {
    expect(getPolygonDefect(reviewPolygon)).toBeUndefined();
    const triangles = [
      [0, 3, 2],
      [1, 3, 4]
    ];
    // Areas 6 and 4.5: the polygon area.
    expect(getTriangleAreaSum(reviewPolygon, triangles)).toBe(10.5);
    expect(getBoundaryChainDefect(reviewPolygon, triangles)).toMatch(
      /^the triangle edges do not cancel to the polygon boundary at vertex/
    );
    expect(getTilingDefect(reviewPolygon, triangles)).toMatch(
      /^the triangle edges do not cancel to the polygon boundary at vertex/
    );
  });

  it('accepts a tiling of the same polygon, in either triangle winding', () => {
    // V C B (4.5), A V E (1.5) and V B E (4.5).
    const triangles = [
      [1, 2, 3],
      [0, 1, 4],
      [1, 3, 4]
    ];
    expect(getTriangleAreaSum(reviewPolygon, triangles)).toBe(10.5);
    expect(getTilingDefect(reviewPolygon, triangles)).toBeUndefined();
    expect(
      getTilingDefect(
        reviewPolygon,
        triangles.map(([first, second, third]) => [third, second, first])
      )
    ).toBeUndefined();
  });

  it('accepts a triangle side that runs past a collinear polygon vertex', () => {
    // Vertex 1 is inside the bottom side of the square; no triangle uses it.
    const square: SearchPoint[][] = [
      [
        [0, 0],
        [2, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ];
    const triangles = [
      [0, 2, 3],
      [0, 3, 4]
    ];
    expect(getTilingDefect(square, triangles)).toBeUndefined();
    expect(getBoundaryChainDefect(square, triangles)).toBeUndefined();
  });

  it('reports a triangle of three collinear vertices unless told to skip it', () => {
    const square: SearchPoint[][] = [
      [
        [0, 0],
        [2, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ];
    const triangles = [
      [0, 1, 2],
      [0, 2, 3],
      [0, 3, 4]
    ];
    expect(getBoundaryChainDefect(square, triangles)).toBe('triangle 0,1,2 has zero area');
    expect(getBoundaryChainDefect(square, triangles, true)).toBeUndefined();
    expect(getTilingDefect(square, triangles, true)).toBeUndefined();
  });

  it('rejects a gap, an overlap, and a doubled triangle that hides a gap', () => {
    const square: SearchPoint[][] = [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ];
    // One triangle short.
    expect(getBoundaryChainDefect(square, [[0, 1, 2]])).toBeDefined();
    // Both diagonals: the four triangles cover the square twice.
    expect(
      getBoundaryChainDefect(square, [
        [0, 1, 2],
        [0, 2, 3],
        [0, 1, 3],
        [1, 2, 3]
      ])
    ).toBeDefined();
    // The same half twice: the area is right and half of the square is uncovered.
    const doubled = [
      [0, 1, 2],
      [2, 1, 0]
    ];
    expect(getTriangleAreaSum(square, doubled)).toBe(16);
    expect(getBoundaryChainDefect(square, doubled)).toBeDefined();
    expect(getTilingDefect(square, doubled)).toBeDefined();
  });

  // A hole whose vertex (4, 9) lies inside the edge (5, 9) - (3, 9) of another hole: the first
  // input of finding frc-geometry-01, polygon area 33.
  const touchingPolygon: SearchPoint[][] = [
    [
      [7, 2],
      [8, 0],
      [8, 7],
      [3, 12],
      [0, 11]
    ],
    [
      [5, 9],
      [3, 9],
      [5, 10]
    ],
    [
      [7, 4],
      [4, 9],
      [7, 5]
    ]
  ];

  it('rejects the wrong triangulation of the review for rings that touch inside an edge', () => {
    // What the kernel returned with valid = 1 before the review fixes: 11 triangles, area 52.
    const triangles = [
      [6, 4, 0],
      [0, 1, 2],
      [3, 4, 9],
      [3, 9, 10],
      [8, 9, 4],
      [4, 6, 7],
      [5, 6, 0],
      [7, 5, 0],
      [8, 4, 0],
      [10, 0, 2],
      [10, 2, 3]
    ];
    expect(getTriangleAreaSum(touchingPolygon, triangles)).toBe(52);
    expect(getBoundaryChainDefect(touchingPolygon, triangles)).toBeDefined();
    expect(getTilingDefect(touchingPolygon, triangles)).toBeDefined();
  });

  it('accepts a tiling of that polygon, on the lattice and at float32 resolution of projected metres', () => {
    // shapely.constrained_delaunay_triangles of the polygon (Shapely 2.2.0 on GEOS 3.14.1), as
    // rows: shell 0..4, first hole 5..7, second hole 8..10. Shapely: 11 triangles, area 33, and
    // their union equals the polygon.
    const triangles = [
      [6, 3, 7],
      [6, 9, 8],
      [10, 9, 5],
      [5, 7, 2],
      [8, 1, 0],
      [3, 6, 4],
      [10, 5, 2],
      [0, 4, 6],
      [8, 10, 2],
      [0, 6, 8],
      [1, 8, 2]
    ];
    expect(getTriangleAreaSum(touchingPolygon, triangles)).toBe(33);
    expect(getTilingDefect(touchingPolygon, triangles)).toBeUndefined();
    const mapped = mapToFrame(METRE_RESOLUTION_FRAME, touchingPolygon);
    expect(getTilingDefect(mapped, triangles)).toBeUndefined();
    // One wrong row: triangle 0 4 6 becomes 0 4 9, which leaves a gap beside the first hole.
    const wrong = triangles.map(triangle =>
      triangle[0] === 0 && triangle[1] === 4 ? [0, 4, 9] : triangle
    );
    expect(getBoundaryChainDefect(mapped, wrong)).toBeDefined();
  });
});
