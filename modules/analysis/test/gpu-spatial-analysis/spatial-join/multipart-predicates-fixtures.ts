// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Shapely 2.0.6 literals; regenerate with multipart-predicates.oracle.py.
export const MULTIPART_PREDICATES = [
  'intersects',
  'within',
  'coveredBy',
  'contains',
  'covers',
  'containsProperly',
  'touches',
  'crosses',
  'overlaps',
  'equals'
] as const;
export const MULTIPART_PREDICATE_FIXTURES = [
  {
    name: 'point subset',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 3]
    },
    matrix: '0FFFFF0F2',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'point subset reversed',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 3]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F0FFFFF2',
    predicates: [true, false, false, true, true, true, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'equal point sets',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
      featureOffsets: [0, 3]
    },
    right: {
      kind: 'points',
      positions: [2.0, 0.0, 0.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: '0FFFFFFF2',
    predicates: [true, true, true, true, true, true, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'equal point sets reversed',
    left: {
      kind: 'points',
      positions: [2.0, 0.0, 0.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
      featureOffsets: [0, 3]
    },
    matrix: '0FFFFFFF2',
    predicates: [true, true, true, true, true, true, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'partial point sets',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [2.0, 0.0, 4.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F0FFF0F2',
    predicates: [true, false, false, false, false, false, false, false, true, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'partial point sets reversed',
    left: {
      kind: 'points',
      positions: [2.0, 0.0, 4.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F0FFF0F2',
    predicates: [true, false, false, false, false, false, false, false, true, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'disjoint point sets',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 2.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 2.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: 'FF0FFF0F2',
    predicates: [false, false, false, false, false, false, false, false, false, false],
    dwithin: false,
    pattern: false
  },
  {
    name: 'disjoint point sets reversed',
    left: {
      kind: 'points',
      positions: [0.0, 2.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 2.0],
      featureOffsets: [0, 2]
    },
    matrix: 'FF0FFF0F2',
    predicates: [false, false, false, false, false, false, false, false, false, false],
    dwithin: false,
    pattern: false
  },
  {
    name: 'point single',
    left: {
      kind: 'points',
      positions: [0.0, 0.0]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: '0FFFFF0F2',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'point single reversed',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0]
    },
    matrix: '0F0FFFFF2',
    predicates: [true, false, false, true, true, true, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'cancelled endpoint',
    left: {
      kind: 'points',
      positions: [1.0, 0.0],
      featureOffsets: [0, 1]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '0FFFFF102',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'cancelled endpoint reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [1.0, 0.0],
      featureOffsets: [0, 1]
    },
    matrix: '0F1FF0FF2',
    predicates: [true, false, false, true, true, true, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line boundary points',
    left: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: 'F0FFFF1F2',
    predicates: [true, false, true, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'line boundary points reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [0.0, 0.0, 2.0, 0.0],
      featureOffsets: [0, 2]
    },
    matrix: 'FF10FFFF2',
    predicates: [true, false, false, false, true, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'point line crossing',
    left: {
      kind: 'points',
      positions: [1.0, 0.0, 1.0, 1.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    matrix: '0F0FFF102',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'point line crossing reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [1.0, 0.0, 1.0, 1.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F1FF00F2',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'point polygon boundary',
    left: {
      kind: 'points',
      positions: [0.0, 1.0, 2.0, 1.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: 'F0FFFF212',
    predicates: [true, false, true, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'point polygon boundary reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'points',
      positions: [0.0, 1.0, 2.0, 1.0],
      featureOffsets: [0, 2]
    },
    matrix: 'FF20F1FF2',
    predicates: [true, false, false, false, true, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'point polygon mixed',
    left: {
      kind: 'points',
      positions: [1.0, 1.0, 3.0, 1.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: '0F0FFF212',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'point polygon mixed reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'points',
      positions: [1.0, 1.0, 3.0, 1.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F2FF10F2',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'union line coverage',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '1FFF0FFF2',
    predicates: [true, true, true, true, true, false, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'union line coverage reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    matrix: '1FFF0FFF2',
    predicates: [true, true, true, true, true, false, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'union oblique coverage',
    left: {
      kind: 'lines',
      positions: [0.10000000149011612, 0.20000000298023224, 0.4000000059604645, 0.800000011920929],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [
        0.10000000149011612, 0.20000000298023224, 0.20000000298023224, 0.4000000059604645,
        0.20000000298023224, 0.4000000059604645, 0.4000000059604645, 0.800000011920929
      ],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '1FFF0FFF2',
    predicates: [true, true, true, true, true, false, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'union oblique coverage reversed',
    left: {
      kind: 'lines',
      positions: [
        0.10000000149011612, 0.20000000298023224, 0.20000000298023224, 0.4000000059604645,
        0.20000000298023224, 0.4000000059604645, 0.4000000059604645, 0.800000011920929
      ],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.10000000149011612, 0.20000000298023224, 0.4000000059604645, 0.800000011920929],
      lineOffsets: [0, 2]
    },
    matrix: '1FFF0FFF2',
    predicates: [true, true, true, true, true, false, false, false, false, true],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line outside part',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    matrix: '1F1000102',
    predicates: [true, false, false, false, false, false, false, false, true, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line outside part reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '101F00102',
    predicates: [true, false, false, false, false, false, false, false, true, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'touching lines',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [1.0, 0.0, 2.0, 1.0],
      lineOffsets: [0, 2]
    },
    matrix: 'FF1F00102',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'touching lines reversed',
    left: {
      kind: 'lines',
      positions: [1.0, 0.0, 2.0, 1.0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 4.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: 'FF1F00102',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'crossing lines',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '0F1FF0102',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'crossing lines reversed',
    left: {
      kind: 'lines',
      positions: [1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '0F1FF0102',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'closed lines',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0],
      lineOffsets: [0, 4],
      featureOffsets: [0, 1]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, -1.0, 0.0],
      lineOffsets: [0, 2]
    },
    matrix: 'F01FFF102',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'closed lines reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, -1.0, 0.0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0],
      lineOffsets: [0, 4],
      featureOffsets: [0, 1]
    },
    matrix: 'FF10F01F2',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'line polygon mixed',
    left: {
      kind: 'lines',
      positions: [0.5, 1.0, 1.5, 1.0, 3.0, 1.0, 4.0, 1.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: '1F10F0212',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line polygon mixed reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'lines',
      positions: [0.5, 1.0, 1.5, 1.0, 3.0, 1.0, 4.0, 1.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '102FF1102',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line polygon boundary',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: 'F1FF0F212',
    predicates: [true, false, true, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'line polygon boundary reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: 'FF2101FF2',
    predicates: [true, false, false, false, true, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'line polygon crossing',
    left: {
      kind: 'lines',
      positions: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: '101FF0212',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'line polygon crossing reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'lines',
      positions: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '1F20F1102',
    predicates: [true, false, false, false, false, false, false, true, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'multipart polygon',
    left: {
      kind: 'points',
      positions: [1.0, 1.0, 5.0, 1.0],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 4.0, 0.0, 6.0, 0.0, 6.0, 2.0, 4.0, 2.0],
      featureOffsets: [0, 2],
      polygonOffsets: [0, 1, 2],
      ringOffsets: [0, 4, 8]
    },
    matrix: '0FFFFF212',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'multipart polygon reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 4.0, 0.0, 6.0, 0.0, 6.0, 2.0, 4.0, 2.0],
      featureOffsets: [0, 2],
      polygonOffsets: [0, 1, 2],
      ringOffsets: [0, 4, 8]
    },
    right: {
      kind: 'points',
      positions: [1.0, 1.0, 5.0, 1.0],
      featureOffsets: [0, 2]
    },
    matrix: '0F2FF1FF2',
    predicates: [true, false, false, true, true, true, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'polygon hole filler',
    left: {
      kind: 'polygons',
      positions: [1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 2],
      ringOffsets: [0, 4, 8]
    },
    matrix: 'FF2F1F212',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'polygon hole filler reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 2],
      ringOffsets: [0, 4, 8]
    },
    right: {
      kind: 'polygons',
      positions: [1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: 'FF2F112F2',
    predicates: [true, false, false, false, false, false, true, false, false, false],
    dwithin: true,
    pattern: false
  },
  {
    name: 'polygon contained',
    left: {
      kind: 'polygons',
      positions: [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'polygons',
      positions: [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: '2FF1FF212',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'polygon contained reversed',
    left: {
      kind: 'polygons',
      positions: [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    right: {
      kind: 'polygons',
      positions: [1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
      featureOffsets: [0, 1],
      polygonOffsets: [0, 1],
      ringOffsets: [0, 4]
    },
    matrix: '212FF1FF2',
    predicates: [true, false, false, true, true, true, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'empty feature',
    left: {
      kind: 'points',
      positions: [],
      featureOffsets: [0, 0]
    },
    right: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: 'FFFFFF102',
    predicates: [false, false, false, false, false, false, false, false, false, false],
    dwithin: false,
    pattern: false
  },
  {
    name: 'empty feature reversed',
    left: {
      kind: 'lines',
      positions: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'points',
      positions: [],
      featureOffsets: [0, 0]
    },
    matrix: 'FF1FF0FF2',
    predicates: [false, false, false, false, false, false, false, false, false, false],
    dwithin: false,
    pattern: false
  },
  {
    name: 'long line one-unit gap',
    left: {
      kind: 'lines',
      positions: [-100000000, 0, 100000000, 0],
      lineOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [-100000000, 0, 0, 0, 1, 0, 100000000, 0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    matrix: '101F0FFF2',
    predicates: [true, false, false, true, true, false, false, false, false, false],
    dwithin: true,
    pattern: true
  },
  {
    name: 'long line one-unit gap reversed',
    left: {
      kind: 'lines',
      positions: [-100000000, 0, 0, 0, 1, 0, 100000000, 0],
      lineOffsets: [0, 2, 4],
      featureOffsets: [0, 2]
    },
    right: {
      kind: 'lines',
      positions: [-100000000, 0, 100000000, 0],
      lineOffsets: [0, 2]
    },
    matrix: '1FF00F1F2',
    predicates: [true, true, true, false, false, false, false, false, false, false],
    dwithin: true,
    pattern: true
  }
] as const;
