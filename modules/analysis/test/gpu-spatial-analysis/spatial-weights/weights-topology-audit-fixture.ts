// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {preparePolygonContiguityTopology} from '../../../src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology';

/** Eight source rows: shared edge, binary64 gap, empty, masked, malformed and multipart. */
export function createWeightsTopologyAuditFixture(shift = 0, scale = 1) {
  const vertex = new arrow.FixedSizeList(2, new arrow.Field('coordinate', new arrow.Float64()));
  const ring = new arrow.List(new arrow.Field('vertex', vertex));
  const polygon = new arrow.List(new arrow.Field('ring', ring));
  const feature = new arrow.List(new arrow.Field('polygon', polygon));
  const square = (left: number) => [
    [
      [left, 0],
      [left + 1, 0],
      [left + 1, 1],
      [left, 1],
      [left, 0]
    ].map(([coordinateX, coordinateY]) => [
      shift + scale * coordinateX,
      shift + scale * coordinateY
    ])
  ];
  const rows = [
    [square(0)],
    [square(1)],
    [square(1 + 2 ** -30)],
    null,
    [square(90)],
    [
      [
        [
          [0, 0],
          [1, 0],
          [0, 0]
        ]
      ]
    ],
    [square(10)],
    [square(11), square(20)]
  ];
  // Nonzero Arrow slices make a row/offset indexing error observable.
  const vector = arrow.vectorFromArray([null, ...rows, null], feature);
  return preparePolygonContiguityTopology({
    metadata: {rowCount: 8, invalidRows: [{sourceRowIndex: 4, reason: 'masked source'}]},
    batches: [
      {id: 'first', rowCount: 4, sourceRowOffset: 0, polygons: vector.slice(1, 5)},
      {id: 'second', rowCount: 4, sourceRowOffset: 4, polygons: vector.slice(5, 9)}
    ]
  });
}

// Hand-derived: only squares 0/1 and 10/11 share a complete edge (also a vertex).
// The row at 1+2^-30 has a strictly positive source-coordinate gap.
export const TOPOLOGY_AUDIT_EXPECTED = {
  offsets: [0, 1, 2, 2, 2, 2, 2, 3, 4],
  neighbors: [1, 0, 7, 6, 0, 0],
  weights: [1, 1, 1, 1, 0, 0]
};

/** Four squares: edge 0/1, corner-only 1/2, and isolated 3. */
export function createWeightsCornerContactAuditFixture(shift = 0, scale = 1) {
  const vertex = new arrow.FixedSizeList(2, new arrow.Field('coordinate', new arrow.Float64()));
  const ring = new arrow.List(new arrow.Field('vertex', vertex));
  const polygon = new arrow.List(new arrow.Field('ring', ring));
  const feature = new arrow.List(new arrow.Field('polygon', polygon));
  const rows = [
    [0, 0],
    [1, 0],
    [2, 1],
    [10, 10]
  ].map(([left, bottom]) => [
    [
      [
        [left, bottom],
        [left + 1, bottom],
        [left + 1, bottom + 1],
        [left, bottom + 1],
        [left, bottom]
      ].map(([x, y]) => [shift + scale * x, shift + scale * y])
    ]
  ]);
  return preparePolygonContiguityTopology({
    metadata: {rowCount: 4, invalidRows: []},
    batches: [
      {
        id: 'corner-contact',
        rowCount: 4,
        sourceRowOffset: 0,
        polygons: arrow.vectorFromArray(rows, feature)
      }
    ]
  });
}

// Shapely boundary intersection: rook requires positive length; queen any nonempty contact.
export const CORNER_CONTACT_EXPECTED = {
  rook: {offsets: [0, 1, 2, 2, 2], neighbors: [1, 0], weights: [1, 1]},
  queen: {offsets: [0, 1, 3, 4, 4], neighbors: [1, 0, 2, 1], weights: [1, 1, 1, 1]}
};
