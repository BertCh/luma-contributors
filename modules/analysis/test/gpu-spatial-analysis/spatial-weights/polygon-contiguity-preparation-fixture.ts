// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {preparePolygonContiguityTopology} from '../../../src/gpu-spatial-analysis/spatial-weights/index';

export function makeExactPolygonTopology(empty = false) {
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
    ]
  ];
  const values = empty ? [null, null, null] : [[square(0)], [square(1)], [square(1 + 2 ** -30)]];
  const polygons = arrow.vectorFromArray(values, feature);
  return preparePolygonContiguityTopology({
    metadata: {rowCount: 3, invalidRows: []},
    batches: [{id: 'source', rowCount: 3, sourceRowOffset: 0, polygons}]
  });
}
