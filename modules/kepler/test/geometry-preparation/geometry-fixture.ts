// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const geometryBatches = [
  {
    id: 'first',
    rows: [
      {
        type: 'MultiPoint',
        coordinates: [
          [1, 2],
          [3, 4, 0]
        ]
      },
      {
        type: 'MultiLineString',
        coordinates: [
          [
            [0, 0, 7],
            [1, 1, 8]
          ],
          [
            [2, 2],
            [3, 3]
          ]
        ]
      }
    ]
  },
  {id: 'empty', rows: []},
  {
    id: 'last',
    rows: [
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
            [0, 0]
          ],
          [
            [2, 2],
            [2, 4],
            [4, 4],
            [4, 2],
            [2, 2]
          ]
        ]
      },
      null,
      {
        type: 'GeometryCollection',
        geometries: [
          {type: 'Point', coordinates: [20, 20]},
          {
            type: 'MultiPolygon',
            coordinates: [
              [
                [
                  [20, 20],
                  [21, 20],
                  [20, 21],
                  [20, 20]
                ]
              ]
            ]
          }
        ]
      }
    ]
  }
];
