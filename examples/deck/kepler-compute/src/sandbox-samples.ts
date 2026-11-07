// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {CoordinateRoles} from './source-publication';
import type {SandboxTripOptions} from './trip-source-options';

export type SandboxSampleId =
  | 'points-time'
  | 'districts'
  | 'connections'
  | 'trips'
  | 'trip-observations'
  | 'h3-cells'
  | 'numeric-grid'
  | 'directed-network';
export type SandboxSampleDescription = {
  id: SandboxSampleId;
  title: string;
  description: string;
  workflow: string;
  limitation?: string;
  fields: readonly {name: string; meaning: string}[];
};
export const SANDBOX_SAMPLES: readonly SandboxSampleDescription[] = [
  {
    id: 'h3-cells',
    title: 'Exact H3 cells',
    description:
      'Four H3 cells near San Francisco, stored as exact Uint64 IDs across two source batches with missing IDs.',
    workflow:
      'Review the explicit H3 index role, publish, then style cells by measurement and inspect original IDs in the data table.',
    limitation:
      'Missing IDs retain their source rows and omit map geometry. Source cells are not aggregated together.',
    fields: [
      {name: 'h3_cell', meaning: 'Exact H3 Uint64 cell ID; never converted to a Number'},
      {name: 'measurement', meaning: 'Synthetic cell measurement'},
      {name: 'source_row_id', meaning: 'Original source row identity'}
    ]
  },
  {
    id: 'points-time',
    title: 'Observations through time',
    description:
      '256 synthetic sensor observations in four groups, with UTC timestamps and missing readings.',
    workflow:
      'Publish as points, add native category/time filters, then analyze reading over all or selected observations.',
    fields: [
      {name: 'longitude / latitude', meaning: 'WGS84 degrees'},
      {name: 'timestamp', meaning: 'UTC epoch milliseconds'},
      {name: 'reading', meaning: 'Synthetic measurement; null means missing'},
      {name: 'region', meaning: 'Four categorical groups'}
    ]
  },
  {
    id: 'districts',
    title: 'District counts and populations',
    description:
      '16 synthetic polygons with counts and population, including a hole and a multipolygon.',
    workflow:
      'Publish geometry, color by events or population, and compare excess risk using events and population.',
    fields: [
      {name: 'geometry', meaning: 'GeoJSON Polygon or MultiPolygon in WGS84'},
      {name: 'events', meaning: 'Synthetic count'},
      {name: 'population', meaning: 'Synthetic population at risk'},
      {name: 'district', meaning: 'Stable district label'}
    ]
  },
  {
    id: 'connections',
    title: 'Origin and destination flows',
    description:
      '24 synthetic connections with endpoint fields, line geometry, volume and travel duration.',
    workflow:
      'Publish line geometry and filter volume. Endpoint columns are also available for native connection layer configuration.',
    fields: [
      {name: 'longitude / latitude', meaning: 'Origin, WGS84 degrees'},
      {name: 'targetLongitude / targetLatitude', meaning: 'Destination, WGS84 degrees'},
      {name: 'volume', meaning: 'Synthetic flow count'},
      {name: 'duration_minutes', meaning: 'Travel duration'},
      {name: 'geometry', meaning: 'GeoJSON LineString'}
    ]
  },
  {
    id: 'trips',
    title: 'Timed routes',
    description:
      'Eight synthetic routes with increasing per-vertex times and a separate departure timestamp.',
    workflow:
      'Keep Timed paths enabled, review epoch-second timestamps, then publish to use native Trips playback.',
    limitation:
      'Timed paths require one LineString per row with explicit timestamp units. Invalid rows retain their source identity and omit trip geometry.',
    fields: [
      {
        name: 'geometry',
        meaning: 'GeoJSON LineString: longitude, latitude, altitude metres, epoch seconds'
      },
      {name: 'departure', meaning: 'UTC timestamp for the first vertex'},
      {name: 'duration_seconds', meaning: 'Last minus first vertex time'},
      {name: 'passengers', meaning: 'Synthetic occupancy'}
    ]
  },
  {
    id: 'trip-observations',
    title: 'Grouped trip observations',
    description:
      'Forty-eight timed position rows for eight vehicles, interleaved across two source batches with the second batch in reverse time order.',
    workflow:
      'Choose Grouped timed rows (Trips), use vehicle_id, observed_at, longitude and latitude, select Unix milliseconds, then publish. Filtering speed hides rows and shortens trips like native Kepler.',
    limitation:
      'Trips are grouped by the explicit ID column. Rows with a missing ID, time or position contribute no vertex.',
    fields: [
      {name: 'vehicle_id', meaning: 'Trip ID'},
      {name: 'observed_at', meaning: 'UTC timestamp with millisecond resolution'},
      {name: 'longitude / latitude', meaning: 'WGS84 degrees'},
      {name: 'speed', meaning: 'Synthetic speed'}
    ]
  },
  {
    id: 'numeric-grid',
    title: 'Numeric grid',
    description:
      'A 16 × 16 row-major scalar grid with two smooth peaks and explicit missing cells.',
    workflow:
      'Publish cell centers as points. Analyze value, or configure lattice hot spots with width 16 and height 16.',
    limitation:
      'This is a scalar cell table. Raster textures, reprojection and tiled raster analysis are separate workflows.',
    fields: [
      {
        name: 'grid_row / grid_column',
        meaning: 'Zero-based row-major position; row increases northward'
      },
      {name: 'longitude / latitude', meaning: 'WGS84 cell centers'},
      {name: 'value', meaning: 'Synthetic scalar; null marks nodata'}
    ]
  },
  {
    id: 'directed-network',
    title: 'Directed network edges',
    description: 'A 4 × 4 node network with 48 directed edges and asymmetric travel costs.',
    workflow:
      'Publish edge geometry, inspect source/target IDs and filter cost_seconds. Original direction and costs remain in the table.',
    limitation:
      'Routing and accessibility controls still require a network binding; drawing edges does not execute a graph algorithm.',
    fields: [
      {name: 'source_node / target_node', meaning: 'Directed node IDs, 0–15'},
      {name: 'cost_seconds', meaning: 'Positive directed travel cost'},
      {name: 'length_metres', meaning: 'Synthetic segment length'},
      {name: 'geometry', meaning: 'GeoJSON LineString from source to target'}
    ]
  }
];

export type SandboxSample = {
  description: SandboxSampleDescription;
  table: arrow.Table;
  coordinates?: CoordinateRoles;
  geometryColumn?: string;
  tripOptions?: SandboxTripOptions;
  h3Column?: string;
};
const SAMPLE_EPOCH = 1767225600000;

/** Deterministic CPU sources only. Samples use the same preview, upload and analysis as user data. */
export function createSandboxSample(id: SandboxSampleId): SandboxSample {
  const description = SANDBOX_SAMPLES.find(sample => sample.id === id);
  if (!description) throw new Error('Unknown sandbox sample');
  let table: arrow.Table;
  let coordinates: CoordinateRoles | undefined;
  let geometryColumn: string | undefined;
  let h3Column: string | undefined;
  switch (id) {
    case 'h3-cells': {
      const identifiers = [
        null,
        '8928308280fffff',
        '8928308280bffff',
        null,
        '89283082807ffff',
        '89283082803ffff'
      ];
      table = new arrow.Table({
        source_row_id: arrow.makeVector(
          Uint32Array.from({length: identifiers.length}, (_, row) => row)
        ),
        h3_cell: arrow.vectorFromArray(
          identifiers.map(identifier => (identifier === null ? null : BigInt(`0x${identifier}`))),
          new arrow.Uint64()
        ),
        measurement: arrow.makeVector(new Float32Array([0, 10, 25, 0, 40, 15]))
      });
      h3Column = 'h3_cell';
      break;
    }
    case 'points-time': {
      const count = 256;
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row)),
        longitude: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => -74.04 + (row % 16) * 0.006)
        ),
        latitude: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => 40.7 + Math.floor(row / 16) * 0.006)
        ),
        timestamp: makeTimestamp(count, row => SAMPLE_EPOCH + row * 15 * 60000),
        reading: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) =>
            row % 31 === 0 ? null : 20 + (row % 17) * 2 + Math.sin(row * 0.4) * 5
          ),
          new arrow.Float32()
        ),
        region: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) => ['North', 'East', 'South', 'West'][row % 4])
        )
      });
      coordinates = {longitude: 'longitude', latitude: 'latitude'};
      break;
    }
    case 'districts': {
      const geometries = Array.from({length: 16}, (_, row) => {
        const longitude = -74.04 + (row % 4) * 0.025;
        const latitude = 40.7 + Math.floor(row / 4) * 0.025;
        const outer = rectangle(longitude, latitude, 0.022);
        if (row === 10)
          return JSON.stringify({
            type: 'MultiPolygon',
            coordinates: [
              [rectangle(longitude, latitude, 0.01)],
              [rectangle(longitude + 0.013, latitude + 0.013, 0.009)]
            ]
          });
        const rings =
          row === 5
            ? [outer, rectangle(longitude + 0.007, latitude + 0.007, 0.008).reverse()]
            : [outer];
        return JSON.stringify({type: 'Polygon', coordinates: rings});
      });
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: 16}, (_, row) => row)),
        district: arrow.vectorFromArray(
          Array.from({length: 16}, (_, row) => `District ${String(row + 1).padStart(2, '0')}`)
        ),
        events: arrow.makeVector(Float32Array.from({length: 16}, (_, row) => 8 + ((row * 7) % 29))),
        population: arrow.makeVector(
          Float32Array.from({length: 16}, (_, row) => 500 + (row % 5) * 150)
        ),
        geometry: arrow.vectorFromArray(geometries, new arrow.Utf8())
      });
      geometryColumn = 'geometry';
      break;
    }
    case 'connections': {
      const count = 24;
      const longitude = Float32Array.from({length: count}, (_, row) => -74.04 + (row % 6) * 0.014);
      const latitude = Float32Array.from(
        {length: count},
        (_, row) => 40.7 + Math.floor(row / 6) * 0.02
      );
      const targetLongitude = Float32Array.from(
        longitude,
        (value, row) => value + 0.015 + (row % 3) * 0.004
      );
      const targetLatitude = Float32Array.from(
        latitude,
        (value, row) => value + 0.012 + (row % 2) * 0.008
      );
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row)),
        longitude: arrow.makeVector(longitude),
        latitude: arrow.makeVector(latitude),
        targetLongitude: arrow.makeVector(targetLongitude),
        targetLatitude: arrow.makeVector(targetLatitude),
        volume: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => 10 + ((row * 23) % 90))
        ),
        duration_minutes: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => 5 + (row % 11))
        ),
        geometry: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) =>
            JSON.stringify({
              type: 'LineString',
              coordinates: [
                [longitude[row], latitude[row]],
                [targetLongitude[row], targetLatitude[row]]
              ]
            })
          ),
          new arrow.Utf8()
        )
      });
      geometryColumn = 'geometry';
      break;
    }
    case 'trips': {
      const count = 8;
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row)),
        route: arrow.vectorFromArray(Array.from({length: count}, (_, row) => `Route ${row + 1}`)),
        departure: makeTimestamp(count, row => SAMPLE_EPOCH + row * 120000),
        duration_seconds: arrow.makeVector(new Float32Array(count).fill(600)),
        passengers: arrow.makeVector(Float32Array.from({length: count}, (_, row) => 2 + (row % 5))),
        geometry: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) =>
            JSON.stringify({
              type: 'LineString',
              coordinates: Array.from({length: 6}, (_, vertex) => [
                -74.04 + vertex * 0.014,
                40.7 + row * 0.01 + Math.sin(vertex) * 0.003,
                0,
                SAMPLE_EPOCH / 1000 + row * 120 + vertex * 120
              ])
            })
          ),
          new arrow.Utf8()
        )
      });
      geometryColumn = 'geometry';
      break;
    }
    case 'trip-observations': {
      const count = 48;
      const vertexOf = (row: number) =>
        row < 24 ? Math.floor(row / 8) : 5 - Math.floor((row - 24) / 8);
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row)),
        vehicle_id: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) => `Vehicle ${(row % 8) + 1}`)
        ),
        observed_at: makeTimestamp(
          count,
          row => SAMPLE_EPOCH + (row % 8) * 120000 + vertexOf(row) * 120000
        ),
        longitude: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => -74.04 + vertexOf(row) * 0.014)
        ),
        latitude: arrow.makeVector(
          Float32Array.from(
            {length: count},
            (_, row) => 40.7 + (row % 8) * 0.01 + Math.sin(vertexOf(row)) * 0.003
          )
        ),
        speed: arrow.makeVector(Float32Array.from({length: count}, (_, row) => 5 + (row % 7)))
      });
      geometryColumn = 'vehicle_id';
      break;
    }
    case 'numeric-grid': {
      const count = 256;
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row)),
        grid_row: arrow.makeVector(
          Uint32Array.from({length: count}, (_, row) => Math.floor(row / 16))
        ),
        grid_column: arrow.makeVector(Uint32Array.from({length: count}, (_, row) => row % 16)),
        longitude: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => -74.04 + (row % 16) * 0.006)
        ),
        latitude: arrow.makeVector(
          Float32Array.from({length: count}, (_, row) => 40.7 + Math.floor(row / 16) * 0.006)
        ),
        value: arrow.vectorFromArray(
          Array.from({length: count}, (_, row) => {
            if (row === 0 || row === 17 || row === 255) return null;
            const gridRow = Math.floor(row / 16);
            const gridColumn = row % 16;
            return (
              2 +
              35 * Math.exp(-((gridColumn - 4) ** 2 + (gridRow - 5) ** 2) / 12) +
              22 * Math.exp(-((gridColumn - 11) ** 2 + (gridRow - 10) ** 2) / 8)
            );
          }),
          new arrow.Float32()
        )
      });
      coordinates = {longitude: 'longitude', latitude: 'latitude'};
      break;
    }
    case 'directed-network': {
      const edges: {source: number; target: number}[] = [];
      for (let node = 0; node < 16; node++) {
        if (node % 4 < 3)
          edges.push({source: node, target: node + 1}, {source: node + 1, target: node});
        if (node < 12)
          edges.push({source: node, target: node + 4}, {source: node + 4, target: node});
      }
      const position = (node: number) => [
        -74.03 + (node % 4) * 0.025,
        40.71 + Math.floor(node / 4) * 0.025
      ];
      table = new arrow.Table({
        source_row_id: arrow.makeVector(Uint32Array.from(edges, (_, row) => row)),
        source_node: arrow.makeVector(Uint32Array.from(edges, edge => edge.source)),
        target_node: arrow.makeVector(Uint32Array.from(edges, edge => edge.target)),
        cost_seconds: arrow.makeVector(
          Float32Array.from(
            edges,
            edge => 120 + (edge.source > edge.target ? 30 : 0) + (edge.source % 3) * 10
          )
        ),
        length_metres: arrow.makeVector(new Float32Array(edges.length).fill(2000)),
        geometry: arrow.vectorFromArray(
          edges.map(edge =>
            JSON.stringify({
              type: 'LineString',
              coordinates: [position(edge.source), position(edge.target)]
            })
          ),
          new arrow.Utf8()
        )
      });
      geometryColumn = 'geometry';
      break;
    }
  }
  const tripOptions: SandboxTripOptions | undefined =
    id === 'trips'
      ? {timestampEncoding: 'unix-second', maximumTimeErrorMilliseconds: 0.1}
      : id === 'trip-observations'
        ? {
            timestampEncoding: 'unix-millisecond',
            maximumTimeErrorMilliseconds: 0,
            originMilliseconds: SAMPLE_EPOCH,
            grouping: {
              timestampColumn: 'observed_at',
              longitudeColumn: 'longitude',
              latitudeColumn: 'latitude'
            }
          }
        : undefined;
  table.schema.metadata.set(
    'luma:sandbox-sample',
    JSON.stringify({
      version: 1,
      id,
      synthetic: true,
      coordinates,
      geometryColumn,
      ...(h3Column ? {h3Column} : {}),
      ...(tripOptions ? {tripOptions} : {}),
      ...(id === 'numeric-grid' ? {grid: {width: 16, height: 16, order: 'row-major'}} : {})
    })
  );
  // All sources intentionally exercise more than one batch without changing row order.
  const middle = Math.floor(table.numRows / 2);
  return {
    description,
    table: table.slice(0, middle).concat(table.slice(middle)),
    coordinates,
    geometryColumn,
    ...(h3Column ? {h3Column} : {}),
    ...(tripOptions ? {tripOptions} : {})
  };
}

function makeTimestamp(
  length: number,
  value: (row: number) => number
): arrow.Vector<arrow.TimestampMillisecond> {
  return arrow.makeVector(
    arrow.makeData({
      type: new arrow.TimestampMillisecond('UTC'),
      data: BigInt64Array.from({length}, (_, row) => BigInt(value(row)))
    })
  );
}
function rectangle(longitude: number, latitude: number, width: number): number[][] {
  return [
    [longitude, latitude],
    [longitude + width, latitude],
    [longitude + width, latitude + width],
    [longitude, latitude + width],
    [longitude, latitude]
  ];
}
