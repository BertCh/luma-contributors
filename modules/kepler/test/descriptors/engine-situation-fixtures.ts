// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Dataset situations written by an independent reviewer of the applicability engine. The
// expectation of each situation was written before the engine was run on it.
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  profileKeplerColumnTable,
  profileKeplerFeatureCollection
} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

export function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random: () => number): number {
  const u = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

const CX = -87.65;
const CY = 41.85;

type Pt = [number, number];
function blobPoints(
  random: () => number,
  count: number,
  blobs: {c: Pt; s: number; w: number}[],
  background: number,
  box: [number, number, number, number]
): Pt[] {
  const totalWeight = blobs.reduce((a, b) => a + b.w, 0);
  const out: Pt[] = [];
  for (let i = 0; i < count; i++) {
    if (random() < background) {
      out.push([box[0] + random() * (box[2] - box[0]), box[1] + random() * (box[3] - box[1])]);
      continue;
    }
    let pick = random() * totalWeight;
    let blob = blobs[0];
    for (const b of blobs) {
      pick -= b.w;
      if (pick <= 0) {
        blob = b;
        break;
      }
    }
    out.push([blob.c[0] + gaussian(random) * blob.s, blob.c[1] + gaussian(random) * blob.s * 0.75]);
  }
  return out;
}

function pointFeature(p: Pt, properties: Record<string, unknown>): CatalogFeature {
  return {geometry: {type: 'Point', coordinates: p}, properties} as CatalogFeature;
}

function squareFeature(
  x: number,
  y: number,
  w: number,
  h: number,
  properties: Record<string, unknown>
): CatalogFeature {
  return {
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [x, y],
          [x + w, y],
          [x + w, y + h],
          [x, y + h],
          [x, y]
        ]
      ]
    },
    properties
  } as CatalogFeature;
}

function lineFeature(coords: Pt[], properties: Record<string, unknown>): CatalogFeature {
  return {geometry: {type: 'LineString', coordinates: coords}, properties} as CatalogFeature;
}

const collection = (features: CatalogFeature[]): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features
});

function isoDate(base: number, dayOffset: number, secondOffset = 0): string {
  return new Date(base + dayOffset * 86400000 + secondOffset * 1000).toISOString();
}

export type SituationDataset = {
  profile: KeplerDatasetProfile;
  collection: CatalogFeatureCollection;
};

export type Expectation = {
  /** Analyst's first suggestion(s): any of these is fine as the leader. */
  first: string[];
  /** Never first (and never `ready` anywhere with the stated fill, when roles are given). */
  never: string[];
  /** Expected columns by (analysis key -> binding parameter -> allowed columns). */
  roles?: Record<string, Record<string, string[]>>;
  /** Columns that must not be filled into ANY value-like role of any suggestion with status ready. */
  forbiddenColumns?: string[];
  note: string;
};

export type Situation = {
  id: string;
  description: string;
  expectation: Expectation;
  datasets: SituationDataset[];
};

function profileCollection(
  id: string,
  label: string,
  features: CatalogFeature[],
  options: Record<string, unknown> = {}
): SituationDataset {
  const fc = collection(features);
  const profile = profileKeplerFeatureCollection(fc, {
    datasetId: id,
    label,
    ...options
  } as never) as unknown as KeplerDatasetProfile;
  return {profile, collection: fc};
}

const BOX: [number, number, number, number] = [CX - 0.12, CY - 0.09, CX + 0.12, CY + 0.09];

export function buildSituations(): Situation[] {
  const situations: Situation[] = [];
  const baseTime = Date.UTC(2023, 0, 1);

  // S01 customers CSV
  {
    const random = makeRandom(1);
    const points = blobPoints(
      random,
      800,
      [
        {c: [CX - 0.03, CY + 0.02], s: 0.01, w: 3},
        {c: [CX + 0.05, CY - 0.03], s: 0.015, w: 2}
      ],
      0.2,
      BOX
    );
    const rows = points.map((p, i) => ({
      lat: p[1],
      lng: p[0],
      customer_id: 100000 + i * 7,
      revenue: Math.round((50 + random() * 900) * 100) / 100,
      visits: 1 + Math.floor(random() * 30),
      segment: ['A', 'B', 'C', 'D'][Math.floor(random() * 4)]
    }));
    const table = {
      rowCount: rows.length,
      columns: ['lat', 'lng', 'customer_id', 'revenue', 'visits', 'segment'].map(name => ({
        name,
        values: rows.map(r => (r as never)[name])
      }))
    };
    const profile = profileKeplerColumnTable(table as never, {
      datasetId: 'customers',
      label: 'customers.csv'
    }) as unknown as KeplerDatasetProfile;
    const fc = collection(
      rows.map(r =>
        pointFeature([r.lng, r.lat], {
          lat: r.lat,
          lng: r.lng,
          customer_id: r.customer_id,
          revenue: r.revenue,
          visits: r.visits,
          segment: r.segment
        })
      )
    );
    situations.push({
      id: 'S01-customers-csv',
      description:
        'CSV of 800 geocoded customers: lat, lng, customer_id, revenue, visits, segment (profileKeplerColumnTable)',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'clusters-and-footprints'],
        never: [
          'idw-surface',
          'kriging-with-diagnostics',
          'regression-with-diagnostics',
          'period-comparison'
        ],
        roles: {
          'density-surface': {weightProperty: ['revenue', 'visits']},
          'hot-spot-map': {valueProperty: ['revenue', 'visits']}
        },
        forbiddenColumns: ['customer_id', 'lat', 'lng'],
        note: 'Customers are objects: density of customers or revenue hot spots. Weight revenue or visits, never customer_id.'
      },
      datasets: [{profile, collection: fc}]
    });
  }

  // S02 census tracts 60 columns
  {
    const random = makeRandom(2);
    const features: CatalogFeature[] = [];
    const side = 20;
    const w = 0.012;
    const h = 0.009;
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const field = Math.sin(c / 3) * Math.cos(r / 4);
        const props: Record<string, unknown> = {
          geoid: 17031000000 + r * 100 + c,
          statefp: 17,
          countyfp: 31,
          tractce: 10000 + r * 100 + c,
          aland: Math.round(500000 + random() * 900000),
          awater: Math.round(random() * 50000),
          intptlat: CY + r * h,
          intptlon: CX + c * w,
          pop_total: Math.round(2000 + random() * 6000),
          households: Math.round(800 + random() * 2500),
          median_income: Math.round(40000 + 20000 * field + random() * 15000),
          median_age: Math.round((30 + 5 * field + random() * 8) * 10) / 10
        };
        for (let k = 0; k < 16; k++) props[`count_group_${k}`] = Math.round(random() * 800);
        for (let k = 0; k < 16; k++) props[`pct_group_${k}`] = Math.round(random() * 1000) / 10;
        props['pct_poverty'] = Math.round((15 + 10 * field + random() * 8) * 10) / 10;
        props['covid_cases'] = Math.round(
          (props['pop_total'] as number) * (0.02 + 0.01 * (field + 1) + random() * 0.01)
        );
        for (let k = 0; k < 14; k++) props[`median_stat_${k}`] = Math.round(random() * 100000);
        features.push(squareFeature(CX + c * w, CY + r * h, w, h, props));
      }
    }
    situations.push({
      id: 'S02-census-tracts-60col',
      description:
        '400 tract polygons, 60 columns: geoid/statefp/countyfp/tractce, aland/awater, intptlat/lon, pop_total, households, median_income, median_age, 16 counts, 16 pcts, pct_poverty, covid_cases, 14 medians',
      expectation: {
        first: ['cluster-and-outlier-map', 'rate-cluster-map'],
        never: ['regression-with-diagnostics:response=pop_total', 'hot-spot on geoid'],
        roles: {
          'cluster-and-outlier-map': {
            valueProperty: ['pct_poverty', 'median_income', 'median_age']
          },
          'rate-cluster-map': {casesProperty: ['covid_cases'], populationProperty: ['pop_total']},
          'excess-risk': {casesProperty: ['covid_cases'], populationProperty: ['pop_total']}
        },
        forbiddenColumns: ['geoid', 'statefp', 'countyfp', 'tractce', 'intptlat', 'intptlon'],
        note: 'Local Moran on a poverty rate or median income; rate map covid_cases/pop_total (not households). Never an identifier or coordinate.'
      },
      datasets: [profileCollection('tracts', 'tracts', features)]
    });
  }

  // S03 crime incidents
  {
    const random = makeRandom(3);
    const points = blobPoints(
      random,
      5000,
      [
        {c: [CX - 0.04, CY + 0.01], s: 0.008, w: 4},
        {c: [CX + 0.03, CY - 0.04], s: 0.012, w: 2},
        {c: [CX, CY], s: 0.02, w: 3}
      ],
      0.15,
      BOX
    );
    const cats = [
      'THEFT',
      'BATTERY',
      'ASSAULT',
      'BURGLARY',
      'ROBBERY',
      'NARCOTICS',
      'MOTOR VEHICLE THEFT',
      'DECEPTIVE PRACTICE'
    ];
    const features = points.map((p, i) =>
      pointFeature(p, {
        incident_id: 5000000 + i,
        date: isoDate(baseTime, Math.floor(random() * 365), Math.floor(random() * 86400)),
        category: cats[Math.floor(random() * 8)],
        severity_code: 1 + Math.floor(random() * 5)
      })
    );
    situations.push({
      id: 'S03-crime-incidents',
      description:
        '5,000 crime incident points: incident_id, date (ISO), category (8), severity_code (1-5)',
      expectation: {
        // The new curated temporal method reads ISO event times, so it is a valid lead.
        first: ['emerging-hot-spots', 'hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        never: ['idw-surface', 'kriging-with-diagnostics', 'regression-with-diagnostics'],
        roles: {'hot-spot-map': {valueProperty: []}, 'density-surface': {weightProperty: []}},
        forbiddenColumns: ['incident_id', 'severity_code'],
        note: 'Events: hot spots of incident count (no value column) or density. Weight must stay empty (severity_code is ordinal, incident_id an id). ISO event time enables emerging hot spots on this one layer; before/after comparison still needs two layers.'
      },
      datasets: [profileCollection('crimes', 'crimes', features)]
    });
  }

  // S04 taxi trips
  {
    const random = makeRandom(4);
    const pick = blobPoints(
      random,
      8000,
      [
        {c: [CX - 0.02, CY + 0.01], s: 0.01, w: 5},
        {c: [CX + 0.06, CY - 0.05], s: 0.02, w: 1}
      ],
      0.2,
      BOX
    );
    const drop = blobPoints(random, 8000, [{c: [CX, CY], s: 0.03, w: 1}], 0.3, BOX);
    const features = pick.map((p, i) => {
      const d = random() * 20;
      return pointFeature(p, {
        pickup_datetime: isoDate(baseTime, Math.floor(random() * 30), Math.floor(random() * 86400)),
        dropoff_datetime: isoDate(
          baseTime,
          Math.floor(random() * 30),
          Math.floor(random() * 86400)
        ),
        dropoff_longitude: drop[i][0],
        dropoff_latitude: drop[i][1],
        passenger_count: 1 + Math.floor(random() * 4),
        trip_distance: Math.round(d * 100) / 100,
        fare_amount: Math.round((3 + d * 2.5 + random() * 3) * 100) / 100,
        tip_amount: Math.round(random() * 800) / 100,
        PULocationID: 1 + Math.floor(random() * 260),
        VendorID: 1 + Math.floor(random() * 2)
      });
    });
    situations.push({
      id: 'S04-taxi-trips',
      description:
        '8,000 taxi pickups (geometry) with dropoff_longitude/latitude, pickup/dropoff datetime, passenger_count, trip_distance, fare_amount, tip_amount, PULocationID, VendorID',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'clusters-and-footprints'],
        never: ['idw-surface', 'kriging-with-diagnostics', 'regression-with-diagnostics'],
        roles: {
          'density-surface': {
            weightProperty: ['fare_amount', 'trip_distance', 'passenger_count', 'tip_amount']
          },
          'hot-spot-map': {
            valueProperty: ['fare_amount', 'trip_distance', 'passenger_count', 'tip_amount']
          }
        },
        forbiddenColumns: ['dropoff_longitude', 'dropoff_latitude', 'PULocationID', 'VendorID'],
        note: 'Pickup density/hot spots. Weight may be fare; never the dropoff coordinates, a zone id or vendor.'
      },
      datasets: [profileCollection('taxi', 'taxi', features)]
    });
  }

  // S05 weather stations
  {
    const random = makeRandom(5);
    const points = blobPoints(random, 60, [{c: [CX, CY], s: 0.1, w: 1}], 0.5, [
      CX - 0.4,
      CY - 0.3,
      CX + 0.4,
      CY + 0.3
    ]);
    const features = points.map((p, i) =>
      pointFeature(p, {
        station_id: `WX${1000 + i}`,
        temperature_c: Math.round((15 + (p[0] - CX) * 10 + random() * 2) * 10) / 10,
        precip_mm: Math.round(random() * 400) / 10,
        elevation_m: Math.round(170 + random() * 80)
      })
    );
    situations.push({
      id: 'S05-weather-stations',
      description: '60 weather stations: station_id (text), temperature_c, precip_mm, elevation_m',
      expectation: {
        first: ['idw-surface', 'kriging-with-diagnostics'],
        never: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        roles: {
          'idw-surface': {valueProperty: ['temperature_c', 'precip_mm']},
          'kriging-with-diagnostics': {valueProperty: ['temperature_c', 'precip_mm']}
        },
        forbiddenColumns: ['station_id'],
        note: 'Samples of a surface: interpolate temperature or precipitation. Never hot spots or density of stations.'
      },
      datasets: [profileCollection('stations', 'stations', features)]
    });
  }

  // S06 AIS
  {
    const random = makeRandom(6);
    const features: CatalogFeature[] = [];
    for (let v = 0; v < 25; v++) {
      const mmsi = 200000000 + v * 1234567;
      let x = CX - 0.1 + random() * 0.2;
      let y = CY - 0.05 + random() * 0.1;
      const heading = Math.floor(random() * 360);
      for (let k = 0; k < 320; k++) {
        x += Math.cos((heading * Math.PI) / 180) * 0.0004;
        y += Math.sin((heading * Math.PI) / 180) * 0.0003;
        features.push(
          pointFeature([x, y], {
            mmsi,
            sog: Math.round(random() * 200) / 10,
            cog: Math.round(random() * 3600) / 10,
            heading: heading + Math.floor(random() * 5),
            ts: isoDate(baseTime, 0, k * 60 + v),
            vessel_type: 30 + 10 * (v % 6)
          })
        );
      }
    }
    situations.push({
      id: 'S06-ais-vessels',
      description:
        '8,000 AIS positions of 25 vessels: mmsi (9-digit), sog (speed), cog, heading, ts (ISO), vessel_type (30..80)',
      expectation: {
        first: ['density-surface', 'clusters-and-footprints'],
        never: ['idw-surface', 'kriging-with-diagnostics', 'regression-with-diagnostics'],
        roles: {'density-surface': {weightProperty: []}},
        forbiddenColumns: ['mmsi', 'heading', 'cog', 'vessel_type'],
        note: 'Traffic density. Never weight by mmsi, heading, cog or the vessel type code. (Track analyses are not available as bindings.)'
      },
      datasets: [profileCollection('ais', 'ais', features)]
    });
  }

  // S07 earthquakes
  {
    const random = makeRandom(7);
    const points = blobPoints(
      random,
      3000,
      [
        {c: [CX - 0.1, CY], s: 0.02, w: 3},
        {c: [CX + 0.1, CY + 0.05], s: 0.05, w: 1}
      ],
      0.1,
      BOX
    );
    const features = points.map((p, i) =>
      pointFeature(p, {
        event_id: `eq${i}`,
        magnitude: Math.round((2 + -Math.log(1 - random() * 0.999) * 0.6) * 10) / 10,
        depth_km: Math.round(random() * 7000) / 10,
        time: isoDate(baseTime, Math.floor(random() * 700), 0)
      })
    );
    situations.push({
      id: 'S07-earthquakes',
      description:
        '3,000 epicentres: event_id (text), magnitude (2-7), depth_km (0-700), time (ISO)',
      expectation: {
        first: ['clusters-and-footprints', 'density-surface', 'hot-spot-map'],
        never: ['idw-surface', 'kriging-with-diagnostics'],
        roles: {'density-surface': {weightProperty: []}, 'hot-spot-map': {valueProperty: []}},
        forbiddenColumns: ['depth_km', 'magnitude'],
        note: 'Events; density of epicentres. A weight by depth is meaningless; magnitude is logarithmic (summing it is wrong), so no weight at all.'
      },
      datasets: [profileCollection('quakes', 'quakes', features)]
    });
  }

  // S08 parcels
  {
    const random = makeRandom(8);
    const features: CatalogFeature[] = [];
    for (let r = 0; r < 25; r++)
      for (let c = 0; c < 24; c++) {
        const area = Math.round(3000 + random() * 9000);
        features.push(
          squareFeature(CX + c * 0.003, CY + r * 0.002, 0.003, 0.002, {
            parcel_id: 1000000 + r * 24 + c,
            assessed_value: Math.round(
              100000 + 300000 * Math.exp(-(((c - 8) ** 2 + (r - 10) ** 2) / 60)) + random() * 80000
            ),
            land_use: [
              'residential',
              'commercial',
              'industrial',
              'park',
              'vacant',
              'mixed',
              'institutional',
              'civic'
            ][Math.floor(random() * 8)],
            area_sqft: area,
            year_built: 1900 + Math.floor(random() * 120)
          })
        );
      }
    situations.push({
      id: 'S08-parcels',
      description:
        '600 parcel polygons: parcel_id, assessed_value, land_use (8 classes), area_sqft, year_built',
      expectation: {
        first: ['cluster-and-outlier-map', 'contiguity-correlogram'],
        never: ['rate-cluster-map', 'excess-risk'],
        roles: {'cluster-and-outlier-map': {valueProperty: ['assessed_value']}},
        forbiddenColumns: ['parcel_id', 'year_built'],
        note: 'Local Moran of assessed_value (an amount; ideally per sqft). Never parcel_id/year_built. No cases/population so rate maps are impossible.'
      },
      datasets: [profileCollection('parcels', 'parcels', features)]
    });
  }

  // S09 roads
  {
    const random = makeRandom(9);
    const features: CatalogFeature[] = [];
    for (let r = 0; r < 20; r++) {
      for (let c = 0; c < 19; c++) {
        const x = CX + c * 0.004;
        const y = CY + r * 0.003;
        features.push(
          lineFeature(
            [
              [x, y],
              [x + 0.004, y]
            ],
            {
              name: `E ${r}-${c}`,
              speed_limit: [15, 25, 30, 35, 45, 55][Math.floor(random() * 6)],
              oneway: random() < 0.3,
              length_m: Math.round(330 + random() * 20),
              highway: ['residential', 'primary', 'secondary'][Math.floor(random() * 3)]
            }
          )
        );
        features.push(
          lineFeature(
            [
              [x, y],
              [x, y + 0.003]
            ],
            {
              name: `N ${r}-${c}`,
              speed_limit: [15, 25, 30, 35, 45, 55][Math.floor(random() * 6)],
              oneway: random() < 0.3,
              length_m: Math.round(330 + random() * 20),
              highway: ['residential', 'primary', 'secondary'][Math.floor(random() * 3)]
            }
          )
        );
      }
    }
    situations.push({
      id: 'S09-roads',
      description:
        '~760 road segments: name, speed_limit (mph int), oneway (boolean), length_m, highway (class)',
      expectation: {
        first: ['network-centrality', 'network-structure'],
        never: ['hot-spot-map', 'density-surface'],
        roles: {'network-service-area': {forwardCostProperty: ['length_m']}},
        forbiddenColumns: [],
        note: 'A street network alone: centrality/structure ready; service area needs origin points (one step away, remedy: load a points layer). Cost by length_m, one-way flag to the boolean.'
      },
      datasets: [profileCollection('roads', 'roads', features)]
    });
  }

  // S10 bus stops + routes
  {
    const random = makeRandom(10);
    const stops = blobPoints(random, 400, [{c: [CX, CY], s: 0.05, w: 1}], 0.3, BOX).map((p, i) =>
      pointFeature(p, {
        stop_id: 9000 + i,
        stop_name: `Stop ${i}`,
        ridership: Math.round(random() * random() * 3000),
        route_count: 1 + Math.floor(random() * 4)
      })
    );
    const routes: CatalogFeature[] = [];
    for (let r = 0; r < 12; r++) {
      const coords: Pt[] = [];
      let x = CX - 0.1 + random() * 0.2;
      let y = CY - 0.07 + random() * 0.14;
      for (let k = 0; k < 15; k++) {
        coords.push([x, y]);
        x += (random() - 0.4) * 0.02;
        y += (random() - 0.5) * 0.015;
      }
      routes.push(
        lineFeature(coords, {route_id: `R${r}`, headway_min: 5 + Math.floor(random() * 25)})
      );
    }
    situations.push({
      id: 'S10-bus-stops-routes',
      description:
        '400 bus stops (stop_id, stop_name, ridership, route_count) + 12 route lines (route_id, headway_min)',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'drive-time-catchments'],
        never: [
          'idw-surface',
          'kriging-with-diagnostics',
          'period-comparison',
          'straight-line-catchments'
        ],
        roles: {
          'density-surface': {weightProperty: ['ridership']},
          'hot-spot-map': {valueProperty: ['ridership']}
        },
        forbiddenColumns: ['stop_id', 'route_count'],
        note: 'Ridership density at stops (weight ridership); route lines alone are not a street graph for accessibility. Never IDW of ridership.'
      },
      datasets: [
        profileCollection('stops', 'stops', stops),
        profileCollection('routes', 'routes', routes)
      ]
    });
  }

  // S11 polygons with single text column
  {
    const features: CatalogFeature[] = [];
    for (let r = 0; r < 10; r++)
      for (let c = 0; c < 10; c++)
        features.push(
          squareFeature(CX + c * 0.01, CY + r * 0.01, 0.01, 0.01, {name: `Area ${r}-${c}`})
        );
    situations.push({
      id: 'S11-polygons-one-text',
      description: '100 polygons with a single text column (name)',
      expectation: {
        first: ['geometry-measures', 'geometry-validity', 'count-by-area'],
        never: [
          'cluster-and-outlier-map',
          'rate-cluster-map',
          'excess-risk',
          'regression-with-diagnostics',
          'contiguity-correlogram'
        ],
        forbiddenColumns: ['name'],
        note: 'No numeric column: no statistical analysis can be ready. Only geometry prep or aggregation if points load later.'
      },
      datasets: [profileCollection('areas', 'areas', features)]
    });
  }

  // S12 twelve points
  {
    const random = makeRandom(12);
    const features = blobPoints(random, 12, [{c: [CX, CY], s: 0.01, w: 1}], 0.2, BOX).map((p, i) =>
      pointFeature(p, {name: `P${i}`, value: Math.round(random() * 100)})
    );
    situations.push({
      id: 'S12-twelve-points',
      description: '12 points: name, value (0-100 integer)',
      expectation: {
        first: ['density-surface', 'idw-surface'],
        never: ['hot-spot-map', 'clusters-and-footprints'],
        forbiddenColumns: [],
        note: 'With 12 points significance and clustering are not meaningful; hot spots/DBSCAN must not lead without a visible small-sample warning, kriging (min 8 rows) is dubious.'
      },
      datasets: [profileCollection('few', 'few', features)]
    });
  }

  // S13 150k points
  {
    const random = makeRandom(13);
    const features: CatalogFeature[] = [];
    for (let i = 0; i < 150000; i++)
      features.push(
        pointFeature([BOX[0] + random() * 0.24, BOX[1] + random() * 0.18], {
          v: Math.floor(random() * 10)
        })
      );
    situations.push({
      id: 'S13-150k-points',
      description: '150,000 points with one integer column v',
      expectation: {
        first: ['cell-aggregation', 'hexbin'],
        never: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        forbiddenColumns: [],
        note: 'Above every cap (20,000 / 100,000): nothing capped may be ready; the remedy should say aggregate to cells or filter. Something that can take the volume (cell aggregation) is the analyst lead.'
      },
      datasets: [profileCollection('big', 'big', features, {maximumSampleSize: 20000})]
    });
  }

  // S14 UTM points
  {
    const random = makeRandom(14);
    const features = Array.from({length: 1000}, () => {
      const east = 440000 + Math.abs(gaussian(random)) * 3000 + random() * 8000;
      const north = 4625000 + random() * 12000;
      return pointFeature([east, north], {
        id: 1,
        reading: Math.round(random() * 1000) / 10,
        count: Math.floor(random() * 20)
      } as never);
    }).map((f, i) => ({
      ...f,
      properties: {
        pid: i,
        reading: (f.properties as never as Record<string, number>)['reading'],
        visits: (f.properties as never as Record<string, number>)['count']
      }
    }));
    situations.push({
      id: 'S14-utm-points',
      description:
        '1,000 points in UTM 16N metres (440000-460000 E, 4625000-4637000 N): pid, reading, visits',
      expectation: {
        first: ['density-surface', 'hot-spot-map'],
        never: ['idw-surface'],
        forbiddenColumns: ['pid'],
        note: 'Coordinates are planar metres: coordinateSystem must be planar (and a guess of planar shown as a guess); cell sizes in metres.'
      },
      datasets: [profileCollection('utm', 'utm', features)]
    });
  }

  // S15 antimeridian
  {
    const random = makeRandom(15);
    const features = Array.from({length: 500}, (_, i) => {
      const lon = random() < 0.5 ? 178 + random() * 2 : -180 + random() * 2;
      return pointFeature([lon, -17.5 + gaussian(random) * 0.5], {
        obs_id: i,
        catch_kg: Math.round(random() * 500) / 10
      });
    });
    situations.push({
      id: 'S15-antimeridian',
      description:
        '500 fishing observations around Fiji crossing the antimeridian (lon 178..180 and -180..-178): obs_id, catch_kg',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'clusters-and-footprints'],
        never: [],
        forbiddenColumns: ['obs_id'],
        note: 'The extent looks 358 degrees wide; a sane engine warns or yields cell sizes that do not blow a 40,000/250,000 cell cap; the run must not produce a world-sized grid.'
      },
      datasets: [profileCollection('fiji', 'fiji', features)]
    });
  }

  // S16 two unrelated point layers
  {
    const random = makeRandom(16);
    const coffee = blobPoints(random, 300, [{c: [CX, CY], s: 0.03, w: 1}], 0.2, BOX).map((p, i) =>
      pointFeature(p, {shop_id: i, rating: Math.round((3 + random() * 2) * 10) / 10})
    );
    const fires = blobPoints(
      random,
      400,
      [{c: [-121.5, 44], s: 0.4, w: 1}],
      0.2,
      [-123, 42.5, -120, 45.5]
    ).map((p, i) => pointFeature(p, {fire_id: i, acres: Math.round(random() * 5000)}));
    situations.push({
      id: 'S16-two-unrelated-layers',
      description:
        'Coffee shops in Chicago (300; shop_id, rating) and wildfires in Oregon (400; fire_id, acres), 2000 km apart',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'clusters-and-footprints'],
        never: [
          'period-comparison',
          'straight-line-catchments',
          'count-by-area',
          'drive-time-catchments'
        ],
        forbiddenColumns: ['shop_id', 'fire_id'],
        note: 'Each layer analysed alone. Pair analyses (before/after, facilities vs observations, points in polygons) of unrelated layers are nonsense and must not lead.'
      },
      datasets: [
        profileCollection('coffee', 'coffee shops', coffee),
        profileCollection('fires', 'wildfires', fires)
      ]
    });
  }

  // S17 only numeric is an id
  {
    const random = makeRandom(17);
    const features = blobPoints(random, 500, [{c: [CX, CY], s: 0.03, w: 1}], 0.2, BOX).map((p, i) =>
      pointFeature(p, {OBJECTID: i + 1, name: `Site ${i % 40}`})
    );
    situations.push({
      id: 'S17-only-id-numeric',
      description: '500 points, columns OBJECTID (1..500) and name (text, 40 values)',
      expectation: {
        first: ['density-surface', 'clusters-and-footprints', 'hot-spot-map'],
        never: ['idw-surface', 'kriging-with-diagnostics', 'regression-with-diagnostics'],
        forbiddenColumns: ['OBJECTID'],
        note: 'No measurement exists: unweighted density/clusters; any value-role analysis must be blocked/needs-mapping, never ready with OBJECTID.'
      },
      datasets: [profileCollection('ids', 'ids', features)]
    });
  }

  // S18 same dataset twice
  {
    const random = makeRandom(3);
    const points = blobPoints(
      random,
      2000,
      [
        {c: [CX - 0.04, CY + 0.01], s: 0.008, w: 4},
        {c: [CX + 0.03, CY - 0.04], s: 0.012, w: 2}
      ],
      0.15,
      BOX
    );
    const features = points.map((p, i) =>
      pointFeature(p, {
        incident_id: i,
        date: isoDate(baseTime, Math.floor(random() * 365)),
        category: ['THEFT', 'BATTERY'][i % 2]
      })
    );
    situations.push({
      id: 'S18-same-dataset-twice',
      description: 'The same 2,000-incident layer loaded twice under two dataset ids',
      expectation: {
        first: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        never: [
          'period-comparison',
          'straight-line-catchments',
          'count-by-area',
          'drive-time-catchments'
        ],
        forbiddenColumns: ['incident_id'],
        note: 'Identical layers: before/after of the same data (zero change) and facilities-vs-observations are nonsense; single-layer analyses lead, and a pair rule should recognise identical datasets.'
      },
      datasets: [
        profileCollection('inc-a', 'incidents', features),
        profileCollection('inc-b', 'incidents (copy)', features)
      ]
    });
  }

  // S19 districts with cases and population
  const districtFeatures = (
    seed: number,
    withCases: boolean,
    names: {cases: string; population: string; rate?: string; extra?: string}
  ): CatalogFeature[] => {
    const random = makeRandom(seed);
    const out: CatalogFeature[] = [];
    for (let r = 0; r < 12; r++)
      for (let c = 0; c < 12; c++) {
        const population = Math.round(1500 + random() * 9000);
        const hot = Math.exp(-(((c - 3) ** 2 + (r - 8) ** 2) / 6)) * 0.04;
        const casesValue = Math.round(population * (0.01 + hot + random() * 0.004));
        const props: Record<string, unknown> = {name: `D${r}-${c}`, [names.population]: population};
        if (withCases) props[names.cases] = casesValue;
        if (names.rate) props[names.rate] = Math.round((casesValue / population) * 100000);
        if (names.extra) props[names.extra] = Math.round(population / 2.6);
        out.push(squareFeature(CX + c * 0.008, CY + r * 0.006, 0.008, 0.006, props));
      }
    return out;
  };
  situations.push({
    id: 'S19-districts-cases-population',
    description: '144 districts: name, population, cases (counts)',
    expectation: {
      first: ['rate-cluster-map', 'excess-risk'],
      never: ['regression-with-diagnostics', 'hot-spot-map', 'cluster-and-outlier-map'],
      roles: {
        'rate-cluster-map': {casesProperty: ['cases'], populationProperty: ['population']},
        'excess-risk': {casesProperty: ['cases'], populationProperty: ['population']}
      },
      forbiddenColumns: [],
      note: 'Counts and population: rate map / excess risk. Regression of cases on population is circular; Moran of raw cases maps population.'
    },
    datasets: [
      profileCollection(
        'districts',
        'districts',
        districtFeatures(19, true, {cases: 'cases', population: 'population'})
      )
    ]
  });

  // S20 districts with rate and population only
  situations.push({
    id: 'S20-districts-rate-population',
    description: '144 districts: name, population, rate_per_100k (no case counts)',
    expectation: {
      first: ['cluster-and-outlier-map', 'contiguity-correlogram'],
      never: ['rate-cluster-map', 'excess-risk', 'regression-with-diagnostics'],
      roles: {'cluster-and-outlier-map': {valueProperty: ['rate_per_100k']}},
      forbiddenColumns: ['population'],
      note: 'A ready-made rate: local Moran on rate_per_100k (population is a count and is the wrong value). Rate-from-counts analyses need cases that do not exist.'
    },
    datasets: [
      profileCollection(
        'districts-rate',
        'districts',
        districtFeatures(20, false, {cases: 'x', population: 'population', rate: 'rate_per_100k'})
      )
    ]
  });

  // S21 constant numeric column
  {
    const random = makeRandom(21);
    const features = blobPoints(random, 200, [{c: [CX, CY], s: 0.03, w: 1}], 0.3, BOX).map((p, i) =>
      pointFeature(p, {
        sensor_gen: 5,
        reading: Math.round(random() * 1000) / 10,
        n_alerts: Math.floor(random() * 6)
      })
    );
    situations.push({
      id: 'S21-constant-column',
      description: '200 points: sensor_gen (constant 5), reading (0..100 float), n_alerts (0..5)',
      expectation: {
        first: ['density-surface', 'hot-spot-map', 'idw-surface'],
        never: [],
        roles: {
          'idw-surface': {valueProperty: ['reading']},
          'hot-spot-map': {valueProperty: ['n_alerts', 'reading']}
        },
        forbiddenColumns: ['sensor_gen'],
        note: 'The constant column must never fill a value role; reading is the measurement.'
      },
      datasets: [profileCollection('sensors', 'sensors', features)]
    });
  }

  // S22 rain gauges
  {
    const random = makeRandom(22);
    const features = blobPoints(random, 150, [{c: [CX, CY], s: 0.1, w: 1}], 0.5, [
      CX - 0.3,
      CY - 0.2,
      CX + 0.3,
      CY + 0.2
    ]).map((p, i) =>
      pointFeature(p, {
        gauge_id: 7000 + i,
        rainfall_mm: Math.round((40 + (p[1] - CY) * 80 + random() * 10) * 10) / 10,
        elevation_m: Math.round(170 + random() * 60)
      })
    );
    situations.push({
      id: 'S22-rain-gauges',
      description: '150 rain gauges: gauge_id (int), rainfall_mm, elevation_m',
      expectation: {
        first: ['idw-surface', 'kriging-with-diagnostics'],
        never: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        roles: {
          'idw-surface': {valueProperty: ['rainfall_mm']},
          'kriging-with-diagnostics': {valueProperty: ['rainfall_mm']}
        },
        forbiddenColumns: ['gauge_id'],
        note: 'Rainfall is a surface: interpolate rainfall_mm, never gauge_id.'
      },
      datasets: [profileCollection('gauges', 'gauges', features)]
    });
  }

  // S23 table without geometry
  situations.push({
    id: 'S23-table-no-geometry',
    description:
      'Attribute table (CSV, no coordinates): address, revenue, visits (profileKeplerColumnTable)',
    expectation: {
      first: [],
      never: ['hot-spot-map', 'density-surface', 'idw-surface'],
      forbiddenColumns: [],
      note: 'Nothing spatial can run. Blocked list should say: give the table coordinate columns or a geometry column (geocode).'
    },
    datasets: [
      {
        profile: profileKeplerColumnTable(
          {
            rowCount: 3,
            columns: [
              {name: 'address', values: ['1 Main St', '2 Oak Ave', '3 Elm Rd']},
              {name: 'revenue', values: [100.5, 200.25, 150]},
              {name: 'visits', values: [3, 5, 9]}
            ]
          } as never,
          {datasetId: 'table', label: 'addresses.csv'}
        ) as unknown as KeplerDatasetProfile,
        collection: collection([])
      }
    ]
  });

  // S24 mixed geometry
  {
    const random = makeRandom(24);
    const features: CatalogFeature[] = [];
    for (let i = 0; i < 300; i++)
      features.push(
        pointFeature([CX + (random() - 0.5) * 0.1, CY + (random() - 0.5) * 0.1], {
          kind: 'p',
          value: Math.round(random() * 100)
        })
      );
    for (let i = 0; i < 50; i++)
      features.push(
        squareFeature(CX + (random() - 0.5) * 0.1, CY + (random() - 0.5) * 0.1, 0.01, 0.01, {
          kind: 'a',
          value: Math.round(random() * 100)
        })
      );
    for (let i = 0; i < 20; i++)
      features.push(
        lineFeature(
          [
            [CX + (random() - 0.5) * 0.1, CY],
            [CX + (random() - 0.5) * 0.1, CY + 0.02]
          ],
          {kind: 'l', value: Math.round(random() * 100)}
        )
      );
    situations.push({
      id: 'S24-mixed-geometry',
      description: 'One layer of 300 points, 50 polygons and 20 lines: kind (text), value',
      expectation: {
        first: ['density-surface', 'clusters-and-footprints', 'hot-spot-map'],
        never: [],
        forbiddenColumns: [],
        note: 'Dominant geometry is points: analyses on points; a warning that 70 rows of other geometry are ignored would be expected.'
      },
      datasets: [profileCollection('mixed', 'mixed', features)]
    });
  }

  // S25 Pacific-centred longitudes
  {
    const random = makeRandom(25);
    const features = Array.from({length: 400}, (_, i) =>
      pointFeature([150 + random() * 100, -10 + random() * 20], {
        buoy_id: i,
        sst: Math.round((26 + random() * 4) * 10) / 10
      })
    );
    situations.push({
      id: 'S25-pacific-0-360',
      description:
        '400 buoys with longitudes 150..250 (0-360 convention), latitudes -10..10: buoy_id, sst',
      expectation: {
        first: ['idw-surface', 'kriging-with-diagnostics'],
        never: [],
        roles: {'idw-surface': {valueProperty: ['sst']}},
        forbiddenColumns: ['buoy_id'],
        note: 'Samples of a surface (sea surface temperature). Longitudes beyond 180 are not valid EPSG:4326: must not be ready without confirmation, and a run must not silently mis-compute.'
      },
      datasets: [profileCollection('buoys', 'buoys', features)]
    });
  }

  // S26 local planar coordinates 0..50 m
  {
    const random = makeRandom(26);
    const features = Array.from({length: 300}, (_, i) =>
      pointFeature([random() * 50, random() * 40], {
        tree_id: i,
        dbh_cm: Math.round(random() * 800) / 10
      })
    );
    situations.push({
      id: 'S26-site-metres-0-50',
      description:
        '300 trees at local site coordinates in metres within 0..50 x 0..40 (inside the degree range): tree_id, dbh_cm',
      expectation: {
        first: ['density-surface', 'clusters-and-footprints'],
        never: [],
        forbiddenColumns: ['tree_id'],
        note: 'Extent fits degrees so the profile guesses geographic. A guessed model must be flagged and ideally needs-confirmation for a distance method; engine must not silently write EPSG:4326 (50 m read as 50 degrees).'
      },
      datasets: [profileCollection('site', 'site', features)]
    });
  }

  // S27 duplicated geocoded locations
  {
    const random = makeRandom(27);
    const addresses = blobPoints(random, 40, [{c: [CX, CY], s: 0.04, w: 1}], 0.2, BOX);
    const features = Array.from({length: 2000}, (_, i) =>
      pointFeature(addresses[Math.floor(random() * 40)], {
        order_id: i,
        amount: Math.round(random() * 5000) / 100
      })
    );
    situations.push({
      id: 'S27-duplicate-locations',
      description: '2,000 orders geocoded to only 40 distinct addresses: order_id, amount',
      expectation: {
        first: ['count-by-area', 'density-surface', 'hot-spot-map'],
        never: ['clusters-and-footprints', 'kriging-with-diagnostics', 'idw-surface'],
        roles: {'density-surface': {weightProperty: ['amount']}},
        forbiddenColumns: ['order_id'],
        note: 'Heavy duplication: distance defaults derived from nearest-neighbour distance are 0; engine must not produce a zero bandwidth/cell size and should warn about duplicate locations.'
      },
      datasets: [profileCollection('orders', 'orders', features)]
    });
  }

  // S28 grid cells with counts
  {
    const random = makeRandom(28);
    const features: CatalogFeature[] = [];
    for (let r = 0; r < 15; r++)
      for (let c = 0; c < 15; c++)
        features.push(
          squareFeature(CX + c * 0.005, CY + r * 0.005, 0.005, 0.005, {
            cell_id: r * 15 + c,
            n_events: Math.round(Math.exp(-(((c - 4) ** 2 + (r - 9) ** 2) / 8)) * 40 + random() * 6)
          })
        );
    situations.push({
      id: 'S28-grid-cells-counts',
      description: '225 equal-area grid cells: cell_id, n_events (counts)',
      expectation: {
        first: ['hot-spot-map', 'cluster-and-outlier-map'],
        never: ['rate-cluster-map', 'excess-risk', 'regression-with-diagnostics'],
        roles: {'cluster-and-outlier-map': {valueProperty: ['n_events']}},
        forbiddenColumns: ['cell_id'],
        note: 'Counts per equal-area cell are a legitimate Gi*/Moran input (no population to normalise by); engine should not demand a rate here, or at least resolve it by one user choice.'
      },
      datasets: [profileCollection('cells', 'cells', features)]
    });
  }

  // S29 year column only
  {
    const random = makeRandom(29);
    const features = blobPoints(
      random,
      3000,
      [{c: [CX - 0.03, CY + 0.02], s: 0.01, w: 1}],
      0.5,
      BOX
    ).map((p, i) =>
      pointFeature(p, {year: 2010 + Math.floor(random() * 11), category: ['a', 'b', 'c'][i % 3]})
    );
    situations.push({
      id: 'S29-year-column',
      description: '3,000 events with year (2010-2020 integer) and category',
      expectation: {
        first: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
        never: ['idw-surface', 'kriging-with-diagnostics', 'regression-with-diagnostics'],
        roles: {'density-surface': {weightProperty: []}, 'hot-spot-map': {valueProperty: []}},
        forbiddenColumns: ['year'],
        note: 'year is a calendar year, never a weight/value.'
      },
      datasets: [profileCollection('yearly', 'yearly', features)]
    });
  }

  // S30 districts residents/incidents/households
  situations.push({
    id: 'S30-districts-residents-incidents',
    description:
      '144 districts: name, residents, incidents, households (rate pair named residents/incidents)',
    expectation: {
      first: ['rate-cluster-map', 'excess-risk'],
      never: ['regression-with-diagnostics', 'hot-spot-map'],
      roles: {
        'rate-cluster-map': {casesProperty: ['incidents'], populationProperty: ['residents']},
        'excess-risk': {casesProperty: ['incidents'], populationProperty: ['residents']}
      },
      forbiddenColumns: [],
      note: 'incidents per resident (households is the wrong denominator when residents exists).'
    },
    datasets: [
      profileCollection(
        'districts-r',
        'districts',
        districtFeatures(30, true, {
          cases: 'incidents',
          population: 'residents',
          extra: 'households'
        })
      )
    ]
  });

  // S31 aggregated counts at points
  {
    const random = makeRandom(31);
    const features = blobPoints(random, 1500, [{c: [CX, CY], s: 0.03, w: 1}], 0.3, BOX).map(
      (p, i) =>
        pointFeature(p, {
          intersection_id: 4000 + i,
          num_crashes: Math.floor(random() * random() * 40)
        })
    );
    situations.push({
      id: 'S31-crash-counts-at-points',
      description: '1,500 intersections with num_crashes (count), intersection_id',
      expectation: {
        first: ['hot-spot-map', 'density-surface'],
        never: ['idw-surface', 'kriging-with-diagnostics'],
        roles: {
          'hot-spot-map': {valueProperty: ['num_crashes']},
          'density-surface': {weightProperty: ['num_crashes']}
        },
        forbiddenColumns: ['intersection_id'],
        note: 'Counts at points: weighted hot spots / density by num_crashes. Kriging of a count of events is wrong (a count at fixed sites is not a surface sample).'
      },
      datasets: [profileCollection('crashes', 'crashes', features)]
    });
  }

  // S32 county polygons with pct and median
  {
    const random = makeRandom(32);
    const features: CatalogFeature[] = [];
    for (let r = 0; r < 6; r++)
      for (let c = 0; c < 7; c++) {
        const population = Math.round(20000 + random() * 900000);
        features.push(
          squareFeature(CX + c * 0.1, CY + r * 0.1, 0.1, 0.1, {
            county: `C${r}-${c}`,
            population,
            median_income: Math.round(38000 + random() * 50000),
            pct_65plus: Math.round((10 + random() * 15) * 10) / 10,
            cases: Math.round(population * (0.01 + random() * 0.01))
          })
        );
      }
    situations.push({
      id: 'S32-counties-small',
      description: '42 counties: county, population, median_income, pct_65plus, cases',
      expectation: {
        first: ['rate-cluster-map', 'excess-risk', 'cluster-and-outlier-map'],
        never: ['hot-spot-map'],
        roles: {
          'regression-with-diagnostics': {
            responseProperty: ['median_income', 'pct_65plus'],
            predictorProperties: ['median_income', 'pct_65plus']
          }
        },
        forbiddenColumns: [],
        note: 'Case rates by county; if regression is offered, response and predictor must be median_income and pct_65plus, never cases~population.'
      },
      datasets: [profileCollection('counties', 'counties', features)]
    });
  }

  return situations;
}
