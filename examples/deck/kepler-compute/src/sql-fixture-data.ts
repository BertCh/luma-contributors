// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Deterministic string and join fixture for the native SQL explorer (`?sqlFixture=strings`).
 * Pure data: no imports, so node specs and the Playwright script (node type stripping) can load it.
 */

export const SQL_FIXTURE_URL_PARAMETER = 'sqlFixture';
export const SQL_FIXTURE_NAME = 'strings';
export const SQL_TRIPS_TABLE_ID = 'sql_trips';
export const SQL_ZONES_TABLE_ID = 'sql_zones';
export const SQL_FIXTURE_TRIP_COUNT = 2000;
export const SQL_FIXTURE_ZONE_COUNT = 50;
/** First trip timestamp base: 2023-11-14T22:13:20Z. */
export const SQL_FIXTURE_EPOCH_SECONDS_BASE = 1700000000;

const ADJECTIVES = [
  'North',
  'South',
  'East',
  'West',
  'Upper',
  'Lower',
  'Central',
  'Old',
  'New',
  'Little',
  'Grand',
  'Royal'
];
const NOUNS = ['Park', 'Heights', 'Village', 'Bay', 'Point'];
const HOME_BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'];
const REGION_BY_BOROUGH: Record<string, string> = {
  Manhattan: 'Core',
  Brooklyn: 'East',
  Queens: 'East',
  Bronx: 'North',
  'Staten Island': 'South'
};

/** All 60 zone names; sql_zones holds indexes 0..49 and trips use indexes 5..59. */
export const SQL_FIXTURE_ZONE_NAMES: readonly string[] = ADJECTIVES.flatMap(adjective =>
  NOUNS.map(noun => `${adjective} ${noun}`)
);

export type SqlFixtureTrips = {
  id: Int32Array;
  borough: (string | null)[];
  zone: (string | null)[];
  fare: Float32Array;
  passengers: Int32Array;
  code: Int32Array;
  pickupEpoch: Int32Array;
};

export type SqlFixtureZones = {
  zone: string[];
  zoneId: Int32Array;
  region: string[];
  capacity: Int32Array;
  borough: string[];
};

export type SqlFixtureData = {trips: SqlFixtureTrips; zones: SqlFixtureZones};

/** Home borough of a zone index; shared by the trip generator and the sql_zones table. */
function getHomeBorough(zoneIndex: number): string {
  return HOME_BOROUGHS[zoneIndex % HOME_BOROUGHS.length];
}

/** Generates the same tables on every call (seeded LCG, no Math.random or Date). */
export function createSqlFixtureData(): SqlFixtureData {
  let randomState = 20260;
  const random = (): number => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState / 4294967296;
  };

  const tripCount = SQL_FIXTURE_TRIP_COUNT;
  const trips: SqlFixtureTrips = {
    id: new Int32Array(tripCount),
    borough: [],
    zone: [],
    fare: new Float32Array(tripCount),
    passengers: new Int32Array(tripCount),
    code: new Int32Array(tripCount),
    pickupEpoch: new Int32Array(tripCount)
  };
  for (let rowIndex = 0; rowIndex < tripCount; rowIndex++) {
    const zoneIndex = 5 + Math.floor(random() * 55);
    const home = getHomeBorough(zoneIndex);
    const boroughRoll = random();
    let borough: string | null;
    if (boroughRoll < 0.65) borough = home;
    else if (boroughRoll < 0.73) borough = home.toLowerCase();
    else if (boroughRoll < 0.78) borough = home.toUpperCase();
    else if (boroughRoll < 0.84) borough = ` ${home} `;
    else if (boroughRoll < 0.93) borough = getHomeBorough(zoneIndex + 1 + Math.floor(random() * 4));
    else if (boroughRoll < 0.96) borough = '';
    else borough = null;

    const zoneRoll = random();
    let zone: string | null = SQL_FIXTURE_ZONE_NAMES[zoneIndex];
    // Rows whose zone equals their borough make borough = zone non-empty; some zones are null.
    if (zoneRoll < 0.03 && borough) zone = borough;
    else if (zoneRoll < 0.06) zone = null;

    trips.id[rowIndex] = rowIndex + 1;
    trips.borough.push(borough);
    trips.zone.push(zone);
    // Quarter-dollar steps are exact in float32.
    trips.fare[rowIndex] = 2.5 + Math.floor(random() * 470) * 0.25;
    trips.passengers[rowIndex] = 1 + Math.floor(random() * 6);
    trips.code[rowIndex] = Math.floor(random() * 10);
    trips.pickupEpoch[rowIndex] =
      SQL_FIXTURE_EPOCH_SECONDS_BASE + Math.floor(random() * 30 * 86400);
  }

  const zones: SqlFixtureZones = {
    zone: [],
    zoneId: new Int32Array(SQL_FIXTURE_ZONE_COUNT),
    region: [],
    capacity: new Int32Array(SQL_FIXTURE_ZONE_COUNT),
    borough: []
  };
  for (let zoneIndex = 0; zoneIndex < SQL_FIXTURE_ZONE_COUNT; zoneIndex++) {
    const borough = getHomeBorough(zoneIndex);
    zones.zone.push(SQL_FIXTURE_ZONE_NAMES[zoneIndex]);
    zones.zoneId[zoneIndex] = zoneIndex + 1;
    zones.region.push(REGION_BY_BOROUGH[borough]);
    zones.capacity[zoneIndex] = 100 + ((zoneIndex * 37) % 400);
    zones.borough.push(borough);
  }
  return {trips, zones};
}

export type SqlFixtureRow = Record<string, string | number | null>;

/** Row objects keyed by SQL column name, for oracle evaluation. */
export function makeSqlFixtureRows(data: SqlFixtureData): {
  trips: SqlFixtureRow[];
  zones: SqlFixtureRow[];
} {
  const trips = Array.from(data.trips.id, (id, rowIndex) => ({
    id,
    borough: data.trips.borough[rowIndex],
    zone: data.trips.zone[rowIndex],
    fare: data.trips.fare[rowIndex],
    passengers: data.trips.passengers[rowIndex],
    code: data.trips.code[rowIndex],
    pickup_epoch: data.trips.pickupEpoch[rowIndex]
  }));
  const zones = Array.from(data.zones.zoneId, (zoneId, rowIndex) => ({
    zone: data.zones.zone[rowIndex],
    zone_id: zoneId,
    region: data.zones.region[rowIndex],
    capacity: data.zones.capacity[rowIndex],
    borough: data.zones.borough[rowIndex]
  }));
  return {trips, zones};
}
