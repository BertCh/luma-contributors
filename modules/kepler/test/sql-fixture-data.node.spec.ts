// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  SQL_FIXTURE_TRIP_COUNT,
  SQL_FIXTURE_ZONE_COUNT,
  createSqlFixtureData,
  makeSqlFixtureRows
} from '../../../examples/deck/kepler-compute/src/sql-fixture-data';
import {
  EXPECTED_BACKEND_BY_GROUP,
  SQL_FIXTURE_QUERIES
} from '../../../examples/deck/kepler-compute/scripts/native-sql-fixture-queries.mjs';

test('sql fixture data is deterministic', () => {
  expect(makeSqlFixtureRows(createSqlFixtureData())).toEqual(
    makeSqlFixtureRows(createSqlFixtureData())
  );
});

test('sql fixture trips cover strings, nulls and numeric columns', () => {
  const {trips} = createSqlFixtureData();
  expect(trips.id.length).toBe(SQL_FIXTURE_TRIP_COUNT);
  expect(trips.borough).toHaveLength(SQL_FIXTURE_TRIP_COUNT);
  expect(trips.borough).toContain(null);
  expect(trips.borough).toContain('');
  expect(trips.borough).toContain(' Queens ');
  expect(trips.borough).toContain('manhattan');
  expect(trips.borough).toContain('BROOKLYN');
  expect(trips.zone).toContain(null);
  expect(new Set(trips.borough).size).toBeLessThan(40);
  expect(trips.borough.some((borough, index) => borough && borough === trips.zone[index])).toBe(
    true
  );
  expect(Math.min(...trips.passengers)).toBeGreaterThanOrEqual(1);
  expect(new Set(trips.code).size).toBe(10);
  expect(Math.min(...trips.pickupEpoch)).toBeGreaterThanOrEqual(1700000000);
  expect(trips.fare.every(fare => fare === Math.fround(fare))).toBe(true);
});

test('sql fixture zones are unique and partially matched by trips', () => {
  const {trips, zones} = createSqlFixtureData();
  expect(zones.zone).toHaveLength(SQL_FIXTURE_ZONE_COUNT);
  expect(new Set(zones.zone).size).toBe(SQL_FIXTURE_ZONE_COUNT);
  expect(new Set(zones.zoneId).size).toBe(SQL_FIXTURE_ZONE_COUNT);
  const zoneNames = new Set(zones.zone);
  const tripZones = new Set(trips.zone.filter(zone => zone !== null));
  expect([...tripZones].some(zone => !zoneNames.has(zone))).toBe(true);
  expect([...zoneNames].some(zone => !tripZones.has(zone))).toBe(true);
  // Multi-key join matches fewer rows than the zone-only join.
  const zoneByName = new Map(zones.zone.map((zone, index) => [zone, zones.borough[index]]));
  const zoneMatches = trips.zone.filter(zone => zone !== null && zoneByName.has(zone)).length;
  const multiKeyMatches = trips.zone.filter(
    (zone, index) => zone !== null && zoneByName.get(zone) === trips.borough[index]
  ).length;
  expect(multiKeyMatches).toBeGreaterThan(0);
  expect(multiKeyMatches).toBeLessThan(zoneMatches);
});

test('sql fixture query oracles are non-trivial and consistent', () => {
  const names = SQL_FIXTURE_QUERIES.map(query => query.name);
  expect(new Set(names).size).toBe(names.length);
  for (const query of SQL_FIXTURE_QUERIES) {
    expect(Object.keys(EXPECTED_BACKEND_BY_GROUP)).toContain(query.group);
    expect(query.rowCount, query.name).toBeGreaterThan(0);
    expect(query.rowCount, query.name).toBeLessThanOrEqual(SQL_FIXTURE_TRIP_COUNT);
  }
  const rowCount = (name: string) =>
    SQL_FIXTURE_QUERIES.find(query => query.name === name)?.rowCount;
  expect(rowCount('semi-join')! + rowCount('anti-join')!).toBe(SQL_FIXTURE_TRIP_COUNT);
  expect(rowCount('join-multi-key')!).toBeLessThan(rowCount('semi-join')!);
  expect(SQL_FIXTURE_QUERIES.filter(query => query.backend === 'duckdb')).toHaveLength(2);
});
