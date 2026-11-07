// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSqlFixtureData, makeSqlFixtureRows} from '../src/sql-fixture-data.ts';

/**
 * String and join SQL queries for the `?sqlFixture=strings` dataset, each with the expected route
 * and an oracle row count computed in plain JavaScript from the exported fixture data.
 * Semantics follow DuckDB: NULL never satisfies a predicate, ANTI JOIN keeps NULL keys, CAST to
 * INTEGER rounds half away from zero, epoch columns are seconds.
 *
 * Flip a group (or one query) from 'gpu' to 'duckdb' here, or per run with
 * KEPLER_SQL_BACKEND_OVERRIDES="name=duckdb,group:join=duckdb".
 */
export const EXPECTED_BACKEND_BY_GROUP = {
  'string-predicate': 'gpu',
  'string-cast': 'gpu',
  'date-cast': 'gpu',
  'expression-cast': 'gpu',
  between: 'gpu',
  'string-function': 'gpu',
  'string-compare': 'gpu',
  'string-group': 'gpu',
  join: 'gpu',
  'semi-anti-join': 'gpu',
  fallback: 'duckdb'
};

const {trips, zones} = makeSqlFixtureRows(createSqlFixtureData());
const zoneByName = new Map(zones.map(zone => [zone.zone, zone]));
const matchedTrips = trips
  .map(trip => ({trip, zone: trip.zone === null ? undefined : zoneByName.get(trip.zone)}))
  .filter(pair => pair.zone);

function matchesLike(value, pattern, ignoreCase = false) {
  if (value === null) return false;
  const source = pattern
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    .replace(/%/g, '.*')
    .replace(/_/g, '.');
  return new RegExp(`^${source}$`, ignoreCase ? 'is' : 's').test(value);
}
const roundHalfAwayFromZero = value => Math.sign(value) * Math.round(Math.abs(value));
const countTrips = predicate => trips.filter(predicate).length;
const secondsOfDay = (year, monthIndex, day) => Date.UTC(year, monthIndex, day) / 1000;

const T = 'sql_trips';
const Z = 'sql_zones';
const JOIN_ON_ZONE = `${T} t JOIN ${Z} z ON t.zone = z.zone`;

/** @type {{name: string, group: string, sql: string, rowCount: number, backend?: string, fieldTypes?: Record<string, string>}[]} */
const queryDefinitions = [
  // String predicates.
  {name: 'eq', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE borough = 'Queens'`,
    rowCount: countTrips(r => r.borough === 'Queens')},
  {name: 'eq-empty', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE borough = ''`,
    rowCount: countTrips(r => r.borough === '')},
  {name: 'is-null', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE borough IS NULL`,
    rowCount: countTrips(r => r.borough === null)},
  {name: 'in-list', group: 'string-predicate',
    sql: `SELECT id FROM ${T} WHERE borough IN ('Queens', 'Bronx', 'manhattan')`,
    rowCount: countTrips(r => ['Queens', 'Bronx', 'manhattan'].includes(r.borough))},
  {name: 'not-in-list', group: 'string-predicate',
    sql: `SELECT id FROM ${T} WHERE borough NOT IN ('Queens', 'Bronx')`,
    rowCount: countTrips(r => r.borough !== null && !['Queens', 'Bronx'].includes(r.borough))},
  {name: 'like-suffix', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE zone LIKE '%Park'`,
    rowCount: countTrips(r => matchesLike(r.zone, '%Park'))},
  {name: 'like-single-char', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE zone LIKE 'E_st%'`,
    rowCount: countTrips(r => matchesLike(r.zone, 'E_st%'))},
  {name: 'not-like', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE zone NOT LIKE '%Park'`,
    rowCount: countTrips(r => r.zone !== null && !matchesLike(r.zone, '%Park'))},
  {name: 'ilike', group: 'string-predicate', sql: `SELECT id FROM ${T} WHERE borough ILIKE 'manhattan'`,
    rowCount: countTrips(r => matchesLike(r.borough, 'manhattan', true))},

  // CAST numeric to VARCHAR.
  {name: 'cast-varchar-filter', group: 'string-cast',
    sql: `SELECT id FROM ${T} WHERE CAST(code AS VARCHAR) = '3'`,
    rowCount: countTrips(r => r.code === 3)},
  {name: 'cast-varchar-project', group: 'string-cast',
    sql: `SELECT id, CAST(passengers AS VARCHAR) AS passengers_text FROM ${T} WHERE passengers > 4`,
    rowCount: countTrips(r => r.passengers > 4)},

  // CAST epoch seconds to DATE and TIMESTAMP.
  {name: 'cast-timestamp', group: 'date-cast',
    sql: `SELECT id, CAST(pickup_epoch AS TIMESTAMP) AS pickup_time FROM ${T} WHERE CAST(pickup_epoch AS TIMESTAMP) >= '2023-11-20 00:00:00'`,
    rowCount: countTrips(r => r.pickup_epoch >= secondsOfDay(2023, 10, 20)),
    fieldTypes: {pickup_time: 'timestamp'}},
  {name: 'cast-date', group: 'date-cast',
    sql: `SELECT id, CAST(pickup_epoch AS DATE) AS pickup_date FROM ${T} WHERE CAST(pickup_epoch AS DATE) >= '2023-12-01'`,
    rowCount: countTrips(r => r.pickup_epoch >= secondsOfDay(2023, 11, 1))},

  // CAST of an expression.
  {name: 'cast-expression-integer', group: 'expression-cast',
    sql: `SELECT id, CAST(fare + passengers AS INTEGER) AS total FROM ${T} WHERE CAST(fare + passengers AS INTEGER) > 100`,
    rowCount: countTrips(r => roundHalfAwayFromZero(r.fare + r.passengers) > 100)},

  // BETWEEN.
  {name: 'between-numeric', group: 'between', sql: `SELECT id FROM ${T} WHERE fare BETWEEN 10 AND 20`,
    rowCount: countTrips(r => r.fare >= 10 && r.fare <= 20)},
  {name: 'between-string', group: 'between', sql: `SELECT id FROM ${T} WHERE borough BETWEEN 'B' AND 'N'`,
    rowCount: countTrips(r => r.borough !== null && r.borough >= 'B' && r.borough <= 'N')},

  // String functions.
  {name: 'upper', group: 'string-function', sql: `SELECT id FROM ${T} WHERE UPPER(borough) = 'QUEENS'`,
    rowCount: countTrips(r => r.borough !== null && r.borough.toUpperCase() === 'QUEENS')},
  {name: 'lower', group: 'string-function', sql: `SELECT id FROM ${T} WHERE LOWER(zone) = 'east park'`,
    rowCount: countTrips(r => r.zone !== null && r.zone.toLowerCase() === 'east park')},
  {name: 'trim', group: 'string-function', sql: `SELECT id FROM ${T} WHERE TRIM(borough) = 'Queens'`,
    rowCount: countTrips(r => r.borough !== null && r.borough.trim() === 'Queens')},
  {name: 'length', group: 'string-function', sql: `SELECT id FROM ${T} WHERE LENGTH(zone) > 11`,
    rowCount: countTrips(r => r.zone !== null && r.zone.length > 11)},
  {name: 'substr', group: 'string-function', sql: `SELECT id FROM ${T} WHERE SUBSTR(zone, 1, 4) = 'East'`,
    rowCount: countTrips(r => r.zone !== null && r.zone.slice(0, 4) === 'East')},
  {name: 'replace', group: 'string-function',
    sql: `SELECT id FROM ${T} WHERE REPLACE(zone, ' ', '_') = 'East_Park'`,
    rowCount: countTrips(r => r.zone !== null && r.zone.replaceAll(' ', '_') === 'East_Park')},
  {name: 'concat-constant', group: 'string-function',
    sql: `SELECT id FROM ${T} WHERE borough || '!' = 'Queens!'`,
    rowCount: countTrips(r => r.borough === 'Queens')},
  {name: 'upper-project', group: 'string-function',
    sql: `SELECT id, UPPER(borough) AS borough_upper, TRIM(zone) AS zone_trimmed FROM ${T} WHERE id <= 100`,
    rowCount: 100},
  {name: 'spot-padded-row', group: 'string-function',
    sql: `SELECT id FROM ${T} WHERE TRIM(borough) = 'Queens' AND id = ${trips.find(r => r.borough === ' Queens ')?.id}`,
    rowCount: 1},

  // Two string columns.
  {name: 'column-equals-column', group: 'string-compare', sql: `SELECT id FROM ${T} WHERE borough = zone`,
    rowCount: countTrips(r => r.borough !== null && r.borough === r.zone)},

  // GROUP BY a string column.
  {name: 'group-by-string', group: 'string-group',
    sql: `SELECT borough, COUNT(*) AS trip_count FROM ${T} WHERE borough IS NOT NULL GROUP BY borough`,
    rowCount: new Set(trips.filter(r => r.borough !== null).map(r => r.borough)).size},

  // Joins.
  {name: 'join-where', group: 'join',
    sql: `SELECT t.id, z.region FROM ${JOIN_ON_ZONE} WHERE z.capacity > 300`,
    rowCount: matchedTrips.filter(pair => pair.zone.capacity > 300).length},
  {name: 'join-group-order-limit', group: 'join',
    sql: `SELECT z.region, COUNT(*) AS trip_count FROM ${JOIN_ON_ZONE} GROUP BY z.region ORDER BY trip_count DESC LIMIT 3`,
    rowCount: Math.min(3, new Set(matchedTrips.map(pair => pair.zone.region)).size)},
  {name: 'join-expression', group: 'join',
    sql: `SELECT t.id, t.fare * z.capacity AS load FROM ${JOIN_ON_ZONE} WHERE t.fare * z.capacity > 20000`,
    rowCount: matchedTrips.filter(pair => pair.trip.fare * pair.zone.capacity > 20000).length},
  {name: 'join-multi-key', group: 'join',
    sql: `SELECT t.id, z.zone_id FROM ${T} t JOIN ${Z} z ON t.borough = z.borough AND t.zone = z.zone`,
    rowCount: matchedTrips.filter(pair => pair.trip.borough === pair.zone.borough).length},
  {name: 'join-left', group: 'join',
    sql: `SELECT t.id, z.zone_id FROM ${T} t LEFT JOIN ${Z} z ON t.zone = z.zone`,
    rowCount: trips.length},
  {name: 'semi-join', group: 'semi-anti-join',
    sql: `SELECT t.id FROM ${T} t SEMI JOIN ${Z} z ON t.zone = z.zone`,
    rowCount: matchedTrips.length},
  {name: 'anti-join', group: 'semi-anti-join',
    sql: `SELECT t.id FROM ${T} t ANTI JOIN ${Z} z ON t.zone = z.zone`,
    rowCount: trips.length - matchedTrips.length},

  // SQL6: numeric multi-key outer/semi/anti joins (exact ranks), three-column string expressions,
  // CAST to VARCHAR beyond the old 65,536 limit is covered by the headless spec (fixture is 2,000 rows).
  {name: 'join-numeric-multi-key-left', group: 'join',
    sql: `SELECT t.id, z.zone_id FROM ${T} t LEFT JOIN ${Z} z ON t.id = z.zone_id AND t.passengers = z.zone_id`,
    rowCount: trips.reduce(
      (total, trip) =>
        total + Math.max(1, zones.filter(zone => zone.zone_id === trip.id && zone.zone_id === trip.passengers).length),
      0
    )},
  {name: 'semi-join-numeric-multi-key', group: 'semi-anti-join',
    sql: `SELECT t.id FROM ${T} t SEMI JOIN ${T} u ON t.passengers = u.code AND t.code = u.passengers`,
    rowCount: trips.filter(trip => trips.some(other => other.code === trip.passengers && other.passengers === trip.code)).length},
  {name: 'anti-join-numeric-multi-key', group: 'semi-anti-join',
    sql: `SELECT t.id FROM ${T} t ANTI JOIN ${T} u ON t.passengers = u.code AND t.code = u.passengers`,
    rowCount: trips.filter(trip => !trips.some(other => other.code === trip.passengers && other.passengers === trip.code)).length},
  {name: 'string-concat-three-columns', group: 'string-function',
    sql: `SELECT t.id, t.borough || '/' || t.zone || '/' || z.region AS path FROM ${T} t JOIN ${Z} z ON t.zone = z.zone`,
    // NULL || x is NULL but the joined row is still returned, so the count is every matched trip.
    rowCount: matchedTrips.length},

  // Documented unsupported forms: window functions must keep falling back to DuckDB.
  {name: 'fallback-window-row-number', group: 'fallback',
    sql: `SELECT id, ROW_NUMBER() OVER (PARTITION BY borough ORDER BY fare DESC) AS row_number FROM ${T}`,
    rowCount: trips.length},
  {name: 'fallback-window-sum', group: 'fallback',
    sql: `SELECT id, SUM(fare) OVER (PARTITION BY zone) AS zone_fare FROM ${T} WHERE id <= 500`,
    rowCount: 500}
];

export const SQL_FIXTURE_QUERIES = queryDefinitions.map(query => ({
  ...query,
  backend: query.backend ?? EXPECTED_BACKEND_BY_GROUP[query.group]
}));
