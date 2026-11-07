// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  SQL_FIXTURE_NAME,
  SQL_FIXTURE_URL_PARAMETER,
  SQL_TRIPS_TABLE_ID,
  SQL_ZONES_TABLE_ID,
  createSqlFixtureData
} from './sql-fixture-data';

export type SqlFixtureHost = {
  /** True when the runtime can accept a publication. */
  isReady(): boolean;
  /** Publishes the table as a resident dataset whose id (and SQL table name) is `datasetId`. */
  publishTable(options: {datasetId: string; label: string; table: arrow.Table}): Promise<void>;
};

/** Arrow tables of the fixture; string columns are Utf8 so the host prepares dictionaries. */
export function makeSqlFixtureTables(): {trips: arrow.Table; zones: arrow.Table} {
  const {trips, zones} = createSqlFixtureData();
  return {
    trips: new arrow.Table({
      id: arrow.makeVector(trips.id),
      borough: arrow.vectorFromArray(trips.borough, new arrow.Utf8()),
      zone: arrow.vectorFromArray(trips.zone, new arrow.Utf8()),
      fare: arrow.makeVector(trips.fare),
      passengers: arrow.makeVector(trips.passengers),
      code: arrow.makeVector(trips.code),
      pickup_epoch: arrow.makeVector(trips.pickupEpoch)
    }),
    zones: new arrow.Table({
      zone: arrow.vectorFromArray(zones.zone, new arrow.Utf8()),
      zone_id: arrow.makeVector(zones.zoneId),
      region: arrow.vectorFromArray(zones.region, new arrow.Utf8()),
      capacity: arrow.makeVector(zones.capacity),
      borough: arrow.vectorFromArray(zones.borough, new arrow.Utf8())
    })
  };
}

/** `?sqlFixture=strings` publishes `sql_trips` then `sql_zones` once the runtime is ready. */
export function installSqlFixture(host: SqlFixtureHost, search: string): void {
  if (new URLSearchParams(search).get(SQL_FIXTURE_URL_PARAMETER) !== SQL_FIXTURE_NAME) return;
  const timer = setInterval(() => {
    if (!host.isReady()) return;
    clearInterval(timer);
    const {trips, zones} = makeSqlFixtureTables();
    void host
      .publishTable({datasetId: SQL_TRIPS_TABLE_ID, label: SQL_TRIPS_TABLE_ID, table: trips})
      .then(() =>
        host.publishTable({datasetId: SQL_ZONES_TABLE_ID, label: SQL_ZONES_TABLE_ID, table: zones})
      )
      // biome-ignore lint/suspicious/noConsole: the fixture is a dev sandbox option; surface failures
      .catch(error => console.error('SQL fixture publication failed', error));
  }, 200);
}
