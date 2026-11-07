// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as h3 from 'h3-js';

const dataDirectory = new URL('../public/data/bixi-flows/', import.meta.url);

async function readColumn(name, Type) {
  const bytes = await readFile(new URL(`${name}.bin`, dataDirectory));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Independent source-row accounting: no scene preprocessing or GPU algorithm is imported. */
export async function loadBixiCellFlowReference(resolution) {
  const [positions, origins, destinations, values, departures, manifestText] = await Promise.all([
    readColumn('locations', Float32Array), readColumn('origin', Uint16Array),
    readColumn('destination', Uint16Array), readColumn('count', Uint16Array),
    readColumn('stationDepartures', Uint32Array), readFile(new URL('manifest.json', dataDirectory), 'utf8')
  ]);
  const manifest = JSON.parse(manifestText);
  const stationCells = Array.from({length: positions.length / 2}, (_, row) => h3.latLngToCell(positions[row * 2 + 1], positions[row * 2], resolution));
  const cells = new Map([...new Set(stationCells)].map(key => [key, {outgoing: 0, incoming: 0}]));
  const pairTotals = new Map();
  let sameCell = 0;
  let neighboring = 0;
  let nonneighboring = 0;
  let retainedTotal = 0;
  for (let row = 0; row < values.length; row++) {
    assert.notEqual(origins[row], destinations[row], 'The retained source excludes same-station rides');
    assert.ok(values[row] >= 3);
    const origin = stationCells[origins[row]];
    const destination = stationCells[destinations[row]];
    const value = values[row];
    cells.get(origin).outgoing += value;
    cells.get(destination).incoming += value;
    const pair = `${origin}/${destination}`;
    pairTotals.set(pair, (pairTotals.get(pair) ?? 0) + value);
    retainedTotal += value;
    if (origin === destination) sameCell += value;
    else if (h3.areNeighborCells(origin, destination)) neighboring += value;
    else nonneighboring += value;
  }
  const crossStationTotal = departures.reduce((sum, value) => sum + value, 0);
  assert.equal(sameCell + neighboring + nonneighboring, retainedTotal);
  assert.equal([...cells.values()].reduce((sum, value) => sum + value.outgoing, 0), retainedTotal);
  assert.equal([...cells.values()].reduce((sum, value) => sum + value.incoming, 0), retainedTotal);
  assert.equal(crossStationTotal + manifest.properties.sameStationRides, manifest.properties.totalRides);
  return {
    cells, pairTotals, stationCells,
    summary: {
      resolution, stationCount: stationCells.length, sourcePairCount: values.length,
      cellCount: cells.size, pairCount: pairTotals.size, retainedTotal, sameCell, neighboring,
      nonneighboring, crossStationTotal, excludedRareRides: crossStationTotal - retainedTotal,
      sameStationRides: manifest.properties.sameStationRides, sourceTotalRides: manifest.properties.totalRides
    }
  };
}
