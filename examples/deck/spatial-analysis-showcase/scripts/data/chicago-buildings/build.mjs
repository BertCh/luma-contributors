#!/usr/bin/env node
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Rebuild the bounded Chicago building-footprint extract from the City of Chicago Data Portal.
// No credentials are required. The fixed bbox keeps this teaching dataset compact and reproducible.

import {mkdir, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const outputDirectory = fileURLToPath(new URL('../../../public/data/chicago-buildings/', import.meta.url));
const sourceUrl = 'https://data.cityofchicago.org/resource/syp8-uezg.geojson';
const bbox = [-87.650, 41.870, -87.620, 41.895];
const where = `within_box(the_geom,${bbox[3]},${bbox[0]},${bbox[1]},${bbox[2]}) AND bldg_statu = 'ACTIVE'`;
const query = new URLSearchParams({$where: where, $limit: '10000', $order: 'bldg_id'});
const response = await fetch(`${sourceUrl}?${query}`);
if (!response.ok) {
  throw new Error(`Chicago Data Portal request failed (${response.status} ${response.statusText})`);
}
const collection = await response.json();
if (!Array.isArray(collection.features) || collection.features.length === 0) {
  throw new Error('Chicago Data Portal returned no footprints for the requested bbox');
}
if (collection.features.length >= 10000) {
  throw new Error('Footprint extract reached the 10,000 feature limit; narrow the bbox explicitly');
}

await mkdir(outputDirectory, {recursive: true});
const geojsonText = `${JSON.stringify(collection)}\n`;
const payloadSha256 = createHash('sha256').update(geojsonText).digest('hex');
await writeFile(path.join(outputDirectory, 'buildings.geojson'), geojsonText);
const manifest = {
  id: 'chicago-buildings',
  version: 1,
  kind: 'polygons',
  count: collection.features.length,
  bbox,
  crs: 'EPSG:4326',
  geometry: {type: 'polygons', file: 'buildings.geojson'},
  properties: {
    sourceDataset: 'Building Footprints',
    sourceDatasetId: 'syp8-uezg',
    sourceUrl,
    queryBbox: bbox,
    downloadedAt: new Date().toISOString(),
    payloadSha256,
    attribution: 'City of Chicago Data Portal, Building Footprints (syp8-uezg)',
    license: 'City of Chicago Data Portal Terms of Use',
    notes: 'Bounded query result; available source fields include bldg_id, bldg_statu, stories, year_built and address components f_add1/t_add1. Story counts are incomplete and are not complete building heights.'
  }
};
await writeFile(path.join(outputDirectory, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(
  path.join(outputDirectory, 'SOURCE.md'),
  `# Chicago building footprints\n\n` +
    `Source: [City of Chicago Data Portal — Building Footprints](https://data.cityofchicago.org/Buildings/Building-Footprints-current-/hz9b-7nh8), dataset ID \`syp8-uezg\`.\n\n` +
    `The GeoJSON is queried directly from \`${sourceUrl}\` using the fixed bbox \`${bbox.join(', ')}\` (west, south, east, north) and the source's \`bldg_statu = 'ACTIVE'\` filter. Rebuild with \`node scripts/data/chicago-buildings/build.mjs\`. The script fails if the source is unavailable, empty, or reaches its 10,000-row cap.\n\n` +
    `Attribution: City of Chicago Data Portal, Building Footprints. Review the [dataset terms of use](https://data.cityofchicago.org/terms-of-use) for reuse conditions. The payload SHA-256 is recorded in the manifest; the download timestamp changes on every build. Available source fields include \`bldg_id\`, \`bldg_statu\`, \`stories\`, \`year_built\`, and address components such as \`f_add1\` and \`t_add1\`. Story counts are incomplete and must not be interpreted as complete height coverage.\n`
);
console.log(`Saved ${collection.features.length} real building footprints to ${outputDirectory}`);
