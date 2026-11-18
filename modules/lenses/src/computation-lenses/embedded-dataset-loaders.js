// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Native Node requires JSON attributes; Vite browser modules must omit them. Keep both import
// expressions literal so esbuild and Vite can discover and package every dataset.
const IS_NATIVE_NODE = typeof process !== 'undefined' && Boolean(process.versions?.node);
export const EMBEDDED_DATASET_LOADERS = {
  'f-chicago-nature.json': () =>
    IS_NATIVE_NODE
      ? import('./data/f-chicago-nature.json', {with: {type: 'json'}})
      : import('./data/f-chicago-nature.json'),
  'f-chicago-tracts.json': () =>
    IS_NATIVE_NODE
      ? import('./data/f-chicago-tracts.json', {with: {type: 'json'}})
      : import('./data/f-chicago-tracts.json'),
  'f-ghcn-stations.json': () =>
    IS_NATIVE_NODE
      ? import('./data/f-ghcn-stations.json', {with: {type: 'json'}})
      : import('./data/f-ghcn-stations.json'),
  'n-chicago-loop-crashes.json': () =>
    IS_NATIVE_NODE
      ? import('./data/n-chicago-loop-crashes.json', {with: {type: 'json'}})
      : import('./data/n-chicago-loop-crashes.json'),
  'n-chicago-loop-polylines.json': () =>
    IS_NATIVE_NODE
      ? import('./data/n-chicago-loop-polylines.json', {with: {type: 'json'}})
      : import('./data/n-chicago-loop-polylines.json'),
  'n-chicago-loop-roads.json': () =>
    IS_NATIVE_NODE
      ? import('./data/n-chicago-loop-roads.json', {with: {type: 'json'}})
      : import('./data/n-chicago-loop-roads.json'),
  'n-chicago-loop-traces.json': () =>
    IS_NATIVE_NODE
      ? import('./data/n-chicago-loop-traces.json', {with: {type: 'json'}})
      : import('./data/n-chicago-loop-traces.json'),
  'n-chicago-od-flows.json': () =>
    IS_NATIVE_NODE
      ? import('./data/n-chicago-od-flows.json', {with: {type: 'json'}})
      : import('./data/n-chicago-od-flows.json'),
  's1-chicago-tracts.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s1-chicago-tracts.json', {with: {type: 'json'}})
      : import('./data/s1-chicago-tracts.json'),
  's1-dixie-lattice.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s1-dixie-lattice.json', {with: {type: 'json'}})
      : import('./data/s1-dixie-lattice.json'),
  's2-chicago-north-areas.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-areas.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-areas.json'),
  's2-chicago-north-facilities.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-facilities.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-facilities.json'),
  's2-chicago-north-nature.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-nature.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-nature.json'),
  's2-chicago-north-parks.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-parks.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-parks.json'),
  's2-chicago-north-roads.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-roads.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-roads.json'),
  's2-chicago-north-tracts.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-chicago-north-tracts.json', {with: {type: 'json'}})
      : import('./data/s2-chicago-north-tracts.json'),
  's2-helene-rain.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-helene-rain.json', {with: {type: 'json'}})
      : import('./data/s2-helene-rain.json'),
  's2-midwest-counties.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-midwest-counties.json', {with: {type: 'json'}})
      : import('./data/s2-midwest-counties.json'),
  's2-midwest-county-shapes.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s2-midwest-county-shapes.json', {with: {type: 'json'}})
      : import('./data/s2-midwest-county-shapes.json'),
  's3-chicago-area-roads.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-chicago-area-roads.json', {with: {type: 'json'}})
      : import('./data/s3-chicago-area-roads.json'),
  's3-chicago-community-areas.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-chicago-community-areas.json', {with: {type: 'json'}})
      : import('./data/s3-chicago-community-areas.json'),
  's3-chicago-parks.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-chicago-parks.json', {with: {type: 'json'}})
      : import('./data/s3-chicago-parks.json'),
  's3-chicago-places.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-chicago-places.json', {with: {type: 'json'}})
      : import('./data/s3-chicago-places.json'),
  's3-chicago-roads.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-chicago-roads.json', {with: {type: 'json'}})
      : import('./data/s3-chicago-roads.json'),
  's3-ny-harbor-ais.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-ny-harbor-ais.json', {with: {type: 'json'}})
      : import('./data/s3-ny-harbor-ais.json'),
  's3-ny-harbor-zones.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-ny-harbor-zones.json', {with: {type: 'json'}})
      : import('./data/s3-ny-harbor-zones.json'),
  's3-openflights-ord.json': () =>
    IS_NATIVE_NODE
      ? import('./data/s3-openflights-ord.json', {with: {type: 'json'}})
      : import('./data/s3-openflights-ord.json'),
  't-dixie-bands.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-dixie-bands.json', {with: {type: 'json'}})
      : import('./data/t-dixie-bands.json'),
  't-dixie-dem.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-dixie-dem.json', {with: {type: 'json'}})
      : import('./data/t-dixie-dem.json'),
  't-dixie-landcover.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-dixie-landcover.json', {with: {type: 'json'}})
      : import('./data/t-dixie-landcover.json'),
  't-dixie-perimeter.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-dixie-perimeter.json', {with: {type: 'json'}})
      : import('./data/t-dixie-perimeter.json'),
  't-dixie-terrarium.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-dixie-terrarium.json', {with: {type: 'json'}})
      : import('./data/t-dixie-terrarium.json'),
  't-gfs-wind.json': () =>
    IS_NATIVE_NODE
      ? import('./data/t-gfs-wind.json', {with: {type: 'json'}})
      : import('./data/t-gfs-wind.json')
};
