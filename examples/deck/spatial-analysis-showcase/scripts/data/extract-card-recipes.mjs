// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extract display-sized fixtures from the checked-in, versioned showcase archives.
import {readFileSync, writeFileSync} from 'node:fs';
import earcut from 'earcut';

const source = new URL('../../public/data/', import.meta.url);
const output = new URL('../../../../../website/src/gpu/data/recipes/', import.meta.url);
const read = (dataset, name) => readFileSync(new URL(dataset + '/' + name, source));
const write = (name, value) =>
  writeFileSync(new URL(name + '.json', output), JSON.stringify(value) + '\n');
const round = (value, digits) => Number(value.toFixed(digits));
function floats(dataset, file) {
  const buffer = read(dataset, file);
  return Array.from({length: buffer.length / 4}, (_, index) => buffer.readFloatLE(index * 4));
}
function integers(dataset, file) {
  const buffer = read(dataset, file);
  return Array.from({length: buffer.length / 4}, (_, index) => buffer.readUInt32LE(index * 4));
}

const support = 'chicago-population-support';
const native = read(support, 'worldcover-native.bin');
const width = 96;
const height = 72;
const cover = Buffer.alloc(width * height);
for (let row = 0; row < height; row++) {
  for (let column = 0; column < width; column++) {
    const counts = new Map();
    for (let y = Math.floor((row * 432) / height); y < Math.floor(((row + 1) * 432) / height); y++) {
      for (
        let x = Math.floor((column * 578) / width);
        x < Math.floor(((column + 1) * 578) / width);
        x++
      ) {
        const code = native[y * 578 + x];
        counts.set(code, (counts.get(code) ?? 0) + 1);
      }
    }
    cover[row * width + column] = [...counts].sort((first, second) => second[1] - first[1])[0][0];
  }
}
const fractions = floats(support, 'support-fractions.bin').slice(0, 64 * 64);
write('hyde-park-cover', {
  width,
  height,
  classes: cover.toString('base64'),
  builtFraction: Buffer.from(fractions.map((value) => Math.round(value * 255))).toString('base64')
});

const population = floats(support, 'population.bin');
const focal = floats(support, 'nhBlack.bin');
const tracts = JSON.parse(read(support, 'tracts.geojson')).features.map((feature, index) => {
  const ring = feature.geometry.coordinates[0].slice(0, -1);
  return {
    geoid: feature.properties.GEOID,
    ring,
    triangles: earcut(ring.flat()),
    population: population[index],
    focal: focal[index]
  };
});
write('hyde-park-tracts', tracts);

const ndvi = read('ndvi-timeseries', 'ndvi.bin');
const dates = JSON.parse(read('ndvi-timeseries', 'manifest.json')).properties.dates;
const slices = [7, 9, 12, 15].map((slice) => {
  // Centered 256 x 144 window. Average valid observations over 4 x 4 source pixels.
  const pixels = Buffer.alloc(64 * 36);
  for (let y = 0; y < 36; y++) {
    for (let x = 0; x < 64; x++) {
      let total = 0;
      let count = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const value = ndvi[slice * 256 * 256 + (56 + y * 4 + dy) * 256 + x * 4 + dx];
          if (value > 0) {
            total += value;
            count++;
          }
        }
      }
      pixels[y * 64 + x] = count ? Math.round(total / count) : 0;
    }
  }
  return {date: dates[slice], values: pixels.toString('base64')};
});
write('dixie-ndvi', {width: 64, height: 36, slices});

const earthquakes = 'poopdeck-earthquakes';
const positions = floats(earthquakes, 'position.bin');
const times = integers(earthquakes, 'time.bin');
const magnitudes = floats(earthquakes, 'magnitude.bin');
const events = [];
for (let index = 0; index < times.length; index++) {
  const longitude = positions[index * 2],
    latitude = positions[index * 2 + 1];
  const time = 1577838500 + times[index];
  if (
    time >= Date.UTC(2023, 0, 1) / 1000 &&
    time < Date.UTC(2024, 0, 1) / 1000 &&
    longitude >= 34 &&
    longitude < 44 &&
    latitude >= 35 &&
    latitude < 41
  ) {
    events.push([round(longitude, 3), round(latitude, 3), time, round(magnitudes[index], 1)]);
  }
}
write('turkiye-quakes', events);
