// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFile, writeFile} from 'node:fs/promises';

const root = new URL('../public/data/dixie-spectral/', import.meta.url);
const output = new URL(
  '../../../../dev-docs/research/showcase-overhaul-2026-10-25/earth-spectral-regions-oracle.json',
  import.meta.url
);
const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));
const sourceWidth = manifest.rasters.nir_before.width;
const crop = {row: 50, column: 101, width: 64, height: 64};
const bandNames = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22'];
const dates = {};
for (const date of ['before', 'after']) {
  const bands = {};
  for (const band of bandNames) {
    const bytes = await readFile(new URL(`${band}_${date}.bin`, root));
    bands[band] = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
  const validity = new Uint8Array(await readFile(new URL(`validity_${date}.bin`, root)));
  const cells = [];
  for (let row = 0; row < crop.height; row++) {
    for (let column = 0; column < crop.width; column++) {
      const cell = row * crop.width + column;
      const sourceCell = (crop.row + row) * sourceWidth + crop.column + column;
      const valid = validity[sourceCell] !== 0 && bandNames.every(name => Number.isFinite(bands[name][sourceCell]));
      cells.push({
        cell,
        row,
        column,
        valid,
        nir: valid ? bands.nir[sourceCell] : Number.NaN
      });
    }
  }
  dates[date] = {bands, validity, cells};
}

const zones = [];
const probabilities = [0, 0.25, 0.5, 0.75, 1];
const cellAreaM2 = manifest.rasters.nir_before.cellSizeM ** 2;
for (let region = 0; region < 4; region++) {
  const rowStart = region >= 2 ? 32 : 0;
  const columnStart = region % 2 ? 32 : 0;
  const cells = [];
  for (let row = rowStart; row < rowStart + 32; row++) {
    for (let column = columnStart; column < columnStart + 32; column++) {
      cells.push(row * crop.width + column);
    }
  }
  const summarize = date => {
    const values = cells.map(cell => dates[date].cells[cell]).filter(sample => sample.valid).map(sample => sample.nir).sort((a, b) => a - b);
    const quantiles = probabilities.map(probability => {
      const position = Math.fround(Math.fround(values.length - 1) * Math.fround(probability));
      const lowIndex = Math.floor(position);
      const low = values[lowIndex];
      if (lowIndex + 1 >= values.length) return low;
      const high = values[lowIndex + 1];
      return Math.fround(low + Math.fround((high - low) * Math.fround(position - lowIndex)));
    });
    const frequencies = new Map();
    for (const value of values) frequencies.set(value, (frequencies.get(value) ?? 0) + 1);
    let mode = Number.NaN;
    let maximumFrequency = -1;
    for (const [value, count] of frequencies) {
      if (count > maximumFrequency || (count === maximumFrequency && value < mode)) {
        mode = value;
        maximumFrequency = count;
      }
    }
    return {
      assignedCells: cells.length,
      validRows: values.length,
      validFiniteValues: values.length,
      maskedOutRows: cells.length - values.length,
      uniqueValues: frequencies.size,
      sum: values.reduce((total, value) => total + value, 0),
      mean: values.reduce((total, value) => total + value, 0) / values.length,
      median: quantiles[2],
      mode,
      modeFrequency: maximumFrequency,
      quantiles
    };
  };
  const before = summarize('before');
  const after = summarize('after');
  const common = cells.map(cell => [dates.before.cells[cell], dates.after.cells[cell]]).filter(pair => pair[0].valid && pair[1].valid);
  const deltas = common.map(pair => Math.fround(pair[1].nir - pair[0].nir));
  const meanDelta = deltas.reduce((sum, value) => sum + value, 0) / deltas.length;
  const thresholds = {};
  for (const threshold of [0.02, 0.05, 0.1, 0.2]) {
    const changed = deltas.filter(delta => Math.abs(delta) >= threshold).length;
    thresholds[String(threshold)] = {
      changedCells: changed,
      changedAreaM2: changed * cellAreaM2,
      changedAreaKm2: changed * cellAreaM2 / 1_000_000
    };
  }
  zones.push({
    id: region,
    label: ['north-west', 'north-east', 'south-west', 'south-east'][region],
    rowRangeInclusive: [rowStart, rowStart + 31],
    columnRangeInclusive: [columnStart, columnStart + 31],
    totalAssignedCells: cells.length,
    totalAreaM2: cells.length * cellAreaM2,
    totalAreaKm2: cells.length * cellAreaM2 / 1_000_000,
    before,
    after,
    paired: {
      validCells: common.length,
      validAreaM2: common.length * cellAreaM2,
      validAreaKm2: common.length * cellAreaM2 / 1_000_000,
      coverage: common.length / cells.length,
      meanDifference: meanDelta,
      thresholds
    }
  });
}

const sources = ['validity_before.bin', 'validity_after.bin', 'nir_before.bin', 'nir_after.bin'];
const checksums = {};
for (const name of sources) checksums[name] = createHash('sha256').update(await readFile(new URL(name, root))).digest('hex');
const report = {
  dataset: 'dixie-spectral',
  acquisitionDates: {before: '2021-07-13', after: '2021-09-21'},
  crop: {
    ...crop,
    sourceWidth,
    order: 'row-major, north-first',
    exactProjectedBoundsMeters: [662800 + crop.column * 60, 4456040 - (crop.row + crop.height) * 60, 662800 + (crop.column + crop.width) * 60, 4456040 - crop.row * 60],
    sourceResolutionMeters: 60,
    note: 'Reflectance means from native 10/20 m Sentinel-2 samples on a 60 m UTM grid; not native 60 m measurements.'
  },
  validity: {
    rule: 'per-date source validity byte is nonzero and all six calibrated reflectance bands are finite, matching createSpectralEvidence',
    beforeValid: dates.before.cells.filter(cell => cell.valid).length,
    afterValid: dates.after.cells.filter(cell => cell.valid).length,
    commonValid: dates.before.cells.filter((cell, index) => cell.valid && dates.after.cells[index].valid).length
  },
  zones: {
    definition: 'Four dense uint32 whole-cell quadrants; IDs 0 NW, 1 NE, 2 SW, 3 SE; each exactly 32×32 cells.',
    areaDefinition: `${cellAreaM2} m² per projected UTM cell; sums are planar, not geodesic.`,
    probabilities,
    thresholdDefinition: 'changed when absolute September-minus-July reflectance difference is greater than or equal to the selected threshold; paired common support only.',
    expectedStatus: {
      stages: [
        {stage: 'total-area', overflow: 0},
        {stage: 'valid-area', overflow: 0},
        {stage: 'changed-area', overflow: 0},
        {stage: 'change-summary', overflow: 0},
        {stage: 'change-threshold', invalidCount: 0}
      ],
      reasoning: 'There are four dense zone IDs with four-row output capacity; the selected cutoff is finite and nonnegative.'
    },
    note: 'GPU region recipe passes the strict validity mask into GPUGroupStatistics. Therefore output rowCounts count retained valid-mask rows, not every assigned cell; current readout wording should distinguish assigned cells, valid rows and masked-out rows. CPU row statistics here include only valid six-band cells, as the GPU should.',
    checksums,
    values: zones
  }
};
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
