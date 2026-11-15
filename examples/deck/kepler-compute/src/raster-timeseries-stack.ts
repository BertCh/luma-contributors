// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterTimeStack, type RasterTimeStack} from '@luma-contributors/kepler';
import {
  createNumericRasterSource,
  createNumericRasterTile,
  type NumericRasterSource,
  type NumericRasterTile
} from './raster-tile-ingest';

export type RasterTimeSlice = {time: string; layerId?: string; bands: readonly number[]};
export type BoundRasterTimeStack = {
  stack: RasterTimeStack;
  template: NumericRasterSource;
  sources: readonly (NumericRasterSource | undefined)[];
};

/** Strict local calendar validation before Date.parse applies the explicit timezone offset. */
function parseRasterAcquisitionTime(text: string): number {
  const match =
    typeof text === 'string' &&
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$/.exec(text);
  if (!match) throw new Error('Acquisition dates need ISO timestamps with an explicit timezone');
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, timezone] = match;
  const [year, month, day, hour, minute, second] = [
    yearText,
    monthText,
    dayText,
    hourText,
    minuteText,
    secondText
  ].map(Number);
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const monthLengths = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > monthLengths[month - 1] ||
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    (timezone !== 'Z' && (Number(timezone.slice(1, 3)) > 23 || Number(timezone.slice(4)) > 59))
  )
    throw new Error('Acquisition date has invalid calendar or time components');
  return Date.parse(text);
}

function getGridKey(tile: NumericRasterTile): string {
  return JSON.stringify([tile.width, tile.height, tile.zoom, tile.extent]);
}

/** Explicit acquisition/band mapping. Tile IDs may differ across acquisitions; grid geometry may not. */
export function bindRasterTimeStack(
  defaultLayerId: string,
  slices: readonly RasterTimeSlice[],
  getSource: (layerId: string) => NumericRasterSource | undefined
): BoundRasterTimeStack {
  if (!Array.isArray(slices) || slices.length < 2 || slices.length > 256)
    throw new Error('Time stack needs 2–256 acquisitions');
  const sources = slices.map(slice => getSource(slice.layerId || defaultLayerId));
  const template = sources[0];
  if (!template?.tiles.length) throw new Error('No loaded numeric raster tiles');
  if (template.tiles.length > 256) throw new Error('Time stack supports at most 256 loaded tiles');
  if (new Set(template.tiles.map(tile => tile.zoom)).size > 1)
    throw new Error('Time stack needs a single loaded zoom level');
  const bandCount = slices[0].bands?.length;
  if (!Number.isInteger(bandCount) || bandCount < 1 || bandCount > 16)
    throw new Error('Each date needs 1–16 band indices');
  const keys = template.tiles.map(getGridKey);
  if (new Set(keys).size !== keys.length) throw new Error('Duplicate time-stack tile grids');
  for (let first = 0; first < template.tiles.length; first++)
    for (let second = first + 1; second < template.tiles.length; second++) {
      const left = template.tiles[first].extent;
      const right = template.tiles[second].extent;
      if (
        Math.min(left[2], right[2]) > Math.max(left[0], right[0]) &&
        Math.min(left[3], right[3]) > Math.max(left[1], right[1])
      )
        throw new Error('Time stack refuses overlapping tiles or mixed zoom coverage');
    }
  const cellCount = template.tiles.reduce((sum, tile) => sum + tile.width * tile.height, 0);
  const sliceCount = slices.length;
  if (cellCount > 65536 || cellCount * sliceCount * bandCount > 1048576)
    throw new Error('Time stack exceeds 65536 cells or 1048576 values; load a smaller window');
  const values = new Float32Array(cellCount * sliceCount * bandCount);
  slices.forEach((slice, sliceIndex) => {
    const source = sources[sliceIndex];
    if (!source || source.crs !== template.crs || source.tiles.length !== template.tiles.length)
      throw new Error(`Acquisition ${sliceIndex}: CRS/tile coverage differs`);
    if (
      !Array.isArray(slice.bands) ||
      slice.bands.length !== bandCount ||
      slice.bands.some((index: number) => !Number.isInteger(index) || index < 1)
    )
      throw new Error(
        `Acquisition ${sliceIndex}: band indices must be positive integers with the same count on each date`
      );
    const tiles = new Map(source.tiles.map(tile => [getGridKey(tile), tile]));
    if (tiles.size !== keys.length) throw new Error(`Acquisition ${sliceIndex}: duplicate grids`);
    let cellOffset = 0;
    template.tiles.forEach((reference, tileIndex) => {
      const tile = tiles.get(keys[tileIndex]);
      if (!tile)
        throw new Error(
          `Acquisition ${sliceIndex}: grid dimensions/extent/zoom differ; reprojection is not implicit`
        );
      const bands = slice.bands.map((index: number) => {
        const band = tile.bands.find(candidate => candidate.index === index);
        if (!band) throw new Error(`Acquisition ${sliceIndex}: missing band ${index}`);
        return band;
      });
      for (let cell = 0; cell < reference.width * reference.height; cell++)
        for (let band = 0; band < bandCount; band++) {
          const value = Math.fround(bands[band].values[cell]);
          values[((cellOffset + cell) * sliceCount + sliceIndex) * bandCount + band] =
            bands[band].validity[cell] && Number.isFinite(value) ? value : NaN;
        }
      cellOffset += reference.width * reference.height;
    });
  });
  const times = slices.map(slice => parseRasterAcquisitionTime(slice.time));
  const stack = {cellCount, sliceCount, bandCount, values, times};
  validateRasterTimeStack(stack);
  return {stack, template, sources};
}

/** Restores tile geometry and nodata for one selected cell-aligned output component. */
export function createTimeSeriesOutputSource(
  template: NumericRasterSource,
  values: ArrayLike<number>,
  components: number,
  component: number,
  name: string
): NumericRasterSource {
  if (!Number.isInteger(component) || component < 0 || component >= components)
    throw new Error(`Output component must be 0–${components - 1}`);
  const cellCount = template.tiles.reduce((sum, tile) => sum + tile.width * tile.height, 0);
  if (values.length !== cellCount * components)
    throw new Error('Time-series output length does not match the raster grid');
  let offset = 0;
  return createNumericRasterSource({
    id: `${template.id}:${name}`,
    kind: template.kind,
    crs: template.crs,
    tiles: template.tiles.map(tile => {
      const selected = Float32Array.from({length: tile.width * tile.height}, (_, cell) => {
        const value = values[(offset + cell) * components + component];
        return value === 0xffffffff || !Number.isFinite(value) ? NaN : value;
      });
      offset += selected.length;
      return createNumericRasterTile({
        ...tile,
        id: `${tile.id}:${name}`,
        bands: [{index: 1, name, values: selected}]
      });
    })
  });
}
