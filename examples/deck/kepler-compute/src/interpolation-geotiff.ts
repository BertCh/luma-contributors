// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerAnalysisRecipe} from '@luma-contributors/kepler';

type TIFFEntry = {tag: number; type: number; count: number; data: Uint8Array};

export type InterpolationExportMetric = 'prediction' | 'variance';

export type InterpolationGeoTIFFOptions = {
  width: number;
  height: number;
  bounds: readonly [number, number, number, number];
  values: Float32Array;
  recipe: KeplerAnalysisRecipe;
  /** Required for kriging; omission preserves legacy IDW prediction exports. */
  metric?: InterpolationExportMetric;
};

/**
 * One bounded north-up EPSG:4326 float32 band, with PixelIsArea georeferencing and NaN nodata.
 * Values retain row-major order; this is a numeric raster, not the rendered color palette.
 * Uses classic TIFF, one uncompressed strip and no overviews, masks, reprojection or COG claims.
 * GeoTIFF 1.1: https://docs.ogc.org/is/19-008r4/19-008r4.pdf
 * GDAL nodata extension: https://gdal.org/en/stable/drivers/raster/gtiff.html#nodata-value
 */
export function encodeInterpolationGeoTIFF(
  options: InterpolationGeoTIFFOptions
): Uint8Array<ArrayBuffer> {
  const {width, height, bounds, values, recipe} = options;
  const metric = options.metric ?? 'prediction';
  const [west, south, east, north] = bounds;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > 256 ||
    height > 256 ||
    width * height > 16384 ||
    values.length !== width * height ||
    bounds.length !== 4 ||
    !bounds.every(Number.isFinite) ||
    west < -180 ||
    east > 180 ||
    south < -90 ||
    north > 90 ||
    west >= east ||
    south >= north
  )
    throw new Error('GeoTIFF requires a bounded geographic grid and one value per cell');
  if (
    (recipe.operation.kind !== 'idw-points' && recipe.operation.kind !== 'kriging-points') ||
    recipe.operation.grid.width !== width ||
    recipe.operation.grid.height !== height ||
    recipe.operation.grid.bounds.some((value, index) => value !== bounds[index])
  )
    throw new Error('GeoTIFF grid must match its interpolation recipe');
  if (
    (metric !== 'prediction' && metric !== 'variance') ||
    (recipe.operation.kind === 'kriging-points' && options.metric === undefined) ||
    (recipe.operation.kind === 'idw-points' && metric !== 'prediction')
  )
    throw new Error(
      'Choose a prediction or variance band for kriging; IDW exports prediction only'
    );
  const bandName =
    recipe.operation.kind === 'idw-points'
      ? 'IDW prediction'
      : metric === 'variance'
        ? 'Ordinary kriging variance'
        : 'Ordinary kriging prediction';
  // The response column may have no declared physical unit. Record its exponent without inventing one.
  const valueUnitsPower = metric === 'variance' ? 2 : 1;

  // TIFF ASCII is 7-bit. JSON escapes preserve arbitrary Unicode names without changing meaning.
  const description = JSON.stringify({
    format: 'luma.gl-kepler-interpolation',
    coordinateReferenceSystem: 'EPSG:4326',
    pixelInterpretation: 'area',
    nodata: 'NaN',
    metric,
    band: {index: 1, name: bandName, sourceColumn: recipe.operation.column, valueUnitsPower},
    recipe
  }).replace(
    /[\u007f-\uffff]/g,
    character => `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`
  );
  if (description.length > 32768)
    throw new Error('GeoTIFF recipe provenance exceeds its 32 KiB metadata limit');
  const entries: TIFFEntry[] = [
    numericEntry(256, 4, [width]), // ImageWidth
    numericEntry(257, 4, [height]), // ImageLength
    numericEntry(258, 3, [32]), // BitsPerSample
    numericEntry(259, 3, [1]), // Compression: none
    numericEntry(262, 3, [1]), // PhotometricInterpretation: BlackIsZero
    asciiEntry(270, description), // ImageDescription: provenance
    numericEntry(273, 4, [0]), // StripOffsets, filled after metadata layout
    numericEntry(274, 3, [1]), // Orientation: top-left
    numericEntry(277, 3, [1]), // SamplesPerPixel
    numericEntry(278, 4, [height]), // RowsPerStrip: one complete image
    numericEntry(279, 4, [values.length * 4]), // StripByteCounts
    {tag: 282, type: 5, count: 1, data: numericEntry(0, 4, [1, 1]).data}, // XResolution
    {tag: 283, type: 5, count: 1, data: numericEntry(0, 4, [1, 1]).data}, // YResolution
    numericEntry(284, 3, [1]), // PlanarConfiguration: chunky
    numericEntry(296, 3, [1]), // ResolutionUnit: none; georeferencing is in GeoTIFF tags
    numericEntry(339, 3, [3]), // SampleFormat: IEEE floating point
    numericEntry(33550, 12, [(east - west) / width, (north - south) / height, 0]),
    numericEntry(33922, 12, [0, 0, 0, west, north, 0]),
    numericEntry(34735, 3, [
      1,
      1,
      1,
      3, // GeoTIFF key directory version 1.1, three keys
      1024,
      0,
      1,
      2, // GTModelTypeGeoKey: geographic
      1025,
      0,
      1,
      1, // GTRasterTypeGeoKey: PixelIsArea
      2048,
      0,
      1,
      4326 // GeodeticCRSGeoKey: WGS 84
    ]),
    asciiEntry(
      42112,
      `<GDALMetadata><Item name="DESCRIPTION" sample="0" role="description">${bandName}</Item><Item name="INTERPOLATION_METRIC" sample="0">${metric}</Item><Item name="VALUE_UNITS_POWER" sample="0">${valueUnitsPower}</Item></GDALMetadata>`
    ), // GDAL_METADATA: explicit single-band identity, with source units defined by the recipe column
    asciiEntry(42113, 'nan') // GDAL_NODATA
  ].sort((first, second) => first.tag - second.tag);

  // The directory follows the eight-byte header. Eight-byte alignment also accommodates DOUBLE.
  let nextOffset = alignToEight(8 + 2 + entries.length * 12 + 4);
  const entryOffsets = entries.map(entry => {
    if (entry.data.byteLength <= 4) return 0;
    const offset = nextOffset;
    nextOffset = alignToEight(nextOffset + entry.data.byteLength);
    return offset;
  });
  const stripOffset = nextOffset;
  const output = new Uint8Array(stripOffset + values.length * 4);
  const view = new DataView(output.buffer);
  view.setUint16(0, 0x4949, true); // II, little endian
  view.setUint16(2, 42, true); // Classic TIFF
  view.setUint32(4, 8, true);
  view.setUint16(8, entries.length, true);
  for (const [index, entry] of entries.entries()) {
    const entryOffset = 10 + index * 12;
    view.setUint16(entryOffset, entry.tag, true);
    view.setUint16(entryOffset + 2, entry.type, true);
    view.setUint32(entryOffset + 4, entry.count, true);
    if (entry.tag === 273) view.setUint32(entryOffset + 8, stripOffset, true);
    else if (entry.data.byteLength <= 4) output.set(entry.data, entryOffset + 8);
    else {
      view.setUint32(entryOffset + 8, entryOffsets[index], true);
      output.set(entry.data, entryOffsets[index]);
    }
  }
  // The zero-initialized next-IFD pointer terminates this single-image file.
  for (const [index, value] of values.entries()) {
    const offset = stripOffset + index * 4;
    if (Number.isFinite(value)) view.setFloat32(offset, value, true);
    else view.setUint32(offset, 0x7fc00000, true); // Canonical quiet NaN, including infinities.
  }
  return output;
}

function alignToEight(value: number): number {
  return Math.ceil(value / 8) * 8;
}

function asciiEntry(tag: number, value: string): TIFFEntry {
  const data = new TextEncoder().encode(`${value}\0`);
  return {tag, type: 2, count: data.length, data};
}

function numericEntry(tag: number, type: 3 | 4 | 12, values: readonly number[]): TIFFEntry {
  const componentBytes = type === 3 ? 2 : type === 4 ? 4 : 8;
  const data = new Uint8Array(values.length * componentBytes);
  const view = new DataView(data.buffer);
  for (const [index, value] of values.entries()) {
    const offset = index * componentBytes;
    if (type === 3) view.setUint16(offset, value, true);
    else if (type === 4) view.setUint32(offset, value, true);
    else view.setFloat64(offset, value, true);
  }
  return {tag, type, count: values.length, data};
}
