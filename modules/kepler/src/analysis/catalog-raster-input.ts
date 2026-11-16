// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  type GPURasterCellSizeMode
} from '@luma-contributors/analysis/analysis-core';

/** Decoded numeric samples, structurally compatible with the host's NumericRasterBand. */
export type CatalogRasterBand = Readonly<{
  index: number;
  name?: string;
  values:
    | Float32Array
    | Float64Array
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array;
  noDataValue: number | null;
  /** Authoritative validity, preserved from ingestion. Nonzero means valid. */
  validity: Uint8Array | Uint32Array;
}>;

/** One complete bounded grid and one decoded band; no palette, decoder or point conversion. */
export type CatalogRasterInput = Readonly<{
  type: 'raster';
  id: string;
  grid: GPURasterMetadata;
  band: CatalogRasterBand;
  /** Explicit CRS declaration; `planar` requires caller-confirmed consistent linear units. */
  crs: 'EPSG:4326' | 'EPSG:3857' | 'planar';
  cellSizeMode: GPURasterCellSizeMode;
}>;

/** Checks shape and metadata once at the input boundary, without scanning decoded cells. */
export function validateCatalogRasterInput(
  id: string,
  input: CatalogRasterInput,
  maximumCells: number
): void {
  validateCapacity(id, 'maximumCells', maximumCells);
  const {grid, band} = input;
  validateCapacity(id, 'grid.width', grid.width);
  validateCapacity(id, 'grid.height', grid.height);
  const cellCount = grid.width * grid.height;
  validateMaximumCount(id, 'grid cell count', cellCount, maximumCells);
  validateRequirement(
    id,
    grid.affine.length === 6 && grid.affine.every(Number.isFinite),
    'grid.affine must contain six finite numbers'
  );
  validateRequirement(
    id,
    grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3] !== 0,
    'grid.affine must be invertible'
  );
  validateRequirement(
    id,
    grid.pixelInterpretation === 'area' || grid.pixelInterpretation === 'point',
    'grid.pixelInterpretation must be area or point'
  );
  validateRequirement(
    id,
    band.values.length === cellCount,
    'band.values length must equal grid cell count'
  );
  validateRequirement(
    id,
    band.validity.length === cellCount,
    'band.validity length must equal grid cell count'
  );
  validateCapacity(id, 'band.index', band.index);
  validateRequirement(
    id,
    band.noDataValue === null || Number.isFinite(band.noDataValue),
    'band.noDataValue must be null or finite'
  );
  const expectedMode =
    input.crs === 'EPSG:4326'
      ? 'geographic'
      : input.crs === 'EPSG:3857'
        ? 'web-mercator'
        : input.crs === 'planar'
          ? 'uniform'
          : undefined;
  validateRequirement(
    id,
    expectedMode !== undefined && input.cellSizeMode === expectedMode,
    'crs and cellSizeMode must agree (EPSG:4326/geographic, EPSG:3857/web-mercator, planar/uniform)'
  );
  validateRequirement(
    id,
    input.cellSizeMode === 'uniform' || (grid.affine[1] === 0 && grid.affine[3] === 0),
    'geographic and web-mercator grids must be axis aligned'
  );
}

/** Wraps caller-owned decoded arrays. The caller must keep them immutable after hand-off. */
export function makeCatalogRasterInput(
  input: Omit<CatalogRasterInput, 'type'>,
  maximumCells: number
): CatalogRasterInput {
  const result: CatalogRasterInput = {type: 'raster', ...input};
  validateCatalogRasterInput(input.id, result, maximumCells);
  return result;
}

/** Exact grid alignment, required before recipes compare cells by row index. */
export function areCatalogRasterGridsAligned(
  first: CatalogRasterInput,
  second: CatalogRasterInput
): boolean {
  return (
    first.crs === second.crs &&
    first.cellSizeMode === second.cellSizeMode &&
    first.grid.width === second.grid.width &&
    first.grid.height === second.grid.height &&
    first.grid.pixelInterpretation === second.grid.pixelInterpretation &&
    first.grid.affine.every((value, index) => value === second.grid.affine[index])
  );
}
