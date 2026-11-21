// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {
  GPURasterBufferBand,
  GPURasterMetadata,
  GPURasterScalarFormat
} from '@luma.gl/graph/gpu-raster';

/** Options for {@link makeGPURasterMetadata}. */
export type GPURasterMetadataOptions = {
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /**
   * Cell size in world units, `[x, y]` or one value for square cells. Defaults to 1. Meters for
   * `cellSizeMode: 'uniform'`, EPSG:3857 meters for `'web-mercator'`, degrees for `'geographic'`.
   */
  cellSize?: number | readonly [number, number];
  /** World `[x, y]` of the top-left corner of pixel `(0, 0)`. Defaults to `[0, 0]`. */
  origin?: readonly [number, number];
  /**
   * Direction in which rows advance in world y. Defaults to `'south'`: row 0 is the top row and
   * world y decreases down the grid (a north-up raster).
   */
  rowDirection?: 'south' | 'north';
};

/**
 * Returns the metadata of an axis-aligned area grid: `affine = [cellX, 0, originX, 0, -cellY,
 * originY]` (`+cellY` when rows advance north).
 *
 * Every raster and terrain analysis contributor takes its shape from `grid`, and every parameter
 * packer that needs a cell size or a latitude band reads them from the same object.
 */
export function makeGPURasterMetadata(options: GPURasterMetadataOptions): GPURasterMetadata {
  const cellSize = options.cellSize ?? 1;
  const [cellX, cellY] = typeof cellSize === 'number' ? [cellSize, cellSize] : cellSize;
  const [originX, originY] = options.origin ?? [0, 0];
  const rowStep = (options.rowDirection ?? 'south') === 'south' ? -cellY : cellY;
  return {
    width: options.width,
    height: options.height,
    affine: [cellX, 0, originX, 0, rowStep, originY],
    pixelInterpretation: 'area'
  };
}

/** Optional band metadata for {@link makeGPURasterBand}. */
export type GPURasterBandOptions = {
  /** Band identifier. Defaults to `'band'`. */
  id?: string;
  /** Optional validity flags, one per pixel: zero rejects a sample. */
  validity?: GraphDataView<'uint32'>;
  /** Raw nodata value, compared before calibration. */
  noDataValue?: number;
  /** Calibration multiplier. Defaults to one. */
  scale?: number;
  /** Calibration offset. Defaults to zero. */
  offset?: number;
  /**
   * Vertical datum of height values, such as `'ellipsoidal'` or `'EPSG:5773'` (EGM96).
   * Descriptive only: operators compare declared datums and never convert. Absent means
   * undeclared; derived bands do not inherit it.
   */
  verticalDatum?: string;
};

/**
 * Wraps a packed per-pixel view, typically the output of one contributor, as a buffer raster band
 * on `grid`, so it can feed the next contributor without a hand-written band literal. For
 * `GPUTerrainHeightAboveDrainage`, wrap `GPUTerrainFlow`'s `filledElevation` as the `elevation`
 * band and pass its `flowDirections` view directly as `flowDirections`.
 *
 * Borrows the grid-sized prefix when values or validity have spare capacity. No data is copied;
 * byte offsets and band metadata are preserved. Exact-sized views retain their identity.
 *
 * Values must be packed float32, sint32 or uint32; validity must be packed uint32.
 * @throws If a format is unsupported, either packed view is shorter than the grid, or graphs differ.
 */
export function makeGPURasterBand<Format extends GPURasterScalarFormat>(
  values: GraphDataView<Format>,
  grid: GPURasterMetadata,
  options: GPURasterBandOptions = {}
): GPURasterBufferBand<Format> {
  const pixelCount = grid.width * grid.height;
  const id = options.id ?? 'band';
  const graph = values.buffer.graph as GPUCommandGraph;
  const borrowPrefix = <ViewFormat extends GPURasterScalarFormat>(
    view: GraphDataView<ViewFormat>,
    name: string,
    expectedFormats: readonly GPURasterScalarFormat[]
  ) => {
    validateSinglePackedView(view, expectedFormats, {id, contributor: id}, name);
    validateRequirement(
      id,
      view.length >= pixelCount,
      `${name} must hold at least one value per grid pixel`
    );
    validateRequirement(
      id,
      view.buffer.graph === graph,
      `${name} must belong to the same graph as values`
    );
    return view.length === pixelCount
      ? view
      : graph.createDataView(view.buffer, {
          format: view.format,
          byteOffset: view.byteOffset,
          length: pixelCount
        });
  };
  const boundedValues = borrowPrefix(values, 'values', ['float32', 'sint32', 'uint32']);
  const validity = options.validity
    ? borrowPrefix(options.validity, 'validity', ['uint32'])
    : undefined;
  return {
    id: options.id ?? 'band',
    format: values.format,
    storage: {kind: 'buffer', values: boundedValues},
    ...(validity ? {validity} : {}),
    ...(options.noDataValue !== undefined ? {noDataValue: options.noDataValue} : {}),
    ...(options.scale !== undefined ? {scale: options.scale} : {}),
    ...(options.offset !== undefined ? {offset: options.offset} : {}),
    ...(options.verticalDatum !== undefined ? {verticalDatum: options.verticalDatum} : {})
  } as GPURasterBufferBand<Format>;
}
