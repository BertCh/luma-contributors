// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBand,
  validateRasterBandBelongsToGraph,
  validateRasterGrid
} from '../raster-band-utils';
import {validateRasterAlgebraView} from './raster-algebra-utils';

/** Calibration and float no-data are emitted as finite float32 shader literals. */
const MAXIMUM_RASTER_ALGEBRA_FLOAT32_VALUE = 3.4028234663852886e38;
/** Integer no-data must retain the exact native sample representation. */
const MAXIMUM_RASTER_ALGEBRA_UINT32_VALUE = 0xffffffff;
const MINIMUM_RASTER_ALGEBRA_SINT32_VALUE = -0x80000000;
const MAXIMUM_RASTER_ALGEBRA_SINT32_VALUE = 0x7fffffff;
/** Published raster conversion reads one 2D mip, one array layer, and one sample. */
const RASTER_ALGEBRA_TEXTURE_LAYOUT_COUNT = 1;
/** RGBA raster textures have four zero-indexed channels. */
const MAXIMUM_RASTER_ALGEBRA_RGBA_CHANNEL = 3;

/** Accepts a legacy float column or a published raster band. @internal */
export type RasterAlgebraInput = GraphDataView<'float32'> | GPURasterBand;

/** Narrows a published raster band without inspecting any cell contents. @internal */
export function isRasterAlgebraBand(input: RasterAlgebraInput): input is GPURasterBand {
  return 'storage' in input;
}

/** Returns the original input views for aliasing and graph-ownership checks. @internal */
export function getRasterAlgebraInputViews(input: RasterAlgebraInput | undefined): GraphDataView[] {
  return input ? (isRasterAlgebraBand(input) ? getRasterBandViews(input) : [input]) : [];
}

/** Validates a column or band at construction and returns its logical row count. @internal */
export function validateRasterAlgebraInput(
  id: string,
  name: string,
  input: RasterAlgebraInput,
  grid: GPURasterMetadata | undefined,
  rowCount?: number
): number {
  validateRequirement(id, Boolean(input), `${name} must be a float32 view or raster band`);
  if (!isRasterAlgebraBand(input)) {
    validateRasterAlgebraView(id, name, input, 'float32', rowCount ?? 0);
    return input.length;
  }
  validateRequirement(id, Boolean(grid), `${name} requires grid for a raster band`);
  const cellCount = validateRasterGrid(id, grid!);
  validateRequirement(
    id,
    rowCount === undefined || rowCount <= cellCount,
    `${name} grid must cover rowCount ${rowCount}`
  );
  for (const [property, value] of [
    ['scale', input.scale ?? 1],
    ['offset', input.offset ?? 0]
  ] as const) {
    validateFiniteNumber(
      id,
      `${name}.${property}`,
      value,
      -MAXIMUM_RASTER_ALGEBRA_FLOAT32_VALUE,
      MAXIMUM_RASTER_ALGEBRA_FLOAT32_VALUE
    );
  }
  if (input.noDataValue !== undefined) {
    if (input.format === 'float32') {
      if (!Number.isNaN(input.noDataValue)) {
        validateFiniteNumber(
          id,
          `${name}.noDataValue`,
          input.noDataValue,
          -MAXIMUM_RASTER_ALGEBRA_FLOAT32_VALUE,
          MAXIMUM_RASTER_ALGEBRA_FLOAT32_VALUE
        );
      }
    } else {
      validateCapacity(
        id,
        `${name}.noDataValue`,
        input.noDataValue,
        input.format === 'uint32' ? 0 : MINIMUM_RASTER_ALGEBRA_SINT32_VALUE,
        input.format === 'uint32'
          ? MAXIMUM_RASTER_ALGEBRA_UINT32_VALUE
          : MAXIMUM_RASTER_ALGEBRA_SINT32_VALUE
      );
    }
  }
  validateRasterBand(input, grid!, `${id} ${name}`);
  if (input.storage.kind === 'buffer') {
    validateSinglePackedView(input.storage.values, [input.format], `${id} ${name}.storage.values`);
  } else {
    const {view, channel = 0} = input.storage;
    validateRequirement(id, view.dimension === '2d', `${name}.storage.view.dimension must be 2d`);
    for (const [property, value] of [
      ['mipLevelCount', view.mipLevelCount],
      ['arrayLayerCount', view.arrayLayerCount],
      ['texture.samples', view.texture.samples]
    ] as const) {
      validateCapacity(
        id,
        `${name}.storage.view.${property}`,
        value,
        RASTER_ALGEBRA_TEXTURE_LAYOUT_COUNT,
        RASTER_ALGEBRA_TEXTURE_LAYOUT_COUNT
      );
    }
    const textureSuffix = {float32: 'float', uint32: 'uint', sint32: 'sint'}[input.format];
    const textureFormats = [`r32${textureSuffix}`, `rgba32${textureSuffix}`];
    validateRequirement(
      id,
      textureFormats.includes(view.format),
      `${name}.storage.view.format must be ${textureFormats.join(' or ')} for ${name}.format ${input.format}`
    );
    validateCapacity(
      id,
      `${name}.storage.channel`,
      channel,
      0,
      view.format.startsWith('rgba') ? MAXIMUM_RASTER_ALGEBRA_RGBA_CHANNEL : 0
    );
  }
  validateRasterAlgebraView(id, `${name}.validity`, input.validity, 'uint32', cellCount);
  const owner =
    input.storage.kind === 'buffer'
      ? input.storage.values.buffer.graph
      : input.storage.view.texture.graph;
  validateRequirement(
    id,
    !input.validity || input.validity.buffer.graph === owner,
    `${name}.validity must belong to the same graph as ${name}.storage`
  );
  return cellCount;
}

/**
 * Resolves band validity, native nodata and calibration through published raster primitives.
 * Invalid samples become NaN, so downstream algebra needs no extra storage binding. Legacy
 * columns pass through unchanged (including their documented infinity handling).
 * @internal
 */
export function getRasterAlgebraInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: RasterAlgebraInput,
  grid: GPURasterMetadata | undefined
) {
  if (!isRasterAlgebraBand(input)) {
    return {values: input, nodes: []};
  }
  validateRasterBandBelongsToGraph(id, graph, input, []);
  const cellCount = grid!.width * grid!.height;
  // Canonicalization always writes full-grid float32 values and uint32 validity, even when
  // the consumer requests fewer rows. All operands share the grid, so the first converted
  // band checks both equal-sized scratch bindings before any conversion reserves resources.
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'band canonicalization',
    cellCount * Float32Array.BYTES_PER_ELEMENT,
    'reduce grid dimensions'
  );
  // Published raster conversions require exact logical lengths. Borrow the grid prefix of
  // padded caller views while retaining their byte offsets and the original band metadata.
  const validity = input.validity
    ? graph.createDataView(input.validity.buffer, {
        format: 'uint32',
        byteOffset: input.validity.byteOffset,
        length: cellCount
      })
    : undefined;
  let boundedInput: GPURasterBand = {...input, validity};
  if (input.storage.kind === 'buffer') {
    const {values} = input.storage;
    boundedInput = {
      ...boundedInput,
      storage: {
        kind: 'buffer',
        values: graph.createDataView(values.buffer, {
          format: values.format,
          byteOffset: values.byteOffset,
          length: cellCount
        })
      }
    } as GPURasterBand;
  }
  const source = getRasterBufferBandNodes(graph, id, boundedInput, grid!, true);
  return {values: source.band.storage.values as GraphDataView<'float32'>, nodes: source.nodes};
}
