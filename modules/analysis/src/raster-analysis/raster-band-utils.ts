// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import {GPURaster, GPURasterNeighborhood} from '@luma.gl/graph/gpu-raster';
import {GPURasterTextureToBuffer} from '@luma.gl/graph/gpu-raster';
import type {
  GPURasterBand,
  GPURasterBufferBand,
  GPURasterMetadata,
  GPURasterTextureBand
} from '@luma.gl/graph/gpu-raster';
import type {CompiledProjection, ProjectionShader} from '@luma.gl/graph/gpu-project';
import {
  captureGraphCommandNodes,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateGraphViewsBelongToGraph
} from '../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {snapshotConstructionFields} from '../analysis-core/kernels/construction-snapshot';
import type {GPURasterCellSizeMode} from '../analysis-core/contracts/raster-grid';
import {validateRasterWritableDispatches} from './raster-grid/raster-view-validation';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';

/** Native nearest-label inputs must retain identity calibration. @internal */
export function validateRasterLabelBand(
  id: string,
  band: GPURasterBand,
  grid: GPURasterMetadata
): void {
  validateRequirement(
    id,
    band.format === 'uint32' || band.format === 'sint32',
    `input.format must be uint32 or sint32 for output.labels (got ${band.format})`
  );
  validateRequirement(
    id,
    (band.scale ?? 1) === 1 && (band.offset ?? 0) === 0,
    `input scale/offset must be identity (1/0) for output.labels (got ${band.scale ?? 1}/${band.offset ?? 0})`
  );
  if (band.noDataValue !== undefined) {
    validateRequirement(
      id,
      Number.isInteger(band.noDataValue) &&
        band.noDataValue >= (band.format === 'uint32' ? 0 : -2147483648) &&
        band.noDataValue <= (band.format === 'uint32' ? 4294967295 : 2147483647),
      `input.noDataValue must be an exact ${band.format} label (got ${band.noDataValue})`
    );
  }
  new GPURaster({id: `${id} input`, metadata: grid, bands: [band]});
}

/** Gathers native integers and evaluates native nodata before any float conversion. @internal */
export function getRasterLabelBandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  band: GPURasterBand,
  grid: GPURasterMetadata
): RasterBufferBandSource<Parameters> {
  const format = band.format === 'uint32' ? 'uint32' : 'sint32';
  const type: 'u32' | 'i32' = format === 'uint32' ? 'u32' : 'i32';
  const count = grid.width * grid.height;
  const noDataLiteral =
    band.noDataValue === -2147483648 && format === 'sint32'
      ? 'bitcast<i32>(0x80000000u)'
      : `${band.noDataValue}${format === 'uint32' ? 'u' : 'i'}`;
  const values = createTransientView(graph, `${id}-label-values`, format, count);
  const validity = createTransientView(graph, `${id}-label-validity`, 'uint32', count);
  const nodes =
    band.storage.kind === 'texture'
      ? captureGraphCommandNodes(graph, () =>
          new GPURasterTextureToBuffer({
            id: `${id}-labels`,
            input: band as GPURasterTextureBand,
            output: values,
            outputValidity: validity,
            applyCalibration: false
          }).addToGraph(graph)
        )
      : [
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-labels`,
            operation: 'GPURasterReprojection',
            invocationCount: count,
            bindings: [
              {name: 'source', view: band.storage.values, type, access: 'read'},
              ...(band.validity
                ? [
                    {
                      name: 'sourceValidity',
                      view: band.validity,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {name: 'values', view: values, type, access: 'read_write'},
              {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
            ],
            body: `let value=source[sourceOffset+index];values[valuesOffset+index]=value;
validity[validityOffset+index]=u32(${band.validity ? 'sourceValidity[sourceValidityOffset+index]!=0u' : 'true'}${band.noDataValue === undefined ? '' : `&&value!=${noDataLiteral}`});`
          })
        ];
  return {
    band: {id, format, storage: {kind: 'buffer', values}, validity} as GPURasterBufferBand,
    nodes
  };
}

/** WGSL helper shared by raster and terrain kernels. @internal */
export const RASTER_WGSL_HELPERS = /* wgsl */ IS_FINITE_VALUE_WGSL;

/** Maximum total grid cells addressable by uint32 raster indices. @internal */
export const RASTER_GRID_MAXIMUM_CELL_COUNT = 4294967295;

/** Validates runtime string settings before selecting a route or packing. @internal */
export function validateRasterOption(
  id: string,
  name: string,
  value: string,
  allowed: readonly string[]
): void {
  validateRequirement(
    id,
    allowed.includes(value),
    `${name} must be ${allowed.join(', ')} (got ${value})`
  );
}

/** Compares JSON values by member value, retaining array order. @internal */
function equalRasterJSON(first: unknown, second: unknown): boolean {
  if (first === second) return true;
  if (first === null || second === null || typeof first !== 'object' || typeof second !== 'object')
    return false;
  if (Array.isArray(first) || Array.isArray(second)) {
    return (
      Array.isArray(first) &&
      Array.isArray(second) &&
      first.length === second.length &&
      first.every((value, index) => equalRasterJSON(value, second[index]))
    );
  }
  const firstRecord = first as Record<string, unknown>;
  const secondRecord = second as Record<string, unknown>;
  const keys = Object.keys(firstRecord);
  return (
    keys.length === Object.keys(secondRecord).length &&
    keys.every(
      key =>
        Object.hasOwn(secondRecord, key) && equalRasterJSON(firstRecord[key], secondRecord[key])
    )
  );
}

/** Rejects same-lattice warp CRS mismatches before scratch is reserved. @internal */
export function validateRasterReprojectionCRS(
  id: string,
  source: GPURasterMetadata,
  destination: GPURasterMetadata,
  projected: boolean
): void {
  const first = source.coordinateReferenceSystem;
  const second = destination.coordinateReferenceSystem;
  validateRequirement(
    id,
    projected ||
      (first === undefined && second === undefined) ||
      (first !== undefined &&
        second !== undefined &&
        first.authority === second.authority &&
        first.wellKnownText === second.wellKnownText &&
        equalRasterJSON(first.projectionJson, second.projectionJson)),
    'sourceGrid and destinationGrid must share a coordinate reference system unless inverseProjection is given'
  );
}

/** Validates projection compatibility before direct or tiled graph allocation. @internal */
export function getRasterReprojectionShader(
  id: string,
  sourceGrid: GPURasterMetadata,
  destinationGrid: GPURasterMetadata,
  inverseProjection: CompiledProjection | undefined
): ProjectionShader | undefined {
  validateRasterReprojectionCRS(id, sourceGrid, destinationGrid, inverseProjection !== undefined);
  validateRequirement(
    id,
    !inverseProjection || inverseProjection.inputFormat === 'float32x2',
    `inverseProjection.inputFormat must be 'float32x2' (got ${inverseProjection?.inputFormat})`
  );
  if (!inverseProjection) return undefined;
  if (inverseProjection.precision === 'double-single') {
    validateFiniteNumber(id, 'sourceGrid.affine[2]', sourceGrid.affine[2]);
    validateFiniteNumber(id, 'sourceGrid.affine[5]', sourceGrid.affine[5]);
  }
  const shader = inverseProjection.getShader({namespace: 'raster'});
  const declaration = `@group(0) @binding(auto) var<storage, read> ${shader.bindingName}: array<u32>;`;
  validateRequirement(
    id,
    shader.source.includes(declaration),
    'inverseProjection.getShader() must declare its uint32 parameter storage binding'
  );
  return {...shader, source: shader.source.replace(declaration, '')};
}

/** Captures declared grid fields and CRS JSON without retaining caller metadata. @internal */
export function snapshotRasterGrid(grid: GPURasterMetadata): GPURasterMetadata {
  const captured = snapshotConstructionFields(grid, [
    'width',
    'height',
    'affine',
    'pixelInterpretation',
    'coordinateReferenceSystem',
    'level',
    'levelZeroOrigin'
  ]);
  return Object.freeze({
    ...captured,
    affine: Object.freeze([...captured.affine]) as typeof grid.affine,
    levelZeroOrigin: captured.levelZeroOrigin
      ? (Object.freeze([...captured.levelZeroOrigin]) as typeof grid.levelZeroOrigin)
      : undefined,
    coordinateReferenceSystem: captured.coordinateReferenceSystem
      ? Object.freeze(
          JSON.parse(
            JSON.stringify(
              snapshotConstructionFields(captured.coordinateReferenceSystem, [
                'authority',
                'wellKnownText',
                'projectionJson'
              ])
            )
          ) as NonNullable<typeof grid.coordinateReferenceSystem>
        )
      : undefined
  });
}

/**
 * Throws unless the grid width and height are positive integers whose product fits in uint32 and
 * the affine is finite.
 *
 * @returns The pixel count.
 * @internal
 */
export function validateRasterGrid(id: string, grid: GPURasterMetadata): number {
  const {width, height, affine} = grid;
  validateRequirement(
    id,
    Number.isSafeInteger(width) && Number.isSafeInteger(height) && width >= 1 && height >= 1,
    'grid dimensions must be positive integers'
  );
  const cellCount = width * height;
  validateMaximumCount(
    id,
    `grid cell count (${cellCount})`,
    cellCount,
    RASTER_GRID_MAXIMUM_CELL_COUNT
  );
  if (affine.length !== 6 || !affine.every(Number.isFinite)) {
    throw new Error(`${id} grid affine must be six finite values`);
  }
  return width * height;
}

/**
 * Throws when a latitude-dependent cell size mode meets a rotated or sheared grid: the
 * `'web-mercator'` and `'geographic'` row scales assume rows of constant world y.
 *
 * @internal
 */
export function validateRasterGridAxisAligned(
  id: string,
  grid: GPURasterMetadata,
  cellSizeMode: GPURasterCellSizeMode | undefined
): void {
  if ((cellSizeMode ?? 'uniform') !== 'uniform' && (grid.affine[1] !== 0 || grid.affine[3] !== 0)) {
    throw new Error(`${id} ${cellSizeMode} cell sizes need an axis-aligned grid`);
  }
}

/**
 * Largest `|cos|` of the angle between the two pixel axes accepted as orthogonal. A rotated grid
 * built from float64 sines and cosines is far inside it; the diagonal step it admits is wrong by
 * less than one part in a million.
 *
 * @internal
 */
export const RASTER_GRID_ORTHOGONALITY_TOLERANCE = 1e-6;

/**
 * Throws when the two pixel axes of the grid affine are not orthogonal (a sheared grid). The D8
 * step lengths `[hypot(a, d), hypot(b, e), sqrt of their squares]` are the ground lengths of the
 * moves only when the column axis `(a, d)` and the row axis `(b, e)` are perpendicular; on a
 * sheared grid the two diagonals have different lengths, `hypot(a + b, d + e)` and
 * `hypot(a - b, d - e)`. Rotated grids pass. Both lengths must be positive finite float32
 * values. The normalized-axis dot product avoids overflow/underflow from coordinate units.
 *
 * @internal
 */
export function validateRasterGridOrthogonal(id: string, grid: GPURasterMetadata): void {
  const [a, b, , d, e] = grid.affine;
  const columnLength = Math.hypot(a, d);
  const rowLength = Math.hypot(b, e);
  const packedColumnLength = Math.fround(columnLength);
  const packedRowLength = Math.fround(rowLength);
  validateRequirement(
    id,
    packedColumnLength > 0 &&
      Number.isFinite(packedColumnLength) &&
      packedRowLength > 0 &&
      Number.isFinite(packedRowLength),
    'grid.affine pixel-axis lengths must be positive and finite after float32 conversion'
  );
  const normalizedDotProduct =
    (a / columnLength) * (b / rowLength) + (d / columnLength) * (e / rowLength);
  if (!(Math.abs(normalizedDotProduct) <= RASTER_GRID_ORTHOGONALITY_TOLERANCE)) {
    throw new Error(
      `${id} grid.affine must have orthogonal pixel axes (normalized axis dot product within ${RASTER_GRID_ORTHOGONALITY_TOLERANCE} of zero): a sheared grid has two different diagonal step lengths; resample it to a rotated or axis-aligned grid`
    );
  }
}

/** Throws unless `parameters` is packed float32 with at least `length` values. @internal */
export function validateRasterParameters(
  id: string,
  parameters: GraphDataView,
  length: number,
  name: string = 'parameters'
): void {
  validatePackedView(parameters, ['float32'], `${id} ${name}`);
  if (parameters.length < length) {
    throw new Error(`${id} ${name} must contain at least ${length} values`);
  }
}

/** Throws unless a packer target holds at least `length` values. @internal */
export function validateParameterTarget(
  name: string,
  target: ArrayLike<number>,
  length: number
): void {
  if (target.length < length) {
    throw new Error(`${name} target must hold ${length} values`);
  }
}

/** Checks logical ranges and co-bound storage usage, resolving imported physical buffers. @internal */
export function validateRasterBuffersDistinct(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  validateRasterWritableDispatches(
    id,
    inputs.filter(view => view !== undefined),
    outputs.filter(view => view !== undefined)
  );
}

/** Returns the buffer views of a raster band (values for buffer bands, plus validity). @internal */
export function getRasterBandViews(band: GPURasterBand): GraphDataView[] {
  return [
    ...(band.storage.kind === 'buffer' ? [band.storage.values] : []),
    ...(band.validity ? [band.validity] : [])
  ];
}

/** Validates texture-backed analysis inputs through the public raster contract, without allocation. @internal */
export function validateRasterTextureBand(
  id: string,
  band: GPURasterBand,
  grid: GPURasterMetadata
): void {
  if (band.storage.kind === 'texture') {
    validateRequirement(
      id,
      (band.storage.view.texture.usage & Texture.SAMPLE) !== 0,
      'input.storage.view.texture.usage must include Texture.SAMPLE'
    );
    if (band.scale !== undefined) validateFiniteNumber(id, 'input.scale', band.scale);
    if (band.offset !== undefined) validateFiniteNumber(id, 'input.offset', band.offset);
    new GPURaster({id: `${id} input`, metadata: grid, bands: [band]});
  }
}

/** Captures band metadata and bindings while retaining GPU view identities and per-frame contents. @internal */
export function snapshotRasterBand(band: GPURasterBand): GPURasterBand {
  const captured = snapshotConstructionFields(band, [
    'id',
    'format',
    'storage',
    'validity',
    'noDataValue',
    'scale',
    'offset'
  ]);
  const storage = captured.storage;
  // Read declared fields even when supplied by a prototype or non-enumerable getter.
  return {
    ...captured,
    storage:
      storage.kind === 'buffer'
        ? snapshotConstructionFields(storage, ['kind', 'values'])
        : snapshotConstructionFields(storage, ['kind', 'view', 'channel'])
  } as GPURasterBand;
}

/** Throws when band storage or any texture belongs to a different graph. @internal */
export function validateRasterBandBelongsToGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  band: GPURasterBand,
  textures: readonly (GraphTextureView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, getRasterBandViews(band));
  const allTextures = [
    ...textures,
    band.storage.kind === 'texture' ? band.storage.view : undefined
  ];
  for (const texture of allTextures) {
    if (texture && texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
  }
}

/** Throws unless a storage texture has an allowed format and the grid extent. @internal */
export function validateRasterOutputTexture(
  id: string,
  name: string,
  texture: GraphTextureView | undefined,
  formats: readonly string[],
  grid: GPURasterMetadata
): void {
  if (
    texture &&
    (!formats.includes(texture.format) ||
      texture.width !== grid.width ||
      texture.height !== grid.height)
  ) {
    throw new Error(`${id} ${name} must be ${formats.join(' or ')} with the grid extent`);
  }
}

/** Buffer-backed band and the nodes that produce it. @internal */
export type RasterBufferBandSource<Parameters> = {
  band: GPURasterBufferBand;
  nodes: GPUCommandNode<Parameters>[];
};

/**
 * Produces a buffer-backed band (typically elevation) for downstream raster operators.
 *
 * Texture bands are gathered with `GPURasterTextureToBuffer`. Buffer bands pass through unless
 * `canonicalize` is set, in which case an identity `GPURasterNeighborhood` resolves scale, offset,
 * nodata, validity, and non-finite samples into canonical float32 values plus uint32 validity.
 *
 * `canonicalize: 'reuse-canonical'` behaves like `true`, except that a float32 buffer band that
 * already has validity and no scale, offset, or noData passes through without a copy. Its values
 * may still hold non-finite samples, so the consumer must treat them as invalid itself.
 *
 * @internal
 */
export function getRasterBufferBandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  band: GPURasterBand,
  grid: GPURasterMetadata,
  canonicalize: boolean | 'reuse-canonical'
): RasterBufferBandSource<Parameters> {
  const {width, height} = grid;
  if (band.storage.kind === 'buffer' && !canonicalize) {
    return {band: band as GPURasterBufferBand, nodes: []};
  }
  if (
    band.storage.kind === 'buffer' &&
    canonicalize === 'reuse-canonical' &&
    band.format === 'float32' &&
    band.validity &&
    band.scale === undefined &&
    band.offset === undefined &&
    band.noDataValue === undefined
  ) {
    return {band: band as GPURasterBufferBand, nodes: []};
  }
  const pixelCount = width * height;
  const values = createTransientView(graph, `${id}-elevation-values`, 'float32', pixelCount);
  const validity = createTransientView(graph, `${id}-elevation-validity`, 'uint32', pixelCount);
  const nodes =
    band.storage.kind === 'texture'
      ? captureGraphCommandNodes(graph, () =>
          new GPURasterTextureToBuffer({
            id: `${id}-elevation`,
            input: band as GPURasterTextureBand,
            output: values,
            outputValidity: validity,
            applyCalibration: true
          }).addToGraph(graph)
        )
      : captureGraphCommandNodes(graph, () =>
          new GPURasterNeighborhood({
            id: `${id}-elevation`,
            width,
            height,
            input: band as GPURasterBufferBand,
            output: values,
            outputValidity: validity,
            radius: 0,
            kernel: [1],
            borderMode: 'clamp',
            noDataPolicy: 'propagate',
            normalize: false
          }).addToGraph(graph)
        );
  return {
    band: {
      id: `${id}-elevation`,
      format: 'float32',
      storage: {kind: 'buffer', values},
      validity
    },
    nodes
  };
}

/** Returns a raster parameter local origin, defaulting to zero, or throws if it is not finite. @internal */
export function readRasterParameterOrigin(
  label: string,
  localOrigin: readonly [number, number] | undefined
): readonly [number, number] {
  if (!localOrigin) {
    return [0, 0];
  }
  if (!(Number.isFinite(localOrigin[0]) && Number.isFinite(localOrigin[1]))) {
    throw new Error(`${label} localOrigin must be finite`);
  }
  return localOrigin;
}

/** Throws unless a parameter extent is `[minX, minY, maxX, maxY]` with finite, increasing bounds. @internal */
export function validateRasterParameterExtent(label: string, extent: readonly number[]): void {
  const [minX, minY, maxX, maxY] = extent;
  if (extent.length !== 4 || !extent.every(Number.isFinite) || !(maxX > minX) || !(maxY > minY)) {
    throw new Error(`${label} extent must be finite with maxX > minX and maxY > minY`);
  }
}

/**
 * Cheap pre-check for focal-pair inputs: an identifier, a 32-bit scalar format, finite scale and
 * offset, and storage covering the grid. Graph ownership is checked separately when the node is
 * added to a graph; the published raster primitives validate the remainder.
 *
 * @internal
 */
export function validateRasterBand(
  band: GPURasterBand,
  grid: Pick<GPURasterMetadata, 'width' | 'height'>,
  label: string
): void {
  if (!band.id) {
    throw new Error(`${label} band requires an identifier`);
  }
  if (!['float32', 'uint32', 'sint32'].includes(band.format)) {
    throw new Error(`${label} format must be float32, uint32 or sint32`);
  }
  if (
    (band.scale !== undefined && !Number.isFinite(band.scale)) ||
    (band.offset !== undefined && !Number.isFinite(band.offset))
  ) {
    throw new Error(`${label} scale and offset must be finite`);
  }
  if (band.storage.kind === 'buffer') {
    if (band.storage.values.length < grid.width * grid.height) {
      throw new Error(`${label} values must cover the raster grid`);
    }
  } else if (band.storage.kind === 'texture') {
    const {view} = band.storage;
    if (view.width !== grid.width || view.height !== grid.height) {
      throw new Error(`${label} texture extent must match the raster grid`);
    }
  } else {
    throw new Error(`${label} storage must be a buffer or texture`);
  }
}
